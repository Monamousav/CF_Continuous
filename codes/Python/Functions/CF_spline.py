import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from patsy import dmatrix  # for spline basis like mgcv::s()

# make parent directory visible for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from run_CF_c import run_CF_c_py
import torch


# === MODEL: Causal Forest with spline-transformed treatment (normalized per field) === #

def train_CF_spline_from_splits(data_2nd_stage, evall_N_seq, split_csv_path, folder_name, device):
    # --- Resolve project root automatically ---
# Finds the "CF_Continuous" folder no matter where the script is launched from
    project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

# --- Create subfolder for model output ---
    output_folder = os.path.join(results_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)


    splits_df = pd.read_csv(split_csv_path)
    p_corn = 6.25 / 25.4  # $/kg
    p_N = 1 / 0.453592    # $/kg

    # For quick test only 1 iteration:
    #row = splits_df[splits_df['test_id'] == 1].iloc[0:1].copy()
    #for _, row in row.iterrows():
    for _, row in tqdm(splits_df.iterrows(), total=len(splits_df)):
        test_id = row["test_id"]
        train_ids = row[[c for c in row.index if c.startswith("train_")]].dropna().values

        # ---- Split data ----
        train_df = data_2nd_stage[data_2nd_stage["sim"].isin(train_ids)].reset_index(drop=True)
        test_df = data_2nd_stage[data_2nd_stage["sim"] == test_id].reset_index(drop=True)

        # ---- Normalize N per field to [0,1] ----
        mm = train_df.groupby("sim")["N"].agg(Nmin="min", Nmax="max").reset_index()
        train_df = train_df.merge(mm, on="sim", how="left")
        test_df = test_df.merge(mm, on="sim", how="left")  # may be missing if test sim not in train set

        # fallback: pooled min/max for unseen test sim
        pool_min, pool_max = train_df["N"].min(), train_df["N"].max()
        test_df["Nmin"] = np.where(test_df["Nmin"].isna(), pool_min, test_df["Nmin"])
        test_df["Nmax"] = np.where(test_df["Nmax"].isna(), pool_max, test_df["Nmax"])

        eps = 1e-8
        train_df["N_star"] = (train_df["N"] - train_df["Nmin"]) / (train_df["Nmax"] - train_df["Nmin"] + eps)
        train_df["N_star"] = train_df["N_star"].clip(0, 1)
        test_df["N_star"] = (test_df["N"] - test_df["Nmin"]) / (test_df["Nmax"] - test_df["Nmin"] + eps)
        test_df["N_star"] = test_df["N_star"].clip(0, 1)

        # ---- Construct spline basis for normalized N_star ----
        T_train = dmatrix("bs(N_star, df=4, degree=3, include_intercept=False)", data=train_df)
        T_test = dmatrix("bs(N_star, df=4, degree=3, include_intercept=False)", data=test_df)

        # ---- Prepare data ----
        Y = train_df["yield"].to_numpy()
        X = train_df[["Nk", "plateau", "b0", "Nmin", "Nmax"]].to_numpy()
        W = X  # same controls

        X_scaler = StandardScaler().fit(X)
        X_scaled = X_scaler.transform(X)
        X_test_scaled = X_scaler.transform(test_df[["Nk", "plateau", "b0", "Nmin", "Nmax"]])

        # ---- Train causal forest ----
        est = run_CF_c_py(
            Y=Y,
            T=T_train,
            X=X_scaled,
            W=W,
            n_estimators=1000,
            min_samples_leaf=20,     # stronger regularization
            random_state=78343
        )

        # ---- Validation prediction ----
        te_hat = est.const_marginal_effect(X_scaled)
        val_out = pd.DataFrame(te_hat, columns=[f"theta_{i+1}" for i in range(te_hat.shape[1])])
        val_out["true"] = Y
        val_out.to_csv(os.path.join(output_folder, f"validation_{test_id}.csv"), index=False)

        # ---- EONR estimation (using normalized basis) ----
        estEONR = []

        for i in range(len(test_df)):
            Nmin_i, Nmax_i = float(test_df.loc[i, "Nmin"]), float(test_df.loc[i, "Nmax"])
            Nseq = np.linspace(Nmin_i, Nmax_i, 200)                     # physical N range
            Nseq_star = (Nseq - Nmin_i) / (Nmax_i - Nmin_i + eps)       # normalized [0,1]

            eval_spline = dmatrix("bs(N_star, df=4, degree=3, include_intercept=False)",
                                  data=pd.DataFrame({"N_star": Nseq_star}))
            base_X = np.repeat(X_test_scaled[[i], :], len(Nseq), axis=0)

            # predicted yield response curve
            y_hat = est.const_marginal_effect(base_X) @ eval_spline.T
            y_hat = np.mean(y_hat, axis=0)

            MP = y_hat * p_corn - Nseq * p_N
            estEONR.append(Nseq[np.argmax(MP)])

        pd.DataFrame({"pred": estEONR, "true": test_df["opt_N"].to_numpy()}) \
            .to_csv(os.path.join(output_folder, f"EONR_{test_id}.csv"), index=False)


# === Main wrapper ===
def run_model(model_type, n_fields, data_2nd_stage, evall_N_seq, device):
    split_csv_path = f"./data/train_test_split/train_test_splits_{n_fields}fields.csv"
    folder_map = {
        ("CF_spline", 1): "CF_spline_outcome_one_field",
        ("CF_spline", 3): "CF_spline_outcome_three_fields",
        ("CF_spline", 5): "CF_spline_outcome_five_fields",
        ("CF_spline", 10): "CF_spline_outcome_ten_fields",
        ("CF_spline", 20): "CF_spline_outcome_twenty_fields",
    }
    folder_name = folder_map.get((model_type, n_fields))
    if folder_name is None:
        raise ValueError("Unsupported model/field count combination.")

    if model_type == "CF_spline":
        train_CF_spline_from_splits(data_2nd_stage, evall_N_seq, split_csv_path, folder_name, device)
