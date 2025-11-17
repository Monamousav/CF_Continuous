import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from patsy import dmatrix, build_design_matrices
from sklearn.preprocessing import StandardScaler

# make parent directory visible for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from run_CF_c import run_CF_c_py

# ==============================================================
# CONFIGURATION
# ==============================================================
k_splines = 4
spline_formula = f"cr(N, df={k_splines}) - 1"
p_corn = 6.25 / 25.4  # $/kg
p_N = 1 / 0.453592    # $/kg

# ==============================================================
# Prepare spline basis (same as mgcv s(N, k=4))
# ==============================================================
from patsy import cr
def prepare_T_mat_py(train_df, test_df):
    N_train = train_df["N"].to_numpy()
    N_min, N_max = float(np.min(N_train)), float(np.max(N_train))

    T_train_df = dmatrix(spline_formula, {"N": N_train}, return_type="dataframe")
    design_info = T_train_df.design_info

    N_test_clip = np.clip(test_df["N"].to_numpy(), N_min, N_max)
    T_test_mat = build_design_matrices([design_info], {"N": N_test_clip})[0]
    T_test_df = pd.DataFrame(np.asarray(T_test_mat), columns=T_train_df.columns)

    col_names = [f"T_{i+1}" for i in range(T_train_df.shape[1])]
    T_train_df.columns = col_names
    T_test_df.columns = col_names

    return T_train_df, T_test_df, design_info, (N_min, N_max)

# ==============================================================
# TRAIN + TEST LOOP (Mimics R “CF Loop”)
# ==============================================================
def train_CF_R_like(data_2nd_stage, split_csv_path, folder_name):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    output_folder = os.path.join(results_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    progress_csv = os.path.join(output_folder, "progress.csv")

    splits_df = pd.read_csv(split_csv_path)
    x_vars = ["Nk", "plateau", "b0"]

    for _, row in tqdm(splits_df.iterrows(), total=len(splits_df)):
    #row = splits_df[splits_df['test_id'] == 1].iloc[0:1].copy()
    #for _, row in row.iterrows():    
        test_sim_id = int(row["test_id"])
        out_fn_eonr = os.path.join(output_folder, f"sim_{test_sim_id:03d}_eonr.csv")
        if os.path.exists(out_fn_eonr):
            print(f"Skipping sim {test_sim_id} (already saved)")
            continue

        try:
            # ===== Train/test data =====
            train_ids = row[[c for c in row.index if c.startswith("train_")]].dropna().astype(int).tolist()
            train_df = data_2nd_stage[data_2nd_stage["sim"].isin(train_ids)].copy()
            test_df = data_2nd_stage[data_2nd_stage["sim"] == test_sim_id].copy()

            # drop old spline columns if exist
            train_df = train_df.loc[:, ~train_df.columns.str.startswith("T_")]
            test_df = test_df.loc[:, ~test_df.columns.str.startswith("T_")]

            # ===== Build spline basis =====
            T_train_df, T_test_df, design_info, (N_min_tr, N_max_tr) = prepare_T_mat_py(train_df, test_df)
            T_vars = T_train_df.columns.tolist()
            train_df = pd.concat([train_df.reset_index(drop=True), T_train_df], axis=1)
            test_df = pd.concat([test_df.reset_index(drop=True), T_test_df], axis=1)

            # ===== Prepare data =====
            Y = train_df["yield"].to_numpy()
            X = train_df[x_vars].to_numpy()
            T_mat = train_df[T_vars].to_numpy()
            W = X
            X_test = test_df[x_vars].to_numpy()

            # ===== Scale X for better stability =====
            X_scaler = StandardScaler().fit(X)
            X_scaled = X_scaler.transform(X)
            X_test_scaled = X_scaler.transform(X_test)

            # ===== Train CF =====
            te_hat_cf = run_CF_c_py(
                Y=Y,
                T=T_mat,
                X=X_scaled,
                W=W,
                n_estimators=2000,
                min_samples_leaf=20,   
                random_state=78343
            )

            # ===== Predict treatment effects =====
            te_hat = te_hat_cf.const_marginal_effect(X_test_scaled)
            te_hat = np.atleast_2d(te_hat)
            n_T = T_mat.shape[1]

            if te_hat.shape[1] != n_T:
                print(f"Adjusting te_hat columns from {te_hat.shape[1]} → {n_T}")
                te_hat = np.resize(te_hat, (te_hat.shape[0], n_T))

            # ===== Evaluate response curve =====
            T_seq = np.linspace(N_min_tr, N_max_tr, 100)
            eval_T_mat = build_design_matrices([design_info], {"N": T_seq})[0]
            eval_T_full = np.asarray(eval_T_mat)
            if eval_T_full.shape[1] != n_T:
                eval_T_full = eval_T_full[:, :n_T]

            curv = te_hat @ eval_T_full.T
            curv_df = (
                pd.DataFrame(curv, columns=T_seq)
                .assign(aunit_id=test_df["aunit_id"].to_numpy())
                .melt(id_vars="aunit_id", var_name="T", value_name="est")
            )
            curv_df["T"] = curv_df["T"].astype(float)
            curv_df["profit"] = curv_df["est"] * p_corn - curv_df["T"] * p_N

            # ===== Site-specific EONR =====
            ss_eonr = (
                curv_df.sort_values(["aunit_id", "profit"])
                .groupby("aunit_id", as_index=False)
                .tail(1)[["aunit_id", "T"]]
                .rename(columns={"T": "opt_N_hat"})
            )

            # ===== True EONR =====
            true_ss = test_df[["aunit_id", "b0", "b1", "b2", "Nk"]].copy()
            true_ss["opt_N"] = (p_N / p_corn - true_ss["b1"]) / (2.0 * true_ss["b2"])
            true_ss["opt_N"] = true_ss[["opt_N", "Nk"]].min(axis=1)
            true_ss["opt_N"] = true_ss["opt_N"].clip(lower=0.0)

            combined = ss_eonr.merge(true_ss[["aunit_id", "opt_N"]], on="aunit_id", how="left")
            rmse_val = np.sqrt(np.mean((combined["opt_N_hat"] - combined["opt_N"]) ** 2))
            corr_val = combined[["opt_N", "opt_N_hat"]].corr().iloc[0, 1]

            # ===== Save results =====
            combined["sim"] = test_sim_id
            combined.to_csv(out_fn_eonr, index=False)

            prog = pd.DataFrame({
                "test_sim_id": [test_sim_id],
                "corr": [corr_val],
                "rmse": [rmse_val],
                "n_train": [len(train_df)],
                "file": [out_fn_eonr],
                "ts": [pd.Timestamp.now().isoformat()]
            })
            prog.to_csv(progress_csv, mode="a", header=not os.path.exists(progress_csv), index=False)

            print(f"✅ Finished sim {test_sim_id} (corr = {corr_val:.3f}, RMSE = {rmse_val:.2f})")

        except Exception as ex:
            print(f"❌ ERROR in sim {test_sim_id}: {ex}")
            err_csv = os.path.join(output_folder, "errors.csv")
            pd.DataFrame({
                "test_sim_id": [test_sim_id],
                "error": [str(ex)],
                "ts": [pd.Timestamp.now().isoformat()]
            }).to_csv(err_csv, mode="a", header=not os.path.exists(err_csv), index=False)


# ==============================================================
# Unified wrapper for external call
# ==============================================================
def run_model(model_type, n_fields, data_2nd_stage, evall_N_seq, device):
    split_csv_path = f"./data/train_test_split/train_test_splits_{n_fields}fields.csv"
    folder_map = {
        ("test", 1): "test_outcome_one_field",
        ("test", 3): "test_outcome_three_fields",
        ("test", 5): "test_outcome_five_fields",
        ("test", 10): "test_outcome_ten_fields",
        ("test", 20): "test_outcome_twenty_fields",
    }

    folder_name = folder_map.get((model_type, n_fields))
    if folder_name is None:
        raise ValueError("Unsupported model/field count combination.")

    train_CF_R_like(data_2nd_stage, split_csv_path, folder_name)


