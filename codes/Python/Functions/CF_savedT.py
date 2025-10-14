# %% [markdown]
# # === MODEL: CF_savedT ===

# %%
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

# %%
def train_cf_savedT_model_from_splits(data_2nd_stage, evall_N_seq, split_csv_path, folder_name):
    """
    Train EconML CausalForestDML using saved spline columns (T_1_tilde, T_2_tilde, T_3_tilde)
    """

    # === Output setup ===
    base_results_dir = os.path.join(os.getcwd(), "results")
    output_folder = os.path.join(base_results_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    # Subfolders for TE and response matrices
    te_dir = os.path.join(output_folder, "te_hat")
    resp_dir = os.path.join(output_folder, "response")
    eonr_dir = os.path.join(output_folder, "eonr")

    os.makedirs(te_dir, exist_ok=True)
    os.makedirs(resp_dir, exist_ok=True)
    os.makedirs(eonr_dir, exist_ok=True)

    # === Load split file ===
    splits_df = pd.read_csv(split_csv_path)

    # === Constants ===
    p_corn = 6.25 / 25.4   # $/kg
    p_N = 1 / 0.453592     # $/kg

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # === Loop over each split row ===
    #for _, row in tqdm(splits_df.iloc[:1].iterrows(), total=1):
    for _, row in tqdm(splits_df.iterrows(), total=len(splits_df)):
        test_id = int(row["test_id"])
        train_ids = row[[c for c in row.index if c.startswith("train_")]].dropna().astype(int).values

        # --- Split train/test data ---
        train_df = data_2nd_stage[data_2nd_stage["sim"].isin(train_ids)].reset_index(drop=True)
        test_df  = data_2nd_stage[data_2nd_stage["sim"] == test_id].reset_index(drop=True)

        # --- Prepare variables ---
        Y = train_df["yield"].to_numpy()
        X = train_df[["Nk", "plateau", "b0"]].to_numpy()
        W = X.copy()
        T = train_df[["T_1_tilde", "T_2_tilde", "T_3_tilde"]].to_numpy()

        # --- Causal Forest (orthogonalized, tuned) ---
        model_y = GradientBoostingRegressor(n_estimators=100, max_depth=3, min_samples_leaf=20)
        model_t = MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, max_depth=3, min_samples_leaf=20)
        )

        cf_model = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            cv=5,
            n_estimators=1000,
            min_samples_leaf=10,
            min_impurity_decrease=0.001,
            random_state=78343,
        )

        cf_model.tune(Y, T, X=X, W=W)
        cf_model.fit(Y, T, X=X, W=W)

        # --- Extract marginal treatment effects (like get_te() in R) ---
        X_test = test_df[["Nk", "plateau", "b0"]].to_numpy()
        te_hat = cf_model.const_marginal_effect(X_test)  # shape (n_cells, 3)

        # Save TE matrix
        te_df = pd.DataFrame(te_hat, columns=["TE_T1", "TE_T2", "TE_T3"])
        te_df["cell_id"] = np.arange(len(test_df))
        te_df.to_csv(os.path.join(te_dir, f"te_hat_{test_id}.csv"), index=False)

        # --- Generate yield response curves (like find_response_semi()) ---
        eval_seq = evall_N_seq[evall_N_seq["sim"] == test_id].reset_index(drop=True)
        T_seq = eval_seq["N"].values
        spline_basis = eval_seq[["T_1", "T_2", "T_3"]].to_numpy()  # shape (100, 3)

        pred_yield = np.dot(te_hat, spline_basis.T)  # shape (n_cells, 100)

        # Save yield response matrix
        response_df = pd.DataFrame(pred_yield, columns=[f"N_{n:.1f}" for n in T_seq])
        response_df["cell_id"] = np.arange(len(test_df))
        response_df.to_csv(os.path.join(resp_dir, f"response_{test_id}.csv"), index=False)

        # --- Profit and EONR ---
        profit = pred_yield * p_corn - T_seq * p_N
        best_indices = np.argmax(profit, axis=1)
        estEONR_vector = T_seq[best_indices]

        # --- Save EONR results ---
        eonr_df = pd.DataFrame({
            "cell_id": np.arange(len(test_df)),
            "pred": estEONR_vector,
            "true": test_df["opt_N"].values
        })
        eonr_df.to_csv(os.path.join(eonr_dir, f"EONR_{test_id}.csv"), index=False)

        # --- Optional validation-like check ---
        val_df = pd.DataFrame({
            "true_yield": test_df["yield"].values,
            "Nk": test_df["Nk"].values,
            "plateau": test_df["plateau"].values,
            "b0": test_df["b0"].values
        })
        val_df.to_csv(os.path.join(output_folder, f"validation_{test_id}.csv"), index=False)

        print(f"✅ Finished sim {test_id}: saved EONR, TE, and response files.")


# %%
# === MODEL DISPATCHER ===
def run_model(model_type, n_fields, data_2nd_stage, evall_N_seq):
    split_csv_path = f"./data/train_test_split/train_test_splits_{n_fields}fields.csv"

    folder_map = {
        ("CF_savedT", 1): "CF_savedT_outcome_one_field",
        ("CF_savedT", 3): "CF_savedT_outcome_three_fields",
        ("CF_savedT", 5): "CF_savedT_outcome_five_fields",
        ("CF_savedT", 10): "CF_savedT_outcome_ten_fields",
        ("CF_savedT", 20): "CF_savedT_outcome_twenty_fields",
    }

    folder_name = folder_map.get((model_type, n_fields))
    if folder_name is None:
        raise ValueError("Unsupported combination of model type and field count.")

    if model_type == "CF_savedT":
        train_cf_savedT_model_from_splits(data_2nd_stage, evall_N_seq, split_csv_path, folder_name)
