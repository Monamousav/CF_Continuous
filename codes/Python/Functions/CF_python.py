import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import subprocess
import sys

# ===  ipywidgets is installed (for tqdm / Jupyter progress bar) ===
import subprocess
import sys

# ===  ipywidgets is installed (for tqdm / Jupyter progress bar) ===
try:
    import ipywidgets
except ImportError:
    print("Installing ipywidgets...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ipywidgets", "-q"])
    # Try enabling the Jupyter widget extension, but skip if it fails (VS Code doesn’t need it)
    try:
        subprocess.check_call([sys.executable, "-m", "jupyter", "nbextension", "enable", "--py", "widgetsnbextension"])
    except Exception as e:
        print(f"(Skipping nbextension enable: {e})")




# ================================================================
# === MODEL: Causal Forest (Python / raw inputs)
# ================================================================
def train_cf_model_from_splits(data_2nd_stage, eval_N_seq, split_csv_path, folder_name):
    """
    Train EconML CausalForestDML on raw (non-orthogonalized) data
    
    """

    # --- Setup results folder ---
    base_results_dir = os.path.join(os.getcwd(), "results")
    output_folder = os.path.join(base_results_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    splits_df = pd.read_csv(split_csv_path)
    p_corn = 6.25 / 25.4
    p_N = 1 / 0.453592

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    for _, row in tqdm(splits_df.iterrows(), total=len(splits_df)):
    #for _, row in tqdm(splits_df.iloc[:1].iterrows(), total=1):
        test_id = int(row["test_id"])
        train_ids = row[[c for c in row.index if c.startswith("train_")]].values

        # --- Split train/test ---
        train_df = data_2nd_stage[data_2nd_stage["sim"].isin(train_ids)].reset_index(drop=True)
        test_df = data_2nd_stage[data_2nd_stage["sim"] == test_id].reset_index(drop=True)

        # --- Variables ---
        Y_train = train_df["yield"].to_numpy()
        X_train = train_df[["Nk", "plateau", "b0"]].to_numpy()
        W_train = X_train.copy()

        # Treatment = spline basis (non-residualized)
        T_train = train_df[["T_1_tilde", "T_2_tilde", "T_3_tilde"]].to_numpy()

        Y_test = test_df["yield"].to_numpy()
        X_test = test_df[["Nk", "plateau", "b0"]].to_numpy()
        #T_test = test_df[["T_1_tilde", "T_2_tilde", "T_3_tilde"]].to_numpy()

        # --- Scale X for stability ---
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- Define models ---
        model_y = GradientBoostingRegressor(n_estimators=100, max_depth=3, min_samples_leaf=20)
        model_t = MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, max_depth=3, min_samples_leaf=20)
        )

        # --- Causal forest (internal orthogonalization) ---
        cf_model = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            n_estimators=2000,
            min_samples_leaf=10,
            min_impurity_decrease=0.001,
            cv=5,
            random_state=3747823
        )

        cf_model.fit(Y_train, T_train, X=X_train_scaled, W=W_train)

        # --- Validation prediction ---
        val_pred = cf_model.effect(X_test_scaled)
        val_df = pd.DataFrame({
            "true": Y_test,
            "pred": val_pred.mean(axis=1) if val_pred.ndim > 1 else val_pred
        })
        val_df.to_csv(
            os.path.join(output_folder, f"validation_{test_id}.csv"), index=False
        )

        # =======================================================
        # === Estimate site-specific EONR ===
        # =======================================================
        test_eval_seq = eval_N_seq[eval_N_seq["sim"] == test_id].reset_index(drop=True)
        N_seq = test_eval_seq["N"].to_numpy()
        est_EONR = []

        for i in range(test_df.shape[0]):
            # repeat field covariates across 100 N levels
            base_feat = test_df[["Nk", "plateau", "b0"]].iloc[[i]]
            repeated = pd.concat([base_feat] * len(N_seq), ignore_index=True)
            repeated_scaled = scaler.transform(repeated.to_numpy())


            # spline basis for this N_seq (T_1, T_2, T_3)
            T_eval = test_eval_seq[["T_1", "T_2", "T_3"]].to_numpy()


            # predict yield response
            T0 = np.zeros_like(T_eval)
            y_hat = cf_model.effect(repeated_scaled, T0=T0, T1=T_eval).reshape(-1)

            profit = y_hat * p_corn - N_seq * p_N
            est_EONR.append(N_seq[np.argmax(profit)])

        eonr_df = pd.DataFrame({
            "pred": est_EONR,
            "true": test_df["opt_N"].to_numpy()
        })
        eonr_df.to_csv(
            os.path.join(output_folder, f"EONR_{test_id}.csv"), index=False
        )

        print(f"Finished sim {test_id}: saved validation and EONR files.")


# ================================================================
# === RUNNER FUNCTION (called from main script)
# ================================================================
def run_model(model_type, n_fields, data_2nd_stage, eval_N_seq):
    split_csv_path = f"./data/train_test_split/train_test_splits_{n_fields}fields.csv"

    folder_map = {
        ("CF_python", 1): "CF_python_outcome_one_field",
        ("CF_python", 3): "CF_python_outcome_three_fields",
        ("CF_python", 5): "CF_python_outcome_five_fields",
        ("CF_python", 10): "CF_python_outcome_ten_fields",
        ("CF_python", 20): "CF_python_outcome_twenty_fields",
    }
    folder_name = folder_map.get((model_type, n_fields))
    if folder_name is None:
        raise ValueError("Unsupported combination of model and field count.")

    train_cf_model_from_splits(data_2nd_stage, eval_N_seq, split_csv_path, folder_name)
