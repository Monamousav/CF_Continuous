#!/usr/bin/env python3
"""
CF Loop (Python port)

This script replicates the R "CF_Loop" workflow in Python using:
- pandas / numpy for data handling
- sklearn SplineTransformer to build a fixed spline basis for N (fit on train only)
- econml.dml.CausalForestDML for treatment effect estimation

It expects:
- Train/test split CSV at data/train_test_split/train_test_splits_{n_fields}fields.csv
- A flat table of observations with columns at least:
  sim, aunit_id, N, yield, Nk, plateau, b0, b1, b2
  available at either data/reg_data.parquet OR data/reg_data.csv

Outputs per test sim:
- results/CF_Loop_p_{n_fields}_fields/sim_{test_sim_id}_eonr.csv
- results/CF_Loop_p_{n_fields}_fields/progress.csv (appended)
- results/CF_Loop_p_{n_fields}_fields/errors.csv (if any)

Notes:
- Thread env vars are limited similar to the R script to avoid BLAS over‑subscription.
- The spline basis is *fit on the current TRAIN set* and then applied to both train and test.
- If you already have a saved basis you want to reuse across runs, persist the fitted
  SplineTransformer with joblib and load it instead of re-fitting (marked below).
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

# --- Limit math library threads (macOS/BLAS friendliness) ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

from sklearn.preprocessing import SplineTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

try:
    from econml.dml import CausalForestDML
except Exception as e:  # helpful import message
    raise SystemExit(
        "Failed to import econml. Install with `pip install econml` in your cf_conda311 env.\n"
        f"Original error: {e}"
    )

# =========================
# Paths & config helpers
# =========================

def find_project_root() -> Path:
    cwd = Path.cwd()
    for p in [cwd, *cwd.parents]:
        if (p / ".here").exists():
            return p
    return cwd

@dataclass
class Config:
    n_fields: int = 1

    @property
    def project_root(self) -> Path:
        return find_project_root()

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def splits_path(self) -> Path:
        return self.data_dir / "train_test_split" / f"train_test_splits_{self.n_fields}fields.csv"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "results" / f"CF_Loop_p_{self.n_fields}_fields"

    @property
    def progress_csv(self) -> Path:
        return self.output_dir / "progress.csv"

    @property
    def errors_csv(self) -> Path:
        return self.output_dir / "errors.csv"
 
# =========================
# Data loading
# =========================

def load_reg_data(data_dir: Path) -> pd.DataFrame:
    pq = data_dir / "reg_data.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    csv = data_dir / "reg_data.csv"
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(
        f"Could not find reg_data at {pq} or {csv}. Provide a flat file with columns: "
        "sim, aunit_id, N, yield, Nk, plateau, b0, b1, b2 (plus any X covariates)."
    )

# =========================
# Spline basis
# =========================

@dataclass
class TInfo:
    transformer: SplineTransformer
    T_sp_train: pd.DataFrame
    T_var_name: str = "N"


def prepare_T_mat(train_df: pd.DataFrame, T_var_name: str = "N", *, degree: int = 3, n_knots: int = 4) -> TInfo:
    x = train_df[[T_var_name]].to_numpy()
    tr = SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        include_bias=False,
        extrapolation="continue",
    )
    T_sp_train = tr.fit_transform(x)
    cols = [f"T_{i+1}" for i in range(T_sp_train.shape[1])]
    T_sp_train = pd.DataFrame(T_sp_train, index=train_df.index, columns=cols)
    return TInfo(transformer=tr, T_sp_train=T_sp_train, T_var_name=T_var_name)

# =========================
# econml CF wrapper
# =========================

def run_CF_c_py(
    Y: np.ndarray,
    T: np.ndarray | pd.DataFrame,
    X: np.ndarray,
    W: np.ndarray,
    model_y: GradientBoostingRegressor | None = None,
    model_t: MultiOutputRegressor | None = None,
    cv: int = 5,
    criterion: str = "mse",
    n_estimators: int = 1000,
    min_samples_leaf: int = 10,
    min_impurity_decrease: float = 0.001,
    random_state: int = 78343,
):
    if model_y is None:
        model_y = GradientBoostingRegressor(n_estimators=100, max_depth=3, min_samples_leaf=20)
    if model_t is None:
        model_t = MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, max_depth=3, min_samples_leaf=20)
        )

    est = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        cv=int(cv),
        criterion=criterion,
        n_estimators=int(n_estimators),
        min_samples_leaf=int(min_samples_leaf),
        min_impurity_decrease=min_impurity_decrease,
        random_state=int(random_state),
    )
    est.tune(Y, T, X=X, W=W)
    est.fit(Y, T, X=X, W=W)
    return est

# =========================
# Helpers
# =========================

def get_te(trained_model: CausalForestDML, test_df: pd.DataFrame, x_vars: List[str], id_var: str = "aunit_id"):
    X_test = test_df[x_vars].to_numpy()
    te_hat = trained_model.const_marginal_effect(X_test)
    te_df = pd.DataFrame(te_hat, index=test_df.index)
    ids = test_df[[id_var]].copy()
    return {"te_hat": te_df, "id_data": ids}


def evaluate_T_grid(T_seq: np.ndarray, tinfo: TInfo) -> pd.DataFrame:
    grid_df = pd.DataFrame({tinfo.T_var_name: T_seq})
    basis = tinfo.transformer.transform(grid_df[[tinfo.T_var_name]].to_numpy())
    cols = [f"T_{i+1}" for i in range(basis.shape[1])]
    return pd.DataFrame(basis, index=grid_df.index, columns=cols)


def find_response_semi(T_seq: np.ndarray, tinfo: TInfo, te_info: dict, id_var: str = "aunit_id") -> pd.DataFrame:
    eval_T = evaluate_T_grid(T_seq, tinfo)
    te_hat = te_info["te_hat"].to_numpy()
    resp = te_hat @ eval_T.to_numpy().T
    resp_df = (
        pd.DataFrame(resp, index=te_info["te_hat"].index, columns=T_seq.astype(float))
        .reset_index(drop=True)
    )
    long_df = resp_df.melt(value_name="est", var_name="T")
    ids = te_info["id_data"].reset_index(drop=True).loc[long_df.index // len(T_seq)].reset_index(drop=True)
    final = pd.concat([ids, long_df], axis=1)
    return final[[id_var, "T", "est"]]

# =========================
# Main loop (one iteration check)
# =========================

def cf_loop(cfg: Config):
    prj = cfg.project_root
    outdir = cfg.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    splits = pd.read_csv(cfg.splits_path)
    required_cols = ["test_id"] + [f"train_{i}" for i in range(1, cfg.n_fields + 1)]
    missing = [c for c in required_cols if c not in splits.columns]
    if missing:
        raise ValueError(f"Splits file {cfg.splits_path} missing columns: {missing}")

    reg = load_reg_data(cfg.data_dir)
    needed = {"sim", "aunit_id", "N", "yield", "Nk", "plateau", "b0", "b1", "b2"}
    missing_cols = needed - set(reg.columns)
    if missing_cols:
        raise ValueError(f"reg_data missing columns: {sorted(missing_cols)}")

    x_vars = ["Nk", "plateau", "b0"]


    # Run only one iteration for testing
    if len(splits) > 0:
        splits = splits.head(1)

    for i, row in splits.iterrows():
        test_sim_id = int(row["test_id"])
        fn_eonr = outdir / f"sim_{test_sim_id:03d}_eonr.csv"

        try:
            train_sim_ids = [int(row[f"train_{k}"]) for k in range(1, cfg.n_fields + 1)]
            sim_id_ls = train_sim_ids + [test_sim_id]

            data = reg.loc[reg["sim"].isin(sim_id_ls)].copy()
            train_data = data.loc[data["sim"] != test_sim_id].copy()
            test_data = reg.loc[reg["sim"] == test_sim_id].copy()

            tinfo = prepare_T_mat(train_data, T_var_name="N", degree=3, n_knots=4)

            Y = train_data["yield"].to_numpy()
            T_sp = tinfo.T_sp_train.to_numpy()
            X = train_data[x_vars].to_numpy()
            W = X

            est = run_CF_c_py(Y=Y, T=T_sp, X=X, W=W, n_estimators=2000)

            te_info = get_te(est, test_data, x_vars=x_vars, id_var="aunit_id")

            T_seq = np.linspace(train_data["N"].min(), train_data["N"].max(), 100)
            response_data = find_response_semi(T_seq, tinfo, te_info, id_var="aunit_id")

            pCorn = 6.25 / 25.4
            pN = 1.0 / 0.453592

            df_profit = response_data.copy()
            df_profit["profit"] = df_profit["est"] * pCorn - pN * df_profit["T"]
            idx = df_profit.groupby("aunit_id")["profit"].idxmax()
            ss_eonr = df_profit.loc[idx, ["aunit_id", "T"]].rename(columns={"T": "opt_N_hat"})

            true_cols = ["aunit_id", "b0", "b1", "b2", "Nk"]
            true_df = test_data[true_cols].copy()
            true_df["opt_N"] = (pN / pCorn - true_df["b1"]) / (2 * true_df["b2"])
            true_df["opt_N"] = true_df[["opt_N", "Nk"]].min(axis=1)
            true_df["opt_N"] = true_df["opt_N"].clip(lower=0)
            true_ss_eonr = true_df[["aunit_id", "opt_N"]]

            combined = ss_eonr.merge(true_ss_eonr, on="aunit_id", how="left")
            corr_val = combined[["opt_N_hat", "opt_N"]].corr().iloc[0, 1]

            e = combined[["aunit_id", "opt_N_hat", "opt_N"]].copy()
            e["sim"] = int(test_sim_id)
            e = e[["aunit_id", "opt_N_hat", "opt_N", "sim"]]

            tmp = fn_eonr.with_suffix(fn_eonr.suffix + ".tmp")
            e.to_csv(tmp, index=False)
            os.replace(tmp, fn_eonr)

            prog = pd.DataFrame(
                {
                    "test_sim_id": [test_sim_id],
                    "corr": [float(corr_val) if pd.notnull(corr_val) else np.nan],
                    "n_train": [len(train_data)],
                    "saved_file": [str(fn_eonr.with_suffix("").name)],
                    "ts": [time.strftime("%Y-%m-%d %H:%M:%S")],
                }
            )
            header = not cfg.progress_csv.exists()
            prog.to_csv(cfg.progress_csv, mode="a", index=False, header=header)

            print(f"Finished sim {test_sim_id} (corr = {corr_val:.3f} if not NaN)")

        except Exception as e:
            err = pd.DataFrame(
                {
                    "test_sim_id": [test_sim_id],
                    "error": [str(e)],
                    "ts": [time.strftime("%Y-%m-%d %H:%M:%S")],
                }
            )
            header = not cfg.errors_csv.exists()
            err.to_csv(cfg.errors_csv, mode="a", index=False, header=header)
            print(f"ERROR in sim {test_sim_id}: {e}")


if __name__ == "__main__":
    # Handle both command-line and VS Code interactive/Jupyter modes
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        n_fields = int(sys.argv[1])
    else:
        n_fields = 1  # default when no numeric argument provided (e.g., in VS Code)

    cfg = Config(n_fields=n_fields)
    print(f"Project root: {cfg.project_root}")
    print(f"Splits:       {cfg.splits_path}")
    print(f"Output dir:   {cfg.output_dir}")
    cf_loop(cfg)

