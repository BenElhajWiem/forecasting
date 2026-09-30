from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from experiments.eval.significance_testing import mae_ci, rmse_ci, extract_errors

ROOT = Path("experiments")
EVAL_DIR = ROOT / "eval" / "predicted_vs_gt"
BASELINE_CSV = ROOT / "baselines" / "baseline_results.csv"
NBOOT = 2000
CI = 0.95
SEED = 42

BASELINE_COLS = ["persistence", "seasonal_naive", "sarima", "ets", "theta", "prophet",
                  "chronos", "timesfm", "moirai", "tft"]
BASELINE_LABELS = {
    "persistence": "Persistence", "seasonal_naive": "Seasonal Naive", "sarima": "SARIMA",
    "ets": "ETS", "theta": "Theta", "prophet": "Prophet", "chronos": "Chronos",
    "timesfm": "TimesFM", "moirai": "Moirai", "tft": "TFT",
}


def baseline_errors(metric: str) -> dict[str, np.ndarray]:
    df = pd.read_csv(BASELINE_CSV)
    sub = df[df["metric"] == metric].copy()
    sub["gt"] = pd.to_numeric(sub["ground_truth"], errors="coerce")
    sub = sub.dropna(subset=["gt"])
    out = {}
    for col in BASELINE_COLS:
        if col not in sub.columns:
            continue
        vals = pd.to_numeric(sub[col], errors="coerce")
        valid = sub.assign(pred=vals).dropna(subset=["pred"])
        errs = (valid["pred"] - valid["gt"]).abs().to_numpy(dtype=float)
        if len(errs):
            out[col] = errs
    return out


def llm_errors(metric: str) -> dict[str, np.ndarray]:
    out = {}
    for csv_path in sorted(EVAL_DIR.glob("*_eval_with_gt.csv")):
        model = csv_path.stem.replace("_eval_with_gt", "")
        df = pd.read_csv(csv_path)
        if "stage" in df.columns:
            df = df[df["stage"] == "reproducibility"]
        errs = extract_errors(df, metric=metric).to_numpy(dtype=float)
        errs = errs[~np.isnan(errs)]
        if len(errs):
            out[model] = errs
    return out


def main():
    for metric in ["TOTALDEMAND", "RRP"]:
        print(f"\n{'='*70}\n METRIC: {metric}\n{'='*70}")
        rows = []
        for col, errs in baseline_errors(metric).items():
            mae_pt, mae_lo, mae_hi = mae_ci(errs, n_bootstrap=NBOOT, ci=CI, seed=SEED)
            rmse_pt, rmse_lo, rmse_hi = rmse_ci(errs, n_bootstrap=NBOOT, ci=CI, seed=SEED)
            rows.append((BASELINE_LABELS[col], mae_pt, mae_lo, mae_hi, rmse_pt, rmse_lo, rmse_hi, len(errs)))
        for model, errs in llm_errors(metric).items():
            mae_pt, mae_lo, mae_hi = mae_ci(errs, n_bootstrap=NBOOT, ci=CI, seed=SEED)
            rmse_pt, rmse_lo, rmse_hi = rmse_ci(errs, n_bootstrap=NBOOT, ci=CI, seed=SEED)
            rows.append((model, mae_pt, mae_lo, mae_hi, rmse_pt, rmse_lo, rmse_hi, len(errs)))

        print(f"  {'Method':16s} {'MAE':>10s} {'MAE CI':>22s} {'RMSE':>10s} {'RMSE CI':>22s} {'n':>4s}")
        for name, mae_pt, mae_lo, mae_hi, rmse_pt, rmse_lo, rmse_hi, n in rows:
            print(f"  {name:16s} {mae_pt:10.2f} [{mae_lo:8.2f},{mae_hi:8.2f}] {rmse_pt:10.2f} [{rmse_lo:8.2f},{rmse_hi:8.2f}] {n:4d}")

        pd.DataFrame(rows, columns=["method", "MAE", "MAE_lo", "MAE_hi", "RMSE", "RMSE_lo", "RMSE_hi", "n"]) \
          .to_csv(ROOT / "eval" / f"table8_rmse_{metric}.csv", index=False)


if __name__ == "__main__":
    main()
