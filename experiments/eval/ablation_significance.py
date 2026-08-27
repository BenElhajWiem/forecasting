from __future__ import annotations

import ast
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "experiments" / "eval" / "predicted_vs_gt"
NBOOT = 2000
CI = 0.95
SEED = 42

STAGE_TO_EXP = {
    "reproducibility": "FULL",
    "horizon": "L1_horizon",
    "pattern": "L1_pattern",
    "stats": "L1_stats",
    "summary": "L1_summarizer",
}


# ─────────────────────────────────────────────────────────────────────────────
# Scalar extraction -- same convention as full_significance_analysis.py
# ─────────────────────────────────────────────────────────────────────────────

def _parse_dict_field(raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return ast.literal_eval(raw)
        except Exception:
            return None
    return None


def _first_scalar(entry):
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, list) and entry:
        return float(entry[0]) if not isinstance(entry[0], list) else float(entry[0][0])
    return None


def _parse_metric(raw, metric: str):
    d = _parse_dict_field(raw)
    if not isinstance(d, dict) or metric not in d:
        return None
    return _first_scalar(d[metric])


# ─────────────────────────────────────────────────────────────────────────────
# Stats (mirrors significance_testing.py / full_significance_analysis.py)
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(arr: np.ndarray, stat=np.mean, n=NBOOT, ci=CI, seed=SEED):
    rng = np.random.default_rng(seed)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    boot = np.array([stat(rng.choice(arr, len(arr), replace=True)) for _ in range(n)])
    lo, hi = np.percentile(boot, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(stat(arr)), float(lo), float(hi)


def wilcoxon_paired(a: np.ndarray, b: np.ndarray):
    d = a - b
    d = d[d != 0]
    if len(d) < 5:
        return np.nan, np.nan, len(d)
    try:
        stat, p = wilcoxon(d, alternative="two-sided")
        return float(stat), float(p), len(d)
    except Exception:
        return np.nan, np.nan, len(d)


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction. NaNs pass through unchanged."""
    idx = [i for i, p in enumerate(pvals) if not np.isnan(p)]
    m = len(idx)
    order = sorted(idx, key=lambda i: pvals[i])
    adjusted = [np.nan] * len(pvals)
    running_max = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i])
        running_max = max(running_max, adj)
        adjusted[i] = running_max
    return adjusted


# ─────────────────────────────────────────────────────────────────────────────
# Load + build per-row normalized errors
# ─────────────────────────────────────────────────────────────────────────────

def load_all_errors() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(EVAL_DIR / "*_eval_with_gt.csv"))):
        backend = Path(path).stem.replace("_eval_with_gt", "")
        df = pd.read_csv(path)
        if "stage" not in df.columns:
            continue
        for _, r in df.iterrows():
            stage = r.get("stage")
            exp_id = STAGE_TO_EXP.get(stage)
            if exp_id is None:
                continue
            for metric in ("TOTALDEMAND", "RRP"):
                gt = _parse_metric(r.get("ground_truth"), metric)
                pred = _parse_metric(r.get("predicted"), metric)
                if gt is None or pred is None:
                    continue
                rows.append({
                    "backend": backend,
                    "exp_id": exp_id,
                    "query_id": r.get("query_id"),
                    "seed": r.get("seed"),
                    "metric": metric,
                    "gt": gt,
                    "abs_err": abs(pred - gt),
                })
    df = pd.DataFrame(rows)
    # Global per-(backend, metric) NMAE scale factor: mean(|ground truth|)
    # over the FULL-system rows (the reference condition), applied uniformly
    # to that backend's ablated rows too so FULL and ablated share one scale.
    df["err_norm"] = np.nan
    for (backend, metric), g in df.groupby(["backend", "metric"]):
        full_gt = df[(df["backend"] == backend) & (df["metric"] == metric) & (df["exp_id"] == "FULL")]["gt"]
        scale = full_gt.abs().mean()
        if not scale or np.isnan(scale) or scale <= 0:
            continue
        mask = (df["backend"] == backend) & (df["metric"] == metric)
        df.loc[mask, "err_norm"] = df.loc[mask, "abs_err"] / scale
    return df.dropna(subset=["err_norm"])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    df = load_all_errors()
    out_dir = Path(__file__).resolve().parent
    df.to_csv(out_dir / "ablation_errors_raw.csv", index=False)
    print(f"Parsed {len(df)} (backend, exp_id, query_id, seed, metric) error records")
    print(f"Saved: {out_dir / 'ablation_errors_raw.csv'}")

    result_rows = []
    for backend, bdf in df.groupby("backend"):
        ablated_ids = sorted(e for e in bdf["exp_id"].unique() if e != "FULL")

        for metric in ("TOTALDEMAND", "RRP"):
            full_sub = bdf[(bdf["exp_id"] == "FULL") & (bdf["metric"] == metric)]
            if full_sub.empty:
                continue
            full_series = full_sub.groupby(["query_id", "seed"])["err_norm"].mean()
            full_mae, full_lo, full_hi = bootstrap_ci(full_series.to_numpy())

            backend_block = []
            for exp_id in ablated_ids:
                abl_sub = bdf[(bdf["exp_id"] == exp_id) & (bdf["metric"] == metric)]
                if abl_sub.empty:
                    continue
                abl_series = abl_sub.groupby(["query_id", "seed"])["err_norm"].mean()
                abl_mae, abl_lo, abl_hi = bootstrap_ci(abl_series.to_numpy())

                shared = full_series.index.intersection(abl_series.index)
                if len(shared) >= 5:
                    stat, p, n_paired = wilcoxon_paired(
                        abl_series.loc[shared].to_numpy(), full_series.loc[shared].to_numpy()
                    )
                else:
                    stat, p, n_paired = np.nan, np.nan, len(shared)

                backend_block.append({
                    "backend": backend,
                    "metric": metric,
                    "component_removed": exp_id,
                    "full_nmae": full_mae, "full_ci_lo": full_lo, "full_ci_hi": full_hi,
                    "ablated_nmae": abl_mae, "ablated_ci_lo": abl_lo, "ablated_ci_hi": abl_hi,
                    "delta": abl_mae - full_mae if not (np.isnan(abl_mae) or np.isnan(full_mae)) else np.nan,
                    "wilcoxon_stat": stat, "wilcoxon_p": p, "n_paired": n_paired,
                    "n_full": len(full_series), "n_ablated": len(abl_series),
                })

            pvals = [r["wilcoxon_p"] for r in backend_block]
            adj = holm_bonferroni(pvals)
            for r, a in zip(backend_block, adj):
                r["wilcoxon_p_holm"] = a
                r["sig_holm"] = (
                    "***" if (not np.isnan(a) and a < 0.001) else
                    "**" if (not np.isnan(a) and a < 0.01) else
                    "*" if (not np.isnan(a) and a < 0.05) else "n.s."
                )
            result_rows.extend(backend_block)

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(out_dir / "ablation_significance.csv", index=False)

    print("\n" + "=" * 100)
    print(" Ablation significance: FULL vs. each ablated variant")
    print(" NMAE = mean(|pred-gt|) / mean(|gt|), global per (backend, metric); Holm-Bonferroni corrected")
    print("=" * 100)
    if not result_df.empty:
        cols = ["backend", "metric", "component_removed", "full_nmae", "ablated_nmae",
                "delta", "wilcoxon_p", "wilcoxon_p_holm", "sig_holm", "n_paired"]
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(result_df[cols].to_string(index=False))
    print(f"\nSaved: {out_dir / 'ablation_significance.csv'}")


if __name__ == "__main__":
    main()
