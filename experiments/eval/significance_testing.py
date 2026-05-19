"""
Statistical significance testing for forecast comparison.

Implements:
  1. Bootstrap confidence intervals for MAE and RMSE
  2. Wilcoxon signed-rank test for pairwise model comparison
  3. Diebold-Mariano test for forecast accuracy comparison
  4. Summary tables suitable for paper reporting

Usage (standalone):
    python -m experiments.eval.significance_testing \
        --inputs Claude_eval_with_gt.csv Gemini_eval_with_gt.csv ... \
        --stage reproducibility \
        --output significance_results/

Usage (as library):
    from experiments.eval.significance_testing import run_significance_analysis
    results = run_significance_analysis(model_dfs, metric="TOTALDEMAND")
"""
from __future__ import annotations

import ast
import itertools
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_scalar(raw) -> Optional[float]:
    """Extract a single float from various encoded formats."""
    if isinstance(raw, (int, float)):
        return float(raw)
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    try:
        return float(s)
    except ValueError:
        pass
    # Try dict/list literal e.g. {'TOTALDEMAND': [9207.15, '...']}
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, list) and v and isinstance(v[0], (int, float)):
                    return float(v[0])
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, (int, float)):
                    return float(item)
                if isinstance(item, list) and item and isinstance(item[0], (int, float)):
                    return float(item[0])
    except Exception:
        pass
    return None


def _parse_metric_from_gt(gt_raw, metric: str) -> Optional[float]:
    """Extract a specific metric value from a ground_truth field."""
    if isinstance(gt_raw, str):
        try:
            gt_raw = ast.literal_eval(gt_raw)
        except Exception:
            return _parse_scalar(gt_raw)
    if isinstance(gt_raw, dict):
        entry = gt_raw.get(metric)
        if entry is None:
            return None
        if isinstance(entry, (int, float)):
            return float(entry)
        if isinstance(entry, list):
            if entry and isinstance(entry[0], (int, float)):
                return float(entry[0])
            if entry and isinstance(entry[0], list):
                return float(entry[0][0]) if entry[0] else None
    return _parse_scalar(gt_raw)


def _parse_metric_from_predicted(pred_raw, metric: str) -> Optional[float]:
    """Extract a specific metric value from a predicted field."""
    if isinstance(pred_raw, str):
        # Try literal eval first
        try:
            obj = ast.literal_eval(pred_raw)
            if isinstance(obj, dict) and metric in obj:
                v = obj[metric]
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, list) and v:
                    if isinstance(v[0], (int, float)):
                        return float(v[0])
                    if isinstance(v[0], list) and v[0]:
                        return float(v[0][0])
        except Exception:
            pass
        # Scan for "METRIC=<value>" pattern
        import re
        pattern = rf"{re.escape(metric)}\s*[=:]\s*([-+]?\d*\.?\d+)"
        m = re.search(pattern, pred_raw, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return _parse_scalar(pred_raw)


def extract_errors(
    df: pd.DataFrame,
    metric: str = "TOTALDEMAND",
    predicted_col: str = "predicted",
    gt_col: str = "ground_truth",
) -> pd.Series:
    """
    Return a pd.Series of absolute prediction errors for a given metric,
    aligned by index to df.
    """
    errors = []
    for _, row in df.iterrows():
        gt = _parse_metric_from_gt(row.get(gt_col), metric)
        pred = _parse_metric_from_predicted(row.get(predicted_col), metric)
        if gt is None or pred is None or np.isnan(gt) or np.isnan(pred):
            errors.append(np.nan)
        else:
            errors.append(abs(gt - pred))
    return pd.Series(errors, index=df.index)


def extract_signed_errors(
    df: pd.DataFrame,
    metric: str = "TOTALDEMAND",
    predicted_col: str = "predicted",
    gt_col: str = "ground_truth",
) -> pd.Series:
    """Return signed errors (gt − pred) for Diebold-Mariano."""
    errors = []
    for _, row in df.iterrows():
        gt = _parse_metric_from_gt(row.get(gt_col), metric)
        pred = _parse_metric_from_predicted(row.get(predicted_col), metric)
        if gt is None or pred is None or np.isnan(gt) or np.isnan(pred):
            errors.append(np.nan)
        else:
            errors.append(gt - pred)
    return pd.Series(errors, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    errors: np.ndarray,
    stat_fn,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute a bootstrap confidence interval for stat_fn applied to absolute errors.

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    valid = errors[~np.isnan(errors)]
    if len(valid) == 0:
        return np.nan, np.nan, np.nan
    point = stat_fn(valid)
    boots = [stat_fn(rng.choice(valid, size=len(valid), replace=True)) for _ in range(n_bootstrap)]
    alpha = 1.0 - ci
    lower = float(np.percentile(boots, 100 * alpha / 2))
    upper = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(point), lower, upper


def mae_ci(errors: np.ndarray, **kwargs) -> Tuple[float, float, float]:
    return bootstrap_ci(errors, np.mean, **kwargs)


def rmse_ci(errors: np.ndarray, **kwargs) -> Tuple[float, float, float]:
    return bootstrap_ci(errors, lambda e: np.sqrt(np.mean(e**2)), **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Wilcoxon signed-rank test
# ─────────────────────────────────────────────────────────────────────────────

def wilcoxon_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
) -> Dict:
    """
    Wilcoxon signed-rank test on paired absolute errors (model A vs model B).
    H₀: median difference in absolute errors = 0.
    A small p-value (< 0.05) means the models differ significantly.

    Returns dict with: statistic, p_value, n, model_a_better (bool | None)
    """
    valid = ~(np.isnan(errors_a) | np.isnan(errors_b))
    a = errors_a[valid]
    b = errors_b[valid]
    n = int(valid.sum())
    if n < 10:
        return {"statistic": np.nan, "p_value": np.nan, "n": n, "model_a_better": None, "note": "too few samples"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p = stats.wilcoxon(a, b, zero_method="wilcox", correction=False, alternative="two-sided")
        return {
            "statistic": float(stat),
            "p_value": float(p),
            "n": n,
            "model_a_better": bool(np.median(a) < np.median(b)) if p < 0.05 else None,
        }
    except Exception as exc:
        return {"statistic": np.nan, "p_value": np.nan, "n": n, "model_a_better": None, "note": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Diebold-Mariano test
# ─────────────────────────────────────────────────────────────────────────────

def diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    horizon: int = 1,
    power: int = 2,
) -> Dict:
    """
    Diebold-Mariano test comparing forecast accuracy of two models.
    Uses MSE loss (power=2) by default; set power=1 for MAE loss.

    H₀: equal predictive accuracy.
    Returns dict with: dm_statistic, p_value, model_a_better (bool | None)
    """
    valid = ~(np.isnan(errors_a) | np.isnan(errors_b))
    a = errors_a[valid]
    b = errors_b[valid]
    n = int(valid.sum())
    if n < 10:
        return {"dm_statistic": np.nan, "p_value": np.nan, "n": n, "model_a_better": None}

    loss_a = a**power
    loss_b = b**power
    d = loss_a - loss_b
    d_bar = np.mean(d)

    # Newey-West HAC variance (truncation lag = horizon - 1)
    T = len(d)
    gamma_0 = np.var(d, ddof=1)
    gamma = [np.cov(d[h:], d[:-h])[0, 1] if h > 0 else gamma_0 for h in range(horizon)]
    var_d = (gamma_0 + 2 * sum(gamma[1:])) / T
    if var_d <= 0:
        return {"dm_statistic": np.nan, "p_value": np.nan, "n": n, "model_a_better": None}

    dm_stat = d_bar / np.sqrt(var_d)
    p_val = 2 * stats.norm.sf(abs(dm_stat))
    return {
        "dm_statistic": float(dm_stat),
        "p_value": float(p_val),
        "n": n,
        "model_a_better": bool(d_bar < 0) if p_val < 0.05 else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility: coefficient of variation across seeds
# ─────────────────────────────────────────────────────────────────────────────

def compute_reproducibility(
    df: pd.DataFrame,
    metric: str = "TOTALDEMAND",
    predicted_col: str = "predicted",
    group_col: str = "query_id",
    seed_col: str = "seed",
) -> pd.DataFrame:
    """
    For each query, compute the standard deviation and CV of predictions
    across repeated runs (seeds). Matches the paper's Table II metric.
    """
    records = []
    for qid, grp in df.groupby(group_col):
        preds = []
        for _, row in grp.iterrows():
            v = _parse_metric_from_predicted(row.get(predicted_col), metric)
            if v is not None and not np.isnan(v):
                preds.append(v)
        if len(preds) < 2:
            continue
        mu = float(np.mean(preds))
        sigma = float(np.std(preds, ddof=1))
        cv = sigma / abs(mu) if abs(mu) > 1e-9 else np.nan
        records.append({"query_id": qid, "n_seeds": len(preds), "mean": mu, "std": sigma, "cv": cv})
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Full analysis runner
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SignificanceConfig:
    metric: str = "TOTALDEMAND"
    stage: str = "reproducibility"
    n_bootstrap: int = 2000
    ci: float = 0.95
    alpha: float = 0.05


def run_significance_analysis(
    model_dfs: Dict[str, pd.DataFrame],
    cfg: SignificanceConfig = SignificanceConfig(),
) -> Dict:
    """
    Run full significance analysis across all model pairs.

    Args:
        model_dfs: dict mapping model name → DataFrame (loaded from eval CSV)
        cfg: analysis configuration

    Returns:
        dict with keys:
          - 'accuracy': per-model MAE/RMSE with 95% bootstrap CIs
          - 'pairwise_wilcoxon': all model pairs
          - 'pairwise_dm': all model pairs (Diebold-Mariano)
          - 'reproducibility': per-model CV across seeds
    """
    # Filter to target stage
    filtered: Dict[str, pd.DataFrame] = {}
    for name, df in model_dfs.items():
        if "stage" in df.columns:
            sub = df[df["stage"] == cfg.stage].copy()
        else:
            sub = df.copy()
        filtered[name] = sub

    # Build per-model error series keyed by (query_id, seed) for alignment
    model_error_series: Dict[str, pd.Series] = {}
    accuracy_rows = []

    for name, df in filtered.items():
        df = df.copy()
        df["_err"] = extract_errors(df, metric=cfg.metric).values

        # Build a stable key per row for alignment across models
        if "query_id" in df.columns and "seed" in df.columns:
            df["_key"] = df["query_id"].astype(str) + "__" + df["seed"].astype(str)
        elif "query_id" in df.columns:
            df["_key"] = df["query_id"].astype(str)
        else:
            df["_key"] = df.index.astype(str)

        err_series = df.set_index("_key")["_err"]
        # Aggregate duplicates by mean (multiple steps in a multi-step query)
        err_series = err_series.groupby(err_series.index).mean()
        model_error_series[name] = err_series

        errs = err_series.to_numpy(dtype=float)
        mae_pt, mae_lo, mae_hi = mae_ci(errs, n_bootstrap=cfg.n_bootstrap, ci=cfg.ci)
        rmse_pt, rmse_lo, rmse_hi = rmse_ci(errs, n_bootstrap=cfg.n_bootstrap, ci=cfg.ci)
        n_valid = int(np.sum(~np.isnan(errs)))
        accuracy_rows.append({
            "model": name,
            "metric": cfg.metric,
            "n": n_valid,
            "MAE": mae_pt,
            "MAE_lo": mae_lo,
            "MAE_hi": mae_hi,
            "RMSE": rmse_pt,
            "RMSE_lo": rmse_lo,
            "RMSE_hi": rmse_hi,
        })
    accuracy_df = pd.DataFrame(accuracy_rows).sort_values("MAE")

    def _aligned_pair(name_a: str, name_b: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return error arrays aligned on their shared keys."""
        sa = model_error_series[name_a]
        sb = model_error_series[name_b]
        shared = sa.index.intersection(sb.index)
        if len(shared) == 0:
            # Fall back to unaligned — may differ in length, tests will flag as too few
            return sa.to_numpy(dtype=float), sb.to_numpy(dtype=float)
        return sa.loc[shared].to_numpy(dtype=float), sb.loc[shared].to_numpy(dtype=float)

    # Pairwise Wilcoxon
    wilcoxon_rows = []
    for a, b in itertools.combinations(model_error_series.keys(), 2):
        ea, eb = _aligned_pair(a, b)
        result = wilcoxon_test(ea, eb)
        wilcoxon_rows.append({"model_a": a, "model_b": b, **result})
    wilcoxon_df = pd.DataFrame(wilcoxon_rows)

    # Pairwise Diebold-Mariano
    dm_rows = []
    for a, b in itertools.combinations(model_error_series.keys(), 2):
        ea, eb = _aligned_pair(a, b)
        result = diebold_mariano_test(ea, eb)
        dm_rows.append({"model_a": a, "model_b": b, **result})
    dm_df = pd.DataFrame(dm_rows)

    # Reproducibility across seeds
    repro_rows = []
    for name, df in filtered.items():
        repro = compute_reproducibility(df, metric=cfg.metric)
        if not repro.empty:
            repro_rows.append({
                "model": name,
                "mean_cv": float(repro["cv"].mean()),
                "mean_std": float(repro["std"].mean()),
                "n_queries": len(repro),
            })
    repro_df = pd.DataFrame(repro_rows)

    return {
        "accuracy": accuracy_df,
        "pairwise_wilcoxon": wilcoxon_df,
        "pairwise_dm": dm_df,
        "reproducibility": repro_df,
    }


def format_ci(pt, lo, hi, decimals=4) -> str:
    """Format a point estimate with CI as 'value [lo, hi]'."""
    if any(np.isnan(x) for x in [pt, lo, hi]):
        return "N/A"
    return f"{pt:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def print_significance_report(results: Dict, metric: str = "TOTALDEMAND"):
    """Print a human-readable significance report."""
    sep = "─" * 72

    print(f"\n{'═' * 72}")
    print(f"  Significance Report — {metric}")
    print(f"{'═' * 72}")

    print(f"\n{sep}")
    print("  Accuracy with 95% Bootstrap CIs (2000 resamples)")
    print(sep)
    acc = results["accuracy"]
    for _, r in acc.iterrows():
        mae_s = format_ci(r["MAE"], r["MAE_lo"], r["MAE_hi"])
        rmse_s = format_ci(r["RMSE"], r["RMSE_lo"], r["RMSE_hi"])
        print(f"  {r['model']:20s}  MAE: {mae_s}   RMSE: {rmse_s}   n={r['n']}")

    print(f"\n{sep}")
    print("  Pairwise Wilcoxon Signed-Rank Tests (H₀: equal median |error|)")
    print(sep)
    for _, r in results["pairwise_wilcoxon"].iterrows():
        sig = "✓" if (not np.isnan(r.get("p_value", np.nan)) and r.get("p_value", 1) < 0.05) else " "
        better = r.get("model_a_better")
        winner = r["model_a"] if better else r["model_b"] if better is False else "—"
        pv = f"{r.get('p_value', np.nan):.4f}" if not np.isnan(r.get("p_value", np.nan)) else "N/A"
        print(f"  [{sig}] {r['model_a']:15s} vs {r['model_b']:15s}  p={pv:>8s}  winner={winner}")

    print(f"\n{sep}")
    print("  Pairwise Diebold-Mariano Tests (MSE loss, H₀: equal predictive accuracy)")
    print(sep)
    for _, r in results["pairwise_dm"].iterrows():
        sig = "✓" if (not np.isnan(r.get("p_value", np.nan)) and r.get("p_value", 1) < 0.05) else " "
        better = r.get("model_a_better")
        winner = r["model_a"] if better else r["model_b"] if better is False else "—"
        pv = f"{r.get('p_value', np.nan):.4f}" if not np.isnan(r.get("p_value", np.nan)) else "N/A"
        dm = f"{r.get('dm_statistic', np.nan):.3f}" if not np.isnan(r.get("dm_statistic", np.nan)) else "N/A"
        print(f"  [{sig}] {r['model_a']:15s} vs {r['model_b']:15s}  DM={dm:>8s}  p={pv:>8s}  winner={winner}")

    if not results["reproducibility"].empty:
        print(f"\n{sep}")
        print("  Reproducibility (mean CV across queries)")
        print(sep)
        for _, r in results["reproducibility"].iterrows():
            cv_s = f"{r['mean_cv']:.4f}" if not np.isnan(r["mean_cv"]) else "N/A"
            print(f"  {r['model']:20s}  mean CV={cv_s}  n_queries={r['n_queries']}")

    print(f"\n{'═' * 72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Statistical significance testing for forecast comparison")
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more eval CSVs (e.g. Claude_eval_with_gt.csv)")
    parser.add_argument("--stage", default="reproducibility", help="Stage to filter (default: reproducibility)")
    parser.add_argument("--metric", default="TOTALDEMAND", choices=["TOTALDEMAND", "RRP"])
    parser.add_argument("--output", default="experiments/eval/significance_results", help="Output directory")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--ci", type=float, default=0.95)
    args = parser.parse_args()

    model_dfs: Dict[str, pd.DataFrame] = {}
    for path in args.inputs:
        name = os.path.splitext(os.path.basename(path))[0].replace("_eval_with_gt", "")
        try:
            df = pd.read_csv(path, low_memory=False)
            model_dfs[name] = df
            print(f"Loaded {name}: {len(df)} rows")
        except Exception as exc:
            print(f"WARNING: Could not load {path}: {exc}", file=__import__("sys").stderr)

    if not model_dfs:
        print("ERROR: No data loaded.", file=__import__("sys").stderr)
        return 1

    cfg = SignificanceConfig(
        metric=args.metric,
        stage=args.stage,
        n_bootstrap=args.n_bootstrap,
        ci=args.ci,
    )
    results = run_significance_analysis(model_dfs, cfg)
    print_significance_report(results, metric=args.metric)

    os.makedirs(args.output, exist_ok=True)
    results["accuracy"].to_csv(os.path.join(args.output, f"accuracy_{args.metric}.csv"), index=False)
    results["pairwise_wilcoxon"].to_csv(os.path.join(args.output, f"wilcoxon_{args.metric}.csv"), index=False)
    results["pairwise_dm"].to_csv(os.path.join(args.output, f"dm_{args.metric}.csv"), index=False)
    results["reproducibility"].to_csv(os.path.join(args.output, "reproducibility.csv"), index=False)
    print(f"Results saved to {args.output}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
