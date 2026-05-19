"""
Full significance analysis:
  - Bootstrap 95% CIs on MAE/RMSE for every method
  - Wilcoxon signed-rank + Diebold-Mariano for every LLM×baseline pair
  - Produces LaTeX-ready comparison table
"""
from __future__ import annotations

import ast
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
from scipy.special import ndtri   # for normal quantile

ROOT = Path("experiments")
EVAL_DIR = ROOT / "eval" / "predicted_vs_gt"
BASELINE_CSV = ROOT / "baselines" / "baseline_results.csv"
NBOOT = 2000
CI = 0.95
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_gt(raw, metric: str) -> float | None:
    if isinstance(raw, dict):
        entry = raw.get(metric)
    elif isinstance(raw, str) and raw.strip():
        try:
            d = ast.literal_eval(raw)
        except Exception:
            return None
        entry = d.get(metric) if isinstance(d, dict) else None
    else:
        return None
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, list) and entry:
        return float(entry[0]) if not isinstance(entry[0], list) else float(entry[0][0])
    return None


def _parse_pred(raw, metric: str) -> float | None:
    if isinstance(raw, (int, float)) and not np.isnan(raw):
        return float(raw)
    if isinstance(raw, str):
        # try dict literal first
        try:
            d = ast.literal_eval(raw)
            if isinstance(d, dict) and metric in d:
                v = d[metric]
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, list) and v:
                    return float(v[0]) if not isinstance(v[0], list) else float(v[0][0])
        except Exception:
            pass
        # regex fallback
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", raw.replace(",", ""))
        if nums:
            return float(nums[0])
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Load LLM model errors
# ─────────────────────────────────────────────────────────────────────────────

def load_llm_errors(metric: str) -> dict[str, np.ndarray]:
    """Returns {model_name: array of absolute errors} for reproducibility rows."""
    model_errors: dict[str, list[float]] = {}
    for csv_path in sorted(EVAL_DIR.glob("*_eval_with_gt.csv")):
        model = csv_path.stem.replace("_eval_with_gt", "")
        df = pd.read_csv(csv_path)
        df = df[df["stage"] == "reproducibility"] if "stage" in df.columns else df
        errs: list[float] = []
        for _, row in df.iterrows():
            gt = _parse_gt(row.get("ground_truth"), metric)
            pred = _parse_pred(row.get("predicted"), metric)
            if gt is not None and pred is not None:
                mean_gt = abs(gt)
                if mean_gt > 0:
                    errs.append(abs(pred - gt) / mean_gt)
        if errs:
            model_errors[model] = np.array(errs)
    return model_errors


def load_llm_errors_by_query(metric: str) -> dict[str, pd.Series]:
    """Returns {model: Series indexed by query_id of mean normalised abs error}."""
    out: dict[str, pd.Series] = {}
    for csv_path in sorted(EVAL_DIR.glob("*_eval_with_gt.csv")):
        model = csv_path.stem.replace("_eval_with_gt", "")
        df = pd.read_csv(csv_path)
        if "stage" in df.columns:
            df = df[df["stage"] == "reproducibility"]
        rows: list[dict] = []
        for _, row in df.iterrows():
            gt = _parse_gt(row.get("ground_truth"), metric)
            pred = _parse_pred(row.get("predicted"), metric)
            if gt is not None and pred is not None and abs(gt) > 0:
                rows.append({
                    "query_id": str(row.get("query_id", row.get("exp_id", "?"))),
                    "seed": str(row.get("seed", "0")),
                    "err": abs(pred - gt) / abs(gt),
                })
        if rows:
            tmp = pd.DataFrame(rows)
            tmp["_key"] = tmp["query_id"] + "__" + tmp["seed"]
            s = tmp.groupby("_key")["err"].mean()
            out[model] = s
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Load baseline errors
# ─────────────────────────────────────────────────────────────────────────────

def load_baseline_errors(metric: str) -> dict[str, pd.Series]:
    """
    Returns {baseline_name: Series indexed by query_id of normalised abs error}.
    Also returns the mean GT for normalisation.
    """
    bdf = pd.read_csv(BASELINE_CSV)
    sub = bdf[bdf["metric"] == metric].copy()
    sub["gt"] = pd.to_numeric(sub["ground_truth"], errors="coerce")
    sub = sub.dropna(subset=["gt"])
    sub = sub[sub["gt"].abs() > 0]

    baseline_cols = [c for c in ["persistence", "seasonal_naive", "sarima", "chronos", "tft"]
                     if c in sub.columns]
    out: dict[str, pd.Series] = {}
    for col in baseline_cols:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        valid = sub.dropna(subset=[col])
        if valid.empty:
            continue
        errs = (valid[col] - valid["gt"]).abs() / valid["gt"].abs()
        s = pd.Series(errs.values, index=valid["query_id"].values, name=col)
        out[col] = s
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(arr: np.ndarray, stat=np.mean, n=NBOOT, ci=CI, seed=SEED):
    rng = np.random.default_rng(seed)
    boot = np.array([stat(rng.choice(arr, len(arr), replace=True)) for _ in range(n)])
    lo, hi = np.percentile(boot, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return stat(arr), lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# Diebold-Mariano test (Harvey et al. small-sample correction)
# ─────────────────────────────────────────────────────────────────────────────

def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> dict:
    d = np.abs(e1) - np.abs(e2)
    n = len(d)
    if n < 4:
        return {"statistic": np.nan, "p_value": np.nan, "n": n}
    # Newey-West HAC variance
    gamma0 = np.var(d, ddof=1)
    nw = gamma0
    for lag in range(1, min(h, n // 4) + 1):
        cov = np.cov(d[lag:], d[:-lag])[0, 1]
        nw += 2 * (1 - lag / (h + 1)) * cov
    var_d = nw / n
    if var_d <= 0:
        return {"statistic": np.nan, "p_value": np.nan, "n": n}
    dm = d.mean() / np.sqrt(var_d)
    # Two-sided p from standard normal
    p = 2 * (1 - 0.5 * (1 + np.sign(dm) * (1 - 2 * (1 - abs(dm) < 0 and True or
        float(__import__("scipy.stats", fromlist=["norm"]).norm.cdf(abs(dm)) - 0.5) / 0.5))))
    from scipy.stats import norm
    p = 2 * norm.sf(abs(dm))
    return {"statistic": float(dm), "p_value": float(p), "n": int(n)}


def wilcoxon_test(e1: np.ndarray, e2: np.ndarray) -> dict:
    d = np.abs(e1) - np.abs(e2)
    d = d[d != 0]
    if len(d) < 10:
        return {"statistic": np.nan, "p_value": np.nan, "n": len(d)}
    try:
        stat, p = wilcoxon(d, alternative="two-sided")
        return {"statistic": float(stat), "p_value": float(p), "n": len(d)}
    except Exception:
        return {"statistic": np.nan, "p_value": np.nan, "n": len(d)}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_full_analysis():
    for metric in ["TOTALDEMAND", "RRP"]:
        print(f"\n{'='*90}")
        print(f" METRIC: {metric}")
        print(f"{'='*90}")

        llm_flat   = load_llm_errors(metric)
        llm_series = load_llm_errors_by_query(metric)
        bl_series  = load_baseline_errors(metric)

        # ── 1. Bootstrap CIs for every method ────────────────────────────────
        print(f"\n{'─'*90}")
        print(f" Bootstrap 95% CIs (MAE, normalised)  —  {NBOOT} resamples")
        print(f"{'─'*90}")
        print(f"  {'Method':22s}  {'MAE':>8s}  {'CI_lo':>8s}  {'CI_hi':>8s}  {'n':>5s}")
        print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*5}")

        ci_results: dict[str, tuple] = {}

        # LLM models
        for model, errs in sorted(llm_flat.items()):
            if len(errs) == 0:
                continue
            mae, lo, hi = bootstrap_ci(errs)
            ci_results[model] = (mae, lo, hi, len(errs))
            print(f"  {model:22s}  {mae:.4f}    [{lo:.4f}, {hi:.4f}]  {len(errs):5d}")

        # Baselines (use raw errors per query, concatenate)
        for bl_name, bl_ser in sorted(bl_series.items()):
            errs = bl_ser.values.astype(float)
            errs = errs[~np.isnan(errs)]
            if len(errs) == 0:
                continue
            mae, lo, hi = bootstrap_ci(errs)
            ci_results[bl_name] = (mae, lo, hi, len(errs))
            print(f"  {bl_name:22s}  {mae:.4f}    [{lo:.4f}, {hi:.4f}]  {len(errs):5d}")

        # ── 2. Pairwise significance tests (LLM vs baselines) ────────────────
        print(f"\n{'─'*90}")
        print(f" Pairwise significance: each LLM vs each baseline")
        print(f" Wilcoxon (W) + Diebold-Mariano (DM)  |  * p<.05  ** p<.01  *** p<.001")
        print(f"{'─'*90}")
        print(f"  {'LLM':12s}  {'Baseline':20s}  {'W-stat':>8s}  {'W-p':>7s}  {'DM-stat':>8s}  {'DM-p':>7s}  sig  n")

        sig_results = []
        for model_name, model_ser in sorted(llm_series.items()):
            for bl_name, bl_ser in sorted(bl_series.items()):
                shared = model_ser.index.intersection(bl_ser.index)
                if len(shared) < 5:
                    continue
                e_llm = model_ser.loc[shared].values.astype(float)
                e_bl  = bl_ser.loc[shared].values.astype(float)
                valid = ~(np.isnan(e_llm) | np.isnan(e_bl))
                e_llm, e_bl = e_llm[valid], e_bl[valid]
                if len(e_llm) < 5:
                    continue

                w  = wilcoxon_test(e_llm, e_bl)
                dm = diebold_mariano(e_llm, e_bl)

                p_min = min(
                    w["p_value"]  if not np.isnan(w["p_value"])  else 1,
                    dm["p_value"] if not np.isnan(dm["p_value"]) else 1,
                )
                sig = "***" if p_min < 0.001 else "**" if p_min < 0.01 else "*" if p_min < 0.05 else "n.s."

                # direction: negative DM → LLM better; positive → baseline better
                direction = "LLM<BL" if dm["statistic"] < 0 else "LLM>BL"

                print(f"  {model_name:12s}  {bl_name:20s}  {w['statistic']:8.2f}  {w['p_value']:7.4f}  "
                      f"{dm['statistic']:8.3f}  {dm['p_value']:7.4f}  {sig:4s}  {len(e_llm)}")
                sig_results.append({
                    "metric": metric, "llm": model_name, "baseline": bl_name,
                    "llm_mae": ci_results.get(model_name, (np.nan,))[0],
                    "bl_mae":  ci_results.get(bl_name, (np.nan,))[0],
                    "wilcoxon_stat": w["statistic"], "wilcoxon_p": w["p_value"],
                    "dm_stat": dm["statistic"], "dm_p": dm["p_value"],
                    "sig": sig, "direction": direction, "n_shared": len(e_llm),
                })

        # ── 3. LaTeX table rows ───────────────────────────────────────────────
        print(f"\n{'─'*90}")
        print(f" LaTeX-ready comparison table rows ({metric})")
        print(f"{'─'*90}")
        llm_order = ["Claude", "Gemini", "OpenAI", "DeepSeek"]
        bl_order  = [b for b in ["persistence","seasonal_naive","sarima","chronos","tft"]
                     if b in ci_results]
        # header
        bl_labels = {"persistence":"Persistence","seasonal_naive":"Seasonal Naive",
                     "sarima":"SARIMA","chronos":"Chronos","tft":"TFT"}
        header_cols = " & ".join(f"\\textbf{{{bl_labels.get(b, b)}}}" for b in bl_order)
        print(f"  Method & {header_cols} \\\\")
        print(f"  \\midrule")
        for m_name in llm_order:
            if m_name not in ci_results:
                continue
            mae_m, lo_m, hi_m, _ = ci_results[m_name]
            cells = [f"\\textbf{{{m_name}}} & {mae_m:.4f} [{lo_m:.4f}, {hi_m:.4f}]"]
            for bl_name in bl_order:
                # find sig result
                match = [r for r in sig_results
                         if r["metric"]==metric and r["llm"]==m_name and r["baseline"]==bl_name]
                if match:
                    r = match[0]
                    p = min(r["wilcoxon_p"] if not np.isnan(r["wilcoxon_p"]) else 1,
                            r["dm_p"]       if not np.isnan(r["dm_p"])       else 1)
                    star = "^{***}" if p < 0.001 else "^{**}" if p < 0.01 else "^{*}" if p < 0.05 else ""
                    bl_mae, bl_lo, bl_hi, _ = ci_results.get(bl_name, (np.nan, np.nan, np.nan, 0))
                    cells.append(f"${star}$" if star else "$-$")
                else:
                    cells.append("$-$")
            print("  " + " & ".join(cells) + " \\\\")

        # Baseline rows
        print(f"  \\midrule")
        for bl_name in bl_order:
            if bl_name not in ci_results:
                continue
            mae, lo, hi, n = ci_results[bl_name]
            label = bl_labels.get(bl_name, bl_name)
            print(f"  {label} & {mae:.4f} [{lo:.4f}, {hi:.4f}] \\\\ % n={n}")

        # Save sig_results
        sig_df = pd.DataFrame(sig_results)
        out_path = ROOT / "eval" / f"significance_{metric}.csv"
        sig_df.to_csv(out_path, index=False)
        print(f"\n  Saved significance table to {out_path}")


if __name__ == "__main__":
    run_full_analysis()
