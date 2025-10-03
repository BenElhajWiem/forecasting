from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

# ===================== Config =====================
@dataclass
class StatConfig:
    """Configuration for the Statistical Agent."""
    tz: str = "Australia/Sydney"
    percentiles: List[float] = (1, 5, 10, 25, 50, 75, 90, 95, 99)
    acf_lags_hours: List[int] = (1, 24, 168)  # 1h, 1d, 7d
    non_metric_cols: Tuple[str, ...] = (
        "REGION","SETTLEMENTDATE","ret_block","ret_score","PERIODTYPE",
        "PRIOR_YEAR","LAG_DAYS","PRIOR_WOY"
    )
    weighted_by_ret_score: bool = True

# ===================== Time helpers =====================
def localize_to_timezone(series: pd.Series, tz: str) -> pd.Series:
    """Return tz-aware timestamps localized to `tz` (no conversion assumed)."""
    s = pd.to_datetime(series, errors="coerce", utc=False)
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    return s

def infer_sampling_interval_seconds(ts: pd.Series) -> Optional[float]:
    """Infer the dominant sample interval (in seconds) from a timestamp series."""
    if ts is None or ts.empty:
        return None
    diffs = ts.sort_values().diff().dropna().dt.total_seconds()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return None
    rounded_minutes = (diffs / 60.0).round().astype(int)
    if rounded_minutes.empty:
        return float(diffs.median())
    step_min = int(rounded_minutes.mode().iloc[0])
    return float(step_min * 60)

def convert_hours_to_steps(hours: int, step_seconds: Optional[float]) -> Optional[int]:
    """Convert a duration in hours to number of samples given a step size in seconds."""
    if not step_seconds or step_seconds <= 0:
        return None
    steps = int(round((hours * 3600) / step_seconds))
    return steps if steps > 0 else None

# ===================== Metric helpers =====================
def detect_metric_columns(df: pd.DataFrame, preferred: Optional[List[str]], non_metrics: Tuple[str, ...]) -> List[str]:
    """Infer metric columns (numeric) excluding known non-metric fields."""
    if preferred:
        return [m for m in preferred if m in df.columns]
    candidates = [c for c in df.columns if c not in non_metrics]
    out = []
    for c in candidates:
        try:
            if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0:
                out.append(c)
        except Exception:
            pass
    return out

def compute_basic_statistics(series: pd.Series, percentiles: List[float]) -> Dict[str, Any]:
    """Compute count/mean/std/min/max and requested percentiles for a numeric series."""
    x = pd.to_numeric(series, errors="coerce").astype(float).dropna()
    if x.empty:
        return {"count": 0}
    q = np.percentile(x, percentiles, method="linear")
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "min": float(x.min()),
        **{f"p{int(p)}": float(q[i]) for i, p in enumerate(percentiles)},
        "max": float(x.max()),
    }

def compute_weighted_mean(values: pd.Series, weights: pd.Series) -> Optional[float]:
    """Compute a weighted mean (e.g., by retrieval score)."""
    v = pd.to_numeric(values, errors="coerce").astype(float)
    w = pd.to_numeric(weights, errors="coerce").astype(float).clip(lower=0)
    mask = v.notna() & w.notna()
    if mask.sum() == 0 or w[mask].sum() == 0:
        return None
    return float(np.average(v[mask], weights=w[mask]))

def compute_autocorrelation_at_lags(series: pd.Series, lags_in_steps: List[int]) -> Dict[str, Optional[float]]:
    """Compute autocorrelation for specific integer lags (in samples)."""
    out: Dict[str, Optional[float]] = {}
    s = pd.to_numeric(series, errors="coerce").astype(float).dropna()
    if s.size < 3:
        return {f"lag{L}": None for L in lags_in_steps}
    ps = pd.Series(s)
    for L in lags_in_steps:
        out[f"lag{L}"] = None if L is None or L <= 0 or L >= len(ps) else float(ps.autocorr(L))
    return out

def compute_time_profiles(df: pd.DataFrame, metric_cols: List[str], tz: str) -> Dict[str, Any]:
    """Median profiles by hour, weekday, month for each metric."""
    if df.empty:
        return {"hourly": {}, "weekday": {}, "month": {}}
    frame = df.copy()
    ts = localize_to_timezone(frame["SETTLEMENTDATE"], tz)
    frame["__hour"] = ts.dt.hour
    frame["__wk"]   = ts.dt.weekday
    frame["__mon"]  = ts.dt.month
    profiles = {"hourly": {}, "weekday": {}, "month": {}}
    for m in metric_cols:
        if m not in frame.columns:
            continue
        s = pd.to_numeric(frame[m], errors="coerce")
        profiles["hourly"][m]  = [float(x) if pd.notna(x) else None for x in s.groupby(frame["__hour"]).median().reindex(range(24)).values]
        profiles["weekday"][m] = [float(x) if pd.notna(x) else None for x in s.groupby(frame["__wk"]).median().reindex(range(7)).values]
        profiles["month"][m]   = [float(x) if pd.notna(x) else None for x in s.groupby(frame["__mon"]).median().reindex(range(1,13)).values]
    return profiles

def compute_correlation_matrix(df: pd.DataFrame, metric_cols: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    """Pearson correlation matrix among metric columns."""
    if df.empty or not metric_cols:
        return {}
    sub = df[metric_cols].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr(method="pearson", min_periods=8)
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for a in metric_cols:
        out[a] = {}
        for b in metric_cols:
            val = corr.loc[a, b] if (a in corr.index and b in corr.columns) else np.nan
            out[a][b] = float(val) if pd.notna(val) else None
    return out

def compute_gap_report(ts: pd.Series, expected_step_seconds: Optional[float]) -> Dict[str, Any]:
    """Report count/size of gaps vs expected step, and duplicate timestamp count."""
    if ts is None or ts.empty:
        return {"expected_step_seconds": expected_step_seconds, "gap_count": 0, "max_gap_steps": 0, "dup_ts": 0}
    ordered = ts.sort_values()
    diffs = ordered.diff().dropna().dt.total_seconds()
    duplicates = int((ordered.size - ordered.drop_duplicates().size))
    if not expected_step_seconds or expected_step_seconds <= 0 or diffs.empty:
        return {"expected_step_seconds": expected_step_seconds, "gap_count": 0, "max_gap_steps": 0, "dup_ts": duplicates}
    gaps = diffs[diffs > 1.5 * expected_step_seconds]
    max_gap_steps = int(round(gaps.max() / expected_step_seconds)) if not gaps.empty else 0
    return {
        "expected_step_seconds": expected_step_seconds,
        "gap_count": int(gaps.size),
        "max_gap_steps": max_gap_steps,
        "dup_ts": duplicates
    }

# ===================== Main Agent =====================
class StatisticalAgent:
    """
    Statistical Agent:
      Input: retrieval_out (dict of DataFrames + meta)
      Output: per-origin stats, profiles, ACF, gaps, correlations.
    """
    def __init__(self, cfg: StatConfig = StatConfig()):
        self.cfg = cfg

    def run(self, retrieval_out: Dict[str, Any], metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compute statistics for each retrieved origin and global correlations."""
        origins = {k: v for k, v in retrieval_out.items() if isinstance(v, pd.DataFrame) and not v.empty}
        meta = retrieval_out.get("meta", {}) or {}
        if not origins:
            return {"meta": meta, "per_origin": {}, "global": {}}

        # Decide metrics from union if not provided
        if metrics is None:
            union_cols = set().union(*[set(df.columns) for df in origins.values()])
            metrics = detect_metric_columns(pd.DataFrame(columns=sorted(union_cols)), None, self.cfg.non_metric_cols)
            preferred = [m for m in ["RRP","TOTALDEMAND"] if m in union_cols]
            metrics = preferred + [m for m in metrics if m not in preferred]

        per_origin: Dict[str, Any] = {}
        global_df = origins.get("combined", pd.concat(list(origins.values()), ignore_index=True))

        for origin_name, frame in origins.items():
            per_origin[origin_name] = self.summarize_origin_block(origin_name, frame, metrics)

        global_out = {
            "metrics": metrics,
            "correlations": compute_correlation_matrix(global_df, metrics),
        }

        return {
            "meta": {
                "route": meta.get("route"),
                "anchor_iso": meta.get("anchor_iso"),
                "freq_hint": meta.get("freq"),
                "tz": meta.get("tz") or self.cfg.tz,
                "origins": list(origins.keys()),
            },
            "per_origin": per_origin,
            "global": global_out
        }

    def summarize_origin_block(self, origin: str, df: pd.DataFrame, metrics: List[str]) -> Dict[str, Any]:
        """Produce a compact statistical summary for a single origin block."""
        tz = self.cfg.tz
        frame = df.copy()
        ts_local = localize_to_timezone(frame["SETTLEMENTDATE"], tz)
        frame["__ts"] = ts_local
        frame = frame.dropna(subset=["__ts"]).sort_values("__ts")

        step_seconds = infer_sampling_interval_seconds(frame["__ts"])
        lags_in_steps = [
            s for s in [convert_hours_to_steps(h, step_seconds) for h in self.cfg.acf_lags_hours] if s is not None
        ]

        # Overall stats per metric
        stats = {m: compute_basic_statistics(frame[m], list(self.cfg.percentiles)) for m in metrics if m in frame.columns}

        # Optional weighted mean (ret_score)
        if self.cfg.weighted_by_ret_score and "ret_score" in frame.columns:
            for m in metrics:
                if m in frame.columns:
                    wmean = compute_weighted_mean(frame[m], frame["ret_score"])
                    if wmean is not None:
                        stats[m]["w_mean_ret_score"] = wmean

        # Autocorrelation per metric
        acf = {m: compute_autocorrelation_at_lags(frame[m], lags_in_steps) for m in metrics if m in frame.columns}

        # Profiles (hourly / weekday / month)
        profiles = compute_time_profiles(frame, metrics, tz)

        # Gaps / duplicates
        gaps = compute_gap_report(frame["__ts"], step_seconds)

        # Time span & regions
        tmin, tmax = frame["__ts"].min(), frame["__ts"].max()
        regions = sorted([str(x).upper() for x in frame["REGION"].unique()]) if "REGION" in frame.columns else []

        # Per-region light stats
        by_region: Dict[str, Any] = {}
        if "REGION" in frame.columns:
            for r, sub in frame.groupby(frame["REGION"]):
                by_region[str(r).upper()] = {
                    "rows": int(len(sub)),
                    "stats": {m: compute_basic_statistics(sub[m], list(self.cfg.percentiles)) for m in metrics if m in sub.columns}
                }

        # Small correlation matrix for this origin
        corr = compute_correlation_matrix(frame, metrics)

        preview_cols = [c for c in ["REGION","SETTLEMENTDATE","RRP","TOTALDEMAND","ret_block","ret_score"] if c in frame.columns]
        return {
            "origin": origin,
            "rows": int(len(frame)),
            "regions": regions,
            "time_span": {
                "start": tmin.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(tmin) else None,
                "end":   tmax.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(tmax) else None,
                "duration_hours": float((tmax - tmin).total_seconds()/3600.0) if (pd.notna(tmin) and pd.notna(tmax)) else None,
            },
            "inferred_step_seconds": step_seconds,
            "stats": stats,
            "acf": acf,
            "profiles": profiles,
            "gaps": gaps,
            "corr": corr,
            "by_region": by_region,
            "preview_cols": preview_cols,
            "preview_head": frame[preview_cols].head(5).to_dict(orient="records")
        }
