# pattern_agent.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import json
import math

# =======================
# 1) Configuration
# =======================
@dataclass
class PatternConfig:
    tz: str = "Australia/Sydney"
    metrics: Tuple[str, ...] = ("TOTALDEMAND", "RRP")

    # Rolling-z anomalies
    z_threshold: float = 3.0
    rolling_window: str = "7D"
    min_points: int = 24
    top_k_anomalies: int = 20

    # Trend & ACF
    resample_for_trend: str = "1h"      # resample to stabilize slope
    acf_lags_hours: Tuple[int, ...] = (1, 2, 3, 6, 12, 24, 48, 72, 168)  # include daily/weekly taps

    # Payload limits
    max_regions_per_origin: int = 6
    max_metrics_per_region: int = 2
    max_points_in_thumbnail: int = 360  # series thumbnail length for LLM

@dataclass
class LLMConfig:
    model: str = None
    temperature: float = 0.2
    max_tokens_out: int = None
    json_mode: bool = True              # we want JSON labels back
    model_override: Optional[str] = None

# =======================
# 2) Small utilities
# =======================
def localize_to_timezone(series: pd.Series, tz: str) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", utc=False)
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    return s

def series_indexed_by_time(df: pd.DataFrame, metric: str, tz: str) -> pd.Series:
    ts = localize_to_timezone(df["SETTLEMENTDATE"], tz)
    x = pd.to_numeric(df[metric], errors="coerce")
    return pd.Series(x.values, index=ts.values).sort_index()

def downsample_to_thumbnail(s: pd.Series, max_points: int) -> List[float]:
    s = s.dropna()
    if len(s) <= max_points:
        return [float(x) for x in s.values]
    step = int(np.ceil(len(s) / max_points))
    return [float(x) for x in s.iloc[::step].values[:max_points]]

# =======================
# 3) Numeric feature builders (lightweight)
# =======================
def compute_trend_slope_per_hour(s: pd.Series, resample: str) -> Optional[float]:
    """OLS slope on resampled mean series (value per hour)."""
    y = s.resample(resample).mean().dropna()
    if len(y) < 5:
        return None
    t0 = y.index[0]
    t_hours = (y.index - t0) / np.timedelta64(1, "h")
    X = np.vstack([t_hours, np.ones_like(t_hours)]).T
    beta, *_ = np.linalg.lstsq(X, y.values, rcond=None)
    return float(beta[0])

def compute_autocorrelation_scores(s: pd.Series, acf_lags_hours: Tuple[int, ...]) -> Dict[str, float]:
    """Return ACF at requested hour lags using normalized correlation."""
    y = s.asfreq(pd.infer_freq(s.index) or "h")  # try to regularize; fallback hourly
    y = y.interpolate(limit_direction="both").dropna()
    out: Dict[str, float] = {}
    if len(y) < 10:
        return {f"lag_{h}h": 0.0 for h in acf_lags_hours}
    arr = y.values.astype(float)
    arr = arr - arr.mean()
    denom = float((arr ** 2).sum()) or 1.0

    def acf_k(k: int) -> float:
        if k <= 0 or k >= len(arr): return 0.0
        return float((arr[:-k] * arr[k:]).sum() / denom)

    # estimate step hours from index
    step_hours = max(1, int(round((y.index[1] - y.index[0]) / np.timedelta64(1, "h"))))
    for h in acf_lags_hours:
        k = int(round(h / step_hours))
        out[f"lag_{h}h"] = acf_k(k)
    return out

def detect_rolling_z_anomalies(s: pd.Series, window: str, min_points: int, z_thresh: float, top_k: int) -> List[Dict[str, Any]]:
    mu = s.rolling(window, min_periods=min_points).mean()
    sd = s.rolling(window, min_periods=min_points).std(ddof=1).replace(0, np.nan)
    z = (s - mu) / sd
    mask = (z.abs() >= z_thresh).fillna(False)
    hits = []
    for t in s.index[mask]:
        hits.append({
            "ts": pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S"),
            "z": float(z.loc[t]) if pd.notna(z.loc[t]) else None,
            "value": float(s.loc[t]) if pd.notna(s.loc[t]) else None,
            "type": "spike" if (pd.notna(z.loc[t]) and z.loc[t] > 0) else "drop"
        })
    # keep top |z|
    hits.sort(key=lambda e: abs(e.get("z") or 0.0), reverse=True)
    return hits[:top_k]

def compute_hourly_weekday_profiles(df: pd.DataFrame, metric: str, tz: str) -> Dict[str, List[Optional[float]]]:
    ts = localize_to_timezone(df["SETTLEMENTDATE"], tz)
    x = pd.to_numeric(df[metric], errors="coerce")
    frame = pd.DataFrame({"x": x, "__hour": ts.dt.hour, "__wk": ts.dt.weekday})
    hourly = frame.groupby("__hour")["x"].median().reindex(range(24)).tolist()
    weekday = frame.groupby("__wk")["x"].median().reindex(range(7)).tolist()
    return {"hourly_median": [float(v) if pd.notna(v) else None for v in hourly],
            "weekday_median": [float(v) if pd.notna(v) else None for v in weekday]}

# =======================
# 4) Evidence builder per origin/region/metric
# =======================
def build_llm_evidence_for_origin(df: pd.DataFrame, pcfg: PatternConfig) -> Dict[str, Any]:
    """Compute minimal numeric features the LLM will interpret into labels."""
    out: Dict[str, Any] = {"summary": {}, "items": []}
    ts = localize_to_timezone(df["SETTLEMENTDATE"], pcfg.tz)
    out["summary"] = {
        "rows": int(len(df)),
        "start": ts.min().strftime("%Y-%m-%d %H:%M:%S") if len(ts) else None,
        "end":   ts.max().strftime("%Y-%m-%d %H:%M:%S") if len(ts) else None,
        "regions": sorted(df["REGION"].astype(str).str.upper().unique()) if "REGION" in df.columns else []
    }

    region_groups = list(df.groupby(df["REGION"].astype(str).str.upper()))[:pcfg.max_regions_per_origin]
    for reg, sub in region_groups:
        for m in pcfg.metrics[:pcfg.max_metrics_per_region]:
            if m not in sub.columns: 
                continue
            s = series_indexed_by_time(sub, m, pcfg.tz).dropna()
            if s.empty: 
                continue

            slope = compute_trend_slope_per_hour(s, pcfg.resample_for_trend)
            acf = compute_autocorrelation_scores(s, pcfg.acf_lags_hours)
            anomalies = detect_rolling_z_anomalies(s, pcfg.rolling_window, pcfg.min_points, pcfg.z_threshold, pcfg.top_k_anomalies)
            profiles = compute_hourly_weekday_profiles(sub, m, pcfg.tz)
            thumb = downsample_to_thumbnail(s, pcfg.max_points_in_thumbnail)

            out["items"].append({
                "region": reg, "metric": m,
                "trend_slope_per_hour": slope,
                "acf": acf,
                "anomalies": anomalies,
                "profiles": profiles,
                "series_thumbnail": thumb  # short numeric trace for pattern “feel”
            })
    return out

# =======================
# 5) LLM: classify patterns from evidence
# =======================
def ask_llm_to_label_patterns(
    adapter,
    evidence: Dict[str, Any],
    pcfg: PatternConfig,
    lcfg: LLMConfig
) -> Dict[str, Any]:
    """
    Ask the LLM to decide: trend, seasonality (daily/weekly), cycles, anomalies.
    Returns structured JSON only.
    """
    system = (
        "You are a time-series expert. "
        "Given numeric evidence (trend slopes, ACF taps, profiles, short series thumbnails, anomaly counts), "
        "classify patterns WITHOUT inventing numbers. Return valid JSON only."
    )

    # Clear rules to keep LLM consistent
    rules = {
        "trend_strength_by_abs_slope_per_hour": {"none":"<0.01", "weak":"0.01-0.05", "moderate":"0.05-0.15", "strong":">0.15"},
        "seasonality_presence_threshold": 0.3,  # ACF >= 0.3 present, >=0.5 strong
        "strong_seasonality_threshold": 0.5,
        "anomaly_count_field": "len(anomalies)",
        "notes": [
            "Use ACF lag_24h for daily, lag_168h for weekly.",
            "Profiles (hourly/weekday medians) can support or refute ACF conclusions.",
            "If thumbnails look highly irregular with low ACF, call it 'random/high-noise'.",
        ]
    }

    schema = {
        "origin": "string",
        "summaries": [
            {
                "region": "string",
                "metric": "string",
                "trend": {"direction":"up|down|flat", "strength":"none|weak|moderate|strong", "slope_per_hour": "float|null"},
                "seasonality": {
                    "daily": {"present": "bool", "strength": "none|weak|moderate|strong", "acf": "float|null"},
                    "weekly":{"present": "bool", "strength": "none|weak|moderate|strong", "acf": "float|null"}
                },
                "cycles": [
                    {"period_hours":"int", "evidence":"acf|profile|visual", "confidence":"low|medium|high"}
                ],
                "anomalies": {"count":"int","types_present":["spike|drop"], "max_abs_z":"float|null"},
                "noise_level": "low|medium|high",
                "remarks": "short text"
            }
        ]
    }

    user = f"""
RULES (strict, use thresholds):
{json.dumps(rules, ensure_ascii=False)}

SCHEMA (return this shape, JSON only):
{json.dumps(schema, ensure_ascii=False)}

EVIDENCE:
{json.dumps(evidence, ensure_ascii=False)[:15000]}
"""

    messages = [{"role":"system","content":system},{"role":"user","content":user}]
    # We expect JSON back
    out = adapter.chat_json_loose(
        messages,
        temperature=lcfg.temperature,
        max_tokens=lcfg.max_tokens_out,
        model_override=(lcfg.model_override or lcfg.model),
        strict_json_first=True,
    )
    return out if isinstance(out, dict) else {"error": "LLM did not return JSON"}

# =======================
# 6) End-to-end: detect patterns after retrieval
# =======================
def detect_patterns_with_llm_after_retrieval(
    adapter,
    retrieval_out: Dict[str, Any],
    *,
    pcfg: PatternConfig = PatternConfig(),
    lcfg: LLMConfig = LLMConfig(),
    origins: Optional[List[str]] = None,
    return_bundle: bool = True
) -> Dict[str, Any] | str:
    """
    Build minimal numeric evidence from retrieved slices and let the LLM classify:
      - trends (direction/strength)
      - seasonality (daily/weekly)
      - cycles (arbitrary periods in hours)
      - anomalies (counts, max |z|)
    Returns a bundle (default) with features+LLM labels, or just LLM JSON if return_bundle=False.
    """
    if origins is None:
        origins = [
            "recent_window",
            "same_hour_previous_days",
            "same_weekday_recent_weeks",
            "prior_years_same_dates",
            "prior_years_same_week",
            "same_woy_prior_years",
            "same_month_prev_years",
        ]

    evidence_all: Dict[str, Any] = {}
    for origin in origins:
        df = retrieval_out.get(origin)
        if isinstance(df, pd.DataFrame) and not df.empty and {"SETTLEMENTDATE","REGION"}.issubset(df.columns):
            evidence_all[origin] = build_llm_evidence_for_origin(df, pcfg)

    llm_labels: Dict[str, Any] = {}
    for origin, ev in evidence_all.items():
        labeled = ask_llm_to_label_patterns(adapter, {"origin": origin, **ev}, pcfg, lcfg)
        llm_labels[origin] = labeled

    meta = retrieval_out.get("meta", {}) or {}
    bundle = {"meta": meta, "evidence": evidence_all, "llm_patterns": llm_labels}
    return bundle if return_bundle else json.dumps(llm_labels, ensure_ascii=False)
