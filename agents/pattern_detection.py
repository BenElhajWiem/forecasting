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

    # Spikes/Drops (rolling z)
    z_threshold: float = 3.0
    rolling_window: str = "7d"
    min_points: int = 24

    # Ramps (delta vs sigma over horizon)
    ramp_horizon: str = "1h"
    ramp_abs_threshold: float = 0.0
    ramp_sigma_threshold: float = 2.0

    # Level shifts (left/right windows)
    level_window: str = "6h"
    level_sigma_threshold: float = 2.5

    # Vol regimes (fast/slow std ratio)
    vol_fast: str = "6h"
    vol_slow: str = "3d"
    vol_ratio_threshold: float = 2.0

    # Trend & periodicity
    trend_resample: str = "1h"

    # Payload hygiene
    max_events_per_metric: int = 40
    context_points_per_event: int = 6

@dataclass
class LLMConfig:
    model: str = None
    temperature: float = 0.2
    max_tokens_out: int = None
    json_mode: bool = False          # If True, asks for JSON; else plain text
    model_override: Optional[str] = None

# =======================
# 2) Low-level utilities
# =======================
def _freq(s: str | None) -> str | None:
    return str(s).lower() if s is not None else None

def _to_local(series: pd.Series, tz: str) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", utc=False)
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    return s

def _series_indexed(df: pd.DataFrame, metric: str, tz: str) -> pd.Series:
    ts = _to_local(df["SETTLEMENTDATE"], tz)
    x = pd.to_numeric(df[metric], errors="coerce")
    return pd.Series(x.values, index=ts.values).sort_index()

def _quantiles(v: pd.Series) -> Dict[str, float]:
    v = pd.to_numeric(v, errors="coerce").dropna()
    if v.empty:
        return {"mean": np.nan, "std": 0.0, "min": np.nan, "p10": np.nan, "p25": np.nan,
                "p50": np.nan, "p75": np.nan, "p90": np.nan, "max": np.nan, "count": 0}
    qs = v.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
    return {"mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()),
            "p10": float(qs.get(0.1, np.nan)),
            "p25": float(qs.get(0.25, np.nan)),
            "p50": float(qs.get(0.5, np.nan)),
            "p75": float(qs.get(0.75, np.nan)),
            "p90": float(qs.get(0.9, np.nan)),
            "max": float(v.max()),
            "count": int(v.size)}

# =======================
# 3) Numeric Candidate Detectors (per metric)
# =======================
def detect_spike_drop(s: pd.Series, cfg: PatternConfig) -> List[Dict[str, Any]]:
    mu = s.rolling(_freq(cfg.rolling_window), min_periods=cfg.min_points).mean()
    sd = s.rolling(_freq(cfg.rolling_window), min_periods=cfg.min_points).std(ddof=1).replace(0, np.nan)
    z = (s - mu) / sd
    mask = (z.abs() >= float(cfg.z_threshold)).fillna(False)
    out = []
    for t in s.index[mask]:
        out.append({
            "ts": pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S"),
            "z": float(z.loc[t]) if pd.notna(z.loc[t]) else None,
            "value": float(s.loc[t]) if pd.notna(s.loc[t]) else None,
            "kind_hint": "spike" if (pd.notna(z.loc[t]) and z.loc[t] > 0) else "drop"
        })
    return out

def detect_ramps(s: pd.Series, cfg: PatternConfig) -> List[Dict[str, Any]]:
    prev = s.shift(freq=_freq(cfg.ramp_horizon))
    delta = s - prev
    horizon_hours = pd.Timedelta(_freq(cfg.ramp_horizon)).total_seconds() / 3600.0
    rate = delta / horizon_hours
    sd_r = s.rolling(_freq(cfg.rolling_window), min_periods=cfg.min_points).std(ddof=1)
    thr_sigma = sd_r.mul(float(cfg.ramp_sigma_threshold))
    thr = thr_sigma.clip(lower=float(cfg.ramp_abs_threshold)).reindex(delta.index)
    mask = (delta.abs() >= thr).fillna(False)
    out = []
    for t in delta.index[mask]:
        out.append({
            "ts": pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S"),
            "delta": float(delta.loc[t]) if pd.notna(delta.loc[t]) else None,
            "prev": float(prev.loc[t]) if pd.notna(prev.loc[t]) else None,
            "value": float(s.loc[t]) if pd.notna(s.loc[t]) else None,
            "rate_per_h": float(rate.loc[t]) if pd.notna(rate.loc[t]) else None
        })
    return out

def detect_level_shifts(s: pd.Series, cfg: PatternConfig) -> List[Dict[str, Any]]:
    L = _freq(cfg.level_window)
    mean_left  = s.rolling(L, min_periods=cfg.min_points).mean()
    mean_right = mean_left.shift(freq=pd.Timedelta(L))
    std_left   = s.rolling(L, min_periods=cfg.min_points).std(ddof=1)
    std_right  = std_left.shift(freq=pd.Timedelta(L))
    pooled = pd.concat([std_left, std_right], axis=1).max(axis=1).replace(0, np.nan)
    diff = (mean_right - mean_left)
    pooled = pooled.reindex(diff.index)
    rhs = pooled.mul(float(cfg.level_sigma_threshold))
    mask = (diff.abs() >= rhs) & diff.notna() & rhs.notna()
    ml = mean_left.reindex(diff.index); mr = mean_right.reindex(diff.index)
    out = []
    for t in diff.index[mask.fillna(False)]:
        out.append({
            "ts": pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S"),
            "delta_mean": float(diff.loc[t]),
            "mean_left": (float(ml.loc[t]) if pd.notna(ml.loc[t]) else None),
            "mean_right": (float(mr.loc[t]) if pd.notna(mr.loc[t]) else None)
        })
    return out

def detect_volatility_regimes(s: pd.Series, cfg: PatternConfig) -> List[Dict[str, Any]]:
    fast = s.rolling(_freq(cfg.vol_fast), min_periods=cfg.min_points).std(ddof=1)
    slow = s.rolling(_freq(cfg.vol_slow), min_periods=cfg.min_points).std(ddof=1).reindex(fast.index)
    ratio = (fast / slow).replace([np.inf, -np.inf], np.nan)
    hi = (ratio >= float(cfg.vol_ratio_threshold)).fillna(False)
    lo = (ratio <= (1.0 / max(float(cfg.vol_ratio_threshold), 1e-9))).fillna(False)
    out = []
    for t in ratio.index[hi]:
        out.append({"ts": pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S"),
                    "ratio": float(ratio.loc[t]), "direction": "up"})
    for t in ratio.index[lo]:
        out.append({"ts": pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S"),
                    "ratio": float(ratio.loc[t]), "direction": "down"})
    return out

def detect_trend_and_periodicity(s: pd.Series, cfg: PatternConfig) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    # Trend on resampled mean
    hourly = s.resample(_freq(cfg.trend_resample) or "1h").mean().dropna()
    trend = None
    if len(hourly) >= 5:
        t0 = hourly.index[0]
        t_hours = (hourly.index - t0) / np.timedelta64(1, "h")
        X = np.vstack([t_hours, np.ones_like(t_hours)]).T
        y = hourly.values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        trend = {"slope_per_hour": float(beta[0])}

    # Periodicity via ACF taps
    periodicity = None
    def acf(x: np.ndarray, k: int) -> float:
        x = x - x.mean()
        denom = float((x ** 2).sum())
        if denom == 0 or k >= len(x): return 0.0
        return float((x[:-k] * x[k:]).sum() / denom)
    if len(hourly) > 30:
        arr = hourly.values
        daily = acf(arr, 24)
        weekly = acf(arr, 24 * 7) if len(arr) > 24 * 7 else 0.0
        periodicity = {"daily_acf": float(daily), "weekly_acf": float(weekly)}
    return trend, periodicity

# =======================
# 4) Per-metric evidence builder (uses detectors)
# =======================
def build_numeric_candidates_for_metric(df: pd.DataFrame, metric: str, cfg: PatternConfig) -> Dict[str, Any]:
    s = _series_indexed(df, metric, cfg.tz)
    out = {
        "stats": _quantiles(df[metric]),
        "spike_drop": [], "ramp": [], "level_shift": [], "volatility": [], "trend": [], "periodicity": []
    }
    if s.empty: return out
    out["spike_drop"] = detect_spike_drop(s, cfg)
    out["ramp"] = detect_ramps(s, cfg)
    out["level_shift"] = detect_level_shifts(s, cfg)
    out["volatility"] = detect_volatility_regimes(s, cfg)
    trend, periodicity = detect_trend_and_periodicity(s, cfg)
    if trend: out["trend"].append(trend)
    if periodicity: out["periodicity"].append(periodicity)

    # Cap event lists
    for k in ("spike_drop", "ramp", "level_shift", "volatility"):
        if len(out[k]) > cfg.max_events_per_metric:
            out[k] = out[k][:cfg.max_events_per_metric]
    return out

def build_numeric_evidence_for_origin(df: pd.DataFrame, cfg: PatternConfig) -> Dict[str, Any]:
    ev: Dict[str, Any] = {"summary": {}, "by_region_metric": {}}
    ts = _to_local(df["SETTLEMENTDATE"], cfg.tz)
    ev["summary"] = {
        "n_rows": int(len(df)),
        "start": ts.min().strftime("%Y-%m-%d %H:%M:%S") if len(ts) else None,
        "end":   ts.max().strftime("%Y-%m-%d %H:%M:%S") if len(ts) else None,
        "regions": sorted(df["REGION"].astype(str).str.upper().unique()) if "REGION" in df.columns else []
    }
    for reg, sub in df.groupby(df["REGION"].astype(str).str.upper()):
        ev["by_region_metric"].setdefault(reg, {})
        for m in cfg.metrics:
            if m in sub.columns:
                ev["by_region_metric"][reg][m] = build_numeric_candidates_for_metric(sub, m, cfg)
    return ev

# =======================
# 5) Compact numeric references (period-anchored)
# =======================
def _init_metric_ref() -> Dict[str, Any]:
    return {
        "spike":       {"count":0, "z_max":None, "z_mean":None, "support_count":0, "window_mean":None, "window_std":None},
        "ramp":        {"count":0, "delta_abs_max":None, "delta_sign":0, "ramp_rate_per_h":None},
        "level_shift": {"count":0, "level_shift_delta_mean":None, "mean_left":None, "mean_right":None},
        "volatility":  {"count":0, "vol_ratio_max":None},
        "trend":       {"slope_per_h":None},
        "periodicity": {"daily_acf":None, "weekly_acf":None},
    }

def _apply_aggregate(ref: Dict[str, Any], kind: str, feats: Dict[str, Any]):
    if kind in ("spike","drop"):
        ref["spike"]["count"] += 1
        for k in ("z_max","z_mean","window_mean","window_std"):
            v = feats.get(k)
            if v is not None:
                if k == "z_max":
                    cur = ref["spike"].get(k)
                    if cur is None or v > cur: ref["spike"][k] = float(v)
                elif k == "z_mean":
                    ref["spike"][k] = float(v)
                else:
                    if ref["spike"].get(k) is None:
                        ref["spike"][k] = float(v)
        ref["spike"]["support_count"] += int(feats.get("support_count") or 0)

    elif kind == "ramp":
        ref["ramp"]["count"] += 1
        for k in ("delta_abs_max","ramp_rate_per_h"):
            v = feats.get(k)
            if v is not None:
                cur = ref["ramp"].get(k)
                if cur is None or v > cur: ref["ramp"][k] = float(v)
        ds = feats.get("delta_sign")
        if ds not in (None, 0):
            ref["ramp"]["delta_sign"] = int(np.sign(ds))

    elif kind == "level_shift":
        ref["level_shift"]["count"] += 1
        for k in ("level_shift_delta_mean","mean_left","mean_right"):
            v = feats.get(k)
            if v is not None and ref["level_shift"].get(k) is None:
                ref["level_shift"][k] = float(v)

    elif kind.startswith("volatility"):
        ref["volatility"]["count"] += 1
        v = feats.get("vol_ratio_max")
        if v is not None:
            cur = ref["volatility"].get("vol_ratio_max")
            if cur is None or v > cur: ref["volatility"]["vol_ratio_max"] = float(v)

    elif kind == "trend":
        v = feats.get("trend_slope_per_h")
        cur = ref["trend"].get("slope_per_h")
        if v is not None and (cur is None or abs(v) > abs(cur)):
            ref["trend"]["slope_per_h"] = float(v)

    elif kind == "periodicity":
        for k in ("daily_acf","weekly_acf"):
            v = feats.get(k)
            cur = ref["periodicity"].get(k)
            if v is not None and (cur is None or abs(v) > abs(cur)):
                ref["periodicity"][k] = float(v)

def build_numeric_references(evidence_by_origin: Dict[str, Any]) -> Dict[str, Any]:
    refs: Dict[str, Any] = {}
    for origin, ev in evidence_by_origin.items():
        period = {k: ev.get("summary", {}).get(k) for k in ("start","end","n_rows")}
        block = {"period": period, "by_region_metric": {}}

        for reg, metdict in (ev.get("by_region_metric") or {}).items():
            block["by_region_metric"].setdefault(reg, {})
            for metric, buckets in metdict.items():
                ref = _init_metric_ref()

                # spikes/drops
                s_list = buckets.get("spike_drop") or []
                stats = buckets.get("stats") or {}
                if s_list:
                    z_vals = [e.get("z") for e in s_list if e.get("z") is not None]
                    if z_vals:
                        _apply_aggregate(ref, "spike",
                            {"z_max": max(z_vals),
                             "z_mean": float(np.mean(z_vals)),
                             "support_count": len(s_list),
                             "window_mean": stats.get("mean"),
                             "window_std": stats.get("std")})

                # ramps
                r_list = buckets.get("ramp") or []
                if r_list:
                    delta_abs_max = max([abs(e.get("delta")) for e in r_list if e.get("delta") is not None] or [None])
                    ramp_rate_per_h = max([e.get("rate_per_h") for e in r_list if e.get("rate_per_h") is not None] or [None])
                    signs = [np.sign(e.get("delta")) for e in r_list if e.get("delta") is not None]
                    delta_sign = int(np.sign(np.median(signs))) if signs else 0
                    _apply_aggregate(ref, "ramp",
                        {"delta_abs_max": delta_abs_max,
                         "ramp_rate_per_h": ramp_rate_per_h,
                         "delta_sign": delta_sign})

                # level shifts
                l_list = buckets.get("level_shift") or []
                if l_list:
                    dmeans = [e.get("delta_mean") for e in l_list if e.get("delta_mean") is not None]
                    mlefts = [e.get("mean_left") for e in l_list if e.get("mean_left") is not None]
                    mrights= [e.get("mean_right") for e in l_list if e.get("mean_right") is not None]
                    _apply_aggregate(ref, "level_shift",
                        {"level_shift_delta_mean": float(np.median(dmeans)) if dmeans else None,
                         "mean_left": float(np.median(mlefts)) if mlefts else None,
                         "mean_right": float(np.median(mrights)) if mrights else None})

                # volatility
                v_list = buckets.get("volatility") or []
                if v_list:
                    vr = max([e.get("ratio") for e in v_list if e.get("ratio") is not None] or [None])
                    _apply_aggregate(ref, "volatility_up", {"vol_ratio_max": vr})

                # trend
                tr_list = buckets.get("trend") or []
                if tr_list:
                    _apply_aggregate(ref, "trend", {"trend_slope_per_h": tr_list[0].get("slope_per_hour")})

                # periodicity
                p_list = buckets.get("periodicity") or []
                if p_list:
                    _apply_aggregate(ref, "periodicity",
                        {"daily_acf": p_list[0].get("daily_acf"),
                         "weekly_acf": p_list[0].get("weekly_acf")})

                block["by_region_metric"][reg][metric] = ref
        refs[origin] = block
    return refs

# =======================
# 6) LLM narrator (interpret numbers only) — Adapter
# =======================
def _round(v, nd=2):
    if v is None: return None
    try:
        if isinstance(v, (int, float)):
            if math.isnan(v) or math.isinf(v): return None
            return round(float(v), nd)
        return None
    except Exception:
        return None

def _compact_metric_block(block: Dict[str, Any]) -> Dict[str, Any]:
    sp = block.get("spike", {}); rp = block.get("ramp", {}); ls = block.get("level_shift", {})
    vo = block.get("volatility", {}); tr = block.get("trend", {}); pr = block.get("periodicity", {})
    return {
        "spike": {"count": int(sp.get("count",0)), "z_max": _round(sp.get("z_max")),
                  "z_mean": _round(sp.get("z_mean")), "window_mean": _round(sp.get("window_mean")),
                  "window_std": _round(sp.get("window_std"))},
        "ramp": {"count": int(rp.get("count",0)), "delta_abs_max": _round(rp.get("delta_abs_max")),
                 "delta_sign": int(rp.get("delta_sign",0)), "ramp_rate_per_h": _round(rp.get("ramp_rate_per_h"))},
        "level_shift": {"count": int(ls.get("count",0)), "level_shift_delta_mean": _round(ls.get("level_shift_delta_mean")),
                        "mean_left": _round(ls.get("mean_left")), "mean_right": _round(ls.get("mean_right"))},
        "volatility": {"count": int(vo.get("count",0)), "vol_ratio_max": _round(vo.get("vol_ratio_max"))},
        "trend": {"slope_per_h": _round(tr.get("slope_per_h"))},
        "periodicity": {"daily_acf": _round(pr.get("daily_acf")), "weekly_acf": _round(pr.get("weekly_acf"))},
    }

def _shrink_reference(reference: Dict[str, Any], max_origins: int = 5, max_regions: int = 8, max_metrics_per_region: int = 2) -> Dict[str, Any]:
    small: Dict[str, Any] = {}
    for origin in list(reference.keys())[:max_origins]:
        obj = reference[origin] or {}
        period = obj.get("period") or {}
        by_rm  = obj.get("by_region_metric") or {}
        regs = sorted(by_rm.keys())[:max_regions]
        items = []
        for reg in regs:
            met_dict = by_rm.get(reg, {})
            for metric in list(sorted(met_dict.keys()))[:max_metrics_per_region]:
                items.append({"region": reg, "metric": metric, "numbers": _compact_metric_block(met_dict[metric])})
        small[origin] = {"period": period, "items": items}
    return small

def narrate_patterns_with_llm(
    adapter,
    references: Dict[str, Any],
    pcfg: PatternConfig,
    lcfg: LLMConfig,
    *,
    max_origins: int = 5,
    max_regions: int = 8,
    max_metrics_per_region: int = 2,
) -> str:
    evidence = _shrink_reference(references, max_origins=max_origins, max_regions=max_regions, max_metrics_per_region=max_metrics_per_region)

    system = (
        "You are a precise energy time-series expert. "
        "Interpret the provided numbers. "
        "Do NOT invent values; only use numbers you see."
    )
    user = f"""
Write concise sentences for EACH origin in the evidence, in order.
For each origin, do:
1) Start with the period: [start → end; n_rows].
2) For each salient region/metric (non-zero counts or non-null stats), report:
   • Trend: use slope_per_h (resample={pcfg.trend_resample}).
     - Classify as ↑ or ↓ if slope_per_h is nonzero; strength: weak/moderate/strong by magnitude.
   • Seasonality/Cycles: use periodicity.daily_acf and periodicity.weekly_acf.
     - Daily seasonality present if daily_acf ≥ 0.3 (strong if ≥ 0.5).
     - Weekly cycle present if weekly_acf ≥ 0.3 (strong if ≥ 0.5).
   • Randomness/Noise: infer "high/medium/low" from low ACFs (<0.3), near-zero slope, low spike/ramp counts, and vol_ratio_max ≈ 1.
     - If vol_ratio_max ≥ {pcfg.vol_ratio_threshold}, note elevated volatility.
   • Regime/Level changes: if level_shift.count > 0, note that baseline shifted (use level_shift_delta_mean).

Cite methods inline:
  spikes/drops: rolling-z over {pcfg.rolling_window}
  ramps: over {pcfg.ramp_horizon}
  level shifts: windows {pcfg.level_window} each
  volatility: σ_fast/σ_slow (fast={pcfg.vol_fast}, slow={pcfg.vol_slow})
  trend: resample={pcfg.trend_resample}

Heuristics (strict):
- Trend strength (|slope_per_h|): <0.01 → none; 0.01–0.05 → weak; 0.05–0.15 → moderate; >0.15 → strong.
- Seasonality present if ACF ≥ 0.3 (strong if ≥ 0.5).
- Randomness high if daily_acf<0.3 AND weekly_acf<0.3 AND |slope_per_h|<0.01 AND spike/ramp counts are low AND vol_ratio_max<1.3.
- If nothing salient: 'no salient patterns detected for <region>/<metric>'.

Return {"JSON" if lcfg.json_mode else "plain text"} only.

EVIDENCE (pure numbers + calc windows):
{json.dumps(evidence, ensure_ascii=False)[:15000]}
"""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if lcfg.json_mode:
        out = adapter.chat_json_loose(
            messages,
            temperature=lcfg.temperature,
            max_tokens=lcfg.max_tokens_out,
            model_override=(lcfg.model_override or lcfg.model),
            strict_json_first=True,
        )
        # If JSON, convert to small text paragraph anyway (compact)
        return json.dumps(out, ensure_ascii=False)
    else:
        text = adapter.chat(
            messages,
            temperature=lcfg.temperature,
            max_tokens=lcfg.max_tokens_out,
            model_override=(lcfg.model_override or lcfg.model),
        )
        return (text or "").strip()

# =======================
# 7) End-to-end runner (AFTER retrieval agent)
# =======================
from typing import Any

def run_pattern_detection_after_retrieval(
    adapter,                                   # <-- renamed: this is your LLMClientAdapter
    retrieval_out: Dict[str, Any],
    *,
    pcfg: PatternConfig = PatternConfig(),
    lcfg: LLMConfig = LLMConfig(),
    origins: Optional[List[str]] = None,       # optional override of which slices to narrate
    return_bundle: bool = False,               # <-- NEW: get full struct if True
) -> Any:
    """
    Build numeric evidence per origin, compact into references, and ask the LLM to narrate.

    Expects retrieval_out to contain DataFrames for any of:
      - 'recent_window'
      - 'prior_years_same_dates'
      - 'prior_years_same_week'
      - 'same_hour_previous_days'
      - 'same_woy_prior_years'
      plus 'meta' (dict)

    Returns:
      - paragraph string when return_bundle=False (default)
      - full bundle dict when return_bundle=True
    """
    # default origins (includes your new ones)
    if origins is None:
        origins = [
            "recent_window",
            "prior_years_same_dates",
            "prior_years_same_week",
            "same_hour_previous_days",
            "same_woy_prior_years",
        ]

    # --- Build evidence per origin
    evidence_by_origin: Dict[str, Any] = {}
    for origin in origins:
        df = retrieval_out.get(origin)
        if isinstance(df, pd.DataFrame) and not df.empty and {"SETTLEMENTDATE", "REGION"}.issubset(df.columns):
            evidence_by_origin[origin] = build_numeric_evidence_for_origin(df, pcfg)

    # --- Compact references
    references = build_numeric_references(evidence_by_origin)

    # --- LLM narration (adapter version)
    paragraph = narrate_patterns_with_llm(adapter, references, pcfg, lcfg)

    bundle = {
        "evidence": evidence_by_origin,   # full per-event numeric candidates
        "references": references,         # compact, period-anchored numeric refs
        "paragraph": (paragraph or "").strip(),
        "meta": retrieval_out.get("meta", {}),
    }
    return bundle if return_bundle else bundle["paragraph"]