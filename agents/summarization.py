from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Iterable
import pandas as pd
import numpy as np
import json
from datetime import timedelta

# -----------------------------------------------------------
# Config
# -----------------------------------------------------------
@dataclass
class SummarizeConfig:
    model: str = None             # default; adapter can override per call
    temperature: float = 0.2
    max_tokens_out: int = None               # small, deterministic JSON
    target_tokens_per_chunk: int = 500      # rough input budget per chunk
    hard_max_rows_per_chunk: int = 1200     # guardrail
    time_chunk_hours: Optional[int] = 24  # if set, chunk by time windows
    row_chunk_size_fallback: int = 600     # if not time chunking
    json_mode: bool = True
    tz: str = "Australia/Sydney"
    model_override: Optional[str] = None    # force model for this stage if needed

@dataclass
class AnomalyConfig:
    z_threshold: float = 3.0          # abs(z) threshold
    rolling_window: str = "7D"        # rolling mean/std horizon
    min_points: int = 24              # min points for rolling stats

# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------
def _coerce_ts_local(series: pd.Series, tz: str) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", utc=False)
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    return s

def _df_to_compact_csv(df: pd.DataFrame, max_rows: int = 1000) -> str:
    if df is None or df.empty:
        return ""
    cols_pref = ["REGION", "SETTLEMENTDATE", "TOTALDEMAND", "RRP", "PERIODTYPE", "PRIOR_YEAR", "LAG_DAYS", "PRIOR_WOY"]
    cols = [c for c in cols_pref if c in df.columns] or list(df.columns[:8])
    slim = df[cols].head(max_rows).copy()
    if "SETTLEMENTDATE" in slim.columns:
        slim["SETTLEMENTDATE"] = pd.to_datetime(slim["SETTLEMENTDATE"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    return slim.to_csv(index=False)

def _dict_compact(d: Dict[str, Any], max_chars: int = 6000) -> str:
    s = json.dumps(d, ensure_ascii=False, default=str)
    if len(s) <= max_chars:
        return s
    try:
        dd = json.loads(s)
        for k, v in list(dd.items()):
            if isinstance(v, list) and len(v) > 12:
                dd[k] = v[:6] + ["…"] + v[-3:]
        s = json.dumps(dd, ensure_ascii=False, default=str)
    except Exception:
        s = s[:max_chars] + "…"
    return s

# -----------------------------------------------------------
# Precompute numerics (cheap, offline)
# -----------------------------------------------------------
def _basic_stats(df: pd.DataFrame, metrics: List[str]) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    out = []
    for reg, sub in df.groupby(df["REGION"].astype(str).str.upper()):
        for m in metrics:
            if m not in sub.columns:
                continue
            v = pd.to_numeric(sub[m], errors="coerce").dropna()
            if v.empty:
                continue
            q = v.quantile([0.1, 0.5, 0.9]).to_dict()
            out.append({
                "region": reg, "metric": m,
                "count": int(v.size),
                "mean": float(v.mean()),
                "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                "min": float(v.min()),
                "p10": float(q.get(0.1, np.nan)),
                "p50": float(q.get(0.5, np.nan)),
                "p90": float(q.get(0.9, np.nan)),
                "max": float(v.max()),
            })
    return out

def _rolling_anomalies(
    df: pd.DataFrame,
    metrics: List[str],
    acfg: AnomalyConfig,
    tz: str
) -> List[Dict[str, Any]]:
    if df is None or df.empty or "REGION" not in df.columns or "SETTLEMENTDATE" not in df.columns:
        return []
    work = df.copy()
    ts = _coerce_ts_local(work["SETTLEMENTDATE"], tz)
    work["__ts"] = ts
    work = work.dropna(subset=["__ts"]).sort_values("__ts")
    out: List[Dict[str, Any]] = []

    for reg, sub in work.groupby(work["REGION"].astype(str).str.upper()):
        sub = sub.sort_values("__ts")
        for m in metrics:
            if m not in sub.columns:
                continue
            x = pd.to_numeric(sub[m], errors="coerce")
            s = pd.Series(x.values, index=sub["__ts"].values)  # DatetimeIndex
            mu = s.rolling(acfg.rolling_window, min_periods=acfg.min_points).mean()
            sd = s.rolling(acfg.rolling_window, min_periods=acfg.min_points).std(ddof=1).replace(0, np.nan)
            z = (s - mu) / sd
            mask = (z.abs() >= acfg.z_threshold).fillna(False)
            if mask.any():
                z_series = z[mask]
                for ts_i, z_val in z_series.items():
                    out.append({
                        "region": reg,
                        "metric": m,
                        "ts": pd.to_datetime(ts_i).strftime("%Y-%m-%d %H:%M:%S"),
                        "z": float(z_val),
                        "type": "spike" if z_val > 0 else "drop",
                        "detector": "rolling"
                    })
    return out

# -----------------------------------------------------------
# Chunking strategies
# -----------------------------------------------------------
def _chunk_by_time(df: pd.DataFrame, hours: int, tz: str) -> Iterable[pd.DataFrame]:
    if df is None or df.empty:
        return []
    s = _coerce_ts_local(df["SETTLEMENTDATE"], tz)
    df = df.copy()
    df["__ts"] = s
    df = df.dropna(subset=["__ts"]).sort_values("__ts")
    start, end = df["__ts"].min(), df["__ts"].max()
    cur = start
    while cur <= end:
        nxt = cur + timedelta(hours=hours)
        yield df[(df["__ts"] >= cur) & (df["__ts"] < nxt)].drop(columns="__ts")
        cur = nxt

def _chunk_by_rows(df: pd.DataFrame, size: int) -> Iterable[pd.DataFrame]:
    if df is None or df.empty:
        return []
    n = len(df)
    for i in range(0, n, size):
        yield df.iloc[i:i + size]

def _auto_chunk_df(df: pd.DataFrame, cfg: SummarizeConfig) -> List[pd.DataFrame]:
    if df is None or df.empty:
        return []
    if cfg.time_chunk_hours:
        chunks = [c for c in _chunk_by_time(df, cfg.time_chunk_hours, cfg.tz) if not c.empty]
    else:
        # ~25 tokens/row crude; times 4 chars/token ~100 chars/row
        approx_rows = max(200, int((cfg.target_tokens_per_chunk * 4) / 100))
        size = min(cfg.hard_max_rows_per_chunk, approx_rows if approx_rows > 0 else cfg.row_chunk_size_fallback)
        chunks = [c for c in _chunk_by_rows(df, size) if not c.empty]
    return chunks

# -----------------------------------------------------------
# LLM JSON helpers (Adapter + Loose JSON)
# -----------------------------------------------------------
def _llm_json_adapter(
    adapter,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: Optional[int],
    json_mode: bool,
    model_override: Optional[str],
) -> Dict[str, Any]:
    """
    Calls adapter.chat_json_loose → tolerant to malformed JSON.
    """
    messages = [{"role":"system","content":system},{"role":"user","content":user}]
    out = adapter.chat_json_loose(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        model_override=(model_override or model),
        strict_json_first=bool(json_mode),
    )
    return out if isinstance(out, dict) else {"summary": str(out)[: (max_tokens or 512) ]}

# -----------------------------------------------------------
# Prompts (deterministic, retrieval-only)
# -----------------------------------------------------------
def _prompt_map(origin: str, chunk_csv: str, pre_stats: List[Dict[str, Any]], pre_anoms: List[Dict[str, Any]]) -> Tuple[str, str]:
    system = "You compress electricity time-series slices into JSON. No prose. Deterministic."
    user = f"""
ORIGIN: {origin}

# PRECOMPUTED_STATS (use instead of recalculating)
{_dict_compact({"stats": pre_stats}, 3000)}

# PRECOMPUTED_ANOMALIES (subset; may be empty)
{_dict_compact({"anomalies": pre_anoms[:100]}, 3000)}

# CSV SLICE (truncated rows)
{chunk_csv}

Return ONLY JSON with schema:
{{
  "slice_stats": [{{"region":"...", "metric":"...", "mean":float, "std":float, "min":float, "max":float}}],
  "events": [{{"ts":"YYYY-MM-DD HH:MM:SS","region":"...","metric":"...","type":"spike|drop|level_shift","note":"short"}}]
}}
"""
    return system, user

def _prompt_reduce(origin: str, partials: List[Dict[str, Any]], pre_stats: List[Dict[str, Any]], pre_anoms: List[Dict[str, Any]]) -> Tuple[str, str]:
    system = "You merge partial JSON summaries for one origin into a compact JSON. No prose."
    user = f"""
You are merging MAP outputs for ORIGIN={origin}.
Use the PRECOMPUTED stats/anomalies as ground truth.

PRECOMPUTED:
{_dict_compact({"stats": pre_stats, "anomalies": pre_anoms[:200]}, 8000)}

PARTIALS:
{json.dumps(partials, ensure_ascii=False)[:9000]}

Return ONLY JSON with schema:
{{
  "origin": "{origin}",
  "global_stats": [{{"region":"...","metric":"...","mean":float,"std":float,"min":float,"max":float}}],
  "global_events": [{{"ts":"...","region":"...","metric":"...","type":"spike|drop|level_shift","note":"short"}}],
  "bullets": ["insight 1","insight 2","insight 3"]
}}
"""
    return system, user

def _prompt_cross(per_origin_reduced: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
    system = "You synthesize per-origin summaries into a single JSON. No prose."
    user = f"""
You receive per-origin JSONs keyed by origin names (e.g., recent_window, prior_years_same_dates, prior_years_same_week,
same_hour_previous_days, same_woy_prior_years). Items are already reduced.

DATA:
{json.dumps(per_origin_reduced, ensure_ascii=False)[:12000]}

Return ONLY JSON with schema:
{{
  "headlines": ["short bullet","short bullet","short bullet"],
  "comparative": [{{"origin":"...", "metric":"RRP|TOTALDEMAND","pattern":"recent vs prior-years","note":"short"}}],
  "risks": ["YYYY-MM-DD HH:MM:SS NSW1 RRP spike (z=3.1)", "..."]
}}
"""
    return system, user

# -----------------------------------------------------------
# Text builder (compact narrative for forecaster conditioning)
# -----------------------------------------------------------
def make_pattern_summary(result_bundle: dict) -> str:
    """
    Build a compact human-readable summary from the summarize_from_retrieval_strategy() bundle.
    Robust to missing origins or unexpected shapes.
    """
    if not isinstance(result_bundle, dict):
        return "No summaries available."

    cross = result_bundle.get("cross_origin", {}) or {}
    headlines = cross.get("headlines") or []
    if not isinstance(headlines, list):
        headlines = []
    general = "; ".join([str(h) for h in headlines[:5]]) if headlines else None

    po = result_bundle.get("per_origin", {}) or {}

    def _bullets_for(origin_key: str) -> Optional[str]:
        blk = po.get(origin_key, {}) or {}
        if not isinstance(blk, dict):
            return None
        reduced = blk.get("reduced", {}) or {}
        if not isinstance(reduced, dict):
            return None
        bullets = reduced.get("bullets") or []
        if not isinstance(bullets, list):
            return None
        return "; ".join([str(b) for b in bullets[:5]]) or None

    parts = []
    if general:
        parts.append(f"Overall: {general}.")

    # Show the most common origins if present
    for origin in ["recent_window",
                   "same_hour_previous_days",
                   "prior_years_same_week",
                   "prior_years_same_dates",
                   "same_woy_prior_years"]:
        bt = _bullets_for(origin)
        if bt:
            label = origin.replace("_", " ")
            parts.append(f"{label}: {bt}.")

    return " ".join(parts) if parts else "No summaries available."

# -----------------------------------------------------------
# Main entry (retrieval-only) — Adapter
# -----------------------------------------------------------
def summarize_from_retrieval_strategy(
    adapter,
    retrieval_out: Dict[str, Any],
    *,
    cfg: SummarizeConfig = SummarizeConfig(),
    acfg: AnomalyConfig = AnomalyConfig(),
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Token-lean summarization using ONLY retrieval output:
      - Precompute stats + rolling anomalies per origin
      - Chunk each origin slice
      - Map -> Reduce per origin
      - Cross-origin synthesis

    Adapter lets you switch between OpenAI / DeepSeek / Gemini without changes here.
    """
    # ---- NEW: include two extra origins
    origins_all = [
        "recent_window",
        "prior_years_same_dates",
        "prior_years_same_week",
        "same_hour_previous_days",    # NEW
        "same_woy_prior_years",       # NEW
    ]

    present = {
        k: v for k, v in retrieval_out.items()
        if k in origins_all and isinstance(v, pd.DataFrame) and not v.empty
    }

    # discover metrics present
    if metrics is None:
        m_pref = ["TOTALDEMAND", "RRP"]
        cols = set().union(*[set(df.columns) for df in present.values()]) if present else set()
        metrics = [m for m in m_pref if m in cols]

    per_origin_results: Dict[str, Dict[str, Any]] = {}
    debug = {"origin_rows": {k: int(len(v)) for k, v in present.items()}}

    for origin, df in present.items():
        # PRECOMPUTE numerics (per origin slice)
        stats = _basic_stats(df, metrics)
        anomalies = _rolling_anomalies(df, metrics, acfg, cfg.tz)

        # CHUNK
        chunks = _auto_chunk_df(df, cfg)

        # MAP
        partials: List[Dict[str, Any]] = []
        for ch in chunks:
            csv_text = _df_to_compact_csv(ch, max_rows=cfg.row_chunk_size_fallback)
            sys_m, usr_m = _prompt_map(origin, csv_text, stats, anomalies)
            out_m = _llm_json_adapter(
                adapter, cfg.model, sys_m, usr_m,
                cfg.temperature, cfg.max_tokens_out, cfg.json_mode,
                cfg.model_override
            )
            partials.append(out_m)

        # REDUCE (per-origin)
        sys_r, usr_r = _prompt_reduce(origin, partials, stats, anomalies)
        reduced = _llm_json_adapter(
            adapter, cfg.model, sys_r, usr_r,
            cfg.temperature, cfg.max_tokens_out, cfg.json_mode,
            cfg.model_override
        )

        per_origin_results[origin] = {
            "precomputed": {"stats": stats, "anomalies": anomalies},
            "partials": partials,
            "reduced": reduced,
            "n_chunks": len(chunks),
        }

    # CROSS-ORIGIN synthesis (only on reduced views)
    sys_x, usr_x = _prompt_cross({k: v["reduced"] for k, v in per_origin_results.items()})
    cross = _llm_json_adapter(
        adapter, cfg.model, sys_x, usr_x,
        cfg.temperature, cfg.max_tokens_out, cfg.json_mode,
        cfg.model_override
    )

    summaries_bundle = {
        "per_origin": per_origin_results,
        "cross_origin": cross,
        "debug": debug,
        "config": {
            "summarize": vars(cfg),
            "anomaly": vars(acfg),
            "metrics": metrics,
        },
    }

    summary_text = make_pattern_summary(summaries_bundle)
    return {"bundle": summaries_bundle, "text": summary_text}