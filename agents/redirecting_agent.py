from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List, Union
import pandas as pd
import json
from datetime import datetime

@dataclass
class HorizonConfig:
    # Boundaries (in days)
    short_days: int = 2        # <= short_days  => short_term
    long_days: int = 60        # >  long_days   => long_term (else mid_term)

    # Data column config (domain-agnostic)
    timestamp_col: str = "timestamp"           # column holding timestamps
    group_col: Optional[str] = None            # optional filter column
    group_values: Optional[List[str]] = None   # values to keep in group_col (case-insensitive)

    # Optional override: classify relative to this time instead of dataset latest
    # Accepts pd.Timestamp, datetime, or ISO-like string (e.g., "2025-09-17 12:00:00")
    reference_time: Optional[Union[pd.Timestamp, datetime, str]] = None

    # LLM runtime
    model: Optional[str] = None
    model_override: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 200

# -----------------------------------
# Utilities
# -----------------------------------
def _coerce_ts(x: Union[pd.Timestamp, datetime, str]) -> pd.Timestamp:
    if isinstance(x, pd.Timestamp):
        return x
    if isinstance(x, datetime):
        return pd.Timestamp(x)
    return pd.to_datetime(str(x), errors="raise")

def _latest_anchor(df: pd.DataFrame, cfg: HorizonConfig) -> pd.Timestamp:
    if cfg.timestamp_col not in df.columns:
        raise ValueError(f"Expected timestamp column '{cfg.timestamp_col}' in df.")

    work = df
    if cfg.group_col and cfg.group_col in df.columns and cfg.group_values:
        want = {str(v).lower() for v in cfg.group_values}
        work = df[df[cfg.group_col].astype(str).str.lower().isin(want)]

    if work.empty:
        raise ValueError("No rows available to determine latest anchor (after optional filtering).")

    ts = pd.to_datetime(work[cfg.timestamp_col], errors="coerce")
    last = ts.max()
    if pd.isna(last):
        raise ValueError(f"Could not determine latest observed timestamp in '{cfg.timestamp_col}'.")
    return last

def _horizon_messages(query: str, ref_ts: pd.Timestamp, cfg: HorizonConfig) -> List[Dict[str, str]]:
    system = (
        "You are a deterministic horizon classifier.\n"
        "Given a query and a reference timestamp, output EXACTLY one word from the allowed list: [short_term, mid_term, or long_term]"
        "Rules:\n"
        "No punctuation, no explanations, no JSON."
    )
    user = (
        "Use the provided reference timestamp as the anchor.\n"
        f"reference_time: {ref_ts.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"short_term: hours/days (<= {cfg.short_days} days)\n"
        f"mid_term:   weeks/months ({cfg.short_days+1}–{cfg.long_days} days)\n"
        f"long_term:  years (> {cfg.long_days} days)\n"
        f"query: {json.dumps(query, ensure_ascii=False)}\n\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

# -----------------------------------
# Public API
# -----------------------------------
def classify_horizon(
    adapter,
    query: str,
    df: pd.DataFrame,
    cfg: HorizonConfig = HorizonConfig(),
) -> str:
    """
    Classify a query's horizon relative to an anchor time : "short_term", "mid_term", "long_term".
    """
    if cfg.reference_time is not None:
        ref_ts = _coerce_ts(cfg.reference_time)
    else:
        ref_ts = _latest_anchor(df, cfg)

    messages = _horizon_messages(query, ref_ts, cfg)

    raw = adapter.chat(
        messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        model_override=(cfg.model_override or cfg.model),
    )

    out = str(raw).strip().lower()
    if out in {"short_term", "mid_term", "long_term"}:
        return out
    return "mid_term"  # safe fallback