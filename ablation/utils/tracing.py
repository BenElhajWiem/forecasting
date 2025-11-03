# utils/tracing.py
from __future__ import annotations
import time, numbers
from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np

# Keep a canonical order of retrieval blocks for previews & counts
BLOCK_KEYS = [
    "recent_window",
    "same_hour_previous_days",
    "same_weekday_recent_weeks",
    "prior_years_same_dates",
    "prior_years_same_week",
    "same_woy_prior_years",
    "same_month_prev_years",
    "macro_trend_blocks",
]

class Timer:
    """Lightweight wall-clock timer."""
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
    def ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0
    def sec(self) -> float:
        return (time.perf_counter() - self.t0)

def jsonable(x: Any, *, max_list: int = 50) -> Any:
    """
    Make objects JSON-serializable and compact:
    - Truncate long lists/dicts
    - Convert DataFrames to small row previews with important columns
    - Convert timestamps via isoformat
    """
    if x is None or isinstance(x, (str, bool, numbers.Number)):
        return x
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x[:max_list]]
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in list(x.items())[:max_list]}
    if isinstance(x, pd.DataFrame):
        cols = [c for c in x.columns if c in ("SETTLEMENTDATE","REGION","TOTALDEMAND","RRP","ret_block","ret_score")]
        cols = cols or list(x.columns)[:6]
        def _cell(v):
            if hasattr(v, "isoformat"):
                return v.isoformat()
            if isinstance(v, (np.floating,)):
                return float(v)
            return v
        return (
            x.head(20)[cols]
             .astype(object)
             .applymap(_cell)
             .to_dict(orient="records")
        )
    if hasattr(x, "isoformat"):
        return x.isoformat()
    try:
        return float(x)
    except Exception:
        return str(x)[:1000]

def topk_by_block(combined: Optional[pd.DataFrame], *, k: int = 100) -> Dict[str, list]:
    """
    Produce a per-block top-K preview (by ret_score) from retrieval_out['combined'].
    Returns JSON-serializable lists.
    """
    if combined is None or not isinstance(combined, pd.DataFrame) or combined.empty:
        return {}
    df = combined.copy()
    if "ret_score" not in df.columns:
        df["ret_score"] = 0.0
    out: Dict[str, list] = {}
    for b in BLOCK_KEYS:
        sub = df[df.get("ret_block") == b].sort_values("ret_score", ascending=False).head(k)
        out[b] = jsonable(sub)
    return out
