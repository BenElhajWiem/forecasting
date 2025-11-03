# eval/backtesting.py
from __future__ import annotations
from typing import List, Dict, Optional
import pandas as pd

def monthly_rolling_splits(start: str, end: str, warmup_months: int = 1) -> List[Dict]:
    """
    Produce rolling monthly splits:
      train: up to month t (optional warmup)   test: month t+1
    Returns list of dicts with train_start, train_end, test_start, test_end (ISO strings).
    """
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()

    # list first day of each month between s and e (inclusive)
    months = pd.period_range(s, e, freq="M").to_timestamp("M") + pd.offsets.MonthBegin(0)
    if len(months) == 0:
        return []

    splits = []
    # For each month boundary m, test on [m, m+1M) and train is [start, m) (with warmup guard)
    for i in range(len(months) - 1):
        test_start = months[i]
        test_end   = months[i + 1]
        train_end  = test_start
        train_start = (train_end - pd.DateOffset(months=warmup_months)).normalize() if warmup_months > 0 else s

        splits.append(dict(
            train_start=train_start.isoformat(),
            train_end=train_end.isoformat(),
            test_start=test_start.isoformat(),
            test_end=test_end.isoformat(),
        ))
    return splits