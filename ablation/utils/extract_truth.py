#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import json
import argparse
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

AEMO_TZ = "Australia/Sydney"

def prepare_df(
    df: pd.DataFrame,
    *,
    time_col: str = "SETTLEMENTDATE",
    region_col: str = "REGION",
    tz: str = AEMO_TZ,
) -> pd.DataFrame:
    """Normalize dtype/timezone and sort."""
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    if df[time_col].dt.tz is None:
        df[time_col] = df[time_col].dt.tz_localize(tz, nonexistent="shift_forward")
    else:
        df[time_col] = df[time_col].dt.tz_convert(tz)

    df[region_col] = df[region_col].astype(str).str.strip().str.upper()
    return df.sort_values([region_col, time_col]).reset_index(drop=True)

def _ensure_ts(ts_str: str, tz: str) -> pd.Timestamp:
    ts = pd.Timestamp(ts_str)
    if ts.tz is None:
        ts = ts.tz_localize(tz)
    else:
        ts = ts.tz_convert(tz)
    return ts

def _snap_30m(ts: pd.Timestamp) -> pd.Timestamp:
    """Snap to closest :00 or :30."""
    minute = ts.minute
    snapped_min = 0 if minute < 15 else (30 if minute < 45 else 60)
    if snapped_min == 60:
        ts = (ts + pd.Timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    else:
        ts = ts.replace(minute=snapped_min, second=0, microsecond=0)
    return ts

def _infer_freq(description: str) -> str:
    s = (description or "").lower()
    if "every 30 minutes" in s or "30 minutes" in s:
        return "30min"
    if "every 2 hours" in s or "2 hours" in s or "2-hour" in s:
        return "2H"
    if "hourly" in s or "every hour" in s or "1-hour" in s:
        return "1H"
    return "1H"

def _nearest_row_for_region(
    df_region: pd.DataFrame,
    time_col: str,
    target_ts: pd.Timestamp,
    tolerance: pd.Timedelta,
    strict: bool = False,
) -> Optional[pd.Series]:
    s = df_region[time_col]
    if strict:
        # exact timestamp match only
        hit = df_region.loc[s == target_ts]
        return None if hit.empty else hit.iloc[0]

    pos = s.searchsorted(target_ts)
    candidates = []
    if pos > 0: candidates.append(pos - 1)
    if pos < len(s): candidates.append(pos)
    if pos + 1 < len(s): candidates.append(pos + 1)

    best_idx, best_diff = None, None
    for i in candidates:
        diff = abs((s.iat[i] - target_ts))
        if best_diff is None or diff < best_diff:
            best_idx, best_diff = i, diff
    if best_idx is None or best_diff > tolerance:
        return None
    return df_region.iloc[best_idx]

def extract_truth_for_queries(
    df: pd.DataFrame,
    queries: List[Dict[str, Any]],
    *,
    time_col: str = "SETTLEMENTDATE",
    region_col: str = "REGION",
    tz: str = AEMO_TZ,
    tolerance_minutes_point: int = 20,
    tolerance_minutes_series: int = 20,
    snap_to_30m: bool = True,
    strict_time_match: bool = False,
) -> pd.DataFrame:
    """
    Returns tidy DataFrame with:
    ['id','kind','region','request_ts','matched_settlementdate','abs_diff_min','TOTALDEMAND','RRP','status']
    """
    df = prepare_df(df, time_col=time_col, region_col=region_col, tz=tz)
    out_rows: List[Dict[str, Any]] = []

    by_region = {r: g.sort_values(time_col).reset_index(drop=True) for r, g in df.groupby(region_col, sort=False)}
    tol_point = pd.Timedelta(minutes=tolerance_minutes_point)
    tol_series = pd.Timedelta(minutes=tolerance_minutes_series)

    for q in queries:
        qid = q.get("id")
        region = (q.get("region") or "").strip().upper()
        if region not in by_region:
            out_rows.append({
                "id": qid, "kind": "point" if "timestamp" in q else "series",
                "region": region, "request_ts": q.get("timestamp") or q.get("start_timestamp"),
                "matched_settlementdate": pd.NaT, "abs_diff_min": np.nan,
                "TOTALDEMAND": np.nan, "RRP": np.nan,
                "status": f"REGION {region} not in dataframe"
            })
            continue

        df_r = by_region[region]

        # POINT
        if "timestamp" in q and q["timestamp"]:
            req_ts = _ensure_ts(q["timestamp"], tz)
            if snap_to_30m:
                req_ts = _snap_30m(req_ts)

            row = _nearest_row_for_region(
                df_r, time_col, req_ts, tol_point, strict=strict_time_match
            )
            if row is None:
                out_rows.append({
                    "id": qid, "kind": "point", "region": region, "request_ts": req_ts,
                    "matched_settlementdate": pd.NaT, "abs_diff_min": np.nan,
                    "TOTALDEMAND": np.nan, "RRP": np.nan, "status": "no match within tolerance" if not strict_time_match else "no exact-timestamp match"
                })
            else:
                diff_min = abs((row[time_col] - req_ts).total_seconds())/60.0
                out_rows.append({
                    "id": qid, "kind": "point", "region": region, "request_ts": req_ts,
                    "matched_settlementdate": row[time_col], "abs_diff_min": diff_min,
                    "TOTALDEMAND": row.get("TOTALDEMAND", np.nan),
                    "RRP": row.get("RRP", np.nan),
                    "status": "ok"
                })

        # SERIES
        elif all(k in q for k in ("start_timestamp", "forecast_horizon_hours", "description")):
            start_ts = _ensure_ts(q["start_timestamp"], tz)
            if snap_to_30m:
                start_ts = _snap_30m(start_ts)
            freq = _infer_freq(q.get("description", "hourly"))
            end_ts = start_ts + pd.Timedelta(hours=float(q["forecast_horizon_hours"]))
            grid = pd.date_range(start_ts, end_ts, freq=freq, tz=tz, inclusive="left")

            for t in grid:
                t_snap = _snap_30m(t) if snap_to_30m else t
                row = _nearest_row_for_region(
                    df_r, time_col, t_snap, tol_series, strict=strict_time_match
                )
                if row is None:
                    out_rows.append({
                        "id": qid, "kind": "series", "region": region, "request_ts": t,
                        "matched_settlementdate": pd.NaT, "abs_diff_min": np.nan,
                        "TOTALDEMAND": np.nan, "RRP": np.nan,
                        "status": "no match within tolerance" if not strict_time_match else "no exact-timestamp match"
                    })
                else:
                    diff_min = abs((row[time_col] - t_snap).total_seconds())/60.0
                    out_rows.append({
                        "id": qid, "kind": "series", "region": region, "request_ts": t,
                        "matched_settlementdate": row[time_col], "abs_diff_min": diff_min,
                        "TOTALDEMAND": row.get("TOTALDEMAND", np.nan),
                        "RRP": row.get("RRP", np.nan),
                        "status": "ok"
                    })
        else:
            out_rows.append({
                "id": qid, "kind": "unknown", "region": region, "request_ts": None,
                "matched_settlementdate": pd.NaT, "abs_diff_min": np.nan,
                "TOTALDEMAND": np.nan, "RRP": np.nan, "status": "query missing required fields"
            })

    result = pd.DataFrame(out_rows)
    cols = ["id","kind","region","request_ts","matched_settlementdate","abs_diff_min","TOTALDEMAND","RRP","status"]
    result = result[cols].sort_values(["id","request_ts"], kind="stable").reset_index(drop=True)
    return result

def main():
    p = argparse.ArgumentParser(description="Extract ground-truth values for queries from an AEMO-like dataframe.")
    p.add_argument("--df", required=True, help="Path to dataframe file (CSV or Parquet).")
    p.add_argument("--queries", required=True, help="Path to queries JSON file (list[dict]).")
    p.add_argument("--out", required=True, help="Output path for CSV.")
    p.add_argument("--out-parquet", default=None, help="Optional Parquet output path.")
    p.add_argument("--time-col", default="SETTLEMENTDATE", help="Timestamp column name in df.")
    p.add_argument("--region-col", default="REGION", help="Region column name in df.")
    p.add_argument("--tz", default=AEMO_TZ, help="Timezone to normalize to (default Australia/Sydney).")
    p.add_argument("--tolerance-point", type=int, default=20, help="Minutes tolerance for point queries.")
    p.add_argument("--tolerance-series", type=int, default=20, help="Minutes tolerance for series queries.")
    p.add_argument("--no-snap", action="store_true", help="Disable snapping to :00/:30.")
    p.add_argument("--strict", action="store_true", help="Require exact timestamp matches (no nearest-in-tolerance).")
    args = p.parse_args()

    # Load df
    if args.df.lower().endswith(".parquet"):
        df = pd.read_parquet(args.df)
    else:
        df = pd.read_csv(args.df)

    # Parse dates if needed
    if args.time_col in df.columns and not np.issubdtype(df[args.time_col].dtype, np.datetime64):
        df[args.time_col] = pd.to_datetime(df[args.time_col], errors="coerce")

    # Load queries JSON
    with open(args.queries, "r") as f:
        data = json.load(f)
    # Accept either a top-level list or {"queries": [...]}
    queries = data if isinstance(data, list) else data.get("queries", [])

    # Extract
    truth = extract_truth_for_queries(
        df=df,
        queries=queries,
        time_col=args.time_col,
        region_col=args.region_col,
        tz=args.tz,
        tolerance_minutes_point=args.tolerance_point,
        tolerance_minutes_series=args.tolerance_series,
        snap_to_30m=not args.no_snap,
        strict_time_match=args.strict,
    )

    # Save
    truth.to_csv(args.out, index=False)
    if args.out_parquet:
        truth.to_parquet(args.out_parquet, index=False)

    # Quick summary
    ok = (truth["status"] == "ok").sum()
    total = len(truth)
    print(f"Saved {args.out} ({ok}/{total} matched rows).")
    if args.out_parquet:
        print(f"Also wrote {args.out_parquet}.")

if __name__ == "__main__":
    main()
