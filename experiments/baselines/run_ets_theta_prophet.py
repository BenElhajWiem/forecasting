"""
Run ETS, Theta, and Prophet baselines against the exact same rows already
present in experiments/baselines/baseline_results.csv (same query_id, region,
metric, target_ts, ground_truth as every other baseline in the paper), and
add three new prediction columns (ets, theta, prophet) to that file.

Fitting is cached per (region, metric) series, same pattern as SARIMA in
classical_baselines.py, since fitting is independent of which query targets
that series.

Usage:
    python -m experiments.baselines.run_ets_theta_prophet
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.baselines.classical_baselines import (
    load_historical,
    ETSConfig, ets_fit, ets_predict_from_fit,
    ThetaConfig, theta_fit, theta_predict_from_fit,
    ProphetConfig, prophet_fit, prophet_predict_from_fit,
)

CSV_PATH = "data/processed_data.csv"
CUTOFF = "2025-04-30 23:30:00"
TZ = "Australia/Sydney"
BASELINE_RESULTS = "experiments/baselines/baseline_results.csv"


def main() -> int:
    df = pd.read_csv(BASELINE_RESULTS)
    # Timestamps carry mixed AEST/AEDT offsets (Australia/Sydney observes DST across
    # the query set's date range), so parse via UTC first, then convert uniformly.
    df["target_ts"] = pd.to_datetime(df["target_ts"], utc=True).dt.tz_convert(TZ)

    series_cache: dict[tuple[str, str], pd.Series] = {}
    ets_cache: dict[tuple[str, str], tuple] = {}
    theta_cache: dict[tuple[str, str], tuple] = {}
    prophet_cache: dict[tuple[str, str], tuple] = {}

    def get_series(region: str, metric: str) -> pd.Series:
        key = (region.upper(), metric)
        if key not in series_cache:
            series_cache[key] = load_historical(CSV_PATH, region, metric, CUTOFF, TZ)
        return series_cache[key]

    ets_col, theta_col, prophet_col = [], [], []

    for i, row in df.iterrows():
        region, metric = row["region"], row["metric"]
        key = (region.upper(), metric)
        ts = row["target_ts"]

        series = get_series(region, metric)
        if series.empty:
            ets_col.append(np.nan); theta_col.append(np.nan); prophet_col.append(np.nan)
            print(f"  [{i+1}/{len(df)}] {row['query_id']:25s} {metric:12s} EMPTY SERIES")
            continue

        if key not in ets_cache:
            print(f"  fitting ETS for {key} ...", flush=True)
            ets_cache[key] = ets_fit(series, ETSConfig())
        if key not in theta_cache:
            print(f"  fitting Theta for {key} ...", flush=True)
            theta_cache[key] = theta_fit(series, ThetaConfig())
        if key not in prophet_cache:
            print(f"  fitting Prophet for {key} ...", flush=True)
            prophet_cache[key] = prophet_fit(series, ProphetConfig())

        e_res, e_last = ets_cache[key]
        t_res, t_last = theta_cache[key]
        p_res, p_last = prophet_cache[key]

        e_pred = ets_predict_from_fit(e_res, e_last, [ts]).get(ts, np.nan)
        t_pred = theta_predict_from_fit(t_res, t_last, [ts]).get(ts, np.nan)
        p_pred = prophet_predict_from_fit(p_res, p_last, [ts]).get(ts, np.nan)

        ets_col.append(e_pred); theta_col.append(t_pred); prophet_col.append(p_pred)
        print(f"  [{i+1}/{len(df)}] {row['query_id']:25s} {metric:12s} "
              f"ETS={e_pred:.2f} Theta={t_pred:.2f} Prophet={p_pred:.2f}  (gt={row['ground_truth']})")

    df["ets"] = ets_col
    df["theta"] = theta_col
    df["prophet"] = prophet_col

    df.to_csv(BASELINE_RESULTS, index=False)
    print(f"\nSaved updated results (with ets/theta/prophet columns) to {BASELINE_RESULTS}")
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    raise SystemExit(main())
