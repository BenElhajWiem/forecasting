"""
TimesFM (Google, zero-shot) baseline for AEMO electricity forecasting.

Motivation (EAAI-26-14664 revision, Reviewer #4.2): the paper previously
compared only against classical statistical baselines (Persistence, Seasonal
Naive, SARIMA) and one time series foundation model (Chronos). This adds a
second, architecturally distinct zero-shot TSFM explicitly named by the
reviewer, without any training.

Runs in the isolated `tsfm_env` venv (Python >=3.10 required by timesfm;
the main project venv is Python 3.9) -- see README note in that directory.
Data loading mirrors tft_baseline.py's convention (same cutoff, same
fixed-origin protocol per the paper's Evaluation Protocol, Sec 5.3) but is
duplicated rather than imported because tft_baseline.py's other imports
(pytorch_forecasting, lightning) are not installed in tsfm_env.

Usage:
    source tsfm_env/bin/activate
    python -m experiments.baselines.timesfm_baseline
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import timesfm

CSV_PATH = "data/processed_data.csv"
CUTOFF = "2025-04-30 23:30:00"
TZ = "Australia/Sydney"
CONTEXT_DAYS = 90          # same context window as the TFT baseline
MAX_HORIZON_STEPS = 512    # TimesFM 2.5 supports long horizons; capped for tractability
STEP = "30min"

MODEL_REPO = "google/timesfm-2.5-200m-pytorch"


def load_series(csv_path: str, region: str, metric: str, cutoff_ts: pd.Timestamp) -> pd.Series:
    df = pd.read_csv(csv_path, low_memory=False)
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"], errors="coerce")
    if df["SETTLEMENTDATE"].dt.tz is None:
        df["SETTLEMENTDATE"] = df["SETTLEMENTDATE"].dt.tz_localize(
            TZ, ambiguous="NaT", nonexistent="shift_forward")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    mask = (df["REGION"].str.upper() == region.upper()) & (df["SETTLEMENTDATE"] < cutoff_ts)
    sub = df.loc[mask, ["SETTLEMENTDATE", metric]].dropna().drop_duplicates("SETTLEMENTDATE")
    sub = sub.sort_values("SETTLEMENTDATE").set_index("SETTLEMENTDATE")[metric]

    train_start = cutoff_ts - pd.Timedelta(days=CONTEXT_DAYS)
    sub = sub[sub.index >= train_start]
    return sub.resample(STEP).mean().interpolate("time")


def load_model():
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_REPO)
    model.compile(
        timesfm.ForecastConfig(
            max_context=CONTEXT_DAYS * 48,
            max_horizon=MAX_HORIZON_STEPS,
            normalize_inputs=True,
            use_continuous_quantile_head=False,
            force_flip_invariance=True,
            infer_is_positive=True,
        )
    )
    return model


def timesfm_predict_all(queries_csv: str, output_csv: str) -> pd.DataFrame:
    cutoff_ts = pd.Timestamp(CUTOFF).tz_localize(TZ)
    bdf = pd.read_csv(queries_csv)
    bdf["target_ts"] = pd.to_datetime(bdf["target_ts"], utc=True).dt.tz_convert(TZ)

    model = load_model()
    preds: list[float] = [np.nan] * len(bdf)
    series_cache: dict[tuple[str, str], pd.Series] = {}

    for idx, row in bdf.iterrows():
        region = row.region.upper()
        metric = row.metric
        key = (region, metric)
        if key not in series_cache:
            try:
                series_cache[key] = load_series(CSV_PATH, region, metric, cutoff_ts)
            except Exception as exc:
                print(f"  Could not load series for {key}: {exc}")
                series_cache[key] = pd.Series(dtype=float)
        series = series_cache[key]
        if series.empty:
            continue

        steps_needed = max(1, int((row.target_ts - series.index[-1]) / pd.Timedelta(STEP)))
        horizon = min(steps_needed, MAX_HORIZON_STEPS)

        try:
            point_forecast, _ = model.forecast(horizon=horizon, inputs=[series.to_numpy(dtype=float)])
            pred_val = float(point_forecast[0][-1])
            preds[idx] = pred_val
            print(f"  {row.query_id:30s} {metric:12s} TimesFM={pred_val:.2f}  (horizon={horizon}, steps_needed={steps_needed})")
        except Exception as exc:
            print(f"  {row.query_id:30s} {metric:12s} TimesFM=ERROR: {exc}")

    bdf["timesfm"] = preds
    bdf.to_csv(output_csv, index=False)
    print(f"\nSaved to {output_csv}")
    return bdf


if __name__ == "__main__":
    # Writes to a separate file (not baseline_results.csv directly) to avoid
    # a write-write race with the other new baseline scripts run alongside
    # this one; merged into baseline_results.csv afterward.
    timesfm_predict_all(
        "experiments/baselines/baseline_results.csv",
        "experiments/baselines/baseline_results_timesfm.csv",
    )
