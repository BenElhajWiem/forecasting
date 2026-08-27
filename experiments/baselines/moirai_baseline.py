from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.pandas import PandasDataset  # pyright: ignore[reportMissingImports]

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule  # pyright: ignore[reportMissingImports]  # runs in tsfm_env, not the main venv
CSV_PATH = "data/processed_data.csv"
CUTOFF = "2025-04-30 23:30:00"
TZ = "Australia/Sydney"
CONTEXT_DAYS = 90
MAX_HORIZON_STEPS = 512
STEP = "30min"

MODEL_REPO = "Salesforce/moirai-1.0-R-small"
PATCH_SIZE = "auto"


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
    # Resample/strip tz in UTC, not local Australia/Sydney time: the 90-day
    # window crosses the April DST fall-back transition, which makes a
    # local-time-based 30-min grid briefly non-monotonic after stripping tz
    # (observed as a single "-30min" step) -- GluonTS's PandasDataset then
    # rejects the index as non-uniformly spaced. UTC has no DST transitions,
    # so resampling there gives a genuinely uniform physical-time grid.
    sub_utc = sub.tz_convert("UTC")
    series = sub_utc.resample(STEP).mean().interpolate("time")
    series.index = series.index.tz_localize(None)  # GluonTS PandasDataset expects naive index
    return series


def build_predictor(prediction_length: int, context_length: int):
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(MODEL_REPO),
        prediction_length=prediction_length,
        context_length=context_length,
        patch_size=PATCH_SIZE,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.create_predictor(batch_size=8, device=device)


def moirai_predict_all(queries_csv: str, output_csv: str) -> pd.DataFrame:
    cutoff_ts = pd.Timestamp(CUTOFF).tz_localize(TZ)
    bdf = pd.read_csv(queries_csv)
    bdf["target_ts"] = pd.to_datetime(bdf["target_ts"], utc=True).dt.tz_convert(TZ)

    series_cache: dict[tuple[str, str], pd.Series] = {}
    for _, row in bdf.iterrows():
        key = (row.region.upper(), row.metric)
        if key not in series_cache:
            try:
                series_cache[key] = load_series(CSV_PATH, row.region, row.metric, cutoff_ts)
            except Exception as exc:
                print(f"  Could not load series for {key}: {exc}")
                series_cache[key] = pd.Series(dtype=float)

    # One predictor per (region, metric): prediction_length = the max horizon
    # actually needed among that series' queries, capped at MAX_HORIZON_STEPS.
    preds: list[float] = [np.nan] * len(bdf)
    for key, series in series_cache.items():
        if series.empty:
            continue
        region, metric = key
        sub_mask = (bdf["region"].str.upper() == region) & (bdf["metric"] == metric)
        rows = bdf[sub_mask]
        if rows.empty:
            continue

        last_ts_utc = series.index[-1].tz_localize("UTC")  # series index is naive UTC, see load_series
        steps_needed = {
            idx: max(1, int((row.target_ts - last_ts_utc) / pd.Timedelta(STEP)))
            for idx, row in rows.iterrows()
        }
        horizon = min(max(steps_needed.values()), MAX_HORIZON_STEPS)

        try:
            predictor = build_predictor(prediction_length=horizon, context_length=len(series))
            # GluonTS's freq auto-inference (pd.infer_freq) unreliably returns
            # None on this index (observed on tz-stripped Australia/Sydney
            # 30-min data, likely a DST-transition artifact) -- pass it
            # explicitly rather than relying on inference.
            ds = PandasDataset({f"{region}_{metric}": series}, freq=STEP)
            forecast = next(iter(predictor.predict(ds)))
            median_path = np.median(forecast.samples, axis=0)  # (horizon,)
        except Exception as exc:
            print(f"  Moirai failed for {key}: {exc}")
            continue

        for idx, row in rows.iterrows():
            step = min(steps_needed[idx], horizon) - 1
            pred_val = float(median_path[step])
            preds[idx] = pred_val
            print(f"  {row.query_id:30s} {metric:12s} Moirai={pred_val:.2f}  (step={step + 1}/{horizon})")

    bdf["moirai"] = preds
    bdf.to_csv(output_csv, index=False)
    print(f"\nSaved to {output_csv}")
    return bdf


if __name__ == "__main__":
    # Writes to a separate file (not baseline_results.csv directly) to avoid
    # a write-write race with the other new baseline scripts run alongside
    # this one; merged into baseline_results.csv afterward.
    moirai_predict_all(
        "experiments/baselines/baseline_results.csv",
        "experiments/baselines/baseline_results_moirai.csv",
    )
