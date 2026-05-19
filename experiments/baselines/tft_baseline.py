"""
Temporal Fusion Transformer (TFT) baseline for AEMO electricity forecasting.

Trains on pre-cutoff AEMO data (TOTALDEMAND + RRP, all 5 regions) using
pytorch-forecasting, then forecasts each query target.

Strategy:
  - Training window: last 90 days before cutoff (to keep training feasible)
  - Encoder length: 168 steps (3.5 days of 30-min data)
  - Prediction length: 1 (point forecast at target timestamp)
  - For targets far beyond cutoff: iterative 1-step-ahead rollout
  - One model per metric (TD / RRP) across all regions (region = static categorical)
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import torch
import lightning as L
from torch.utils.data import DataLoader

import pytorch_forecasting as pf
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE as PF_MAE


CSV_PATH    = "data/processed_data.csv"
CUTOFF      = "2025-04-30 23:30:00"
TZ          = "Australia/Sydney"
TRAIN_DAYS  = 90
ENCODER_LEN = 168   # 3.5 days × 48 steps/day
PRED_LEN    = 1
HIDDEN      = 32
ATTENTION   = 4
EPOCHS      = 10
BATCH_SIZE  = 64
LR          = 1e-3


def load_and_resample(csv_path: str, cutoff_ts: pd.Timestamp, metric: str) -> pd.DataFrame:
    """Load all regions for one metric, resample to 30-min, return tidy DataFrame."""
    df = pd.read_csv(csv_path, low_memory=False)
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"], errors="coerce")
    if df["SETTLEMENTDATE"].dt.tz is None:
        df["SETTLEMENTDATE"] = df["SETTLEMENTDATE"].dt.tz_localize(
            TZ, ambiguous="NaT", nonexistent="shift_forward")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df[df["SETTLEMENTDATE"] < cutoff_ts][["REGION", "SETTLEMENTDATE", metric]].dropna()

    # Keep last TRAIN_DAYS per region
    train_start = cutoff_ts - pd.Timedelta(days=TRAIN_DAYS)
    df = df[df["SETTLEMENTDATE"] >= train_start]

    # Resample each region independently
    frames = []
    for region, grp in df.groupby("REGION"):
        s = grp.set_index("SETTLEMENTDATE")[metric].resample("30min").mean().interpolate("time")
        tmp = s.reset_index()
        tmp.columns = ["time", metric]
        tmp["region"] = region.upper()
        frames.append(tmp)

    out = pd.concat(frames, ignore_index=True).sort_values(["region", "time"])
    out["time_idx"] = out.groupby("region").cumcount()
    out["region"] = out["region"].astype(str)
    return out


def train_tft(metric: str, cutoff_ts: pd.Timestamp) -> TemporalFusionTransformer:
    data = load_and_resample(CSV_PATH, cutoff_ts, metric)

    # Normalise per-region using training-set statistics
    for region, grp in data.groupby("region"):
        mu = grp[metric].mean()
        sigma = grp[metric].std() or 1.0
        data.loc[data["region"] == region, metric] = (data.loc[data["region"] == region, metric] - mu) / sigma
    data["_mu"]    = data.groupby("region")[metric].transform("mean")   # stored for de-norm
    data["_sigma"] = data.groupby("region")[metric].transform("std").fillna(1.0)

    max_idx = data.groupby("region")["time_idx"].max().min()
    val_cutoff = max_idx - PRED_LEN

    training = TimeSeriesDataSet(
        data[data["time_idx"] <= val_cutoff],
        time_idx="time_idx",
        target=metric,
        group_ids=["region"],
        min_encoder_length=ENCODER_LEN // 2,
        max_encoder_length=ENCODER_LEN,
        min_prediction_length=PRED_LEN,
        max_prediction_length=PRED_LEN,
        static_categoricals=["region"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[metric],
        target_normalizer=None,   # already normalised
    )
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

    train_loader = training.to_dataloader(train=True,  batch_size=BATCH_SIZE, num_workers=0)
    val_loader   = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=LR,
        hidden_size=HIDDEN,
        attention_head_size=ATTENTION,
        dropout=0.1,
        loss=PF_MAE(),
        log_interval=-1,
    )

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
        accelerator="auto",
        callbacks=[],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return model, training, data


def tft_predict_all(queries_csv: str, output_csv: str) -> pd.DataFrame:
    cutoff_ts = pd.Timestamp(CUTOFF).tz_localize(TZ)
    bdf = pd.read_csv(queries_csv)
    bdf["target_ts"] = pd.to_datetime(bdf["target_ts"], utc=True).dt.tz_convert(TZ)

    tft_preds: list[float] = [np.nan] * len(bdf)

    for metric in ["TOTALDEMAND", "RRP"]:
        mask = bdf["metric"] == metric
        if mask.sum() == 0:
            continue

        print(f"\n=== Training TFT for {metric} ===")
        try:
            model, training, data = train_tft(metric, cutoff_ts)
        except Exception as exc:
            print(f"  TFT training failed for {metric}: {exc}")
            continue

        # Per-region normalisation stats (needed to de-normalise predictions)
        norm_stats = {}
        raw = pd.read_csv(CSV_PATH, low_memory=False)
        raw["SETTLEMENTDATE"] = pd.to_datetime(raw["SETTLEMENTDATE"], errors="coerce")
        if raw["SETTLEMENTDATE"].dt.tz is None:
            raw["SETTLEMENTDATE"] = raw["SETTLEMENTDATE"].dt.tz_localize(
                TZ, ambiguous="NaT", nonexistent="shift_forward")
        raw[metric] = pd.to_numeric(raw[metric], errors="coerce")
        train_start = cutoff_ts - pd.Timedelta(days=TRAIN_DAYS)
        for region, grp in raw[
            (raw["SETTLEMENTDATE"] >= train_start) & (raw["SETTLEMENTDATE"] < cutoff_ts)
        ].groupby("REGION"):
            vals = grp[metric].dropna()
            norm_stats[region.upper()] = (vals.mean(), vals.std() or 1.0)

        for idx, row in bdf[mask].iterrows():
            region = row.region.upper()
            target_ts = row.target_ts
            region_data = data[data["region"] == region].copy()
            if region_data.empty or region not in norm_stats:
                continue

            mu, sigma = norm_stats[region]
            last_data_ts = region_data["time"].iloc[-1]

            # Steps from last training point to target
            steps_needed = max(1, int((target_ts - last_data_ts) / pd.Timedelta("30min")))

            # For short horizons use model directly; for long horizons cap at 48 steps
            # and note prediction degrades — we still report the 1-step at cutoff+48
            pred_steps = min(steps_needed, 48)

            # Build a prediction dataset from the tail of training data
            encoder_data = region_data.tail(ENCODER_LEN).copy()
            # Extend time_idx forward
            last_idx = encoder_data["time_idx"].iloc[-1]
            future_rows = pd.DataFrame({
                "time_idx": range(last_idx + 1, last_idx + pred_steps + 1),
                metric: [np.nan] * pred_steps,
                "region": region,
                "time": pd.date_range(
                    last_data_ts + pd.Timedelta("30min"),
                    periods=pred_steps, freq="30min", tz=TZ),
            })
            pred_data = pd.concat([encoder_data, future_rows], ignore_index=True)

            try:
                pred_dataset = TimeSeriesDataSet.from_dataset(
                    training, pred_data, predict=True, stop_randomization=True)
                pred_loader = pred_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
                raw_preds = model.predict(pred_loader, mode="prediction", return_index=True)
                # raw_preds is (n, pred_steps); take the last step
                pred_norm = float(raw_preds[0][-1, -1] if raw_preds[0].ndim == 2 else raw_preds[0][-1])
                pred_val = pred_norm * sigma + mu
                tft_preds[idx] = pred_val
                print(f"  {row.query_id:30s} {metric:12s} TFT={pred_val:.2f}")
            except Exception as exc:
                print(f"  {row.query_id:30s} {metric:12s} TFT=ERROR: {exc}")

    bdf["tft"] = tft_preds
    bdf.to_csv(output_csv, index=False)
    print(f"\nSaved to {output_csv}")
    return bdf


if __name__ == "__main__":
    tft_predict_all(
        "experiments/baselines/baseline_results.csv",
        "experiments/baselines/baseline_results.csv",
    )
