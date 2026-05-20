"""
Classical forecasting baselines for electricity demand (TOTALDEMAND) and price (RRP).

Baselines implemented:
  - Persistence: predict the last observed value before the query timestamp
  - Seasonal Naive: predict the value observed at the same time one season ago
      * short-term  
      * mid-term    
      * long-term   
  - SARIMA: seasonal ARIMA fitted per (region, metric) on the pre-cutoff data
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Data loading helper
# ─────────────────────────────────────────────────────────────────────────────

def load_historical(
    csv_path: str,
    region: str,
    metric: str,
    cutoff: str,
    tz: str = "Australia/Sydney",
) -> pd.Series:
    """
    Load a single (region, metric) time-series from the processed CSV,
    filtering to data strictly before *cutoff*.

    Returns a tz-aware pd.Series indexed by SETTLEMENTDATE, sorted ascending,
    with duplicates and NaNs removed.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"], errors="coerce")

    if getattr(df["SETTLEMENTDATE"].dt, "tz", None) is None:
        df["SETTLEMENTDATE"] = df["SETTLEMENTDATE"].dt.tz_localize(
            tz, ambiguous="NaT", nonexistent="shift_forward"
        )

    cutoff_ts = pd.Timestamp(cutoff).tz_localize(tz) if pd.Timestamp(cutoff).tzinfo is None else pd.Timestamp(cutoff)
    mask = (df["REGION"].str.upper() == region.upper()) & (df["SETTLEMENTDATE"] < cutoff_ts)
    sub = df.loc[mask, ["SETTLEMENTDATE", metric]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna().drop_duplicates("SETTLEMENTDATE").sort_values("SETTLEMENTDATE")
    return sub.set_index("SETTLEMENTDATE")[metric]


# ─────────────────────────────────────────────────────────────────────────────
# Persistence baseline
# ─────────────────────────────────────────────────────────────────────────────

def persistence_predict(
    series: pd.Series,
    target_timestamps: List[pd.Timestamp],
) -> Dict[pd.Timestamp, float]:
    """
    For each target timestamp, return the last observed value in *series*
    that is strictly earlier than the target.
    """
    preds: Dict[pd.Timestamp, float] = {}
    for ts in target_timestamps:
        before = series[series.index < ts]
        preds[ts] = float(before.iloc[-1]) if not before.empty else np.nan
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# Seasonal Naive baseline
# ─────────────────────────────────────────────────────────────────────────────

_HORIZON_LAG_DAYS: Dict[str, int] = {
    "short_term": 7,
    "mid_term": 28,
    "long_term": 364,
}


def seasonal_naive_predict(
    series: pd.Series,
    target_timestamps: List[pd.Timestamp],
    horizon_hint: str = "short_term",
    tolerance_minutes: int = 30,
) -> Dict[pd.Timestamp, float]:
    """
    For each target timestamp t, return the most recent value in *series* that
    shares the same day-of-week and hour (within ±tolerance_minutes).

    This handles the common case where the target is beyond the data cutoff:
    instead of t − N days (which may also be beyond the cutoff), we search
    backwards through the available history for the last observation at the
    same weekday+hour slot.
    """
    tol = pd.Timedelta(minutes=tolerance_minutes)

    preds: Dict[pd.Timestamp, float] = {}
    for ts in target_timestamps:
        target_dow = ts.weekday()
        target_hour = ts.hour
        target_minute = ts.minute

        # Candidates: same weekday, same hour, within ±tolerance on minutes
        candidates = series[
            (series.index.weekday == target_dow)
            & (series.index.hour == target_hour)
            & (np.abs((series.index.minute - target_minute)) <= tolerance_minutes)
        ]

        if candidates.empty:
            # Relax: same hour only (any weekday)
            candidates = series[
                (series.index.hour == target_hour)
                & (np.abs((series.index.minute - target_minute)) <= tolerance_minutes)
            ]

        if not candidates.empty:
            preds[ts] = float(candidates.iloc[-1])
        else:
            preds[ts] = np.nan
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# SARIMA baseline
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SARIMAConfig:
    order: Tuple[int, int, int] = (1, 1, 1)
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 0, 48)  # 48 half-hours per day
    resample_freq: str = "30min"
    max_train_rows: int = 4 * 48 * 7  # 4 weeks of half-hourly data


def sarima_predict(
    series: pd.Series,
    target_timestamps: List[pd.Timestamp],
    cfg: SARIMAConfig = SARIMAConfig(),
) -> Dict[pd.Timestamp, float]:
    """
    Fit a SARIMA model on the most recent *max_train_rows* observations and
    forecast forward to each target timestamp.

    Requires: statsmodels >= 0.13
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as e:
        raise ImportError("statsmodels is required for SARIMA: pip install statsmodels") from e

    # Resample to regular grid
    resampled = series.resample(cfg.resample_freq).mean().interpolate(method="time")
    train = resampled.tail(cfg.max_train_rows).dropna()

    if len(train) < cfg.seasonal_order[3] * 2:
        return {ts: np.nan for ts in target_timestamps}

    preds: Dict[pd.Timestamp, float] = {}
    last_train_ts = train.index[-1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = SARIMAX(
                train,
                order=cfg.order,
                seasonal_order=cfg.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=50)
        except Exception:
            return {ts: np.nan for ts in target_timestamps}

    for ts in target_timestamps:
        if ts <= last_train_ts:
            # Target is within training window — use in-sample fitted value
            if ts in result.fittedvalues.index:
                preds[ts] = float(result.fittedvalues[ts])
            else:
                nearest = result.fittedvalues.index.get_indexer([ts], method="nearest")[0]
                preds[ts] = float(result.fittedvalues.iloc[nearest])
        else:
            steps = int((ts - last_train_ts) / pd.Timedelta(cfg.resample_freq))
            if steps < 1:
                steps = 1
            try:
                fc = result.forecast(steps=steps)
                # fc is indexed by integer steps; pick the last one that is ≤ ts
                preds[ts] = float(fc.iloc[-1])
            except Exception:
                preds[ts] = np.nan

    return preds


# ─────────────────────────────────────────────────────────────────────────────
# Unified runner
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BaselineConfig:
    csv_path: str = "data/processed_data.csv"
    cutoff: str = "2025-04-30 23:30:00"
    tz: str = "Australia/Sydney"
    run_sarima: bool = True
    sarima_cfg: SARIMAConfig = field(default_factory=SARIMAConfig)


def run_all_baselines(
    queries: List[Dict],
    cfg: BaselineConfig = BaselineConfig(),
) -> pd.DataFrame:
    """
    Run Persistence, Seasonal Naive, and (optionally) SARIMA on *queries*.

    Each query dict must have:
      - id, region, metric(s), timestamp (ISO string), horizon_hint
      - For multi-step: a list of timestamps under 'target_timestamps'

    Returns a DataFrame with columns:
      query_id, region, metric, target_ts, persistence, seasonal_naive, sarima, ground_truth
    """
    rows = []

    # Cache series per (region, metric) to avoid redundant CSV reads
    series_cache: Dict[Tuple[str, str], pd.Series] = {}

    def get_series(region: str, metric: str) -> pd.Series:
        key = (region.upper(), metric)
        if key not in series_cache:
            try:
                series_cache[key] = load_historical(cfg.csv_path, region, metric, cfg.cutoff, cfg.tz)
            except Exception as exc:
                warnings.warn(f"Could not load series for {region}/{metric}: {exc}")
                series_cache[key] = pd.Series(dtype=float)
        return series_cache[key]

    for q in queries:
        qid = q.get("id", "")
        region = q.get("region", "")
        horizon_hint = q.get("horizon_hint", "short_term")
        metrics = q.get("metrics", ["TOTALDEMAND"])

        # Resolve target timestamps
        raw_ts = q.get("target_timestamps") or q.get("timestamp") or q.get("start_timestamp")
        if isinstance(raw_ts, list):
            target_tss = [pd.Timestamp(t).tz_localize(cfg.tz) if pd.Timestamp(t).tzinfo is None else pd.Timestamp(t) for t in raw_ts]
        elif raw_ts:
            ts = pd.Timestamp(raw_ts)
            target_tss = [ts.tz_localize(cfg.tz) if ts.tzinfo is None else ts]
        else:
            continue

        gt_map: Dict = q.get("ground_truth", {})

        for metric in metrics:
            series = get_series(region, metric)
            if series.empty:
                continue

            p_preds = persistence_predict(series, target_tss)
            sn_preds = seasonal_naive_predict(series, target_tss, horizon_hint)
            sa_preds = (
                sarima_predict(series, target_tss, cfg.sarima_cfg) if cfg.run_sarima else {}
            )

            for ts in target_tss:
                ts_str = ts.isoformat()
                gt_val = None
                if isinstance(gt_map, dict) and metric in gt_map:
                    entry = gt_map[metric]
                    if isinstance(entry, list):
                        # [[value, ts], ...] or [value, ts]
                        if entry and isinstance(entry[0], list):
                            for pair in entry:
                                if pd.Timestamp(pair[1]) == ts:
                                    gt_val = pair[0]
                                    break
                        elif len(entry) == 2:
                            gt_val = entry[0]

                rows.append({
                    "query_id": qid,
                    "region": region,
                    "metric": metric,
                    "horizon_hint": horizon_hint,
                    "target_ts": ts_str,
                    "persistence": p_preds.get(ts, np.nan),
                    "seasonal_naive": sn_preds.get(ts, np.nan),
                    "sarima": sa_preds.get(ts, np.nan),
                    "ground_truth": gt_val,
                })

    return pd.DataFrame(rows)


def compute_baseline_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MAE and RMSE for each baseline method over rows where ground_truth is available.
    Returns a summary DataFrame with columns: method, metric, MAE, RMSE, n.
    """
    methods = ["persistence", "seasonal_naive", "sarima"]
    records = []
    for metric in df["metric"].unique():
        sub = df[df["metric"] == metric].dropna(subset=["ground_truth"])
        gt = pd.to_numeric(sub["ground_truth"], errors="coerce")
        for method in methods:
            if method not in sub.columns:
                continue
            pred = pd.to_numeric(sub[method], errors="coerce")
            valid = gt.notna() & pred.notna()
            if valid.sum() == 0:
                continue
            errors = (gt[valid] - pred[valid]).abs()
            records.append({
                "method": method,
                "metric": metric,
                "MAE": float(errors.mean()),
                "RMSE": float(np.sqrt((errors**2).mean())),
                "n": int(valid.sum()),
            })
    return pd.DataFrame(records)
