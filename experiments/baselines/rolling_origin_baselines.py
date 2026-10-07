"""
Rolling-origin re-run of the classical baselines (EAAI-26-14664R1, Reviewer #2).

The paper evaluates every method from a fixed origin (the data cutoff
2025-04-30 23:30). Here the origin is instead moved to (target - lead) for each
target in baseline_results.csv, and Persistence, Seasonal Naive and SARIMA are
refitted on the history available at that origin. Persistence and Seasonal
Naive use the functions of classical_baselines.py unchanged; SARIMA uses the
same specification (SARIMAConfig: order, seasonal order, 30-minute resampling,
26-week training window, maxiter=50) and is fitted with statsmodels'
low_memory option, which leaves the parameter estimates and out-of-sample
forecasts unchanged but does not store the filter history (several GB per fit
for a seasonal period of 48). Leads of 1 day (day-ahead) and 7 days are evaluated.

History after the cutoff is read from the public AEMO monthly files in
data/post_cutoff/ (PRICE_AND_DEMAND_YYYYMM_REGION.csv).

Output: experiments/baselines/rolling_origin_results.csv
Usage:  python experiments/baselines/rolling_origin_baselines.py [n_workers]
"""
import os

# One thread per process: several fits run in parallel, and multithreaded
# linear algebra in every process oversubscribes the CPU.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import glob
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "baselines"))
from classical_baselines import SARIMAConfig, persistence_predict, seasonal_naive_predict  # noqa: E402

TZ = "Australia/Sydney"
LEADS_DAYS = (1, 7)
HISTORY_WEEKS = 27  # slightly more than the 26-week SARIMA window
BASELINE_CSV = os.path.join(ROOT, "experiments", "baselines", "baseline_results.csv")
OUT_CSV = os.path.join(ROOT, "experiments", "baselines", "rolling_origin_results.csv")


def load_series() -> dict:
    """(region, metric) -> raw series: pre-cutoff archive plus post-cutoff AEMO files, localized as in the fixed-origin baselines."""
    parts = [pd.read_csv(os.path.join(ROOT, "data", "processed_data.csv"), low_memory=False,
                         usecols=["REGION", "SETTLEMENTDATE", "TOTALDEMAND", "RRP"])]
    parts += [pd.read_csv(f, usecols=["REGION", "SETTLEMENTDATE", "TOTALDEMAND", "RRP"])
              for f in sorted(glob.glob(os.path.join(ROOT, "data", "post_cutoff", "PRICE_AND_DEMAND_*.csv")))]
    df = pd.concat(parts, ignore_index=True)
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"], format="mixed", errors="coerce")
    df["SETTLEMENTDATE"] = df["SETTLEMENTDATE"].dt.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
    df["REGION"] = df["REGION"].astype(str).str.upper()
    out = {}
    for (region, ), g in df.groupby(["REGION"]):
        for metric in ("TOTALDEMAND", "RRP"):
            s = g[["SETTLEMENTDATE", metric]].copy()
            s[metric] = pd.to_numeric(s[metric], errors="coerce")
            s = s.dropna().drop_duplicates("SETTLEMENTDATE").sort_values("SETTLEMENTDATE")
            out[(region, metric)] = s.set_index("SETTLEMENTDATE")[metric]
    return out


def sarima_forecast(series: pd.Series, target: pd.Timestamp, cfg: SARIMAConfig = SARIMAConfig()) -> float:
    """Same specification as classical_baselines.sarima_fit / sarima_predict_from_fit, fitted with low_memory=True."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    train = series.resample(cfg.resample_freq).mean().interpolate(method="time").tail(cfg.max_train_rows).dropna()
    if len(train) < cfg.seasonal_order[3] * 2:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = SARIMAX(train, order=cfg.order, seasonal_order=cfg.seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=50, low_memory=True)
            steps = max(1, int((target - train.index[-1]) / pd.Timedelta(cfg.resample_freq)))
            return float(res.forecast(steps=steps).iloc[-1])
        except Exception:
            return np.nan


def _task(args):
    row, target, horizon_hint, lead_days, history = args
    warnings.filterwarnings("ignore")
    return {
        "row": row,
        "lead_days": lead_days,
        "origin": str(target - pd.Timedelta(days=lead_days)),
        "last_observation": str(history.index[-1]),
        "sarima_rolling": sarima_forecast(history, target),
        "persistence_rolling": persistence_predict(history, [target])[target],
        "seasonal_naive_rolling": seasonal_naive_predict(history, [target], horizon_hint)[target],
    }


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    series = load_series()
    base = pd.read_csv(BASELINE_CSV)
    jobs, observed = [], {}
    for i, r in base.iterrows():
        target = pd.Timestamp(r.target_ts).tz_convert(TZ)
        s = series[(r.region.upper(), r.metric)]
        observed[i] = float(s.get(target, np.nan))
        for lead in LEADS_DAYS:
            origin = target - pd.Timedelta(days=lead)
            hist = s[(s.index < origin) & (s.index >= origin - pd.Timedelta(weeks=HISTORY_WEEKS))]
            jobs.append((i, target, r.horizon_hint, lead, hist))
    del series
    results = []
    with Pool(processes=n_workers) as pool:
        for k, res in enumerate(pool.imap_unordered(_task, jobs), start=1):
            results.append(res)
            if k % 5 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} forecasts done", flush=True)
    res = pd.DataFrame(results)
    res["observed_aemo"] = res.row.map(observed)
    out = base.reset_index().rename(columns={"index": "row"}).merge(res, on="row")
    out.to_csv(OUT_CSV, index=False)
    print("saved", OUT_CSV, len(out), "rows")


if __name__ == "__main__":
    main()
