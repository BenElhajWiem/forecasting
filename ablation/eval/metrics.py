# Computes all quantitative metrics from the forecasts.

import numpy as np, pandas as pd

def compute_point_metrics(df: pd.DataFrame) -> dict:
    y, yhat = df["y_true"].astype(float).values, df["y_pred"].astype(float).values
    mae  = float(np.mean(np.abs(y - yhat)))
    rmse = float(np.sqrt(np.mean((y - yhat)**2)))
    smape = float(np.mean(2*np.abs(y - yhat)/(np.abs(y)+np.abs(yhat)+1e-9))) * 100
    mape  = float(np.mean(np.abs((y - yhat)/(y + 1e-9)))) * 100
    return {"MAE": mae, "RMSE": rmse, "sMAPE": smape, "MAPE": mape}

def compute_prob_metrics(df: pd.DataFrame) -> dict:
    need = {"q10","q50","q90"}
    if not need.issubset(df.columns): return {}
    y, q50 = df["y_true"].values, df["q50"].values
    pinball = lambda q, qhat: np.mean(np.maximum(q*(y-qhat), (q-1)*(y-qhat)))
    return {"Pinball@0.5": float(pinball(0.5, q50))}

def compute_coverage(df: pd.DataFrame) -> dict:
    need = {"y_lo","y_hi"}
    if not need.issubset(df.columns): return {}
    y = df["y_true"].values
    cov = float(np.mean((y >= df["y_lo"].values) & (y <= df["y_hi"].values)))
    width = float(np.mean(df["y_hi"].values - df["y_lo"].values))
    return {"PI_cov": cov, "PI_width": width}
