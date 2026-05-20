"""
Run all classical baselines against the evaluation query set and produce
a results CSV.

Usage:
    python -m experiments.baselines.run_baselines \
        --csv data/processed_data.csv \
        --queries experiments/queries/queries_verif_25.json \
        --output experiments/baselines/baseline_results.csv \
        [--no-sarima]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys

import numpy as np
import pandas as pd

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.baselines.classical_baselines import (
    BaselineConfig,
    SARIMAConfig,
    compute_baseline_metrics,
    run_all_baselines,
)


def _parse_gt(gt_raw) -> dict:
    """Parse a raw ground_truth string into a dict if necessary."""
    if isinstance(gt_raw, dict):
        return gt_raw
    if isinstance(gt_raw, str) and gt_raw.strip():
        try:
            return ast.literal_eval(gt_raw)
        except Exception:
            return {}
    return {}


def _infer_metrics(query: dict) -> list[str]:
    """Infer which metrics are requested from the query text/answer."""
    text = (query.get("text", "") + " " + query.get("answer", "")).upper()
    metrics = []
    if "TOTALDEMAND" in text:
        metrics.append("TOTALDEMAND")
    if "RRP" in text:
        metrics.append("RRP")
    return metrics or ["TOTALDEMAND"]


def _infer_region(query: dict) -> str:
    """Infer region code from the query."""
    region = query.get("region", "")
    if region:
        return region.upper()
    text = query.get("text", "").upper()
    for r in ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]:
        if r in text:
            return r
    return "NSW1"


def _infer_horizon(query: dict) -> str:
    h = query.get("horizon_hint", "")
    if h:
        return h
    qid = query.get("id", "")
    if "short" in qid:
        return "short_term"
    if "mid" in qid:
        return "mid_term"
    if "long" in qid:
        return "long_term"
    return "short_term"


def load_queries(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "queries" in data:
        data = data["queries"]
    return data


def prepare_queries_for_baseline(raw_queries: list[dict]) -> list[dict]:
    """Enrich raw query dicts with fields expected by run_all_baselines."""
    prepared = []
    for q in raw_queries:
        qid = q.get("id", "")
        region = _infer_region(q)
        metrics = _infer_metrics(q)
        horizon = _infer_horizon(q)

        # Resolve target timestamps (single or multi-step)
        start_ts = q.get("start_timestamp") or q.get("timestamp")
        gt_raw = _parse_gt(q.get("ground_truth", {}))

        prepared.append({
            "id": qid,
            "region": region,
            "metrics": metrics,
            "horizon_hint": horizon,
            "timestamp": start_ts,
            "target_timestamps": None,
            "ground_truth": gt_raw,
        })
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser(description="Run classical forecasting baselines")
    parser.add_argument("--csv", default="data/processed_data.csv", help="Path to processed data CSV")
    parser.add_argument("--queries", default="experiments/queries/queries_verif_25.json")
    parser.add_argument("--output", default="experiments/baselines/baseline_results.csv")
    parser.add_argument("--cutoff", default="2025-04-30 23:30:00", help="Data availability cutoff (strict <)")
    parser.add_argument("--no-sarima", action="store_true", help="Skip SARIMA (faster)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: data CSV not found at {args.csv}. Please provide the processed data file.", file=sys.stderr)
        return 1

    print(f"Loading queries from {args.queries} ...")
    raw_queries = load_queries(args.queries)
    queries = prepare_queries_for_baseline(raw_queries)
    print(f"  {len(queries)} queries prepared.")

    cfg = BaselineConfig(
        csv_path=args.csv,
        cutoff=args.cutoff,
        run_sarima=not args.no_sarima,
    )

    print("Running baselines ...")
    results_df = run_all_baselines(queries, cfg)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results_df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

    # Print summary metrics
    summary = compute_baseline_metrics(results_df)
    if not summary.empty:
        print("\n=== Baseline Summary Metrics ===")
        print(summary.to_string(index=False))

        summary_path = args.output.replace(".csv", "_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
