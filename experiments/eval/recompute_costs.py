"""
Recompute per-run cost, latency, and API-call-count statistics from the raw
experiment logs (experiments/outputs/<Backend>/<batch>/<exp_id>/{results.jsonl,calls/}),
using the corrected pricing table in experiments/utils/cost.py.

Motivation (EAAI-26-14664 revision):
  - Reviewer #3 flagged Gemini 2.5 Flash's per-run cost ($15.63) as implausibly
    high relative to Claude Sonnet 4.5 ($4.16) -- more than 10x.
  - Reviewer #2 asked for the number of API calls per run and their variance,
    which the paper did not report.
  - experiments/utils/cost.py had a pricing bug: gemini-2.5-flash input was
    priced at $25/1M tokens instead of the published $0.30/1M (see the NOTE in
    that file). tokens_in/tokens_out logged per call are unaffected by the
    bug -- only the derived cost_usd was wrong -- so costs can be recomputed
    exactly from the raw token counts already on disk.

This script does not call any API or modify the raw logs; it only re-derives
cost_usd from tokens_in/tokens_out and reports summary statistics.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.utils.cost import estimate_cost  # noqa: E402

OUTPUTS_ROOT = Path(__file__).resolve().parents[1] / "outputs"
BACKEND_DIRS = ["Claude", "OpenAI", "Deepseek", "Gemini"]


def find_results_files() -> list[Path]:
    files = []
    for backend in BACKEND_DIRS:
        backend_dir = OUTPUTS_ROOT / backend
        if not backend_dir.exists():
            continue
        files.extend(backend_dir.glob("*/*/results.jsonl"))
    return sorted(files)


def load_calls_count(calls_dir: Path, run_id: str) -> int | None:
    calls_path = calls_dir / f"{run_id}.json"
    if not calls_path.exists():
        return None
    try:
        with open(calls_path) as f:
            calls = json.load(f)
        return len(calls) if isinstance(calls, list) else None
    except Exception:
        return None


def main() -> None:
    rows = []
    for rf in find_results_files():
        exp_dir = rf.parent          # .../<batch>/<exp_id>/
        batch_dir = exp_dir.parent   # .../<batch>/
        backend = batch_dir.parent.name  # Claude / OpenAI / Deepseek / Gemini
        calls_dir = exp_dir / "calls"

        with open(rf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("error"):
                    continue
                tokens_in = d.get("tokens_in") or 0.0
                tokens_out = d.get("tokens_out") or 0.0
                model = d.get("model", "")
                corrected_cost = estimate_cost(model, tokens_in, tokens_out)
                n_calls = load_calls_count(calls_dir, d.get("run_id", ""))

                rows.append({
                    "backend": backend,
                    "batch": batch_dir.name,
                    "exp_id": d.get("exp_id"),
                    "seed": d.get("seed"),
                    "query_id": d.get("query_id"),
                    "run_id": d.get("run_id"),
                    "model": model,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "cost_usd_logged": d.get("cost_usd"),
                    "cost_usd_corrected": corrected_cost,
                    "latency_sec": d.get("latency_sec"),
                    "wall_clock_sec": d.get("wall_clock_sec"),
                    "n_api_calls": n_calls,
                })

    df = pd.DataFrame(rows)
    out_dir = Path(__file__).resolve().parent
    df.to_csv(out_dir / "cost_latency_raw.csv", index=False)
    print(f"Loaded {len(df)} run records from {len(find_results_files())} results.jsonl files")

    # ---- Per-backend summary for the FULL (non-ablated) system -------------
    full_mask = df["exp_id"].str.startswith("FULL", na=False)
    full = df[full_mask]

    print("\n" + "=" * 88)
    print(" Per-run cost / latency / API-call-count -- FULL system, logged vs. corrected pricing")
    print("=" * 88)
    summary_rows = []
    for backend, g in full.groupby("backend"):
        row = {
            "backend": backend,
            "n_runs": len(g),
            "mean_cost_logged_usd": g["cost_usd_logged"].mean(),
            "mean_cost_corrected_usd": g["cost_usd_corrected"].mean(),
            "std_cost_corrected_usd": g["cost_usd_corrected"].std(),
            "mean_n_api_calls": g["n_api_calls"].mean(),
            "std_n_api_calls": g["n_api_calls"].std(),
            "mean_latency_sec": g["latency_sec"].mean(),
            "std_latency_sec": g["latency_sec"].std(),
            "mean_wall_clock_sec": g["wall_clock_sec"].mean(),
        }
        summary_rows.append(row)
        print(f"\n{backend}  (n={row['n_runs']} runs)")
        print(f"  cost/run   logged (buggy pricing):  ${row['mean_cost_logged_usd']:.4f}")
        print(f"  cost/run   corrected:                ${row['mean_cost_corrected_usd']:.4f}  (SD ${row['std_cost_corrected_usd']:.4f})")
        print(f"  API calls/run:                       {row['mean_n_api_calls']:.1f}  (SD {row['std_n_api_calls']:.1f})")
        print(f"  latency/run (sum of call latencies):  {row['mean_latency_sec']:.1f}s")
        print(f"  wall-clock/run:                       {row['mean_wall_clock_sec']:.1f}s")

    summary_df = pd.DataFrame(summary_rows).sort_values("backend")
    summary_df.to_csv(out_dir / "cost_latency_summary_full.csv", index=False)

    # ---- Per (backend, exp_id) summary across ALL ablation configs --------
    print("\n" + "=" * 88)
    print(" Per-run cost / API-call-count -- every ablation configuration")
    print("=" * 88)
    abl_rows = []
    for (backend, exp_id), g in df.groupby(["backend", "exp_id"]):
        abl_rows.append({
            "backend": backend,
            "exp_id": exp_id,
            "n_runs": len(g),
            "mean_cost_corrected_usd": g["cost_usd_corrected"].mean(),
            "std_cost_corrected_usd": g["cost_usd_corrected"].std(),
            "mean_n_api_calls": g["n_api_calls"].mean(),
            "std_n_api_calls": g["n_api_calls"].std(),
            "mean_latency_sec": g["latency_sec"].mean(),
            "mean_latency_ms": g["latency_sec"].mean() * 1000.0,
        })
    abl_df = pd.DataFrame(abl_rows).sort_values(["backend", "exp_id"])
    abl_df.to_csv(out_dir / "cost_latency_summary_by_ablation.csv", index=False)
    print(abl_df.to_string(index=False))

    print(f"\nSaved: {out_dir / 'cost_latency_raw.csv'}")
    print(f"Saved: {out_dir / 'cost_latency_summary_full.csv'}")
    print(f"Saved: {out_dir / 'cost_latency_summary_by_ablation.csv'}")


if __name__ == "__main__":
    main()
