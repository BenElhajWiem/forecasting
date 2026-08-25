"""
Prompt-only LLM baseline: sends each evaluation query as a single,
unscaffolded prompt directly to the model (no retrieval, no statistical
grounding, no summarization, no pattern detection, no output-format
constraints, no few-shot exemplars) and checks whether a numeric forecast
for each requested metric can be recovered by the SAME parser used
throughout the evaluation framework (experiments/eval/significance_testing.py).

This quantifies the "responses frequently contained malformed or unusable
outputs" claim in the paper (Section 6, Prompt-Only Evaluation) with real
per-backend counts instead of qualitative language.

Usage:
    python -m experiments.baselines.prompt_only_baseline \
        --queries experiments/queries/queries_eval_25.json \
        --backends openai-mini deepseek-chat \
        --output experiments/baselines/prompt_only_results.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.stubs.model_registry_instrumented import registry, LLMClientAdapter
from experiments.eval.significance_testing import _parse_metric_from_predicted


def _infer_metrics(text: str) -> list[str]:
    t = text.upper()
    metrics = []
    if "TOTALDEMAND" in t:
        metrics.append("TOTALDEMAND")
    if "RRP" in t:
        metrics.append("RRP")
    return metrics or ["TOTALDEMAND"]


SYSTEM_MSG = "You are a time-series forecasting assistant for the Australian electricity market."


def run_backend(preset: str, queries: list[dict]) -> list[dict]:
    spec = registry.get(preset)
    adapter = LLMClientAdapter(spec)
    rows = []
    for q in queries:
        text = q["text"]
        metrics = _infer_metrics(text)
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": text},
        ]
        try:
            raw = adapter.chat(messages, temperature=0.0, max_tokens=4000)
            err = None
        except Exception as e:
            raw = ""
            err = str(e)

        parsed = {}
        for m in metrics:
            parsed[m] = _parse_metric_from_predicted(raw, m) if raw else None

        valid = bool(raw) and all(parsed[m] is not None for m in metrics)

        rows.append({
            "backend": preset,
            "query_id": q["id"],
            "requested_metrics": ",".join(metrics),
            "raw_response": raw,
            "parsed_values": json.dumps(parsed),
            "valid": valid,
            "error": err,
        })
        print(f"  [{preset}] {q['id']}: metrics={metrics} valid={valid}"
              + (f" ERROR={err}" if err else ""))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="experiments/queries/queries_eval_25.json")
    parser.add_argument("--backends", nargs="+", default=["openai-mini", "deepseek-chat"])
    parser.add_argument("--output", default="experiments/baselines/prompt_only_results.csv")
    args = parser.parse_args()

    with open(args.queries, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = data["queries"] if isinstance(data, dict) and "queries" in data else data
    print(f"Loaded {len(queries)} queries from {args.queries}")

    all_rows = []
    for preset in args.backends:
        print(f"\n=== Running backend: {preset} ===")
        all_rows.extend(run_backend(preset, queries))

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved raw results to {args.output}")

    print("\n=== Summary: validation failure rate per backend ===")
    summary = df.groupby("backend")["valid"].agg(["count", "sum"])
    summary["failed"] = summary["count"] - summary["sum"]
    summary["failure_rate_pct"] = (summary["failed"] / summary["count"] * 100).round(1)
    summary = summary.rename(columns={"count": "total", "sum": "passed"})
    print(summary[["total", "passed", "failed", "failure_rate_pct"]])
    summary.to_csv(args.output.replace(".csv", "_summary.csv"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
