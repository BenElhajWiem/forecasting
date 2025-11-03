# ablate.py
from __future__ import annotations
import os, json, time, argparse, hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.io import load_yaml, ensure_dir
from utils.logger import get_logger
from ablation.utils.cost import estimate_cost
from eval.metrics import compute_point_metrics, compute_prob_metrics, compute_coverage
from eval import backtesting

# IMPORTANT: use the instrumented stub so the original pipeline stays untouched
from stubs.orchestration_stub import orchestration_agent
from stubs.model_registry_instrumented import Registry as InstrumentedRegistry, LLMClientAdapter as InstrumentedAdapter

log = get_logger("ablate")

@dataclass
class RunSpec:
    run_id: str
    exp_id: str
    seed: int
    setup: Dict[str, Any]

def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed % (2**32 - 1))

def _extract_query_timestamp(q: Dict[str, Any]) -> str:
    return q.get("timestamp") or q.get("start_timestamp") or ""

def _model_and_adapter(preset: str):
    reg = InstrumentedRegistry()
    spec = reg.get(preset)
    adapter = InstrumentedAdapter(spec)
    try:
        adapter.reset_usage()
    except Exception:
        pass
    return spec, adapter

def _exp_runtime_kwargs(setup: Dict[str, Any], query_iso_hint: str) -> Dict[str, Any]:
    # Map YAML toggles -> stub kwargs. Adjust or extend as you add more switches.
    return dict(
        use_sector_detector=setup.get("sector_detector", True),
        use_horizon_classifier=setup.get("horizon_classifier", True),
        use_statistics_agent=(setup.get("stats_mode","summary_acf") != "off"),
        use_pattern_agent=setup.get("pattern_detection", True),
        use_summarizer=(setup.get("summarization","16sent") != "off"),
        # If you added retrieval blocks in YAML, pass them through:
        retrieval_blocks=setup.get("retrieval_blocks"),
    )

def run_once(spec: RunSpec, queries: List[Dict[str, Any]], backtest_cfg: Dict[str, Any], outdir: str) -> Dict[str, Any]:
    set_seed(spec.seed)
    setup = spec.setup
    model_preset = setup["model"]

    # adapter
    model_spec, adapter = _model_and_adapter(model_preset)

    # rolling splits
    splits = backtesting.monthly_rolling_splits(
        backtest_cfg["start"], backtest_cfg["end"],
        warmup_months=backtest_cfg.get("warmup_months", 1)
    )
    if not splits:
        splits = [dict(train_start="", train_end="", test_start=backtest_cfg["start"], test_end=backtest_cfg["end"])]

    # trace writer
    trace_dir = Path(outdir) / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{spec.exp_id}__seed{spec.seed}.jsonl"
    trace_fh = open(trace_path, "w", encoding="utf-8")

    rows = []
    total_tokens_in = 0.0
    total_tokens_out = 0.0
    t0_all = time.time()

    for sp in splits:
        bt_ctx = {
            "train_start": sp.get("train_start",""),
            "train_end":   sp.get("train_end",""),
            "test_start":  sp.get("test_start",""),
            "test_end":    sp.get("test_end",""),
        }

        for q in queries:
            q_iso = _extract_query_timestamp(q)
            kwargs = _exp_runtime_kwargs(setup, q_iso)

            t_q = time.time()
            out = orchestration_agent(
                user_query=q["text"],
                adapter=adapter,
                return_trace=True,
                trace_topk_per_block=50,
                **kwargs
            )
            latency_q = time.time() - t_q

            y_true = out.get("y_true", np.nan)
            y_pred = out.get("y_pred", np.nan)
            tokens_in = float(out.get("tokens_in", 0.0))
            tokens_out = float(out.get("tokens_out", 0.0))
            total_tokens_in += tokens_in
            total_tokens_out += tokens_out

            rec = {
                "exp_id": spec.exp_id,
                "run_id": spec.run_id,
                "seed": spec.seed,
                "model": model_preset,
                "query_id": q.get("id",""),
                "region": q.get("region",""),
                "horizon_hint": q.get("horizon_hint",""),
                "timestamp": q_iso or bt_ctx["test_start"],
                "train_start": bt_ctx["train_start"],
                "train_end": bt_ctx["train_end"],
                "test_start": bt_ctx["test_start"],
                "test_end": bt_ctx["test_end"],
                "y_true": y_true,
                "y_pred": y_pred,
                "latency_sec": latency_q,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
            }

            # retrieval composition (for EDA later)
            meta = (out.get("trace",{}) or {}).get("retrieval",{}).get("meta",{})
            counts = meta.get("counts",{}) if isinstance(meta, dict) else {}
            rec.update({
                "ret_count_recent":   counts.get("recent_window", 0),
                "ret_count_samehour": counts.get("same_hour_previous_days", 0),
                "ret_count_wkprof":   counts.get("same_weekday_recent_weeks", 0),
                "ret_count_pysamed":  counts.get("prior_years_same_dates", 0),
                "ret_count_pyweek":   counts.get("prior_years_same_week", 0),
                "ret_count_woy":      counts.get("same_woy_prior_years", 0),
                "ret_count_month":    counts.get("same_month_prev_years", 0),
                "ret_count_macro":    counts.get("macro_trend_blocks", 0),
                "ret_total":          meta.get("total_after_merge", 0),
            })

            rows.append(rec)

            # per-query JSONL trace
            trace_obj = {
                "run_id": spec.run_id,
                "exp_id": spec.exp_id,
                "seed": spec.seed,
                "model": model_preset,
                "query_id": rec["query_id"],
                "region": rec["region"],
                "timestamp": rec["timestamp"],
                "trace": out.get("trace", {})
            }
            trace_fh.write(json.dumps(trace_obj, ensure_ascii=False) + "\n")

    trace_fh.close()

    elapsed = time.time() - t0_all
    df = pd.DataFrame(rows)

    # metrics
    pt = compute_point_metrics(df)
    pr = compute_prob_metrics(df)
    cv = compute_coverage(df)

    # cost
    cost_usd = estimate_cost(model_preset, total_tokens_in, total_tokens_out)

    summary = dict(
        run_id=spec.run_id,
        exp_id=spec.exp_id,
        seed=spec.seed,
        model=model_preset,
        latency_sec=elapsed,
        tokens_in=total_tokens_in,
        tokens_out=total_tokens_out,
        cost_usd=cost_usd,
        pt_MAE=pt.get("MAE", np.nan),
        pt_RMSE=pt.get("RMSE", np.nan),
        pt_sMAPE=pt.get("sMAPE", np.nan),
        pt_MAPE=pt.get("MAPE", np.nan),
    )
    for k, v in pr.items(): summary[f"pr_{k}"] = v
    for k, v in cv.items(): summary[f"cv_{k}"] = v

    return dict(summary=summary, detailed=df)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="ablations.yaml")
    ap.add_argument("--queries", default="ablation/utils/queries_eval_25.json")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    ensure_dir(args.outdir)

    with open(args.queries, "r") as f:
        queries = json.load(f)

    summaries: List[Dict[str, Any]] = []
    detailed_rows: List[Dict[str, Any]] = []

    for exp in cfg["experiments"]:
        exp_id = exp["id"]; setup = exp["setup"]
        for seed in cfg["global"]["seeds"]:
            run_id = hashlib.md5(f"{exp_id}-{seed}".encode()).hexdigest()[:10]
            spec = RunSpec(run_id=run_id, exp_id=exp_id, seed=seed, setup=setup)
            res = run_once(spec, queries, cfg["global"]["backtest"], args.outdir)
            summaries.append(res["summary"])
            detailed_rows += res["detailed"].to_dict(orient="records")

    pd.DataFrame(summaries).to_csv(os.path.join(args.outdir, "ablation_results.csv"), index=False)
    pd.DataFrame(detailed_rows).to_csv(os.path.join(args.outdir, "ablation_results_detailed.csv"), index=False)
    print("Saved outputs/ablation_results.csv, outputs/ablation_results_detailed.csv and per-run JSONL traces in outputs/traces/")

if __name__ == "__main__":
    main()