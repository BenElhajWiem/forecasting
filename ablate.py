from __future__ import annotations
import os, sys, json, time, uuid, random, csv, traceback
from typing import Any, Dict, List
import numpy as np
# ---------------------------------------------------------------------
import logging

os.environ["GRPC_VERBOSITY"] = "ERROR"       # Only errors from gRPC
os.environ["GLOG_minloglevel"] = "3"         # 0=INFO,1=WARNING,2=ERROR,3=FATAL
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"       # Suppress absl info/warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # (optional) silence TensorFlow if imported
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # avoids some C++ log spam

# silence grpc & absl loggers inside Python
logging.getLogger("grpc").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.CRITICAL)

# optional: if absl-py is installed, suppress “pre-init STDERR” warning entirely
try:
    from absl import logging as absl_logging
    absl_logging.use_absl_handler()
    absl_logging.set_verbosity(absl_logging.FATAL)
    absl_logging._warn_preinit_stderr = 0  # type: ignore[attr-defined]
except Exception:
    pass
# ---------------------------------------------------------------------


# --- local utils ---
from ablation.utils.io import ensure_dir, load_yaml
from ablation.utils.logger import get_logger
from ablation.stubs.model_registry_instrumented import Registry as InstrumentedRegistry, LLMClientAdapter as InstrumentedAdapter
from ablation.stubs.orchestration_stub import orchestration_agent

log = get_logger("ablate")

# ---------------------------------------------------------------------
# default paths and constants
# ---------------------------------------------------------------------
DEFAULT_YAML = "ablation/ablations.yaml"
DEFAULT_QUERIES = "ablation/queries_eval_25.json"
OUTPUT_ROOT = "ablation/outputs"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def fix_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def _open_csv_append(path: str, fieldnames: List[str]):
    exists = os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not exists:
        writer.writeheader()
    return f, writer

def _append_jsonl(path: str, obj: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_queries(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "queries" in data:
        data = data["queries"]
    if not isinstance(data, list):
        raise ValueError("queries file must be a list[dict] or {'queries': [...]}")

    out = []
    for i, q in enumerate(data):
        if not isinstance(q, dict):
            continue
        if "id" not in q:
            q = {"id": f"q_{i:04d}", **q}
        out.append(q)
    return out

def extract_main_answer(forecast: Dict[str, Any]) -> Any:
    for k in ("forecast", "answer", "point", "value", "text"):
        if k in forecast and forecast[k] is not None:
            return forecast[k]
    return forecast

# ---------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------
def run_single(
    *,
    exp_id: str,
    model_name: str,
    seed: int,
    query: Dict[str, Any],
    exp_setup: Dict[str, Any],
    registry: InstrumentedRegistry,
    output_dir: str,
    csv_writer,
    csv_file,
    jsonl_path: str,
    csv_fields: List[str],
) -> None:

    run_id = uuid.uuid4().hex[:12]
    qid = str(query.get("id"))
    region = query.get("region")
    region_name = query.get("region_name")
    horizon_hint = query.get("horizon_hint")
    start_ts = query.get("start_timestamp") or query.get("timestamp")

    spec = registry.get(model_name)
    adapter = InstrumentedAdapter(spec)

    orch_kwargs: Dict[str, Any] = {}
    for k in (
        "use_sector_detector",
        "use_horizon_classifier",
        "use_statistics_agent",
        "use_pattern_agent",
        "use_summarizer",
    ):
        if k in exp_setup:
            orch_kwargs[k] = bool(exp_setup[k])

    if "retrieval_blocks" in exp_setup:
        orch_kwargs["retrieval_blocks"] = exp_setup["retrieval_blocks"]

    user_query = query.get("text") or query.get("query") or json.dumps(query)

    started = time.time()
    error = None
    forecast_out = None
    trace_obj = None
    main_answer = None
    tokens_in = tokens_out = 0.0
    cost_usd = latency_sec = 0.0

    try:
        fix_seed(seed)
        forecast_out = orchestration_agent(
            user_query=user_query,
            adapter=adapter,
            return_trace=True,
            **orch_kwargs,
        )

        if isinstance(forecast_out, dict):
            trace_obj = forecast_out.get("trace")
            tokens_in = float(forecast_out.get("tokens_in", 0.0))
            tokens_out = float(forecast_out.get("tokens_out", 0.0))

            totals = {}
            try:
                totals = adapter.totals()
            except Exception:
                pass
            if totals:
                cost_usd = float(totals.get("cost_usd", 0.0))
                latency_sec = float(totals.get("latency_sec", 0.0))

            main_answer = extract_main_answer(forecast_out)
        else:
            main_answer = str(forecast_out)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        log.error(f"[{exp_id}] seed={seed} qid={qid} ERROR: {error}")
        log.debug(traceback.format_exc())

    ended = time.time()
    wall_clock = ended - started

    traces_dir = os.path.join(output_dir, "traces")
    calls_dir = os.path.join(output_dir, "calls")
    ensure_dir(traces_dir)
    ensure_dir(calls_dir)

    trace_path = os.path.join(traces_dir, f"{run_id}.json")
    run_trace = {
        "run_id": run_id,
        "exp_id": exp_id,
        "seed": seed,
        "model": model_name,
        "query": query,
        "error": error,
        "started_ts": started,
        "ended_ts": ended,
        "wall_clock_sec": wall_clock,
        "tokens": {"in": tokens_in, "out": tokens_out},
        "cost_usd": cost_usd,
        "latency_sec": latency_sec,
        "trace": trace_obj or {},
    }
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(run_trace, f, ensure_ascii=False, indent=2)

    calls_path = os.path.join(calls_dir, f"{run_id}.json")
    try:
        call_log = adapter.call_log()
    except Exception:
        call_log = []
    with open(calls_path, "w", encoding="utf-8") as f:
        json.dump(call_log, f, ensure_ascii=False, indent=2)

    row = {
        "run_id": run_id,
        "exp_id": exp_id,
        "seed": seed,
        "model": model_name,
        "query_id": qid,
        "region": region,
        "region_name": region_name,
        "horizon_hint": horizon_hint,
        "timestamp": start_ts,
        "answer": main_answer if isinstance(main_answer, (str, int, float)) else json.dumps(main_answer),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost_usd,
        "latency_sec": latency_sec,
        "wall_clock_sec": wall_clock,
        "error": error,
        "trace_path": trace_path,
        "calls_path": calls_path,
    }
    csv_writer.writerow(row)
    csv_file.flush()
    _append_jsonl(os.path.join(output_dir, "results.jsonl"), row)


# ---------------------------------------------------------------------
# Per-experiment summary
# ---------------------------------------------------------------------
def write_summary(output_dir: str, csv_path: str):
    import pandas as pd
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    def safe_sum(c): return float(df[c].fillna(0).sum())
    def safe_mean(c): return float(df[c].astype(float).replace([np.inf, -np.inf], np.nan).dropna().mean() or 0.0)

    summary = {
        "num_runs": len(df),
        "num_errors": int(df["error"].notna().sum() if "error" in df.columns else 0),
        "total_tokens_in": safe_sum("tokens_in"),
        "total_tokens_out": safe_sum("tokens_out"),
        "total_cost_usd": safe_sum("cost_usd"),
        "avg_cost_usd_per_run": safe_mean("cost_usd"),
        "avg_latency_sec": safe_mean("latency_sec"),
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> int:
    cfg = load_yaml(DEFAULT_YAML)
    seeds = cfg.get("global", {}).get("seeds", [7, 21, 42, 63, 84])
    queries = load_queries(DEFAULT_QUERIES)
    log.info(f"Loaded {len(queries)} queries from {DEFAULT_QUERIES}")

    registry = InstrumentedRegistry()
    experiments = cfg.get("experiments", [])
    if not experiments:
        log.error("No experiments found in ablation/ablations.yaml")
        return 2

    csv_fields = [
        "run_id","exp_id","seed","model","query_id","region","region_name",
        "horizon_hint","timestamp","answer","tokens_in","tokens_out",
        "cost_usd","latency_sec","wall_clock_sec","error","trace_path","calls_path",
    ]

    for exp in experiments:
        exp_id = str(exp.get("id"))
        setup = exp.get("setup", {}) if isinstance(exp.get("setup"), dict) else {}
        model_name = setup.get("model") or exp.get("model")
        if not model_name:
            log.warning(f"Experiment {exp_id} missing 'model'; skipping.")
            continue

        mapped_setup = {}
        if "sector_detector" in setup:
            mapped_setup["use_sector_detector"] = bool(setup["sector_detector"])
        if "horizon_classifier" in setup:
            mapped_setup["use_horizon_classifier"] = bool(setup["horizon_classifier"])
        if "pattern_detection" in setup:
            mapped_setup["use_pattern_agent"] = bool(setup["pattern_detection"])
        if "stats_mode" in setup:
            mapped_setup["use_statistics_agent"] = (setup["stats_mode"] != "off")
        if "summarization" in setup:
            mapped_setup["use_summarizer"] = (setup["summarization"] != "off")
        if "retrieval_blocks" in setup:
            mapped_setup["retrieval_blocks"] = setup["retrieval_blocks"]

        exp_dir = os.path.join(OUTPUT_ROOT, exp_id)
        ensure_dir(exp_dir)
        csv_path = os.path.join(exp_dir, "results.csv")

        csv_file, csv_writer = _open_csv_append(csv_path, csv_fields)
        log.info(f"=== Running experiment {exp_id} with model={model_name} ===")

        try:
            for seed in seeds:
                for q in queries:
                    run_single(
                        exp_id=exp_id,
                        model_name=model_name,
                        seed=seed,
                        query=q,
                        exp_setup=mapped_setup,
                        registry=registry,
                        output_dir=exp_dir,
                        csv_writer=csv_writer,
                        csv_file=csv_file,
                        jsonl_path=os.path.join(exp_dir, "results.jsonl"),
                        csv_fields=csv_fields,
                    )
        finally:
            try: csv_file.close()
            except Exception: pass

        write_summary(exp_dir, csv_path)
        log.info(f"[{exp_id}] wrote summary and results to {exp_dir}")

    log.info("All experiments completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())