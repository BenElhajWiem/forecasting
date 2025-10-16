# repro_benchmark.py (10x-only)
from __future__ import annotations
import os, csv, json, time, hashlib, logging, argparse
from typing import Any, Dict, List, Optional

from utils.model_registry import Registry, LLMClientAdapter
from agents.orchestration_agent import orchestration_agent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("repro")

# =========================
# Your queries
# =========================
QUERIES: List[str] = [
    "Predict the TOTALDEMAND and RRP for NSW1 on 2025-04-01T12:00:00.",
    "Estimate the TOTALDEMAND for VIC1 on 2025-04-12T18:00:00.",
    "What is the forecasted RRP in NSW1 on 2025-04-05T07:00:00?",
    "Estimate the TOTALDEMAND in QLD1 on 2025-04-10T00:00:00.",
    "What will the RRP be in TAS1 on 2025-04-20T16:00:00?",
    "Provide TOTALDEMAND and RRP in NSW1 on 2025-04-20T22:00:00.",
    "What will the RRP be in NSW1 on 2025-04-25T06:00:00?",
    "What will the RRP be in TAS1 on 2025-04-20T18:00:00?",
    "Forecast next 24 hours (hourly) of TOTALDEMAND for NSW1 starting 2025-04-12T00:00:00.",
    "Between 2025-04-12T12:00:00 and 2025-04-13T12:00:00, provide RRP every 30 minutes in VIC1.",
    "Show the TOTALDEMAND forecast for QLD1 for the next 48 hours (hourly) starting now.",
    "Generate day-ahead hourly TOTALDEMAND and RRP forecasts for QLD1 for 2025-04-15.",
    "Describe the expected trend in TOTALDEMAND for QLD1 during 2025-05.",
    "Provide the forecasted TOTALDEMAND at 18:00 on each weekend day in VIC1 during 2025-04.",
    "Predict daily RRP in VIC1 throughout 2025-06.",
    "Show weekly TOTALDEMAND forecasts for NSW1 across 2025-04.",
    "Show the wholesale RRP forecast across all regions (NSW1,VIC1,QLD1,SA1,TAS1) during 2025-06.",
    "Describe how TOTALDEMAND evolves in SA1 from 2025-06-01 to 2025-07-31.",
    "Predict RRP for QLD1 during 2025-06 to 2025-07.",
    "Identify high-demand periods expected in TAS1 between 2025-05-01 and 2025-07-31.",
    "What is the expected trend in TOTALDEMAND for QLD1 on 2025-05-02?",
    "Describe how load evolves in SA1 from 2025-04-01 to 2025-07-31.",
    "Predict RRP during 06:00–12:00 (morning hours) in NSW1 throughout 2025-05 (hourly).",
    "What is the electricity demand expected on public holidays in QLD1 during 2025-Q2?",
    "What is the electricity demand expected on public holidays in NSW1 during 2025-Q4?",
    "Describe how TOTALDEMAND evolves in NSW1 from 2025-04-01 to 2025-07-31.",
    "Identify high-demand periods expected in NSW1 between 2025-05-01 and 2025-07-31.",
]

# =========================
# Helpers
# =========================
def sha1(text: str) -> str:
    import hashlib as _h
    return _h.sha1(text.encode("utf-8")).hexdigest()

def serialize_output(o: Any) -> str:
    if isinstance(o, (dict, list)):
        return json.dumps(o, ensure_ascii=False, sort_keys=True, default=str)
    return str(o)

def ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# =========================
# Core
# =========================
def run_10_reps_for_query(
    adapter: LLMClientAdapter,
    preset_name: str,
    provider: str,
    model_name: str,
    sdk: str,
    query: str,
    out_dir: str,
    temperature: float,
    max_tokens: Optional[int],
) -> Dict[str, Any]:
    reps = 10
    run_stamp = ts()
    jsonl_path = os.path.join(out_dir, f"{preset_name}.reps{reps}.{run_stamp}.jsonl")
    ensure_dir(out_dir)

    hashes: List[str] = []
    rows: List[Dict[str, Any]] = []

    with open(jsonl_path, "a", encoding="utf-8") as jf:
        for i in range(reps):
            t0 = time.perf_counter()
            try:
                out = orchestration_agent(user_query=query, adapter=adapter)
                error = None
            except Exception as e:
                out = f"ERROR: {e}"
                error = str(e)
            latency = round(time.perf_counter() - t0, 3)

            output_text = serialize_output(out)
            h = sha1(output_text)
            hashes.append(h)

            record = {
                "timestamp": run_stamp,
                "provider": provider,
                "model": model_name,
                "preset": preset_name,
                "sdk": sdk,
                "query": query,
                "rep": i + 1,
                "output": output_text,
                "hash": h,
                "latency_s": latency,
                "error": error,
            }
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows.append(record)

    stable = len(set(hashes)) == 1
    return {
        "jsonl_path": jsonl_path,
        "stable": stable,
        "unique_hashes": len(set(hashes)),
        "rows": rows,
    }

def append_csv(csv_path: str, rows: List[Dict[str, Any]]):
    ensure_dir(os.path.dirname(csv_path) or ".")
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow([
                "timestamp","provider","model","preset","sdk",
                "rep_total","rep_idx","latency_s","hash","output_len","query","error","output"
            ])
        rep_total = len(rows)
        for r in rows:
            w.writerow([
                r["timestamp"], r["provider"], r["model"], r["preset"], r["sdk"],
                rep_total, r["rep"], r["latency_s"], r["hash"], len(r["output"]),
                r["query"], r["error"], r["output"]
            ])

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description="LLM reproducibility benchmark (10x per query)")
    ap.add_argument("--presets", default="ALL",
                    help='Comma-separated preset keys, or "ALL" for every preset with an API key.')
    ap.add_argument("--out_dir", default="repro_logs_10x", help="Directory for logs")
    ap.add_argument("--csv", default="repro_summary_10x.csv", help="Summary CSV filename (inside out_dir)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir
    csv_path = os.path.join(out_dir, args.csv)

    registry = Registry()

    # Choose presets
    if args.presets.strip().upper() == "ALL":
        presets = [name for name, spec in registry.presets.items() if spec.api_key]
    else:
        presets = [p.strip() for p in args.presets.split(",") if p.strip()]

    log.info(f"Running 10x reproducibility for presets: {presets}")
    ensure_dir(out_dir)

    for preset in presets:
        spec = registry.get(preset)
        adapter = LLMClientAdapter(spec)

        provider = spec.provider
        model_name = spec.model
        sdk = getattr(spec, "sdk", "unknown")

        log.info(f"=== Testing '{preset}' (provider={provider}, sdk={sdk}, model={model_name}) ===")

        # track per-query stability
        per_query_stable: List[bool] = []

        for q in QUERIES:
            result = run_10_reps_for_query(
                adapter=adapter,
                preset_name=preset,
                provider=provider,
                model_name=model_name,
                sdk=sdk,
                query=q,
                out_dir=os.path.join(out_dir, preset),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            per_query_stable.append(result["stable"])
            append_csv(csv_path, result["rows"])

        # summary for this preset
        stable_count = sum(per_query_stable)
        total = len(per_query_stable)
        log.info(f"✅ Reproducible queries for {preset}: {stable_count}/{total}")
        log.info(f"Logs saved to: {os.path.join(out_dir, preset)}\n")

    log.info(f"All tests done.\nLogs directory: {out_dir}\nCSV summary: {csv_path}")

if __name__ == "__main__":
    main()
