from __future__ import annotations
import os, csv, json, time, hashlib, logging, argparse
from typing import Any, Dict, List, Optional

from utils.model_registry import Registry, LLMClientAdapter
from agents.orchestration_agent import orchestration_agent

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("reproducibility_bench")

# =========================
# Config Variables
# =========================

QUERIES: List[str] = [
    "Predict the demand and price for New South Wales on April the 1st, 2025 at mid-day.",
    "Estimate the total demand for Victoria on April 12, 2025 at 18:00.",
    "What is the forecasted RRP in NSW1 on April 5th, 2025 at 7am ?",
    "Estimate the electricity demand in QLD1 on April 10th, 2025 at midnight.",
    "How much will the RRP cost in TAS1 on April 20th at 4 PM?",
    "What are the TOTALDEMAND and RRP in NSW1 in 2025/04/20 in the late evening ?",
    "RRP in NSW1 on April 25th, 2025 at 6 AM?",
    "What will RRP be in TAS1 at 18:00 on April 20th?",

    "Forecast the next 24 hours of hourly demand for NSW1 starting April 12, 2025 at 00:00",
    "Between 2025-04-12 12:00 and 2025-04-13 12:00, provide price every 30 minutes in VIC1.",
    "Show the demand forecast for QLD1 over the next 48 hours.",
    "Generate day-ahead hourly price and demand forecasts for Queensland region for April 15, 2025."
    "What is the expected trend in TOTALDEMAND for QLD1 during May 2025?",
    "What is the forecasted TOTALDEMAND at 18:00 in VIC1 during the weekends of April 2025?",
    
]

DEFAULT_REPS = [3, 5, 10]

# =========================
# Helpers
# =========================

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def serialize_output(o: Any) -> str:
    if isinstance(o, (dict, list)):
        return json.dumps(o, ensure_ascii=False, sort_keys=True, default=str)
    return str(o)

def ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def print_matrix(preset: str, model_name: str, queries: List[str], results: Dict[int, List[bool]]):
    reps_list = sorted(results.keys())
    header = ["#", "Query (truncated)"] + [f"{r}x" for r in reps_list]
    widths = [3, 48] + [5]*len(reps_list)

    def fmt_row(cols, ws): return "  ".join(str(c)[:w].ljust(w) for c, w in zip(cols, ws))

    print(f"\n=== STABILITY MATRIX — {preset} ({model_name}) ===")
    print(fmt_row(header, widths))
    print(fmt_row(["-"*w for w in widths], widths))
    for i, q in enumerate(queries, 1):
        row = [i, (q[:46] + "…") if len(q) > 47 else q]
        for r in reps_list:
            row.append("OK" if results[r][i-1] else "X")
        print(fmt_row(row, widths))

# =========================
# Core
# =========================

def run_reps_for_query(
    adapter: LLMClientAdapter,
    preset_name: str,
    provider: str,
    model_name: str,
    sdk: str,
    query: str,
    reps: int,
    out_dir: str,
    temperature: float,
    max_tokens: Optional[int],
) -> Dict[str, Any]:
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
    ap = argparse.ArgumentParser(description="LLM reproducibility benchmark")
    ap.add_argument("--presets", default="ALL",
                    help='Comma-separated preset keys, or "ALL" for every preset with an API key.')
    ap.add_argument("--reps", default="3,5,10", help="Repetitions per query, e.g. 3,5,10")
    ap.add_argument("--out_dir", default="repro_logs", help="Directory for logs")
    ap.add_argument("--csv", default="repro_summary.csv", help="Summary CSV filename (inside out_dir)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir
    csv_path = os.path.join(out_dir, args.csv)
    reps_list = [int(x.strip()) for x in args.reps.split(",") if x.strip()]

    registry = Registry()

    # Choose presets
    if args.presets.strip().upper() == "ALL":
        presets = [name for name, spec in registry.presets.items() if spec.api_key]
    else:
        presets = [p.strip() for p in args.presets.split(",") if p.strip()]

    log.info(f"Running presets: {presets}")
    ensure_dir(out_dir)

    # Run per preset
    for preset in presets:
        spec = registry.get(preset)
        adapter = LLMClientAdapter(spec)

        provider = spec.provider
        model_name = spec.model
        sdk = spec.sdk

        log.info(f"=== Testing '{preset}' (provider={provider}, sdk={sdk}, model={model_name}) ===")

        # For the console matrix: reps -> [stable flags per query]
        stability_by_reps: Dict[int, List[bool]] = {r: [] for r in reps_list}

        for q in QUERIES:
            for r in reps_list:
                result = run_reps_for_query(
                    adapter=adapter,
                    preset_name=preset,
                    provider=provider,
                    model_name=model_name,
                    sdk=sdk,
                    query=q,
                    reps=r,
                    out_dir=os.path.join(out_dir, preset),
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                stability_by_reps[r].append(result["stable"])
                append_csv(csv_path, result["rows"])

        print_matrix(preset, model_name, QUERIES, stability_by_reps)

    print(f"\nLogs directory: {out_dir}\nCSV summary: {csv_path}")

if __name__ == "__main__":
    main()