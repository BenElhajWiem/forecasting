"""
Automated groundedness metric for forecast rationales.

Motivation (EAAI-26-14664 revision, Reviewer #4.3): "Explainability claims
are evaluated qualitatively only. A quantitative evaluation or expert
assessment is needed." This provides a quantitative, automated proxy:
for each generated forecast, what fraction of the numeric values cited in
its rationale (`answer`) are traceable to the statistical/pattern evidence
the pipeline actually retrieved and computed upstream (StatisticalAgent /
pattern_detection, logged in each run's trace file), versus untethered
numbers with no matching evidence value.

This is a TRACEABILITY metric, not a correctness metric: a "grounded" number
is one that matches a value the pipeline actually computed, independent of
whether that was the statistically "right" number to cite. It directly
operationalizes the paper's traceability claim (Introduction, Proposed
System) rather than leaving it as a qualitative/structural assertion.

Data source: experiments/outputs/<Backend>/<batch>/<exp_id>/results.jsonl,
matched to their trace file by run_id (not by the `trace_path` field logged
in results.jsonl, which was found to record an inconsistent/incorrect batch
label -- see note in main()).

Usage:
    python -m experiments.eval.groundedness
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = ROOT / "experiments" / "outputs"
REL_TOL = 0.005  # 0.5% relative tolerance for a numeric "match"
ABS_TOL_FLOOR = 0.5  # minimum absolute tolerance, for values near zero (e.g. RRP)

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _numbers_in_text(text: str) -> list[float]:
    """Extract candidate numeric claims from a forecast rationale.

    Skips small bare integers (years like 2024, counts like "5-year",
    percentages under 1 written as whole numbers) which are common but
    uninformative as "cited evidence" -- keeping them would inflate both
    the match and mismatch counts with numbers nobody intends as evidence.
    """
    out = []
    for m in _NUM_RE.finditer(text or ""):
        s = m.group().replace(",", "")
        try:
            v = float(s)
        except ValueError:
            continue
        if abs(v) < 100 and v == int(v):
            continue
        out.append(v)
    return out


def _flatten_numbers(obj: Any, out: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_numbers(v, out)


def _evidence_numbers(trace: dict) -> np.ndarray:
    out: list[float] = []
    for key in ("statistics", "patterns", "retrieval"):
        if key in trace:
            _flatten_numbers(trace[key], out)
    return np.array(out, dtype=float)


def _match_fraction(claims: list[float], evidence: np.ndarray) -> tuple[Optional[float], int, int]:
    if not claims or evidence.size == 0:
        return None, 0, len(claims)
    n_matched = 0
    for c in claims:
        tol = max(abs(c) * REL_TOL, ABS_TOL_FLOOR)
        if np.any(np.abs(evidence - c) <= tol):
            n_matched += 1
    return n_matched / len(claims), n_matched, len(claims)


def _find_trace_file(run_id: str) -> Optional[Path]:
    matches = list(OUTPUTS_ROOT.glob(f"**/traces/{run_id}.json"))
    return matches[0] if matches else None


def score_all() -> pd.DataFrame:
    rows = []
    for results_file in sorted(OUTPUTS_ROOT.glob("*/*/*/results.jsonl")):
        backend = results_file.parents[2].name
        exp_id = results_file.parent.name
        with open(results_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("error") or not rec.get("answer"):
                    continue
                run_id = rec.get("run_id")
                trace_file = _find_trace_file(run_id) if run_id else None
                if trace_file is None:
                    continue
                try:
                    trace = json.load(open(trace_file)).get("trace", {})
                except Exception:
                    continue

                claims = _numbers_in_text(rec["answer"])
                evidence = _evidence_numbers(trace)
                frac, n_matched, n_claims = _match_fraction(claims, evidence)

                rows.append({
                    "backend": backend,
                    "exp_id": exp_id,
                    "query_id": rec.get("query_id"),
                    "seed": rec.get("seed"),
                    "run_id": run_id,
                    "n_claims": n_claims,
                    "n_matched": n_matched,
                    "groundedness": frac,
                })

    return pd.DataFrame(rows)


def main() -> None:
    df = score_all()
    out_dir = Path(__file__).resolve().parent
    df.to_csv(out_dir / "groundedness_raw.csv", index=False)
    print(f"Scored {len(df)} runs across {df['backend'].nunique() if not df.empty else 0} backends")
    print(f"Saved: {out_dir / 'groundedness_raw.csv'}")

    if df.empty:
        return

    valid = df.dropna(subset=["groundedness"])
    print("\n" + "=" * 78)
    print(" Groundedness (fraction of cited numeric claims traceable to retrieved/")
    print(" computed evidence) -- mean +/- std, by backend, FULL system only")
    print("=" * 78)
    summary_rows = []
    full = valid[valid["exp_id"].str.contains("FULL|reproducibility", case=False, na=False, regex=True)]
    if full.empty:
        full = valid  # fall back to all configs if no FULL-labelled rows found
    for backend, g in full.groupby("backend"):
        mean_g, std_g, n = g["groundedness"].mean(), g["groundedness"].std(), len(g)
        summary_rows.append({"backend": backend, "n_runs": n, "mean_groundedness": mean_g, "std_groundedness": std_g,
                              "mean_n_claims": g["n_claims"].mean()})
        print(f"  {backend:12s}  groundedness = {mean_g:.3f} +/- {std_g:.3f}  (n={n} runs, "
              f"~{g['n_claims'].mean():.1f} numeric claims/rationale)")

    pd.DataFrame(summary_rows).to_csv(out_dir / "groundedness_summary.csv", index=False)
    print(f"\nSaved: {out_dir / 'groundedness_summary.csv'}")


if __name__ == "__main__":
    main()
