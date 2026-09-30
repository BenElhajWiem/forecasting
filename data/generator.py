# ================================================================
# This script generates all evaluation queries for ablation study:
#  -  daily-level forecasting queries (short/mid/long term)
#  -  hourly-level forecasting queries (6h, 12h, 24h, 48h horizons)
# ================================================================

from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter
from datetime import datetime, timedelta

T_MAX = datetime(2025, 4, 30, 23, 30)  # data cutoff (Australia/Sydney local time)

# -------------------- Region and time settings (as in generator.py, plus SA1) --------------------

REGIONS = [
    {"code": "NSW1", "aliases": ["New South Wales", "NSW", "NSW1"]},
    {"code": "VIC1", "aliases": ["Victoria", "VIC", "VIC1"]},
    {"code": "QLD1", "aliases": ["Queensland", "QLD", "QLD1"]},
    {"code": "SA1",  "aliases": ["South Australia", "SA", "SA1"]},
    {"code": "TAS1", "aliases": ["Tasmania", "TAS", "TAS1"]},
]
TARGETS = ["TOTALDEMAND", "RRP"]

# Lead-time windows (days after T_MAX) for the first target timestamp.
HORIZON_WINDOWS = {
    "short_term": (1, 15),
    "mid_term": (16, 365),
    "long_term": (366, 450),  # upper bound keeps targets within published AEMO data
}

# Time bins of generator.py; the base time of each bin is used without jitter.
TIME_BINS = {
    "early morning":  "07:00:00",
    "morning":        "09:00:00",
    "late morning":   "11:00:00",
    "noon":           "12:00:00",
    "afternoon":      "15:00:00",
    "late afternoon": "17:00:00",
    "evening":        "19:00:00",
    "early night":    "20:00:00",
    "late evening":   "21:00:00",
    "night":          "22:00:00",
    "midnight":       "00:00:00",
}
BIN_SEQUENCE = list(TIME_BINS.keys())

# Multi-step spans and cadences, as in generator.py.
MULTISTEP_HORIZONS = [
    (6,  "6-hour forecast every hour"),
    (12, "12-hour forecast every 30 minutes"),
    (24, "24-hour hourly forecast"),
    (48, "48-hour every 2 hours forecast"),
]

# Single-target templates, as in generator.py.
POINT_TEMPLATES = {
    "TOTALDEMAND": [
        "Estimate the TOTALDEMAND for {region_name} ({region_code}) on {nice}.",
        "What is the TOTALDEMAND in {region_name} ({region_code}) at {nice}?",
    ],
    "RRP": [
        "What is the forecasted RRP in {region_name} ({region_code}) on {nice}?",
        "Estimate the RRP for {region_name} ({region_code}) at {nice}.",
    ],
}
# Multi-step templates of generator.py (both targets in every query).
MULTI_TEMPLATES = [
    "Generate a {desc} for TOTALDEMAND and RRP in {region_name} ({region_code}) starting at {nice}.",
    "Forecast {desc} for {region_name} ({region_code}), beginning at {nice}, including TOTALDEMAND and RRP.",
]

# -------------------- Helper functions --------------------

def fmt_iso(dt: datetime) -> str:
    return dt.strftime("%Y/%m/%d %H:%M:%S")


def fmt_nice(dt: datetime) -> str:
    return dt.strftime("%B %d, %Y at %H:%M")


def pick_timestamp(rng: random.Random, horizon: str, slot: int) -> tuple[datetime, str]:
    """Draw a date in the horizon window, then set a rotated day type and time bin."""
    lo, hi = HORIZON_WINDOWS[horizon]
    # Draw the base date at least 6 days before the window end, so that moving
    # forward to the requested day type keeps the lead time inside the window.
    day = T_MAX.replace(hour=0, minute=0) + timedelta(days=rng.randint(lo, hi - 6))
    want_weekend = slot % 2 == 1
    for _ in range(7):
        if (day.weekday() >= 5) == want_weekend:
            break
        day += timedelta(days=1)
    bin_name = BIN_SEQUENCE[slot % len(BIN_SEQUENCE)]
    hh, mm, ss = map(int, TIME_BINS[bin_name].split(":"))
    ts = day.replace(hour=hh, minute=mm, second=ss)
    lead = (ts - T_MAX).total_seconds() / 86400
    if not (lo - 1 < lead <= hi):
        raise ValueError(f"lead {lead:.2f} d outside window {HORIZON_WINDOWS[horizon]}")
    return ts, bin_name

# -------------------- Generator --------------------

def generate(per_cell: int, multistep: bool, seed: int) -> tuple[list[dict], list[dict]]:
    """Return (point queries, multi-step queries) in the structure of generator.py.

    Point queries request one target and are crossed over region x target x horizon.
    Multi-step queries request both targets (templates of generator.py) and are
    crossed over region x horizon, with 2 * per_cell queries per combination, so
    that each target is requested by the same number of queries of each type.
    """
    rng = random.Random(seed)
    point, multi = [], []
    # Separate rotation counters per query type, so that day type, time bin,
    # and multi-step shape rotate independently within each type.
    p_slot = m_slot = 0
    for reg, target, horizon in itertools.product(REGIONS, TARGETS, HORIZON_WINDOWS):
        code = reg["code"]
        for _ in range(per_cell):
            ts, bin_name = pick_timestamp(rng, horizon, p_slot)
            rname = rng.choice(reg["aliases"])
            tpl = rng.choice(POINT_TEMPLATES[target])
            point.append({
                "id": f"{horizon}_{code}_{len(point) + 1:03d}",
                "text": tpl.format(region_name=rname, region_code=code, nice=fmt_nice(ts)),
                "region": code, "region_name": rname, "horizon_hint": horizon,
                "timestamp": fmt_iso(ts), "hour_bin": bin_name,
            })
            p_slot += 1

    if multistep:
        for reg, horizon in itertools.product(REGIONS, HORIZON_WINDOWS):
            code = reg["code"]
            for _ in range(2 * per_cell):
                ts, bin_name = pick_timestamp(rng, horizon, m_slot)
                hours, desc = MULTISTEP_HORIZONS[m_slot % len(MULTISTEP_HORIZONS)]
                rname = rng.choice(reg["aliases"])
                tpl = rng.choice(MULTI_TEMPLATES)
                multi.append({
                    "id": f"hourly_{code}_{len(multi) + 1:03d}",
                    "text": tpl.format(desc=desc, region_name=rname,
                                       region_code=code, nice=fmt_nice(ts)),
                    "region": code, "region_name": rname, "forecast_horizon_hours": hours,
                    "description": desc, "start_timestamp": fmt_iso(ts),
                    "horizon_hint": horizon, "start_hour_bin": bin_name,
                })
                m_slot += 1
    return point, multi

# -------------------- MAIN --------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=1,
                    help="point queries per region x target x horizon combination "
                         "(multi-step: twice this number per region x horizon combination)")
    ap.add_argument("--no-multistep", action="store_true", help="point queries only")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--output", default="experiments/queries/queries_eval_balanced.json")
    args = ap.parse_args()

    point, multi = generate(args.per_cell, not args.no_multistep, args.seed)
    all_q = point + multi
    with open(args.output, "w") as f:
        json.dump(all_q, f, indent=2)

    print(f"Generated {len(point)} point + {len(multi)} multi-step = {len(all_q)} total queries")
    print(f"Saved to {args.output}")
    requested = Counter(m for q in all_q for m in TARGETS if m in q["text"])
    print("  region      ", dict(Counter(q["region"] for q in all_q)))
    print("  target      ", dict(requested), "(queries requesting each target)")
    print("  horizon_hint", dict(Counter(q["horizon_hint"] for q in all_q)))


if __name__ == "__main__":
    main()
