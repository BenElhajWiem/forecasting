# ================================================================
# This script generates all evaluation queries for ablation study:
#  -  daily-level forecasting queries (short/mid/long term)
#  -  hourly-level forecasting queries (6h, 12h, 24h, 48h horizons)
#  - Merges them into a master file: queries_eval.json
#  - All dates cover May–July 2025 across NSW1, VIC1, QLD1, TAS1.
#  -  Generated 192 daily + 96 hourly = 288 total queries
# ================================================================


import json, random
from datetime import datetime, timedelta

# -------------------- Region and time settings --------------------

REGIONS = [
    {"code": "NSW1", "aliases": ["New South Wales", "NSW", "NSW1"]},
    {"code": "VIC1", "aliases": ["Victoria", "VIC", "VIC1"]},
    {"code": "QLD1", "aliases": ["Queensland", "QLD", "QLD1"]},
    {"code": "TAS1", "aliases": ["Tasmania", "TAS", "TAS1"]},
]

TIME_BINS = {
    "early morning": ("07:00:00",   0),
    "morning":       ("09:00:00", 180),
    "late morning":  ("11:00:00", 120),
    "noon":          ("12:00:00",  60),
    "afternoon":     ("15:00:00", 180),
    "late afternoon":("17:00:00", 120),
    "evening":       ("19:00:00", 180),
    "early night":   ("20:00:00", 120),
    "late evening":  ("21:00:00", 120),
    "night":         ("22:00:00", 180),
    "midnight":      ("00:00:00",  60),
}
BIN_SEQUENCE = list(TIME_BINS.keys())

# -------------------- Helper functions --------------------

def fmt_iso(dt): return dt.strftime("%Y-%m-%dT%H:%M:%S")
def fmt_nice(dt): return dt.strftime("%B %d, %Y at %H:%M")

def apply_time_bin(base_date, bin_name, rng):
    tstr, jitter = TIME_BINS[bin_name]
    h, m, s = map(int, tstr.split(":"))
    dt = base_date.replace(hour=h, minute=m, second=s, microsecond=0)
    if jitter > 0:
        dt += timedelta(minutes=rng.randint(-jitter//2, jitter//2))
    return dt

def next_date_matching_weekend(start, weekend):
    dt = start
    while True:
        if (dt.weekday() >= 5) == weekend:
            return dt
        dt += timedelta(days=1)

# -------------------- DAILY QUERY GENERATOR --------------------

def generate_daily():
    rng = random.Random(7)
    start, end = datetime(2025,5,1), datetime(2025,7,30)
    horizons = ["short_term", "mid_term", "long_term"]

    both_tpl = [
        "Predict the TOTALDEMAND and RRP for {region_name} on {iso}.",
        "Provide a point forecast of TOTALDEMAND and RRP for {region_name} on {iso}.",
        "What are the TOTALDEMAND and RRP in {region_name} at {nice}? ({iso})"
    ]
    demand_tpl = [
        "Estimate the TOTALDEMAND for {region_name} on {iso}.",
        "TOTALDEMAND in {region_name} at {nice}? ({iso})"
    ]
    price_tpl = [
        "What is the forecasted RRP in {region_name} on {iso}?",
        "Estimate the RRP for {region_name} at {nice}. ({iso})"
    ]

    def make_slots(seed_shift):
        slots, cur = [], start + timedelta(days=seed_shift)
        weekend_flags = [False, True]*8
        for i in range(16):
            bin_name = BIN_SEQUENCE[i % len(BIN_SEQUENCE)]
            weekend = weekend_flags[i]
            dt = next_date_matching_weekend(cur, weekend)
            cur = dt + timedelta(days=2)
            if cur > end: cur = start + timedelta(days=(i+seed_shift)%10)
            ts = apply_time_bin(dt, bin_name, rng)
            slots.append((ts, bin_name, weekend))
        return slots[:16]

    horizon_slots = {"short_term": make_slots(0),
                     "mid_term": make_slots(4),
                     "long_term": make_slots(8)}

    queries, qid = [], 1
    cycle = ["both", "demand", "price"]

    for reg in REGIONS:
        code = reg["code"]
        for hzn in horizons:
            for dt, bin_name, is_wkd in horizon_slots[hzn]:
                typ = cycle[(qid-1)%3]
                tpl = rng.choice(both_tpl if typ=="both" else demand_tpl if typ=="demand" else price_tpl)
                rname = rng.choice(reg["aliases"])
                iso, nice = fmt_iso(dt), fmt_nice(dt)
                queries.append({
                    "id": f"{hzn}_{code}_{qid:03d}",
                    "text": tpl.format(region_name=rname, region_code=code, iso=iso, nice=nice),
                    "region": code, "region_name": rname, "horizon_hint": hzn,
                    "timestamp": iso, "hour_bin": bin_name, "is_weekend": is_wkd
                })
                qid += 1
    return queries

# -------------------- HOURLY QUERY GENERATOR --------------------

def generate_hourly():
    rng = random.Random(18)
    start_dates = [
        datetime(2025,5,5,0,0), datetime(2025,5,20,6,0),
        datetime(2025,6,3,12,0), datetime(2025,6,18,18,0),
        datetime(2025,7,1,0,0),  datetime(2025,7,15,0,0)
    ]
    horizons = [
        (6,  "6-hour forecast every hour"),
        (12, "12-hour forecast every 30 minutes"),
        (24, "24-hour hourly forecast"),
        (48, "48-hour every 2 hours forecast")
    ]
    templates = [
        "Generate a {desc} for TOTALDEMAND and RRP in {region_name} starting {iso}.",
        "Forecast {desc} for {region_name}, beginning at {iso}, including TOTALDEMAND and RRP."
    ]

    queries, qid = [], 1
    for reg in REGIONS:
        code = reg["code"]
        for base_date in start_dates:
            for hours, desc in horizons:
                tpl = rng.choice(templates)
                rname = rng.choice(reg["aliases"])
                bin_name = BIN_SEQUENCE[qid % len(BIN_SEQUENCE)]
                ts = apply_time_bin(base_date, bin_name, rng)
                iso = fmt_iso(ts)
                queries.append({
                    "id": f"hourly_{code}_{qid:03d}",
                    "text": tpl.format(region_name=rname, region_code=code, iso=iso, desc=desc),
                    "region": code, "region_name": rname, "forecast_horizon_hours": hours,
                    "description": desc, "start_timestamp": iso,
                    "horizon_hint": "short_term", "start_hour_bin": bin_name
                })
                qid += 1
    return queries

# -------------------- MAIN --------------------

if __name__ == "__main__":
    daily = generate_daily()    
    hourly = generate_hourly()  
    all_q = daily + hourly      

    with open("ablation/queries_eval.json", "w") as f:
        json.dump(all_q, f, indent=2)

    print(f"✅ Generated {len(daily)} daily + {len(hourly)} hourly = {len(all_q)} total queries")
    print("📄 Saved to queries_eval.json")
