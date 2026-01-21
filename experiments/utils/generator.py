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

def fmt_iso(dt): return dt.strftime("%Y/%m/%d %H:%M:%S")
def fmt_nice(dt): return dt.strftime("%B %d, %Y at %H:%M")

def apply_time_bin(base_date, bin_name, rng):
    """
    Apply the time bin and jitter, then round to the nearest 30 minutes.
    """
    tstr, jitter = TIME_BINS[bin_name]
    h, m, s = map(int, tstr.split(":"))
    dt = base_date.replace(hour=h, minute=m, second=s, microsecond=0)
    if jitter > 0:
        dt += timedelta(minutes=rng.randint(-jitter // 2, jitter // 2))

    # ---- round to nearest 30 minutes ----
    minute = dt.minute
    if minute < 15:
        minute = 0
    elif minute < 45:
        minute = 30
    else:
        dt += timedelta(hours=1)
        minute = 0

    dt = dt.replace(minute=minute, second=0, microsecond=0)
    return dt


def next_date_matching_weekend(start, weekend):
    dt = start
    while True:
        if (dt.weekday() >= 5) == weekend:
            return dt
        dt += timedelta(days=1)

# -------------------- DAILY QUERY GENERATOR (25 queries) --------------------
def generate_daily():
    rng = random.Random(7)
    start, end = datetime(2025,5,1), datetime(2025,7,30)
    horizons = ["short_term", "mid_term", "long_term"]

    both_tpl = [
    "Predict the TOTALDEMAND and RRP for {region_name} ({region_code}) on {nice}.",
    "Provide a point forecast of TOTALDEMAND and RRP for {region_name} ({region_code}) on {nice}.",
    "What are the TOTALDEMAND and RRP in {region_name} ({region_code}) at {nice}?"
]

    demand_tpl = [
        "Estimate the TOTALDEMAND for {region_name} ({region_code}) on {nice}.",
        "What is the TOTALDEMAND in {region_name} ({region_code}) at {nice}?"
    ]

    price_tpl = [
        "What is the forecasted RRP in {region_name} ({region_code}) on {nice}?",
        "Estimate the RRP for {region_name} ({region_code}) at {nice}."
    ]


    queries, qid = [], 1
    cycle = ["both", "demand", "price"]

    # evenly sample 25 slots across regions/horizons
    total_needed = 12
    slots_per_region = total_needed // len(REGIONS)
    extra = total_needed % len(REGIONS)

    for i, reg in enumerate(REGIONS):
        count = slots_per_region + (1 if i < extra else 0)
        for _ in range(count):
            code = reg["code"]
            rname = rng.choice(reg["aliases"])
            hzn = rng.choice(horizons)
            dt = start + timedelta(days=rng.randint(0, 90))
            bin_name = rng.choice(list(TIME_BINS.keys()))
            ts = apply_time_bin(dt, bin_name, rng)
            iso, nice = fmt_iso(ts), fmt_nice(ts)
            typ = cycle[(qid-1) % 3]
            tpl = rng.choice(both_tpl if typ=="both" else demand_tpl if typ=="demand" else price_tpl)

            queries.append({
                "id": f"{hzn}_{code}_{qid:03d}",
                "text": tpl.format(region_name=rname, region_code=code, iso=iso, nice=nice),
                "region": code, "region_name": rname, "horizon_hint": hzn,
                "timestamp": iso, "hour_bin": bin_name
            })
            qid += 1
    rng = random.Random(123)
    selected = rng.sample(queries,12)
    return selected


# -------------------- HOURLY QUERY GENERATOR (25 queries) --------------------

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
    "Generate a {desc} for TOTALDEMAND and RRP in {region_name} ({region_code}) starting at {nice}.",
    "Forecast {desc} for {region_name} ({region_code}), beginning at {nice}, including TOTALDEMAND and RRP."
]

    queries, qid = [], 1
    total_needed = 13
    slots_per_region = total_needed // len(REGIONS)
    extra = total_needed % len(REGIONS)

    for i, reg in enumerate(REGIONS):
        count = slots_per_region + (1 if i < extra else 0)
        for _ in range(count):
            code = reg["code"]
            rname = rng.choice(reg["aliases"])
            hours, desc = rng.choice(horizons)
            base_date = rng.choice(start_dates)
            bin_name = rng.choice(list(TIME_BINS.keys()))
            ts = apply_time_bin(base_date, bin_name, rng)
            iso = fmt_iso(ts)
            nice = fmt_nice(ts)
            tpl = rng.choice(templates)

            queries.append({
                "id": f"hourly_{code}_{qid:03d}",
                "text": tpl.format(region_name=rname, region_code=code, iso=iso, nice=nice, desc=desc),
                "region": code, "region_name": rname, "forecast_horizon_hours": hours,
                "description": desc, "start_timestamp": iso,
                "horizon_hint": "short_term", "start_hour_bin": bin_name
            })
            qid += 1
    rng = random.Random(123)
    selected = rng.sample(queries,13)
    return selected


if __name__ == "__main__":
    daily = generate_daily()
    hourly = generate_hourly()
    all_q = daily + hourly

    with open("queries_eval_25.json", "w") as f:
        json.dump(all_q, f, indent=2)

    print(f"✅ Generated {len(daily)} daily + {len(hourly)} hourly = {len(all_q)} total queries")
    print("📄 Saved to queries_eval_25.json")
