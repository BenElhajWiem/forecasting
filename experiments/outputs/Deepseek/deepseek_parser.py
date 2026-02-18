import re
import pandas as pd
from datetime import datetime

# ============================================================
# Paths
# ============================================================

INPUT_PATH = "DeepSeek_parser.csv"
OUTPUT_PATH = "DeepSeek_eval.csv"

# ============================================================
# Timestamp normalization (AM/PM + month names)
# ============================================================

_MONTH_TS_RE = re.compile(
    r"\b(?P<mon>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+(?P<day>\d{1,2}),\s*(?P<year>\d{4})"
    r"(?:\s*,?\s*(?:at\s*)?(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?(?::(?P<second>\d{2}))?\s*(?P<ampm>am|pm)?)?\b",
    re.IGNORECASE,
)
_NUMERIC_TS_RE = re.compile(r"\b(\d{4}[-/]\d{2}[-/]\d{2})[ T](\d{2}:\d{2})(?::(\d{2}))?\b")

def _normalize_timestamp(ts: str) -> str | None:
    if not isinstance(ts, str):
        return None
    s = ts.strip()
    s = s.replace("**", "").replace("(", " ").replace(")", " ").strip()
    s = re.sub(r"([+-]\d{2}:\d{2}|Z)\b", "", s).strip()
    s = re.sub(r"\s+", " ", s)

    m = _MONTH_TS_RE.search(s)
    if m:
        mon = (m.group("mon") or "").lower()
        day = int(m.group("day"))
        year = int(m.group("year"))

        hour_raw = m.group("hour")
        minute_raw = m.group("minute")
        second_raw = m.group("second")
        ampm = (m.group("ampm") or "").lower()

        hour = int(hour_raw) if hour_raw else 0
        minute = int(minute_raw) if minute_raw else 0
        second = int(second_raw) if second_raw else 0

        if ampm in ("am", "pm"):
            if hour == 12:
                hour = 0 if ampm == "am" else 12
            else:
                hour = hour + 12 if ampm == "pm" else hour

        mon_map = {
            "jan": 1, "january": 1,
            "feb": 2, "february": 2,
            "mar": 3, "march": 3,
            "apr": 4, "april": 4,
            "may": 5,
            "jun": 6, "june": 6,
            "jul": 7, "july": 7,
            "aug": 8, "august": 8,
            "sep": 9, "sept": 9, "september": 9,
            "oct": 10, "october": 10,
            "nov": 11, "november": 11,
            "dec": 12, "december": 12,
        }
        mm = mon_map.get(mon[:3], None) if mon else None
        if mm is None:
            return None

        try:
            dt = datetime(year, mm, day, hour, minute, second)
            return dt.strftime("%Y/%m/%d %H:%M:%S")
        except ValueError:
            return None

    m2 = _NUMERIC_TS_RE.search(s)
    if m2:
        date_part = m2.group(1).replace("-", "/")
        time_part = m2.group(2)
        sec = m2.group(3) or "00"
        return f"{date_part} {time_part}:{sec}"

    fmts = [
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]
    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            return dt.strftime("%Y/%m/%d %H:%M:%S")
        except ValueError:
            pass

    return None


# ============================================================
# Numeric parsing (commas, ~, negatives, currency)
# ============================================================

def _to_float(num_str: str) -> float | None:
    if not isinstance(num_str, str):
        return None
    s = num_str.strip()
    s = s.replace("\u202f", "")  # narrow no-break space
    s = s.replace(",", "")
    s = s.replace("~", "").strip()
    s = s.replace("$", "")
    s = re.sub(r"[^\d\.\-\+]+", "", s)
    if s.count(".") > 1:
        first = s.find(".")
        s = s[:first + 1] + s[first + 1:].replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


def _convert_demand_to_mw(val: float, unit: str | None) -> float:
    if not unit:
        return val
    u = unit.strip().lower()
    if u == "gw":
        return val * 1000.0
    return val


# ============================================================
# Extraction regex
# ============================================================

_METRICS = ("TOTALDEMAND", "RRP")

_TS_ANY_RE = re.compile(
    r"(\d{4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}(?::\d{2})?)|"
    r"((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2},\s*\d{4}(?:\s*,?\s*(?:at\s*)?\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?)",
    re.IGNORECASE,
)

_METRIC_VAL_RE = re.compile(
    r"\b(?P<metric>TOTALDEMAND|RRP)\b\s*[:=]\s*\**\s*(?P<val>~?\s*[-+]?\$?\d[\d,]*\.?\d*)\s*(?P<unit>GW|MW|\$/MWh|USD/MWh|/MWh|MWh)?",
    re.IGNORECASE,
)

_ANY_VAL_WITH_UNIT_RE = re.compile(
    r"(?P<val>~?\s*[-+]?\$?\d[\d,]*\.?\d*)\s*(?P<unit>GW|MW|\$/MWh|USD/MWh|/MWh)\b",
    re.IGNORECASE,
)

_TABLE_ROW_TS_RE = re.compile(r"\b(\d{4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}(?::\d{2})?)\b")

_REJECT_PRE_CTX = ("average", "mean", "median", "range", "observed", "historical", "prior", "evidence", "window")
_ACCEPT_CTX = ("forecast", "prediction", "predictions", "ahead", "forecasts", "here are")

def _extract_all_timestamps(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    out, seen = [], set()
    for m in _TS_ANY_RE.finditer(text):
        raw = m.group(0)
        if not raw:
            continue
        n = _normalize_timestamp(raw)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _append(pred: dict, metric: str, val: float, ts: str):
    pred.setdefault(metric, [])
    pred[metric].append([val, ts])


def _is_reject_context(text: str, idx_start: int, idx_end: int) -> bool:
    w0 = max(0, idx_start - 80)
    w1 = min(len(text), idx_end + 40)
    window = text[w0:w1].lower()
    has_reject = any(k in window for k in _REJECT_PRE_CTX)
    has_accept = any(k in window for k in _ACCEPT_CTX)
    return has_reject and not has_accept


# ============================================================
# Table parsing
# ============================================================

def _split_md_row(line: str) -> list[str]:
    cells = [c.strip() for c in line.split("|")]
    cells = [c for c in cells if c != ""]
    return cells

def _infer_table_schema(header_cells: list[str]) -> dict:
    """
    Returns mapping:
      {'ts_idx': int, 'demand_idx': int|None, 'rrp_idx': int|None, 'demand_unit': 'MW'|'GW'|None}
    """
    header_l = [c.lower() for c in header_cells]
    ts_idx = 0
    demand_idx = None
    rrp_idx = None
    demand_unit = None

    for i, c in enumerate(header_l):
        if "time" in c or "timestamp" in c:
            ts_idx = i
        if "totaldemand" in c:
            demand_idx = i
            if "gw" in c:
                demand_unit = "GW"
            elif "mw" in c:
                demand_unit = "MW"
        if "rrp" in c:
            rrp_idx = i


    if demand_idx is None or rrp_idx is None:
        if len(header_cells) >= 3:

            if demand_idx is None:
                demand_idx = 1
                if "gw" in header_l[1]:
                    demand_unit = "GW"
                elif "mw" in header_l[1]:
                    demand_unit = "MW"
            if rrp_idx is None:
                rrp_idx = 2

    return {
        "ts_idx": ts_idx,
        "demand_idx": demand_idx,
        "rrp_idx": rrp_idx,
        "demand_unit": demand_unit,
    }

_NUM_IN_CELL_RE = re.compile(r"[-+]?\$?\d[\d,]*\.?\d*")

def _parse_number_from_cell(cell: str) -> float | None:
    if not isinstance(cell, str):
        return None
    m = _NUM_IN_CELL_RE.search(cell)
    if not m:
        return None
    return _to_float(m.group(0))

def _parse_markdown_table(text: str) -> dict:
    pred = {}
    if not isinstance(text, str) or "|" not in text:
        return pred

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    n = len(lines)
    i = 0

    while i < n:
        line = lines[i]
        if "|" not in line:
            i += 1
            continue

        cells = _split_md_row(line)
        if len(cells) < 2:
            i += 1
            continue

        # header candidate: contains "time" or "totaldemand" or "rrp"
        header_l = " ".join(cells).lower()
        is_header = ("time" in header_l) and (("totaldemand" in header_l) or ("rrp" in header_l))
        if not is_header:
            i += 1
            continue

        schema = _infer_table_schema(cells)

        # skip separator line like |---|---|---|
        i += 1
        if i < n and re.fullmatch(r"[:\-\|\s]+", lines[i]):
            i += 1

        # parse rows until non-table line
        while i < n and "|" in lines[i]:
            row_cells = _split_md_row(lines[i])
            if len(row_cells) < 2:
                i += 1
                continue

            # locate timestamp cell
            ts_cell = row_cells[schema["ts_idx"]] if schema["ts_idx"] < len(row_cells) else row_cells[0]
            ts_norm = _normalize_timestamp(ts_cell)
            if not ts_norm:
                i += 1
                continue

            # demand
            if schema["demand_idx"] is not None and schema["demand_idx"] < len(row_cells):
                dv = _parse_number_from_cell(row_cells[schema["demand_idx"]])
                if dv is not None:
                    dv = _convert_demand_to_mw(dv, schema["demand_unit"])
                    _append(pred, "TOTALDEMAND", dv, ts_norm)

            # rrp
            if schema["rrp_idx"] is not None and schema["rrp_idx"] < len(row_cells):
                rv = _parse_number_from_cell(row_cells[schema["rrp_idx"]])
                if rv is not None:
                    _append(pred, "RRP", rv, ts_norm)

            i += 1

        continue

    return pred


# ============================================================
# Main per-answer parser
# ============================================================

def parse_answer_to_predicted(answer: str, fallback_row_ts: str | None) -> dict:
    pred: dict = {}
    if not isinstance(answer, str) or not answer.strip():
        return pred

    text = answer
    row_ts_norm = _normalize_timestamp(str(fallback_row_ts)) if fallback_row_ts else None
    ts_all = _extract_all_timestamps(text)

    # 1) Tables
    table_pred = _parse_markdown_table(text)
    for k, v in table_pred.items():
        pred.setdefault(k, []).extend(v)

    # 2) Metric-labeled values (bullets / inline)
    pending = []
    for m in _METRIC_VAL_RE.finditer(text):
        metric = (m.group("metric") or "").upper()
        if metric not in _METRICS:
            continue
        idx0, idx1 = m.start(), m.end()
        if _is_reject_context(text, idx0, idx1):
            continue

        val = _to_float(m.group("val") or "")
        if val is None:
            continue

        unit = (m.group("unit") or "").upper() or None
        w0 = max(0, idx0 - 140)
        w1 = min(len(text), idx1 + 180)
        local_ts = _extract_all_timestamps(text[w0:w1])
        local_ts_norm = local_ts[0] if local_ts else None
        pending.append((metric, val, unit, idx0, idx1, local_ts_norm))

    # 3) Unlabeled values with explicit units
    for m in _ANY_VAL_WITH_UNIT_RE.finditer(text):
        idx0, idx1 = m.start(), m.end()
        if _is_reject_context(text, idx0, idx1):
            continue

        val = _to_float(m.group("val") or "")
        if val is None:
            continue

        unit = (m.group("unit") or "").upper()
        if unit in ("GW", "MW"):
            metric = "TOTALDEMAND"
            val = _convert_demand_to_mw(val, unit)
        else:
            metric = "RRP"

        # avoid duplicates near labeled entries
        if any(pm == metric and abs(p0 - idx0) <= 6 for (pm, _, _, p0, _, _) in pending):
            continue

        w0 = max(0, idx0 - 140)
        w1 = min(len(text), idx1 + 180)
        local_ts = _extract_all_timestamps(text[w0:w1])
        local_ts_norm = local_ts[0] if local_ts else None
        pending.append((metric, val, unit, idx0, idx1, local_ts_norm))

    # 4) Resolve timestamp
    single_answer_ts = ts_all[0] if len(ts_all) == 1 else None

    for metric, val, unit, _, _, local_ts_norm in pending:
        ts_use = local_ts_norm or single_answer_ts

        # safe fallback to row timestamp only when NOT a multi-point table case
        if not ts_use and row_ts_norm and not table_pred:
            ts_use = row_ts_norm

        if not ts_use:
            continue

        if metric == "TOTALDEMAND":
            val = _convert_demand_to_mw(val, unit)

        _append(pred, metric, float(val), ts_use)

    # 5) Clean + dedup + sort
    for metric in list(pred.keys()):
        cleaned = []
        seen = set()
        for v, t in pred[metric]:
            tn = _normalize_timestamp(str(t))
            if tn is None:
                continue
            key = (metric, float(v), tn)
            if key in seen:
                continue
            seen.add(key)
            cleaned.append([float(v), tn])

        try:
            cleaned.sort(key=lambda x: datetime.strptime(x[1], "%Y/%m/%d %H:%M:%S"))
        except Exception:
            pass

        if cleaned:
            pred[metric] = cleaned
        else:
            pred.pop(metric, None)

    return pred


def main():
    df = pd.read_csv(INPUT_PATH)

    if "answer" not in df.columns:
        raise ValueError("CSV must contain an 'answer' column.")
    if "timestamp" not in df.columns:
        df["timestamp"] = None

    predicted_col = []
    for _, row in df.iterrows():
        ans = row.get("answer", "")
        ts = row.get("timestamp", None)
        pred = parse_answer_to_predicted(ans, ts)
        predicted_col.append(repr(pred) if pred else "")

    df["predicted"] = predicted_col
    df.to_csv(OUTPUT_PATH, index=False)

    non_empty = (df["predicted"].astype(str).str.len() > 0).sum()
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Non-empty predicted rows: {non_empty} out of {len(df)}")


if __name__ == "__main__":
    main()