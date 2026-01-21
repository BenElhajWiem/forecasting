import re
import pandas as pd
from datetime import datetime

INPUT_CSV = "Gemini_parser.csv"
OUTPUT_CSV = "Gemini_eval.csv"

ANSWER_COL = "answer"
ROW_TS_COL = "timestamp"
OUT_COL    = "predicted"

# ============================================================
# REGEX
# ============================================================
# Timestamps:
# - 2025-09-22 02:00
# - 2025-09-22 02:00:00
# - 2025/09/22 02:00
# - supports "T" between date/time
TS_RE = re.compile(
    r"(?<!\d)"
    r"(20\d{2})[/-](\d{1,2})[/-](\d{1,2})"
    r"(?:[ T])"
    r"(\d{1,2}):(\d{2})"
    r"(?::(\d{2}))?"
    r"(?!\d)"
)

# Numbers (accept commas)
NUM_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")

# Units (strict)
MW_RE       = re.compile(r"\bMW\b", re.IGNORECASE)
RRP_UNIT_RE = re.compile(r"\$/\s*MWh|\$/MWh", re.IGNORECASE)

# Keywords
TD_KW_RE  = re.compile(r"\bTOTALDEMAND\b", re.IGNORECASE)
RRP_KW_RE = re.compile(r"\bRRP\b", re.IGNORECASE)

# ============================================================
# HELPERS
# ============================================================
def to_float(s: str):
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None

def normalize_timestamp(s: str):
    """
    Returns 'YYYY/MM/DD HH:MM:SS' or None.
    """
    if not s:
        return None
    m = TS_RE.search(s)
    if not m:
        return None
    y, mo, d, hh, mm, ss = m.groups()
    ss = ss or "00"
    try:
        dt = datetime(int(y), int(mo), int(d), int(hh), int(mm), int(ss))
    except Exception:
        return None
    return dt.strftime("%Y/%m/%d %H:%M:%S")

def append(out: dict, key: str, val, ts: str):
    if val is None or ts is None:
        return
    out.setdefault(key, []).append([val, ts])

def dedupe_sort(out: dict):
    for k, pairs in list(out.items()):
        seen = set()
        cleaned = []
        for v, ts in pairs:
            sig = (k, v, ts)
            if sig not in seen:
                seen.add(sig)
                cleaned.append([v, ts])
        try:
            cleaned.sort(key=lambda p: datetime.strptime(p[1], "%Y/%m/%d %H:%M:%S"))
        except Exception:
            pass
        out[k] = cleaned
    return out

def strip_num(cell: str):
    """
    Extract first number from a cell like '6388.715 MW' or '266.255 $/MWh' or '6251.37'.
    """
    if not cell:
        return None
    m = NUM_RE.search(cell)
    if not m:
        return None
    return to_float(m.group(0))

def is_alignment_row(cells):
    def is_align(c):
        c = (c or "").strip().replace(" ", "")
        return bool(re.fullmatch(r":?-{2,}:?", c))
    return len(cells) >= 2 and all(is_align(c) for c in cells)

def norm_header(s: str):
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def remove_timestamp_from_line(line: str):
    """
    Prevent grabbing HH/MM/SS from timestamp as forecast values.
    """
    m = TS_RE.search(line or "")
    if not m:
        return line or ""
    return (line[:m.start()] + " " + line[m.end():]).strip()

# ============================================================
# TABLE EXTRACTION (covers all examples)
# ============================================================
def extract_from_markdown_tables(text: str):
    """
    Supports:
    A) | Time | TOTALDEMAND (MW) | RRP ($/MWh) |
       rows like: | 2025-11-03 22:00:00 | 6251.37 | 78.86 |
       (units may be in header only; cells may be pure numbers)

    B) | Time | Metric | Forecast | Unit |
       rows like: | 2025-06-03 12:00:00 | TOTALDEMAND | 7200.485 | MW |
       also handles lines where Time cell is blank for RRP: '| | RRP | 54.8 | $/MWh |'
       by carrying forward last seen time.

    C) | Time | TOTALDEMAND Forecast | RRP Forecast |
       rows like: | 2025-07-01 20:30:00 | 6388.715 MW | 266.255 $/MWh |
       (cells include units)
    """
    if not text or "|" not in text:
        return {}

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out = {}

    i = 0
    while i < len(lines):
        if "|" not in lines[i]:
            i += 1
            continue

        header_cells = [c.strip() for c in lines[i].split("|") if c.strip()]
        if len(header_cells) < 2:
            i += 1
            continue

        header_norm = [norm_header(c) for c in header_cells]
        header_join = " ".join(header_norm)

        # detect forecast table header
        looks_like_table = ("time" in header_join) and (
            ("totaldemand" in header_join and "rrp" in header_join)
            or ("metric" in header_join and "forecast" in header_join)
            or ("forecast" in header_join and ("totaldemand" in header_join or "rrp" in header_join))
        )
        if not looks_like_table:
            i += 1
            continue

        # header-based unit flags (important for tables where cells have no units)
        header_has_mw = ("mw" in header_join)
        header_has_rrp_unit = ("/mwh" in header_join) or ("$/mwh" in header_join) or ("$/ mwh" in header_join)

        # index mapping
        idx_time = next((k for k, h in enumerate(header_norm) if h.startswith("time")), None)
        idx_metric = next((k for k, h in enumerate(header_norm) if h == "metric"), None)
        idx_forecast = next((k for k, h in enumerate(header_norm) if h == "forecast"), None)
        idx_unit = next((k for k, h in enumerate(header_norm) if h == "unit"), None)

        # schema A/C indices
        idx_td = next((k for k, h in enumerate(header_norm) if "totaldemand" in h), None)
        idx_rrp = next((k for k, h in enumerate(header_norm) if h == "rrp" or h.startswith("rrp ")), None)

        # skip alignment row if present
        j = i + 1
        if j < len(lines) and "|" in lines[j]:
            align_cells = [c.strip() for c in lines[j].split("|") if c.strip()]
            has_align = is_alignment_row(align_cells)
        else:
            has_align = False

        k = i + 1 + (1 if has_align else 0)

        last_ts = None
        any_row = False

        while k < len(lines) and "|" in lines[k]:
            row_cells = [c.strip() for c in lines[k].split("|") if c.strip() or c == ""]
            # Preserve empty cells: splitting with filter drops empties; redo carefully.
            # We'll do a safer parse here:
            raw_parts = [p.strip() for p in lines[k].split("|")]
            # drop leading/trailing empties caused by leading/trailing '|'
            if raw_parts and raw_parts[0] == "":
                raw_parts = raw_parts[1:]
            if raw_parts and raw_parts[-1] == "":
                raw_parts = raw_parts[:-1]
            cells = raw_parts

            if not cells or is_alignment_row([c for c in cells if c.strip()]):
                k += 1
                continue

            # stop if next header begins
            row_join = " ".join(norm_header(c) for c in cells if c)
            if ("time" in row_join and "totaldemand" in row_join) or ("time" in row_join and "metric" in row_join and "forecast" in row_join):
                break

            # ---------- Schema B: Time | Metric | Forecast | Unit ----------
            if idx_metric is not None and idx_forecast is not None:
                raw_time = cells[idx_time] if (idx_time is not None and idx_time < len(cells)) else ""
                ts_norm = normalize_timestamp(raw_time) if raw_time and normalize_timestamp(raw_time) else None
                if ts_norm:
                    last_ts = ts_norm
                ts_use = ts_norm or last_ts
                if not ts_use:
                    k += 1
                    continue

                metric_cell = cells[idx_metric] if idx_metric < len(cells) else ""
                forecast_cell = cells[idx_forecast] if idx_forecast < len(cells) else ""
                unit_cell = cells[idx_unit] if (idx_unit is not None and idx_unit < len(cells)) else ""

                val = strip_num(forecast_cell)
                if val is None:
                    k += 1
                    continue

                metric_u = (metric_cell or "").strip().upper()
                if metric_u == "TOTALDEMAND" and (MW_RE.search(unit_cell) or header_has_mw):
                    append(out, "TOTALDEMAND", val, ts_use); any_row = True
                elif metric_u == "RRP" and (RRP_UNIT_RE.search(unit_cell) or header_has_rrp_unit):
                    append(out, "RRP", val, ts_use); any_row = True

                k += 1
                continue

            # ---------- Schema A/C: Time | TD | RRP ----------
            if idx_time is None or idx_time >= len(cells):
                k += 1
                continue

            ts_use = normalize_timestamp(cells[idx_time])
            if not ts_use:
                k += 1
                continue

            # TD column
            if idx_td is not None and idx_td < len(cells):
                td_cell = cells[idx_td]
                td_val = strip_num(td_cell)
                td_ok = (td_val is not None) and (MW_RE.search(td_cell) or header_has_mw)
                # if header has MW, allow pure numeric cells
                if td_val is not None and (td_ok or header_has_mw):
                    append(out, "TOTALDEMAND", td_val, ts_use); any_row = True

            # RRP column
            if idx_rrp is not None and idx_rrp < len(cells):
                rrp_cell = cells[idx_rrp]
                rrp_val = strip_num(rrp_cell)
                rrp_ok = (rrp_val is not None) and (RRP_UNIT_RE.search(rrp_cell) or header_has_rrp_unit)
                # if header has $/MWh, allow pure numeric cells
                if rrp_val is not None and (rrp_ok or header_has_rrp_unit):
                    append(out, "RRP", rrp_val, ts_use); any_row = True

            k += 1

        if any_row:
            i = k
        else:
            i += 1

    return out

# ============================================================
# NON-TABLE EXTRACTION (strict: must include unit OR keyword+unit context)
# ============================================================
def extract_non_table(text: str, row_ts_norm: str):
    out = {}
    if not text:
        return out

    # 1) Bullet/inline with timestamp + value + unit (preferred)
    for ln in text.splitlines():
        ts = normalize_timestamp(ln)
        if not ts:
            continue

        ln_wo_ts = remove_timestamp_from_line(ln)
        nums = [to_float(n) for n in NUM_RE.findall(ln_wo_ts)]
        nums = [n for n in nums if n is not None]
        if not nums:
            continue

        has_td_kw = bool(TD_KW_RE.search(ln))
        has_rrp_kw = bool(RRP_KW_RE.search(ln))
        has_mw = bool(MW_RE.search(ln))
        has_rrp_unit = bool(RRP_UNIT_RE.search(ln))

        # strict mapping
        if has_mw and (has_td_kw or (not has_rrp_kw)):
            append(out, "TOTALDEMAND", nums[0], ts)
        if has_rrp_unit and (has_rrp_kw or (not has_td_kw)):
            append(out, "RRP", nums[0], ts)

    # 2) Sentences: "... on <timestamp> ... is <value> MW" and "... is <value> $/MWh"
    # TOTALDEMAND
    for m in re.finditer(
        r"(?is)\bTOTALDEMAND\b.*?(" + TS_RE.pattern + r").*?(" + NUM_RE.pattern + r")\s*MW\b",
        text,
    ):
        ts = normalize_timestamp(m.group(1))
        val = to_float(m.group(m.lastindex - 0))  # last capture is numeric
        append(out, "TOTALDEMAND", val, ts)

    # RRP
    for m in re.finditer(
        r"(?is)\bRRP\b.*?(" + TS_RE.pattern + r").*?(" + NUM_RE.pattern + r")\s*\$/\s*MWh",
        text,
    ):
        ts = normalize_timestamp(m.group(1))
        val = to_float(m.group(m.lastindex - 0))
        append(out, "RRP", val, ts)

    # 3) Single-point fallback: value + unit but no explicit timestamp -> use row timestamp
    # TOTALDEMAND
    if row_ts_norm and "TOTALDEMAND" not in out:
        m = re.search(r"(?i)\bTOTALDEMAND\b.*?(" + NUM_RE.pattern + r")\s*MW\b", text)
        if m:
            append(out, "TOTALDEMAND", to_float(m.group(1)), row_ts_norm)

    # RRP
    if row_ts_norm and "RRP" not in out:
        m = re.search(r"(?i)\bRRP\b.*?(" + NUM_RE.pattern + r")\s*\$/\s*MWh", text)
        if m:
            append(out, "RRP", to_float(m.group(1)), row_ts_norm)

    return out

# ============================================================
# CORE
# ============================================================
def extract_predicted(answer_text: str, row_timestamp: str):
    text = answer_text or ""
    row_ts_norm = normalize_timestamp(row_timestamp or "")

    # Prefer tables for multi-step
    table_out = extract_from_markdown_tables(text)
    table_out = {k: v for k, v in table_out.items() if k in ("TOTALDEMAND", "RRP") and v}
    if table_out:
        return dedupe_sort(table_out)

    # Otherwise strict non-table parsing
    out = extract_non_table(text, row_ts_norm)
    out = {k: v for k, v in out.items() if k in ("TOTALDEMAND", "RRP") and v}
    return dedupe_sort(out) if out else {}

# ============================================================
# RUN
# ============================================================
def main():
    df = pd.read_csv(INPUT_CSV)

    if ANSWER_COL not in df.columns:
        raise KeyError(f"Missing '{ANSWER_COL}' column.")
    if ROW_TS_COL not in df.columns:
        raise KeyError(f"Missing '{ROW_TS_COL}' column.")

    df[OUT_COL] = [
        extract_predicted(a, t)
        for a, t in zip(df[ANSWER_COL].astype(str), df[ROW_TS_COL].astype(str))
    ]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

import re
import pandas as pd
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
INPUT_CSV  = "Gemini_parser.csv"
OUTPUT_CSV = "Gemini_eval.csv"

ANSWER_COL = "answer"
ROW_TS_COL = "timestamp"
OUT_COL    = "predicted"

# ============================================================
# REGEX
# ============================================================
TS_RE = re.compile(
    r"(?<!\d)"
    r"(20\d{2})[/-](\d{1,2})[/-](\d{1,2})"
    r"(?:[ T])"
    r"(\d{1,2}):(\d{2})"
    r"(?::(\d{2}))?"
    r"(?!\d)"
)

NUM_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")

MW_RE        = re.compile(r"\bMW\b", re.IGNORECASE)
RRP_UNIT_RE  = re.compile(r"\$/\s*MWh|\$/MWh", re.IGNORECASE)

TD_KW_RE  = re.compile(r"\bTOTALDEMAND\b", re.IGNORECASE)
RRP_KW_RE = re.compile(r"\bRRP\b", re.IGNORECASE)

# Strict "value + unit" (prevents grabbing timestamp HH/MM like 23 or 30)
TD_VAL_UNIT_RE  = re.compile(r"(" + NUM_RE.pattern + r")\s*MW\b", re.IGNORECASE)
RRP_VAL_UNIT_RE = re.compile(r"(" + NUM_RE.pattern + r")\s*\$/\s*MWh", re.IGNORECASE)

# "TOTALDEMAND ... <num> MW"
TD_LABELED_RE = re.compile(r"(?i)\bTOTALDEMAND\b[^0-9\-+]{0,80}(" + NUM_RE.pattern + r")\s*MW\b")
# "RRP ... <num> $/MWh"
RRP_LABELED_RE = re.compile(r"(?i)\bRRP\b[^0-9\-+]{0,80}(" + NUM_RE.pattern + r")\s*\$/\s*MWh")

# ============================================================
# HELPERS
# ============================================================
def to_float(s: str):
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None

def normalize_timestamp(s: str):
    if not s:
        return None
    m = TS_RE.search(s)
    if not m:
        return None
    y, mo, d, hh, mm, ss = m.groups()
    ss = ss or "00"
    try:
        dt = datetime(int(y), int(mo), int(d), int(hh), int(mm), int(ss))
    except Exception:
        return None
    return dt.strftime("%Y/%m/%d %H:%M:%S")

def append(out: dict, key: str, val, ts: str):
    if val is None or ts is None:
        return
    out.setdefault(key, []).append([val, ts])

def dedupe_sort(out: dict):
    for k, pairs in list(out.items()):
        seen = set()
        cleaned = []
        for v, ts in pairs:
            sig = (k, v, ts)
            if sig not in seen:
                seen.add(sig)
                cleaned.append([v, ts])
        try:
            cleaned.sort(key=lambda p: datetime.strptime(p[1], "%Y/%m/%d %H:%M:%S"))
        except Exception:
            pass
        out[k] = cleaned
    return out

def strip_num(cell: str):
    if not cell:
        return None
    m = NUM_RE.search(cell)
    return to_float(m.group(0)) if m else None

def is_alignment_row(cells):
    def is_align(c):
        c = (c or "").strip().replace(" ", "")
        return bool(re.fullmatch(r":?-{2,}:?", c))
    return len(cells) >= 2 and all(is_align(c) for c in cells)

def norm_header(s: str):
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def remove_timestamp_substring(line: str):
    m = TS_RE.search(line or "")
    if not m:
        return line or ""
    return (line[:m.start()] + " " + line[m.end():]).strip()

# ============================================================
# TABLE EXTRACTION
# ============================================================
def extract_from_markdown_tables(text: str):
    if not text or "|" not in text:
        return {}

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out = {}
    i = 0

    while i < len(lines):
        if "|" not in lines[i]:
            i += 1
            continue

        # header parse (preserve empties carefully later)
        header_cells = [c.strip() for c in lines[i].split("|")]
        if header_cells and header_cells[0] == "":
            header_cells = header_cells[1:]
        if header_cells and header_cells[-1] == "":
            header_cells = header_cells[:-1]
        header_cells_clean = [c for c in header_cells if c.strip()]

        if len(header_cells_clean) < 2:
            i += 1
            continue

        header_norm = [norm_header(c) for c in header_cells_clean]
        header_join = " ".join(header_norm)

        looks_like_table = ("time" in header_join) and (
            ("totaldemand" in header_join and "rrp" in header_join)
            or ("metric" in header_join and "forecast" in header_join)
            or ("forecast" in header_join and ("totaldemand" in header_join or "rrp" in header_join))
        )
        if not looks_like_table:
            i += 1
            continue

        header_has_mw = ("mw" in header_join)
        header_has_rrp_unit = ("/mwh" in header_join) or ("$/mwh" in header_join) or ("$/ mwh" in header_join)

        idx_time = next((k for k, h in enumerate(header_norm) if h.startswith("time")), None)
        idx_metric = next((k for k, h in enumerate(header_norm) if h == "metric"), None)
        idx_forecast = next((k for k, h in enumerate(header_norm) if h == "forecast"), None)
        idx_unit = next((k for k, h in enumerate(header_norm) if h == "unit"), None)

        idx_td = next((k for k, h in enumerate(header_norm) if "totaldemand" in h), None)
        idx_rrp = next((k for k, h in enumerate(header_norm) if h == "rrp" or h.startswith("rrp ")), None)

        # alignment row?
        j = i + 1
        has_align = False
        if j < len(lines) and "|" in lines[j]:
            tmp = [c.strip() for c in lines[j].split("|")]
            if tmp and tmp[0] == "":
                tmp = tmp[1:]
            if tmp and tmp[-1] == "":
                tmp = tmp[:-1]
            tmp_clean = [c for c in tmp if c.strip()]
            has_align = is_alignment_row(tmp_clean)

        k = i + 1 + (1 if has_align else 0)
        last_ts = None
        any_row = False

        while k < len(lines) and "|" in lines[k]:
            raw = [p.strip() for p in lines[k].split("|")]
            if raw and raw[0] == "":
                raw = raw[1:]
            if raw and raw[-1] == "":
                raw = raw[:-1]
            cells = raw  # keep empties

            if not cells:
                k += 1
                continue
            if is_alignment_row([c for c in cells if c.strip()]):
                k += 1
                continue

            # Schema B
            if idx_metric is not None and idx_forecast is not None:
                raw_time = cells[idx_time] if (idx_time is not None and idx_time < len(cells)) else ""
                ts = normalize_timestamp(raw_time) if raw_time else None
                if ts:
                    last_ts = ts
                ts_use = ts or last_ts
                if not ts_use:
                    k += 1
                    continue

                metric_cell = cells[idx_metric] if idx_metric < len(cells) else ""
                forecast_cell = cells[idx_forecast] if idx_forecast < len(cells) else ""
                unit_cell = cells[idx_unit] if (idx_unit is not None and idx_unit < len(cells)) else ""

                val = strip_num(forecast_cell)
                if val is None:
                    k += 1
                    continue

                metric_u = (metric_cell or "").strip().upper()
                if metric_u == "TOTALDEMAND" and (MW_RE.search(unit_cell) or header_has_mw):
                    append(out, "TOTALDEMAND", val, ts_use); any_row = True
                elif metric_u == "RRP" and (RRP_UNIT_RE.search(unit_cell) or header_has_rrp_unit):
                    append(out, "RRP", val, ts_use); any_row = True

                k += 1
                continue

            # Schema A/C
            if idx_time is None or idx_time >= len(cells):
                k += 1
                continue

            ts = normalize_timestamp(cells[idx_time])
            if not ts:
                k += 1
                continue

            if idx_td is not None and idx_td < len(cells):
                td_cell = cells[idx_td]
                td_val = strip_num(td_cell)
                if td_val is not None and (MW_RE.search(td_cell) or header_has_mw):
                    append(out, "TOTALDEMAND", td_val, ts); any_row = True

            if idx_rrp is not None and idx_rrp < len(cells):
                rrp_cell = cells[idx_rrp]
                rrp_val = strip_num(rrp_cell)
                if rrp_val is not None and (RRP_UNIT_RE.search(rrp_cell) or header_has_rrp_unit):
                    append(out, "RRP", rrp_val, ts); any_row = True

            k += 1

        if any_row:
            i = k
        else:
            i += 1

    return out

# ============================================================
# NON-TABLE EXTRACTION (FIXED FOR YOUR FAILING CASES)
# ============================================================
def extract_non_table(text: str, row_ts_norm: str):
    out = {}
    if not text:
        return out

    # 1) For lines that contain a timestamp, ONLY take numbers that are immediately followed by MW or $/MWh
    for ln in text.splitlines():
        ts = normalize_timestamp(ln)
        if not ts:
            continue

        ln_wo_ts = remove_timestamp_substring(ln)

        # TOTALDEMAND (must have MW)
        for m in TD_VAL_UNIT_RE.finditer(ln_wo_ts):
            val = to_float(m.group(1))
            # if line has RRP only (no TOTALDEMAND keyword and contains $/MWh), do not mis-assign
            append(out, "TOTALDEMAND", val, ts)

        # RRP (must have $/MWh)
        for m in RRP_VAL_UNIT_RE.finditer(ln_wo_ts):
            val = to_float(m.group(1))
            append(out, "RRP", val, ts)

        # Also allow "TOTALDEMAND ... <num> MW" / "RRP ... <num> $/MWh" on same line
        mtd = TD_LABELED_RE.search(ln)
        if mtd:
            append(out, "TOTALDEMAND", to_float(mtd.group(1)), ts)

        mrrp = RRP_LABELED_RE.search(ln)
        if mrrp:
            append(out, "RRP", to_float(mrrp.group(1)), ts)

    # 2) Sentence with both metrics (your Queensland example)
    #    Require units to avoid cross-contamination.
    for m in re.finditer(
        r"(?is)(" + TS_RE.pattern + r").{0,120}?\bTOTALDEMAND\b.{0,60}?(" + NUM_RE.pattern + r")\s*MW\b.{0,160}?\bRRP\b.{0,60}?(" + NUM_RE.pattern + r")\s*\$/\s*MWh",
        text,
    ):
        ts = normalize_timestamp(m.group(1))
        td_val = to_float(m.group(m.lastindex - 1))  # TOTALDEMAND number
        rrp_val = to_float(m.group(m.lastindex))     # RRP number
        append(out, "TOTALDEMAND", td_val, ts)
        append(out, "RRP", rrp_val, ts)

    # 3) Single-point fallback (no explicit timestamp in answer): use row timestamp,
    #    but ONLY if value includes the correct unit.
    if row_ts_norm:
        if "TOTALDEMAND" not in out:
            m = TD_VAL_UNIT_RE.search(text)
            if m:
                append(out, "TOTALDEMAND", to_float(m.group(1)), row_ts_norm)

        if "RRP" not in out:
            m = RRP_VAL_UNIT_RE.search(text)
            if m:
                append(out, "RRP", to_float(m.group(1)), row_ts_norm)

        # labeled fallback
        if "TOTALDEMAND" not in out:
            m = TD_LABELED_RE.search(text)
            if m:
                append(out, "TOTALDEMAND", to_float(m.group(1)), row_ts_norm)

        if "RRP" not in out:
            m = RRP_LABELED_RE.search(text)
            if m:
                append(out, "RRP", to_float(m.group(1)), row_ts_norm)

    return out

# ============================================================
# CORE
# ============================================================
def extract_predicted(answer_text: str, row_timestamp: str):
    text = answer_text or ""
    row_ts_norm = normalize_timestamp(row_timestamp or "")

    # Prefer tables
    table_out = extract_from_markdown_tables(text)
    table_out = {k: v for k, v in table_out.items() if k in ("TOTALDEMAND", "RRP") and v}
    if table_out:
        return dedupe_sort(table_out)

    out = extract_non_table(text, row_ts_norm)
    out = {k: v for k, v in out.items() if k in ("TOTALDEMAND", "RRP") and v}
    return dedupe_sort(out) if out else {}

# ============================================================
# RUN
# ============================================================
def main():
    df = pd.read_csv(INPUT_CSV)

    if ANSWER_COL not in df.columns:
        raise KeyError(f"Missing '{ANSWER_COL}' column.")
    if ROW_TS_COL not in df.columns:
        raise KeyError(f"Missing '{ROW_TS_COL}' column.")

    df[OUT_COL] = [
        extract_predicted(a, t)
        for a, t in zip(df[ANSWER_COL].astype(str), df[ROW_TS_COL].astype(str))
    ]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()