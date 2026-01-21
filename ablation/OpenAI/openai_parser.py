import re
import pandas as pd
from datetime import datetime

INPUT_PATH = "OpenAI_parser.csv"
OUTPUT_PATH = "OpenAI_eval.csv"

# ============================
# TIMESTAMP NORMALIZATION
# ============================

def _normalize_timestamp(ts: str) -> str | None:
    if not isinstance(ts, str):
        return None
    ts = ts.strip()
    ts = re.sub(r"[)\],;.!]+$", "", ts)

    fmts = [
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]
    for f in fmts:
        try:
            return datetime.strptime(ts, f).strftime("%Y/%m/%d %H:%M:%S")
        except ValueError:
            pass
    return None

_MONTH_TS_RE = re.compile(
    r"\b(?P<mon>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+(?P<day>\d{1,2}),\s*(?P<year>\d{4})\s*,?\s*(?:at\s*)?(?P<hour>\d{1,2}):(?P<min>\d{2})\b",
    flags=re.IGNORECASE,
)

def _normalize_month_ts(m) -> str | None:
    try:
        mon = m.group("mon")
        fmt = "%b %d %Y %H:%M" if len(mon) <= 3 else "%B %d %Y %H:%M"
        dt = datetime.strptime(
            f"{mon} {int(m.group('day')):02d} {m.group('year')} {int(m.group('hour')):02d}:{m.group('min')}",
            fmt,
        )
        return dt.strftime("%Y/%m/%d %H:%M:%S")
    except Exception:
        return None

_NUMERIC_TS_RE = re.compile(
    r"\b(?P<ts>\d{4}[/-]\d{2}[/-]\d{2}\s+\d{2}:\d{2}(?::\d{2})?)\b"
)

def _find_any_timestamp(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    m = _NUMERIC_TS_RE.search(text)
    if m:
        ts = _normalize_timestamp(m.group("ts"))
        if ts:
            return ts
    m2 = _MONTH_TS_RE.search(text)
    if m2:
        ts = _normalize_month_ts(m2)
        if ts:
            return ts
    return None

# ============================
# NUMBER
# ============================

def _parse_number(s: str) -> float | None:
    if not isinstance(s, str):
        return None
    s = s.replace(",", "")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None

# ============================
# TABLE EXTRACTION (AUTHORITATIVE)
# ============================

_TABLE4_ROW_RE = re.compile(
    r"\|\s*(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)\s*"
    r"\|\s*(?P<td>[-+]?\d[\d,]*(?:\.\d+)?)\s*"
    r"\|\s*(?P<rrp>[-+]?\d[\d,]*(?:\.\d+)?)\s*"
    r"\|\s*(?P<unit>[^|]*?)\s*\|",
    flags=re.IGNORECASE,
)

_TABLE3_VAL_UNIT_RE = re.compile(
    r"\|\s*(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)\s*"
    r"\|\s*(?P<val>[-+]?\d[\d,]*(?:\.\d+)?)\s*"
    r"\|\s*(?P<unit>[^|]*?)\s*\|",
    flags=re.IGNORECASE,
)

def _unit_tag(unit: str) -> str:
    u = (unit or "").lower().replace(" ", "")
    if "mw" in u and "mwh" not in u:
        return "TD"
    if "mwh" in u and ("$" in u or "/mwh" in u):
        return "RRP"
    return ""

def _dedup_sort(lst: list[list]) -> list[list]:
    seen = set()
    out = []
    for v, t in lst:
        k = (v, t)
        if k in seen:
            continue
        seen.add(k)
        out.append([v, t])
    return sorted(out, key=lambda x: x[1])

def _extract_from_tables(answer: str) -> tuple[dict, bool]:
    """
    IMPORTANT CHANGE:
    - Parse BOTH table styles in a SINGLE pass and MERGE results.
    - Never overwrite one with the other.
    """
    td, rrp = [], []
    has_table = False

    # 4-col rows (TD and RRP on same row)
    for m in _TABLE4_ROW_RE.finditer(answer):
        unit = (m.group("unit") or "")
        u = unit.lower().replace(" ", "")
        if ("mw" in u) and ("mwh" in u):  # MW / $/MWh
            ts = _normalize_timestamp(m.group("ts"))
            if not ts:
                continue
            vtd = _parse_number(m.group("td"))
            vrrp = _parse_number(m.group("rrp"))
            if vtd is not None:
                td.append([float(vtd), ts])
            if vrrp is not None:
                rrp.append([float(vrrp), ts])
            has_table = True

    # 3-col rows (TD block and/or RRP block)
    for m in _TABLE3_VAL_UNIT_RE.finditer(answer):
        unit = m.group("unit") or ""
        tag = _unit_tag(unit)
        if not tag:
            continue
        ts = _normalize_timestamp(m.group("ts"))
        if not ts:
            continue
        v = _parse_number(m.group("val"))
        if v is None:
            continue
        if tag == "TD":
            td.append([float(v), ts])
            has_table = True
        elif tag == "RRP":
            rrp.append([float(v), ts])
            has_table = True

    out = {}
    td = _dedup_sort(td)
    rrp = _dedup_sort(rrp)
    if td:
        out["TOTALDEMAND"] = td
    if rrp:
        out["RRP"] = rrp
    return out, has_table

# ============================
# NON-TABLE EXTRACTION (STRICT UNIT-REQUIRED)
# ============================

_TD_SPAN_RE = re.compile(
    r"\b(?P<val>[-+]?\d[\d,]*(?:\.\d+)?)\s*(?:\|\s*)?(?:MW|M\s*W)\b",
    flags=re.IGNORECASE,
)

_RRP_SPAN_RE = re.compile(
    r"(?:\$\s*)?(?P<val>[-+]?\d[\d,]*(?:\.\d+)?)\s*(?:\|\s*)?(?:\$\s*)?(?:/|\s*)\s*(?:MWh|M\s*W\s*h)\b",
    flags=re.IGNORECASE,
)

def _extract_strict_units(answer: str, row_ts_norm: str | None) -> dict:
    td, rrp = [], []
    ts = _find_any_timestamp(answer) or row_ts_norm
    if not ts:
        return {}

    for m in _TD_SPAN_RE.finditer(answer):
        v = _parse_number(m.group("val"))
        if v is not None:
            td.append([float(v), ts])

    for m in _RRP_SPAN_RE.finditer(answer):
        span_start, span_end = m.span()
        window = answer[max(0, span_start - 8): min(len(answer), span_end + 8)]
        if "$" not in window and "$" not in m.group(0):
            continue
        v = _parse_number(m.group("val"))
        if v is not None:
            rrp.append([float(v), ts])

    out = {}
    td = _dedup_sort(td)
    rrp = _dedup_sort(rrp)
    if td:
        out["TOTALDEMAND"] = td
    if rrp:
        out["RRP"] = rrp
    return out

# ============================
# HOURLY vs SINGLE
# ============================

def _force_single(out: dict, row_ts_norm: str | None) -> dict:
    if not row_ts_norm:
        return out
    single = {}
    if "TOTALDEMAND" in out and out["TOTALDEMAND"]:
        single["TOTALDEMAND"] = [[out["TOTALDEMAND"][-1][0], row_ts_norm]]
    if "RRP" in out and out["RRP"]:
        single["RRP"] = [[out["RRP"][-1][0], row_ts_norm]]
    return single

# ============================
# MAIN PARSER
# ============================

def parse_answer_to_predicted(answer: str, row_timestamp: str | None, query_id: str | None) -> dict:
    if not isinstance(answer, str) or not answer.strip():
        return {}

    is_hourly = isinstance(query_id, str) and "hourly" in query_id.lower()
    row_ts_norm = _normalize_timestamp(row_timestamp) if row_timestamp else None

    # 1) TABLES OVERRIDE EVERYTHING (BUT NOW MERGED TD+RRP)
    table_out, has_table = _extract_from_tables(answer)
    if has_table:
        return table_out if is_hourly else _force_single(table_out, row_ts_norm)

    # 2) STRICT NON-TABLE extraction: unit-required spans only (TD+RRP merged)
    out = _extract_strict_units(answer, row_ts_norm)
    return out if is_hourly else _force_single(out, row_ts_norm)

# ============================
# RUN
# ============================

def main():
    df = pd.read_csv(INPUT_PATH)
    required = {"answer", "timestamp", "query_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["predicted"] = df.apply(
        lambda r: parse_answer_to_predicted(r["answer"], r["timestamp"], r["query_id"]),
        axis=1,
    )

    df.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()