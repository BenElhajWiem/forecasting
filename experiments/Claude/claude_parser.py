import re
import json
from typing import Any, Dict, List, Optional
import pandas as pd
from dateutil import parser as dtparser

# ============================================================
# Regex / config
# ============================================================

TZ_TOKENS_RE = re.compile(r"\b(AEST|AEDT|ACST|ACDT|AWST|UTC)\b", flags=re.IGNORECASE)

HDR_RE = re.compile(r"^\s*(?:-{3,}\s*)?(?P<level>#{1,6})\s+(?P<title>.+?)\s*$", re.MULTILINE)
RATIONALE_ANY_HDR_RE = re.compile(r"^\s*(?:-{3,}\s*)?#{1,6}\s*Rationale\b.*$", re.IGNORECASE | re.MULTILINE)

TS_ISO_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?\b")
TS_SLASH_RE = re.compile(r"\b\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}(?::\d{2})?\b")

TABLE_ROW_RE = re.compile(r"^\|\s*(.*?)\s*\|\s*(.*?)\s*\|(?:\s*(.*?)\s*\|)?\s*$", re.MULTILINE)

INLINE_ROW_START_RE = re.compile(
    r"(?<=\s)\|\s*(?=(?:Time\b|------|20\d{2}-\d{2}-\d{2}|\d{4}/\d{2}/\d{2}))",
    flags=re.IGNORECASE
)

# ============================================================
# Helpers
# ============================================================

def strip_md(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("`", "")
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
    s = re.sub(r"\*(.*?)\*", r"\1", s)
    return s.strip()

def norm_ts(ts: Any) -> Optional[str]:
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    s = strip_md(str(ts))
    if not s:
        return None

    su = s.upper()
    if "TOTALDEMAND" in su or "RRP" in su:
        return None

    s = TZ_TOKENS_RE.sub("", s).strip()

    m = TS_SLASH_RE.search(s) or TS_ISO_RE.search(s)
    if m:
        try:
            dt = dtparser.parse(m.group(0), fuzzy=True)
            return dt.strftime("%Y/%m/%d %H:%M:%S")
        except Exception:
            return None

    try:
        dt = dtparser.parse(s, fuzzy=True)
        return dt.strftime("%Y/%m/%d %H:%M:%S")
    except Exception:
        return None

def to_float_num(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = strip_md(str(x))
    if not s:
        return None
    s = s.replace("$", "")
    s = re.sub(r"(?i)\b(MW|MWh|/MWh|\$/MWh)\b", "", s)
    s = s.replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def dedupe_pairs(out: Dict[str, List[List[Any]]]) -> Dict[str, List[List[Any]]]:
    for k in list(out.keys()):
        seen = set()
        uniq = []
        for v, ts in out[k]:
            t = (v, ts)
            if t not in seen:
                uniq.append([v, ts])
                seen.add(t)
        out[k] = uniq
        if not out[k]:
            out.pop(k, None)
    return out

def extract_ts_from_text(s: str) -> Optional[str]:
    if not s:
        return None
    s2 = TZ_TOKENS_RE.sub("", s).strip()
    m = TS_SLASH_RE.search(s2) or TS_ISO_RE.search(s2)
    if m:
        return norm_ts(m.group(0))
    try:
        return norm_ts(dtparser.parse(s2, fuzzy=True))
    except Exception:
        return None

# ============================================================
# Normalization (inline --- ## Forecast | ... | ... cases)
# ============================================================

def normalize_answer_text(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"\s*---\s*", "\n---\n", text)
    t = re.sub(r"(##\s*Forecast)\s*\|", r"\1\n|", t, flags=re.IGNORECASE)
    t = INLINE_ROW_START_RE.sub("\n| ", t)
    return t

def cut_before_rationale(text: str) -> str:
    m = RATIONALE_ANY_HDR_RE.search(text)
    if m:
        return text[:m.start()]
    return text

# ============================================================
# Forecast-ish detection (includes your markers)
# ============================================================

def is_forecastish_title(title: str) -> bool:
    t = strip_md(title).lower()
    if "rationale" in t:
        return False
    if "forecast" in t:
        return True
    extras = ["i forecast", "**forecast"]
    return any(x in t for x in extras)

def extract_forecast_text(answer: str) -> str:
    if not answer:
        return ""

    norm = normalize_answer_text(answer)
    pre = cut_before_rationale(norm)

    matches = list(HDR_RE.finditer(pre))
    blocks: List[str] = []

    for i, m in enumerate(matches):
        title = m.group("title")
        if not is_forecastish_title(title):
            continue
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(pre)
        blocks.append(pre[start:end])

    if re.search(r"(?i)\bforecast\s*:|\bi\s+forecast\b|\*\*forecast", pre):
        blocks.append(pre)

    seen = set()
    uniq = []
    for b in blocks:
        k = b.strip()
        if k and k not in seen:
            uniq.append(b)
            seen.add(k)

    return "\n\n".join(uniq).strip()

def default_ts_from_text(forecast_text: str, row_ts: Optional[str]) -> Optional[str]:
    if not forecast_text:
        return row_ts
    s = TZ_TOKENS_RE.sub("", forecast_text)
    m = TS_SLASH_RE.search(s) or TS_ISO_RE.search(s)
    if m:
        return norm_ts(m.group(0)) or row_ts
    m2 = re.search(r"(?i)\bTime\b\s*:\s*([^\n\r]+)", forecast_text)
    if m2:
        return norm_ts(m2.group(1)) or row_ts
    return row_ts

# ============================================================
# Parsing
# ============================================================

def parse_tables(forecast_text: str, row_ts: Optional[str]) -> Dict[str, List[List[Any]]]:
    out: Dict[str, List[List[Any]]] = {}
    if not forecast_text:
        return out

    default_ts = default_ts_from_text(forecast_text, row_ts)

    rows: List[List[str]] = []
    for m in TABLE_ROW_RE.finditer(forecast_text):
        c1 = strip_md(m.group(1))
        c2 = strip_md(m.group(2))
        c3 = strip_md(m.group(3)) if m.group(3) is not None else None
        rows.append([c1, c2] + ([c3] if c3 is not None else []))

    if not rows:
        return out

    ts_like = sum(1 for r in rows[:15] if len(r) >= 3 and norm_ts(r[0]) is not None)
    time_series_mode = ts_like >= 1

    for r in rows:
        if all((not c) or (set(c) <= set("-: ")) for c in r):
            continue

        if time_series_mode and len(r) >= 3:
            ts = norm_ts(r[0])
            if ts is not None:
                td = to_float_num(r[1])
                rrp = to_float_num(r[2])
                if td is not None:
                    out.setdefault("TOTALDEMAND", []).append([td, ts])
                if rrp is not None:
                    out.setdefault("RRP", []).append([rrp, ts])
                continue

        k = r[0].upper()
        if "TOTALDEMAND" in k:
            v = to_float_num(r[1]) if len(r) >= 2 else None
            if v is not None:
                out.setdefault("TOTALDEMAND", []).append([v, default_ts])
            continue

        if "RRP" in k:
            v = to_float_num(r[1]) if len(r) >= 2 else None
            if v is not None:
                out.setdefault("RRP", []).append([v, default_ts])
            continue

    return dedupe_pairs(out)

def parse_lines(forecast_text: str, row_ts: Optional[str]) -> Dict[str, List[List[Any]]]:
    out: Dict[str, List[List[Any]]] = {}
    if not forecast_text:
        return out

    default_ts = default_ts_from_text(forecast_text, row_ts)

    # (A) Forecast: 6,950 MW
    for m in re.finditer(r"(?i)\bforecast\s*:\s*([0-9][0-9,]*(?:\.\d+)?)\s*MW\b", forecast_text):
        v = to_float_num(m.group(1))
        if v is not None:
            out.setdefault("TOTALDEMAND", []).append([v, default_ts])

    # (B) Time: ... TOTALDEMAND: ... RRP: ...
    m_time = re.search(r"(?i)\bTime\b\s*:\s*([^\n\r*]+)", forecast_text)
    local_ts = norm_ts(m_time.group(1)) if m_time else default_ts

    for m in re.finditer(r"(?i)\bTOTALDEMAND\b\s*:\s*\**\s*([0-9][0-9,]*(?:\.\d+)?)\s*MW", forecast_text):
        v = to_float_num(m.group(1))
        if v is not None:
            out.setdefault("TOTALDEMAND", []).append([v, local_ts])

    for m in re.finditer(r"(?i)\bRRP\b\s*:\s*\**\s*\$?\s*([0-9][0-9,]*(?:\.\d+)?)", forecast_text):
        v = to_float_num(m.group(1))
        if v is not None:
            out.setdefault("RRP", []).append([v, local_ts])

    # ========================================================
    # MISSING CASES FIX (your TAS1_011 / NSW1_001 examples)
    # ========================================================

    # (C) **<date/time>: <value> MW**
    for m in re.finditer(
        r"\*\*\s*([^*\n]{0,260}?)\s*:\s*([0-9][0-9,]*(?:\.\d+)?)\s*MW\s*\*\*",
        forecast_text,
        flags=re.IGNORECASE,
    ):
        left = strip_md(m.group(1))
        v = to_float_num(m.group(2))
        ts = extract_ts_from_text(left) or default_ts
        if v is not None:
            out.setdefault("TOTALDEMAND", []).append([v, ts])

    # (D) **<date/time> → <value> MW**
    for m in re.finditer(
        r"\*\*\s*([^*\n]{0,260}?)(?:→|->)\s*([0-9][0-9,]*(?:\.\d+)?)\s*MW\s*\*\*",
        forecast_text,
        flags=re.IGNORECASE,
    ):
        left = strip_md(m.group(1))
        v = to_float_num(m.group(2))
        ts = extract_ts_from_text(left) or default_ts
        if v is not None:
            out.setdefault("TOTALDEMAND", []).append([v, ts])

    # (E) Non-bold: <date/time>: <value> MW
    for m in re.finditer(
        r"(?i)\b(.{0,200}?)\b(?:\d{4}-\d{2}-\d{2}|\d{4}/\d{2}/\d{2}).{0,60}?\b:\s*([0-9][0-9,]*(?:\.\d+)?)\s*MW\b",
        forecast_text
    ):
        left = m.group(0)
        v = to_float_num(m.group(2))
        ts = extract_ts_from_text(left) or default_ts
        if v is not None:
            out.setdefault("TOTALDEMAND", []).append([v, ts])

    # (F) After "## Forecast" header, capture first bold MW in next lines
    lines = forecast_text.splitlines()
    for i, ln in enumerate(lines):
        if re.search(r"(?i)^\s*##\s*forecast\b", ln.strip()):
            seen = 0
            for nxt in lines[i+1:]:
                t = nxt.strip()
                if not t:
                    continue
                if re.match(r"^\s*#{1,6}\s+", t):
                    break
                seen += 1
                if seen > 12:
                    break
                m2 = re.search(r"\*\*\s*([0-9][0-9,]*(?:\.\d+)?)\s*MW\s*\*\*", t, flags=re.IGNORECASE)
                if m2:
                    v2 = to_float_num(m2.group(1))
                    if v2 is not None:
                        out.setdefault("TOTALDEMAND", []).append([v2, default_ts])
                    break

    return dedupe_pairs(out)

# ============================================================
# Build predicted as list-of-dicts
# ============================================================

def build_predicted(answer: Any, timestamp: Any) -> List[Dict[str, List[List[Any]]]]:
    text = "" if pd.isna(answer) else str(answer)
    row_ts = norm_ts(timestamp)

    forecast_text = extract_forecast_text(text)
    if not forecast_text:
        return []

    out: Dict[str, List[List[Any]]] = {}
    for k, v in parse_tables(forecast_text, row_ts).items():
        out.setdefault(k, []).extend(v)
    for k, v in parse_lines(forecast_text, row_ts).items():
        out.setdefault(k, []).extend(v)

    out = dedupe_pairs(out)

    result: List[Dict[str, List[List[Any]]]] = []
    if out.get("TOTALDEMAND"):
        result.append({"TOTALDEMAND": out["TOTALDEMAND"]})
    if out.get("RRP"):
        result.append({"RRP": out["RRP"]})
    return result

# ============================================================
# Main
# ============================================================

def main(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)

    if "answer" not in df.columns:
        raise ValueError("Input must contain an 'answer' column.")
    if "timestamp" not in df.columns:
        raise ValueError("Input must contain a 'timestamp' column.")

    df["predicted"] = df.apply(
        lambda r: json.dumps(build_predicted(r["answer"], r["timestamp"]), ensure_ascii=False),
        axis=1
    )

    non_empty = (df["predicted"] != "[]").sum()
    both = df["predicted"].str.contains("TOTALDEMAND") & df["predicted"].str.contains("RRP")

    df.to_csv(output_path, index=False)
    print("Saved:", output_path)
    print("Non-empty predicted rows:", non_empty, "out of", len(df))
    print("Rows with BOTH metrics:", int(both.sum()))
    if non_empty:
        print("Example predicted:", df.loc[df["predicted"] != "[]", "predicted"].iloc[0])

if __name__ == "__main__":
    INPUT = "claude_evaluation_all.csv"
    OUTPUT = "Claude_eval.csv"
    main(INPUT, OUTPUT)