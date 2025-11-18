from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json
import pandas as pd

# -------------------------
# Config
# -------------------------
@dataclass
class ForecastConfig:
    model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    tz: str = "Australia/Sydney"
    model_override: Optional[str] = None
    enforce_unique_timestamps: bool = True  # kept for future use
    units_map: Dict[str, str] = field(default_factory=lambda: {
        "TOTALDEMAND": "MW",
        "RRP": "$/MWh",
    })

# -------------------------
# Helpers
# -------------------------
def _cap_text(obj: Any, max_chars: int = 15000) -> str:
    """Stringify and cap long evidence blocks to keep prompts compact."""
    try:
        s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    if len(s) > max_chars:
        keep = max_chars - 200
        return s[:keep] + f"\n...[truncated {len(s)-keep} chars]..."
    return s


def _tzify(ts_like: Any, tz: str) -> Optional[pd.Timestamp]:
    """Parse to Timestamp and ensure timezone; returns None on failure/empty."""
    if ts_like in (None, "", "null"):
        return None
    try:
        t = pd.Timestamp(ts_like)
        return t.tz_convert(tz) if t.tzinfo else t.tz_localize(tz)
    except Exception:
        return None

def _derive_targets_from_filters(filters: Dict[str, Any], tz: str) -> Dict[str, Any]:
    """Normalize regions/metrics/freq and build a readable window label."""
    regs = [str(r).upper() for r in (filters.get("regions") or [])] or []
    mets = [str(m).upper() for m in (filters.get("metrics") or [])] or []
    freq = (filters.get("freq") or "").strip() or None

    start = _tzify(filters.get("start"), tz)
    end   = _tzify(filters.get("end"), tz)

    # Allow (date_start,date_end) + times as fallback
    if start is None and end is None:
        ds = _tzify(filters.get("date_start"), tz)
        de = _tzify(filters.get("date_end"), tz)
        times = filters.get("times") or []
        t0 = None
        if times:
            tstr = str(times[0])
            if len(tstr) == 5:  # "HH:MM" -> add seconds
                tstr += ":00"
            try:
                hh, mm, ss = [int(x) for x in tstr.split(":")]
                t0 = (hh, mm, ss)
            except Exception:
                t0 = None
        if ds is not None:
            start = ds.replace(hour=t0[0], minute=t0[1], second=t0[2]) if t0 else ds
        if de is not None:
            end = de.replace(hour=t0[0], minute=t0[1], second=t0[2]) if t0 else de

    def _fmt(ts: Optional[pd.Timestamp]) -> Optional[str]:
        return ts.strftime("%Y-%m-%d %H:%M") if ts is not None else None

    if start and end:
        win = f"{_fmt(start)} → {_fmt(end)} ({freq or 'unspecified'})"
    elif start and not end:
        win = f"{_fmt(start)} (single anchor; {freq or 'unspecified'})"
    elif end and not start:
        win = f"up to {_fmt(end)} ({freq or 'unspecified'})"
    else:
        win = f"forward horizon at {freq or 'default cadence'} from the latest available point"

    return {"regions": regs, "metrics": mets, "freq": freq, "window_text": win}

def _render_units_lines(metrics: List[str], units_map: Dict[str, str]) -> str:
    """One unit per line for readability in the prompt."""
    if not metrics:
        metrics = ["TOTALDEMAND", "RRP"]
    return "\n".join([f"- {m}: {units_map.get(m, '')}" for m in metrics])

def _format_targets_header(tgt: Dict[str, Any], route: Optional[str]) -> str:
    regions = tgt["regions"] or ["UNKNOWN-REGION"]
    metrics = tgt["metrics"] or ["TOTALDEMAND", "RRP"]
    return (
        "TARGETS (from filters)\n"
        f"- Regions: {', '.join(regions)}\n"
        f"- Metrics: {', '.join(metrics)}\n"
        f"- Window:  {tgt['window_text']}\n"
        f"- Route hint: {route or 'unspecified'}\n"
        f"- Freq: {tgt['freq'] or 'unspecified'}\n"
        "- Context slices: recent_window, same_hour_previous_days, prior_years_same_dates/week, same_woy_prior_years"
    )

# -------------------------
# Forecaster (adapter)
# -------------------------
def forecast_with_llm(
    adapter,
    user_query: str,
    summary: Any,
    stats: Any,
    patterns: Any,
    filters: Dict[str, Any],
    cfg: ForecastConfig = ForecastConfig(),
    route: Optional[str] = None,
    prior_history = None,
    # Optional one-off overrides (rarely needed; keeps call-site simple)
    temperature: Optional[float] = 0.0,
    max_tokens: Optional[int] = None,
    model_override: Optional[str] = None,
) -> str:
    """
    Generate a clear, structured NATURAL-LANGUAGE forecast.
    """
    tgt = _derive_targets_from_filters(filters, cfg.tz)
    units_lines = _render_units_lines(tgt["metrics"], cfg.units_map)

    summary_block  = summary
    stats_block    = stats 
    patterns_block = patterns
    targets_header = _format_targets_header(tgt, route)

    system_msg = (
        "You are a precise time-series forecaster. "
        "Generate timestamped forecasts for each requested metric in NATURAL LANGUAGE. "
        "Be concise but specific. Prefer ~3 significant digits and include units. "
        "Respect requested frequency and window. "
        "Do NOT repeat the same numeric value for consecutive timestamps unless the evidence is identical; if identical, justify equality explicitly."
        "If multiple timestamps are requested, present them as short bullet points."
        "Do NOT fabricate new calculations; rely only on the provided evidence."
        "Base statements only on the provided evidence; do not invent new calculations. "
    )
    
    user_msg = f"""
USER QUERY: {user_query}

UNITS: {units_lines}

{targets_header}

PRIOR HISTORY (same dates prior years):
{prior_history}

EVIDENCE (use as-is; do not recalc):

- Statistical Insights:
{stats_block}

- Historical Summary:
{summary_block}

- Detected Patterns:
{patterns_block}

OUTPUT STYLE — NATURAL LANGUAGE ONLY
- Write a clear **narrative report**.
- Start with a short overview sentence (region(s), metric(s), window/frequency).
- Then give the forecasts:
- If multiple timestamps are requested:
    * Present them in a **Markdown table** with columns: Time | Forecast | Unit
    * Keep it concise and readable.
- If only one timestamp: a single sentence with the forecast.
Rules:
- Produce a single point UNLESS the user asked for multiple timestamps across the requested window/frequency.
- Use real values per timestamp unless you explicitly justify equality (e.g., flat night hours).
- Always include the requested metric(s) or default to both metrics ['TOTALDEMAND','RRP'] when unspecified.
"""
    messages = [{"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}]
    
    out = adapter.chat(
        messages,
        temperature=cfg.temperature if temperature is None else temperature,
        max_tokens=cfg.max_tokens if max_tokens is None else max_tokens,
        model_override=(cfg.model_override or cfg.model) if model_override is None else model_override,
    )
    return (out or "").strip()