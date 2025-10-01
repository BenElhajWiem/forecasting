from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import json
import pandas as pd

# -------------------------
# Config
# -------------------------
@dataclass
class ForecastConfig:
    model: str = None
    temperature: float = 0.2
    max_tokens: int = None
    tz: str = "Australia/Sydney"
    model_override: Optional[str] = None
    enforce_unique_timestamps: bool = True   # de-duplicate or nudge duplicates
    units_map: Dict[str, str] = None

    def __post_init__(self):
        if self.units_map is None:
            self.units_map = {"TOTALDEMAND": "MW", "RRP": "$/MWh"}

# -------------------------
# Helpers
# -------------------------
def _cap_text(obj: Any, max_chars: int = 15000) -> str:
    try:
        s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    if len(s) > max_chars:
        keep = max_chars - 200
        return s[:keep] + f"\n...[truncated {len(s)-keep} chars]..."
    return s

def _tzify(ts_like: Any, tz: str) -> Optional[pd.Timestamp]:
    if ts_like in (None, "", "null"):
        return None
    try:
        t = pd.Timestamp(ts_like)
        return t.tz_convert(tz) if t.tzinfo else t.tz_localize(tz)
    except Exception:
        return None

def _derive_targets_from_filters(filters: Dict[str, Any], tz: str) -> Dict[str, Any]:
    regs = [str(r).upper() for r in (filters.get("regions") or [])]
    mets = [str(m).upper() for m in (filters.get("metrics") or [])]
    freq = (filters.get("freq") or "").strip() or None

    start = _tzify(filters.get("start"), tz)
    end   = _tzify(filters.get("end"), tz)
    if start is None and end is None:
        ds = _tzify(filters.get("date_start"), tz)
        de = _tzify(filters.get("date_end"), tz)
        times = filters.get("times") or []
        t0 = None
        if times:
            tstr = str(times[0])
            if len(tstr) == 5: tstr += ":00"
            try:
                hh, mm, ss = [int(x) for x in tstr.split(":")]
                t0 = (hh, mm, ss)
            except Exception:
                t0 = None
        if ds is not None:
            start = ds.replace(hour=t0[0], minute=t0[1], second=t0[2]) if t0 else ds
        if de is not None:
            end = de.replace(hour=t0[0], minute=t0[1], second=t0[2]) if t0 else de

    if start and end:
        win = f"{start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')} ({freq or 'unspecified'})"
    elif start and not end:
        win = f"{start.strftime('%Y-%m-%d %H:%M')} (single anchor; {freq or 'unspecified'})"
    elif end and not start:
        win = f"up to {end.strftime('%Y-%m-%d %H:%M')} ({freq or 'unspecified'})"
    else:
        win = f"forward horizon at {freq or 'default cadence'} from the latest available point"

    return {"regions": regs, "metrics": mets, "freq": freq, "window_text": win}

def _format_targets_header(tgt: Dict[str, Any], route: Optional[str]) -> str:
    return (
        "TARGETS (from filters)\n"
        f"- Regions: {', '.join(tgt['regions'] or ['UNKNOWN-REGION'])}\n"
        f"- Metrics: {', '.join(tgt['metrics'] or ['TOTALDEMAND','RRP'])}\n"
        f"- Window:  {tgt['window_text']}\n"
        f"- Route hint: {route or 'unspecified'}\n"
        f"- Freq: {tgt['freq'] or 'unspecified'}\n"
        "- Context slices: recent_window, same_hour_previous_days, prior_years_same_dates/week, same_woy_prior_years"
    )

def _nudge_duplicate_timestamps(obj: Dict[str, Any]) -> Dict[str, Any]:
    """If model repeats exact timestamps, slightly nudge seconds to ensure uniqueness."""
    for mkey, arr in (obj.get("metrics") or {}).items():
        seen = {}
        new_arr = []
        for item in arr:
            ts = item.get("timestamp")
            if not ts:
                new_arr.append(item); continue
            if ts not in seen:
                seen[ts] = 0
                new_arr.append(item)
            else:
                seen[ts] += 1
                try:
                    t = pd.Timestamp(ts)
                    t = t + pd.Timedelta(seconds=seen[ts])  # +1s, +2s, ...
                    item2 = dict(item); item2["timestamp"] = t.isoformat()
                    new_arr.append(item2)
                except Exception:
                    # fallback: append suffix
                    item2 = dict(item); item2["timestamp"] = f"{ts} (+{seen[ts]}s)"
                    new_arr.append(item2)
        obj["metrics"][mkey] = new_arr
    return obj

# -------------------------
# Forecaster (adapter)
# -------------------------
def forecast_narrative_from_filters(
    adapter,
    *,
    user_query: str,
    summary: Any,
    stats: Any,
    patterns: Any,
    filters: Dict[str, Any],
    route: Optional[str] = None,
    cfg: ForecastConfig = ForecastConfig(),
) -> Dict[str, Any]:
    """
    Returns a STRICT-JSON dict:
    {
      "metrics": {
        "RRP": [ {timestamp, frequency, forecast, justification}, ... ],
        "TOTALDEMAND": [ ... ]
      }
    }
    """
    tgt = _derive_targets_from_filters(filters, cfg.tz)
    regs = tgt["regions"] or ["UNKNOWN-REGION"]
    mets = tgt["metrics"] or ["TOTALDEMAND", "RRP"]

    units_lines = "\n".join([f"- {m}: {cfg.units_map.get(m, '')}" for m in mets])

    summary_block  = _cap_text(summary)
    stats_block    = _cap_text(stats)
    patterns_block = _cap_text(patterns)

    system_msg = (
        "You are a precise time-series forecaster. "
        "Generate **timestamped** forecasts for each requested metric. "
        "Prefer ~3 significant digits and include units. "
        "Respect requested frequency and window. "
        "Do NOT repeat the same numeric value for consecutive timestamps unless the evidence is identical; if identical, justify equality explicitly."
    )

    targets_header = _format_targets_header(tgt, route)

    user_msg = f"""
{targets_header}

UNITS
{units_lines}

USER QUERY
{user_query}

EVIDENCE (use as-is; do not recalc):
- Statistical Insights:
{stats_block}

- Historical Summary:
{summary_block}

- Detected Patterns:
{patterns_block}

OUTPUT STYLE — STRICT JSON ONLY (no prose)
Schema:
{{
  "metrics": {{
    "RRP": [
      {{
        "timestamp": "<ISO timestamp or explicit date/time>",
        "frequency": "<e.g. 15min|30min|1h|1d>",
        "forecast": "<value with unit, e.g. $56.4/MWh>",
        "justification": "tie to the EVIDENCE (patterns/stats/summary)"
      }}
    ],
    "TOTALDEMAND": [
      {{
        "timestamp": "<ISO timestamp or explicit date/time>",
        "frequency": "<e.g. 15min|30min|1h|1d>",
        "forecast": "<value with unit, e.g. 8,930 MW>",
        "justification": "tie to the EVIDENCE"
      }}
    ]
  }}
}}

Rules:
- Produce **multiple timestamps** across the requested window/frequency (no single point unless the user asked for just one).
- Use different values per timestamp unless you explicitly justify equality (e.g., flat night hours).
- Always include both metrics if requested or default to ['TOTALDEMAND','RRP'] when unspecified.
"""
    # Call adapter with strict JSON first (falls back loosely inside adapter)
    messages = [{"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}]
    out = adapter.chat_json_loose(
        messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        model_override=(cfg.model_override or cfg.model),
        strict_json_first=True,
    )
    if not isinstance(out, dict):
        # very defensive
        try:
            out = json.loads(out)
        except Exception:
            out = {"metrics": {"RRP": [], "TOTALDEMAND": []}}

    # Optionally nudge duplicates
    if cfg.enforce_unique_timestamps and isinstance(out, dict) and "metrics" in out:
        out = _nudge_duplicate_timestamps(out)

    return out