from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Callable, Set
from datetime import datetime, date
import json, re, itertools

# bring your adapter
from utils.model_registry import LLMClientAdapter

# Optional holidays provider (pip install holidays)
try:
    import holidays as _holidays_pkg  # type: ignore
except Exception:
    _holidays_pkg = None

# -------------------------
# Shared constants & helpers
# -------------------------
DEFAULT_VAGUE_TIME_MAP: Dict[str, Tuple[str, int]] = {
    "early morning": ("07:00:00", 120),
    "morning":       ("09:00:00", 180),
    "late morning":  ("11:00:00", 120),
    "noon":          ("12:00:00", 60),
    "afternoon":     ("15:00:00", 180),
    "late afternoon":("17:00:00", 120),
    "evening":       ("19:00:00", 180),
    "late evening":  ("21:00:00", 120),
    "night":         ("22:00:00", 180),
    "midnight":      ("00:00:00", 60),
    "early night":   ("20:00:00", 120),
    "commute":       ("08:30:00", 60),
}


def _normalize_freq(freq_raw: Optional[str]) -> Optional[str]:
    if not freq_raw: return None
    s = str(freq_raw).strip().lower()
    alias = {
    "5":"5min","5m":"5min","5 min":"5min","5mins":"5min","5 minutes":"5min",
    "20":"20min","20m":"20min","20 min":"20min","20mins":"20min","20 minutes":"20min",
    "15":"15min","15m":"15min","15 min":"15min","15mins":"15min","15 minutes":"15min",
    "quarter hour":"15min","quarter-hour":"15min",
    "30":"30min","30m":"30min","30 min":"30min","30mins":"30min","30 minutes":"30min",
    "half hour":"30min","half-hour":"30min",
    "h":"1H","1h":"1H","1hr":"1H","hour":"1H","hourly":"1H","per hour":"1H","hrs":"1H",
    "d":"1D","1d":"1D","day":"1D","daily":"1D","per day":"1D"
}

    if s in alias: return alias[s]
    m = re.match(r"^\s*(\d+)\s*(h|hr|hrs|hour|hours)\s*$", s)
    if m: return f"{int(m.group(1))}H"
    m = re.match(r"^\s*(\d+)\s*(m|min|mins|minute|minutes)\s*$", s)
    if m: return f"{int(m.group(1))}min"
    m = re.match(r"^\s*(\d+)\s*(d|day|days)\s*$", s)
    if m: return f"{int(m.group(1))}D"
    s2 = s.replace(" ","")
    if re.match(r"^\d+min$", s2): return s2
    if re.match(r"^\d+[hHdD]$", s2): return s2[:-1] + s2[-1].upper()
    return freq_raw

def _date_or_none(s: Any) -> Optional[str]:
    if not s: return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
    except Exception:
        return None

def _iso_or_none(s: Any) -> Optional[str]:
    if not s: return None
    try:
        datetime.fromisoformat(str(s).replace("Z","+00:00"))
        return str(s)
    except Exception:
        return None

def _iso_dt_or_none(s: Any) -> Optional[datetime]:
    if not s: return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None

def _date_to_day_bounds(d_str: str) -> Tuple[str, str]:
    # Naive ISO (no tz). If you prefer tz-aware, localize/convert using your cfg.tz.
    d = datetime.strptime(d_str, "%Y-%m-%d")
    start = d.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    end   = d.replace(hour=23, minute=59, second=59, microsecond=0).isoformat()
    return start, end

def _apply_vague_time(label: Optional[str], vague_map: Dict[str, Tuple[str,int]]) -> Tuple[Optional[str], Optional[int]]:
    if not label: return (None, None)
    lab = str(label).strip().lower()
    if lab in vague_map:
        center, tol = vague_map[lab]
        return (center, int(tol))
    return (None, None)

def _as_list(x):
    if x is None or x == "": return []
    if isinstance(x, list): return x
    return [x]

def _as_int_list(x):
    out = []
    for v in _as_list(x):
        try:
            out.append(int(str(v).strip()))
        except Exception:
            pass
    return out

def _as_str_list(x):
    out = []
    for v in _as_list(x):
        s = str(v).strip()
        if s: out.append(s)
    return out

def _as_bool(x):
    if isinstance(x, bool): return x
    s = str(x).strip().lower()
    if s in {"true","1","yes","y"}: return True
    if s in {"false","0","no","n"}: return False
    return False

def _pad_hhmm(s: str) -> str:
    # "07:00" -> "07:00:00"
    if re.match(r"^\d{2}:\d{2}:\d{2}$", s): return s
    if re.match(r"^\d{2}:\d{2}$", s): return s + ":00"
    return s


def _coerce_time_filters_for_schema(obj: dict) -> dict:
    """
    Make a best-effort cast to the types your TIME_JSON_SCHEMA expects.
    This runs BEFORE validation so that routine LLM shortcuts don't crash validation.
    """
    d = dict(obj or {})

    # Arrays of ints/strings
    d["years"]   = _as_int_list(d.get("years"))
    d["months"]  = _as_int_list(d.get("months"))
    d["days"]    = _as_int_list(d.get("days"))
    d["times"]   = [_pad_hhmm(s) for s in _as_str_list(d.get("times"))]
    d["weekdays"]= [str(w).strip().upper() for w in _as_str_list(d.get("weekdays"))]

    # Booleans
    d["weekends"] = _as_bool(d.get("weekends"))
    d["holidays"] = _as_bool(d.get("holidays"))

    # Date lists
    d["weekend_dates"] = _as_str_list(d.get("weekend_dates"))
    d["holiday_dates"] = _as_str_list(d.get("holiday_dates"))

    # Scalars (nullable strings)
    for k in ("date_start","date_end","start_time","end_time","freq","vague_label","notes","modality"):
        v = d.get(k)
        if v is None or v == "": 
            d[k] = None
        else:
            d[k] = str(v)

    # Vague tolerance (nullable int)
    vtol = d.get("vague_tolerance_minutes")
    if isinstance(vtol, int):
        d["vague_tolerance_minutes"] = vtol
    elif isinstance(vtol, str) and vtol.strip().isdigit():
        d["vague_tolerance_minutes"] = int(vtol.strip())
    else:
        d["vague_tolerance_minutes"] = None

    # single_timestamps (list of strings)
    d["single_timestamps"] = _as_str_list(d.get("single_timestamps"))

    # timestamp_ranges / date_ranges (list of {start,end} dicts)
    def _mk_ts_range(x):
        if isinstance(x, dict):
            s = x.get("start_time") or x.get("start") or x.get("from") or x.get("timestamp_start")
            e = x.get("end_time")   or x.get("end")   or x.get("to")   or x.get("timestamp_end")
        else:
            s = e = None
        return {"start_time": (str(s) if s else None), "end_time": (str(e) if e else None)}

    def _mk_date_range(x):
        if isinstance(x, dict):
            s = x.get("start_date") or x.get("start") or x.get("from") or x.get("date_start")
            e = x.get("end_date")   or x.get("end")   or x.get("to")   or x.get("date_end")
        else:
            s = e = None
        return {"start_date": (str(s) if s else None), "end_date": (str(e) if e else None)}

    d["timestamp_ranges"] = [_mk_ts_range(x) for x in _as_list(d.get("timestamp_ranges"))]
    d["date_ranges"]      = [_mk_date_range(x) for x in _as_list(d.get("date_ranges"))]

    # horizon (object with steps:int|null, units:str|null)
    hz = d.get("horizon")
    if isinstance(hz, dict):
        h = hz
    elif isinstance(hz, str):
        h = {"hint": hz}
    elif isinstance(hz, (list, tuple)) and hz:
        # handle cases like ["short_term"] or [{"hint":"short_term"}]
        if isinstance(hz[0], dict):
            h = hz[0]
        elif isinstance(hz[0], str):
            h = {"hint": hz[0]}
        else:
            h = {}
    else:
        h = {}
    steps_raw = h.get("steps")
    if isinstance(steps_raw, int):
        steps_val = steps_raw
    elif isinstance(steps_raw, str) and steps_raw.strip().isdigit():
        steps_val = int(steps_raw.strip())
    else:
        steps_val = None
    units_raw = h.get("units")
    units_val = str(units_raw).lower() if isinstance(units_raw, str) and units_raw.strip() else None
    if units_val not in {None,"hours","days","weeks","months"}:
        units_val = None
    d["horizon"] = {"steps": steps_val, "units": units_val}

    return d



# -------------------------
# Lightweight schema + JSON enforcement
# -------------------------
def _light_validate(obj: dict, schema: dict) -> List[str]:
    errs: List[str] = []
    if not isinstance(obj, dict):
        return ["Output is not a JSON object"]

    required = schema.get("required", [])
    props = schema.get("properties", {})
    addl_ok = schema.get("additionalProperties", True)

    for k in required:
        if k not in obj:
            errs.append(f"Missing required key: {k}")

    if not addl_ok:
        extra = set(obj.keys()) - set(props.keys())
        if extra:
            errs.append(f"Unexpected keys: {sorted(extra)}")

    def _is_type(v, t):
        if t == "string": return isinstance(v, str)
        if t == "integer": return isinstance(v, int) and not isinstance(v, bool)
        if t == "array": return isinstance(v, list)
        if t == "object": return isinstance(v, dict)
        if t == "boolean": return isinstance(v, bool)
        if t == "null": return v is None
        if isinstance(t, list): return any(_is_type(v, tt) for tt in t)
        return True

    for k, spec in props.items():
        if k not in obj: continue
        v = obj[k]; t = spec.get("type")
        if t and not _is_type(v, t):
            errs.append(f"Key '{k}' has wrong type; expected {t}")
        if "items" in spec and isinstance(v, list):
            it = spec["items"].get("type")
            if it:
                for i, val in enumerate(v):
                    if not _is_type(val, it):
                        errs.append(f"Key '{k}[{i}]' has wrong element type; expected {it}")
        if "minimum" in spec and isinstance(v, int) and v < spec["minimum"]:
            errs.append(f"Key '{k}' below minimum {spec['minimum']}")
        if "maximum" in spec and isinstance(v, int) and v > spec["maximum"]:
            errs.append(f"Key '{k}' above maximum {spec['maximum']}")
    return errs

def _json_salvage(s: str) -> Optional[dict]:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            i = s.index("{"); j = s.rindex("}") + 1
            return json.loads(s[i:j])
        except Exception:
            return None

def _chat_json_required(
    adapter: LLMClientAdapter,
    messages: List[Dict[str, str]],
    json_schema: dict,
    *,
    temperature: float = 0.0,
    max_tokens: int = 600,
    use_json_mode: bool = True,
    max_retries: int = 2,
) -> dict:
    """
    Enforce JSON via:
      1) ask provider for JSON mode (adapter strips if unsupported)
      2) salvage -> schema-validate -> retry with a concise fix prompt
    """
    rf = {"type": "json_object"} if use_json_mode else None
    err_last = None
    local_messages = list(messages)
    for _ in range(max_retries + 1):
        try:
            txt = adapter.chat(
                local_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=rf,
            )
            obj = _json_salvage(txt)
            if obj is None:
                raise ValueError("Non-JSON response")
            
            obj = _coerce_time_filters_for_schema(obj)

            errs = _light_validate(obj, json_schema)
            if not errs:
                return obj

            local_messages = local_messages + [{
                "role": "user",
                "content": (
                    "Your previous output did not match the JSON schema.\n"
                    f"Errors: {errs}\n"
                    "Return ONLY a valid JSON object that satisfies the schema."
                ),
            }]
            err_last = ValueError("; ".join(errs))
        except Exception as e:
            err_last = e
    raise err_last or RuntimeError("Unknown error enforcing JSON output")

@dataclass
class TimeSeriesConfig:
    temperature: float = 0.0
    max_tokens: int = None
    vague_time_map: Dict[str, Tuple[str,int]] = field(default_factory=lambda: DEFAULT_VAGUE_TIME_MAP)
    holiday_fn: Optional[Callable[[date], bool]] = None
    holiday_country: Optional[str] = "AU"
    holiday_state: Optional[str] = None  

TIME_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "years","months","days","times",
        "date_start","date_end","notes","start_time","end_time","freq",
        "weekdays","weekends","holidays",
        "horizon","modality",
        "single_timestamps","timestamp_ranges","date_ranges",
        "vague_label","vague_tolerance_minutes",
    ],
    "properties": {
        "years":  {"type":"array","items":{"type":"integer"}},
        "months": {"type":"array","items":{"type":"integer","minimum":1,"maximum":12}},
        "days":   {"type":"array","items":{"type":"integer","minimum":1,"maximum":31}},
        "times":  {"type":"array","items":{"type":"string"}},

        "date_start":{"type":["string","null"]},
        "date_end":  {"type":["string","null"]},
        "notes":     {"type":["string","null"]},
        "start_time":{"type":["string","null"]},
        "end_time":  {"type":["string","null"]},
        "freq":      {"type":["string","null"]},

        "weekdays": {"type":"array","items":{"type":"string"}},
        "weekends": {"type":"boolean"},
        "weekend_dates": { "type":"array", "items":{"type":"string"}},    # ISO YYYY-MM-DD
        "holidays": {"type":"boolean"},
        "holiday_dates": { "type":"array", "items":{"type":"string"}},    # ISO YYYY-MM-DD
        
        "horizon": {
            "type": ["object", "null"],
            "required": ["steps","units"],
            "additionalProperties": False,
            "properties": {
                "steps": {"type":["integer","null"]},
                "units": {"type":["string","null"]}
            }
        },
        "modality": {"type":["string","null"]},

        "single_timestamps": {"type":"array","items":{"type":"string"}},
        "timestamp_ranges": {
            "type":"array",
            "items":{
                "type":"object",
                "required":["start_time","end_time"],
                "additionalProperties": False,
                "properties":{
                    "start_time":{"type":["string","null"]},
                    "end_time":{"type":["string","null"]},
                }
            }
        },
        "date_ranges": {
            "type":"array",
            "items":{
                "type":"object",
                "required":["start_date","end_date"],
                "additionalProperties": False,
                "properties":{
                    "start_date":{"type":["string","null"]},
                    "end_date":{"type":["string","null"]},
                }
            }
        },

        "vague_label":{"type":["string","null"]},
        "vague_tolerance_minutes":{"type":["integer","null"]},
    },
}

class TimeSeriesFilterExtractor:
    """Extracts ONLY temporal constraints. `now_iso` must be provided by the caller (anchored outside)."""
    
    def __init__(self, adapter: LLMClientAdapter, cfg: Optional[TimeSeriesConfig] = None):
        self.adapter = adapter
        self.cfg = cfg or TimeSeriesConfig()

    
    # ---------- Prompt ----------
    def _prompt(self, user_query: str, now_iso: str) -> List[Dict[str,str]]:
        sys = (
            "You extract ONLY temporal constraints for time-series queries. "
            "Return STRICT JSON that matches the provided schema. No prose."
        )
        usr = f"""
Rules:
- If both endpoints of a 'from ... to ...' contain times, use ISO 'start_time'/'end_time'.
- If endpoints are dates only, use 'date_start'/'date_end' (YYYY-MM-DD).
- Relative windows ('next 48 hours', 'next week'): compute ISO 'start_time'/'end_time' using NOW="{now_iso}".
- Cadence: if stated (e.g., 'every 30 minutes', 'hourly', '1H'), put it in 'freq'; else null.
- Vague times ('morning', 'afternoon', etc.) → set 'vague_label' and leave tolerance null.
- Week filters: fill 'weekdays' with 3-letter uppercase codes (MON..SUN). Set 'weekends' boolean if user asked for weekends.
- Horizon: if a forecast/lead is mentioned (e.g., '24 hours ahead', '2 weeks'), set horizon as steps+units.
- Modality: is the forecast univariate/multivariate, set 'modality' accordingly; else null.
- Single timestamps → exact ISO points in 'single_timestamps'.
- Timestamp spans → list under 'timestamp_ranges' with ISO 'start_time'/'end_time'.
- Date spans → list under 'date_ranges' with 'start_date'/'end_date'.
- Output ONLY temporal info; no regions/metrics. Extra cues go in 'notes'.
- if notes is not null, search for extra cues in notes and add it to 'notes'.
- if it is one-point in time, set both date_start and date_end to that date.

Return a JSON object with keys:
{json.dumps(list(TIME_JSON_SCHEMA["properties"].keys()))}

User Query: {user_query.strip()}
""".strip()
        return [{"role":"system","content":sys},{"role":"user","content":usr}]
    
    # ---------- Public API ----------
    
    def extract(self, user_query: str, *, now_iso: Optional[str]) -> Dict[str, Any]:
        """
        Run the temporal extractor.

        Args:
            user_query: user text
            now_iso: ISO-8601 string used to anchor relative windows (computed externally)

        Returns:
            Dict[str, Any]: temporal-only structure with date_start/date_end as ISO datetimes
        """
        if not isinstance(user_query, str) or not user_query.strip():
            raise ValueError("user_query must be a non-empty string")
        if not now_iso or not _iso_dt_or_none(now_iso):
            raise ValueError("now_iso must be a valid ISO-8601 datetime string")

        # LLM step (schema enforced)
        obj = _chat_json_required(
            self.adapter,
            self._prompt(user_query, now_iso),
            TIME_JSON_SCHEMA,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            use_json_mode=True,
        )

        # Normalize core fields
        out = self._normalize_llm_payload(obj)

        # Collapse to ISO datetimes in date_start/date_end (central requirement)
        out = self._collapse_datetimes(out)

        # Deterministic calendar inference from concrete dates
        derived_dates = self._collect_concrete_dates(out)
        out = self._enrich_from_dates(out, derived_dates)

        # Apply vague time mapping → keep as helpers
        vcenter, vtol = _apply_vague_time(out.get("vague_label"), self.cfg.vague_time_map)
        out["_vague_center"] = vcenter
        out["_vague_tolerance_minutes"] = vtol
        # If LLM didn’t supply tolerance, fall back to our map
        if out.get("vague_label") and out.get("vague_tolerance_minutes") is None:
            out["vague_tolerance_minutes"] = vtol

        return out

    # ---------- Normalization ----------
    
    def _normalize_llm_payload(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        years  = [int(y) for y in (obj.get("years") or []) if str(y).isdigit()]
        months = [m for m in (obj.get("months") or []) if isinstance(m,int) and 1<=m<=12]
        days   = [d for d in (obj.get("days") or []) if isinstance(d,int) and 1<=d<=31]

        times: List[str] = []
        for t in obj.get("times") or []:
            s = str(t).strip()
            if re.match(r"^\d{2}:\d{2}:\d{2}$", s):
                times.append(s)
            elif re.match(r"^\d{2}:\d{2}$", s):
                times.append(s + ":00")

        out = {
            "years": years,
            "months": months,
            "days": days,
            "times": times,
            "date_start": _date_or_none(obj.get("date_start")),
            "date_end":   _date_or_none(obj.get("date_end")),
            "notes":      (obj.get("notes") or None),
            "start_time": _iso_or_none(obj.get("start_time")),
            "end_time":   _iso_or_none(obj.get("end_time")),
            "freq":       _normalize_freq(obj.get("freq")),

            "weekdays": [str(x).strip().upper() for x in (obj.get("weekdays") or [])],
        "weekends": bool(obj.get("weekends", False)),
        "weekend_dates": [d for d in (obj.get("weekend_dates") or []) if _date_or_none(d)],

        "holidays": bool(obj.get("holidays", False)),
        "holiday_dates": [d for d in (obj.get("holiday_dates") or []) if _date_or_none(d)],

        "modality": (str(obj.get("modality")).lower() if obj.get("modality") else None),

        # keep the model's vague tolerance if it provided one
        "vague_label": obj.get("vague_label"),
        "vague_tolerance_minutes": (
            int(obj.get("vague_tolerance_minutes"))
            if str(obj.get("vague_tolerance_minutes") or "").strip().isdigit()
            else None
        ),

            "single_timestamps": [st for st in (obj.get("single_timestamps") or []) if _iso_or_none(st)],
            "timestamp_ranges":  [{"start_time": _iso_or_none((r or {}).get("start_time")),
                                  "end_time":   _iso_or_none((r or {}).get("end_time"))}
                                  for r in (obj.get("timestamp_ranges") or [])],
            "date_ranges":       [{"start_date": _date_or_none((r or {}).get("start_date")),
                                  "end_date":   _date_or_none((r or {}).get("end_date"))}
                                  for r in (obj.get("date_ranges") or [])],
                                  }

        # Horizon
        hraw = obj.get("horizon") or {}
        steps_raw = hraw.get("steps")
        if isinstance(steps_raw, int):
            steps_val = steps_raw
        elif isinstance(steps_raw, str) and steps_raw.strip().isdigit():
            steps_val = int(steps_raw.strip())
        else:
            steps_val = None

        units_raw = hraw.get("units")
        units_val = None
        if isinstance(units_raw, str) and units_raw.strip():
            units_val = units_raw.strip().lower()
            if units_val not in {"hours","days","weeks","months"}:
                units_val = None

        out["horizon"] = {"steps": steps_val, "units": units_val}

        # Clean modality
        if out["modality"] not in {None,"univariate","multivariate"}:
            out["modality"] = None

        # Drop invalid weekday tokens
        valid_wd = {"MON","TUE","WED","THU","FRI","SAT","SUN"}
        wd = [w for w in out["weekdays"] if w in valid_wd]
        out["weekdays"] = list(dict.fromkeys(wd))
        return out

    # ---------- Collapse ----------
    def _collapse_datetimes(self, out: Dict[str, Any]) -> Dict[str, Any]:
        st_iso  = _iso_dt_or_none(out.get("start_time"))
        en_iso  = _iso_dt_or_none(out.get("end_time"))
        ds_date = _date_or_none(out.get("date_start"))
        de_date = _date_or_none(out.get("date_end"))

        if st_iso or en_iso:
            if st_iso:
                out["date_start"] = st_iso.isoformat()
            elif ds_date:
                out["date_start"] = _date_to_day_bounds(ds_date)[0]
            if en_iso:
                out["date_end"] = en_iso.isoformat()
            elif de_date:
                out["date_end"] = _date_to_day_bounds(de_date)[1]
        else:
            if ds_date and de_date:
                s_iso, _ = _date_to_day_bounds(ds_date)
                _, e_iso = _date_to_day_bounds(de_date)
                out["date_start"] = s_iso
                out["date_end"]   = e_iso
            elif ds_date and not de_date:
                s_iso, e_iso = _date_to_day_bounds(ds_date)
                out["date_start"] = s_iso
                out["date_end"]   = e_iso
            elif de_date and not ds_date:
                s_iso, e_iso = _date_to_day_bounds(de_date)
                out["date_start"] = s_iso
                out["date_end"]   = e_iso
        return out

    # ---------- Calendar enrichment ----------
    def _collect_concrete_dates(self, out: Dict[str, Any]) -> Set[date]:
        dates: Set[date] = set()

        def add_date_span(d0: Optional[str], d1: Optional[str]):
            if not (d0 or d1): return
            try:
                sd = datetime.fromisoformat((d0 or d1) + "T00:00:00").date() if d0 else None
                ed = datetime.fromisoformat((d1 or d0) + "T00:00:00").date() if d1 else None
            except Exception:
                return
            if sd and not ed: ed = sd
            if ed and not sd: sd = ed
            if sd and ed and sd <= ed:
                cur = sd
                while cur <= ed:
                    dates.add(cur)
                    cur = cur.fromordinal(cur.toordinal()+1)

        # From date_start/date_end (ISO with times) — derive dates
        for key in ("date_start", "date_end"):
            if out.get(key):
                try:
                    d = datetime.fromisoformat(out[key].replace("Z","+00:00")).date()
                    dates.add(d)
                except Exception:
                    pass

        # date_ranges
        for r in out.get("date_ranges") or []:
            add_date_span((r or {}).get("start_date"), (r or {}).get("end_date"))

        # start_time/end_time
        def add_ts_span(ts0: Optional[str], ts1: Optional[str]):
            if not (ts0 or ts1): return
            try:
                sdt = datetime.fromisoformat((ts0 or ts1).replace("Z","+00:00"))
                edt = datetime.fromisoformat((ts1 or ts0).replace("Z","+00:00"))
            except Exception:
                return
            if sdt > edt: sdt, edt = edt, sdt
            d = sdt.date()
            while d <= edt.date():
                dates.add(d)
                d = d.fromordinal(d.toordinal()+1)

        add_ts_span(out.get("start_time"), out.get("end_time"))

        # timestamp_ranges
        for r in (out.get("timestamp_ranges") or []):
            add_ts_span((r or {}).get("start_time"), (r or {}).get("end_time"))

        # single_timestamps
        for ts in (out.get("single_timestamps") or []):
            try:
                d = datetime.fromisoformat(ts.replace("Z","+00:00")).date()
                dates.add(d)
            except Exception:
                pass

        # Cartesian expansion
        years  = out.get("years") or []
        months = out.get("months") or []
        days   = out.get("days") or []
        MAX_COMBOS = 366
        if years and months and days:
            combos = list(itertools.product(years, months, days))
            if len(combos) <= MAX_COMBOS:
                for y,m,d in combos:
                    try:
                        dates.add(date(int(y), int(m), int(d)))
                    except Exception:
                        pass
        return dates

    def _enrich_from_dates(self, out: Dict[str, Any], dates: Set[date]) -> Dict[str, Any]:
        if not dates:
            out["weekends"] = bool(out.get("weekends", False))
            out["holidays"] = bool(out.get("holidays", False))
            return out

        years  = set(out.get("years") or [])
        months = set(out.get("months") or [])
        days   = set(out.get("days") or [])
        for d in dates:
            years.add(d.year); months.add(d.month); days.add(d.day)
        out["years"]  = sorted(years)
        out["months"] = sorted(months)
        out["days"]   = sorted(days)

        wd_codes = ["MON","TUE","WED","THU","FRI","SAT","SUN"]
        wd_seen: Set[str] = set(wd_codes[d.weekday()] for d in dates)
        out["weekdays"] = sorted(wd_seen, key=lambda x: wd_codes.index(x))

        is_all_weekend = all((d.weekday() >= 5) for d in dates)
        has_weekend    = any((d.weekday() >= 5) for d in dates)
        out["weekends"] = bool(out.get("weekends")) or is_all_weekend or has_weekend

        out["holidays"] = bool(out.get("holidays")) or self._dates_have_holiday(dates, out.get("notes"))
        return out

    def _dates_have_holiday(self, dates: Set[date], notes: Optional[str]) -> bool:
        if callable(self.cfg.holiday_fn):
            if any(self.cfg.holiday_fn(d) for d in dates):
                return True
        if _holidays_pkg and self.cfg.holiday_country:
            try:
                cal = getattr(_holidays_pkg, self.cfg.holiday_country, None)
                if cal:
                    hd = cal()  # type: ignore
                    if any(d in hd for d in dates):
                        return True
            except Exception:
                pass
        tokens = (notes or "").lower()
        HOLI_HINTS = ["holiday","public holiday","christmas","new year","boxing day","easter","good friday","labor day","anzac"]
        return any(h in tokens for h in HOLI_HINTS)

    def _list_dates_between(self, start_iso: Optional[str], end_iso: Optional[str]) -> List[date]:
        if not (start_iso and end_iso): return []
        try:
            sdt = datetime.fromisoformat(start_iso.replace("Z","+00:00")).date()
            edt = datetime.fromisoformat(end_iso.replace("Z","+00:00")).date()
        except Exception:
            return []
        if sdt > edt: sdt, edt = edt, sdt
        out, cur = [], sdt
        while cur <= edt:
            out.append(cur)
            cur = cur.fromordinal(cur.toordinal()+1)
        return out

    def _derive_weekend_and_holidays(self, out: Dict[str,Any], dates: Set[date]) -> Dict[str,Any]:
        # Base set = union of all concrete dates we’ve already inferred
        window_dates: Set[date] = set(dates)

        # If the user gave a concrete outer window (date_start/date_end), fill the full span
        ds, de = out.get("date_start"), out.get("date_end")
        span = self._list_dates_between(ds, de)
        window_dates.update(span)

        # Derive weekend dates
        weekend_dates = sorted({d for d in window_dates if d.weekday() >= 5})
        out["weekend_dates"] = [d.isoformat() for d in weekend_dates]

        # Derive holiday dates
        holiday_dates: List[date] = []
        if callable(self.cfg.holiday_fn):
            holiday_dates = [d for d in window_dates if self.cfg.holiday_fn(d)]
        elif _holidays_pkg and self.cfg.holiday_country:
            try:
                cal_cls = getattr(_holidays_pkg, self.cfg.holiday_country, None)
                if cal_cls:
                    # Python-holidays AU supports state='QLD'/etc.
                    if self.cfg.holiday_country == "AU":
                        hd = cal_cls(state=self.cfg.holiday_state)  # may be None
                    else:
                        hd = cal_cls()
                    holiday_dates = [d for d in window_dates if d in hd]
            except Exception:
                pass

        out["holiday_dates"] = sorted({d.isoformat() for d in holiday_dates})
        # Flags remain truthful even if lists are empty
        out["weekends"] = bool(out.get("weekends") or len(out["weekend_dates"]) > 0)
        out["holidays"] = bool(out.get("holidays") or len(out["holiday_dates"]) > 0)
        return out  