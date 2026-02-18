from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import json
from utils.model_registry import LLMClientAdapter


# -----------------------------------------------------------
# helpers
# -----------------------------------------------------------
REGION_ALIASES: Dict[str, str] = {
    "nsw1":"NSW1","qld1":"QLD1","sa1":"SA1","vic1":"VIC1","tas1":"TAS1",
    "nsw":"NSW1","new south wales":"NSW1","sydney":"NSW1",
    "qld":"QLD1","queensland":"QLD1","brisbane":"QLD1",
    "sa":"SA1","south australia":"SA1","adelaide":"SA1",
    "vic":"VIC1","victoria":"VIC1","melbourne":"VIC1",
    "tas":"TAS1","tasmania":"TAS1","hobart":"TAS1",
    "nsw.":"NSW1","vic.":"VIC1","qld.":"QLD1","s.a.":"SA1",
}

METRIC_ALIASES: Dict[str, str] = {
    "rrp":"RRP","price":"RRP","$/mwh":"RRP","spot price":"RRP","wholesale price":"RRP",
    "totaldemand":"TOTALDEMAND","demand":"TOTALDEMAND","load":"TOTALDEMAND","mw":"TOTALDEMAND",
}
ALLOWED_METRICS = {"TOTALDEMAND","RRP"}

def _norm_regions(regions: List[str]) -> List[str]:
    """
    Normalize region identifiers to canonical AEMO region codes.

    This function maps a list of potentially noisy or user-provided region strings
    (e.g., "NSW", "Sydney", "vic.", "tasmania") to canonical region codes used in
    downstream processing (e.g., "NSW1", "VIC1", "TAS1").

    Parameters
    ----------
    regions : List[str]
        Raw region-like strings extracted from user text or other upstream steps.

    Returns
    -------
    List[str]
        Canonicalized region codes with stable ordering and no duplicates.
    """
    out, seen = [], set()
    for r in regions or []:
        key = str(r).strip().lower()
        canon = REGION_ALIASES.get(key, str(r).strip().upper())
        if canon not in seen:
            out.append(canon); seen.add(canon)
    return out

def _norm_metrics(metrics: List[str]) -> List[str]:
    """
    Normalize metric identifiers to the allowed energy forecasting targets.

    This function maps metric synonyms to the two supported canonical metrics:
    - "TOTALDEMAND" (e.g., "demand", "load", "MW")
    - "RRP" (e.g., "price", "$/MWh", "spot price")

    Parameters
    ----------
    metrics : List[str]
        Raw metric-like strings extracted from user text or other upstream steps.

    Returns
    -------
    List[str]
        Canonical metric names restricted to {"TOTALDEMAND","RRP"} with stable
        ordering and no duplicates.
    """
    out, seen = [], set()
    for m in (metrics or []):
        k = str(m).strip().lower()
        canon = METRIC_ALIASES.get(k, str(m).strip().upper())
        if canon in ALLOWED_METRICS and canon not in seen:
            out.append(canon); seen.add(canon)
    return out

# -------------------------
# JSON enforcement
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
    max_tokens: int = 800,
    use_json_mode: bool = True,
    max_retries: int = 2,
) -> dict:
    """
    Call an LLM and strictly enforce a JSON-object response matching a schema.

    Parameters
    ----------
    adapter : LLMClientAdapter
        Adapter that exposes a `chat(messages, ...) -> str` interface.
    messages : List[Dict[str, str]]
        Chat messages in OpenAI-style format: [{"role":"system|user|assistant","content": "..."}].
    json_schema : dict
        JSON schema-like dict (subset) used by _light_validate().
    temperature : float, default=0.0
        Decoding temperature. Keep at 0.0 for deterministic extraction.
    max_tokens : int, default=800
        Output token budget for the extraction call.
    use_json_mode : bool, default=True
        If True, request JSON mode (where supported) via response_format.
    max_retries : int, default=2
        Number of repair attempts after the initial call.

    Returns
    -------
    dict
        A JSON object satisfying the schema.
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
class EnergyConfig:
    temperature: float = 0.0
    max_tokens: int = 800

ENERGY_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["regions","metrics","notes","period_type"],
    "properties": {
        "regions": {"type":"array","items":{"type":"string"}},
        "metrics": {"type":"array","items":{"type":"string"}},
        "notes":   {"type":["string","null"]},
        "period_type": {"type":["string","null"]},
    }
}

class EnergyFilterExtractor:
    """
    Extract and normalize energy-domain constraints (regions/metrics/period_type) from a user query.

    Output contract
    ---------------
    Returns a dict:
      - regions: canonical region codes (e.g., ["NSW1","VIC1"])
      - metrics: subset of ["TOTALDEMAND","RRP"]
      - notes: optional string; if period_type was present, it is appended as "PERIODTYPE=<...>"

    Temporal constraints are intentionally not extracted here.
    """

    def __init__(self, adapter: LLMClientAdapter, cfg: Optional[EnergyConfig] = None):
        self.adapter = adapter
        self.cfg = cfg or EnergyConfig()

    def _prompt(self, user_query: str) -> List[Dict[str,str]]:
        sys = (
            "You extract ONLY energy domain constraints (regions, metrics, optional period_type). "
            "Return STRICT JSON matching the schema. No prose."
        )
        usr = f"""
Rules:
- Regions: return any region/place mentions literally (e.g., "NSW1","NSW","Sydney","Victoria"). DO NOT normalize.
- Metrics: choose from ["TOTALDEMAND","RRP"] using user wording and synonyms
  (demand/load/MW → TOTALDEMAND; price/RRP/$ per MWh → RRP). If unclear, return [].
- If the user mentions a 'period type' (like "PEAK/OFF-PEAK", "PERIODTYPE"), set "period_type" to that token; else null.
- Do not include any temporal information here.

Return a JSON object with keys:
{json.dumps(list(ENERGY_JSON_SCHEMA["properties"].keys()))}

User Query: {user_query.strip()}
""".strip()
        return [{"role":"system","content":sys},{"role":"user","content":usr}]

    def extract(self, user_query: str) -> Dict[str, Any]:
        if not isinstance(user_query, str) or not user_query.strip():
            raise ValueError("user_query must be a non-empty string")

        obj = _chat_json_required(
            self.adapter,
            self._prompt(user_query),
            ENERGY_JSON_SCHEMA,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            use_json_mode=True,
        )

        regions_raw = [str(r).strip() for r in (obj.get("regions") or []) if str(r).strip()]
        metrics_raw = [str(m).strip() for m in (obj.get("metrics") or []) if str(m).strip()]
        period_type = (obj.get("period_type") or None)
        notes = (obj.get("notes") or None)

        regions = _norm_regions(regions_raw)
        metrics = _norm_metrics(metrics_raw)

        merged_notes = notes if not period_type else (f"{notes} | PERIODTYPE={period_type}" if notes else f"PERIODTYPE={period_type}")

        return {
            "regions": regions,
            "metrics": metrics,
            "notes": merged_notes,
        }