import json, math, re
from typing import Any, Iterable, Optional, Union

JSONLike = Union[str, dict, list]

def clean_tokens_ultra(
    obj: JSONLike,
    *,
    float_decimals: int = 3,
    drop_keys: Optional[Iterable[str]] = None,
    max_chars: Optional[int] = None,
    ensure_ascii: bool = False,
    # remove quotes, backslashes, braces, brackets by default
    remove_chars: str = '"\\{}[]',
    remove_escaped_whitespace: bool = True,
    squash_whitespace: bool = True,
    strip_punct_spaces: bool = True,
) -> str:
    """
    Produce an *extremely* lean, human-readable string for prompts.
    WARNING: Output is NOT valid JSON (quotes/braces/brackets removed).

    Steps:
      1) If dict/list/JSON string -> compact JSON and round floats.
      2) Remove literal \\n \\t \\r.
      3) Remove characters in `remove_chars` (default: quotes, backslashes, {}, []).
      4) Strip spaces around , : { } [ ] then collapse whitespace.
    """

    drop_keys = set(drop_keys or ())

    def _round_floats(x: Any) -> Any:
        if isinstance(x, float):
            if math.isnan(x) or math.isinf(x):
                return None
            return round(x, float_decimals)
        if isinstance(x, dict):
            return {k: _round_floats(v) for k, v in x.items() if k not in drop_keys}
        if isinstance(x, list):
            return [_round_floats(v) for v in x]
        try:
            import numpy as np
            if isinstance(x, (np.floating,)):
                fv = float(x)
                return None if (math.isnan(fv) or math.isinf(fv)) else round(fv, float_decimals)
        except Exception:
            pass
        return x

    def _minidumps(x: Any) -> str:
        return json.dumps(x, ensure_ascii=ensure_ascii, separators=(",", ":"))

    def _try_json_minify(s: str) -> Optional[str]:
        s = s.strip()
        # 1) direct parse
        try:
            return _minidumps(_round_floats(json.loads(s)))
        except Exception:
            pass
        # 2) salvage first {...}
        try:
            i, j = s.find("{"), s.rfind("}")
            if i != -1 and j != -1 and j > i:
                return _minidumps(_round_floats(json.loads(s[i:j+1])))
        except Exception:
            pass
        # 3) salvage first [...]
        try:
            i, j = s.find("["), s.rfind("]")
            if i != -1 and j != -1 and j > i:
                return _minidumps(_round_floats(json.loads(s[i:j+1])))
        except Exception:
            pass
        return None

    # --- main compaction ---
    if isinstance(obj, (dict, list)):
        compact = _minidumps(_round_floats(obj))
    elif isinstance(obj, str):
        compact = _try_json_minify(obj)
        if compact is None:
            compact = re.sub(r"\s+", " ", obj.strip())
    else:
        compact = _minidumps(_round_floats(obj))

    # --- post-processing ---
    if remove_escaped_whitespace:
        compact = re.sub(r"\\[nrt]+", " ", compact)  # kill literal \n \r \t

    if strip_punct_spaces:
        compact = re.sub(r"\s*([,\:\{\}\[\]])\s*", r"\1", compact)

    if remove_chars:
        compact = compact.translate(str.maketrans("", "", remove_chars))

    if squash_whitespace:
        compact = re.sub(r"\s+", " ", compact).strip()

    if max_chars and len(compact) > max_chars:
        compact = compact[: max_chars - 3] + "..."

    return compact