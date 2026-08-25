# Cost estimation utilities for LLM API calls.
#
# NOTE (revision, 2026-08): the gemini-2.5-flash input rate below was 0.025/1k
# (~$25 per 1M tokens) -- about 80x the published rate and higher than its own
# output rate, which is backwards for every major provider. Corrected against
# Google's published Gemini 2.5 Flash pricing ($0.30/1M input, $2.50/1M output;
# ai.google.dev/gemini-api/docs/pricing, checked 2026-08-04). This is very
# likely the root cause of the inflated per-run cost Reviewer #3 flagged
# (Gemini $15.63/run vs Claude $4.16/run). Only the Gemini rate is changed
# here -- the other three providers' rates were not flagged as anomalous and
# are left as originally recorded.

PRICES = {
    "gpt-4o-mini":       {"input_per_1k": 0.00015,   "output_per_1k": 0.00066},
    "deepseek-chat":     {"input_per_1k": 0.00028,   "output_per_1k": 0.00028},  # placeholder
    "gemini-2.5-flash":  {"input_per_1k": 0.0003,    "output_per_1k": 0.0025},
    "claude-sonnet-4-5": {"input_per_1k": 0.003,     "output_per_1k": 0.015},
}

ALIASES = {
    "openai-mini":          "gpt-4o-mini",
    "deepseek-chat":        "deepseek-chat",
    "gemini-flash-native":  "gemini-2.5-flash",
    "claude-api":           "claude-sonnet-4-5",
}

def _price_key(model_or_preset: str) -> str:
    """Resolve a registry preset key or model name to a PRICES key."""
    return ALIASES.get(model_or_preset, model_or_preset)

def estimate_cost(model_or_preset: str, tokens_in: float, tokens_out: float) -> float:
    """
    Estimate USD cost for a single call.
    """
    key = _price_key(model_or_preset)
    p = PRICES.get(key, {"input_per_1k": 0.0, "output_per_1k": 0.0})
    return (tokens_in / 1000.0) * p["input_per_1k"] + (tokens_out / 1000.0) * p["output_per_1k"]

# (Optional) batch helper
def estimate_batch_cost(calls: list[dict]) -> float:
    """
    calls: [{"model": "openai-mini", "in": 3200, "out": 900}, ...]
    """
    return sum(estimate_cost(c["model"], c["in"], c["out"]) for c in calls)
