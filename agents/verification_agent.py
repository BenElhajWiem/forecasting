"""
Verification Agent: closes the loop between the Forecast agent and the
upstream evidence-producing agents (Statistical, Pattern Detection).

Motivation (EAAI-26-14664 revision, Reviewer #3.1): the original orchestration
pipeline was a strictly sequential call chain (sector -> horizon -> features ->
retrieval -> summarize -> stats -> patterns -> forecast) with no agent ever
reading or revising another agent's output. This module adds a genuine
agent-to-agent feedback loop: the Verification Agent checks a candidate
forecast against the Statistical Agent's evidence and the Pattern Detection
Agent's labels, and -- if it finds the forecast inconsistent with the
evidence -- returns a structured critique that the orchestrator feeds back
into a bounded number of Forecast agent revision attempts.

This is intentionally conservative in scope: it does not change the Forecast
agent's prompt contract for the (default, unchanged) non-verified path, and
it is opt-in via `enable_verification=False` by default in
orchestration_agent, so existing evaluation results and the published tables
remain reproducible unless verification is explicitly turned on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import re


@dataclass
class VerificationConfig:
    max_revisions: int = 2
    # Numeric plausibility: flag a forecast value more than this many
    # standard deviations from the recent-window mean for its metric.
    outlier_z_threshold: float = 4.0
    metrics: Tuple[str, ...] = ("TOTALDEMAND", "RRP")
    # LLM semantic check (does the narrative's stated direction/rationale
    # match the Pattern Detection Agent's trend/seasonality labels).
    use_llm_semantic_check: bool = True
    temperature: float = 0.0
    max_tokens: int = 300
    model: Optional[str] = None
    model_override: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Best-effort numeric extraction from the Forecast agent's free-text/markdown
# narrative (see forecast_narrative.py's OUTPUT STYLE contract). This is a
# heuristic used only to gate the plausibility check below -- it is NOT used
# as the authoritative evaluation parser (that lives in the eval pipeline).
# ─────────────────────────────────────────────────────────────────────────────

_NUM_RE = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*\.?\d*")
# Numbers immediately followed by a known output unit (see forecast_narrative.py's
# ForecastConfig.units_map) are far more likely to be the actual forecast value
# than an arbitrary earlier number in the sentence (a date, an hour, or a digit
# trailing a region code like "NSW1" -- the (?<![A-Za-z]) above only excludes the
# latter; a plain first-number-in-line heuristic still grabs dates/times first).
_UNIT_NUM_RE = re.compile(r"(?<![A-Za-z])([-+]?\d[\d,]*\.?\d*)\s*(?:MW|\$/MWh|\$|MWh)\b(?!/)")
# The (?!/) above excludes rate units like "MW/h" or "MW/hour" (trend slopes
# quoted in the rationale, e.g. "slope -0.423 MW/h") from matching as if they
# were absolute forecast values -- observed: a slope of -0.423 MW/h was
# extracted as a 0.423 MW forecast and flagged as a bogus outlier.
# The Forecast agent's prompt (forecast_narrative.py) does not fix a single
# output template, so a bolded "**Forecast:** <value>" label line -- observed
# in practice -- often carries the value without repeating the metric name on
# that same line, which the metric+line joint search below would otherwise miss.
_LABELED_LINE_RE = re.compile(r"\*{0,2}forecast\*{0,2}\s*:", re.IGNORECASE)
# Forecast narratives routinely restate the target date/timestamp on the same
# line as the metric name (e.g. an "Overview: ... TOTALDEMAND ... 2025-06-19
# 23:30 ..." sentence) -- stripped out before numeric search so the
# fallback branch doesn't mistake a calendar date for the forecast value.
_ISO_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?")


def extract_forecast_values(narrative: str, metric: str) -> List[float]:
    values: List[float] = []
    if not narrative:
        return values
    if metric.upper() not in narrative.upper():
        # The generic "**Forecast:**" label (_LABELED_LINE_RE) is metric-agnostic,
        # so without this guard a narrative that forecasts only one of cfg.metrics
        # (e.g. a TOTALDEMAND-only query) has its single value attributed to every
        # other metric being checked too (observed: a TOTALDEMAND-only narrative's
        # "**Forecast:** 8,900 MW" line was picked up as an RRP value of 8900,
        # flagged as a 24.7-std outlier against an RRP mean of ~92).
        return values
    metric_colon_re = re.compile(rf"\*{{0,2}}{re.escape(metric)}\*{{0,2}}\s*:", re.IGNORECASE)
    for line in narrative.splitlines():
        is_table_row = "|" in line
        # A free-text line only counts as a "metric line" when the metric name
        # actually labels a value (e.g. "**TOTALDEMAND:** 5,240 MW"), not merely
        # mentioned in passing (e.g. an Overview sentence naming the metrics
        # being forecast) -- otherwise the bare-number fallback below grabs the
        # first unrelated digit on that line (observed: "a single 5-minute
        # trading interval" produced a spurious forecast value of 5).
        is_metric_line = (metric.upper() in line.upper()) if is_table_row else bool(metric_colon_re.search(line))
        is_labeled_line = bool(_LABELED_LINE_RE.search(line))
        if not (is_metric_line or is_labeled_line):
            continue
        cells = [c.strip() for c in line.split("|") if c.strip()] if "|" in line else [line]
        for cell in cells:
            clean = _ISO_TS_RE.sub(" ", cell).replace(",", "")
            unit_nums = _UNIT_NUM_RE.findall(clean)
            nums = unit_nums if unit_nums else (_NUM_RE.findall(clean) if is_metric_line else [])
            if nums:
                try:
                    values.append(float(nums[0]))
                except ValueError:
                    pass
                break
    return values


def _find_metric_stats(stats_out: Dict[str, Any], metric: str) -> Optional[Dict[str, Any]]:
    """Prefer recent_window stats for `metric`; fall back to any origin that has it."""
    per_origin = (stats_out or {}).get("per_origin", {}) or {}
    for preferred in ("recent_window", "same_hour_previous_days", "same_weekday_recent_weeks"):
        block = per_origin.get(preferred, {})
        s = block.get("stats", {}).get(metric)
        if s and s.get("count", 0) > 1:
            return s
    for block in per_origin.values():
        s = block.get("stats", {}).get(metric)
        if s and s.get("count", 0) > 1:
            return s
    return None


def check_numeric_plausibility(
    narrative: str,
    stats_out: Dict[str, Any],
    cfg: VerificationConfig,
) -> Tuple[bool, List[str]]:
    """Flag forecast values that are extreme outliers relative to retrieved evidence."""
    issues: List[str] = []
    for metric in cfg.metrics:
        values = extract_forecast_values(narrative, metric)
        if not values:
            continue
        mstats = _find_metric_stats(stats_out, metric)
        if not mstats or not mstats.get("std"):
            continue
        mean, std = mstats["mean"], mstats["std"]
        if std <= 0:
            continue
        for v in values:
            z = abs(v - mean) / std
            if z > cfg.outlier_z_threshold:
                issues.append(
                    f"{metric} forecast value {v:g} is {z:.1f} std from the retrieved "
                    f"recent-window mean ({mean:.2f} +/- {std:.2f}); this exceeds the "
                    f"plausibility threshold ({cfg.outlier_z_threshold:.1f} std) without "
                    f"an explicit justification in the rationale."
                )
    return (len(issues) == 0, issues)


def _extract_trend_labels(patterns: Any) -> List[Dict[str, Any]]:
    """Pull {region, metric, trend.direction, seasonality} labels out of the
    Pattern Detection Agent's bundle (see pattern_detection.py's schema)."""
    out: List[Dict[str, Any]] = []
    if not isinstance(patterns, dict):
        return out
    llm_patterns = patterns.get("llm_patterns", {}) or {}
    for _, labeled in llm_patterns.items():
        if not isinstance(labeled, dict):
            continue
        for summary in labeled.get("summaries", []) or []:
            if isinstance(summary, dict):
                out.append(summary)
    return out


def check_semantic_consistency(
    adapter,
    narrative: str,
    patterns: Any,
    cfg: VerificationConfig,
) -> Tuple[bool, List[str]]:
    """Ask the LLM whether the forecast narrative's stated direction/rationale
    is consistent with the Pattern Detection Agent's trend/seasonality labels.
    Mirrors the schema-constrained JSON convention used elsewhere in the
    pipeline (see pattern_detection.py::ask_llm_to_label_patterns)."""
    trend_labels = _extract_trend_labels(patterns)
    if not trend_labels:
        return True, []

    system = (
        "You are a consistency checker for a time-series forecasting pipeline. "
        "You will be given (a) detected trend/seasonality labels computed from "
        "historical data, and (b) a candidate forecast narrative. "
        "Decide whether the narrative's direction and reasoning plausibly follow "
        "from the detected evidence. Do not re-derive the forecast yourself. "
        "Return JSON only: {\"consistent\": bool, \"issues\": [string, ...]}."
    )
    import json as _json
    user = (
        f"DETECTED PATTERN LABELS:\n{_json.dumps(trend_labels, ensure_ascii=False)[:6000]}\n\n"
        f"CANDIDATE FORECAST NARRATIVE:\n{narrative[:4000]}\n"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    try:
        result = adapter.chat_json_loose(
            messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            model_override=(cfg.model_override or cfg.model),
            strict_json_first=True,
        )
    except Exception as exc:
        return True, [f"semantic check skipped (LLM error: {exc})"]

    if not isinstance(result, dict) or "consistent" not in result:
        return True, ["semantic check skipped (unparseable verifier output)"]

    consistent = bool(result.get("consistent", True))
    issues = [str(x) for x in (result.get("issues") or [])]
    return consistent, issues


@dataclass
class VerificationResult:
    consistent: bool
    issues: List[str] = field(default_factory=list)
    checked_numeric: bool = True
    checked_semantic: bool = False


class VerificationAgent:
    """Checks a candidate forecast against upstream evidence and produces a
    critique the Forecast agent can act on. See module docstring."""

    def __init__(self, cfg: VerificationConfig = VerificationConfig()):
        self.cfg = cfg

    def run(
        self,
        adapter,
        narrative: str,
        stats_out: Dict[str, Any],
        patterns: Any,
    ) -> VerificationResult:
        num_ok, num_issues = check_numeric_plausibility(narrative, stats_out, self.cfg)

        sem_ok, sem_issues = True, []
        checked_semantic = False
        if self.cfg.use_llm_semantic_check:
            sem_ok, sem_issues = check_semantic_consistency(adapter, narrative, patterns, self.cfg)
            checked_semantic = True

        issues = num_issues + sem_issues
        return VerificationResult(
            consistent=(num_ok and sem_ok),
            issues=issues,
            checked_numeric=True,
            checked_semantic=checked_semantic,
        )


def format_critique(result: VerificationResult) -> str:
    return "\n".join(f"- {issue}" for issue in result.issues)
