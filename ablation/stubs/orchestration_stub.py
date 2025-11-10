# ablation/stubs/orchestration_stub.py
# Instrumented orchestration that mirrors the real pipeline but returns a rich TRACE.
from __future__ import annotations
from typing import Any, Dict, Optional, Iterable
import pandas as pd

# --- Tracing helpers ---
from ablation.utils.tracing import Timer, jsonable, topk_by_block, BLOCK_KEYS

# --- Real agents / configs ---
from data.data_processing import ElectricityDataLoader
from agents.redirecting_agent import classify_horizon, HorizonConfig
from agents.sector_detector import SectorDetector

from agents.timeseries_features import TimeSeriesFilterExtractor
from agents.energy_features import EnergyFilterExtractor

from agents.retrieval import retrieve_context, RetrievalConfig
from agents.summarization import summarize_from_retrieval_strategy, SummarizeConfig, AnomalyConfig
from agents.pattern_detection import detect_patterns_with_llm_after_retrieval, PatternConfig
from agents.statistics_calculation import StatisticalAgent, StatConfig
from agents.forecast_narrative import forecast_with_llm, ForecastConfig


# -----------------------------------------
# Optional: block-level filtering for ablations
# -----------------------------------------
def _filter_retrieval_blocks(
    retrieval_out: Dict[str, Any],
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    topk_per_block: Optional[int] = None,
) -> Dict[str, Any]:
    """Return a shallow-copied retrieval_out with 'combined' & 'meta' filtered to chosen blocks."""
    ro = dict(retrieval_out)  # shallow copy ok
    combined: pd.DataFrame = ro.get("combined", pd.DataFrame())

    inc = set(include) if include else set(BLOCK_KEYS)
    exc = set(exclude) if exclude else set()
    keep_blocks = [b for b in BLOCK_KEYS if (b in inc) and (b not in exc)]

    # If nothing to filter or no data, just update meta counts to reflect selection
    if not isinstance(combined, pd.DataFrame) or combined.empty:
        meta = dict(ro.get("meta", {}))
        counts = {k: (int(len(ro.get(k, []))) if (k in keep_blocks) else 0) for k in BLOCK_KEYS}
        meta["counts"] = counts
        meta["total_after_merge"] = 0
        meta["filtered_blocks"] = {"include": sorted(list(inc)), "exclude": sorted(list(exc))}
        ro["meta"] = meta
        return ro

    # Keep only selected blocks
    filtered = combined[combined["ret_block"].isin(keep_blocks)].copy()

    # Optional: per-block top-K by ret_score
    if topk_per_block and "ret_score" in filtered.columns:
        filtered = (
            filtered.sort_values(["ret_block", "ret_score"], ascending=[True, False])
                    .groupby("ret_block", as_index=False, group_keys=False)
                    .head(topk_per_block)
        )

    # Update meta counts
    meta = dict(ro.get("meta", {}))
    counts = {}
    for b in BLOCK_KEYS:
        counts[b] = int(len(filtered[filtered["ret_block"] == b])) if b in keep_blocks else 0

    meta["counts"] = counts
    meta["total_after_merge"] = int(len(filtered))
    meta["filtered_blocks"] = {"include": sorted(list(inc)), "exclude": sorted(list(exc))}
    ro["meta"] = meta
    ro["combined"] = filtered
    return ro


# -----------------------------------------
# Instrumented orchestration (stub)
# -----------------------------------------
def orchestration_agent(*,
    user_query: str,
    adapter,                     # instrumented or vanilla adapter
    csv_path: str = "data/Final_Data_2025.CSV",

    # Configs
    retrieval_cfg: Optional[RetrievalConfig] = None,
    summarize_cfg: Optional[SummarizeConfig] = None,
    anomaly_cfg: Optional[AnomalyConfig] = None,
    pattern_cfg: Optional[PatternConfig] = None,
    narrator_cfg: Optional[ForecastConfig] = None,
    forecast_cfg: Optional[ForecastConfig] = None,
    route_cfg: Optional[HorizonConfig] = None,

    # Tracing options
    return_trace: bool = True,
    trace_topk_per_block: int = 50,

    # Ablation toggles
    use_sector_detector: bool = True,
    use_horizon_classifier: bool = True,
    use_statistics_agent: bool = True,
    use_pattern_agent: bool = True,
    use_summarizer: bool = True,

    # Optional block selection for retrieval ablations
    retrieval_blocks: Optional[Dict[str, Any]] = None,

    # Ensure clean metrics per orchestration call
    reset_adapter_usage_at_start: bool = True,
) -> Dict[str, Any]:
    """
    Returns forecast dict augmented with a TRACE (no metrics pipeline coupling).
    TRACE contains: timings_ms, sector, horizon, filters, retrieval{meta, blocks_topk},
                    patterns (compact), tokens{in,out}, llm_calls[], llm_totals{}.
    """
    # Defaults
    retrieval_cfg = retrieval_cfg or RetrievalConfig()
    summarize_cfg = summarize_cfg or SummarizeConfig()
    anomaly_cfg   = anomaly_cfg   or AnomalyConfig()
    pattern_cfg   = pattern_cfg   or PatternConfig()
    narrator_cfg  = narrator_cfg  or ForecastConfig()
    forecast_cfg  = forecast_cfg  or ForecastConfig()
    route_cfg     = route_cfg     or HorizonConfig()

    # Reset per-call accounting if instrumented adapter is used
    if reset_adapter_usage_at_start and hasattr(adapter, "reset_call_log"):
        try:
            adapter.reset_call_log()
        except Exception:
            pass

    t_all = Timer()
    trace: Dict[str, Any] = {"timings_ms": {}}

    # 1) Load & preprocess
    t = Timer()
    loader = ElectricityDataLoader(csv_path)
    df = loader.load_and_preprocess()
    trace["timings_ms"]["load_preprocess"] = t.ms()

    # 2) Sector classification
    t = Timer()
    sector = "energy"
    if use_sector_detector:
        if hasattr(adapter, "stage"):
            with adapter.stage("sector"):
                sector = SectorDetector().classify(adapter, user_query)
        else:
            sector = SectorDetector().classify(adapter, user_query)
    trace["sector"] = sector
    trace["timings_ms"]["sector"] = t.ms()

    # 3) Horizon classification
    t = Timer()
    hcfg = HorizonConfig(timestamp_col="SETTLEMENTDATE", reference_time="2025/04/01 00:00:00")
    horizon = "short_term"
    if use_horizon_classifier:
        if hasattr(adapter, "stage"):
            with adapter.stage("horizon"):
                horizon = classify_horizon(adapter, user_query, df, hcfg)
        else:
            horizon = classify_horizon(adapter, user_query, df, hcfg)
    trace["horizon"] = horizon
    trace["timings_ms"]["horizon"] = t.ms()

    # 4) Compute "now" anchor
    t = Timer()
    now_iso = loader.compute_anchor_now_iso()
    trace["now_iso"] = now_iso
    trace["timings_ms"]["anchor_now"] = t.ms()

    # 5) Feature extraction
    t = Timer()
    if hasattr(adapter, "stage"):
        with adapter.stage("features_timeseries"):
            timeseries_filters = TimeSeriesFilterExtractor(adapter).extract(user_query, now_iso=now_iso)
        with adapter.stage("features_energy"):
            domain_filters = EnergyFilterExtractor(adapter).extract(user_query)
    else:
        timeseries_filters = TimeSeriesFilterExtractor(adapter).extract(user_query, now_iso=now_iso)
        domain_filters = EnergyFilterExtractor(adapter).extract(user_query)

    filters = {**(timeseries_filters or {}), **(domain_filters or {})}
    trace["filters"] = jsonable(filters)
    trace["timings_ms"]["feature_extract"] = t.ms()

    # 6) Retrieval (no LLM)
    t = Timer()
    retrieval_out = retrieve_context(df, filters, route=horizon, cfg=retrieval_cfg)
    trace["timings_ms"]["retrieval"] = t.ms()

    # Optional: filter retrieval blocks for ablation
    if retrieval_blocks:
        retrieval_out = _filter_retrieval_blocks(
            retrieval_out,
            include=retrieval_blocks.get("include"),
            exclude=retrieval_blocks.get("exclude"),
            topk_per_block=retrieval_blocks.get("topk_per_block"),
        )

    # Save retrieval meta + per-block topK preview
    meta = retrieval_out.get("meta", {}) if isinstance(retrieval_out, dict) else {}
    combined = retrieval_out.get("combined") if isinstance(retrieval_out, dict) else None
    trace["retrieval"] = {
        "meta": jsonable(meta),
        "blocks_topk": topk_by_block(combined, k=trace_topk_per_block),
    }

    # 6b) Summarization (optional)
    t = Timer()
    summaries = None
    if use_summarizer:
        if hasattr(adapter, "stage"):
            with adapter.stage("summarization"):
                summaries = summarize_from_retrieval_strategy(
                    adapter,
                    retrieval_out=retrieval_out,
                    cfg=summarize_cfg,
                    acfg=anomaly_cfg,
                    metrics=filters.get("metrics"),
                )
        else:
            summaries = summarize_from_retrieval_strategy(
                adapter,
                retrieval_out=retrieval_out,
                cfg=summarize_cfg,
                acfg=anomaly_cfg,
                metrics=filters.get("metrics"),
            )
    trace["timings_ms"]["summarization"] = t.ms()

    # 7) Statistics (non-LLM)
    t = Timer()
    stats_out = None
    if use_statistics_agent:
        stats_out = StatisticalAgent(StatConfig(tz="Australia/Sydney")).run(retrieval_out)
    trace["statistics"] = jsonable(getattr(stats_out, "summary", stats_out))
    trace["timings_ms"]["statistics"] = t.ms()

    # 8) Pattern detection (LLM stage if your impl uses LLM)
    t = Timer()
    patterns = None
    if use_pattern_agent:
        if hasattr(adapter, "stage"):
            with adapter.stage("patterns"):
                patterns = detect_patterns_with_llm_after_retrieval(adapter, retrieval_out=retrieval_out)
        else:
            patterns = detect_patterns_with_llm_after_retrieval(adapter, retrieval_out=retrieval_out)

    recent_llm = None
    try:
        recent_llm = patterns["llm_patterns"]["recent_window"]
    except Exception:
        pass
    trace["patterns"] = jsonable({"recent_window": recent_llm}) if patterns else {}
    trace["timings_ms"]["patterns"] = t.ms()

    # 9) Forecast (LLM stage)
    t = Timer()
    if hasattr(adapter, "stage"):
        with adapter.stage("forecast"):
            forecast = forecast_with_llm(
                adapter=adapter,
                user_query=user_query,
                summary=summaries,
                stats=stats_out,
                patterns=patterns,
                filters=filters,
                cfg=forecast_cfg,
                route=horizon,
            )
    else:
        forecast = forecast_with_llm(
            adapter=adapter,
            user_query=user_query,
            summary=summaries,
            stats=stats_out,
            patterns=patterns,
            filters=filters,
            cfg=forecast_cfg,
            route=horizon,
        )
    trace["timings_ms"]["forecast"] = t.ms()
    trace["timings_ms"]["total"] = t_all.ms()

    # Tokens and LLM-call accounting (instrumented adapter only)
    usage = {}
    totals = {}
    calls = []
    if hasattr(adapter, "usage"):
        try: usage = adapter.usage()
        except Exception: pass
    if hasattr(adapter, "totals"):
        try: totals = adapter.totals()
        except Exception: pass
    if hasattr(adapter, "call_log"):
        try: calls = adapter.call_log()
        except Exception: pass

    tin = float(usage.get("in", 0.0))
    tout = float(usage.get("out", 0.0))
    trace["tokens"] = {"in": tin, "out": tout}
    if calls:
        trace["llm_calls"] = calls
    if totals:
        trace["llm_totals"] = totals  # {"tokens_in","tokens_out","cost_usd","latency_sec"}

    # Attach trace and return forecast as-is (no metrics coupling)
    if isinstance(forecast, dict):
        forecast.setdefault("tokens_in", tin)
        forecast.setdefault("tokens_out", tout)
        if return_trace:
        # Only keep the main LLM-generated answer (single value or text)
            main_answer = None
            for k in ("point", "forecast", "answer", "text", "value"):
                if k in forecast:
                    main_answer = forecast[k]
                    break

            trace["forecast"] = jsonable({"answer": main_answer})
            forecast["trace"] = trace
        return forecast

# Fallback if forecaster returned non-dict
    return {
        "tokens_in": tin,
        "tokens_out": tout,
        "trace": trace if return_trace else {},
    }