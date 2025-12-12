from __future__ import annotations
from typing import Any, Dict, Optional
import pandas as pd
import time, random 

# --- Tracing helpers ---
from ablation.utils.tracing import Timer, jsonable, topk_by_block

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
# Helper: retry LLM stages until success
# -----------------------------------------
def call_llm_stage_until_success(
    stage_name: str,
    fn,
    max_backoff: float = 60.0,
):
    """
    Repeatedly execute `fn()` (an LLM stage) until it succeeds.
    Retries only on transient API issues: timeouts, DeadlineExceeded,
    503/504, "overloaded", etc.

    Each *internal* LLM call should still have its own per-call timeout
    configured inside the adapter.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            transient = any(
                key in msg
                for key in (
                    "timeout",
                    "deadline",
                    "504",
                    "503",
                    "overloaded",
                    "temporarily unavailable",
                    "server error",
                )
            )
            if not transient:
                # Real bug / validation error: don't spin forever
                raise

            sleep = min(max_backoff, 2 * attempt) + random.random()
            print(
                f"[orchestration] transient LLM error in '{stage_name}' "
                f"(attempt {attempt}): {e} – retrying in {sleep:.1f}s..."
            )
            time.sleep(sleep)
            # loop continues

# -----------------------------------------
# Instrumented orchestration 
# -----------------------------------------
def orchestration_agent(*,
    user_query: str,
    adapter,                     
    csv_path: str = "data/processed_data.csv",

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
    direct_forecast_only: bool = False,

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
    trace: Dict[str, Any] = {"timings_ms": {}, "mode": "full_pipeline"}

    # -------------------------------------------------
    # Direct-forecast-only ablation:
    # -------------------------------------------------

    if direct_forecast_only:
        trace["mode"] = "direct_forecast_only"
        trace["sector"] = None
        trace["horizon"] = "direct"
        trace["now_iso"] = None
        trace["filters"] = jsonable({})
        trace["retrieval"] = {
            "meta": jsonable({}),
            "blocks_topk": {},
        }
        trace["statistics"] = jsonable(None)
        trace["patterns"] = jsonable({})

        # Forecast only
        t = Timer()
        if hasattr(adapter, "stage"):
            with adapter.stage("forecast"):
                forecast = forecast_with_llm(
                    adapter=adapter,
                    user_query=user_query,
                    summary=None,
                    stats=None,
                    patterns=None,
                    filters={},
                    cfg=forecast_cfg,
                    route=None,
                )
        else:
            forecast = forecast_with_llm(
                adapter=adapter,
                user_query=user_query,
                summary=None,
                stats=None,
                patterns=None,
                filters={},
                cfg=forecast_cfg,
                route=None,
            )
        trace["timings_ms"]["forecast"] = t.ms()
        trace["timings_ms"]["total"] = t_all.ms()

        # Tokens and LLM-call accounting
        usage: Dict[str, Any] = {}
        totals: Dict[str, Any] = {}
        calls: list = []

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

        # Attach trace and return forecast
        if isinstance(forecast, dict):
            forecast.setdefault("tokens_in", tin)
            forecast.setdefault("tokens_out", tout)
            if return_trace:
                main_answer = None
                for k in ("point", "forecast", "answer", "text", "value"):
                    v = forecast.get(k)
                    if v is not None:
                        main_answer = v
                        break
                forecast["llm_answer"] = main_answer
                if return_trace:
                    trace["forecast"] = jsonable({"answer": main_answer})
                    forecast["trace"] = trace
                return forecast

        return {
            "tokens_in": tin,
            "tokens_out": tout,
            "forecast": forecast,
            "trace": trace if return_trace else {},
        }
    
    # =========================================================
    # FULL PIPELINE PATH (normal behaviour with toggles)
    # =========================================================
    # 1) Load & preprocess
    t = Timer()
    loader = ElectricityDataLoader(csv_path)
    df = loader.load_and_preprocess()
    trace["timings_ms"]["load_preprocess"] = t.ms()

    # 2) Sector classification
    t = Timer()
    sector = "energy"
    if use_sector_detector:
        def _sector():
            return SectorDetector().classify(adapter, user_query)

        if hasattr(adapter, "stage"):
            with adapter.stage("sector"):
                sector = call_llm_stage_until_success("sector", _sector)
        else:
            sector = call_llm_stage_until_success("sector", _sector)

    trace["sector"] = sector
    trace["timings_ms"]["sector"] = t.ms()

    # 3) Horizon classification
    t = Timer()
    hcfg = HorizonConfig(timestamp_col="SETTLEMENTDATE", reference_time="2025/04/01 00:00:00")
    horizon = "long_term"
    if use_horizon_classifier:
        def _horizon():
            return classify_horizon(adapter, user_query, df, hcfg)

        if hasattr(adapter, "stage"):
            with adapter.stage("horizon"):
                horizon = call_llm_stage_until_success("horizon", _horizon)
        else:
            horizon = call_llm_stage_until_success("horizon", _horizon)

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
            timeseries_filters = call_llm_stage_until_success(
                "features_timeseries",
                lambda: TimeSeriesFilterExtractor(adapter).extract(user_query, now_iso=now_iso),
            )
        with adapter.stage("features_energy"):
            domain_filters = call_llm_stage_until_success(
                "features_energy",
                lambda: EnergyFilterExtractor(adapter).extract(user_query),
            )
    else:
        timeseries_filters = call_llm_stage_until_success(
            "features_timeseries",
            lambda: TimeSeriesFilterExtractor(adapter).extract(user_query, now_iso=now_iso),
        )
        domain_filters = call_llm_stage_until_success(
            "features_energy",
            lambda: EnergyFilterExtractor(adapter).extract(user_query),
        )

    filters = {**(timeseries_filters or {}), **(domain_filters or {})}
    trace["filters"] = jsonable(filters)
    trace["timings_ms"]["feature_extract"] = t.ms()
    print("🔍_____________________Extracted filters:")

    # 6) Retrieval (no LLM)
    t = Timer()
    retrieval_out = retrieve_context(df, filters, route=horizon, cfg=retrieval_cfg)
    trace["timings_ms"]["retrieval"] = t.ms()

    # Save retrieval meta + per-block topK preview
    meta = retrieval_out.get("meta", {}) if isinstance(retrieval_out, dict) else {}
    combined = retrieval_out.get("combined") if isinstance(retrieval_out, dict) else None
    trace["retrieval"] = {
        "meta": jsonable(meta),
        "blocks_topk": topk_by_block(combined, k=trace_topk_per_block),
    }

    # +) data prior years same dates
    prior_years_same_dates=retrieval_out["prior_years_same_dates"]
    print("📥_____________________Prior Years Same Dates Retrieved", prior_years_same_dates)

    # 7) Summarization  (LLM stage)
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
    print("📝_____________________Summarization Done")

    # 8) Statistics (non-LLM)
    t = Timer()
    stats_out = None
    if use_statistics_agent:
        stats_out = StatisticalAgent(StatConfig(tz="Australia/Sydney")).run(retrieval_out)
    trace["statistics"] = jsonable(getattr(stats_out, "summary", stats_out))
    trace["timings_ms"]["statistics"] = t.ms()
    print("📊_____________________Statistics calculated")

    # 9) Pattern detection (LLM stage)
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

    print("🧩_____________________Pattern detection Done")

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
                prior_history=prior_years_same_dates,
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
            prior_history=prior_years_same_dates,
        )
    trace["timings_ms"]["forecast"] = t.ms()
    trace["timings_ms"]["total"] = t_all.ms()

    # Tokens and LLM-call accounting (instrumented adapter only)
    usage: Dict[str, Any] = {}
    totals: Dict[str, Any] = {}
    calls: list = []

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

        main_answer = None
        for k in ("point", "forecast", "answer", "text", "value"):
            v = forecast.get(k)
            if v is not None:
                main_answer = v
                break
        
        forecast["llm_answer"] = main_answer

        if return_trace:
            trace["forecast"] = jsonable({"answer": main_answer})
            forecast["trace"] = trace
        return forecast

# Fallback if forecaster returned non-dict
    return {
        "tokens_in": tin,
        "tokens_out": tout,
        "forecast": forecast,
        "trace": trace if return_trace else {},
    }