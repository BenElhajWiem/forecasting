from data.data_processing import ElectricityDataLoader
from utils.text_utils import clean_tokens_ultra

from redirecting_agent import classify_horizon, HorizonConfig
from sector_detector import SectorDetector

from timeseries_features import TimeSeriesFilterExtractor
from energy_features import EnergyFilterExtractor

from retrieval import retrieve_context, RetrievalConfig
from summarization import summarize_from_retrieval_strategy, SummarizeConfig, AnomalyConfig
from pattern_detection import run_pattern_detection_after_retrieval, PatternConfig
from statistics_calculation import generate_statistical_summary
from forecast_narrative import forecast_narrative_from_filters, ForecastConfig

from utils.model_registry import LLMConfig

from typing import Any, Dict, Optional
import pandas as pd
import json

def orchestration_agent(*,
    user_query: str,
    adapter,                     
    csv_path: str = "data/Final_Data_2025.CSV",
    retrieval_cfg: Optional[RetrievalConfig] = None,
    summarize_cfg: Optional[SummarizeConfig] = None,
    anomaly_cfg: Optional[AnomalyConfig] = None,
    pattern_cfg: Optional[PatternConfig] = None,
    narrator_cfg: Optional[LLMConfig] = None,
    forecast_cfg: Optional[ForecastConfig] = None,
    route_cfg: Optional[HorizonConfig] = None,) -> Dict[str, Any]:
    # defaults
    retrieval_cfg = retrieval_cfg or RetrievalConfig()
    summarize_cfg = summarize_cfg or SummarizeConfig()
    anomaly_cfg   = anomaly_cfg   or AnomalyConfig()
    pattern_cfg   = pattern_cfg   or PatternConfig()
    narrator_cfg  = narrator_cfg  or LLMConfig()
    forecast_cfg  = forecast_cfg  or ForecastConfig()
    route_cfg     = route_cfg     or HorizonConfig()
    sector_detector = SectorDetector()

    # 1) Load & preprocess
    loader = ElectricityDataLoader(csv_path)
    df = loader.load_and_preprocess()

    # 2) Sector classification
    sector = sector_detector.classify(adapter, user_query)
    print("🏷️ Sector classification:", sector)

    # 3) Route classification
    hcfg = HorizonConfig(timestamp_col="SETTLEMENTDATE", reference_time="2025/04/01 00:00:00")
    horizon = classify_horizon(adapter, user_query, df, hcfg)
    print("🧭 Forecast type classification:", horizon)

    now_iso = loader.compute_anchor_now_iso()
    timeseries_agent = TimeSeriesFilterExtractor(adapter)
    timeseries_filters = timeseries_agent.extract(user_query, now_iso=now_iso)

    domain_agent = EnergyFilterExtractor(adapter)
    domain_filters = domain_agent.extract(user_query)

    # merge filters (timeseries + domain-specific)
    filters = timeseries_filters | domain_filters

    # 5) Retrieval
    retrieval_out = retrieve_context(df, filters ,route=horizon, cfg=retrieval_cfg)
    print("📥 Data Retrieved")

    # 5) Summarization
    print("📝 Summarizing output")
    summaries = summarize_from_retrieval_strategy(
        adapter,
        retrieval_out=retrieval_out,
        cfg=summarize_cfg,
        acfg=anomaly_cfg,
        metrics=domain_filters.get("metrics"),
    )

    # compact text for LLM-conditioning
    per_origin = summaries.get("bundle", {}).get("per_origin", {})
    cross_headlines = summaries.get("bundle", {}).get("cross_origin", {}).get("headlines", [])
    general_summary = clean_tokens_ultra(cross_headlines, remove_chars='\"\\')

    def _safe_reduce(origin_key: str) -> str:
        blk = per_origin.get(origin_key, {}) or {}
        red = blk.get("reduced", {}) or {}
        return clean_tokens_ultra(red, remove_chars='\"\\')

    # ---- NEW: gather all origins (short + long analogs)
    stats_recent         = _safe_reduce("recent_window")
    stats_pysw           = _safe_reduce("prior_years_same_week")
    stats_pysd           = _safe_reduce("prior_years_same_dates")
    stats_same_hour_prev = _safe_reduce("same_hour_previous_days")   # NEW
    stats_same_woy_prior = _safe_reduce("same_woy_prior_years")      # NEW

    if horizon == "short_term":
      stats_llm = (
          f"stats_recent_window={stats_recent}\n"
          f"stats_same_hour_previous_days={stats_same_hour_prev}\n"   # NEW
          f"stats_prior_years_same_week={stats_pysw}\n"
          f"stats_prior_years_same_dates={stats_pysd}\n"
          f"stats_same_woy_prior_years={stats_same_woy_prior}"        # NEW
      )
    else:
      stats_llm = (
          f"stats_prior_years_same_week={stats_pysw}\n"
          f"stats_prior_years_same_dates={stats_pysd}\n"
          f"stats_same_woy_prior_years={stats_same_woy_prior}\n"      # NEW
          f"stats_same_hour_previous_days={stats_same_hour_prev}"      # (optional) helps long-term for daily cycles
      )

    summary_text = general_summary + stats_llm
    print("summary", summary_text)

    # 6) Pattern detection (numeric detectors + LLM narrator)
    print("🧩 Pattern detection")
    patterns_paragraph = run_pattern_detection_after_retrieval(
        adapter, retrieval_out=retrieval_out, pcfg=pattern_cfg, lcfg=narrator_cfg, return_bundle=False)
    print(patterns_paragraph[:1000])

    # 7) Simple statistics (numeric strings)
    def _safe_stat(df_slice):
        return generate_statistical_summary(df_slice) if isinstance(df_slice, pd.DataFrame) and not df_slice.empty else "No data"
    statistics = {
        "Recent time window statistics": _safe_stat(retrieval_out.get("recent_window")),
        "Same hour previous days statistics": _safe_stat(retrieval_out.get("same_hour_previous_days")),
        "Prior years same week statistics": _safe_stat(retrieval_out.get("prior_years_same_week")),
        "Prior years same dates statistics": _safe_stat(retrieval_out.get("prior_years_same_dates")),
        "Same WOY prior years statistics": _safe_stat(retrieval_out.get("same_woy_prior_years")),
    }
    print("📊 Statistics")

    # 8) Forecast
    forecast_json = forecast_narrative_from_filters(
        adapter=adapter,
        user_query=user_query,
        summary=summary_text,
        stats=statistics,
        patterns=patterns_paragraph,
        filters=filters,
        route=horizon,
        cfg=forecast_cfg
    )
    print(json.dumps(forecast_json, indent=2)[:1200])

    # 9) Return
    return {
        "forecast": forecast_json,
    }
