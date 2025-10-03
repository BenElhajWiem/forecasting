from data.data_processing import ElectricityDataLoader
# from utils.text_utils import clean_tokens_ultra

from agents.redirecting_agent import classify_horizon, HorizonConfig
from agents.sector_detector import SectorDetector

from agents.timeseries_features import TimeSeriesFilterExtractor
from agents.energy_features import EnergyFilterExtractor

from agents.retrieval import retrieve_context, RetrievalConfig
#from agents.summarization import summarize_from_retrieval_strategy, SummarizeConfig, AnomalyConfig
from agents.pattern_detection import detect_patterns_with_llm_after_retrieval, PatternConfig
from agents.statistics_calculation import StatisticalAgent, StatConfig 
from agents.forecast_narrative import forecast_with_llm, ForecastConfig

from utils.model_registry import *

from typing import Any, Dict, Optional

def orchestration_agent(*,
    user_query: str,
    adapter,                     
    csv_path: str = "data/Final_Data_2025.CSV",
    retrieval_cfg: Optional[RetrievalConfig] = None,
    # summarize_cfg: Optional[SummarizeConfig] = None,
    # anomaly_cfg: Optional[AnomalyConfig] = None,
    pattern_cfg: Optional[PatternConfig] = None,
    narrator_cfg: Optional[ForecastConfig] = None,
    forecast_cfg: Optional[ForecastConfig] = None,
    route_cfg: Optional[HorizonConfig] = None,) -> Dict[str, Any]:
    # defaults
    retrieval_cfg = retrieval_cfg or RetrievalConfig()
    # summarize_cfg = summarize_cfg or SummarizeConfig()
    # anomaly_cfg   = anomaly_cfg   or AnomalyConfig()
    pattern_cfg   = pattern_cfg   or PatternConfig()
    narrator_cfg  = narrator_cfg  or ForecastConfig()
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

    # 4) Compute "now" anchor
    now_iso = loader.compute_anchor_now_iso()
    print("⏰ Anchor 'now' timestamp:", now_iso)

    # 5) Feature extraction
    timeseries_agent = TimeSeriesFilterExtractor(adapter)
    timeseries_filters = timeseries_agent.extract(user_query, now_iso=now_iso)
    domain_agent = EnergyFilterExtractor(adapter)
    domain_filters = domain_agent.extract(user_query)

    filters = timeseries_filters | domain_filters
    print("🔍 Extracted filters:", filters)

    # 6) Retrieval
    retrieval_out = retrieve_context(df, filters ,route=horizon, cfg=retrieval_cfg)
    print("📥 Data Retrieved", retrieval_out)

    # 7) Statistics calculation (structured)
    stats_out = StatisticalAgent(StatConfig(tz="Australia/Sydney")).run(retrieval_out)
    print("📊 Statistics calculated", stats_out)

    # 8) Pattern detection (numeric detectors + LLM narrator)
    patterns = detect_patterns_with_llm_after_retrieval(adapter, retrieval_out=retrieval_out)
    print("🧩 Pattern detection",patterns["llm_patterns"]["recent_window"])

    # 8) Forecast
    forecast = forecast_with_llm (
        adapter=adapter, user_query=user_query,
        summary=None,
        stats=stats_out,
        patterns=patterns,
        filters=filters,
        cfg=forecast_cfg,
        route=horizon,
        )
    print("🤖 Forecast results", forecast)
    return forecast
