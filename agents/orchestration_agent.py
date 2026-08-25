from data.data_processing import ElectricityDataLoader
from utils.text_utils import clean_tokens_ultra

from agents.redirecting_agent import classify_horizon, HorizonConfig
from agents.sector_detector import SectorDetector

from agents.timeseries_features import TimeSeriesFilterExtractor
from agents.energy_features import EnergyFilterExtractor

from agents.retrieval import retrieve_context, RetrievalConfig
from agents.summarization import summarize_from_retrieval_strategy, SummarizeConfig, AnomalyConfig
from agents.pattern_detection import detect_patterns_with_llm_after_retrieval, PatternConfig
from agents.statistics_calculation import StatisticalAgent, StatConfig
from agents.forecast_narrative import forecast_with_llm, ForecastConfig
from agents.verification_agent import VerificationAgent, VerificationConfig, format_critique

from utils.model_registry import *

from typing import Any, Dict, Optional

def orchestration_agent(*,
    user_query: str,
    adapter,
    csv_path: str = "data/processed_data.csv",
    retrieval_cfg: Optional[RetrievalConfig] = None,
    summarize_cfg: Optional[SummarizeConfig] = None,
    anomaly_cfg: Optional[AnomalyConfig] = None,
    pattern_cfg: Optional[PatternConfig] = None,
    narrator_cfg: Optional[ForecastConfig] = None,
    forecast_cfg: Optional[ForecastConfig] = None,
    route_cfg: Optional[HorizonConfig] = None,
    # Opt-in feedback loop (EAAI-26-14664 revision, R3.1): OFF by default so
    # existing evaluation runs and published numbers remain reproducible
    # unchanged. See agents/verification_agent.py.
    enable_verification: bool = False,
    verification_cfg: Optional[VerificationConfig] = None,
) -> Dict[str, Any]:
    # defaults
    retrieval_cfg = retrieval_cfg or RetrievalConfig()
    summarize_cfg = summarize_cfg or SummarizeConfig()
    anomaly_cfg   = anomaly_cfg   or AnomalyConfig()
    pattern_cfg   = pattern_cfg   or PatternConfig()
    narrator_cfg  = narrator_cfg  or ForecastConfig()
    forecast_cfg  = forecast_cfg  or ForecastConfig()
    route_cfg     = route_cfg     or HorizonConfig()
    verification_cfg = verification_cfg or VerificationConfig()
    sector_detector = SectorDetector()
    verifier = VerificationAgent(verification_cfg) if enable_verification else None


    # 1) Load & preprocess
    loader = ElectricityDataLoader(csv_path)
    df = loader.load_and_preprocess()

    # 2) Sector classification
    sector = sector_detector.classify(adapter, user_query)
    print("🏷️_____________________Sector classification:", sector)

    # 3) Route classification
    hcfg = HorizonConfig(timestamp_col="SETTLEMENTDATE", reference_time="2025/05/01 00:00:00")
    horizon = classify_horizon(adapter, user_query, df, hcfg)
    print("🧭_____________________Forecast type classification:", horizon)

    # 4) Compute "now" anchor
    now_iso = loader.compute_anchor_now_iso()
    print("⏰_____________________Anchor 'now' timestamp:", now_iso)

    # 5) Feature extraction
    timeseries_agent = TimeSeriesFilterExtractor(adapter)
    timeseries_filters = timeseries_agent.extract(user_query, now_iso=now_iso)
    domain_agent = EnergyFilterExtractor(adapter)
    domain_filters = domain_agent.extract(user_query)

    filters = timeseries_filters | domain_filters
    print("🔍_____________________Extracted filters:", filters)

    # 6) Retrieval
    retrieval_out = retrieve_context(df, filters ,route=horizon, cfg=retrieval_cfg)
    print("📥_____________________Data Retrieved", retrieval_out)

    # +) data prior years same dates
    prior_years_same_dates=retrieval_out["prior_years_same_dates"]
    print("📥_____________________Prior Years Same Dates Retrieved", prior_years_same_dates)

    # 7) Summarization
    summaries = summarize_from_retrieval_strategy(adapter, retrieval_out=retrieval_out, cfg=summarize_cfg, acfg=anomaly_cfg, metrics=filters.get("metrics"))
    #summaries=""
    print("📝_____________________Summarization Done", summaries)

    # 8) Statistics calculation (structured)
    stats_out = StatisticalAgent(StatConfig(tz="Australia/Sydney")).run(retrieval_out)
    print("📊_____________________Statistics calculated", stats_out)

    # 9) Pattern detection (numeric detectors + LLM narrator)
    patterns = detect_patterns_with_llm_after_retrieval(adapter, retrieval_out=retrieval_out)
    print("🧩_____________________Pattern detection Done", patterns)
    #patterns=""

    # 10) Forecast, with an optional Verification Agent feedback loop.
    # Unlike steps 1-9 (a fixed sequential chain), this is a genuine
    # agent-to-agent loop: the Verification Agent reads the Forecast agent's
    # output against the Statistical/Pattern evidence and, if it flags an
    # inconsistency, sends a structured critique back to the Forecast agent
    # for a bounded number of revisions (see agents/verification_agent.py).
    verification_history: list = []
    critique = None
    forecast = forecast_with_llm(
        adapter=adapter, user_query=user_query,
        summary=summaries,
        stats=stats_out,
        patterns=patterns,
        filters=filters,
        cfg=forecast_cfg,
        route=horizon,
        prior_history=prior_years_same_dates,
        )
    print("🤖_____________________Forecast results", forecast)

    if verifier is not None:
        for revision in range(verification_cfg.max_revisions + 1):
            result = verifier.run(adapter, forecast, stats_out, patterns)
            verification_history.append({
                "revision": revision,
                "consistent": result.consistent,
                "issues": result.issues,
            })
            print(f"🔎_____________________Verification (attempt {revision}):",
                  "OK" if result.consistent else result.issues)
            if result.consistent or revision == verification_cfg.max_revisions:
                break
            critique = format_critique(result)
            forecast = forecast_with_llm(
                adapter=adapter, user_query=user_query,
                summary=summaries,
                stats=stats_out,
                patterns=patterns,
                filters=filters,
                cfg=forecast_cfg,
                route=horizon,
                prior_history=prior_years_same_dates,
                critique=critique,
                )
            print(f"🤖_____________________Revised forecast (attempt {revision + 1}):", forecast)

    if verifier is not None:
        return {
            "forecast": forecast,
            "verification_history": verification_history,
            "revisions_used": max(0, len(verification_history) - 1),
        }
    return forecast