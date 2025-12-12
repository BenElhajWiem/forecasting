# Agentic Retrieval-Augmented Time Series Forecasting with Large Language Models

## A Modular, Explainable, and Reproducible Framework for Energy Time Series Forecasting
The system decomposes forecasting into specialized, cooperating agents responsible for query understanding, data retrieval, statistical computation, pattern detection, summarization, and forecast generation. This agentic formulation enables explicit reasoning steps, improved explainability, and systematic ablation, while remaining model-agnostic and reproducible.

Although demonstrated on electricity demand and price forecasting, the framework is domain-independent and applicable to general structured temporal data.

### Contributions

This work makes the following contributions:

**1- Agentic Decomposition of Forecasting**
Forecasting is formalized as a sequence of modular reasoning stages, each handled by a dedicated agent.

**2- Retrieval-Conditioned Temporal Reasoning**
Historical data is retrieved dynamically using structured temporal, regional, and horizon-aware constraints.

**3- Explainability-First Pipeline**
Statistical summaries and detected temporal patterns are explicitly computed prior to forecast generation.

**4- Reproducible and Ablation-Friendly Design**
The framework supports controlled ablations, multi-model benchmarking, and repeatable evaluation.

**5- LLM-Provider Agnostic Architecture**
Multiple LLM backends can be evaluated under identical experimental conditions.
