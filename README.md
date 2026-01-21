# Forecasting as Reasoning  
**Retrieval-Augmented Multi-Agent LLMs for Interpretable Time Series Forecasting**

This repository contains the reference implementation for the paper:

> **Forecasting as Reasoning: A Retrieval-Augmented Multi-Agent LLM Framework for Interpretable Time Series Forecasting**

<p align="center">
  <img src="assets/architecture.png" width="85%" alt="Overview of the retrieval-augmented multi-agent forecasting framework">
</p>
The system reformulates time series forecasting as a **structured reasoning workflow** over **deterministically retrieved** historical evidence, coordinated by multiple role-specialized LLM agents. The design emphasizes **interpretability, reproducibility, and ablation-driven analysis**, rather than end-to-end opaque prediction.

---

## Core Principles

- **Deterministic DataFrame Retrieval**  
  Historical context is retrieved via structured filters (time, region, metric), ensuring exact numerical grounding and reproducibility.

- **Tool-Augmented Reasoning**  
  All numerical operations (statistics, aggregation) are computed via code, not inferred by the LLM.

- **Multi-Agent Decomposition**  
  Forecasting is decomposed into interpretable agents (feature extraction, retrieval, statistics, summarization, pattern detection, synthesis).

- **Ablation-Ready by Design**  
  Each agent can be independently disabled to quantify its contribution.

---

## Repository Structure
.
├── ablation/
│ ├── Claude/ # Ablation outputs (Claude)
│ ├── Deepseek/ # Ablation outputs (DeepSeek)
│ ├── Gemini/ # Ablation outputs (Gemini)
│ ├── OpenAI/ # Ablation outputs (OpenAI)
│ ├── eval/ # Evaluation utilities
│ ├── queries/ # Query sets used for evaluation
│ ├── stubs/ # Stubs/mocks for controlled experiments
│ ├── utils/ # Ablation helpers
│ └── yamls/ # Experiment configurations
│
├── agents/
│ ├── orchestration_agent.py # Central orchestrator
│ ├── sector_detector.py # Domain detection using LLMs
│ ├── timeseries_features.py # Temporal parsing and filters using LLMs
│ ├── energy_features.py # Energy-domain feature extraction using LLMs
│ ├── retrieval.py # Deterministic DataFrame retrieval
│ ├── statistics_calculation.py # Tool-based statistics
│ ├── summarization.py # Token-efficient summarization using LLMs
│ ├── pattern_detection.py # Trend/seasonality/anomaly detection using LLMs
│ ├── forecast_narrative.py # Forecast synthesis and formatting using LLMs
│ └── redirecting_agent.py # Horizon routing / fallback logic using LLMs
│
├── data/ # Dataset and processed data
|
├── utils/ # Shared utilities
│
├── ablate_model.py # Run ablation for a single configuration
├── ablate_parallel.py # Parallel ablation runs
├── interface.py # Lightweight interactive interface
├── main.py # Main end-to-end entry point
│
├── requirements.txt
├── .env # Environment variables 
└── README.md

---

## Setup

### 1. Environment

python -m venv .venv
source .venv/bin/activate      
pip install -r requirements.txt 

### 2. Environment Variables

Create a .env file at the project root:

OPENAI_API_KEY="..."
DEEPSEEK_API_KEY="..."
GEMINI_API_KEY="..."
ANTHROPIC_API_KEY="..."


## Running the System

### End-to-End Forecasting
python main.py

### Interactive Mode
python interface.py

## Forecasting Workflow (Per Query)

- **Orchestrator** (agent routing, state propagation, deterministic logging, and ablation control)
- Sector detection (optional, multi-domain)
- Horizon classification (short / mid / long)
- Feature extraction (temporal + domain filters)
- Deterministic DataFrame retrieval
- Statistical grounding (tool-computed)
- Summarization (context compression)
- Pattern detection (trend, seasonality, anomalies)
- Forecast synthesis (prediction + rationale)

All intermediate artifacts are explicit and inspectable.


## Ablation Studies

### Single Configuration
python ablate_model.py

### Parallel Ablations
python ablate_parallel.py

### Configuration Files

All experiments are driven by YAML files in:
ablation/yamls/

Each ablation disables an agent or a tool while keeping other components fixed.