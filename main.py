from utils.model_registry import Registry, LLMClientAdapter
from agents.orchestration_agent import orchestration_agent

import os
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():

    vis = {k: bool(os.getenv(k)) for k in ["OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY"]}
    logger.info(f"API key presence: {vis}")
    registry = Registry()
    spec = registry.presets["openai-mini"]
    adapter = LLMClientAdapter(spec)

    user_query = "Predict the TOTALDEMAND and RRP for NSW1 on April the 1st, 2025 at 12:00."

    result = orchestration_agent(user_query=user_query, adapter=adapter)
    print(result)

if __name__ == "__main__":
    main()