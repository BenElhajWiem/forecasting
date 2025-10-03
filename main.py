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
    spec = registry.presets["deepseek-chat"]
    adapter = LLMClientAdapter(spec)

    user_query = "Estimate the electricity demand and RRP in QLD1 on April 10th, 2025 at 7am."

    result = orchestration_agent(user_query=user_query, adapter=adapter)

if __name__ == "__main__":
    main()