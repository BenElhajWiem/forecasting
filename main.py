from utils.model_registry import Registry, LLMClientAdapter
from agents.orchestration_agent import orchestration_agent

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():


    registry = Registry()
    spec = registry.presets["gemini-flash-native"]
    adapter = LLMClientAdapter(spec)

    user_query = "Estimate the electricity demand and RRP in QLD1 on April 10th, 2025 every 2 hours."

    result = orchestration_agent(user_query=user_query, adapter=adapter)

if __name__ == "__main__":
    main()