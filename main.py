import logging   

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from utils.model_registry import Registry, LLMClientAdapter
from agents.orchestration_agent import orchestration_agent

def main():

    logger.info("_____________________Starting forecasting_____________________")

    registry = Registry()
    spec = registry.presets["deepseek-chat"] # "gemini-flash-native" | "deepseek-chat" | "openai-mini" | "gemini-flash" | "claude-api"
    adapter = LLMClientAdapter(spec)

    user_query = "Generate a 24-hour forecast at 2-hour intervals for Queensland, starting on September 22, 2025 at 2PM, including electricity demand and market price."
    result = orchestration_agent(user_query=user_query, adapter=adapter)

    logger.info("_____________________Orchestration completed_____________________")

if __name__ == "__main__":
    main()