import logging   

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from utils.model_registry import Registry, LLMClientAdapter
from agents.orchestration_agent import orchestration_agent

def main():

    logger.info("_____________________Starting forecasting_____________________")

    registry = Registry()
    spec = registry.presets["gemini-flash-native"] # "gemini-flash-native" | "deepseek-chat" | "openai-mini" | "gemini-flash" | "claude-api"
    adapter = LLMClientAdapter(spec)

    user_query = "Forecast 48-hour every 2 hours forecast for QLD, beginning at September 22, 2025 at 2am, including TOTALDEMAND and RRP."
    result = orchestration_agent(user_query=user_query, adapter=adapter)

    logger.info("_____________________Orchestration completed_____________________")

if __name__ == "__main__":
    main()