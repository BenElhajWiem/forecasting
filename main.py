import os
import logging

os.environ["GRPC_VERBOSITY"] = "ERROR"       # Only errors from gRPC
os.environ["GLOG_minloglevel"] = "3"         # 0=INFO,1=WARNING,2=ERROR,3=FATAL
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"       # Suppress absl info/warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # (optional) silence TensorFlow if imported
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # avoids some C++ log spam

# silence grpc & absl loggers inside Python
logging.getLogger("grpc").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.CRITICAL)

# optional: if absl-py is installed, suppress “pre-init STDERR” warning entirely
try:
    from absl import logging as absl_logging
    absl_logging.use_absl_handler()
    absl_logging.set_verbosity(absl_logging.FATAL)
    absl_logging._warn_preinit_stderr = 0  # type: ignore[attr-defined]
except Exception:
    pass

# -------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from utils.model_registry import Registry, LLMClientAdapter
from agents.orchestration_agent import orchestration_agent

def main():
    registry = Registry()
    spec = registry.presets["deepseek-chat"] # "gemini-flash-native" | "deepseek-chat" | "openai-mini"
    adapter = LLMClientAdapter(spec)

    user_query = "Forecast 48-hour every 2 hours forecast for QLD (QLD1), beginning at July 01, 2025 at 20:30, including TOTALDEMAND and RRP."
    result = orchestration_agent(user_query=user_query, adapter=adapter)

if __name__ == "__main__":
    main()