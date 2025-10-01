from __future__ import annotations

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from dataclasses import dataclass
import os, json, time, logging
from typing import Optional, Dict, Any, List

from openai import OpenAI
from openai import BadRequestError 

logger = logging.getLogger(__name__)

# ---------- Model registry ----------
@dataclass
class ModelSpec:
    provider: str                
    model: str
    api_key: str
    base_url: Optional[str] = None
    supports_response_format: bool = True

class Registry:
    def __init__(self):
        self.presets = {
            # OpenAI (native)
            "openai-mini": ModelSpec(
                provider="openai",
                model="gpt-4o-mini",
                api_key=os.getenv("OPENAI_API_KEY"),   # from .env
                base_url=None,
                supports_response_format=True
            ),
            # DeepSeek
            "deepseek-chat": ModelSpec(
                provider="deepseek",
                model="deepseek-chat",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
                supports_response_format=False
            ),
            # Google Gemini
            "gemini-flash": ModelSpec(
                provider="gemini-openai",
                model="gemini-2.5-flash",
                api_key=os.getenv("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                supports_response_format=True
            ),
        }
    def get(self, name: str) -> ModelSpec:
        if name not in self.presets:
            raise KeyError(f"Unknown preset: {name}")
        return self.presets[name]

registry = Registry()

# ---------- Adapter ----------
class LLMClientAdapter:
    """
    Single adapter for OpenAI-compatible chat APIs (OpenAI, DeepSeek, Gemini OpenAI proxy).
    - Accepts response_format but strips it for providers that don't support it.
    - Retries transient errors; auto-falls back if server rejects JSON mode.
    """

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        api_key = spec.api_key
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {spec.api_key}")
        self.client = OpenAI(api_key=api_key, base_url=spec.base_url) if spec.base_url else OpenAI(api_key=api_key)

    def _retry_chat(self, **kwargs) -> Any:
        """
        Core send with small retry loop and auto-fallback for 'response_format' rejection.
        """
        # Provider-level strip (e.g., DeepSeek)
        if not self.spec.supports_response_format and "response_format" in kwargs:
            kwargs.pop("response_format", None)

        last_err: Optional[Exception] = None

        # Attempt + targeted fallback if server complains about response_format
        for attempt in range(3):
            try:
                return self.client.chat.completions.create(**kwargs)
            except BadRequestError as e:
                msg = str(e)
                # Example error seen: "Invalid parameter: 'response_format' is not supported with this model."
                if "response_format" in msg or "not supported with this model" in msg:
                    if "response_format" in kwargs:
                        rf = kwargs.pop("response_format", None)
                        logger.warning(f"[Adapter] Server rejected response_format; retrying without it. Removed={rf}")
                        # Immediate retry once without counting against attempts
                        try:
                            return self.client.chat.completions.create(**kwargs)
                        except Exception as e2:
                            last_err = e2
                            time.sleep(0.8)
                            continue
                last_err = e
            except Exception as e:
                last_err = e
            time.sleep(0.8)
        assert last_err is not None
        raise last_err

    def chat(
      self,
      messages: List[Dict[str, str]],
      *,
      temperature: float = 0.0,
      max_tokens: Optional[int] = None,
      model_override: Optional[str] = None,
      response_format: Optional[Dict[str, Any]] = None,
      extra_params: Optional[Dict[str, Any]] = None,
      **kwargs: Any,  # Accept unknown keywords to avoid TypeError in callers
  ) -> str:
      """
      Returns assistant message text (string).
      - If provider doesn't support JSON mode, silently ignores `response_format`.
      - If callers pass `response_format` via kwargs, we handle/strip it too.
      - If a TypeError still happens mentioning response_format, we retry once without it.
      """
      params: Dict[str, Any] = {
          "model": (model_override or self.spec.model),
          "messages": messages,
          "temperature": temperature,
      }
      if max_tokens is not None:
          params["max_tokens"] = max_tokens

      # Collect any response_format coming from either explicit arg or kwargs
      rf = response_format if response_format is not None else kwargs.pop("response_format", None)

      # Only include response_format if this provider supports it
      if rf is not None and getattr(self.spec, "supports_response_format", True):
          params["response_format"] = rf  # _retry_chat will also defend/retry

      if extra_params:
          params.update(extra_params)
      if kwargs:
          params.update(kwargs)

      try:
          resp = self._retry_chat(**params)
      except TypeError as e:
          # Some older/stale adapters or proxies might still choke on response_format.
          if "response_format" in str(e):
              params.pop("response_format", None)
              resp = self._retry_chat(**params)
          else:
              raise

      return (resp.choices[0].message.content or "").strip()

    def chat_json_loose(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        model_override: Optional[str] = None,
        strict_json_first: bool = False,  # default False: never attempts JSON mode
    ) -> Dict[str, Any]:
        """
        Provider-agnostic loose JSON parse.
        IMPORTANT: Never passes `response_format` (avoids signature/compat issues).
        """
        txt = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model_override=model_override,)
        return _json_loads_loose(txt)


    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        model_override: Optional[str] = None,
        strict_json_first: bool = False,
    ) -> Dict[str, Any]:
        out = self.chat_json_loose(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model_override=model_override,
            strict_json_first=strict_json_first,
        )
        if not isinstance(out, dict):
            raise ValueError("Expected JSON object from model.")
        return out

# ---------- JSON salvage helper ----------
def _json_loads_loose(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            i = s.index("{")
            j = s.rindex("}") + 1
            return json.loads(s[i:j])
        except Exception:
            return {"text": s[:2000]}