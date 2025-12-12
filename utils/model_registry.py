from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import os, json, time, logging
logger = logging.getLogger(__name__)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from openai import OpenAI
from openai import BadRequestError 
import google.generativeai as genai 

# -------------------------
# Registry
# -------------------------
@dataclass
class ModelSpec:
    provider: str                
    model: str
    api_key: str
    base_url: Optional[str] = None
    supports_response_format: bool = True
    extra_body: Optional[Dict[str, Any]] = None
    reasoning_effort: Optional[str] = None  
    sdk: str = "openai" 

class Registry:
    def __init__(self):
        self.presets = {
            # OpenAI
            "openai-mini": ModelSpec(
                provider="openai",
                model="gpt-4o-mini",
                api_key=os.getenv("OPENAI_API_KEY"),   # from .env
                base_url=None,
                supports_response_format=True,
                sdk="openai",
            ),
            # OpenAI
            "openai-gpt5-pro": ModelSpec(
                provider="openai",
                model="gpt-5-pro",
                api_key=os.getenv("OPENAI_API_KEY"),   # from .env
                base_url=None,
                supports_response_format=True,
                sdk="openai",
            ),
            # DeepSeek
            "deepseek-chat": ModelSpec(
                provider="deepseek",
                model="deepseek-chat",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
                supports_response_format=False,
                sdk="openai",
            ),
            
            # Google Gemini OPENAI-API
            "gemini-flash": ModelSpec(
                provider="gemini",
                model="gemini-2.5-pro",
                api_key=os.getenv("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                supports_response_format=False,
                extra_body=None,
                reasoning_effort="high",
                sdk="openai",
            ),

            # Google Gemini (Native SDK)
            "gemini-flash-native": ModelSpec(
            provider="gemini",
            model="gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url=None,                   
            supports_response_format=True,   
            extra_body=None,
            reasoning_effort=None,          
            sdk="gemini_native",
        ),
            # Claude Anthropic
            "claude-api": ModelSpec(
                provider="claude",
                model="claude-sonnet-4-5",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                base_url="https://api.anthropic.com/v1/",
                supports_response_format=True,
                sdk="openai",
            ),
        }
    def get(self, name: str) -> ModelSpec:
        if name not in self.presets:
            raise KeyError(f"Unknown preset: {name}")
        return self.presets[name]

registry = Registry()

# -------------------------------
# Helpers
# -------------------------------
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
        
def _extract_choice_text(choice) -> str:
    """Robust for OpenAI-compat & Gemini-compat."""
    msg = getattr(choice, "message", None)
    if msg is not None:
        c = getattr(msg, "content", "")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            buf = []
            for p in c:
                if isinstance(p, str):
                    buf.append(p)
                elif isinstance(p, dict) and "text" in p:
                    buf.append(p["text"])
                else:
                    t = getattr(p, "text", None)
                    if isinstance(t, str): buf.append(t)
            return "".join(buf)
        if isinstance(c, dict) and "text" in c:
            return c["text"]
    if hasattr(choice, "text") and isinstance(choice.text, str):
        return choice.text
    delta = getattr(choice, "delta", None)
    if delta is not None:
        dc = getattr(delta, "content", "")
        if isinstance(dc, str): return dc
    return ""

def _is_max_tokens(fr: Any) -> bool:
    """Detect MAX_TOKENS across SDK versions (name, str, or int code=2)."""
    if fr is None: return False
    name = getattr(fr, "name", None)
    if isinstance(name, str) and name.upper() == "MAX_TOKENS": return True
    if isinstance(fr, str) and fr.upper() == "MAX_TOKENS": return True
    if isinstance(fr, int) and fr == 2: return True
    return False

# =========================
# Adapter
# =========================
class LLMClientAdapter:
    """
    Unified adapter:
      - OpenAI-compat providers via OpenAI client
      - Gemini native via google-generativeai
    """
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        api_key = spec.api_key
        if not api_key:
            raise RuntimeError(
                f"Missing API key for preset '{self.spec.model}' (provider={self.spec.provider}). "
                "Check your .env: GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY / LLAMA_API_KEY."
            )
        if spec.sdk == "gemini_native":
            genai.configure(api_key=api_key)
            self.client = None 
        else:
            self.client = OpenAI(api_key=api_key, base_url=spec.base_url) if spec.base_url else OpenAI(api_key=api_key)

    # ---------- Public API ----------
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        model_override: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        if self.spec.sdk == "gemini_native":
            return self._chat_gemini_native(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                model_override=model_override,
                **(extra_params or {}),
                **kwargs,
            )
        return self._chat_openai_compat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model_override=model_override,
            response_format=response_format,
            extra_params=extra_params,
            **kwargs,
        )
    
    def chat_json_loose(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        model_override: Optional[str] = None,
        strict_json_first: bool = False,
    ) -> Dict[str, Any]:
        txt = self.chat(messages, temperature=temperature, max_tokens=max_tokens, model_override=model_override)
        return _json_loads_loose(txt)

    # ---------- OpenAI-compat helpers ----------
    def _chat_openai_compat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        model_override: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        params: Dict[str, Any] = {
            "model": (model_override or self.spec.model),
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        if response_format is not None and getattr(self.spec, "supports_response_format", True):
            params["response_format"] = response_format

        # Compat-only knobs
        if self.spec.reasoning_effort:
            params["reasoning_effort"] = self.spec.reasoning_effort

        # Avoid invalid extra_body for Gemini-compat
        if self.spec.extra_body and self.spec.provider not in ("gemini",):
            params["extra_body"] = self.spec.extra_body

        if extra_params:
            params.update(extra_params)
        if kwargs:
            params.update(kwargs)

        resp = self._retry_chat(**params)
        return (_extract_choice_text(resp.choices[0]) or "").strip()
    
    # ---------- Retry logic for OpenAI-compat ----------
    def _retry_chat(self, **kwargs) -> Any:
        if not self.spec.supports_response_format and "response_format" in kwargs:
            kwargs.pop("response_format", None)

        last_err: Optional[Exception] = None
        for _ in range(3):
            try:
                return self.client.chat.completions.create(**kwargs)
            except BadRequestError as e:
                msg = str(e)
                # Drop response_format if rejected
                if "response_format" in msg and "response_format" in kwargs:
                    kwargs.pop("response_format", None)
                    continue
                # Drop extra_body if invalid for this provider
                if ("INVALID_ARGUMENT" in msg or "Unknown name" in msg) and "extra_body" in kwargs:
                    kwargs.pop("extra_body", None)
                    continue
                last_err = e
            except Exception as e:
                last_err = e
            time.sleep(0.6)
        raise last_err

    
     # ---------- Gemini native ----------
    @staticmethod
    def _split_system_and_contents(messages: List[Dict[str, str]]):
        sys = "\n".join([m["content"] for m in messages if m.get("role") == "system"]).strip() or None
        contents = []
        for m in messages:
            r = m.get("role")
            if r == "system": continue
            if r == "user":
                contents.append({"role": "user", "parts": [m.get("content", "")]})
            elif r in ("assistant", "model"):
                contents.append({"role": "model", "parts": [m.get("content", "")]})
            else:
                contents.append({"role": "user", "parts": [m.get("content", "")]})
        return sys, contents
    
    def _chat_gemini_native(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        model_override: Optional[str],
        response_format: Optional[Dict[str, Any]],
    ) -> str:
        model_id = model_override or self.spec.model
        system_instruction, contents = self._split_system_and_contents(messages)

        out_tokens = max_tokens

        def build_model(tokens: int):
            cfg: Dict[str, Any] = {
                "temperature": temperature if temperature is not None else 0.0,
                "max_output_tokens": tokens,
            }
            # Map simple JSON request
            if response_format and response_format.get("type") == "json_object":
                cfg["response_mime_type"] = "application/json"
            return genai.GenerativeModel(model_id, system_instruction=system_instruction, generation_config=cfg)

        model = build_model(out_tokens)
        resp = model.generate_content(contents)

        # Try to read text parts
        text = self._extract_gemini_text_or_retry(resp, contents, build_model, out_tokens)
        if text:
            return text.strip()
        
        # Fallback: flatten prompt (rare role/parts edge cases)
        flat_prompt = (system_instruction + "\n\n" if system_instruction else "")
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), None)
        flat_prompt += last_user or "\n".join(m["content"] for m in messages if m.get("role") != "system")

        model2 = build_model(16384)
        resp2 = model2.generate_content(flat_prompt)
        t2 = getattr(resp2, "text", None)
        if isinstance(t2, str) and t2.strip():
            return t2.strip()
        
        if getattr(resp2, "candidates", None):
            parts2 = getattr(resp2.candidates[0].content, "parts", []) or []
            buf = [getattr(p, "text", "") for p in parts2 if isinstance(getattr(p, "text", None), str)]
            if any(buf):
                return "".join(buf).strip()

        fr2 = getattr(resp2.candidates[0], "finish_reason", None) if getattr(resp2, "candidates", None) else None
        raise RuntimeError(f"Gemini native: empty after fallback. finish_reason={fr2}")

    
    def _extract_gemini_text_or_retry(self, resp, contents, build_model, out_tokens) -> str:
        
        if not getattr(resp, "candidates", None):
            pf = getattr(resp, "prompt_feedback", None)
            raise RuntimeError(f"Empty Gemini response. prompt_feedback={pf}")

        cand = resp.candidates[0]
        fr = getattr(cand, "finish_reason", None)

        parts = getattr(cand.content, "parts", []) if cand and cand.content else []
        buf = []
        for p in parts or []:
            t = getattr(p, "text", None)
            if isinstance(t, str):
                buf.append(t)
        if buf:
            return "".join(buf)

        # Retry once if we hit MAX_TOKENS with no text parts
        if _is_max_tokens(fr):
            new_tokens = 65536
            model2 = build_model(new_tokens)
            resp2 = model2.generate_content(contents)
            if getattr(resp2, "candidates", None):
                parts2 = getattr(resp2.candidates[0].content, "parts", []) or []
                buf2 = []
                for p in parts2:
                    t2 = getattr(p, "text", None)
                    if isinstance(t2, str): buf2.append(t2)
                if buf2:
                    return "".join(buf2)
            fr2 = getattr(resp2.candidates[0], "finish_reason", None) if getattr(resp2, "candidates", None) else None
            raise RuntimeError(f"Gemini returned no parts on retry. finish_reason={fr2}")

        # Other finish reasons (e.g., safety)
        raise RuntimeError(f"Gemini returned no parts. finish_reason={fr}")