# utils/model_registry_instrumented.py
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import os, json, time, logging

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from openai import OpenAI
from openai import BadRequestError
import google.generativeai as genai

from utils.cost import estimate_cost

logger = logging.getLogger(__name__)
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

# -------------------------
# Helpers
# -------------------------
def _rough_token_estimate(txt: str) -> int:
    if not isinstance(txt, str): return 0
    return max(1, int(len(txt) / 4))

def _json_loads_loose(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            i = s.index("{"); j = s.rindex("}") + 1
            return json.loads(s[i:j])
        except Exception:
            return {"text": s[:2000]}

def _extract_choice_text(choice) -> str:
    msg = getattr(choice, "message", None)
    if msg is not None:
        c = getattr(msg, "content", "")
        if isinstance(c, str): return c
        if isinstance(c, list):
            buf = []
            for p in c:
                if isinstance(p, str): buf.append(p)
                elif isinstance(p, dict) and "text" in p: buf.append(p["text"])
                else:
                    t = getattr(p, "text", None)
                    if isinstance(t, str): buf.append(t)
            return "".join(buf)
        if isinstance(c, dict) and "text" in c: return c["text"]
    if hasattr(choice, "text") and isinstance(choice.text, str): return choice.text
    delta = getattr(choice, "delta", None)
    if delta is not None:
        dc = getattr(delta, "content", "")
        if isinstance(dc, str): return dc
    return ""

def _is_max_tokens(fr: Any) -> bool:
    if fr is None: return False
    name = getattr(fr, "name", None)
    if isinstance(name, str) and name.upper() == "MAX_TOKENS": return True
    if isinstance(fr, str) and fr.upper() == "MAX_TOKENS": return True
    if isinstance(fr, int) and fr == 2: return True
    return False

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
    """
    Instrumented registry with the same keys as your original one.
    Safe to import only in ablation code.
    """
    def __init__(self):
        self.presets = {
            "openai-mini": ModelSpec(
                provider="openai",
                model="gpt-4o-mini",
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=None,
                supports_response_format=True,
                sdk="openai",
            ),
            "deepseek-chat": ModelSpec(
                provider="deepseek",
                model="deepseek-chat",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
                supports_response_format=False,
                sdk="openai",
            ),
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

# keep a module-level default instance if you like parity
registry = Registry()

# -------------------------
# Adapter (instrumented)
# -------------------------
class LLMClientAdapter:
    """
    API-compatible with your original adapter:
      - chat() -> str
      - chat_json_loose() -> dict
    Adds:
      - stage() context manager
      - call_log(), totals(), usage(), reset_call_log()
    """

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self._current_stage = None
        self._call_log: List[Dict[str, Any]] = []
        self._usage = {"in": 0, "out": 0}

        if not spec.api_key:
            raise RuntimeError(
                f"Missing API key for preset '{spec.model}' (provider={spec.provider}). "
                "Set GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY in .env"
            )

        if spec.sdk == "gemini_native":
            genai.configure(api_key=spec.api_key)
            self.client = None
        else:
            self.client = OpenAI(api_key=spec.api_key, base_url=spec.base_url) if spec.base_url else OpenAI(api_key=spec.api_key)

    # ---- stage tagging ----
    @contextmanager
    def stage(self, name: str):
        prev = self._current_stage
        self._current_stage = name
        try:
            yield
        finally:
            self._current_stage = prev

    def set_stage(self, name: Optional[str]):
        self._current_stage = name

    # ---- accounting API ----
    def reset_usage(self): self._usage = {"in": 0, "out": 0}
    def reset_call_log(self): self._call_log = []; self.reset_usage()
    def usage(self) -> dict: return dict(self._usage)
    def call_log(self) -> List[Dict[str, Any]]: return [dict(x) for x in self._call_log]
    def totals(self) -> Dict[str, Any]:
        ti = sum(x["tokens_in"] for x in self._call_log)
        to = sum(x["tokens_out"] for x in self._call_log)
        tc = sum(x["cost_usd"] for x in self._call_log)
        tl = sum(x["latency_sec"] for x in self._call_log)
        return {"tokens_in": ti, "tokens_out": to, "cost_usd": tc, "latency_sec": tl}

    # ---- public chat APIs ----
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
                messages, temperature, max_tokens, model_override, response_format
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

    # ---- internal accounting sink ----
    def _record_call(self, *, prompt_tokens: int, completion_tokens: int, latency_sec: float):
        self._usage["in"] += int(prompt_tokens)
        self._usage["out"] += int(completion_tokens)
        cost = estimate_cost(self.spec.model, prompt_tokens, completion_tokens)
        self._call_log.append({
            "stage": self._current_stage or "unknown",
            "model": self.spec.model,
            "latency_sec": float(latency_sec),
            "tokens_in": int(prompt_tokens),
            "tokens_out": int(completion_tokens),
            "cost_usd": float(cost),
        })

    # ---- OpenAI-compatible path ----
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
        if self.spec.reasoning_effort:
            params["reasoning_effort"] = self.spec.reasoning_effort
        if self.spec.extra_body and self.spec.provider not in ("gemini",):
            params["extra_body"] = self.spec.extra_body
        if extra_params: params.update(extra_params)
        if kwargs: params.update(kwargs)

        t0 = time.perf_counter()
        resp = self._retry_chat(**params)
        latency = time.perf_counter() - t0

        out_text = _extract_choice_text(resp.choices[0]) if getattr(resp, "choices", None) else ""

        # usage robustly
        pt = ct = 0
        ui = getattr(resp, "usage", None)
        try:
            if ui:
                pt = int(getattr(ui, "prompt_tokens", 0) or ui.get("prompt_tokens", 0))
                ct = int(getattr(ui, "completion_tokens", 0) or ui.get("completion_tokens", 0))
            else:
                in_text = "\n".join(m.get("content","") for m in messages)
                pt = _rough_token_estimate(in_text)
                ct = _rough_token_estimate(out_text or "")
        except Exception:
            in_text = "\n".join(m.get("content","") for m in messages)
            pt = _rough_token_estimate(in_text)
            ct = _rough_token_estimate(out_text or "")

        self._record_call(prompt_tokens=pt, completion_tokens=ct, latency_sec=latency)
        return (out_text or "").strip()

    def _retry_chat(self, **kwargs) -> Any:
        if not self.spec.supports_response_format and "response_format" in kwargs:
            kwargs.pop("response_format", None)
        last_err: Optional[Exception] = None
        for _ in range(3):
            try:
                return self.client.chat.completions.create(**kwargs)
            except BadRequestError as e:
                msg = str(e)
                if "response_format" in msg and "response_format" in kwargs:
                    kwargs.pop("response_format", None); continue
                if ("INVALID_ARGUMENT" in msg or "Unknown name" in msg) and "extra_body" in kwargs:
                    kwargs.pop("extra_body", None); continue
                last_err = e
            except Exception as e:
                last_err = e
            time.sleep(0.6)
        raise last_err

    # ---- Gemini native path ----
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

        out_tokens = max_tokens or 256

        def build_model(tokens: int):
            cfg: Dict[str, Any] = {
                "temperature": temperature if temperature is not None else 0.0,
                "max_output_tokens": tokens,
            }
            if response_format and response_format.get("type") == "json_object":
                cfg["response_mime_type"] = "application/json"
            return genai.GenerativeModel(model_id, system_instruction=system_instruction, generation_config=cfg)

        model = build_model(out_tokens)

        t0 = time.perf_counter()
        resp = model.generate_content(contents)
        latency = time.perf_counter() - t0

        # Extract text (with fallback retry)
        text = self._extract_gemini_text_or_retry(resp, contents, build_model, out_tokens) or ""

        # usage
        pt = ct = 0
        try:
            um = getattr(resp, "usage_metadata", None)
            if um:
                pt = int(getattr(um, "prompt_token_count", 0) or um.get("prompt_token_count", 0))
                ct = int(getattr(um, "candidates_token_count", 0) or um.get("candidates_token_count", 0))
            else:
                in_text = "\n".join(m.get("content","") for m in messages)
                pt = _rough_token_estimate(in_text)
                ct = _rough_token_estimate(text or "")
        except Exception:
            in_text = "\n".join(m.get("content","") for m in messages)
            pt = _rough_token_estimate(in_text)
            ct = _rough_token_estimate(text or "")

        self._record_call(prompt_tokens=pt, completion_tokens=ct, latency_sec=latency)
        return text.strip()

    def _extract_gemini_text_or_retry(self, resp, contents, build_model, out_tokens) -> str:
        if not getattr(resp, "candidates", None):
            pf = getattr(resp, "prompt_feedback", None)
            raise RuntimeError(f"Empty Gemini response. prompt_feedback={pf}")

        cand = resp.candidates[0]
        fr = getattr(cand, "finish_reason", None)

        parts = getattr(cand.content, "parts", []) if cand and cand.content else []
        buf = [getattr(p, "text", "") for p in (parts or []) if isinstance(getattr(p, "text", None), str)]
        if any(buf):
            return "".join(buf)

        if _is_max_tokens(fr):
            new_tokens = max(2 * (out_tokens or 1), 64)
            logger.info(f"[Gemini native] MAX_TOKENS; retrying with max_output_tokens={new_tokens}")
            model2 = build_model(new_tokens)
            resp2 = model2.generate_content(contents)
            if getattr(resp2, "candidates", None):
                parts2 = getattr(resp2.candidates[0].content, "parts", []) or []
                buf2 = [getattr(p, "text", "") for p in parts2 if isinstance(getattr(p, "text", None), str)]
                if any(buf2):
                    return "".join(buf2)
            fr2 = getattr(resp2.candidates[0], "finish_reason", None) if getattr(resp2, "candidates", None) else None
            raise RuntimeError(f"Gemini returned no parts on retry. finish_reason={fr2}")

        raise RuntimeError(f"Gemini returned no parts. finish_reason={fr}")
