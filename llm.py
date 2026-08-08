"""Central LLM factory, failover, structured invocation, and cost logging.

Single source of truth for model wiring across all agents (profiler,
analysis planner/executor/reflector, insight, chat).

Design:
  * ``get_chat_model`` -> a provider-backed ``BaseChatModel``. OpenAI-compatible
    base URL is the primary provider (via ``OPENAI_API_KEY``/``MODEL``); Groq is
    the automatic failover on 4xx/5xx; plain Groq or Gemini as stand-alone
    providers when no OpenAI key is present.
  * ``structured_invoke`` runs one chat call and parses the reply into a
    Pydantic schema: tries tool calling (``with_structured_output``) first, then
    a JSON-mode prompt + validation so plain stubs and non-tool-calling models
    still work, and parse failures surface as Pydantic errors, never as
    silently-garbage strings.
  * Every helper appends a ``{call}`` record to ``state["llm_calls"]`` when a
    state dict is supplied: latency, provider/model, tokens, and an estimated
    price (table below). This makes token/cost visible per pipeline run.
  * Optional in-memory response cache (``LLM_CACHE_ENABLED=1``) memoizes exact
    prompt+task replies for cheap deterministic steps (planner / reflector).

The old per-agent ``ResilientFallbackModel`` / ``_build_*_llm`` copies are
superseded by this module; import from here instead of re-declaring overrides.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Type, Union

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from pydantic import BaseModel

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model / pricing registry
# ---------------------------------------------------------------------------

_MODEL_ALIASES: Dict[str, str] = {
    # task/model name -> model string used in prompts / env lookups
    "gpt-4.1-nano": "gpt-4.1-nano",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.0-flash": "gemini-2.0-flash",
}

# USD per 1M tokens. Fallbacks for unknown models: pessimistic estimate.
_PRICE_PER_MTOK: Dict[str, Dict[str, float]] = {
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},  # via Groq
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "default": {"input": 0.80, "output": 2.40},
}


def _price_for(model: str) -> Dict[str, float]:
    key = _MODEL_ALIASES.get(model, model)
    return _PRICE_PER_MTOK.get(key, _PRICE_PER_MTOK["default"])


def _task_model(task: str) -> Optional[str]:
    """Per-task model override from env, e.g. ``PLANNER_MODEL``."""
    raw = os.getenv(f"{task}_MODEL", "").strip()
    return raw or os.getenv("MODEL", "gpt-4.1-nano")


# ---------------------------------------------------------------------------
# Fallback chat model
# ---------------------------------------------------------------------------

class ResilientFallbackModel(BaseChatModel):
    """Calls ``primary`` first; transparently fails over to ``fallback``.

    Uses bounded exponential backoff between retries (steady-state fallback
    retries: up to 4 attempts, growing sleep). ``max_retries`` on the primary
    stays 0 so a 429/5xx fails immediately and crosses over to the fallback
    instead of hanging in OpenAI's own retry loop.
    """

    primary: Any
    fallback: Any
    attempt_limit: int = 4

    @property
    def _llm_type(self) -> str:
        return "resilient_fallback"

    def _generate(self, messages: Any, stop: Any = None, **kwargs: Any) -> Any:
        try:
            return self.primary._generate(messages, stop=stop, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Primary LLM failed (%s); failing over to fallback.", exc)
            for attempt in range(1, self.attempt_limit + 1):
                try:
                    return self.fallback._generate(messages, stop=stop, **kwargs)
                except Exception as fb_err:  # noqa: BLE001
                    if attempt >= self.attempt_limit:
                        raise fb_err
                    time.sleep(min(2 ** attempt, 12))

    @property
    def model(self) -> str:  # informational alias for logging
        return self.primary.model if getattr(self.primary, "model", None) else "resilient_fallback"

    @property
    def provider(self) -> str:
        return "resilient (openai/groq)"


def build_chat_model(task: str = "DEFAULT", temperature: float = 0.2) -> BaseChatModel:
    """Build the agent chat model for ``task`` with automatic failover.

    Returns either a plain model (single provider) or a
    ``ResilientFallbackModel`` when both primary and fallback are present.
    """
    # --- primary: OpenAI-compatible endpoint ---
    openai_key = os.getenv("OPENAI_API_KEY", "")
    default_model = _task_model(task)

    groq_key = os.getenv("GROQ_API_KEY", "")
    groq_llm = None
    if groq_key:
        try:
            from langchain_groq import ChatGroq
            groq_llm = ChatGroq(
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                groq_api_key=groq_key,
                temperature=temperature,
                max_retries=0,
            )
        except Exception:  # noqa: BLE001
            groq_llm = None

    if openai_key:
        from langchain_openai import ChatOpenAI
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        primary = ChatOpenAI(
            model=os.getenv("MODEL", "gpt-4.1-nano"),
            api_key=openai_key,
            base_url=base_url,
            temperature=temperature,
            max_retries=0,  # fail fast -> fallback handles retries
        )
        if groq_llm:
            return ResilientFallbackModel(primary=primary, fallback=groq_llm)
        return primary

    if groq_llm:
        return groq_llm

    if os.getenv("GEMINI_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            temperature=temperature,
        )

    raise EnvironmentError(
        "No LLM configured. Set at least one of OPENAI_API_KEY, GROQ_API_KEY or "
        "GEMINI_API_KEY in .env."
    )


# ---------------------------------------------------------------------------
# Structured invocation
# ---------------------------------------------------------------------------

def _messages_to_str(messages: Union[str, Sequence[Union[dict, BaseMessage]]]) -> str:
    if isinstance(messages, str):
        return messages
    parts = []
    for m in messages:
        if isinstance(m, BaseMessage):
            parts.append(m.content or "")
        elif isinstance(m, dict):
            parts.append(str(m.get("content", "")))
        else:
            parts.append(str(m))
    return "\n".join(parts)


def _extract_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return str(result.get("content") or result)
    if hasattr(result, "content"):
        txt = result.content
        if isinstance(txt, list):  # content blocks
            return "".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in txt
            )
        return str(txt)
    return str(result)


class _Cache:
    """Tiny in-process cache keyed by (task, canonical prompt)."""

    def __init__(self, cap: int = 256) -> None:
        self._cap = cap
        self._data: Dict[str, Any] = {}

    def _key(self, task: str, prompt: str, model: str) -> str:
        digest = hashlib.sha256(f"{model}::{task}::{prompt}".encode("utf-8")).hexdigest()
        return digest

    def get(self, task: str, prompt: str, model: str) -> Optional[Any]:
        if not os.getenv("LLM_CACHE_ENABLED"):
            return None
        return self._data.get(self._key(task, prompt, model))

    def set(self, task: str, prompt: str, model: str, value: Any) -> None:
        if not os.getenv("LLM_CACHE_ENABLED"):
            return
        key = self._key(task, prompt, model)
        self._data[key] = value
        if len(self._data) > self._cap:
            # naive eviction (drop oldest)
            for oldest in list(self._data)[: len(self._data) - self._cap]:
                self._data.pop(oldest, None)


_CACHE = _Cache()


def _usage_stats(model: str, result: Any, latency_s: float) -> Dict[str, Any]:
    """Best-effort token + price accounting for a single call."""
    in_tok = out_tok = None
    meta = getattr(result, "usage_metadata", None) or {}
    if isinstance(meta, dict):
        in_tok = meta.get("input_tokens") or meta.get("prompt_tokens")
        out_tok = meta.get("output_tokens") or meta.get("completion_tokens")
    resp_meta = getattr(result, "response_metadata", {}) or {}
    if in_tok is None and isinstance(resp_meta, dict):
        usage = resp_meta.get("usage") or {}
        in_tok = usage.get("prompt_tokens")
        out_tok = usage.get("completion_tokens")

    price = _price_for(model)
    cost = None
    if in_tok is not None and out_tok is not None:
        cost = round(in_tok / 1e6 * price["input"] + out_tok / 1e6 * price["output"], 6)
    return {
        "model": model,
        "latency_s": round(latency_s, 3),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": cost,
    }


def _record(state: Optional[Dict[str, Any]], record: Dict[str, Any]) -> None:
    if state is None:
        return
    state.setdefault("llm_calls", [])
    if not isinstance(state.get("llm_calls"), list):
        state["llm_calls"] = []
    state["llm_calls"].append(record)


def _llm_model_name(chat: BaseChatModel) -> str:
    for attr in ("model",):
        if getattr(chat, attr, None):
            return str(getattr(chat, attr))
    return getattr(chat, "_llm_type", "unknown")


def structured_invoke(
    task: str,
    messages: Union[str, Sequence[Union[dict, BaseMessage]]],
    schema: type[BaseModel],
    temperature: float = 0.2,
    chat: Optional[BaseChatModel] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Optional[BaseModel]:
    """Invoke ``schema`` still passes through with fallbacks; parse structured output.

    Returns a validated Pydantic model or ``None`` on total failure.
    """
    model = chat or build_chat_model(task, temperature=temperature)
    prompt = _messages_to_str(messages)
    model_name = _llm_model_name(model)

    cached = _CACHE.get(task, prompt, model_name)
    if cached is not None:
        if state is not None:
            rec = {"task": task, "cached": True, "model": model_name}
            _record(state, {**rec, **cached})
        return cached

    t0 = time.monotonic()
    try:
        parsed = _structured_call(model, schema, messages)
    except Exception as exc:  # noqa: BLE001
        logger.warning("structured_invoke(%s) failed: %s", task, exc)
        _record(state, {
            "task": task, "model": model_name, "ok": False,
            "error": str(exc)[:300], "latency_s": round(time.monotonic() - t0, 3),
        })
        return None
    latency = time.monotonic() - t0

    if isinstance(parsed, dict) and "parsed" in parsed:
        parsed = parsed["parsed"]
    if not isinstance(parsed, schema):
        try:
            parsed = schema.model_validate(parsed)
        except Exception:  # noqa: BLE001
            _record(state, {
                "task": task, "model": model_name, "ok": False,
                "error": f"cannot validate as {schema.__name__}",
                "latency_s": round(latency, 3),
            })
            return None

    usage = _usage_stats(model_name, parsed, latency)
    _record(state, {"task": task, "model": model_name, "ok": True, **usage})
    _CACHE.set(task, prompt, model_name, parsed)
    return parsed


def _structured_call(model: BaseChatModel, schema: type[BaseModel], messages: Any) -> Any:
    """Return a tool-calling structured response, or fall back to JSON-in-prompt."""
    try:
        bound = model.with_structured_output(schema)
        result = bound.invoke(messages)
        if isinstance(result, schema) or isinstance(result, dict) or isinstance(result, str):
            return result
    except (NotImplementedError, AttributeError, TypeError):
        pass
    except Exception:  # noqa: BLE001
        pass
    return _json_invoke(model, schema, messages)


def _json_invoke(model: BaseChatModel, schema: type[BaseModel], messages: Any) -> dict:
    """Prompt the model for a plain JSON object and hand it off for validation."""
    call = _messages_to_str(messages)
    json_prompt = (
        call + "\n\nRespond with ONLY a valid JSON object matching this schema:\n"
        + json.dumps(schema.model_json_schema(), default=str)
    )
    reply = model.invoke(json_prompt)
    text = _extract_text(reply)
    data = _json_from_text(text)
    if isinstance(data, list):
        # collections live under the batch container schema field anyway.
        pass
    if isinstance(data, dict):
        return data
    return schema.model_validate_json(text)


def _json_from_text(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        last = text.rfind("```")
        text = text[first_nl + 1 : last].strip() if last > first_nl else text[first_nl + 1 :].strip()
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        pass
    try:
        starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
        ends = [i for i in (text.rfind("}"), text.rfind("]")) if i != -1]
        if starts and ends and min(starts) < max(ends):
            return json.loads(text[min(starts) : max(ends) + 1])
    except Exception:  # noqa: BLE001
        pass
    return None


def plain_invoke(
    task: str,
    messages: Union[str, Sequence[Dict[str, Any]]],
    temperature: float = 0.2,
    chat: Optional[BaseChatModel] = None,
    state: Optional[Dict[str, Any]] = None,
) -> str:
    """Non-structured free-text call; returns the text reply."""
    model = chat or build_chat_model(task, temperature=temperature)
    model_name = _llm_model(model)
    t0 = time.monotonic()
    result = model.invoke(messages if isinstance(messages, (list, tuple)) else [{"role": "user", "content": messages}])
    latency = time.monotonic() - t0
    text = _extract_text(result)
    usage = _usage_stats(model_name, result, latency)
    _record(state, {"task": task, "model": model_name, "ok": True, **usage})
    return text


__all__ = [
    "ResilientFallbackModel",
    "build_chat_model",
    "structured_invoke",
    "plain_invoke",
    "_price_for",
]