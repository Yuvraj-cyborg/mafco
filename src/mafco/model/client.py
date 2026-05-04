"""LLM client for OpenAI (Chat Completions API).

Adapts to GPT-5-class reasoning models (`gpt-5*`, `o1*`, `o3*`):
  - sends `reasoning_effort` instead of `temperature`
  - uses `max_completion_tokens` instead of `max_tokens`
For legacy models (`gpt-4o`, `gpt-4-turbo`) the classic params are used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


_DEFAULT_MODEL = os.environ.get("MAFCO_MODEL", "gpt-5.5")
_DEFAULT_SUB_MODEL = os.environ.get("MAFCO_SUB_MODEL", _DEFAULT_MODEL)
_DEFAULT_MAX_TOKENS = int(os.environ.get("MAFCO_MAX_TOKENS", "4096"))
_DEFAULT_REASONING_EFFORT = os.environ.get("MAFCO_REASONING_EFFORT", "low")


def _is_reasoning_model(model: str) -> bool:
    """gpt-5, gpt-5-mini, gpt-5.5, o1, o3, o4 — they all take reasoning_effort."""
    m = model.lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


def _skip_reasoning_effort(model: str) -> str:
    """The cheapest reasoning_effort each model family accepts."""
    m = model.lower()
    # gpt-5.5+ swapped 'minimal' for 'none'; older gpt-5/o-series uses 'minimal'.
    if m.startswith(("gpt-5.5", "gpt-5.6", "gpt-6")):
        return "none"
    return "minimal"


@dataclass
class ModelConfig:
    """Configuration for which model to call and how."""

    model: str = _DEFAULT_MODEL
    sub_model: str = _DEFAULT_SUB_MODEL
    api_key: str | None = None
    base_url: str | None = None  # None -> OpenAI default endpoint
    # Used for non-reasoning models only.
    temperature: float = 0.6
    max_tokens: int | None = _DEFAULT_MAX_TOKENS
    # GPT-5 family only. "minimal" | "low" | "medium" | "high".
    reasoning_effort: str = _DEFAULT_REASONING_EFFORT
    # Backwards-compat flag.
    enable_reasoning: bool = True
    extra_headers: dict[str, str] = field(default_factory=dict)

    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        v = os.environ.get("OPENAI_API_KEY")
        if v:
            return v
        raise RuntimeError(
            "No API key found. Set OPENAI_API_KEY in env / .env"
        )


@dataclass
class LLMResponse:
    content: str
    raw: Any | None = None
    usage: dict[str, Any] | None = None
    model: str | None = None
    reasoning_details: Any | None = None


class LLMClient:
    """Thin wrapper around `openai.OpenAI().chat.completions.create`."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        kwargs: dict[str, Any] = {"api_key": self.config.resolved_api_key()}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        self._client = OpenAI(**kwargs)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        enable_reasoning: bool | None = None,
    ) -> LLMResponse:
        cfg = self.config
        m = model or cfg.model
        mt = cfg.max_tokens if max_tokens is None else max_tokens

        clean_messages = [_strip_msg(msg) for msg in messages]
        kwargs: dict[str, Any] = {"model": m, "messages": clean_messages}

        if _is_reasoning_model(m):
            if mt is not None:
                kwargs["max_completion_tokens"] = mt
            effort = reasoning_effort or cfg.reasoning_effort
            if enable_reasoning is False:
                effort = _skip_reasoning_effort(m)
            kwargs["reasoning_effort"] = effort
        else:
            if mt is not None:
                kwargs["max_tokens"] = mt
            kwargs["temperature"] = (
                cfg.temperature if temperature is None else temperature
            )

        if cfg.extra_headers:
            kwargs["extra_headers"] = cfg.extra_headers

        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        usage = None
        if getattr(resp, "usage", None) is not None:
            try:
                usage = resp.usage.model_dump()  # type: ignore[union-attr]
            except Exception:
                usage = dict(resp.usage)  # type: ignore[arg-type]
        return LLMResponse(
            content=msg.content or "",
            raw=resp,
            usage=usage,
            model=getattr(resp, "model", m),
        )

    def query(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        **kw: Any,
    ) -> str:
        msgs: list[dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.complete(msgs, model=model, **kw).content


_ALLOWED_MSG_KEYS = {"role", "content", "name", "tool_calls", "tool_call_id"}


def _strip_msg(msg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in msg.items() if k in _ALLOWED_MSG_KEYS}
