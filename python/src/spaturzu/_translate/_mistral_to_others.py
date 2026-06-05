"""Mistral response → other-provider response adapters.

Returns SimpleNamespace via _to_namespace (matches existing pattern in
_openai_to_others / _anthropic_to_others / _bedrock_to_others / _gemini_to_others).
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any


def _to_namespace(payload: Any) -> Any:
    if isinstance(payload, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in payload.items()})
    if isinstance(payload, list):
        return [_to_namespace(item) for item in payload]
    return payload


def mistral_response_to_chat(response: dict[str, Any], caller_model: str) -> Any:
    choices = []
    for c in response.get("choices") or []:
        finish = c.get("finish_reason")
        new_finish = (
            "length" if finish in ("length", "model_length")
            else "tool_calls" if finish == "tool_calls"
            else "stop"
        )
        choices.append({**c, "finish_reason": new_finish})
    payload = {
        "id": response.get("id", ""),
        "object": "chat.completion",
        "created": response.get("created", int(time.time())),
        "model": caller_model,
        "choices": choices,
        "usage": response.get("usage") or {},
    }
    return _to_namespace(payload)


def mistral_response_to_anthropic(
    response: dict[str, Any], caller_model: str
) -> Any:
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    text = (choice.get("message") or {}).get("content", "") or ""
    finish = choice.get("finish_reason")
    usage = response.get("usage") or {}

    def _stop():
        if finish in ("length", "model_length"):
            return "max_tokens"
        if finish == "tool_calls":
            return "tool_use"
        return "end_turn"

    payload = {
        "id": response.get("id", ""),
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": caller_model,
        "stop_reason": _stop(),
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }
    return _to_namespace(payload)


def mistral_response_to_bedrock(
    response: dict[str, Any], caller_model_id: str
) -> Any:
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    text = (choice.get("message") or {}).get("content", "") or ""
    finish = choice.get("finish_reason")
    usage = response.get("usage") or {}

    def _stop():
        if finish in ("length", "model_length"):
            return "max_tokens"
        if finish == "tool_calls":
            return "tool_use"
        return "end_turn"

    payload = {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": _stop(),
        "usage": {
            "inputTokens": usage.get("prompt_tokens", 0),
            "outputTokens": usage.get("completion_tokens", 0),
            "totalTokens": usage.get("total_tokens", 0),
        },
    }
    return _to_namespace(payload)


def mistral_response_to_gemini(
    response: dict[str, Any], caller_model: str
) -> Any:
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    text = (choice.get("message") or {}).get("content", "") or ""
    finish = choice.get("finish_reason")
    usage = response.get("usage") or {}

    def _g_finish():
        if finish in ("length", "model_length"):
            return "MAX_TOKENS"
        return "STOP"

    payload = {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": _g_finish()}
        ],
        "usageMetadata": {
            "promptTokenCount": usage.get("prompt_tokens", 0),
            "candidatesTokenCount": usage.get("completion_tokens", 0),
            "totalTokenCount": usage.get("total_tokens", 0),
        },
    }
    return _to_namespace(payload)
