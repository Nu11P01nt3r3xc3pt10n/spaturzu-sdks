"""Bedrock response → other-provider response adapters.

Called by the dispatcher when bedrock is the TARGET of a fallback:
translate Bedrock's response (boto3-shaped dict) into the caller's
expected shape — SimpleNamespace for openai/anthropic so attribute
access matches the existing translators' return type.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any


def _to_namespace(payload: Any) -> Any:
    """Recursively convert dict → SimpleNamespace, list → list of converted.

    Mirrors the existing ``_to_namespace`` helpers in _translate/_openai.py
    and _translate/_anthropic.py — keep behaviour identical so tests work
    uniformly across translators.
    """
    if isinstance(payload, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in payload.items()})
    if isinstance(payload, list):
        return [_to_namespace(item) for item in payload]
    return payload


def _flatten_bedrock_text(resp: dict[str, Any]) -> str:
    blocks = ((resp.get("output") or {}).get("message") or {}).get("content") or []
    return "".join(
        b["text"] for b in blocks if isinstance(b, dict) and isinstance(b.get("text"), str)
    )


def bedrock_response_to_chat(response: dict[str, Any], caller_model: str) -> Any:
    """Bedrock Converse response → OpenAI ChatCompletion response (SimpleNamespace)."""
    text = _flatten_bedrock_text(response)
    usage = response.get("usage") or {}
    stop_reason = response.get("stopReason")

    def _finish() -> str:
        if stop_reason == "max_tokens":
            return "length"
        if stop_reason == "tool_use":
            return "tool_calls"
        if stop_reason in ("content_filtered", "guardrail_intervened"):
            return "content_filter"
        return "stop"

    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)
    payload = {
        "id": "",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": caller_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": _finish(),
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": usage.get("totalTokens", input_tokens + output_tokens),
        },
    }
    return _to_namespace(payload)


def bedrock_response_to_anthropic(
    response: dict[str, Any], caller_model: str
) -> Any:
    """Bedrock Converse response → Anthropic Messages response (SimpleNamespace)."""
    text = _flatten_bedrock_text(response)
    usage = response.get("usage") or {}
    stop_reason = response.get("stopReason")

    def _stop() -> str:
        if stop_reason == "max_tokens":
            return "max_tokens"
        if stop_reason == "stop_sequence":
            return "stop_sequence"
        if stop_reason == "tool_use":
            return "tool_use"
        return "end_turn"

    payload = {
        "id": "",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": caller_model,
        "stop_reason": _stop(),
        "usage": {
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
        },
    }
    return _to_namespace(payload)
