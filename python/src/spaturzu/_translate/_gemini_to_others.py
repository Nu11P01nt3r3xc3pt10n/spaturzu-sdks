"""Gemini response → other-provider response adapters.

Returns SimpleNamespace (via _to_namespace) so consumers get attribute
access matching the existing translators' return type pattern.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any


def _to_namespace(payload: Any) -> Any:
    """Mirror of the helper in _translate/_openai.py / _anthropic.py /
    _bedrock_to_others.py. Recursively convert dict → SimpleNamespace,
    list → list of converted."""
    if isinstance(payload, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in payload.items()})
    if isinstance(payload, list):
        return [_to_namespace(item) for item in payload]
    return payload


def _flatten_gemini_response_text(resp: Any) -> str:
    candidates = getattr(resp, "candidates", None)
    if candidates is None and isinstance(resp, dict):
        candidates = resp.get("candidates")
    candidates = candidates or []
    if not candidates:
        return ""
    first = candidates[0]
    content = getattr(first, "content", None)
    if content is None and isinstance(first, dict):
        content = first.get("content")
    if content is None:
        return ""
    parts = getattr(content, "parts", None)
    if parts is None and isinstance(content, dict):
        parts = content.get("parts")
    parts = parts or []
    out: list[str] = []
    for p in parts:
        t = getattr(p, "text", None)
        if t is None and isinstance(p, dict):
            t = p.get("text")
        if isinstance(t, str):
            out.append(t)
    return "".join(out)


def _get_finish_reason(resp: Any) -> Any:
    candidates = getattr(resp, "candidates", None)
    if candidates is None and isinstance(resp, dict):
        candidates = resp.get("candidates")
    candidates = candidates or []
    if not candidates:
        return None
    first = candidates[0]
    fr = getattr(first, "finishReason", None)
    if fr is None and isinstance(first, dict):
        fr = first.get("finishReason")
    return fr


def _get_usage_metadata(resp: Any) -> dict[str, Any]:
    um = getattr(resp, "usage_metadata", None)
    if um is None and isinstance(resp, dict):
        um = resp.get("usageMetadata")
    if um is None:
        return {}
    if isinstance(um, dict):
        return um
    # SimpleNamespace or typed model with attribute access
    return {
        "promptTokenCount": getattr(um, "prompt_token_count", None) or getattr(um, "promptTokenCount", None) or 0,
        "candidatesTokenCount": getattr(um, "candidates_token_count", None) or getattr(um, "candidatesTokenCount", None) or 0,
        "totalTokenCount": getattr(um, "total_token_count", None) or getattr(um, "totalTokenCount", None) or 0,
        "cachedContentTokenCount": getattr(um, "cached_content_token_count", None) or getattr(um, "cachedContentTokenCount", None),
    }


def gemini_response_to_chat(response: Any, caller_model: str) -> Any:
    text = _flatten_gemini_response_text(response)
    finish = _get_finish_reason(response)
    u = _get_usage_metadata(response)

    def _finish():
        if finish == "MAX_TOKENS":
            return "length"
        if finish in ("SAFETY", "RECITATION"):
            return "content_filter"
        return "stop"

    pt = u.get("promptTokenCount", 0) or 0
    ct = u.get("candidatesTokenCount", 0) or 0
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
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": u.get("totalTokenCount") or pt + ct,
        },
    }
    return _to_namespace(payload)


def gemini_response_to_anthropic(
    response: Any, caller_model: str
) -> Any:
    text = _flatten_gemini_response_text(response)
    finish = _get_finish_reason(response)
    u = _get_usage_metadata(response)
    stop = "max_tokens" if finish == "MAX_TOKENS" else "end_turn"
    payload = {
        "id": "",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": caller_model,
        "stop_reason": stop,
        "usage": {
            "input_tokens": u.get("promptTokenCount", 0) or 0,
            "output_tokens": u.get("candidatesTokenCount", 0) or 0,
        },
    }
    return _to_namespace(payload)


def gemini_response_to_bedrock(response: Any, caller_model: str) -> Any:
    text = _flatten_gemini_response_text(response)
    finish = _get_finish_reason(response)
    u = _get_usage_metadata(response)

    def _stop():
        if finish == "MAX_TOKENS":
            return "max_tokens"
        if finish in ("SAFETY", "RECITATION"):
            return "content_filtered"
        return "end_turn"

    input_tokens = u.get("promptTokenCount", 0) or 0
    output_tokens = u.get("candidatesTokenCount", 0) or 0
    payload = {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": _stop(),
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": u.get("totalTokenCount") or (input_tokens + output_tokens),
        },
    }
    return _to_namespace(payload)
