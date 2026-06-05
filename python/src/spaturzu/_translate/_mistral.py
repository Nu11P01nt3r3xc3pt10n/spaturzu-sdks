"""Translators where Mistral's chat.complete is the SOURCE shape.

Plan 5 completes the 5×5 matrix. Mistral's API is OpenAI-compatible,
so mistral↔openai is near-identity; other pairs follow the patterns
established in earlier plans.
"""

from __future__ import annotations

import time
from typing import Any, Optional


def _mistral_has_unsupported_shape(params: dict[str, Any]) -> bool:
    tools = params.get("tools")
    if isinstance(tools, list) and tools:
        return True
    if params.get("responseFormat") is not None:
        return True
    for m in params.get("messages") or []:
        if isinstance(m, dict) and m.get("role") == "tool":
            return True
    return False


def _g(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    val = getattr(obj, key, None)
    if val is not None:
        return val
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


# ─── mistral → openai (near-identity) ────────────────────────


def mistral_to_chat_params(
    params: dict[str, Any], to_model: str
) -> Optional[dict[str, Any]]:
    if _mistral_has_unsupported_shape(params):
        return None
    messages = [
        m for m in (params.get("messages") or []) if isinstance(m, dict) and m.get("role") != "tool"
    ]
    out: dict[str, Any] = {"model": to_model, "messages": messages}
    if isinstance(params.get("maxTokens"), int):
        out["max_tokens"] = params["maxTokens"]
    if isinstance(params.get("temperature"), (int, float)):
        out["temperature"] = params["temperature"]
    if isinstance(params.get("topP"), (int, float)):
        out["top_p"] = params["topP"]
    if params.get("stop") is not None:
        out["stop"] = params["stop"]
    return out


def chat_response_from_mistral(
    response: Any, caller_model: str
) -> dict[str, Any]:
    # Near-identity except for caller-model echo and finish_reason mapping.
    choices_in = _g(response, "choices") or []
    out_choices = []
    for c in choices_in:
        finish = _g(c, "finish_reason")
        new_finish = (
            "length" if finish in ("length", "model_length")
            else "tool_calls" if finish == "tool_calls"
            else "stop"
        )
        msg = _g(c, "message")
        content = _g(msg, "content", "") or "" if msg is not None else ""
        index = _g(c, "index", 0)
        out_choices.append({
            "index": index,
            "message": {"role": "assistant", "content": content},
            "finish_reason": new_finish,
        })
    usage = _g(response, "usage") or {}
    return {
        "id": _g(response, "id", ""),
        "object": "chat.completion",
        "created": _g(response, "created", int(time.time())),
        "model": caller_model,
        "choices": out_choices,
        "usage": {
            "prompt_tokens": _g(usage, "prompt_tokens", 0),
            "completion_tokens": _g(usage, "completion_tokens", 0),
            "total_tokens": _g(usage, "total_tokens", 0),
        },
    }


# ─── mistral → anthropic ──────────────────────────────────


def mistral_to_anthropic_params(
    params: dict[str, Any], to_model: str
) -> Optional[dict[str, Any]]:
    if _mistral_has_unsupported_shape(params):
        return None
    system_buf: list[str] = []
    messages: list[dict[str, Any]] = []
    for m in params.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "system":
            system_buf.append(m.get("content") or "")
            continue
        if role not in ("user", "assistant"):
            continue
        messages.append({"role": role, "content": m.get("content") or ""})
    out: dict[str, Any] = {
        "model": to_model,
        "messages": messages,
        "max_tokens": params.get("maxTokens") or 1024,
    }
    if system_buf:
        out["system"] = "\n\n".join(system_buf)
    if isinstance(params.get("temperature"), (int, float)):
        out["temperature"] = params["temperature"]
    if isinstance(params.get("topP"), (int, float)):
        out["top_p"] = params["topP"]
    stop = params.get("stop")
    if isinstance(stop, str):
        out["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        out["stop_sequences"] = stop
    return out


def anthropic_response_from_mistral(
    response: Any, caller_model: str
) -> dict[str, Any]:
    content = _g(response, "content") or []
    text_parts: list[str] = []
    for b in content:
        t = _g(b, "text")
        if isinstance(t, str):
            text_parts.append(t)
    text = "".join(text_parts)
    usage = _g(response, "usage") or {}
    stop_reason = _g(response, "stop_reason")
    finish = "length" if stop_reason == "max_tokens" else "stop"
    in_t = _g(usage, "input_tokens", 0) or 0
    out_t = _g(usage, "output_tokens", 0) or 0
    return {
        "id": _g(response, "id", ""),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": caller_model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
        ],
        "usage": {
            "prompt_tokens": in_t,
            "completion_tokens": out_t,
            "total_tokens": in_t + out_t,
        },
    }


# ─── mistral → bedrock + mistral → gemini ─────────────────


def mistral_to_bedrock_params(
    params: dict[str, Any], to_model_id: str
) -> Optional[dict[str, Any]]:
    if _mistral_has_unsupported_shape(params):
        return None
    system_blocks: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    for m in params.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "system":
            system_blocks.append({"text": m.get("content") or ""})
            continue
        if role not in ("user", "assistant"):
            continue
        messages.append({"role": role, "content": [{"text": m.get("content") or ""}]})
    inf: dict[str, Any] = {}
    if isinstance(params.get("maxTokens"), int):
        inf["maxTokens"] = params["maxTokens"]
    if isinstance(params.get("temperature"), (int, float)):
        inf["temperature"] = params["temperature"]
    if isinstance(params.get("topP"), (int, float)):
        inf["topP"] = params["topP"]
    stop = params.get("stop")
    if isinstance(stop, str):
        inf["stopSequences"] = [stop]
    elif isinstance(stop, list) and stop:
        inf["stopSequences"] = stop
    out: dict[str, Any] = {"modelId": to_model_id, "messages": messages}
    if system_blocks:
        out["system"] = system_blocks
    if inf:
        out["inferenceConfig"] = inf
    return out


def bedrock_response_from_mistral(
    response: dict[str, Any], caller_model: str
) -> dict[str, Any]:
    blocks = ((response.get("output") or {}).get("message") or {}).get("content") or []
    text = "".join(b["text"] for b in blocks if isinstance(b, dict) and isinstance(b.get("text"), str))
    usage = response.get("usage") or {}
    stop_reason = response.get("stopReason")
    finish = "length" if stop_reason == "max_tokens" else "stop"
    return {
        "id": "",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": caller_model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
        ],
        "usage": {
            "prompt_tokens": usage.get("inputTokens", 0),
            "completion_tokens": usage.get("outputTokens", 0),
            "total_tokens": usage.get("totalTokens", 0),
        },
    }


def mistral_to_gemini_params(
    params: dict[str, Any], to_model: str
) -> Optional[dict[str, Any]]:
    if _mistral_has_unsupported_shape(params):
        return None
    contents: list[dict[str, Any]] = []
    system_text = ""
    for m in params.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            system_text = f"{system_text}\n\n{content}" if system_text else content
            continue
        if role not in ("user", "assistant"):
            continue
        contents.append(
            {"role": "model" if role == "assistant" else "user", "parts": [{"text": content}]}
        )
    config: dict[str, Any] = {}
    if system_text:
        config["systemInstruction"] = {"parts": [{"text": system_text}]}
    if isinstance(params.get("maxTokens"), int):
        config["maxOutputTokens"] = params["maxTokens"]
    if isinstance(params.get("temperature"), (int, float)):
        config["temperature"] = params["temperature"]
    if isinstance(params.get("topP"), (int, float)):
        config["topP"] = params["topP"]
    stop = params.get("stop")
    if isinstance(stop, str):
        config["stopSequences"] = [stop]
    elif isinstance(stop, list) and stop:
        config["stopSequences"] = stop
    out: dict[str, Any] = {"model": to_model, "contents": contents}
    if config:
        out["config"] = config
    return out


def gemini_response_from_mistral(
    response: dict[str, Any], caller_model: str
) -> dict[str, Any]:
    candidates = response.get("candidates") or []
    parts = ((candidates[0] or {}).get("content") or {}).get("parts") if candidates else []
    text = "".join(p["text"] for p in (parts or []) if isinstance(p, dict) and isinstance(p.get("text"), str))
    finish = (candidates[0] or {}).get("finishReason") if candidates else None
    u = response.get("usageMetadata") or {}
    return {
        "id": "",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": caller_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "length" if finish == "MAX_TOKENS" else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": u.get("promptTokenCount", 0),
            "completion_tokens": u.get("candidatesTokenCount", 0),
            "total_tokens": u.get("totalTokenCount", 0),
        },
    }
