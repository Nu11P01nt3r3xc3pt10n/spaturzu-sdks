"""Translators where Bedrock's Converse API is the SOURCE shape."""

from __future__ import annotations

from typing import Any, Optional


def _flatten_bedrock_content(blocks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for b in blocks or []:
        if isinstance(b, dict) and isinstance(b.get("text"), str):
            parts.append(b["text"])
    return "".join(parts)


def _bedrock_has_unsupported_shape(params: dict[str, Any]) -> bool:
    if params.get("toolConfig") is not None:
        return True
    if params.get("additionalModelRequestFields") is not None:
        return True
    for m in params.get("messages") or []:
        if not isinstance(m, dict):
            continue
        for b in m.get("content") or []:
            if isinstance(b, dict) and "text" not in b:
                return True
    return False


def _map_bedrock_stop_to_openai_finish(reason: Optional[str]) -> str:
    if reason == "max_tokens":
        return "length"
    if reason == "tool_use":
        return "tool_calls"
    if reason in ("content_filtered", "guardrail_intervened"):
        return "content_filter"
    return "stop"


def _map_bedrock_stop_to_anthropic(reason: Optional[str]) -> str:
    if reason == "max_tokens":
        return "max_tokens"
    if reason == "stop_sequence":
        return "stop_sequence"
    if reason == "tool_use":
        return "tool_use"
    return "end_turn"


# ─── bedrock → openai ───────────────────────────────────────────────


def bedrock_to_chat_params(
    params: dict[str, Any], to_model: str
) -> Optional[dict[str, Any]]:
    """Bedrock Converse params → OpenAI chat-completion params."""
    if _bedrock_has_unsupported_shape(params):
        return None

    messages: list[dict[str, Any]] = []
    system = params.get("system")
    if system:
        sys_text = "\n\n".join(
            b["text"] for b in system if isinstance(b, dict) and isinstance(b.get("text"), str)
        )
        if sys_text:
            messages.append({"role": "system", "content": sys_text})

    for m in params.get("messages") or []:
        if not isinstance(m, dict):
            continue
        messages.append(
            {"role": m.get("role"), "content": _flatten_bedrock_content(m.get("content") or [])}
        )

    out: dict[str, Any] = {"model": to_model, "messages": messages}
    cfg = params.get("inferenceConfig") or {}
    if isinstance(cfg.get("maxTokens"), int):
        out["max_tokens"] = cfg["maxTokens"]
    if isinstance(cfg.get("temperature"), (int, float)):
        out["temperature"] = cfg["temperature"]
    if isinstance(cfg.get("topP"), (int, float)):
        out["top_p"] = cfg["topP"]
    stops = cfg.get("stopSequences")
    if isinstance(stops, list) and stops:
        out["stop"] = stops
    return out


def chat_response_from_bedrock(
    response: Any, caller_model_id: str
) -> dict[str, Any]:
    """OpenAI chat response → Bedrock Converse response.

    Used when bedrock is the PRIMARY and an openai fallback served the
    call. Returns a plain dict matching boto3's Converse response shape.

    ``response`` may be a SimpleNamespace (if it came through
    ``chat_response_to_anthropic``) or a dict. Handle both via getattr-or-getitem.
    """
    def _g(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        v = getattr(obj, key, None)
        if v is not None:
            return v
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    choices = _g(response, "choices") or []
    choice = choices[0] if choices else None
    message = _g(choice, "message") if choice is not None else None
    text = _g(message, "content", "") or ""
    finish = _g(choice, "finish_reason")
    usage = _g(response, "usage") or {}

    def _stop() -> str:
        if finish == "length":
            return "max_tokens"
        if finish == "tool_calls":
            return "tool_use"
        if finish == "content_filter":
            return "content_filtered"
        return "end_turn"

    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": _stop(),
        "usage": {
            "inputTokens": _g(usage, "prompt_tokens", 0),
            "outputTokens": _g(usage, "completion_tokens", 0),
            "totalTokens": _g(usage, "total_tokens", 0),
        },
    }


# ─── bedrock → anthropic ────────────────────────────────────────────


def bedrock_to_anthropic_params(
    params: dict[str, Any], to_model: str
) -> Optional[dict[str, Any]]:
    """Bedrock Converse params → Anthropic Messages params."""
    if _bedrock_has_unsupported_shape(params):
        return None

    messages: list[dict[str, Any]] = []
    for m in params.get("messages") or []:
        if not isinstance(m, dict):
            continue
        messages.append(
            {"role": m.get("role"), "content": _flatten_bedrock_content(m.get("content") or [])}
        )

    cfg = params.get("inferenceConfig") or {}
    out: dict[str, Any] = {
        "model": to_model,
        "messages": messages,
        "max_tokens": cfg.get("maxTokens") or 1024,
    }
    system = params.get("system")
    if system:
        sys_text = "\n\n".join(
            b["text"] for b in system if isinstance(b, dict) and isinstance(b.get("text"), str)
        )
        if sys_text:
            out["system"] = sys_text
    if isinstance(cfg.get("temperature"), (int, float)):
        out["temperature"] = cfg["temperature"]
    if isinstance(cfg.get("topP"), (int, float)):
        out["top_p"] = cfg["topP"]
    stops = cfg.get("stopSequences")
    if isinstance(stops, list) and stops:
        out["stop_sequences"] = stops
    return out


def anthropic_response_from_bedrock(
    response: Any, caller_model_id: str
) -> dict[str, Any]:
    """Anthropic Messages response → Bedrock Converse response.

    Used when bedrock is the PRIMARY and an anthropic fallback served.
    Handles both dict and SimpleNamespace inputs (the latter comes from
    ``anthropic_response_to_chat``).
    """
    def _g(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        v = getattr(obj, key, None)
        if v is not None:
            return v
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    content = _g(response, "content") or []
    text_parts: list[str] = []
    for b in content:
        t = _g(b, "text")
        if isinstance(t, str):
            text_parts.append(t)
    text = "".join(text_parts)

    usage = _g(response, "usage") or {}
    stop_reason = _g(response, "stop_reason")

    def _stop() -> str:
        if stop_reason == "max_tokens":
            return "max_tokens"
        if stop_reason == "stop_sequence":
            return "stop_sequence"
        if stop_reason == "tool_use":
            return "tool_use"
        return "end_turn"

    in_t = _g(usage, "input_tokens", 0) or 0
    out_t = _g(usage, "output_tokens", 0) or 0

    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": _stop(),
        "usage": {
            "inputTokens": in_t,
            "outputTokens": out_t,
            "totalTokens": in_t + out_t,
        },
    }


def bedrock_to_gemini_params(
    params: dict[str, Any], to_model: str
) -> Optional[dict[str, Any]]:
    """Bedrock Converse params → Gemini generate_content params."""
    if _bedrock_has_unsupported_shape(params):
        return None
    contents: list[dict[str, Any]] = []
    for m in params.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = "model" if m.get("role") == "assistant" else "user"
        contents.append(
            {
                "role": role,
                "parts": [{"text": _flatten_bedrock_content(m.get("content") or [])}],
            }
        )
    cfg = params.get("inferenceConfig") or {}
    config: dict[str, Any] = {}
    if isinstance(cfg.get("maxTokens"), int):
        config["maxOutputTokens"] = cfg["maxTokens"]
    if isinstance(cfg.get("temperature"), (int, float)):
        config["temperature"] = cfg["temperature"]
    if isinstance(cfg.get("topP"), (int, float)):
        config["topP"] = cfg["topP"]
    stops = cfg.get("stopSequences")
    if isinstance(stops, list) and stops:
        config["stopSequences"] = stops
    system = params.get("system") or []
    sys_text = "\n\n".join(
        b["text"] for b in system if isinstance(b, dict) and isinstance(b.get("text"), str)
    )
    if sys_text:
        config["systemInstruction"] = {"parts": [{"text": sys_text}]}
    out: dict[str, Any] = {"model": to_model, "contents": contents}
    if config:
        out["config"] = config
    return out


def bedrock_to_mistral_params(
    params: dict[str, Any], to_model: str
) -> Optional[dict[str, Any]]:
    if _bedrock_has_unsupported_shape(params):
        return None
    messages: list[dict[str, Any]] = []
    system = params.get("system") or []
    sys_text = "\n\n".join(
        b["text"] for b in system if isinstance(b, dict) and isinstance(b.get("text"), str)
    )
    if sys_text:
        messages.append({"role": "system", "content": sys_text})
    for m in params.get("messages") or []:
        if not isinstance(m, dict):
            continue
        messages.append(
            {"role": m.get("role"), "content": _flatten_bedrock_content(m.get("content") or [])}
        )
    cfg = params.get("inferenceConfig") or {}
    out: dict[str, Any] = {"model": to_model, "messages": messages}
    if isinstance(cfg.get("maxTokens"), int):
        out["maxTokens"] = cfg["maxTokens"]
    if isinstance(cfg.get("temperature"), (int, float)):
        out["temperature"] = cfg["temperature"]
    if isinstance(cfg.get("topP"), (int, float)):
        out["topP"] = cfg["topP"]
    stops = cfg.get("stopSequences")
    if isinstance(stops, list) and stops:
        out["stop"] = stops
    return out
