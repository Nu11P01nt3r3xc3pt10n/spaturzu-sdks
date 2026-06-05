"""Translators where Gemini's generate_content is the SOURCE shape."""

from __future__ import annotations

from typing import Any, Optional


def _flatten_gemini_parts(parts: Any) -> str:
    if not isinstance(parts, list):
        return ""
    return "".join(p["text"] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str))


def _flatten_gemini_system(sys: Any) -> str:
    if sys is None:
        return ""
    if isinstance(sys, list):
        return _flatten_gemini_parts(sys)
    if isinstance(sys, dict) and isinstance(sys.get("parts"), list):
        return _flatten_gemini_parts(sys["parts"])
    return ""


def _gemini_has_unsupported_shape(params: dict[str, Any]) -> bool:
    for c in params.get("contents") or []:
        if not isinstance(c, dict):
            continue
        for p in c.get("parts") or []:
            if not isinstance(p, dict) or not isinstance(p.get("text"), str):
                return True
    return False


def _g(obj: Any, key: str, default: Any = None) -> Any:
    """Read either ``obj.key`` or ``obj[key]``. Handles both dicts and
    objects with attribute access (e.g. google-genai's typed response models)."""
    if obj is None:
        return default
    val = getattr(obj, key, None)
    if val is not None:
        return val
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


# ─── gemini → openai ─────────────────────────────────────────


def gemini_to_chat_params(
    params: dict[str, Any], to_model: str
) -> Optional[dict[str, Any]]:
    if _gemini_has_unsupported_shape(params):
        return None
    messages: list[dict[str, Any]] = []
    sys_text = _flatten_gemini_system((params.get("config") or {}).get("systemInstruction"))
    if sys_text:
        messages.append({"role": "system", "content": sys_text})
    for c in params.get("contents") or []:
        if not isinstance(c, dict):
            continue
        role = "assistant" if c.get("role") == "model" else "user"
        messages.append({"role": role, "content": _flatten_gemini_parts(c.get("parts") or [])})

    out: dict[str, Any] = {"model": to_model, "messages": messages}
    cfg = params.get("config") or {}
    if isinstance(cfg.get("maxOutputTokens"), int):
        out["max_tokens"] = cfg["maxOutputTokens"]
    if isinstance(cfg.get("temperature"), (int, float)):
        out["temperature"] = cfg["temperature"]
    if isinstance(cfg.get("topP"), (int, float)):
        out["top_p"] = cfg["topP"]
    stops = cfg.get("stopSequences")
    if isinstance(stops, list) and stops:
        out["stop"] = stops
    return out


def chat_response_from_gemini(
    response: Any, caller_model: str
) -> dict[str, Any]:
    """OpenAI chat-completion response → Gemini-shape response (plain dict)."""
    choices = _g(response, "choices") or []
    choice = choices[0] if choices else None
    msg = _g(choice, "message") if choice is not None else None
    text = _g(msg, "content", "") or ""
    finish = _g(choice, "finish_reason") if choice is not None else None
    usage = _g(response, "usage") or {}

    def _g_finish():
        if finish == "length":
            return "MAX_TOKENS"
        if finish == "content_filter":
            return "SAFETY"
        return "STOP"

    return {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": _g_finish()}
        ],
        "usageMetadata": {
            "promptTokenCount": _g(usage, "prompt_tokens", 0),
            "candidatesTokenCount": _g(usage, "completion_tokens", 0),
            "totalTokenCount": _g(usage, "total_tokens", 0),
        },
    }


# ─── gemini → anthropic ──────────────────────────────────────


def gemini_to_anthropic_params(
    params: dict[str, Any], to_model: str
) -> Optional[dict[str, Any]]:
    if _gemini_has_unsupported_shape(params):
        return None
    messages: list[dict[str, Any]] = []
    for c in params.get("contents") or []:
        if not isinstance(c, dict):
            continue
        role = "assistant" if c.get("role") == "model" else "user"
        messages.append({"role": role, "content": _flatten_gemini_parts(c.get("parts") or [])})

    cfg = params.get("config") or {}
    out: dict[str, Any] = {
        "model": to_model,
        "messages": messages,
        "max_tokens": cfg.get("maxOutputTokens") or 1024,
    }
    sys_text = _flatten_gemini_system(cfg.get("systemInstruction"))
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


def anthropic_response_from_gemini(
    response: Any, caller_model: str
) -> dict[str, Any]:
    """Anthropic response → Gemini-shape response (plain dict)."""
    content = _g(response, "content") or []
    text_parts: list[str] = []
    for b in content:
        t = _g(b, "text")
        if isinstance(t, str):
            text_parts.append(t)
    text = "".join(text_parts)

    usage = _g(response, "usage") or {}
    stop_reason = _g(response, "stop_reason")
    finish = "MAX_TOKENS" if stop_reason == "max_tokens" else "STOP"
    in_t = _g(usage, "input_tokens", 0) or 0
    out_t = _g(usage, "output_tokens", 0) or 0
    return {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": finish}
        ],
        "usageMetadata": {
            "promptTokenCount": in_t,
            "candidatesTokenCount": out_t,
            "totalTokenCount": in_t + out_t,
        },
    }


# ─── gemini → bedrock ────────────────────────────────────────


def gemini_to_bedrock_params(
    params: dict[str, Any], to_model_id: str
) -> Optional[dict[str, Any]]:
    if _gemini_has_unsupported_shape(params):
        return None
    messages: list[dict[str, Any]] = []
    for c in params.get("contents") or []:
        if not isinstance(c, dict):
            continue
        role = "assistant" if c.get("role") == "model" else "user"
        messages.append(
            {"role": role, "content": [{"text": _flatten_gemini_parts(c.get("parts") or [])}]}
        )
    cfg = params.get("config") or {}
    inf: dict[str, Any] = {}
    if isinstance(cfg.get("maxOutputTokens"), int):
        inf["maxTokens"] = cfg["maxOutputTokens"]
    if isinstance(cfg.get("temperature"), (int, float)):
        inf["temperature"] = cfg["temperature"]
    if isinstance(cfg.get("topP"), (int, float)):
        inf["topP"] = cfg["topP"]
    stops = cfg.get("stopSequences")
    if isinstance(stops, list) and stops:
        inf["stopSequences"] = stops

    out: dict[str, Any] = {"modelId": to_model_id, "messages": messages}
    sys_text = _flatten_gemini_system(cfg.get("systemInstruction"))
    if sys_text:
        out["system"] = [{"text": sys_text}]
    if inf:
        out["inferenceConfig"] = inf
    return out


def bedrock_response_from_gemini(
    response: dict[str, Any], caller_model: str
) -> dict[str, Any]:
    """Bedrock-shape response → Gemini-shape response (plain dict).

    Used when gemini is PRIMARY and bedrock was the fallback target."""
    blocks = ((response.get("output") or {}).get("message") or {}).get("content") or []
    text = "".join(
        b["text"] for b in blocks if isinstance(b, dict) and isinstance(b.get("text"), str)
    )
    usage = response.get("usage") or {}
    stop_reason = response.get("stopReason")

    def _g_finish():
        if stop_reason == "max_tokens":
            return "MAX_TOKENS"
        if stop_reason in ("content_filtered", "guardrail_intervened"):
            return "SAFETY"
        return "STOP"

    return {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": _g_finish()}
        ],
        "usageMetadata": {
            "promptTokenCount": usage.get("inputTokens", 0),
            "candidatesTokenCount": usage.get("outputTokens", 0),
            "totalTokenCount": usage.get("totalTokens", 0),
        },
    }


def gemini_to_mistral_params(
    params: dict[str, Any], to_model: str
) -> Optional[dict[str, Any]]:
    if _gemini_has_unsupported_shape(params):
        return None
    messages: list[dict[str, Any]] = []
    sys_text = _flatten_gemini_system((params.get("config") or {}).get("systemInstruction"))
    if sys_text:
        messages.append({"role": "system", "content": sys_text})
    for c in params.get("contents") or []:
        if not isinstance(c, dict):
            continue
        role = "assistant" if c.get("role") == "model" else "user"
        messages.append({"role": role, "content": _flatten_gemini_parts(c.get("parts") or [])})
    cfg = params.get("config") or {}
    out: dict[str, Any] = {"model": to_model, "messages": messages}
    if isinstance(cfg.get("maxOutputTokens"), int):
        out["maxTokens"] = cfg["maxOutputTokens"]
    if isinstance(cfg.get("temperature"), (int, float)):
        out["temperature"] = cfg["temperature"]
    if isinstance(cfg.get("topP"), (int, float)):
        out["topP"] = cfg["topP"]
    stops = cfg.get("stopSequences")
    if isinstance(stops, list) and stops:
        out["stop"] = stops
    return out
