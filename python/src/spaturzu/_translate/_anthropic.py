"""Translators where Anthropic's Messages API is the SOURCE shape."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Optional


# ─── helpers ───────────────────────────────────────────────────────────────


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field off either a dict or an object with attributes.

    The real OpenAI/Anthropic Python SDKs return Pydantic objects; our
    fake translator output is :class:`SimpleNamespace`. Both expose
    ``getattr``. Customer-passed params are usually dicts."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _flatten_text(content: Any) -> str:
    """Reduce OpenAI content parts / Anthropic content blocks to a single
    string. Non-text parts are dropped — the translator caller has already
    refused them via the ``_anthropic_has_unsupported_shape`` / equivalent
    check, so this stays defensive but never lossy in practice."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    out: list[str] = []
    for part in content:
        # Tolerate both plain dicts (our translator output) and SDK
        # Pydantic objects (real provider responses) — both expose
        # ``type`` and ``text`` via subscript or attribute.
        ptype = _get(part, "type")
        if ptype == "text":
            txt = _get(part, "text")
            if isinstance(txt, str):
                out.append(txt)
    return "".join(out)


def _anthropic_has_unsupported_shape(params: Any) -> bool:
    if _get(params, "stream") is True:
        return True
    for m in _get(params, "messages") or []:
        content = _get(m, "content")
        if isinstance(content, list):
            for part in content:
                if _get(part, "type") != "text":
                    return True
    return False


def _map_stop_reason_to_finish(reason: Optional[str]) -> str:
    if reason == "max_tokens":
        return "length"
    if reason == "tool_use":
        return "tool_calls"
    # end_turn, stop_sequence, None
    return "stop"


def _to_namespace(d: Any) -> Any:
    """Recursively convert dicts/lists into SimpleNamespace trees so
    callers can use dot-access on the translated response."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_namespace(x) for x in d]
    return d


# ─── translators ───────────────────────────────────────────────────────────


def anthropic_params_to_chat(params: Any, to_model: str) -> Optional[dict[str, Any]]:
    """Anthropic Messages params → OpenAI ChatCompletion params.

    Returns ``None`` when the call uses features v1 doesn't translate.
    """
    if _anthropic_has_unsupported_shape(params):
        return None

    messages: list[dict[str, Any]] = []
    system = _get(params, "system")
    if system is not None:
        messages.append({"role": "system", "content": _flatten_text(system)})
    for m in _get(params, "messages") or []:
        messages.append(
            {"role": _get(m, "role"), "content": _flatten_text(_get(m, "content"))},
        )

    out: dict[str, Any] = {
        "model": to_model,
        "messages": messages,
        "max_tokens": _get(params, "max_tokens"),
    }
    temperature = _get(params, "temperature")
    if temperature is not None:
        out["temperature"] = temperature
    top_p = _get(params, "top_p")
    if top_p is not None:
        out["top_p"] = top_p
    stop_sequences = _get(params, "stop_sequences")
    if isinstance(stop_sequences, list) and stop_sequences:
        out["stop"] = list(stop_sequences)
    return out


def anthropic_to_bedrock_params(
    params: Any, to_model_id: str
) -> Optional[dict[str, Any]]:
    """Anthropic Messages params → Bedrock Converse params."""
    if _anthropic_has_unsupported_shape(params):
        return None

    messages: list[dict[str, Any]] = []
    for m in _get(params, "messages") or []:
        if not isinstance(m, dict):
            continue
        text = _flatten_text(_get(m, "content"))
        messages.append({"role": _get(m, "role"), "content": [{"text": text}]})

    inf: dict[str, Any] = {}
    mt = _get(params, "max_tokens")
    if isinstance(mt, int) and mt > 0:
        inf["maxTokens"] = mt
    temperature = _get(params, "temperature")
    if isinstance(temperature, (int, float)):
        inf["temperature"] = temperature
    top_p = _get(params, "top_p")
    if isinstance(top_p, (int, float)):
        inf["topP"] = top_p
    stops = _get(params, "stop_sequences")
    if isinstance(stops, list) and stops:
        inf["stopSequences"] = stops

    out: dict[str, Any] = {
        "modelId": to_model_id,
        "messages": messages,
    }
    if inf:
        out["inferenceConfig"] = inf

    system = _get(params, "system")
    if isinstance(system, str) and system:
        out["system"] = [{"text": system}]
    elif isinstance(system, list):
        sys_text = "\n\n".join(
            b["text"]
            for b in system
            if isinstance(b, dict) and isinstance(b.get("text"), str)
        )
        if sys_text:
            out["system"] = [{"text": sys_text}]

    return out


def anthropic_response_to_chat(response: Any, caller_model: str) -> SimpleNamespace:
    """Anthropic Messages response → OpenAI ChatCompletion response (as a
    :class:`SimpleNamespace` so attribute access works)."""
    text = _flatten_text(_get(response, "content") or [])
    usage = _get(response, "usage") or {}
    prompt_tokens = _get(usage, "input_tokens") or 0
    completion_tokens = _get(usage, "output_tokens") or 0
    payload = {
        "id": _get(response, "id") or "",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": caller_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": _map_stop_reason_to_finish(
                    _get(response, "stop_reason"),
                ),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return _to_namespace(payload)


def anthropic_to_gemini_params(
    params: Any, to_model: str
) -> Optional[dict[str, Any]]:
    """Anthropic Messages params → Gemini generate_content params."""
    if _anthropic_has_unsupported_shape(params):
        return None
    contents: list[dict[str, Any]] = []
    for m in _get(params, "messages") or []:
        if not isinstance(m, dict):
            continue
        role = "model" if _get(m, "role") == "assistant" else "user"
        contents.append(
            {"role": role, "parts": [{"text": _flatten_text(_get(m, "content"))}]}
        )
    config: dict[str, Any] = {}
    mt = _get(params, "max_tokens")
    if isinstance(mt, int) and mt > 0:
        config["maxOutputTokens"] = mt
    system = _get(params, "system")
    if isinstance(system, str) and system:
        config["systemInstruction"] = {"parts": [{"text": system}]}
    elif isinstance(system, list):
        sys_text = "\n\n".join(
            b["text"] for b in system if isinstance(b, dict) and isinstance(b.get("text"), str)
        )
        if sys_text:
            config["systemInstruction"] = {"parts": [{"text": sys_text}]}
    temperature = _get(params, "temperature")
    if isinstance(temperature, (int, float)):
        config["temperature"] = temperature
    top_p = _get(params, "top_p")
    if isinstance(top_p, (int, float)):
        config["topP"] = top_p
    stops = _get(params, "stop_sequences")
    if isinstance(stops, list) and stops:
        config["stopSequences"] = stops
    out: dict[str, Any] = {"model": to_model, "contents": contents}
    if config:
        out["config"] = config
    return out


def anthropic_to_mistral_params(
    params: dict[str, Any], to_model: str
) -> Optional[dict[str, Any]]:
    if _anthropic_has_unsupported_shape(params):
        return None
    messages: list[dict[str, Any]] = []
    system = params.get("system")
    if isinstance(system, str) and system:
        messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        sys_text = "\n\n".join(
            b["text"] for b in system if isinstance(b, dict) and isinstance(b.get("text"), str)
        )
        if sys_text:
            messages.append({"role": "system", "content": sys_text})
    for m in params.get("messages") or []:
        if not isinstance(m, dict):
            continue
        messages.append({"role": m.get("role"), "content": _flatten_text(m.get("content"))})
    out: dict[str, Any] = {"model": to_model, "messages": messages}
    # Only set maxTokens if a positive int (avoid {maxTokens: None})
    mt = params.get("max_tokens")
    if isinstance(mt, int) and mt > 0:
        out["maxTokens"] = mt
    if isinstance(params.get("temperature"), (int, float)):
        out["temperature"] = params["temperature"]
    if isinstance(params.get("top_p"), (int, float)):
        out["topP"] = params["top_p"]
    stops = params.get("stop_sequences")
    if isinstance(stops, list) and stops:
        out["stop"] = stops
    return out
