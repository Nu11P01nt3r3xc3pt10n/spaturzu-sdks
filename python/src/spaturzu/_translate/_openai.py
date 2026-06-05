"""Translators where OpenAI's chat-completion is the SOURCE shape."""

from __future__ import annotations

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
    refused them via the ``_chat_has_unsupported_shape`` / equivalent
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


def _chat_has_unsupported_shape(params: Any) -> bool:
    if _get(params, "stream") is True:
        return True
    tools = _get(params, "tools")
    if isinstance(tools, list) and len(tools) > 0:
        return True
    if _get(params, "tool_choice") is not None:
        return True
    if _get(params, "response_format") is not None:
        return True
    for m in _get(params, "messages") or []:
        if _get(m, "role") == "tool":
            return True
        content = _get(m, "content")
        if isinstance(content, list):
            for part in content:
                if _get(part, "type") != "text":
                    return True
    return False


def _map_finish_to_stop_reason(finish: Optional[str]) -> str:
    if finish == "length":
        return "max_tokens"
    if finish == "tool_calls":
        return "tool_use"
    return "end_turn"


def _to_namespace(d: Any) -> Any:
    """Recursively convert dicts/lists into SimpleNamespace trees so
    callers can use dot-access on the translated response."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_namespace(x) for x in d]
    return d


# ─── translators ───────────────────────────────────────────────────────────


def chat_to_anthropic_params(params: Any, to_model: str) -> Optional[dict[str, Any]]:
    """OpenAI ChatCompletion params → Anthropic Messages params.

    Returns ``None`` when the call uses features v1 doesn't translate.
    """
    if _chat_has_unsupported_shape(params):
        return None

    system_buf: list[str] = []
    messages: list[dict[str, Any]] = []
    for m in _get(params, "messages") or []:
        role = _get(m, "role")
        content = _get(m, "content")
        if role == "system":
            system_buf.append(_flatten_text(content))
            continue
        if role not in ("user", "assistant"):
            continue
        messages.append({"role": role, "content": _flatten_text(content)})

    out: dict[str, Any] = {
        "model": to_model,
        "messages": messages,
        # Anthropic requires non-zero max_tokens. Default to 1024 when
        # the caller didn't set one — matches the TS translator.
        "max_tokens": _get(params, "max_tokens")
        or _get(params, "max_completion_tokens")
        or 1024,
    }
    if system_buf:
        out["system"] = "\n\n".join(system_buf)

    temperature = _get(params, "temperature")
    if temperature is not None:
        out["temperature"] = temperature
    top_p = _get(params, "top_p")
    if top_p is not None:
        out["top_p"] = top_p
    stop = _get(params, "stop")
    if isinstance(stop, str):
        out["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        out["stop_sequences"] = list(stop)
    return out


def chat_to_bedrock_params(
    params: Any, to_model_id: str
) -> Optional[dict[str, Any]]:
    """OpenAI chat-completion params → Bedrock Converse params."""
    if _chat_has_unsupported_shape(params):
        return None

    system_blocks: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    for m in _get(params, "messages") or []:
        if not isinstance(m, dict):
            continue
        role = _get(m, "role")
        text = _flatten_text(_get(m, "content"))
        if role == "system":
            system_blocks.append({"text": text})
            continue
        if role not in ("user", "assistant"):
            continue
        messages.append({"role": role, "content": [{"text": text}]})

    out: dict[str, Any] = {"modelId": to_model_id, "messages": messages}
    if system_blocks:
        out["system"] = system_blocks

    inf: dict[str, Any] = {}
    mt = _get(params, "max_tokens") or _get(params, "max_completion_tokens")
    if isinstance(mt, int):
        inf["maxTokens"] = mt
    temperature = _get(params, "temperature")
    if isinstance(temperature, (int, float)):
        inf["temperature"] = temperature
    top_p = _get(params, "top_p")
    if isinstance(top_p, (int, float)):
        inf["topP"] = top_p
    stop = _get(params, "stop")
    if isinstance(stop, str):
        inf["stopSequences"] = [stop]
    elif isinstance(stop, list) and stop:
        inf["stopSequences"] = stop
    if inf:
        out["inferenceConfig"] = inf
    return out


def chat_response_to_anthropic(response: Any, caller_model: str) -> SimpleNamespace:
    """OpenAI ChatCompletion response → Anthropic Messages response."""
    choices = _get(response, "choices") or []
    choice = choices[0] if choices else None
    msg = _get(choice, "message") if choice is not None else None
    text = _get(msg, "content") if msg is not None else ""
    usage = _get(response, "usage") or {}
    payload = {
        "id": _get(response, "id") or "",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text or ""}],
        "model": caller_model,
        "stop_reason": _map_finish_to_stop_reason(
            _get(choice, "finish_reason") if choice is not None else None,
        ),
        "usage": {
            "input_tokens": _get(usage, "prompt_tokens") or 0,
            "output_tokens": _get(usage, "completion_tokens") or 0,
        },
    }
    return _to_namespace(payload)


def chat_to_gemini_params(
    params: dict[str, Any], to_model: str
) -> Optional[dict[str, Any]]:
    """OpenAI chat-completion params → Gemini generate_content params."""
    if _chat_has_unsupported_shape(params):
        return None

    contents: list[dict[str, Any]] = []
    system_text = ""
    for m in _get(params, "messages") or []:
        if not isinstance(m, dict):
            continue
        role = _get(m, "role")
        text = _flatten_text(_get(m, "content"))
        if role == "system":
            system_text = f"{system_text}\n\n{text}" if system_text else text
            continue
        if role not in ("user", "assistant"):
            continue
        contents.append(
            {"role": "model" if role == "assistant" else "user", "parts": [{"text": text}]}
        )
    config: dict[str, Any] = {}
    if system_text:
        config["systemInstruction"] = {"parts": [{"text": system_text}]}
    mt = _get(params, "max_tokens") or _get(params, "max_completion_tokens")
    if isinstance(mt, int):
        config["maxOutputTokens"] = mt
    temperature = _get(params, "temperature")
    if isinstance(temperature, (int, float)):
        config["temperature"] = temperature
    top_p = _get(params, "top_p")
    if isinstance(top_p, (int, float)):
        config["topP"] = top_p
    stop = _get(params, "stop")
    if isinstance(stop, str):
        config["stopSequences"] = [stop]
    elif isinstance(stop, list) and stop:
        config["stopSequences"] = stop
    out: dict[str, Any] = {"model": to_model, "contents": contents}
    if config:
        out["config"] = config
    return out


def chat_to_mistral_params(
    params: dict[str, Any], to_model: str
) -> Optional[dict[str, Any]]:
    if _chat_has_unsupported_shape(params):
        return None
    messages = []
    for m in params.get("messages") or []:
        if not isinstance(m, dict) or m.get("role") == "tool":
            continue
        messages.append({"role": m.get("role"), "content": _flatten_text(m.get("content"))})
    out: dict[str, Any] = {"model": to_model, "messages": messages}
    max_t = params.get("max_tokens") or params.get("max_completion_tokens")
    if isinstance(max_t, int):
        out["maxTokens"] = max_t
    if isinstance(params.get("temperature"), (int, float)):
        out["temperature"] = params["temperature"]
    if isinstance(params.get("top_p"), (int, float)):
        out["topP"] = params["top_p"]
    if params.get("stop") is not None:
        out["stop"] = params["stop"]
    return out
