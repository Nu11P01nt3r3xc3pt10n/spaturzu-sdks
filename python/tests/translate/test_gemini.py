"""Unit tests for spaturzu._translate Gemini translators."""

from __future__ import annotations

from spaturzu._translate import (
    anthropic_response_from_gemini,
    anthropic_to_gemini_params,
    bedrock_response_from_gemini,
    bedrock_to_gemini_params,
    chat_response_from_gemini,
    chat_to_gemini_params,
    gemini_response_to_anthropic,
    gemini_response_to_bedrock,
    gemini_response_to_chat,
    gemini_to_anthropic_params,
    gemini_to_bedrock_params,
    gemini_to_chat_params,
)


def _get(obj, key, default=None):
    """Read either obj.key or obj[key] — translators may return dict or SimpleNamespace."""
    val = getattr(obj, key, None)
    if val is not None:
        return val
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def test_gemini_to_chat_converts_roles_and_pulls_system() -> None:
    out = gemini_to_chat_params(
        {
            "model": "gemini-2.5-pro",
            "contents": [
                {"role": "user", "parts": [{"text": "hi"}]},
                {"role": "model", "parts": [{"text": "hello"}]},
            ],
            "config": {
                "systemInstruction": {"parts": [{"text": "be brief"}]},
                "maxOutputTokens": 200,
            },
        },
        "gpt-4o",
    )
    assert out is not None
    assert out["messages"][0] == {"role": "system", "content": "be brief"}
    assert out["messages"][1] == {"role": "user", "content": "hi"}
    assert out["messages"][2] == {"role": "assistant", "content": "hello"}
    assert out["max_tokens"] == 200


def test_gemini_to_anthropic_defaults_max_tokens() -> None:
    out = gemini_to_anthropic_params(
        {"model": "x", "contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        "claude",
    )
    assert out is not None
    assert out["max_tokens"] == 1024


def test_gemini_response_to_chat_maps_max_tokens_to_length() -> None:
    out = gemini_response_to_chat(
        {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": ""}]}, "finishReason": "MAX_TOKENS"}
            ],
            "usageMetadata": {"promptTokenCount": 0, "candidatesTokenCount": 0},
        },
        "gpt-4o",
    )
    choice0 = _get(out, "choices")[0]
    assert _get(choice0, "finish_reason") == "length"


def test_chat_to_gemini_maps_assistant_to_model() -> None:
    out = chat_to_gemini_params(
        {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        },
        "gemini-2.5-pro",
    )
    assert out is not None
    assert out["contents"][1] == {"role": "model", "parts": [{"text": "hello"}]}


def test_anthropic_to_gemini_converts_string_system() -> None:
    out = anthropic_to_gemini_params(
        {
            "model": "claude",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "system": "be brief",
        },
        "gemini-2.5-pro",
    )
    assert out is not None
    assert out["config"]["systemInstruction"] == {"parts": [{"text": "be brief"}]}
    assert out["config"]["maxOutputTokens"] == 100


def test_bedrock_to_gemini_converts_system_array() -> None:
    out = bedrock_to_gemini_params(
        {
            "modelId": "x",
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "system": [{"text": "be brief"}],
            "inferenceConfig": {"maxTokens": 200},
        },
        "gemini-2.5-pro",
    )
    assert out is not None
    assert out["config"]["systemInstruction"] == {"parts": [{"text": "be brief"}]}
    assert out["config"]["maxOutputTokens"] == 200


def test_bedrock_to_gemini_maps_assistant_to_model() -> None:
    out = bedrock_to_gemini_params(
        {
            "modelId": "x",
            "messages": [
                {"role": "user", "content": [{"text": "hi"}]},
                {"role": "assistant", "content": [{"text": "hello"}]},
            ],
        },
        "gemini-2.5-pro",
    )
    assert out is not None
    assert out["contents"][0] == {"role": "user", "parts": [{"text": "hi"}]}
    assert out["contents"][1] == {"role": "model", "parts": [{"text": "hello"}]}


def test_chat_response_from_gemini() -> None:
    out = chat_response_from_gemini(
        {
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
        },
        "gemini-2.5-pro",
    )
    # chat_response_from_gemini returns a plain dict.
    assert out["candidates"][0]["content"]["parts"][0]["text"] == "hi"
    assert out["usageMetadata"] == {
        "promptTokenCount": 4,
        "candidatesTokenCount": 3,
        "totalTokenCount": 7,
    }
