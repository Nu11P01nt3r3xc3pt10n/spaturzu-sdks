"""Unit tests for spaturzu._translate Mistral translators."""

from __future__ import annotations

from spaturzu._translate import (
    anthropic_response_from_mistral,
    anthropic_to_mistral_params,
    bedrock_response_from_mistral,
    bedrock_to_mistral_params,
    chat_response_from_mistral,
    chat_to_mistral_params,
    gemini_response_from_mistral,
    gemini_to_mistral_params,
    mistral_response_to_anthropic,
    mistral_response_to_bedrock,
    mistral_response_to_chat,
    mistral_response_to_gemini,
    mistral_to_anthropic_params,
    mistral_to_bedrock_params,
    mistral_to_chat_params,
    mistral_to_gemini_params,
)


def _get(obj, key, default=None):
    """Read either obj.key or obj[key] — translators may return SimpleNamespace or dict."""
    val = getattr(obj, key, None)
    if val is not None:
        return val
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def test_mistral_to_chat_renames_keys() -> None:
    out = mistral_to_chat_params(
        {
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": "hi"}],
            "maxTokens": 200,
            "temperature": 0.7,
            "topP": 0.9,
        },
        "gpt-4o",
    )
    assert out is not None
    assert out["max_tokens"] == 200
    assert out["temperature"] == 0.7
    assert out["top_p"] == 0.9


def test_mistral_to_chat_returns_none_for_tools() -> None:
    out = mistral_to_chat_params(
        {
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function"}],
        },
        "gpt-4o",
    )
    assert out is None


def test_mistral_to_anthropic_pulls_system() -> None:
    out = mistral_to_anthropic_params(
        {
            "model": "x",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ],
            "maxTokens": 100,
        },
        "claude",
    )
    assert out is not None
    assert out["system"] == "be brief"
    assert out["max_tokens"] == 100
    assert out["messages"] == [{"role": "user", "content": "hi"}]


def test_mistral_to_anthropic_defaults_max_tokens() -> None:
    out = mistral_to_anthropic_params(
        {"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        "claude",
    )
    assert out is not None
    assert out["max_tokens"] == 1024


def test_mistral_to_bedrock_wraps_in_content_blocks() -> None:
    out = mistral_to_bedrock_params(
        {
            "model": "x",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ],
            "maxTokens": 200,
        },
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
    )
    assert out is not None
    assert out["system"] == [{"text": "be brief"}]
    assert out["messages"][0] == {"role": "user", "content": [{"text": "hi"}]}
    assert out["inferenceConfig"]["maxTokens"] == 200


def test_mistral_to_gemini_maps_assistant_to_model() -> None:
    out = mistral_to_gemini_params(
        {
            "model": "x",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            "maxTokens": 100,
        },
        "gemini-2.5-pro",
    )
    assert out is not None
    assert out["config"]["systemInstruction"] == {"parts": [{"text": "be brief"}]}
    assert out["contents"] == [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "hello"}]},
    ]
    assert out["config"]["maxOutputTokens"] == 100


def test_mistral_response_to_chat_maps_model_length_to_length() -> None:
    out = mistral_response_to_chat(
        {
            "id": "msg-1",
            "object": "chat.completion",
            "created": 0,
            "model": "mistral-large-latest",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "model_length"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
        "gpt-4o",
    )
    # Returns SimpleNamespace
    assert _get(out, "model") == "gpt-4o"
    choice0 = _get(out, "choices")[0]
    assert _get(choice0, "finish_reason") == "length"


def test_mistral_response_to_anthropic_maps_tool_calls() -> None:
    out = mistral_response_to_anthropic(
        {
            "id": "msg-1",
            "object": "chat.completion",
            "created": 0,
            "model": "mistral",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "tool"}, "finish_reason": "tool_calls"}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        },
        "claude",
    )
    assert _get(out, "stop_reason") == "tool_use"
    content = _get(out, "content")
    assert _get(content[0], "type") == "text"
    assert _get(content[0], "text") == "tool"


def test_chat_to_mistral_renames_keys() -> None:
    out = chat_to_mistral_params(
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 500,
            "top_p": 0.9,
        },
        "mistral-large-latest",
    )
    assert out is not None
    assert out["maxTokens"] == 500
    assert out["topP"] == 0.9


def test_anthropic_to_mistral_converts_string_system() -> None:
    out = anthropic_to_mistral_params(
        {
            "model": "claude",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "system": "be brief",
        },
        "mistral-large-latest",
    )
    assert out is not None
    assert out["messages"][0] == {"role": "system", "content": "be brief"}
    assert out["maxTokens"] == 100


def test_bedrock_to_mistral_flattens_system_array() -> None:
    out = bedrock_to_mistral_params(
        {
            "modelId": "x",
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "system": [{"text": "be brief"}],
            "inferenceConfig": {"maxTokens": 100},
        },
        "mistral-large-latest",
    )
    assert out is not None
    assert out["messages"][0] == {"role": "system", "content": "be brief"}
    assert out["maxTokens"] == 100


def test_gemini_to_mistral_maps_model_to_assistant() -> None:
    out = gemini_to_mistral_params(
        {
            "model": "gemini-2.5-pro",
            "contents": [
                {"role": "user", "parts": [{"text": "hi"}]},
                {"role": "model", "parts": [{"text": "hello"}]},
            ],
            "config": {
                "systemInstruction": {"parts": [{"text": "be brief"}]},
                "maxOutputTokens": 100,
            },
        },
        "mistral-large-latest",
    )
    assert out is not None
    assert out["messages"][0] == {"role": "system", "content": "be brief"}
    assert out["messages"][1] == {"role": "user", "content": "hi"}
    assert out["messages"][2] == {"role": "assistant", "content": "hello"}
    assert out["maxTokens"] == 100


def test_chat_response_from_mistral_preserves_caller_model() -> None:
    out = chat_response_from_mistral(
        {
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        },
        "mistral-large-latest",
    )
    # plain dict
    assert out["model"] == "mistral-large-latest"
    assert out["choices"][0]["message"]["content"] == "hi"
    assert out["usage"] == {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
