"""Unit tests for spaturzu._translate Bedrock translators."""

from __future__ import annotations

from spaturzu._translate import (
    anthropic_response_from_bedrock,
    anthropic_to_bedrock_params,
    bedrock_response_to_anthropic,
    bedrock_response_to_chat,
    bedrock_to_anthropic_params,
    bedrock_to_chat_params,
    chat_response_from_bedrock,
    chat_to_bedrock_params,
)


def _get(obj, key, default=None):
    """Read either obj.key or obj[key]. Translators return SimpleNamespace or dict."""
    val = getattr(obj, key, None)
    if val is not None:
        return val
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def test_bedrock_to_chat_pulls_system_out() -> None:
    out = bedrock_to_chat_params(
        {
            "modelId": "x",
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "system": [{"text": "be brief"}, {"text": "use lists"}],
            "inferenceConfig": {"maxTokens": 200, "temperature": 0.7, "topP": 0.9, "stopSequences": ["STOP"]},
        },
        "gpt-4o",
    )
    assert out is not None
    assert out["messages"][0] == {"role": "system", "content": "be brief\n\nuse lists"}
    assert out["messages"][1] == {"role": "user", "content": "hi"}
    assert out["max_tokens"] == 200
    assert out["temperature"] == 0.7
    assert out["top_p"] == 0.9
    assert out["stop"] == ["STOP"]


def test_bedrock_to_chat_returns_none_for_tool_config() -> None:
    assert (
        bedrock_to_chat_params(
            {
                "modelId": "x",
                "messages": [{"role": "user", "content": [{"text": "hi"}]}],
                "toolConfig": {"tools": []},
            },
            "gpt-4o",
        )
        is None
    )


def test_bedrock_to_anthropic_defaults_max_tokens() -> None:
    out = bedrock_to_anthropic_params(
        {"modelId": "x", "messages": [{"role": "user", "content": [{"text": "hi"}]}]},
        "claude",
    )
    assert out is not None
    assert out["max_tokens"] == 1024


def test_bedrock_response_to_chat_basic_mapping() -> None:
    out = bedrock_response_to_chat(
        {
            "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 4, "outputTokens": 3, "totalTokens": 7},
        },
        "gpt-4o",
    )
    # Adapt to SimpleNamespace return shape (matches Plan 1's pattern for response translators).
    assert _get(out, "model") == "gpt-4o"
    choice0 = _get(out, "choices")[0]
    assert _get(_get(choice0, "message"), "content") == "hello"
    assert _get(choice0, "finish_reason") == "stop"
    usage = _get(out, "usage")
    assert _get(usage, "prompt_tokens") == 4
    assert _get(usage, "completion_tokens") == 3
    assert _get(usage, "total_tokens") == 7


def test_bedrock_response_to_chat_max_tokens_to_length() -> None:
    out = bedrock_response_to_chat(
        {
            "output": {"message": {"role": "assistant", "content": []}},
            "stopReason": "max_tokens",
            "usage": {"inputTokens": 0, "outputTokens": 0},
        },
        "gpt",
    )
    choice0 = _get(out, "choices")[0]
    assert _get(choice0, "finish_reason") == "length"


def test_bedrock_response_to_anthropic_wraps_in_text_block() -> None:
    out = bedrock_response_to_anthropic(
        {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}, {"text": " there"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 3, "outputTokens": 5},
        },
        "claude-3-5-haiku",
    )
    content = _get(out, "content")
    assert _get(content[0], "type") == "text"
    assert _get(content[0], "text") == "hi there"
    usage = _get(out, "usage")
    assert _get(usage, "input_tokens") == 3
    assert _get(usage, "output_tokens") == 5


def test_chat_to_bedrock_params_pulls_system() -> None:
    out = chat_to_bedrock_params(
        {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ],
            "max_tokens": 500,
        },
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
    )
    assert out is not None
    assert out["modelId"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert out["system"] == [{"text": "be brief"}]
    assert out["messages"][0] == {"role": "user", "content": [{"text": "hi"}]}
    assert out["inferenceConfig"]["maxTokens"] == 500


def test_anthropic_to_bedrock_params_string_system_to_array() -> None:
    out = anthropic_to_bedrock_params(
        {
            "model": "claude",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "system": "be brief",
        },
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
    )
    assert out is not None
    assert out["system"] == [{"text": "be brief"}]
    assert out["inferenceConfig"]["maxTokens"] == 100


def test_chat_response_from_bedrock_translates_to_bedrock_shape() -> None:
    out = chat_response_from_bedrock(
        {
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hi back"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
    )
    # chat_response_from_bedrock returns a plain dict (Bedrock shape).
    assert out["output"]["message"]["content"] == [{"text": "hi back"}]
    assert out["stopReason"] == "end_turn"
    assert out["usage"] == {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7}


def test_anthropic_response_from_bedrock() -> None:
    out = anthropic_response_from_bedrock(
        {
            "id": "msg-1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "model": "claude",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 4, "output_tokens": 3},
        },
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
    )
    assert out["output"]["message"]["content"] == [{"text": "hello"}]
    assert out["stopReason"] == "end_turn"
    assert out["usage"] == {"inputTokens": 4, "outputTokens": 3, "totalTokens": 7}
