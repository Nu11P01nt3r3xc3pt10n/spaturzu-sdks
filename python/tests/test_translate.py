"""Unit tests for spaturzu._translate (mirror of test/translate.test.ts).

The function names in the Python SDK match the TypeScript snake-case
equivalents — see ``sdks/python/src/spaturzu/_translate.py``. If the
names differ, swap the imports below.

Shape notes (verified from production source):
- ``chat_to_anthropic_params`` and ``anthropic_params_to_chat`` return plain
  ``dict`` (or ``None``), so dict-subscript access is correct.
- ``anthropic_response_to_chat`` and ``chat_response_to_anthropic`` return
  ``SimpleNamespace`` trees (recursively converted via ``_to_namespace``), so
  attribute access is required: ``out.choices[0].message.content``.
"""

from __future__ import annotations

from spaturzu._translate import (
    anthropic_params_to_chat,
    anthropic_response_to_chat,
    chat_response_to_anthropic,
    chat_to_anthropic_params,
)


def test_chat_to_anthropic_pulls_system_out() -> None:
    out = chat_to_anthropic_params(
        {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "system", "content": "use lists"},
                {"role": "user", "content": "hi"},
            ],
        },
        "claude-3-5-haiku-20241022",
    )
    assert out is not None
    assert out["system"] == "be brief\n\nuse lists"
    assert out["messages"] == [{"role": "user", "content": "hi"}]


def test_chat_to_anthropic_defaults_max_tokens() -> None:
    out = chat_to_anthropic_params(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        "claude",
    )
    assert out is not None
    assert out["max_tokens"] == 1024


def test_chat_to_anthropic_uses_max_completion_tokens_fallback() -> None:
    out = chat_to_anthropic_params(
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 500,
        },
        "claude",
    )
    assert out is not None
    assert out["max_tokens"] == 500


def test_chat_to_anthropic_converts_stop_string_to_stop_sequences() -> None:
    out = chat_to_anthropic_params(
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": "STOP",
        },
        "claude",
    )
    assert out is not None
    assert out["stop_sequences"] == ["STOP"]


def test_chat_to_anthropic_returns_none_for_streaming() -> None:
    assert (
        chat_to_anthropic_params(
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            "claude",
        )
        is None
    )


def test_chat_to_anthropic_returns_none_for_tools() -> None:
    assert (
        chat_to_anthropic_params(
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function"}],
            },
            "claude",
        )
        is None
    )


def test_chat_to_anthropic_returns_none_for_tool_messages() -> None:
    assert (
        chat_to_anthropic_params(
            {
                "model": "gpt-4o",
                "messages": [{"role": "tool", "content": "x", "tool_call_id": "id"}],
            },
            "claude",
        )
        is None
    )


# ── anthropic_response_to_chat returns SimpleNamespace ──────────────────────


def test_anthropic_response_to_chat_basic_mapping() -> None:
    out = anthropic_response_to_chat(
        {
            "id": "msg-1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "model": "claude-3-5-haiku",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 4, "output_tokens": 3},
        },
        "gpt-4o",
    )
    assert out.model == "gpt-4o"
    assert out.object == "chat.completion"
    assert out.choices[0].message.content == "hello"
    assert out.choices[0].finish_reason == "stop"
    assert out.usage.prompt_tokens == 4
    assert out.usage.completion_tokens == 3
    assert out.usage.total_tokens == 7


def test_anthropic_response_to_chat_maps_max_tokens_to_length() -> None:
    out = anthropic_response_to_chat(
        {
            "id": "x",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "claude",
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        "gpt",
    )
    assert out.choices[0].finish_reason == "length"


# ── anthropic_params_to_chat returns plain dict ──────────────────────────────


def test_anthropic_params_to_chat_pushes_system_as_message() -> None:
    out = anthropic_params_to_chat(
        {
            "model": "claude",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "system": "be brief",
        },
        "gpt-4o",
    )
    assert out is not None
    assert out["messages"][0] == {"role": "system", "content": "be brief"}
    assert out["messages"][1] == {"role": "user", "content": "hi"}
    assert out["max_tokens"] == 100


# ── chat_response_to_anthropic returns SimpleNamespace ──────────────────────


def test_chat_response_to_anthropic_wraps_content_block() -> None:
    out = chat_response_to_anthropic(
        {
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi back"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
        "claude-3-5-haiku",
    )
    assert out.model == "claude-3-5-haiku"
    assert out.content[0].type == "text"
    assert out.content[0].text == "hi back"
    assert out.stop_reason == "end_turn"
    assert out.usage.input_tokens == 5
    assert out.usage.output_tokens == 2
