"""Unit tests for spaturzu._openai.wrap_openai (sync path)."""

from __future__ import annotations

from typing import Any

import pytest

from spaturzu._openai import wrap_openai
from tests.helpers.fake_clients import (
    anthropic_error,
    as_async_iterable,
    fake_anthropic,
    fake_openai,
    openai_error,
)
from tests.helpers.fake_logger import FakeLogger
from tests.helpers.frame_helpers import in_frame


def _get_tags() -> Any:
    return None


def test_logs_success_with_usage_and_frame_attribution() -> None:
    logger = FakeLogger()
    client = fake_openai(ok={
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 4},
        },
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
    })
    wrapped = wrap_openai(client, logger, _get_tags)
    with in_frame("agent-a"):
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert len(logger.entries) == 1
    entry = logger.entries[0]
    assert entry["provider"] == "openai"
    assert entry["model"] == "gpt-4o"
    assert entry["status"] == 200
    assert entry["promptTokens"] == 10
    assert entry["completionTokens"] == 20
    assert entry["cachedInputTokens"] == 4
    assert entry["usageSource"] == "provider"
    assert entry["agentName"] == "agent-a"
    assert entry["agentPath"] == ["agent-a"]
    assert isinstance(entry["id"], str)
    assert isinstance(entry["latencyMs"], int)


def test_logs_error_with_status_then_reraises() -> None:
    logger = FakeLogger()
    err = openai_error(429)
    client = fake_openai(err=err)
    wrapped = wrap_openai(client, logger, _get_tags)
    with pytest.raises(BaseException) as exc_info:
        wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
        )
    assert exc_info.value is err
    assert len(logger.entries) == 1
    assert logger.entries[0]["status"] == 429


def test_clamps_unknown_error_status_to_500() -> None:
    logger = FakeLogger()
    client = fake_openai(err=Exception("mystery"))
    wrapped = wrap_openai(client, logger, _get_tags)
    with pytest.raises(Exception):
        wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
        )
    assert logger.entries[0]["status"] == 500
    # Parity with TS test: error entry must not carry token counts.
    assert "promptTokens" not in logger.entries[0]
    assert "completionTokens" not in logger.entries[0]


def test_streaming_yields_chunks_and_logs_once_on_end() -> None:
    logger = FakeLogger()

    # Use a list (not a generator) so asyncio.iscoroutine() returns False —
    # Python 3.11 returns True for generator objects, which would route into
    # the async code path instead of _observe_sync.
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
    ]

    client = fake_openai(stream=chunks)
    wrapped = wrap_openai(client, logger, _get_tags)
    result = wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    collected = list(result)
    assert len(collected) == 3
    assert len(logger.entries) == 1
    assert logger.entries[0]["promptTokens"] == 5
    assert logger.entries[0]["completionTokens"] == 2
    assert logger.entries[0]["usageSource"] == "provider"


def test_fallback_walks_to_anthropic_on_503() -> None:
    logger = FakeLogger()
    primary = fake_openai(err=openai_error(503))
    secondary = fake_anthropic(ok={
        "id": "msg-1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hi from claude"}],
        "model": "claude-3-5-haiku",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 5},
    })
    wrapped = wrap_openai(
        primary,
        logger,
        _get_tags,
        None,  # guard
        "throw",  # on_breach
        [{"provider": "anthropic", "client": secondary, "model": "claude-3-5-haiku-20241022"}],
    )
    resp = wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
    )
    # Response is a SimpleNamespace (translated from Anthropic to OpenAI shape).
    content = resp.choices[0].message.content
    assert content == "hi from claude"
    # Two log entries: primary error + fallback success
    assert len(logger.entries) == 2
    assert logger.entries[0]["provider"] == "openai"
    assert logger.entries[0]["status"] == 503
    assert logger.entries[1]["provider"] == "anthropic"
    assert logger.entries[1]["model"] == "claude-3-5-haiku-20241022"
    assert logger.entries[1]["status"] == 200
    assert logger.entries[1]["tags"]["via"] == "fallback"
