"""Unit tests for spaturzu._anthropic.wrap_anthropic (sync path)."""

from __future__ import annotations

from typing import Any

import pytest

from spaturzu._anthropic import wrap_anthropic
from tests.helpers.fake_clients import (
    anthropic_error,
    as_async_iterable,
    fake_anthropic,
    fake_openai,
)
from tests.helpers.fake_logger import FakeLogger
from tests.helpers.frame_helpers import in_frame


def _get_tags() -> Any:
    return None


def test_logs_success_with_cache_read_as_cached_input_tokens() -> None:
    logger = FakeLogger()
    client = fake_anthropic(ok={
        "id": "msg-1",
        "type": "message",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 2,
        },
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
    })
    wrapped = wrap_anthropic(client, logger, _get_tags)
    with in_frame("agent"):
        wrapped.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
        )
    entry = logger.entries[0]
    assert entry["provider"] == "anthropic"
    assert entry["model"] == "claude-3-5-haiku-20241022"
    assert entry["status"] == 200
    assert entry["promptTokens"] == 10
    assert entry["completionTokens"] == 5
    assert entry["cachedInputTokens"] == 3
    assert entry["usageSource"] == "provider"
    # cache_creation is the WRITE side and must NOT appear on the wire.
    assert "cacheCreationInputTokens" not in entry


def test_logs_error_status_then_reraises() -> None:
    logger = FakeLogger()
    err = anthropic_error(529, "OverloadedError")
    client = fake_anthropic(err=err)
    wrapped = wrap_anthropic(client, logger, _get_tags)
    with pytest.raises(BaseException) as exc_info:
        wrapped.messages.create(
            model="claude",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
        )
    assert exc_info.value is err
    assert logger.entries[0]["status"] == 529


def test_streaming_aggregates_cumulative_output_tokens() -> None:
    logger = FakeLogger()

    # Use a plain list so the sync path is taken (earlier task discovered
    # that generator-functions interact unexpectedly with `iscoroutine`).
    stream_chunks = [
        {
            "type": "message_start",
            "message": {
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 4,
                }
            },
        },
        {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hi"},
        },
        {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": " there"},
        },
        {
            "type": "message_delta",
            "usage": {"output_tokens": 8},
        },
    ]

    client = fake_anthropic(stream=iter(stream_chunks))
    wrapped = wrap_anthropic(client, logger, _get_tags)
    result = wrapped.messages.create(
        model="claude",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        stream=True,
    )
    list(result)  # drain
    assert len(logger.entries) == 1
    assert logger.entries[0]["promptTokens"] == 12
    assert logger.entries[0]["completionTokens"] == 8
    assert logger.entries[0]["cachedInputTokens"] == 4
    assert logger.entries[0]["usageSource"] == "provider"


def test_fallback_walks_to_openai_on_503() -> None:
    logger = FakeLogger()
    primary = fake_anthropic(err=anthropic_error(503))
    secondary = fake_openai(ok={
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi from gpt"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 6, "total_tokens": 8},
    })
    wrapped = wrap_anthropic(
        primary,
        logger,
        _get_tags,
        None,
        "throw",
        [{"provider": "openai", "client": secondary, "model": "gpt-4o"}],
    )
    resp = wrapped.messages.create(
        model="claude-3-5-haiku-20241022",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
    )
    # `chat_response_to_anthropic` returns SimpleNamespace (see
    # spaturzu._translate._to_namespace); the fallback response is always
    # this shape, never dict.
    assert resp.content[0].text == "hi from gpt"
    assert len(logger.entries) == 2
    assert logger.entries[1]["provider"] == "openai"
    assert logger.entries[1]["model"] == "gpt-4o"
    assert logger.entries[1]["promptTokens"] == 2
    assert logger.entries[1]["completionTokens"] == 6
