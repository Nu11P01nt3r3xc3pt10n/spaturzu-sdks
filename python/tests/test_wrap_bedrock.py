"""Unit tests for spaturzu._bedrock.wrap_bedrock."""

from __future__ import annotations

from typing import Any

import pytest

from spaturzu._bedrock import wrap_bedrock
from tests.helpers.fake_clients import (
    bedrock_error,
    fake_bedrock,
    fake_openai,
)
from tests.helpers.fake_logger import FakeLogger
from tests.helpers.frame_helpers import in_frame


def _get_tags() -> Any:
    return None


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Handle SimpleNamespace OR dict return shapes from translators."""
    val = getattr(obj, key, None)
    if val is not None:
        return val
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def test_converse_logs_usage() -> None:
    logger = FakeLogger()
    client = fake_bedrock(ok={
        "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 20, "cacheReadInputTokenCount": 4},
    })
    wrapped = wrap_bedrock(client, logger, _get_tags)
    with in_frame("agent-a"):
        wrapped.converse(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
        )
    entry = logger.entries[0]
    assert entry["provider"] == "bedrock"
    assert entry["model"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert entry["status"] == 200
    assert entry["promptTokens"] == 10
    assert entry["completionTokens"] == 20
    assert entry["cachedInputTokens"] == 4
    assert entry["usageSource"] == "provider"
    assert entry["agentName"] == "agent-a"


def test_converse_logs_error_with_status_from_response_metadata() -> None:
    logger = FakeLogger()
    err = bedrock_error(429)
    client = fake_bedrock(err=err)
    wrapped = wrap_bedrock(client, logger, _get_tags)
    with pytest.raises(BaseException):
        wrapped.converse(
            modelId="x", messages=[{"role": "user", "content": [{"text": "hi"}]}]
        )
    assert logger.entries[0]["status"] == 429


def test_converse_stream_observes_metadata_event() -> None:
    logger = FakeLogger()

    # Use plain list to avoid the iscoroutine-misroute issue noted in Plan 1.
    stream_chunks = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Hello"}}},
        {"contentBlockDelta": {"delta": {"text": " world"}}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 2}}},
    ]

    client = fake_bedrock(stream=iter(stream_chunks))
    wrapped = wrap_bedrock(client, logger, _get_tags)
    resp = wrapped.converse_stream(
        modelId="x", messages=[{"role": "user", "content": [{"text": "hi"}]}]
    )
    events = list(resp["stream"])
    assert len(events) == 5
    entry = logger.entries[0]
    assert entry["provider"] == "bedrock"
    assert entry["promptTokens"] == 5
    assert entry["completionTokens"] == 2
    assert entry["usageSource"] == "provider"


def test_fallback_to_openai_on_503() -> None:
    logger = FakeLogger()
    primary = fake_bedrock(err=bedrock_error(503, "ServiceUnavailableException"))
    secondary = fake_openai(ok={
        "usage": {"prompt_tokens": 3, "completion_tokens": 5},
        "choices": [
            {"message": {"role": "assistant", "content": "hi from gpt"}, "finish_reason": "stop"}
        ],
    })
    wrapped = wrap_bedrock(
        primary,
        logger,
        _get_tags,
        None,
        "throw",
        [{"provider": "openai", "client": secondary, "model": "gpt-4o"}],
    )
    resp = wrapped.converse(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
    )
    # Response translated back to Bedrock shape via chat_response_from_bedrock,
    # which returns a plain dict.
    assert resp["output"]["message"]["content"][0]["text"] == "hi from gpt"
    assert len(logger.entries) == 2
    assert logger.entries[0]["provider"] == "bedrock"
    assert logger.entries[0]["status"] == 503
    assert logger.entries[1]["provider"] == "openai"
    assert logger.entries[1]["model"] == "gpt-4o"
    assert logger.entries[1]["tags"]["via"] == "fallback"
