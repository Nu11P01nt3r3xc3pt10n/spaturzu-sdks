"""Unit tests for spaturzu._mistral.wrap_mistral (sync path)."""

from __future__ import annotations

from typing import Any

import pytest

from spaturzu._mistral import wrap_mistral
from tests.helpers.fake_clients import fake_mistral, fake_openai, mistral_error
from tests.helpers.fake_logger import FakeLogger
from tests.helpers.frame_helpers import in_frame


def _get_tags() -> Any:
    return None


class _Usage:
    """Mimic mistralai's typed usage object (snake_case attrs)."""

    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


def test_complete_logs_usage() -> None:
    logger = FakeLogger()

    class _Resp:
        id = "cmpl-1"
        object = "chat.completion"
        created = 0
        model = "mistral-large-latest"
        choices: list[Any] = []
        usage = _Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    client = fake_mistral(ok=_Resp())
    wrapped = wrap_mistral(client, logger, _get_tags)
    with in_frame("agent-a"):
        wrapped.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "hi"}],
        )
    entry = logger.entries[0]
    assert entry["provider"] == "mistral"
    assert entry["model"] == "mistral-large-latest"
    assert entry["promptTokens"] == 10
    assert entry["completionTokens"] == 20
    assert entry["usageSource"] == "provider"
    # Mistral has no cache; cachedInputTokens must not be set.
    assert "cachedInputTokens" not in entry


def test_complete_logs_error_with_status() -> None:
    logger = FakeLogger()
    err = mistral_error(429)
    client = fake_mistral(err=err)
    wrapped = wrap_mistral(client, logger, _get_tags)
    with pytest.raises(BaseException):
        wrapped.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert logger.entries[0]["status"] == 429


def test_stream_reads_usage_from_data_envelope() -> None:
    logger = FakeLogger()

    # Use plain list (avoid generator/iscoroutine issue per Plan 1)
    chunks = [
        {"data": {"choices": [{"delta": {"content": "Hello"}}]}},
        {"data": {"choices": [{"delta": {"content": " world"}}]}},
        {"data": {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}},
    ]

    client = fake_mistral(stream=iter(chunks))
    wrapped = wrap_mistral(client, logger, _get_tags)
    out = wrapped.chat.stream(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": "hi"}],
    )
    list(out)  # drain
    entry = logger.entries[0]
    assert entry["provider"] == "mistral"
    assert entry["promptTokens"] == 5
    assert entry["completionTokens"] == 2
    assert entry["usageSource"] == "provider"


def test_fallback_to_openai_on_503() -> None:
    logger = FakeLogger()
    primary = fake_mistral(err=mistral_error(503))
    secondary = fake_openai(ok={
        "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
        "choices": [
            {"message": {"role": "assistant", "content": "hi from gpt"}, "finish_reason": "stop"}
        ],
    })
    wrapped = wrap_mistral(
        primary,
        logger,
        _get_tags,
        None,
        "throw",
        [{"provider": "openai", "client": secondary, "model": "gpt-4o"}],
    )
    resp = wrapped.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": "hi"}],
    )
    # chat_response_from_mistral returns a plain dict (Mistral shape).
    assert resp["choices"][0]["message"]["content"] == "hi from gpt"
    assert len(logger.entries) == 2
    assert logger.entries[0]["provider"] == "mistral"
    assert logger.entries[0]["status"] == 503
    assert logger.entries[1]["provider"] == "openai"
    assert logger.entries[1]["tags"]["via"] == "fallback"
