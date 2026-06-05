"""Unit tests for spaturzu._fallback dispatcher + retryable classifier."""

from __future__ import annotations

import pytest

from spaturzu._fallback import (
    is_retryable_upstream_error,
    try_fallback_sync,
)
from tests.helpers.fake_clients import (
    anthropic_error,
    fake_anthropic,
    fake_openai,
    openai_error,
)


# is_retryable_upstream_error ----------------------------------------------


def test_retryable_for_429() -> None:
    assert is_retryable_upstream_error(openai_error(429))


def test_retryable_for_5xx() -> None:
    assert is_retryable_upstream_error(openai_error(503))
    assert is_retryable_upstream_error(openai_error(500))


def test_retryable_for_408() -> None:
    assert is_retryable_upstream_error(openai_error(408))


def test_not_retryable_for_400_401_422() -> None:
    assert not is_retryable_upstream_error(openai_error(400))
    assert not is_retryable_upstream_error(openai_error(401))
    assert not is_retryable_upstream_error(openai_error(422))


def test_retryable_for_known_sdk_names() -> None:
    assert is_retryable_upstream_error(openai_error(0, "APIConnectionError"))
    assert is_retryable_upstream_error(openai_error(0, "APITimeoutError"))
    assert is_retryable_upstream_error(openai_error(0, "RateLimitError"))
    assert is_retryable_upstream_error(openai_error(0, "InternalServerError"))


def test_retryable_returns_false_for_non_objects() -> None:
    assert not is_retryable_upstream_error(None)
    assert not is_retryable_upstream_error("string")


# try_fallback_sync ------------------------------------------------------


def test_identity_openai_to_openai_passes_params_through() -> None:
    target_client = fake_openai(ok={
        "usage": {"prompt_tokens": 5, "completion_tokens": 7},
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
    })
    outcome = try_fallback_sync(
        "openai",
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        {"provider": "openai", "client": target_client, "model": "gpt-4o-mini"},
    )
    assert outcome["kind"] == "success"
    # Identity → model overridden, rest unchanged.
    call_args = target_client.__create.call_args.kwargs
    assert call_args["model"] == "gpt-4o-mini"


def test_cross_shape_openai_to_anthropic_translates_params_and_response() -> None:
    target_client = fake_anthropic(ok={
        "id": "msg-1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hello"}],
        "model": "claude-3-5-haiku",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 4, "output_tokens": 3},
    })
    outcome = try_fallback_sync(
        "openai",
        {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ],
        },
        {"provider": "anthropic", "client": target_client, "model": "claude-3-5-haiku-20241022"},
    )
    assert outcome["kind"] == "success"
    call_args = target_client.__create.call_args.kwargs
    # Anthropic call: system pulled out, max_tokens defaulted.
    assert call_args["system"] == "be brief"
    assert call_args["max_tokens"] == 1024
    # Response translated back to chat shape, caller-model preserved.
    # anthropic_response_to_chat returns a SimpleNamespace, not a dict.
    resp = outcome["response"]
    assert resp.object == "chat.completion"
    assert resp.model == "gpt-4o"


def test_unsupported_when_source_has_stream(monkeypatch) -> None:
    target_client = fake_anthropic(ok={})
    outcome = try_fallback_sync(
        "openai",
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        {"provider": "anthropic", "client": target_client, "model": "claude"},
    )
    assert outcome["kind"] == "unsupported"
    target_client.__create.assert_not_called()


def test_error_when_target_throws() -> None:
    err = anthropic_error(503)
    target_client = fake_anthropic(err=err)
    outcome = try_fallback_sync(
        "openai",
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        {"provider": "anthropic", "client": target_client, "model": "claude"},
    )
    assert outcome["kind"] == "error"
    assert outcome["error"] is err
    assert outcome["status"] == 503
