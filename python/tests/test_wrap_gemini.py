"""Unit tests for spaturzu._gemini.wrap_gemini (sync path)."""

from __future__ import annotations

from typing import Any

import pytest

from spaturzu._gemini import wrap_gemini
from tests.helpers.fake_clients import fake_gemini, fake_openai, gemini_error
from tests.helpers.fake_logger import FakeLogger
from tests.helpers.frame_helpers import in_frame


def _get_tags() -> Any:
    return None


class _UsageMeta:
    """Mimic google-genai's typed UsageMetadata object (snake_case attrs)."""

    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


def test_generate_content_logs_usage() -> None:
    logger = FakeLogger()

    class _Resp:
        usage_metadata = _UsageMeta(
            prompt_token_count=10,
            candidates_token_count=20,
            cached_content_token_count=4,
        )

    client = fake_gemini(ok=_Resp())
    wrapped = wrap_gemini(client, logger, _get_tags)
    with in_frame("agent-a"):
        wrapped.models.generate_content(
            model="gemini-2.5-pro",
            contents=[{"role": "user", "parts": [{"text": "hi"}]}],
        )
    entry = logger.entries[0]
    assert entry["provider"] == "gemini"
    assert entry["model"] == "gemini-2.5-pro"
    assert entry["promptTokens"] == 10
    assert entry["completionTokens"] == 20
    assert entry["cachedInputTokens"] == 4
    assert entry["usageSource"] == "provider"


def test_generate_content_logs_error_with_status() -> None:
    logger = FakeLogger()
    err = gemini_error(429)
    client = fake_gemini(err=err)
    wrapped = wrap_gemini(client, logger, _get_tags)
    with pytest.raises(BaseException):
        wrapped.models.generate_content(
            model="gemini-2.5-pro",
            contents=[{"role": "user", "parts": [{"text": "hi"}]}],
        )
    assert logger.entries[0]["status"] == 429


def test_generate_content_stream_keeps_last_seen_usage() -> None:
    logger = FakeLogger()

    class _Chunk:
        def __init__(self, text: str, pt: int, ct: int) -> None:
            class _Part:
                pass

            part = _Part()
            part.text = text

            class _Content:
                pass

            content = _Content()
            content.parts = [part]

            class _Candidate:
                pass

            cand = _Candidate()
            cand.content = content

            self.candidates = [cand]
            self.usage_metadata = _UsageMeta(prompt_token_count=pt, candidates_token_count=ct)

    chunks = [_Chunk("Hello", 3, 1), _Chunk(" world", 3, 3)]

    client = fake_gemini(stream=iter(chunks))
    wrapped = wrap_gemini(client, logger, _get_tags)
    out = wrapped.models.generate_content_stream(
        model="gemini-2.5-pro",
        contents=[{"role": "user", "parts": [{"text": "hi"}]}],
    )
    list(out)  # drain
    assert logger.entries[0]["promptTokens"] == 3
    assert logger.entries[0]["completionTokens"] == 3  # last-seen, not summed


def test_fallback_to_openai_on_503() -> None:
    logger = FakeLogger()
    primary = fake_gemini(err=gemini_error(503))
    secondary = fake_openai(ok={
        "usage": {"prompt_tokens": 4, "completion_tokens": 6},
        "choices": [
            {"message": {"role": "assistant", "content": "hi from gpt"}, "finish_reason": "stop"}
        ],
    })
    wrapped = wrap_gemini(
        primary,
        logger,
        _get_tags,
        None,
        "throw",
        [{"provider": "openai", "client": secondary, "model": "gpt-4o"}],
    )
    resp = wrapped.models.generate_content(
        model="gemini-2.5-pro",
        contents=[{"role": "user", "parts": [{"text": "hi"}]}],
    )
    # `chat_response_from_gemini` returns a plain dict (Gemini-shape).
    assert resp["candidates"][0]["content"]["parts"][0]["text"] == "hi from gpt"
    assert len(logger.entries) == 2
    assert logger.entries[0]["provider"] == "gemini"
    assert logger.entries[1]["provider"] == "openai"
    assert logger.entries[1]["tags"]["via"] == "fallback"
