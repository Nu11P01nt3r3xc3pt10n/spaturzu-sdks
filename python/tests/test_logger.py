"""Unit tests for spaturzu._logger (mirror of test/logger.test.ts)."""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest

from spaturzu._logger import HttpError, Logger


ENTRY: dict[str, Any] = {
    "provider": "openai",
    "model": "gpt-4o",
    "status": 200,
}


class _FakeResponse:
    def __init__(self, status: int, text: str = "") -> None:
        self.status_code = status
        self.text = text


def _patch_post(monkeypatch, behaviour) -> list[dict[str, Any]]:
    """Replace ``httpx.Client.post`` so tests can drive responses + capture calls.

    ``behaviour`` is a callable ``(request_index) -> _FakeResponse | raises``.
    Returns a list that accumulates every captured ``(url, json, headers)``.
    """
    captured: list[dict[str, Any]] = []
    counter = {"n": 0}

    def fake_post(self: httpx.Client, url: str, json: Any, headers: dict[str, str]) -> _FakeResponse:  # type: ignore[override]
        n = counter["n"]
        counter["n"] += 1
        captured.append({"url": url, "json": json, "headers": dict(headers), "n": n})
        return behaviour(n)

    monkeypatch.setattr(httpx.Client, "post", fake_post, raising=True)
    return captured


def test_log_returns_synchronously(monkeypatch) -> None:
    _patch_post(monkeypatch, lambda n: _FakeResponse(200))
    logger = Logger("https://gw.example", backoff_ms=[])
    before = time.monotonic()
    logger.log(ENTRY)
    after = time.monotonic()
    assert (after - before) < 0.05  # ~no work on the calling thread
    logger.flush()
    logger.close()


def test_flush_awaits_inflight(monkeypatch) -> None:
    # Slow first response so flush has to wait.
    def behaviour(_n: int) -> _FakeResponse:
        time.sleep(0.05)
        return _FakeResponse(200)

    _patch_post(monkeypatch, behaviour)
    logger = Logger("https://gw.example", backoff_ms=[])
    logger.log(ENTRY)
    start = time.monotonic()
    logger.flush()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.04
    logger.close()


def test_retries_on_503_then_succeeds(monkeypatch) -> None:
    def behaviour(n: int) -> _FakeResponse:
        return _FakeResponse(503) if n == 0 else _FakeResponse(200)

    captured = _patch_post(monkeypatch, behaviour)
    errors: list[BaseException] = []
    logger = Logger(
        "https://gw.example",
        on_error=lambda e, _entry: errors.append(e),
        backoff_ms=[1],  # 1ms backoff
    )
    logger.log(ENTRY)
    logger.flush()
    # 1 failure → retry that succeeds → 2 total POST attempts; but the 503
    # is raised as HttpError, which is retryable, so no on_error.
    assert len([c for c in captured if c]) == 2
    assert errors == []
    logger.close()


def test_no_retry_on_400(monkeypatch) -> None:
    captured = _patch_post(monkeypatch, lambda _n: _FakeResponse(400, "bad request"))
    errors: list[BaseException] = []
    logger = Logger(
        "https://gw.example",
        on_error=lambda e, _entry: errors.append(e),
        backoff_ms=[1, 1, 1],
    )
    logger.log(ENTRY)
    logger.flush()
    assert len(captured) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], HttpError)
    assert errors[0].status == 400
    logger.close()


def test_gives_up_after_backoff_exhausted(monkeypatch) -> None:
    captured = _patch_post(monkeypatch, lambda _n: _FakeResponse(500))
    errors: list[BaseException] = []
    logger = Logger(
        "https://gw.example",
        on_error=lambda e, _entry: errors.append(e),
        backoff_ms=[1, 1],
    )
    logger.log(ENTRY)
    logger.flush()
    assert len(captured) == 3  # 1 initial + 2 retries
    assert len(errors) == 1
    logger.close()


def test_sets_x_spaturzu_key_when_api_key_set(monkeypatch) -> None:
    captured = _patch_post(monkeypatch, lambda _n: _FakeResponse(200))
    logger = Logger("https://gw.example", api_key="secret", backoff_ms=[])
    logger.log(ENTRY)
    logger.flush()
    assert captured[0]["headers"]["x-spaturzu-key"] == "secret"
    assert captured[0]["headers"]["content-type"] == "application/json"
    logger.close()


def test_omits_x_spaturzu_key_when_api_key_none(monkeypatch) -> None:
    captured = _patch_post(monkeypatch, lambda _n: _FakeResponse(200))
    logger = Logger("https://gw.example", backoff_ms=[])
    logger.log(ENTRY)
    logger.flush()
    assert "x-spaturzu-key" not in captured[0]["headers"]
    logger.close()


def test_posts_entry_as_json_to_v1_logs(monkeypatch) -> None:
    captured = _patch_post(monkeypatch, lambda _n: _FakeResponse(200))
    logger = Logger("https://gw.example/", backoff_ms=[])
    logger.log(ENTRY)
    logger.flush()
    assert captured[0]["url"] == "https://gw.example/v1/logs"
    assert captured[0]["json"] == ENTRY
    logger.close()
