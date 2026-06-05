"""Fire-and-forget log POST with retries.

Mirrors `sdks/typescript/src/logger.ts`. Differences from Node:

* **Concurrency model.** Node has one event loop; we use a
  ``ThreadPoolExecutor`` so the SDK works identically for sync and async
  customer code without touching the customer's event loop. The HTTP client
  is sync (``httpx.Client``) inside each worker; ``time.sleep`` for backoff
  blocks only the worker thread, never the customer.
* **No ``Promise.allSettled``.** ``flush()`` snapshots the inflight set
  and waits via ``concurrent.futures.wait``; the snapshot dance handles
  late-arriving entries the same way Node's recursive flush does.
* **No unhandled-rejection equivalent.** Failed sends call ``on_error`` if
  the customer registered one, then drop. The customer's app must never
  crash because the metering plane hiccupped.
"""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Callable, Optional

import httpx


# Default backoff schedule: matches Node's ``[1s, 2s, 4s, 8s, 16s]``.
# Each entry is the wait *before* the next retry attempt.
DEFAULT_BACKOFF_MS: tuple[int, ...] = (1_000, 2_000, 4_000, 8_000, 16_000)
DEFAULT_MAX_CONCURRENT = 50
DEFAULT_TIMEOUT_S = 10.0


class HttpError(Exception):
    """HTTP status carried explicitly so the retry loop can branch on it."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _is_retryable(err: BaseException) -> bool:
    """5xx and 429 → retry; other 4xx → fatal; network errors → retry.

    Validation/auth errors (400, 401, 403) won't resolve by retrying;
    queue space is more useful retrying transient gateway / network blips.
    """
    if isinstance(err, HttpError):
        return err.status >= 500 or err.status == 429
    return True


def _jitter(ms: float) -> float:
    """±25% jitter so a hundred SDKs reconnecting after a gateway blip don't
    all retry on the same boundary."""
    return ms * (0.75 + random.random() * 0.5)


class Logger:
    """Background-thread log poster.

    ``log(entry)`` returns immediately; the POST runs on a worker thread.
    Failures call ``on_error`` (default: silent) so the customer's app
    keeps running through metering-plane outages. ``flush()`` blocks until
    every queued entry has either succeeded or burned its retry budget —
    the escape hatch for short-lived processes (CLIs, lambdas).
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        on_error: Optional[Callable[[BaseException, dict[str, Any]], None]] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        backoff_ms: tuple[int, ...] | list[int] | None = None,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._on_error = on_error or (lambda _e, _entry: None)
        self._timeout_s = timeout_s
        # Tuple-fy + freeze defensively: callers occasionally pass lists.
        self._backoff_ms: tuple[int, ...] = tuple(
            backoff_ms if backoff_ms is not None else DEFAULT_BACKOFF_MS
        )
        # ThreadPoolExecutor caps both concurrency *and* parallelism — its
        # internal queue handles excess submissions FIFO. 50 matches Node's
        # default semaphore.
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent, thread_name_prefix="spaturzu-log"
        )
        # Sync httpx.Client is shared across worker threads — its connection
        # pool is thread-safe per the docs and reuses sockets across calls.
        self._client = httpx.Client(timeout=timeout_s)
        self._inflight: set[Future[Any]] = set()
        self._lock = threading.Lock()
        self._closed = False

    def log(self, entry: dict[str, Any]) -> None:
        """Schedule one log POST. Returns immediately; never raises."""
        if self._closed:
            # After flush + close, refuse new entries rather than queue
            # against a dead executor.
            return
        future = self._executor.submit(self._send_with_retries, entry)
        # Attach the entry to the future so the done-callback can pass it
        # to on_error without a second mapping. ``Future`` accepts arbitrary
        # attribute assignment — a private-namespace attribute is the
        # simplest correlation and avoids holding a separate dict + lock.
        setattr(future, "_spaturzu_entry", entry)
        with self._lock:
            self._inflight.add(future)
        future.add_done_callback(self._on_future_done)

    def _on_future_done(self, future: Future[Any]) -> None:
        with self._lock:
            self._inflight.discard(future)
        # Surface the failure here — exceptions raised inside the worker
        # thread are stored on the future and would otherwise be silent
        # until someone called ``future.result()``.
        if future.exception() is not None:
            err = future.exception()
            if err is not None:
                # Recover the entry from the closure so on_error gets it.
                # We stash the entry on the future via a side dict — see
                # _send_with_retries which does not have access here. To
                # keep things simple, on_error receives ({}, err) when we
                # can't retrieve. (The Node SDK passes both; matching by
                # storing the entry on a custom future attribute below.)
                entry = getattr(future, "_spaturzu_entry", {}) or {}
                try:
                    self._on_error(err, entry)
                except Exception:
                    # An on_error handler that itself raises must not crash
                    # the dispatch thread.
                    pass

    def flush(self, timeout_s: Optional[float] = None) -> None:
        """Block until every queued entry has settled.

        Re-snapshots while waiting: a worker that retries can outlive its
        original ``submit``, and a slow ``flush`` shouldn't return early
        if more entries got queued mid-wait. The loop exits once the
        inflight set is empty.
        """
        deadline = time.monotonic() + timeout_s if timeout_s else None
        while True:
            with self._lock:
                snapshot = list(self._inflight)
            if not snapshot:
                return
            remaining = (
                None
                if deadline is None
                else max(0.0, deadline - time.monotonic())
            )
            wait(snapshot, timeout=remaining)
            if deadline is not None and time.monotonic() >= deadline:
                return

    def close(self) -> None:
        """Signal no more entries; release executor + httpx.Client.

        Customers don't typically call this — Python's process exit
        cleans up. Provided for tests and explicit teardown.
        """
        self._closed = True
        self._executor.shutdown(wait=False)
        self._client.close()

    # ─── internals ──────────────────────────────────────────────────────

    def _send_with_retries(self, entry: dict[str, Any]) -> None:
        last_err: Optional[BaseException] = None
        max_attempts = len(self._backoff_ms) + 1
        for attempt in range(max_attempts):
            try:
                self._send(entry)
                return
            except BaseException as err:  # noqa: BLE001 — we re-raise non-retryable
                last_err = err
                if not _is_retryable(err):
                    raise
                if attempt == max_attempts - 1:
                    raise
                time.sleep(_jitter(self._backoff_ms[attempt]) / 1000.0)
        # Unreachable: loop exits via return or raise.
        if last_err is not None:
            raise last_err

    def _send(self, entry: dict[str, Any]) -> None:
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["x-spaturzu-key"] = self._api_key
        try:
            res = self._client.post(
                f"{self._base_url}/v1/logs",
                json=entry,
                headers=headers,
            )
        except httpx.HTTPError as e:
            # Network / timeout / DNS failures bubble as plain Exception.
            # Preserve the original via __cause__ when wrapping.
            raise RuntimeError(f"spaturzu: POST /v1/logs network error: {e}") from e
        if res.status_code >= 400:
            body = (res.text or "")[:200]
            raise HttpError(
                res.status_code,
                f"spaturzu: POST /v1/logs {res.status_code}: {body}",
            )
