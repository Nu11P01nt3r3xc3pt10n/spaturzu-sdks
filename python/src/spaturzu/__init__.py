"""Public surface of ``spaturzu`` (Python).

Mirrors `sdks/typescript/src/index.ts`. Usage::

    from spaturzu import spaturzu
    from openai import OpenAI

    spaturzu = spaturzu(base_url="...", api_key="...", tags={"env": "prod"})
    openai = spaturzu.wrap_openai(OpenAI())

    with spaturzu.run("researcher"):
        r = openai.chat.completions.create(model=..., messages=[...])

    spaturzu.flush()    # blocks until queued log POSTs settle

The same code works for ``AsyncOpenAI`` if you also use ``async with``::

    async with spaturzu.run("researcher", tags={"team": "search"}):
        r = await openai.chat.completions.create(...)

    await asyncio.to_thread(spaturzu.flush)   # flush is sync — stays
                                              # inside the dedicated thread pool
"""

from __future__ import annotations

import os
from contextvars import Token
from typing import Any, Callable, Optional, Union

from ._anthropic import wrap_anthropic as _wrap_anthropic
from ._bedrock import wrap_bedrock as _wrap_bedrock
from ._gemini import wrap_gemini as _wrap_gemini
from ._mistral import wrap_mistral as _wrap_mistral
from ._budget import BudgetExceededError, BudgetGuard
from ._context import (
    RunFrame,
    close_frame,
    get_current_frame,
    normalise_tags,
    open_frame,
)
from ._fallback import FallbackTarget
from ._logger import HttpError, Logger
from ._openai import wrap_openai as _wrap_openai


# Customers can pass numbers / booleans alongside strings; the wire format
# is ``Record<string, string>`` so we coerce at the boundary.
TagInput = dict[str, Union[str, int, float, bool]]


__all__ = [
    "spaturzu",
    "RunFrame",
    "HttpError",
    "BudgetExceededError",
    "FallbackTarget",
    "get_current_frame",
    "configure",
    "run",
    "flush",
    "shutdown",
    "get_default_spaturzu",
]


# Day 20 — wrap option shape: ``{"hard_cap": True, "on_breach": "throw"}``.
# Re-declared here for clarity; runtime treats it as a plain dict.
BudgetWrapOptions = dict[str, Any]


class _RunCtx:
    """Polymorphic context manager — supports both ``with`` and ``async with``.

    The frame is opened lazily in ``__enter__`` / ``__aenter__`` (not
    ``__init__``) so that ``spaturzu.run("x")`` without a ``with`` doesn't
    silently leak a frame onto the ContextVar.
    """

    __slots__ = ("_agent_name", "_tags", "_token")

    def __init__(
        self, agent_name: str, tags: Optional[dict[str, str]] = None
    ) -> None:
        self._agent_name = agent_name
        self._tags = tags
        self._token: Optional[Token] = None

    def _open(self) -> RunFrame:
        if self._token is not None:
            raise RuntimeError(
                "spaturzu.run: this run context is already open — reusing "
                "the same returned object inside two with-blocks is not "
                "supported. Call spaturzu.run() again for a new frame."
            )
        frame, token = open_frame(self._agent_name, self._tags)
        self._token = token
        return frame

    def _close(self) -> None:
        if self._token is not None:
            close_frame(self._token)
            self._token = None

    # Sync context manager.
    def __enter__(self) -> RunFrame:
        return self._open()

    def __exit__(self, *exc: Any) -> bool:
        self._close()
        return False

    # Async context manager.
    async def __aenter__(self) -> RunFrame:
        return self._open()

    async def __aexit__(self, *exc: Any) -> bool:
        self._close()
        return False


class spaturzu:
    """Top-level SDK client.

    Constructor knobs follow the Node SDK 1:1:

    * ``base_url`` — gateway URL. Falls back to ``SPATURZU_BASE_URL`` env,
      then the hosted gateway ``https://spaturzu-api.superchiu.org``.
    * ``api_key`` — project key. Falls back to ``SPATURZU_API_KEY`` env.
      Optional in dev.
    * ``timeout_s`` — per-attempt POST timeout. Default 10s.
    * ``backoff_ms`` — retry backoff schedule. Default
      ``[1000, 2000, 4000, 8000, 16000]``. Pass ``[]`` to disable retries.
    * ``max_concurrent`` — max in-flight log POSTs. Default 50.
    * ``on_error`` — invoked when a POST fails after all retries.
      Receives ``(exception, log_entry_dict)``. Default: silent.
    * ``tags`` — process-wide tags merged into every entry. Per-run tags
      (passed to ``run()``) override these on key conflict.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 10.0,
        backoff_ms: Optional[list[int]] = None,
        max_concurrent: int = 50,
        on_error: Optional[
            Callable[[BaseException, dict[str, Any]], None]
        ] = None,
        tags: Optional[TagInput] = None,
    ) -> None:
        resolved_base = (
            base_url
            or os.environ.get("SPATURZU_BASE_URL")
            or "https://spaturzu-api.superchiu.org"
        )
        resolved_key = api_key or os.environ.get("SPATURZU_API_KEY")
        self._base_url = resolved_base.rstrip("/")
        self._api_key = resolved_key
        self._on_error_cb = on_error
        self._logger = Logger(
            base_url=self._base_url,
            api_key=resolved_key,
            on_error=on_error,
            timeout_s=timeout_s,
            backoff_ms=backoff_ms,
            max_concurrent=max_concurrent,
        )
        self._process_tags = normalise_tags(tags)
        # Day 20: lazily-constructed BudgetGuard, shared across every wrap
        # on this spaturzu instance. One policy cache + one SSE thread per
        # process is enough.
        self._guard: Optional[BudgetGuard] = None

    # The wrappers consult this getter at log-build time so that future
    # support for ``spaturzu.set_tags(...)`` (mutating process tags) can be
    # added without re-wrapping every client.
    def _get_process_tags(self) -> Optional[dict[str, str]]:
        return self._process_tags

    def run(
        self, agent_name: str, *, tags: Optional[TagInput] = None
    ) -> _RunCtx:
        """Open a frame for ``agent_name`` and return a context manager.

        Use as ``with spaturzu.run("name"):`` (sync) or
        ``async with spaturzu.run("name"):`` (async). Nested runs share
        ``run_id`` and extend ``agent_path``.
        """
        return _RunCtx(agent_name, normalise_tags(tags))

    def _ensure_guard(self) -> BudgetGuard:
        """Lazily construct the shared BudgetGuard. Started on first use."""
        if self._guard is None:
            self._guard = BudgetGuard(
                base_url=self._base_url,
                api_key=self._api_key,
                on_error=(
                    (lambda err: self._on_error_cb(err, {}))
                    if self._on_error_cb is not None
                    else None
                ),
            )
            self._guard.start()
        return self._guard

    def _resolve_budget(
        self, budget: Optional[dict[str, Any]]
    ) -> tuple[Optional[BudgetGuard], str]:
        """Decode the wrap-level ``budget`` kwarg into (guard, on_breach)."""
        if not budget or not budget.get("hard_cap"):
            return None, "throw"
        guard = self._ensure_guard()
        on_breach = str(budget.get("on_breach", "throw"))
        if on_breach not in ("throw", "warn"):
            on_breach = "throw"
        return guard, on_breach

    def wrap_openai(
        self,
        client: Any,
        *,
        budget: Optional[dict[str, Any]] = None,
        fallback: Optional[list[FallbackTarget]] = None,
    ) -> Any:
        """Wrap an OpenAI / AsyncOpenAI client. The original is untouched.

        Day 20: pass ``budget={"hard_cap": True}`` to enforce hard-cap
        budgets. The wrapped ``create`` raises ``BudgetExceededError``
        before hitting the provider when any applicable hard-cap budget
        is breached. ``on_breach="warn"`` logs and lets the call through.

        Day 24: pass ``fallback=[{"provider": "anthropic", "client": ...,
        "model": "..."}]`` to declare a cross-provider failover chain.
        On a retryable upstream error (429 / 5xx / connection-class) the
        wrap walks the chain and translates request/response across
        providers. v1 scope: non-streaming chat completions, text content
        only, no tools / response_format. Mixing sync and async clients
        across the chain is rejected at runtime with a clear error.
        """
        guard, on_breach = self._resolve_budget(budget)
        return _wrap_openai(
            client,
            self._logger,
            self._get_process_tags,
            guard,
            on_breach,
            fallback,
        )

    def wrap_anthropic(
        self,
        client: Any,
        *,
        budget: Optional[dict[str, Any]] = None,
        fallback: Optional[list[FallbackTarget]] = None,
    ) -> Any:
        """Wrap an Anthropic / AsyncAnthropic client. The original is untouched.

        See ``wrap_openai`` for the ``budget`` and ``fallback`` options.
        """
        guard, on_breach = self._resolve_budget(budget)
        return _wrap_anthropic(
            client,
            self._logger,
            self._get_process_tags,
            guard,
            on_breach,
            fallback,
        )

    def wrap_bedrock(
        self,
        client: Any,
        *,
        budget: Optional[dict[str, Any]] = None,
        fallback: Optional[list[FallbackTarget]] = None,
    ) -> Any:
        """Wrap a boto3 ``bedrock-runtime`` client. The original is untouched.

        Targets the named-method form (``client.converse(...)`` /
        ``client.converse_stream(...)``). v1 is sync-only.

        See ``wrap_openai`` for the ``budget`` / ``fallback`` options.

        Note: On the happy path the wrap returns whatever boto3 returns
        (the raw Bedrock ``converse`` response dict with its native shape).
        When a fallback target serves the call, the response is a **plain
        dict** in Bedrock shape
        (``{"output": {"message": ...}, "stopReason": ..., "usage": ...}``) —
        use subscript access (``resp["output"]["message"]["content"][0]["text"]``)
        in code paths that may run after a fallback.
        """
        guard, on_breach = self._resolve_budget(budget)
        return _wrap_bedrock(
            client,
            self._logger,
            self._get_process_tags,
            guard,
            on_breach,
            fallback,
        )

    def wrap_gemini(
        self,
        client: Any,
        *,
        budget: Optional[dict[str, Any]] = None,
        fallback: Optional[list[FallbackTarget]] = None,
    ) -> Any:
        """Wrap a google-genai client. The original is untouched.

        Intercepts both ``client.models.*`` (sync) and ``client.aio.models.*``
        (async). See ``wrap_openai`` for the ``budget`` / ``fallback`` options.

        Note: On the happy path the wrap returns whatever google-genai returns
        (typed ``GenerateContentResponse`` with attribute access). When a
        fallback target serves the call, the response is a **plain dict** in
        Gemini shape (``{"candidates": [...], "usageMetadata": {...}}``) — use
        subscript access (``resp["candidates"][0]["content"]["parts"][0]["text"]``)
        in code paths that may run after a fallback.
        """
        guard, on_breach = self._resolve_budget(budget)
        return _wrap_gemini(
            client,
            self._logger,
            self._get_process_tags,
            guard,
            on_breach,
            fallback,
        )

    def wrap_mistral(
        self,
        client: Any,
        *,
        budget: Optional[dict[str, Any]] = None,
        fallback: Optional[list[FallbackTarget]] = None,
    ) -> Any:
        """Wrap a mistralai client. The original is untouched.

        Intercepts ``client.chat.{complete, stream, complete_async,
        stream_async}``. See ``wrap_openai`` for the ``budget`` /
        ``fallback`` options.

        Note: On the happy path the wrap returns whatever mistralai
        returns (typed response objects with attribute access). When a
        fallback target serves the call, the response is a **plain dict**
        in Mistral shape (``{choices: [...], usage: {...}}``) — use
        subscript access in code paths that may run after a fallback.
        """
        guard, on_breach = self._resolve_budget(budget)
        return _wrap_mistral(
            client,
            self._logger,
            self._get_process_tags,
            guard,
            on_breach,
            fallback,
        )

    def flush(self, timeout_s: Optional[float] = None) -> None:
        """Block until every queued log POST has settled.

        Call before exiting short-lived processes (CLIs, lambdas, CI
        scripts). Long-running servers can ignore it — the executor lives
        for the process lifetime.
        """
        self._logger.flush(timeout_s=timeout_s)

    def close(self) -> None:
        """Stop accepting new entries and release the executor / HTTP pool.

        Most processes don't need to call this — interpreter shutdown
        cleans up. Provided for explicit teardown in tests.
        """
        self._logger.close()

    def shutdown(self) -> None:
        """Day 20: tear down the BudgetGuard's SSE + polling threads plus
        flush the log queue. Call before process exit on short-lived
        scripts so the open SSE socket doesn't keep the event loop / GIL
        alive. Long-running servers can ignore — daemon threads + the
        log executor clean up on interpreter shutdown.
        """
        if self._guard is not None:
            self._guard.close()
            self._guard = None
        self._logger.flush()


# ── Default singleton + top-level helpers (drop-in surface) ──────────────
# Mirrors sdks/typescript/src/default.ts. `spaturzu()` reads SPATURZU_* from env,
# so the zero-config path works with no configure() call.

_default: Optional["spaturzu"] = None
_configured: Optional[dict[str, Any]] = None


def configure(**opts: Any) -> None:
    """Set process-wide spaturzu options for the default instance (tags,
    base_url, api_key, on_error, ...). MUST be called before the first
    spaturzu.<provider> drop-in client is constructed — raises otherwise."""
    global _configured
    if _default is not None:
        raise RuntimeError(
            "spaturzu.configure() must be called before constructing any "
            "spaturzu.<provider> drop-in client"
        )
    _configured = opts


def get_default_spaturzu() -> "spaturzu":
    """Return the lazily-constructed default spaturzu instance (cached)."""
    global _default
    if _default is None:
        _default = spaturzu(**(_configured or {}))
    return _default


def run(agent_name: str, *, tags: Optional[TagInput] = None) -> _RunCtx:
    """Open a frame on the default instance. Use as `with run("agent"):`."""
    return get_default_spaturzu().run(agent_name, tags=tags)


def flush(timeout_s: Optional[float] = None) -> None:
    """Flush the default instance's queued log POSTs."""
    get_default_spaturzu().flush(timeout_s=timeout_s)


def shutdown() -> None:
    """Tear down the default instance's guard + flush its log queue."""
    get_default_spaturzu().shutdown()


def __reset_default_for_tests() -> None:
    """Test-only: reset the singleton + pending config between cases."""
    global _default, _configured
    _default = None
    _configured = None
