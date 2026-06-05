"""OpenAI client wrapper.

Mirrors `sdks/typescript/src/openai.ts`. Wraps both ``OpenAI`` (sync) and
``AsyncOpenAI`` (async). Detection is done once at construction via
``asyncio.iscoroutinefunction(client.chat.completions.create)``; the
returned proxy dispatches to the matching ``create`` implementation.

We don't import ``openai`` here — it's an optional integration. The
wrapper duck-types against the runtime shape we actually use
(``client.chat.completions.create``), so customers can be on any modern
``openai`` version.

Streaming model: returns an iterable / async-iterable that passes chunks
through to the consumer untouched while observing usage + completion text
for the logger. ``stream_options.include_usage = True`` is auto-injected
so OpenAI emits the final usage chunk; on streams that don't honour it
(Together / Groq / Anyscale / vLLM-self-hosted), we fall back to tiktoken
on the captured prompt + completion text and flag the row as estimated.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Callable, Iterator, Optional
from ._uuid import uuid7

from . import _tiktoken
from ._budget import BudgetGuard
from ._context import (
    AgentBinding,
    RunFrame,
    derive_frame,
    get_current_frame,
    merge_tags,
    normalise_tags,
    set_last_request_id,
)
from ._fallback import (
    FallbackTarget,
    build_attempt_entry,
    is_retryable_upstream_error,
    try_fallback_async,
    try_fallback_sync,
)
from ._logger import Logger


ProcessTagsGetter = Callable[[], Optional[dict[str, str]]]


# ─── log-entry builders ─────────────────────────────────────────────────


def _base_entry(
    frame: Optional[RunFrame],
    request_id: str,
    model: str,
    started_at: float,
    get_process_tags: ProcessTagsGetter,
) -> dict[str, Any]:
    """Build the common fields. Frame data + tags merged in once here so
    the streaming and non-streaming branches both go through one path.

    ``frame`` is captured at call-time by the caller and threaded in
    explicitly (rather than read via ``get_current_frame()`` here) so the
    streaming / async paths — whose log entry is built later, after the
    frame has been popped — still attribute to the right agent."""
    tags = merge_tags(get_process_tags(), frame.tags if frame else None)
    entry: dict[str, Any] = {
        "id": request_id,
        "provider": "openai",
        "model": model,
        "status": 200,  # overridden by builders below
    }
    # `started_at` parameter stays — it's the local stopwatch used for
    # latencyMs further down. The wire-level startedAt field was dropped
    # 2026-05-16; gateway now uses defaultNow() for created_at and the
    # advisory-lock dedup means retries see the original row's timestamp.
    if frame is not None:
        entry["runId"] = frame.run_id
        if frame.parent_request_id is not None:
            entry["parentRequestId"] = frame.parent_request_id
        entry["agentName"] = frame.agent_name
        entry["agentPath"] = list(frame.agent_path)
    if tags:
        entry["tags"] = tags
    return entry


def _build_success(
    frame: Optional[RunFrame],
    request_id: str,
    model: str,
    started_at: float,
    get_process_tags: ProcessTagsGetter,
    usage: Optional[Any],
) -> dict[str, Any]:
    entry = _base_entry(frame, request_id, model, started_at, get_process_tags)
    entry["status"] = 200
    entry["latencyMs"] = int((time.time() - started_at) * 1000)
    if usage is not None:
        # The OpenAI Python SDK exposes usage as either a typed object with
        # attribute access OR a plain dict (depending on version + raw chunk).
        # Read both shapes via _attr.
        prompt_tokens = _attr(usage, "prompt_tokens")
        completion_tokens = _attr(usage, "completion_tokens")
        details = _attr(usage, "prompt_tokens_details")
        cached = _attr(details, "cached_tokens") if details is not None else None
        if isinstance(prompt_tokens, int):
            entry["promptTokens"] = prompt_tokens
        if isinstance(completion_tokens, int):
            entry["completionTokens"] = completion_tokens
        if isinstance(cached, int):
            entry["cachedInputTokens"] = cached
        entry["usageSource"] = "provider"
    return entry


def _build_error(
    frame: Optional[RunFrame],
    request_id: str,
    model: str,
    started_at: float,
    get_process_tags: ProcessTagsGetter,
    err: BaseException,
) -> dict[str, Any]:
    entry = _base_entry(frame, request_id, model, started_at, get_process_tags)
    # OpenAI's APIError carries ``status_code``. Fall back to 500 so the
    # column is never null (gateway requires status 100..599).
    raw = _attr(err, "status_code")
    if not isinstance(raw, int):
        raw = _attr(err, "status")
    status = raw if isinstance(raw, int) else 500
    entry["status"] = max(100, min(599, status))
    entry["latencyMs"] = int((time.time() - started_at) * 1000)
    return entry


def _build_estimated(
    frame: Optional[RunFrame],
    request_id: str,
    model: str,
    started_at: float,
    get_process_tags: ProcessTagsGetter,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    entry = _base_entry(frame, request_id, model, started_at, get_process_tags)
    entry["status"] = 200
    entry["latencyMs"] = int((time.time() - started_at) * 1000)
    entry["promptTokens"] = prompt_tokens
    entry["completionTokens"] = completion_tokens
    entry["usageSource"] = "tiktoken"
    return entry


def _attr(obj: Any, name: str) -> Any:
    """Read either ``obj.name`` or ``obj[name]``. The OpenAI / Anthropic
    Python SDKs return Pydantic models for typed responses but raw dicts
    in some streaming paths — this collapses both shapes."""
    if obj is None:
        return None
    val = getattr(obj, name, None)
    if val is not None:
        return val
    if isinstance(obj, dict):
        return obj.get(name)
    return None


# ─── prompt-text extraction (tiktoken fallback input) ───────────────────


def _extract_prompt_text(params: dict[str, Any]) -> str:
    """Plain-text rendering of the prompt for tiktoken estimation.

    Multimodal ``content`` arrays keep only the text parts — images would
    need provider-specific vision pricing and aren't relevant here.
    """
    msgs = params.get("messages")
    if not isinstance(msgs, list):
        return ""
    parts: list[str] = []
    for m in msgs:
        role = m.get("role") if isinstance(m, dict) else None
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            if role:
                parts.append(role)
            parts.append(c)
        elif isinstance(c, list):
            for part in c:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ):
                    if role:
                        parts.append(role)
                    parts.append(part["text"])
    return "\n".join(parts)


# ─── stream observation (sync + async) ──────────────────────────────────


def _process_chunk(chunk: Any, state: dict[str, Any]) -> None:
    """Pull usage / completion text out of one streaming chunk into ``state``."""
    usage = _attr(chunk, "usage")
    if usage is not None:
        state["usage"] = usage
    choices = _attr(chunk, "choices") or []
    for ch in choices:
        delta = _attr(ch, "delta")
        content = _attr(delta, "content") if delta is not None else None
        if isinstance(content, str):
            state["completion_text"] += content


def _emit_log(
    frame: Optional[RunFrame],
    logger: Logger,
    request_id: str,
    model: str,
    started_at: float,
    get_process_tags: ProcessTagsGetter,
    state: dict[str, Any],
) -> None:
    err = state.get("error")
    if err is not None:
        logger.log(
            _build_error(frame, request_id, model, started_at, get_process_tags, err)
        )
        return
    usage = state.get("usage")
    if usage is not None:
        logger.log(
            _build_success(
                frame, request_id, model, started_at, get_process_tags, usage
            )
        )
        return
    # No usage emitted — try tiktoken on the captured text.
    prompt_text = state.get("prompt_text", "")
    completion_text = state.get("completion_text", "")
    pt = _tiktoken.estimate_tokens(model, prompt_text)
    ct = _tiktoken.estimate_tokens(model, completion_text)
    if pt is not None and ct is not None:
        logger.log(
            _build_estimated(
                frame, request_id, model, started_at, get_process_tags, pt, ct
            )
        )
    else:
        # tiktoken not installed or load failed — log without tokens
        # (same as pre-tiktoken behaviour).
        logger.log(
            _build_success(
                frame, request_id, model, started_at, get_process_tags, None
            )
        )


def _observe_sync(
    upstream: Iterator[Any],
    *,
    frame: Optional[RunFrame],
    logger: Logger,
    request_id: str,
    model: str,
    started_at: float,
    get_process_tags: ProcessTagsGetter,
    prompt_text: str,
) -> Iterator[Any]:
    state: dict[str, Any] = {
        "usage": None,
        "completion_text": "",
        "prompt_text": prompt_text,
        "error": None,
    }
    try:
        for chunk in upstream:
            _process_chunk(chunk, state)
            yield chunk
    except BaseException as err:  # noqa: BLE001 — propagated below
        state["error"] = err
        raise
    finally:
        _emit_log(
            frame, logger, request_id, model, started_at, get_process_tags, state
        )


async def _observe_async(
    upstream: AsyncIterator[Any],
    *,
    frame: Optional[RunFrame],
    logger: Logger,
    request_id: str,
    model: str,
    started_at: float,
    get_process_tags: ProcessTagsGetter,
    prompt_text: str,
) -> AsyncIterator[Any]:
    state: dict[str, Any] = {
        "usage": None,
        "completion_text": "",
        "prompt_text": prompt_text,
        "error": None,
    }
    try:
        async for chunk in upstream:
            _process_chunk(chunk, state)
            yield chunk
    except BaseException as err:  # noqa: BLE001 — propagated below
        state["error"] = err
        raise
    finally:
        _emit_log(
            frame, logger, request_id, model, started_at, get_process_tags, state
        )


# ─── wrapper objects ────────────────────────────────────────────────────


def _enrich_stream_options(params: dict[str, Any]) -> dict[str, Any]:
    """Auto-inject ``include_usage`` so OpenAI emits the final usage chunk.

    Customer-set ``stream_options`` wins for any other field, but we
    override ``include_usage`` deliberately — without it we'd have no
    usage data and the whole point is attribution.
    """
    if params.get("stream") is not True:
        return params
    existing = params.get("stream_options") or {}
    if not isinstance(existing, dict):
        existing = {}
    new_opts = {**existing, "include_usage": True}
    return {**params, "stream_options": new_opts}


class _CompletionsWrapper:
    """Wraps ``client.chat.completions``.

    ``create`` is the only intercepted attribute; everything else falls
    through via ``__getattr__``. We detect sync vs async by inspecting
    the *result* of the underlying call (``asyncio.iscoroutine``) rather
    than the function definition: openai's Python SDK declares both
    ``OpenAI`` and ``AsyncOpenAI`` ``create`` as plain ``def`` (the async
    one returns a coroutine), so ``iscoroutinefunction`` is unreliable.
    """

    def __init__(
        self,
        completions: Any,
        logger: Logger,
        get_process_tags: ProcessTagsGetter,
        guard: Optional[BudgetGuard] = None,
        on_breach: str = "throw",
        fallbacks: Optional[list[FallbackTarget]] = None,
        agent_binding: Optional[AgentBinding] = None,
    ) -> None:
        self._completions = completions
        self._logger = logger
        self._get_process_tags = get_process_tags
        self._guard = guard
        self._on_breach = on_breach
        # Day 24: cross-provider fallback chain. Walked on retryable
        # upstream errors. Empty list → previous Day-20 behaviour.
        self._fallbacks: list[FallbackTarget] = list(fallbacks or [])
        # Closure-free agent binding from ``.with_agent()``. When set, each
        # call derives a child frame off the ambient frame so attribution
        # follows without a surrounding ``with run():`` block.
        self._agent_binding = agent_binding

    def __getattr__(self, name: str) -> Any:
        # Called only when the attribute isn't on self. Anything other than
        # `create` and the bound state above (e.g. `stream`, `parse`)
        # passes through to the underlying client.
        return getattr(self._completions, name)

    def create(self, *args: Any, **kwargs: Any) -> Any:
        # Day 20 pre-call gate. Raises BudgetExceededError before the
        # upstream provider is touched when an applicable hard-cap budget
        # is breached. Sync call (briefly blocks the event loop on first
        # cache miss per agent for async users); subsequent calls hit the
        # in-memory cache. No log entry is written for refused calls —
        # they'd appear as zero-token ghost rows otherwise.
        request_id = uuid7()
        # Capture the attribution frame ONCE at call entry. For the bound
        # ``.with_agent()`` path this derives a child frame off the ambient
        # frame; otherwise it's the live frame. Threading it explicitly into
        # every builder/observer means the streaming + async paths attribute
        # correctly even though their log entry is built later (after the
        # ambient frame may have been popped).
        binding = self._agent_binding
        frame = (
            derive_frame(get_current_frame(), binding.agent, binding.tags)
            if binding is not None
            else get_current_frame()
        )
        if self._guard is not None:
            agent_name = frame.agent_name if frame is not None else None
            self._guard.pre_call_check(agent_name, self._on_breach)
        if binding is None:
            set_last_request_id(request_id)
        elif frame is not None:
            # Bound path owns a private derived frame — mutate it directly
            # rather than the parent ContextVar frame.
            frame.last_request_id = request_id
        started_at = time.time()
        params = _merge_args(args, kwargs)
        model = params.get("model") or "unknown"
        is_streaming = params.get("stream") is True
        logger = self._logger
        get_tags = self._get_process_tags

        if is_streaming:
            enriched = _enrich_stream_options(params)
            prompt_text = _extract_prompt_text(params)
            try:
                result_or_coro = self._completions.create(**enriched)
            except BaseException as err:
                logger.log(
                    _build_error(frame, request_id, model, started_at, get_tags, err)
                )
                raise
            if asyncio.iscoroutine(result_or_coro):
                # AsyncOpenAI: returned a coroutine that will resolve to
                # an AsyncStream. Wrap so `await create(...)` awaits the
                # coroutine and yields back the observed async iterator.
                return _await_then_observe_async_stream(
                    result_or_coro,
                    frame=frame,
                    request_id=request_id,
                    model=model,
                    started_at=started_at,
                    get_process_tags=get_tags,
                    logger=logger,
                    prompt_text=prompt_text,
                )
            # Sync OpenAI: returned a Stream we can iterate directly.
            return _observe_sync(
                result_or_coro,
                frame=frame,
                logger=logger,
                request_id=request_id,
                model=model,
                started_at=started_at,
                get_process_tags=get_tags,
                prompt_text=prompt_text,
            )

        try:
            result_or_coro = self._completions.create(**params)
        except BaseException as primary_err:
            # Log the primary failure first — cost attribution stays
            # correct even if every fallback also fails.
            logger.log(
                _build_error(frame, request_id, model, started_at, get_tags, primary_err)
            )
            # Day-24: walk the sync fallback chain.
            if not self._fallbacks or not is_retryable_upstream_error(primary_err):
                raise
            last_err: BaseException = primary_err
            for target in self._fallbacks:
                outcome = try_fallback_sync("openai", params, target)
                if outcome["kind"] == "unsupported":
                    raise primary_err
                entry = build_attempt_entry(target, outcome, get_tags)
                logger.log(entry)
                if outcome["kind"] == "success":
                    set_last_request_id(entry["id"])
                    return outcome["response"]
                last_err = outcome["error"]
                if not is_retryable_upstream_error(last_err):
                    raise last_err
            raise last_err

        if asyncio.iscoroutine(result_or_coro):
            return _await_then_log_async(
                result_or_coro,
                frame=frame,
                request_id=request_id,
                model=model,
                started_at=started_at,
                get_process_tags=get_tags,
                logger=logger,
                fallbacks=self._fallbacks,
                original_params=params,
            )
        # Sync non-streaming path.
        usage = _attr(result_or_coro, "usage")
        logger.log(
            _build_success(frame, request_id, model, started_at, get_tags, usage)
        )
        return result_or_coro


async def _await_then_observe_async_stream(
    coro: Any,
    *,
    frame: Optional[RunFrame],
    request_id: str,
    model: str,
    started_at: float,
    get_process_tags: ProcessTagsGetter,
    logger: Logger,
    prompt_text: str,
) -> AsyncIterator[Any]:
    """Customer awaits this; resolves to the observed async iterator.

    Errors raised during the await (HTTP failure on the upstream call)
    flow through ``_build_error`` so attribution is preserved even on
    failed streams.
    """
    try:
        upstream = await coro
    except BaseException as err:
        logger.log(
            _build_error(frame, request_id, model, started_at, get_process_tags, err)
        )
        raise
    return _observe_async(
        upstream,
        frame=frame,
        logger=logger,
        request_id=request_id,
        model=model,
        started_at=started_at,
        get_process_tags=get_process_tags,
        prompt_text=prompt_text,
    )


async def _await_then_log_async(
    coro: Any,
    *,
    frame: Optional[RunFrame],
    request_id: str,
    model: str,
    started_at: float,
    get_process_tags: ProcessTagsGetter,
    logger: Logger,
    fallbacks: list[FallbackTarget] = (),  # type: ignore[assignment]
    original_params: Optional[dict[str, Any]] = None,
) -> Any:
    """Async non-streaming wrapper: await + log + return the response.

    Day-24: when ``fallbacks`` is non-empty and the primary fails with a
    retryable upstream error, walk the chain. Each attempt logs under its
    own request_id with ``tags.via='fallback'``; a successful fallback
    bumps ``set_last_request_id`` so customers reading the most-recent id
    see the attempt that actually served the request.
    """
    try:
        result = await coro
    except BaseException as primary_err:
        logger.log(
            _build_error(frame, request_id, model, started_at, get_process_tags, primary_err)
        )
        if not fallbacks or not is_retryable_upstream_error(primary_err):
            raise

        last_err: BaseException = primary_err
        for target in fallbacks:
            outcome = await try_fallback_async(
                "openai", original_params or {}, target
            )
            if outcome["kind"] == "unsupported":
                # Translator refused (streaming, tools, …). Original error
                # is what the caller should see — bubbling a different
                # one would hide the real problem.
                raise primary_err
            entry = build_attempt_entry(target, outcome, get_process_tags)
            logger.log(entry)
            if outcome["kind"] == "success":
                set_last_request_id(entry["id"])
                return outcome["response"]
            last_err = outcome["error"]
            if not is_retryable_upstream_error(last_err):
                raise last_err
        raise last_err

    usage = _attr(result, "usage")
    logger.log(
        _build_success(frame, request_id, model, started_at, get_process_tags, usage)
    )
    return result


class _ChatWrapper:
    def __init__(
        self,
        chat: Any,
        logger: Logger,
        get_process_tags: ProcessTagsGetter,
        guard: Optional[BudgetGuard] = None,
        on_breach: str = "throw",
        fallbacks: Optional[list[FallbackTarget]] = None,
        agent_binding: Optional[AgentBinding] = None,
    ) -> None:
        self._chat = chat
        self._completions = _CompletionsWrapper(
            chat.completions,
            logger,
            get_process_tags,
            guard,
            on_breach,
            fallbacks,
            agent_binding,
        )

    @property
    def completions(self) -> _CompletionsWrapper:
        return self._completions

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class _OpenAIWrapper:
    def __init__(
        self,
        client: Any,
        logger: Logger,
        get_process_tags: ProcessTagsGetter,
        guard: Optional[BudgetGuard] = None,
        on_breach: str = "throw",
        fallbacks: Optional[list[FallbackTarget]] = None,
        agent_binding: Optional[AgentBinding] = None,
    ) -> None:
        self._client = client
        # Stored so ``.with_agent()`` can re-wrap the same underlying client
        # with an added agent binding (closure-free re-tagging per call).
        self._logger = logger
        self._get_process_tags = get_process_tags
        self._guard = guard
        self._on_breach = on_breach
        self._fallbacks = fallbacks
        self._chat = _ChatWrapper(
            client.chat,
            logger,
            get_process_tags,
            guard,
            on_breach,
            fallbacks,
            agent_binding,
        )

    @property
    def chat(self) -> _ChatWrapper:
        return self._chat

    def with_agent(self, agent: str, tags: Optional[dict] = None) -> Any:
        """Return a fresh wrapped client whose calls attribute to ``agent``.

        Closure-free agent tagging without a surrounding ``with run():``
        block — the frame is derived + captured at each terminal call.
        """
        return wrap_openai(
            self._client,
            self._logger,
            self._get_process_tags,
            self._guard,
            self._on_breach,
            self._fallbacks,
            AgentBinding(agent, normalise_tags(tags)),
        )

    def __getattr__(self, name: str) -> Any:
        # ``audio``, ``embeddings``, ``files``, ``beta``, ``with_options``
        # — every other knob on the openai client passes through unchanged.
        return getattr(self._client, name)


def _merge_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalise the OpenAI ``create()`` call signature to a kwargs dict.

    The Python OpenAI SDK exclusively uses keyword arguments, but customers
    may forward a single positional dict. Tolerate both.
    """
    if not args:
        return dict(kwargs)
    if len(args) == 1 and isinstance(args[0], dict):
        return {**args[0], **kwargs}
    raise TypeError(
        "spaturzu.wrap_openai: unexpected positional args to create(); "
        "use keyword arguments"
    )


def wrap_openai(
    client: Any,
    logger: Logger,
    get_process_tags: ProcessTagsGetter,
    guard: Optional[BudgetGuard] = None,
    on_breach: str = "throw",
    fallbacks: Optional[list[FallbackTarget]] = None,
    agent_binding: Optional[AgentBinding] = None,
) -> Any:
    """Return a wrapped client. The original is untouched — every other
    attribute proxies through ``__getattr__`` so the customer can use the
    returned object exactly like the underlying ``OpenAI`` /
    ``AsyncOpenAI`` instance.

    Day 20: pass a ``BudgetGuard`` (and optional ``on_breach``) to enable
    hard-cap enforcement. The wrapped ``create`` raises
    ``BudgetExceededError`` before the provider is hit when an applicable
    hard-cap budget is breached.

    Day 24: pass ``fallbacks`` (a list of ``FallbackTarget`` dicts) to
    declare a cross-provider failover chain. Each entry is
    ``{"provider": "anthropic"|"openai", "client": ..., "model": "..."}``.
    On a retryable upstream error (429 / 5xx / connection-class), the
    wrap walks the chain and translates request/response when crossing
    providers. v1 scope: non-streaming, no tools/response_format, text
    content only. Mixing sync and async clients across the chain is
    rejected at runtime with a clear error.
    """
    return _OpenAIWrapper(
        client, logger, get_process_tags, guard, on_breach, fallbacks, agent_binding
    )
