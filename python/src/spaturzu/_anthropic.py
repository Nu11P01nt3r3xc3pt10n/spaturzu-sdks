"""Anthropic client wrapper.

Mirrors `sdks/typescript/src/anthropic.ts`. Wraps both ``Anthropic`` (sync) and
``AsyncAnthropic`` (async).

Stream event shapes:

* ``message_start`` — ``event.message.usage`` carries ``input_tokens`` and
  ``cache_*_input_tokens`` plus an ``output_tokens`` of 0.
* ``content_block_delta`` — ``event.delta.type == 'text_delta'`` then
  ``event.delta.text`` accumulates the assistant's reply text (used for
  the tiktoken fallback when usage events don't arrive).
* ``message_delta`` — ``event.usage.output_tokens`` is *cumulative* for
  the message; we overwrite the running total each time.

Anthropic's mainstream SDK reliably emits usage. Bedrock-proxied calls
sometimes don't — that's the case the tiktoken fallback exists for.
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


def _attr(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    val = getattr(obj, name, None)
    if val is not None:
        return val
    if isinstance(obj, dict):
        return obj.get(name)
    return None


# ─── log-entry builders ─────────────────────────────────────────────────


def _base_entry(
    frame: Optional[RunFrame],
    request_id: str,
    model: str,
    started_at: float,
    get_process_tags: ProcessTagsGetter,
) -> dict[str, Any]:
    tags = merge_tags(get_process_tags(), frame.tags if frame else None)
    entry: dict[str, Any] = {
        "id": request_id,
        "provider": "anthropic",
        "model": model,
        "status": 200,
    }
    # `started_at` parameter stays — local stopwatch for latencyMs. Wire
    # startedAt was dropped 2026-05-16 with the advisory-lock dedup change.
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
    *,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    cache_read: Optional[int],
) -> dict[str, Any]:
    entry = _base_entry(frame, request_id, model, started_at, get_process_tags)
    entry["status"] = 200
    entry["latencyMs"] = int((time.time() - started_at) * 1000)
    if input_tokens is not None:
        entry["promptTokens"] = input_tokens
    if output_tokens is not None:
        entry["completionTokens"] = output_tokens
    # We map ``cache_read_input_tokens`` (the portion served from the prompt
    # cache) to ``cachedInputTokens``. ``cache_creation_input_tokens`` is
    # the *write* side and not a savings vs the uncached rate; leave it for
    # a future "cache write cost" surface.
    if cache_read is not None:
        entry["cachedInputTokens"] = cache_read
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


# ─── prompt-text extraction ─────────────────────────────────────────────


def _extract_prompt_text(params: dict[str, Any]) -> str:
    """Plain-text rendering of system + messages for tiktoken estimation.

    Anthropic's ``messages.create`` accepts either string content or
    content-block arrays; we keep only the text blocks. ``system`` can be
    either a string or a list of text blocks (newer Anthropic SDKs).
    """
    parts: list[str] = []
    sys_field = params.get("system")
    if isinstance(sys_field, str):
        parts.append(sys_field)
    elif isinstance(sys_field, list):
        for b in sys_field:
            if (
                isinstance(b, dict)
                and b.get("type") == "text"
                and isinstance(b.get("text"), str)
            ):
                parts.append(b["text"])
    msgs = params.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            c = m.get("content")
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


# ─── stream observation ─────────────────────────────────────────────────


def _process_event(event: Any, state: dict[str, Any]) -> None:
    """Pull token totals + completion text out of one Anthropic stream event."""
    ev_type = _attr(event, "type")
    if ev_type == "message_start":
        msg = _attr(event, "message")
        usage = _attr(msg, "usage") if msg is not None else None
        if usage is not None:
            it = _attr(usage, "input_tokens")
            ot = _attr(usage, "output_tokens")
            cr = _attr(usage, "cache_read_input_tokens")
            if isinstance(it, int):
                state["input_tokens"] = it
            if isinstance(ot, int):
                state["output_tokens"] = ot
            if isinstance(cr, int):
                state["cache_read"] = cr
    elif ev_type == "message_delta":
        usage = _attr(event, "usage")
        if usage is not None:
            ot = _attr(usage, "output_tokens")
            if isinstance(ot, int):
                # ``output_tokens`` in message_delta is cumulative — overwrite.
                state["output_tokens"] = ot
    elif ev_type == "content_block_delta":
        delta = _attr(event, "delta")
        if delta is not None and _attr(delta, "type") == "text_delta":
            text = _attr(delta, "text")
            if isinstance(text, str):
                state["completion_text"] += text


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
    it = state.get("input_tokens")
    ot = state.get("output_tokens")
    if it is not None and ot is not None:
        logger.log(
            _build_success(
                frame,
                request_id,
                model,
                started_at,
                get_process_tags,
                input_tokens=it,
                output_tokens=ot,
                cache_read=state.get("cache_read"),
            )
        )
        return
    # No usage events arrived — try tiktoken on the captured text.
    pt = _tiktoken.estimate_tokens(model, state.get("prompt_text", ""))
    ct = _tiktoken.estimate_tokens(model, state.get("completion_text", ""))
    if pt is not None and ct is not None:
        logger.log(
            _build_estimated(
                frame, request_id, model, started_at, get_process_tags, pt, ct
            )
        )
    else:
        # Fall back to whatever partial usage we collected (e.g. input only).
        logger.log(
            _build_success(
                frame,
                request_id,
                model,
                started_at,
                get_process_tags,
                input_tokens=it,
                output_tokens=ot,
                cache_read=state.get("cache_read"),
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
        "input_tokens": None,
        "output_tokens": None,
        "cache_read": None,
        "completion_text": "",
        "prompt_text": prompt_text,
        "error": None,
    }
    try:
        for event in upstream:
            _process_event(event, state)
            yield event
    except BaseException as err:
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
        "input_tokens": None,
        "output_tokens": None,
        "cache_read": None,
        "completion_text": "",
        "prompt_text": prompt_text,
        "error": None,
    }
    try:
        async for event in upstream:
            _process_event(event, state)
            yield event
    except BaseException as err:
        state["error"] = err
        raise
    finally:
        _emit_log(
            frame, logger, request_id, model, started_at, get_process_tags, state
        )


# ─── wrapper objects ────────────────────────────────────────────────────


def _merge_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if not args:
        return dict(kwargs)
    if len(args) == 1 and isinstance(args[0], dict):
        return {**args[0], **kwargs}
    raise TypeError(
        "spaturzu.wrap_anthropic: unexpected positional args to create(); "
        "use keyword arguments"
    )


class _MessagesWrapper:
    """Wraps ``client.messages``.

    Sync vs async is detected by inspecting the *result* of the underlying
    call — anthropic's SDK uses ``def`` for both ``Anthropic`` and
    ``AsyncAnthropic`` ``messages.create``, so ``iscoroutinefunction``
    can't distinguish them at construction time.
    """

    def __init__(
        self,
        messages: Any,
        logger: Logger,
        get_process_tags: ProcessTagsGetter,
        guard: Optional[BudgetGuard] = None,
        on_breach: str = "throw",
        fallbacks: Optional[list[FallbackTarget]] = None,
        agent_binding: Optional[AgentBinding] = None,
    ) -> None:
        self._messages = messages
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
        # ``stream`` (Anthropic SDK helper), ``count_tokens``, etc. pass
        # through unchanged.
        return getattr(self._messages, name)

    def create(self, *args: Any, **kwargs: Any) -> Any:
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
        # Day 20 pre-call gate — see _openai.py for the rationale; mirror
        # here so both wrappers share enforcement semantics.
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
            prompt_text = _extract_prompt_text(params)
            try:
                result_or_coro = self._messages.create(**params)
            except BaseException as err:
                logger.log(
                    _build_error(frame, request_id, model, started_at, get_tags, err)
                )
                raise
            if asyncio.iscoroutine(result_or_coro):
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
            result_or_coro = self._messages.create(**params)
        except BaseException as primary_err:
            logger.log(
                _build_error(frame, request_id, model, started_at, get_tags, primary_err)
            )
            # Day-24: sync fallback chain. Same semantics as openai.create.
            if not self._fallbacks or not is_retryable_upstream_error(primary_err):
                raise
            last_err: BaseException = primary_err
            for target in self._fallbacks:
                outcome = try_fallback_sync("anthropic", params, target)
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
        usage = _attr(result_or_coro, "usage")
        logger.log(
            _build_success(
                frame,
                request_id,
                model,
                started_at,
                get_tags,
                input_tokens=_attr(usage, "input_tokens"),
                output_tokens=_attr(usage, "output_tokens"),
                cache_read=_attr(usage, "cache_read_input_tokens"),
            )
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
    """Async non-streaming Anthropic wrapper. See ``_openai.py`` for the
    Day-24 fallback semantics — this is the symmetric mirror."""
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
                "anthropic", original_params or {}, target
            )
            if outcome["kind"] == "unsupported":
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
        _build_success(
            frame,
            request_id,
            model,
            started_at,
            get_process_tags,
            input_tokens=_attr(usage, "input_tokens"),
            output_tokens=_attr(usage, "output_tokens"),
            cache_read=_attr(usage, "cache_read_input_tokens"),
        )
    )
    return result


class _AnthropicWrapper:
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
        self._messages = _MessagesWrapper(
            client.messages,
            logger,
            get_process_tags,
            guard,
            on_breach,
            fallbacks,
            agent_binding,
        )

    @property
    def messages(self) -> _MessagesWrapper:
        return self._messages

    def with_agent(self, agent: str, tags: Optional[dict] = None) -> Any:
        """Return a fresh wrapped client whose calls attribute to ``agent``.

        Closure-free agent tagging without a surrounding ``with run():``
        block — the frame is derived + captured at each terminal call.
        """
        return wrap_anthropic(
            self._client,
            self._logger,
            self._get_process_tags,
            self._guard,
            self._on_breach,
            self._fallbacks,
            AgentBinding(agent, normalise_tags(tags)),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def wrap_anthropic(
    client: Any,
    logger: Logger,
    get_process_tags: ProcessTagsGetter,
    guard: Optional[BudgetGuard] = None,
    on_breach: str = "throw",
    fallbacks: Optional[list[FallbackTarget]] = None,
    agent_binding: Optional[AgentBinding] = None,
) -> Any:
    """Return a wrapped Anthropic client. The original is untouched.

    See ``_openai.wrap_openai`` for the ``fallbacks`` and ``agent_binding``
    semantics — identical here, just mirrored for the Messages-shaped surface.
    """
    return _AnthropicWrapper(
        client, logger, get_process_tags, guard, on_breach, fallbacks, agent_binding
    )
