"""Mistral (mistralai) wrap (sync + async).

Mistral's Python SDK exposes:
* ``client.chat.complete(...)``       — sync
* ``client.chat.stream(...)``         — sync (returns iterator)
* ``client.chat.complete_async(...)`` — async
* ``client.chat.stream_async(...)``   — async (returns async iterator)
"""

from __future__ import annotations

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


def _base_entry(frame: Optional[RunFrame], rid, model, get_tags) -> dict[str, Any]:
    tags = merge_tags(get_tags(), frame.tags if frame else None)
    entry: dict[str, Any] = {"id": rid, "provider": "mistral", "model": model, "status": 200}
    if frame is not None:
        entry["runId"] = frame.run_id
        if frame.parent_request_id is not None:
            entry["parentRequestId"] = frame.parent_request_id
        entry["agentName"] = frame.agent_name
        entry["agentPath"] = list(frame.agent_path)
    if tags:
        entry["tags"] = tags
    return entry


def _build_success(frame: Optional[RunFrame], rid, model, started_at, get_tags, usage) -> dict[str, Any]:
    entry = _base_entry(frame, rid, model, get_tags)
    entry["status"] = 200
    entry["latencyMs"] = int((time.time() - started_at) * 1000)
    if usage is not None:
        pt = _attr(usage, "prompt_tokens")
        ct = _attr(usage, "completion_tokens")
        if isinstance(pt, int):
            entry["promptTokens"] = pt
        if isinstance(ct, int):
            entry["completionTokens"] = ct
        entry["usageSource"] = "provider"
    return entry


def _build_error(frame: Optional[RunFrame], rid, model, started_at, get_tags, err) -> dict[str, Any]:
    entry = _base_entry(frame, rid, model, get_tags)
    raw = _attr(err, "status_code")
    if not isinstance(raw, int):
        raw = _attr(err, "status")
    status = raw if isinstance(raw, int) else 500
    entry["status"] = max(100, min(599, status))
    entry["latencyMs"] = int((time.time() - started_at) * 1000)
    return entry


def _build_estimated(frame: Optional[RunFrame], rid, model, started_at, get_tags, pt, ct) -> dict[str, Any]:
    entry = _base_entry(frame, rid, model, get_tags)
    entry["status"] = 200
    entry["latencyMs"] = int((time.time() - started_at) * 1000)
    entry["promptTokens"] = pt
    entry["completionTokens"] = ct
    entry["usageSource"] = "tiktoken"
    return entry


def _extract_prompt_text(params: dict[str, Any]) -> str:
    parts: list[str] = []
    for m in params.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or ""
        c = m.get("content")
        if isinstance(c, str):
            if role:
                parts.append(role)
            parts.append(c)
    return "\n".join(parts)


def _observe_chunk(chunk: Any, state: dict[str, Any]) -> None:
    """Pull usage + completion text from one Mistral stream chunk.

    Mistral wraps under chunk.data — check that envelope first, fall
    back to the top-level shape for SDK variants that don't envelope.
    """
    data = _attr(chunk, "data")
    if data is None:
        data = chunk  # some SDK versions return the inner shape directly
    usage = _attr(data, "usage")
    if usage is not None:
        state["usage"] = usage
    choices = _attr(data, "choices") or []
    for ch in choices:
        delta = _attr(ch, "delta")
        content = _attr(delta, "content") if delta is not None else None
        if isinstance(content, str):
            state["completion_text"] += content


def _emit_log(frame: Optional[RunFrame], logger, rid, model, started_at, get_tags, state) -> None:
    err = state.get("error")
    if err is not None:
        logger.log(_build_error(frame, rid, model, started_at, get_tags, err))
    else:
        usage = state.get("usage")
        if usage is not None:
            logger.log(_build_success(frame, rid, model, started_at, get_tags, usage))
        else:
            pt = _tiktoken.estimate_tokens(model, state.get("prompt_text", ""))
            ct = _tiktoken.estimate_tokens(model, state.get("completion_text", ""))
            if pt is not None and ct is not None:
                logger.log(_build_estimated(frame, rid, model, started_at, get_tags, pt, ct))
            else:
                logger.log(_build_success(frame, rid, model, started_at, get_tags, None))


def _observe_sync(
    upstream: Iterator[Any],
    *,
    frame: Optional[RunFrame],
    logger: Logger,
    rid: str,
    model: str,
    started_at: float,
    get_tags: ProcessTagsGetter,
    prompt_text: str,
) -> Iterator[Any]:
    state: dict[str, Any] = {"usage": None, "completion_text": "", "prompt_text": prompt_text, "error": None}
    try:
        for chunk in upstream:
            _observe_chunk(chunk, state)
            yield chunk
    except BaseException as err:
        state["error"] = err
        raise
    finally:
        # NB: NO `return` here — would suppress propagating exception.
        _emit_log(frame, logger, rid, model, started_at, get_tags, state)


async def _observe_async(
    upstream: AsyncIterator[Any],
    *,
    frame: Optional[RunFrame],
    logger: Logger,
    rid: str,
    model: str,
    started_at: float,
    get_tags: ProcessTagsGetter,
    prompt_text: str,
) -> AsyncIterator[Any]:
    state: dict[str, Any] = {"usage": None, "completion_text": "", "prompt_text": prompt_text, "error": None}
    try:
        async for chunk in upstream:
            _observe_chunk(chunk, state)
            yield chunk
    except BaseException as err:
        state["error"] = err
        raise
    finally:
        _emit_log(frame, logger, rid, model, started_at, get_tags, state)


class _ChatWrapper:
    """Wraps `client.chat` to intercept complete / stream / *_async methods."""

    def __init__(self, chat, logger, get_tags, guard, on_breach, fallbacks, agent_binding=None):
        self._chat = chat
        self._logger = logger
        self._get_tags = get_tags
        self._guard = guard
        self._on_breach = on_breach
        self._fallbacks = fallbacks
        self._agent_binding: Optional[AgentBinding] = agent_binding

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)

    def complete(self, **params: Any) -> Any:
        rid = uuid7()
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
            set_last_request_id(rid)
        elif frame is not None:
            frame.last_request_id = rid
        started_at = time.time()
        model = params.get("model") or "unknown"
        try:
            result = self._chat.complete(**params)
        except BaseException as primary_err:
            self._logger.log(_build_error(frame, rid, model, started_at, self._get_tags, primary_err))
            if not self._fallbacks or not is_retryable_upstream_error(primary_err):
                raise
            last_err: BaseException = primary_err
            for target in self._fallbacks:
                outcome = try_fallback_sync("mistral", params, target)
                if outcome["kind"] == "unsupported":
                    raise primary_err
                entry = build_attempt_entry(target, outcome, self._get_tags)
                self._logger.log(entry)
                if outcome["kind"] == "success":
                    set_last_request_id(entry["id"])
                    return outcome["response"]
                last_err = outcome["error"]
                if not is_retryable_upstream_error(last_err):
                    raise last_err
            raise last_err
        self._logger.log(_build_success(frame, rid, model, started_at, self._get_tags, _attr(result, "usage")))
        return result

    def stream(self, **params: Any) -> Iterator[Any]:
        rid = uuid7()
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
            set_last_request_id(rid)
        elif frame is not None:
            frame.last_request_id = rid
        started_at = time.time()
        model = params.get("model") or "unknown"
        prompt_text = _extract_prompt_text(params)
        try:
            upstream = self._chat.stream(**params)
        except BaseException as err:
            self._logger.log(_build_error(frame, rid, model, started_at, self._get_tags, err))
            raise
        return _observe_sync(
            upstream,
            frame=frame,
            logger=self._logger,
            rid=rid,
            model=model,
            started_at=started_at,
            get_tags=self._get_tags,
            prompt_text=prompt_text,
        )

    async def complete_async(self, **params: Any) -> Any:
        rid = uuid7()
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
            set_last_request_id(rid)
        elif frame is not None:
            frame.last_request_id = rid
        started_at = time.time()
        model = params.get("model") or "unknown"
        try:
            result = await self._chat.complete_async(**params)
        except BaseException as primary_err:
            self._logger.log(_build_error(frame, rid, model, started_at, self._get_tags, primary_err))
            if not self._fallbacks or not is_retryable_upstream_error(primary_err):
                raise
            last_err: BaseException = primary_err
            for target in self._fallbacks:
                outcome = await try_fallback_async("mistral", params, target)
                if outcome["kind"] == "unsupported":
                    raise primary_err
                entry = build_attempt_entry(target, outcome, self._get_tags)
                self._logger.log(entry)
                if outcome["kind"] == "success":
                    set_last_request_id(entry["id"])
                    return outcome["response"]
                last_err = outcome["error"]
                if not is_retryable_upstream_error(last_err):
                    raise last_err
            raise last_err
        self._logger.log(_build_success(frame, rid, model, started_at, self._get_tags, _attr(result, "usage")))
        return result

    async def stream_async(self, **params: Any) -> AsyncIterator[Any]:
        rid = uuid7()
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
            set_last_request_id(rid)
        elif frame is not None:
            frame.last_request_id = rid
        started_at = time.time()
        model = params.get("model") or "unknown"
        prompt_text = _extract_prompt_text(params)
        try:
            upstream = await self._chat.stream_async(**params)
        except BaseException as err:
            self._logger.log(_build_error(frame, rid, model, started_at, self._get_tags, err))
            raise
        return _observe_async(
            upstream,
            frame=frame,
            logger=self._logger,
            rid=rid,
            model=model,
            started_at=started_at,
            get_tags=self._get_tags,
            prompt_text=prompt_text,
        )


class _MistralWrapper:
    def __init__(self, client, logger, get_tags, guard=None, on_breach="throw", fallbacks=None, agent_binding=None):
        self._client = client
        self._logger = logger
        self._get_process_tags = get_tags
        self._guard = guard
        self._on_breach = on_breach
        self._fallbacks = fallbacks
        self._chat = _ChatWrapper(
            client.chat, logger, get_tags, guard, on_breach, list(fallbacks or []), agent_binding
        )

    @property
    def chat(self) -> _ChatWrapper:
        return self._chat

    def with_agent(self, agent: str, tags: Optional[dict] = None) -> Any:
        return wrap_mistral(
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


def wrap_mistral(client, logger, get_process_tags, guard=None, on_breach="throw", fallbacks=None, agent_binding=None):
    """Wrap a mistralai Mistral client. The original is untouched.

    Intercepts ``client.chat.{complete, stream, complete_async, stream_async}``.

    Note: On the happy path the wrap returns whatever mistralai returns
    (typed response objects with attribute access). When a fallback
    target serves the call, the response is a **plain dict** in Mistral
    shape — use subscript access in code paths that may run after a
    fallback.
    """
    return _MistralWrapper(client, logger, get_process_tags, guard, on_breach, fallbacks, agent_binding)
