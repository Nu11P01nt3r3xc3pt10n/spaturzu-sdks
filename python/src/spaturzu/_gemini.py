"""Gemini (google-genai) wrap (sync + async).

The google-genai Python client exposes a sync surface at
``client.models.generate_content`` / ``client.models.generate_content_stream``
and an async surface at ``client.aio.models.*``. The wrap installs
observers on BOTH.
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
        "provider": "gemini",
        "model": model,
        "status": 200,
    }
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
    usage_metadata: Any,
) -> dict[str, Any]:
    entry = _base_entry(frame, request_id, model, started_at, get_process_tags)
    entry["status"] = 200
    entry["latencyMs"] = int((time.time() - started_at) * 1000)
    if usage_metadata is not None:
        pt = _attr(usage_metadata, "prompt_token_count") or _attr(usage_metadata, "promptTokenCount")
        ct = _attr(usage_metadata, "candidates_token_count") or _attr(usage_metadata, "candidatesTokenCount")
        cached = _attr(usage_metadata, "cached_content_token_count") or _attr(usage_metadata, "cachedContentTokenCount")
        if isinstance(pt, int):
            entry["promptTokens"] = pt
        if isinstance(ct, int):
            entry["completionTokens"] = ct
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


def _extract_prompt_text(params: dict[str, Any]) -> str:
    parts: list[str] = []
    sys = (params.get("config") or {}).get("systemInstruction")
    if isinstance(sys, dict):
        for p in sys.get("parts") or []:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
    elif isinstance(sys, list):
        for p in sys:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
    for c in params.get("contents") or []:
        if not isinstance(c, dict):
            continue
        role = c.get("role") or ""
        for p in c.get("parts") or []:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                if role:
                    parts.append(role)
                parts.append(p["text"])
    return "\n".join(parts)


def _observe_chunk(chunk: Any, state: dict[str, Any]) -> None:
    """Pull last-seen usageMetadata + accumulated text from one stream chunk.

    Per spec §2.2: each chunk carries CUMULATIVE usage; the FINAL chunk's
    value is the full count. We overwrite on every chunk so the last
    wins."""
    um = _attr(chunk, "usage_metadata")
    if um is not None:
        state["usage_metadata"] = um
    candidates = _attr(chunk, "candidates") or []
    if candidates:
        cand = candidates[0]
        content = _attr(cand, "content")
        for p in (_attr(content, "parts") or []):
            t = _attr(p, "text")
            if isinstance(t, str):
                state["completion_text"] += t


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
        logger.log(_build_error(frame, request_id, model, started_at, get_process_tags, err))
        return
    um = state.get("usage_metadata")
    if um is not None:
        logger.log(
            _build_success(frame, request_id, model, started_at, get_process_tags, um)
        )
        return
    pt = _tiktoken.estimate_tokens(model, state.get("prompt_text", ""))
    ct = _tiktoken.estimate_tokens(model, state.get("completion_text", ""))
    if pt is not None and ct is not None:
        logger.log(_build_estimated(frame, request_id, model, started_at, get_process_tags, pt, ct))
    else:
        logger.log(_build_success(frame, request_id, model, started_at, get_process_tags, None))


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
        "usage_metadata": None,
        "completion_text": "",
        "prompt_text": prompt_text,
        "error": None,
    }
    try:
        for chunk in upstream:
            _observe_chunk(chunk, state)
            yield chunk
    except BaseException as err:
        state["error"] = err
        raise
    finally:
        # NB: Plan 3 critical fix — NO `return` inside finally; the raise
        # in `except` must propagate. Use if/elif/else for clarity.
        _emit_log(frame, logger, request_id, model, started_at, get_process_tags, state)


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
        "usage_metadata": None,
        "completion_text": "",
        "prompt_text": prompt_text,
        "error": None,
    }
    try:
        async for chunk in upstream:
            _observe_chunk(chunk, state)
            yield chunk
    except BaseException as err:
        state["error"] = err
        raise
    finally:
        _emit_log(frame, logger, request_id, model, started_at, get_process_tags, state)


# ─── sync wrapper ─────────────────────────────────────────────────────


class _SyncModelsWrapper:
    def __init__(
        self,
        models: Any,
        logger: Logger,
        get_process_tags: ProcessTagsGetter,
        guard: Optional[BudgetGuard],
        on_breach: str,
        fallbacks: list[FallbackTarget],
        agent_binding: Optional[AgentBinding] = None,
    ) -> None:
        self._models = models
        self._logger = logger
        self._get_process_tags = get_process_tags
        self._guard = guard
        self._on_breach = on_breach
        self._fallbacks = fallbacks
        self._agent_binding = agent_binding

    def __getattr__(self, name: str) -> Any:
        return getattr(self._models, name)

    def generate_content(self, **params: Any) -> Any:
        request_id = uuid7()
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
            frame.last_request_id = request_id
        started_at = time.time()
        model = params.get("model") or "unknown"
        try:
            result = self._models.generate_content(**params)
        except BaseException as primary_err:
            self._logger.log(
                _build_error(frame, request_id, model, started_at, self._get_process_tags, primary_err)
            )
            if not self._fallbacks or not is_retryable_upstream_error(primary_err):
                raise
            last_err: BaseException = primary_err
            for target in self._fallbacks:
                outcome = try_fallback_sync("gemini", params, target)
                if outcome["kind"] == "unsupported":
                    raise primary_err
                entry = build_attempt_entry(target, outcome, self._get_process_tags)
                self._logger.log(entry)
                if outcome["kind"] == "success":
                    set_last_request_id(entry["id"])
                    return outcome["response"]
                last_err = outcome["error"]
                if not is_retryable_upstream_error(last_err):
                    raise last_err
            raise last_err
        self._logger.log(
            _build_success(
                frame, request_id, model, started_at, self._get_process_tags,
                _attr(result, "usage_metadata"),
            )
        )
        return result

    def generate_content_stream(self, **params: Any) -> Iterator[Any]:
        request_id = uuid7()
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
            frame.last_request_id = request_id
        started_at = time.time()
        model = params.get("model") or "unknown"
        prompt_text = _extract_prompt_text(params)
        try:
            upstream = self._models.generate_content_stream(**params)
        except BaseException as err:
            self._logger.log(
                _build_error(frame, request_id, model, started_at, self._get_process_tags, err)
            )
            raise
        return _observe_sync(
            upstream,
            frame=frame,
            logger=self._logger,
            request_id=request_id,
            model=model,
            started_at=started_at,
            get_process_tags=self._get_process_tags,
            prompt_text=prompt_text,
        )


# ─── async wrapper ────────────────────────────────────────────────────


class _AsyncModelsWrapper:
    def __init__(
        self,
        models: Any,
        logger: Logger,
        get_process_tags: ProcessTagsGetter,
        guard: Optional[BudgetGuard],
        on_breach: str,
        fallbacks: list[FallbackTarget],
        agent_binding: Optional[AgentBinding] = None,
    ) -> None:
        self._models = models
        self._logger = logger
        self._get_process_tags = get_process_tags
        self._guard = guard
        self._on_breach = on_breach
        self._fallbacks = fallbacks
        self._agent_binding = agent_binding

    def __getattr__(self, name: str) -> Any:
        return getattr(self._models, name)

    async def generate_content(self, **params: Any) -> Any:
        request_id = uuid7()
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
            frame.last_request_id = request_id
        started_at = time.time()
        model = params.get("model") or "unknown"
        try:
            result = await self._models.generate_content(**params)
        except BaseException as primary_err:
            self._logger.log(
                _build_error(frame, request_id, model, started_at, self._get_process_tags, primary_err)
            )
            if not self._fallbacks or not is_retryable_upstream_error(primary_err):
                raise
            last_err: BaseException = primary_err
            for target in self._fallbacks:
                outcome = await try_fallback_async("gemini", params, target)
                if outcome["kind"] == "unsupported":
                    raise primary_err
                entry = build_attempt_entry(target, outcome, self._get_process_tags)
                self._logger.log(entry)
                if outcome["kind"] == "success":
                    set_last_request_id(entry["id"])
                    return outcome["response"]
                last_err = outcome["error"]
                if not is_retryable_upstream_error(last_err):
                    raise last_err
            raise last_err
        self._logger.log(
            _build_success(
                frame, request_id, model, started_at, self._get_process_tags,
                _attr(result, "usage_metadata"),
            )
        )
        return result

    async def generate_content_stream(self, **params: Any) -> AsyncIterator[Any]:
        request_id = uuid7()
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
            frame.last_request_id = request_id
        started_at = time.time()
        model = params.get("model") or "unknown"
        prompt_text = _extract_prompt_text(params)
        try:
            upstream = await self._models.generate_content_stream(**params)
        except BaseException as err:
            self._logger.log(
                _build_error(frame, request_id, model, started_at, self._get_process_tags, err)
            )
            raise
        return _observe_async(
            upstream,
            frame=frame,
            logger=self._logger,
            request_id=request_id,
            model=model,
            started_at=started_at,
            get_process_tags=self._get_process_tags,
            prompt_text=prompt_text,
        )


class _AioWrapper:
    """Wraps `client.aio` to intercept `client.aio.models.*`."""

    def __init__(self, aio: Any, models_wrapper: _AsyncModelsWrapper) -> None:
        self._aio = aio
        self._models = models_wrapper

    @property
    def models(self) -> _AsyncModelsWrapper:
        return self._models

    def __getattr__(self, name: str) -> Any:
        return getattr(self._aio, name)


class _GeminiWrapper:
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
        fb_list: list[FallbackTarget] = list(fallbacks or [])
        self._models = _SyncModelsWrapper(
            client.models, logger, get_process_tags, guard, on_breach, fb_list,
            agent_binding,
        )
        # client.aio may or may not exist depending on google-genai version.
        # If it doesn't exist, the async wrapper is None; consumers using
        # only the sync surface won't notice.
        aio = getattr(client, "aio", None)
        if aio is not None and getattr(aio, "models", None) is not None:
            async_models = _AsyncModelsWrapper(
                aio.models, logger, get_process_tags, guard, on_breach, fb_list,
                agent_binding,
            )
            self._aio: Optional[_AioWrapper] = _AioWrapper(aio, async_models)
        else:
            self._aio = None

    @property
    def models(self) -> _SyncModelsWrapper:
        return self._models

    @property
    def aio(self) -> Any:
        if self._aio is None:
            raise AttributeError(
                "google-genai client has no .aio surface (expected on recent versions)"
            )
        return self._aio

    def with_agent(self, agent: str, tags: Optional[dict] = None) -> Any:
        """Return a fresh wrapped client whose calls attribute to ``agent``.

        Closure-free agent tagging without a surrounding ``with run():``
        block — the frame is derived + captured at each terminal call (on
        BOTH the sync ``.models.*`` and async ``.aio.models.*`` surfaces).
        """
        return wrap_gemini(
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


def wrap_gemini(
    client: Any,
    logger: Logger,
    get_process_tags: ProcessTagsGetter,
    guard: Optional[BudgetGuard] = None,
    on_breach: str = "throw",
    fallbacks: Optional[list[FallbackTarget]] = None,
    agent_binding: Optional[AgentBinding] = None,
) -> Any:
    return _GeminiWrapper(
        client, logger, get_process_tags, guard, on_breach, fallbacks, agent_binding
    )
