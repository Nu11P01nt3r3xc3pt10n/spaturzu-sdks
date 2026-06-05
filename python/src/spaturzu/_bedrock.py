"""AWS Bedrock Converse wrap (sync only, mirrors `_openai.py`).

Customers pass a boto3 ``bedrock-runtime`` client::

    import boto3
    from spaturzu import spaturzu

    spaturzu = spaturzu()
    client = boto3.client("bedrock-runtime")
    wrapped = spaturzu.wrap_bedrock(client)
    resp = wrapped.converse(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
    )

Wraps two methods: ``converse`` (non-stream) and ``converse_stream``
(stream). Everything else on the client passes through unchanged.

v1 is sync only — ``aioboto3`` async support is deferred to a future
iteration (spec §9).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterator, Optional
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
    try_fallback_sync,
)
from ._logger import Logger


ProcessTagsGetter = Callable[[], Optional[dict[str, str]]]


# ─── log-entry builders ─────────────────────────────────────────────────


def _base_entry(
    frame: Optional[RunFrame],
    request_id: str,
    model_id: str,
    get_process_tags: ProcessTagsGetter,
) -> dict[str, Any]:
    tags = merge_tags(get_process_tags(), frame.tags if frame else None)
    entry: dict[str, Any] = {
        "id": request_id,
        "provider": "bedrock",
        "model": model_id,
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
    model_id: str,
    started_at: float,
    get_process_tags: ProcessTagsGetter,
    usage: Optional[dict[str, Any]],
) -> dict[str, Any]:
    entry = _base_entry(frame, request_id, model_id, get_process_tags)
    entry["status"] = 200
    entry["latencyMs"] = int((time.time() - started_at) * 1000)
    if usage:
        if isinstance(usage.get("inputTokens"), int):
            entry["promptTokens"] = usage["inputTokens"]
        if isinstance(usage.get("outputTokens"), int):
            entry["completionTokens"] = usage["outputTokens"]
        if isinstance(usage.get("cacheReadInputTokenCount"), int):
            entry["cachedInputTokens"] = usage["cacheReadInputTokenCount"]
        entry["usageSource"] = "provider"
    return entry


def _build_error(
    frame: Optional[RunFrame],
    request_id: str,
    model_id: str,
    started_at: float,
    get_process_tags: ProcessTagsGetter,
    err: BaseException,
) -> dict[str, Any]:
    entry = _base_entry(frame, request_id, model_id, get_process_tags)
    # botocore ClientError carries status in
    # err.response['ResponseMetadata']['HTTPStatusCode']. Read both that
    # shape and the AWS SDK v3-ish $metadata shape for forward-compat.
    status: Optional[int] = None
    response = getattr(err, "response", None)
    if isinstance(response, dict):
        md = response.get("ResponseMetadata") or {}
        if isinstance(md.get("HTTPStatusCode"), int):
            status = md["HTTPStatusCode"]
    if status is None:
        meta = getattr(err, "$metadata", None) or getattr(err, "metadata", None)
        if isinstance(meta, dict) and isinstance(meta.get("httpStatusCode"), int):
            status = meta["httpStatusCode"]
    if status is None:
        status = 500
    entry["status"] = max(100, min(599, status))
    entry["latencyMs"] = int((time.time() - started_at) * 1000)
    return entry


def _build_estimated(
    frame: Optional[RunFrame],
    request_id: str,
    model_id: str,
    started_at: float,
    get_process_tags: ProcessTagsGetter,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    entry = _base_entry(frame, request_id, model_id, get_process_tags)
    entry["status"] = 200
    entry["latencyMs"] = int((time.time() - started_at) * 1000)
    entry["promptTokens"] = prompt_tokens
    entry["completionTokens"] = completion_tokens
    entry["usageSource"] = "tiktoken"
    return entry


# ─── prompt-text extraction ─────────────────────────────────────────────


def _extract_prompt_text(params: dict[str, Any]) -> str:
    parts: list[str] = []
    for b in params.get("system") or []:
        if isinstance(b, dict) and isinstance(b.get("text"), str):
            parts.append(b["text"])
    for m in params.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or ""
        for b in m.get("content") or []:
            if isinstance(b, dict) and isinstance(b.get("text"), str):
                if role:
                    parts.append(role)
                parts.append(b["text"])
    return "\n".join(parts)


# ─── streaming observer ────────────────────────────────────────────────


def _observe_stream(
    upstream: Iterator[Any],
    *,
    frame: Optional[RunFrame],
    logger: Logger,
    request_id: str,
    model_id: str,
    started_at: float,
    get_process_tags: ProcessTagsGetter,
    prompt_text: str,
) -> Iterator[Any]:
    state: dict[str, Any] = {
        "usage": None,
        "completion_text": "",
        "error": None,
    }
    try:
        for event in upstream:
            if isinstance(event, dict):
                # contentBlockDelta accumulates text for tiktoken fallback.
                cbd = event.get("contentBlockDelta")
                if isinstance(cbd, dict):
                    delta = cbd.get("delta") or {}
                    text = delta.get("text")
                    if isinstance(text, str):
                        state["completion_text"] += text
                # metadata is the terminal event carrying authoritative usage.
                meta = event.get("metadata")
                if isinstance(meta, dict):
                    u = meta.get("usage")
                    if isinstance(u, dict):
                        state["usage"] = u
            yield event
    except BaseException as err:
        state["error"] = err
        raise
    finally:
        err = state.get("error")
        if err is not None:
            logger.log(
                _build_error(frame, request_id, model_id, started_at, get_process_tags, err)
            )
        else:
            usage = state.get("usage")
            if usage:
                logger.log(
                    _build_success(
                        frame, request_id, model_id, started_at, get_process_tags, usage
                    )
                )
            else:
                # No metadata event — try tiktoken on captured text.
                pt = _tiktoken.estimate_tokens(model_id, prompt_text)
                ct = _tiktoken.estimate_tokens(model_id, state["completion_text"])
                if pt is not None and ct is not None:
                    logger.log(
                        _build_estimated(
                            frame, request_id, model_id, started_at, get_process_tags, pt, ct
                        )
                    )
                else:
                    logger.log(
                        _build_success(
                            frame, request_id, model_id, started_at, get_process_tags, None
                        )
                    )


# ─── wrapper ───────────────────────────────────────────────────────────


class _BedrockWrapper:
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
        self._logger = logger
        self._get_process_tags = get_process_tags
        self._guard = guard
        self._on_breach = on_breach
        self._fallbacks: list[FallbackTarget] = list(fallbacks or [])
        self._agent_binding = agent_binding

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def converse(self, **params: Any) -> Any:
        request_id = uuid7()
        # Capture the attribution frame ONCE at call entry. For the bound
        # ``.with_agent()`` path this derives a child frame off the ambient
        # frame; otherwise it's the live frame.
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
        model_id = params.get("modelId") or "unknown"

        try:
            result = self._client.converse(**params)
        except BaseException as primary_err:
            self._logger.log(
                _build_error(
                    frame,
                    request_id,
                    model_id,
                    started_at,
                    self._get_process_tags,
                    primary_err,
                )
            )
            if not self._fallbacks or not is_retryable_upstream_error(primary_err):
                raise
            last_err: BaseException = primary_err
            for target in self._fallbacks:
                outcome = try_fallback_sync("bedrock", params, target)
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

        usage = result.get("usage") if isinstance(result, dict) else None
        self._logger.log(
            _build_success(
                frame, request_id, model_id, started_at, self._get_process_tags, usage
            )
        )
        return result

    def converse_stream(self, **params: Any) -> Any:
        request_id = uuid7()
        # Capture the attribution frame ONCE at call entry (mirrors converse).
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
        model_id = params.get("modelId") or "unknown"
        prompt_text = _extract_prompt_text(params)

        try:
            raw = self._client.converse_stream(**params)
        except BaseException as err:
            self._logger.log(
                _build_error(
                    frame, request_id, model_id, started_at, self._get_process_tags, err
                )
            )
            raise

        # Return a NEW dict mirroring boto3's envelope, with our observer
        # replacing the `stream` member. The caller's `for event in resp['stream']`
        # loop iterates through our observer transparently.
        observed = _observe_stream(
            raw["stream"],
            frame=frame,
            logger=self._logger,
            request_id=request_id,
            model_id=model_id,
            started_at=started_at,
            get_process_tags=self._get_process_tags,
            prompt_text=prompt_text,
        )
        return {**raw, "stream": observed}

    def with_agent(self, agent: str, tags: Optional[dict] = None) -> Any:
        """Return a fresh wrapped client whose calls attribute to ``agent``.

        Closure-free agent tagging without a surrounding ``with run():``
        block — the frame is derived + captured at each terminal call.
        """
        return wrap_bedrock(
            self._client,
            self._logger,
            self._get_process_tags,
            self._guard,
            self._on_breach,
            self._fallbacks,
            AgentBinding(agent, normalise_tags(tags)),
        )


def wrap_bedrock(
    client: Any,
    logger: Logger,
    get_process_tags: ProcessTagsGetter,
    guard: Optional[BudgetGuard] = None,
    on_breach: str = "throw",
    fallbacks: Optional[list[FallbackTarget]] = None,
    agent_binding: Optional[AgentBinding] = None,
) -> Any:
    """Wrap a boto3 ``bedrock-runtime`` client.

    Intercepts ``converse`` and ``converse_stream``; every other method
    passes through. v1 is sync-only; ``aioboto3`` support is deferred.

    Pass ``agent_binding`` (via ``.with_agent()``) for closure-free agent
    tagging without a surrounding ``with run():`` block.
    """
    return _BedrockWrapper(
        client, logger, get_process_tags, guard, on_breach, fallbacks, agent_binding
    )
