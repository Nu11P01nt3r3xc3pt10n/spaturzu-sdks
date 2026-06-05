"""Day-24 SDK-side cross-provider fallback (Python port).

Mirrors :mod:`spaturzu.fallback` in TS. Two entry points — one sync and one
async — because the Python OpenAI / Anthropic clients come in both shapes
and the dispatcher needs to match the primary's shape (you can't ``await``
from sync code, and async callers can't block on a sync ``create``).

The translator and retry classifier are shared; only the invocation step
differs between sync and async.

Why split sync/async here instead of detecting at call time: the primary's
shape is known to the caller (it routes either through ``_observe_sync`` or
``_await_then_log_async`` in the existing wraps). Passing the matching
dispatcher in keeps the per-attempt code path linear instead of branching
on every result.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Literal, Optional, TypedDict
from ._uuid import uuid7

from ._context import get_current_frame, merge_tags
from ._translate import (
    anthropic_params_to_chat,
    anthropic_response_to_chat,
    anthropic_response_from_bedrock,
    anthropic_response_from_gemini,
    anthropic_response_from_mistral,
    anthropic_to_bedrock_params,
    anthropic_to_gemini_params,
    anthropic_to_mistral_params,
    bedrock_response_from_gemini,
    bedrock_response_from_mistral,
    bedrock_response_to_anthropic,
    bedrock_response_to_chat,
    bedrock_to_anthropic_params,
    bedrock_to_chat_params,
    bedrock_to_gemini_params,
    bedrock_to_mistral_params,
    chat_response_from_bedrock,
    chat_response_from_gemini,
    chat_response_from_mistral,
    chat_response_to_anthropic,
    chat_to_anthropic_params,
    chat_to_bedrock_params,
    chat_to_gemini_params,
    chat_to_mistral_params,
    gemini_response_from_mistral,
    gemini_response_to_anthropic,
    gemini_response_to_bedrock,
    gemini_response_to_chat,
    gemini_to_anthropic_params,
    gemini_to_bedrock_params,
    gemini_to_chat_params,
    gemini_to_mistral_params,
    mistral_response_to_anthropic,
    mistral_response_to_bedrock,
    mistral_response_to_chat,
    mistral_response_to_gemini,
    mistral_to_anthropic_params,
    mistral_to_bedrock_params,
    mistral_to_chat_params,
    mistral_to_gemini_params,
)


ProcessTagsGetter = Callable[[], Optional[dict[str, str]]]


ClientShape = Literal["openai", "anthropic", "bedrock", "gemini", "mistral"]


class FallbackTarget(TypedDict):
    """Per-target entry in a fallback chain.

    ``provider`` chooses the request/response translation pair.
    ``client`` is the customer's SDK instance — must implement the
    create-equivalent matching ``provider``:

    * 'openai'    → ``chat.completions.create``
    * 'anthropic' → ``messages.create``
    * 'bedrock'   → ``converse`` / ``converse_stream``
    * 'gemini'    → ``models.generate_content`` / ``models.generate_content_stream``
    * 'mistral'   → ``chat.complete`` / ``chat.stream``

    Duck-typed — peer-dep versions stay flexible.
    """

    provider: ClientShape
    client: Any
    model: str


def is_retryable_upstream_error(err: BaseException) -> bool:
    """Conservative classifier. We retry on rate-limit, server-side, and
    connection-class errors only. Auth / bad-request / context-overflow
    are caller bugs that would just fail again on the next provider —
    surface them eagerly."""
    # Python SDKs expose status in different attributes across versions.
    status = getattr(err, "status_code", None)
    if status is None:
        status = getattr(err, "status", None)
    if isinstance(status, int):
        if status == 429 or status == 408:
            return True
        if 500 <= status < 600:
            return True

    # OpenAI / Anthropic Python SDK error classes — class name is the
    # only stable handle without importing the peer deps.
    cls_name = err.__class__.__name__
    if cls_name in (
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "APIConnectionTimeoutError",
        "InternalServerError",
        # AWS SDK (botocore) exception names. Bedrock surfaces these
        # directly when the underlying ThrottlingException is raised.
        "ThrottlingException",
        "ServiceUnavailableException",
        "ModelTimeoutException",
        "ModelStreamErrorException",
        # google-genai's ApiError carries a .status; the status path
        # above already matches, but adding the class name for
        # defense-in-depth.
        "ApiError",
    ):
        return True

    # OSError subclasses cover network-level failures (ConnectionResetError,
    # TimeoutError, etc.). httpx + requests typically raise from inside
    # these.
    if isinstance(err, (ConnectionError, TimeoutError)):
        return True

    return False


def _status_of(err: BaseException) -> Optional[int]:
    s = getattr(err, "status_code", None)
    if s is None:
        s = getattr(err, "status", None)
    return s if isinstance(s, int) else None


def _normalise_usage_openai(resp: Any) -> dict[str, Optional[int]]:
    usage = _attr(resp, "usage")
    prompt = _attr(usage, "prompt_tokens")
    completion = _attr(usage, "completion_tokens")
    cached_details = _attr(usage, "prompt_tokens_details")
    cached = _attr(cached_details, "cached_tokens") if cached_details else None
    return {
        "prompt_tokens": prompt if isinstance(prompt, int) else None,
        "completion_tokens": completion if isinstance(completion, int) else None,
        "cached_input_tokens": cached if isinstance(cached, int) else None,
    }


def _normalise_usage_anthropic(resp: Any) -> dict[str, Optional[int]]:
    usage = _attr(resp, "usage")
    inp = _attr(usage, "input_tokens")
    out = _attr(usage, "output_tokens")
    cached = _attr(usage, "cache_read_input_tokens")
    return {
        "prompt_tokens": inp if isinstance(inp, int) else None,
        "completion_tokens": out if isinstance(out, int) else None,
        "cached_input_tokens": cached if isinstance(cached, int) else None,
    }


def _normalise_usage_bedrock(resp: Any) -> dict[str, Optional[int]]:
    usage = resp.get("usage") if isinstance(resp, dict) else None
    if not usage:
        return {"prompt_tokens": None, "completion_tokens": None, "cached_input_tokens": None}
    return {
        "prompt_tokens": usage.get("inputTokens"),
        "completion_tokens": usage.get("outputTokens"),
        "cached_input_tokens": usage.get("cacheReadInputTokenCount"),
    }


def _normalise_usage_gemini(resp: Any) -> dict[str, Optional[int]]:
    um = resp.get("usageMetadata") if isinstance(resp, dict) else None
    if um is None:
        um = getattr(resp, "usage_metadata", None) or getattr(resp, "usageMetadata", None)
    if not um:
        return {"prompt_tokens": None, "completion_tokens": None, "cached_input_tokens": None}
    if isinstance(um, dict):
        return {
            "prompt_tokens": um.get("promptTokenCount") or um.get("prompt_token_count"),
            "completion_tokens": um.get("candidatesTokenCount") or um.get("candidates_token_count"),
            "cached_input_tokens": um.get("cachedContentTokenCount") or um.get("cached_content_token_count"),
        }
    # SimpleNamespace / typed model
    return {
        "prompt_tokens": getattr(um, "prompt_token_count", None) or getattr(um, "promptTokenCount", None),
        "completion_tokens": getattr(um, "candidates_token_count", None) or getattr(um, "candidatesTokenCount", None),
        "cached_input_tokens": getattr(um, "cached_content_token_count", None) or getattr(um, "cachedContentTokenCount", None),
    }


def _normalise_usage_mistral(resp: Any) -> dict[str, Optional[int]]:
    usage = resp.get("usage") if isinstance(resp, dict) else None
    if usage is None:
        usage = getattr(resp, "usage", None)
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "cached_input_tokens": None}
    if isinstance(usage, dict):
        return {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_input_tokens": None,  # Mistral has no prompt-cache
        }
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "cached_input_tokens": None,
    }


def _attr(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _translate_inbound(
    primary_shape: ClientShape, target: FallbackTarget, original_params: Any
) -> tuple[Optional[Any], str]:
    """Translate primary's request params → target's params.

    Returns (translated_params, translation_tag).
    translation_tag ∈ {'identity', 'openai→anthropic', 'anthropic→openai',
                       'openai→bedrock', 'bedrock→openai', 'anthropic→bedrock',
                       'bedrock→anthropic', 'openai→gemini', 'gemini→openai',
                       'anthropic→gemini', 'gemini→anthropic', 'bedrock→gemini',
                       'gemini→bedrock', 'unsupported'}.
    translated_params is None when the shape can't be safely crossed —
    caller bubbles the original primary error.
    """
    p = primary_shape
    t = target["provider"]
    if p == t:
        # Identity — model swap only, structural pass-through.
        out = dict(original_params) if isinstance(original_params, dict) else (dict(original_params) if original_params else {})
        out["model"] = target["model"]
        return out, "identity"
    if p == "openai" and t == "anthropic":
        return chat_to_anthropic_params(original_params, target["model"]), "openai→anthropic"
    if p == "anthropic" and t == "openai":
        return anthropic_params_to_chat(original_params, target["model"]), "anthropic→openai"
    if p == "openai" and t == "bedrock":
        return chat_to_bedrock_params(original_params, target["model"]), "openai→bedrock"
    if p == "bedrock" and t == "openai":
        return bedrock_to_chat_params(original_params, target["model"]), "bedrock→openai"
    if p == "anthropic" and t == "bedrock":
        return anthropic_to_bedrock_params(original_params, target["model"]), "anthropic→bedrock"
    if p == "bedrock" and t == "anthropic":
        return bedrock_to_anthropic_params(original_params, target["model"]), "bedrock→anthropic"
    if p == "openai" and t == "gemini":
        return chat_to_gemini_params(original_params, target["model"]), "openai→gemini"
    if p == "gemini" and t == "openai":
        return gemini_to_chat_params(original_params, target["model"]), "gemini→openai"
    if p == "anthropic" and t == "gemini":
        return anthropic_to_gemini_params(original_params, target["model"]), "anthropic→gemini"
    if p == "gemini" and t == "anthropic":
        return gemini_to_anthropic_params(original_params, target["model"]), "gemini→anthropic"
    if p == "bedrock" and t == "gemini":
        return bedrock_to_gemini_params(original_params, target["model"]), "bedrock→gemini"
    if p == "gemini" and t == "bedrock":
        return gemini_to_bedrock_params(original_params, target["model"]), "gemini→bedrock"
    if p == "openai" and t == "mistral":
        return chat_to_mistral_params(original_params, target["model"]), "openai→mistral"
    if p == "mistral" and t == "openai":
        return mistral_to_chat_params(original_params, target["model"]), "mistral→openai"
    if p == "anthropic" and t == "mistral":
        return anthropic_to_mistral_params(original_params, target["model"]), "anthropic→mistral"
    if p == "mistral" and t == "anthropic":
        return mistral_to_anthropic_params(original_params, target["model"]), "mistral→anthropic"
    if p == "bedrock" and t == "mistral":
        return bedrock_to_mistral_params(original_params, target["model"]), "bedrock→mistral"
    if p == "mistral" and t == "bedrock":
        return mistral_to_bedrock_params(original_params, target["model"]), "mistral→bedrock"
    if p == "gemini" and t == "mistral":
        return gemini_to_mistral_params(original_params, target["model"]), "gemini→mistral"
    if p == "mistral" and t == "gemini":
        return mistral_to_gemini_params(original_params, target["model"]), "mistral→gemini"
    # Defense-in-depth — unreachable now that the 5×5 matrix is complete.
    return None, "unsupported"


def _translate_outbound(
    primary_shape: ClientShape,
    target: FallbackTarget,
    raw: Any,
    caller_model: str,
    translation_tag: str,
) -> tuple[Any, dict[str, Optional[int]]]:
    """Translate target's response → primary's response shape.

    Called only on successful invocation; if translation isn't possible
    the caller never reaches here (``_translate_inbound`` already
    returned None and dispatch returned 'unsupported').
    """
    p = primary_shape
    t = target["provider"]
    # 5×5 matrix complete — all identity branches handle their provider.
    if translation_tag == "identity":
        if t == "openai":
            usage = _normalise_usage_openai(raw)
        elif t == "anthropic":
            usage = _normalise_usage_anthropic(raw)
        elif t == "bedrock":
            usage = _normalise_usage_bedrock(raw)
        elif t == "gemini":
            usage = _normalise_usage_gemini(raw)
        elif t == "mistral":
            usage = _normalise_usage_mistral(raw)
        else:
            usage = {"prompt_tokens": None, "completion_tokens": None, "cached_input_tokens": None}
        return raw, usage
    if translation_tag == "openai→anthropic":
        # Primary spoke OpenAI, fallback served via Anthropic. Convert
        # back so the caller still sees ChatCompletion shape.
        return anthropic_response_to_chat(raw, caller_model), _normalise_usage_anthropic(raw)
    if translation_tag == "anthropic→openai":
        # primary spoke Anthropic, fallback served via OpenAI.
        return chat_response_to_anthropic(raw, caller_model), _normalise_usage_openai(raw)
    if translation_tag == "openai→bedrock":
        return bedrock_response_to_chat(raw, caller_model), _normalise_usage_bedrock(raw)
    if translation_tag == "bedrock→openai":
        return chat_response_from_bedrock(raw, caller_model), _normalise_usage_openai(raw)
    if translation_tag == "anthropic→bedrock":
        return bedrock_response_to_anthropic(raw, caller_model), _normalise_usage_bedrock(raw)
    if translation_tag == "bedrock→anthropic":
        return anthropic_response_from_bedrock(raw, caller_model), _normalise_usage_anthropic(raw)
    if translation_tag == "openai→gemini":
        return gemini_response_to_chat(raw, caller_model), _normalise_usage_gemini(raw)
    if translation_tag == "gemini→openai":
        return chat_response_from_gemini(raw, caller_model), _normalise_usage_openai(raw)
    if translation_tag == "anthropic→gemini":
        return gemini_response_to_anthropic(raw, caller_model), _normalise_usage_gemini(raw)
    if translation_tag == "gemini→anthropic":
        return anthropic_response_from_gemini(raw, caller_model), _normalise_usage_anthropic(raw)
    if translation_tag == "bedrock→gemini":
        return gemini_response_to_bedrock(raw, caller_model), _normalise_usage_gemini(raw)
    if translation_tag == "gemini→bedrock":
        return bedrock_response_from_gemini(raw, caller_model), _normalise_usage_bedrock(raw)
    if translation_tag == "openai→mistral":
        return mistral_response_to_chat(raw, caller_model), _normalise_usage_mistral(raw)
    if translation_tag == "mistral→openai":
        return chat_response_from_mistral(raw, caller_model), _normalise_usage_openai(raw)
    if translation_tag == "anthropic→mistral":
        return mistral_response_to_anthropic(raw, caller_model), _normalise_usage_mistral(raw)
    if translation_tag == "mistral→anthropic":
        return anthropic_response_from_mistral(raw, caller_model), _normalise_usage_anthropic(raw)
    if translation_tag == "bedrock→mistral":
        return mistral_response_to_bedrock(raw, caller_model), _normalise_usage_mistral(raw)
    if translation_tag == "mistral→bedrock":
        return bedrock_response_from_mistral(raw, caller_model), _normalise_usage_bedrock(raw)
    if translation_tag == "gemini→mistral":
        return mistral_response_to_gemini(raw, caller_model), _normalise_usage_mistral(raw)
    if translation_tag == "mistral→gemini":
        return gemini_response_from_mistral(raw, caller_model), _normalise_usage_gemini(raw)
    # Defense-in-depth — unreachable now that the 5×5 matrix is complete.
    raise RuntimeError(
        f"unreachable: outbound translation requested for {p}→{t} "
        "but inbound was supposed to refuse it",
    )


def _caller_model(original_params: Any, target: FallbackTarget) -> str:
    m = (
        original_params.get("model")
        if isinstance(original_params, dict)
        else getattr(original_params, "model", None)
    )
    return m if isinstance(m, str) else target["model"]


def _invoke_create(target: FallbackTarget, params: dict[str, Any]) -> Any:
    """Call the target's create-equivalent with the (already-translated) params."""
    prov = target["provider"]
    client = target["client"]
    if prov == "openai":
        return client.chat.completions.create(**params)
    if prov == "anthropic":
        return client.messages.create(**params)
    if prov == "bedrock":
        return client.converse(**params)
    if prov == "gemini":
        return client.models.generate_content(**params)
    if prov == "mistral":
        return client.chat.complete(**params)
    raise RuntimeError(f"unreachable: unknown provider {prov!r}")


# ─── outcomes ──────────────────────────────────────────────────────────────


class FallbackOutcome(TypedDict, total=False):
    """Per-attempt result. Exactly one of ``response`` / ``error`` /
    ``unsupported`` is set."""

    kind: Literal["success", "unsupported", "error"]
    response: Any
    usage: dict[str, Optional[int]]
    latency_ms: int
    error: BaseException
    status: Optional[int]
    reason: str


def _success(response: Any, usage: dict[str, Optional[int]], started_at: float) -> FallbackOutcome:
    return {
        "kind": "success",
        "response": response,
        "usage": usage,
        "latency_ms": max(0, int((time.time() - started_at) * 1000)),
    }


def _error(err: BaseException, started_at: float) -> FallbackOutcome:
    return {
        "kind": "error",
        "error": err,
        "status": _status_of(err),
        "latency_ms": max(0, int((time.time() - started_at) * 1000)),
    }


def _unsupported(reason: str) -> FallbackOutcome:
    return {"kind": "unsupported", "reason": reason}


_UNSUPPORTED_REASON = {
    "openai→anthropic": "v1 cross-provider translation refuses streaming / tools / non-text content",
    "anthropic→openai": "v1 cross-provider translation refuses streaming / non-text content",
    "identity": "translator refused identity translation (unexpected)",
}


# ─── sync dispatcher ───────────────────────────────────────────────────────


def try_fallback_sync(
    primary_shape: ClientShape,
    original_params: Any,
    target: FallbackTarget,
) -> FallbackOutcome:
    translated, tag = _translate_inbound(primary_shape, target, original_params)
    if translated is None:
        reason = (
            f"cross-provider translation for {primary_shape}→{target['provider']} not yet implemented"
            if tag == "unsupported"
            else _UNSUPPORTED_REASON[tag]
        )
        return _unsupported(reason)

    caller_model = _caller_model(original_params, target)
    started_at = time.time()
    try:
        raw = _invoke_create(target, translated)
    except BaseException as err:  # noqa: BLE001 — propagated as outcome
        return _error(err, started_at)

    if asyncio.iscoroutine(raw):
        # Shape mismatch: sync wrap configured with an async fallback
        # client. We can't await without an event loop — close the
        # coroutine cleanly so we don't leak it, then surface a clear
        # error so the caller knows to fix their config.
        raw.close()
        return _error(
            RuntimeError(
                "spaturzu: sync wrap received an async fallback client. "
                "Use a sync client (OpenAI, Anthropic) for sync wraps, "
                "or an async client (AsyncOpenAI, AsyncAnthropic) for "
                "async wraps — don't mix.",
            ),
            started_at,
        )

    response, usage = _translate_outbound(primary_shape, target, raw, caller_model, tag)
    return _success(response, usage, started_at)


# ─── async dispatcher ──────────────────────────────────────────────────────


async def try_fallback_async(
    primary_shape: ClientShape,
    original_params: Any,
    target: FallbackTarget,
) -> FallbackOutcome:
    translated, tag = _translate_inbound(primary_shape, target, original_params)
    if translated is None:
        reason = (
            f"cross-provider translation for {primary_shape}→{target['provider']} not yet implemented"
            if tag == "unsupported"
            else _UNSUPPORTED_REASON[tag]
        )
        return _unsupported(reason)

    caller_model = _caller_model(original_params, target)
    started_at = time.time()
    try:
        raw_or_coro = _invoke_create(target, translated)
        raw = (
            await raw_or_coro
            if asyncio.iscoroutine(raw_or_coro)
            else raw_or_coro
        )
    except BaseException as err:  # noqa: BLE001 — propagated as outcome
        return _error(err, started_at)

    response, usage = _translate_outbound(primary_shape, target, raw, caller_model, tag)
    return _success(response, usage, started_at)


# ─── log entry for a fallback attempt ──────────────────────────────────────


def build_attempt_entry(
    target: FallbackTarget,
    outcome: FallbackOutcome,
    get_process_tags: ProcessTagsGetter,
) -> dict[str, Any]:
    """Build the /v1/logs entry for a single fallback attempt — success or
    failure. Returned dict carries ``id`` (the new request_id); callers
    can ``set_last_request_id`` from it after a successful attempt so
    customer code that fetches the most-recent id sees the row that
    actually served the request."""
    request_id = uuid7()
    frame = get_current_frame()
    base_tags: dict[str, str] = {}
    if frame is not None and frame.tags:
        base_tags.update(frame.tags)
    base_tags["via"] = "fallback"
    tags = merge_tags(get_process_tags(), base_tags)

    latency_ms = outcome.get("latency_ms") or 0

    entry: dict[str, Any] = {
        "id": request_id,
        "provider": target["provider"],
        "model": target["model"],
        "runId": frame.run_id if frame is not None else None,
        "parentRequestId": frame.parent_request_id if frame is not None else None,
        "agentName": frame.agent_name if frame is not None else None,
        "agentPath": list(frame.agent_path) if frame is not None and frame.agent_path else None,
        "latencyMs": latency_ms,
    }
    if tags:
        entry["tags"] = tags

    kind = outcome.get("kind")
    if kind == "success":
        entry["status"] = 200
        entry["usageSource"] = "provider"
        usage = outcome.get("usage") or {}
        if usage.get("prompt_tokens") is not None:
            entry["promptTokens"] = usage["prompt_tokens"]
        if usage.get("completion_tokens") is not None:
            entry["completionTokens"] = usage["completion_tokens"]
        if usage.get("cached_input_tokens") is not None:
            entry["cachedInputTokens"] = usage["cached_input_tokens"]
    else:
        # error — clamp into the validator's 100..599 range
        status = outcome.get("status")
        entry["status"] = max(100, min(599, status)) if isinstance(status, int) else 500

    # Drop None-valued attribution fields so the gateway validator's
    # optional checks see them as absent rather than null.
    return {k: v for k, v in entry.items() if v is not None}
