"""Tests for the closure-free .with_agent() accessor + call-time frame capture."""
from __future__ import annotations
from typing import Any
import pytest
from spaturzu._openai import wrap_openai
from spaturzu._context import open_frame, close_frame
from tests.helpers.fake_clients import fake_openai, as_async_iterable
from tests.helpers.fake_logger import FakeLogger


def _tags() -> Any:
    return None


def test_unbound_sync_stream_consumed_after_run_closes_still_attributes():
    """Frame-capture fix (unbound path): a stream created inside a run() block
    but drained AFTER the block closes still attributes to that agent. Before
    the call-time capture this logged agentName=None (frame popped at emit)."""
    logger = FakeLogger()
    client = fake_openai(stream=iter([
        {"choices": [{"delta": {"content": "hi"}}]},
        {"usage": {"prompt_tokens": 2, "completion_tokens": 2}},
    ]))
    wrapped = wrap_openai(client, logger, _tags)
    _frame, token = open_frame("solo")
    stream = wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True,
    )
    close_frame(token)  # frame popped before the stream is drained
    for _ in stream:
        pass
    assert logger.entries[0]["agentName"] == "solo"
    assert logger.entries[0]["agentPath"] == ["solo"]


def test_with_agent_tags_call_no_ambient_run():
    logger = FakeLogger()
    client = fake_openai(ok={"usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    wrapped = wrap_openai(client, logger, _tags)
    wrapped.with_agent("writer").chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
    )
    assert len(logger.entries) == 1
    assert logger.entries[0]["agentName"] == "writer"
    assert logger.entries[0]["agentPath"] == ["writer"]
    assert isinstance(logger.entries[0]["runId"], str)


def test_with_agent_nests_under_run():
    logger = FakeLogger()
    client = fake_openai(ok={"usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    wrapped = wrap_openai(client, logger, _tags)
    _frame, token = open_frame("planner")
    try:
        wrapped.with_agent("writer").chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
        )
    finally:
        close_frame(token)
    assert logger.entries[0]["agentPath"] == ["planner", "writer"]


def test_with_agent_sync_stream_consumed_outside_frame():
    logger = FakeLogger()
    client = fake_openai(stream=iter([
        {"choices": [{"delta": {"content": "hi"}}]},
        {"usage": {"prompt_tokens": 3, "completion_tokens": 4}},
    ]))
    wrapped = wrap_openai(client, logger, _tags)
    stream = wrapped.with_agent("streamer").chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True,
    )
    for _ in stream:  # drained here, with no active frame
        pass
    assert len(logger.entries) == 1
    assert logger.entries[0]["agentName"] == "streamer"
    assert logger.entries[0]["promptTokens"] == 3
    assert logger.entries[0]["completionTokens"] == 4


async def test_with_agent_async_stream_consumed_outside_frame():
    logger = FakeLogger()

    # AsyncOpenAI's ``create(stream=True)`` returns a *coroutine* that resolves
    # to the AsyncStream. The fake client returns whatever ``stream=`` is given
    # directly, so wrap the async iterator in a coroutine to mirror that shape
    # (the wrapper dispatches on ``asyncio.iscoroutine`` of the result).
    async def _coro() -> Any:
        return as_async_iterable([
            {"choices": [{"delta": {"content": "hi"}}]},
            {"usage": {"prompt_tokens": 5, "completion_tokens": 6}},
        ])

    client = fake_openai(stream=_coro())
    wrapped = wrap_openai(client, logger, _tags)
    stream = await wrapped.with_agent("astreamer").chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True,
    )
    async for _ in stream:
        pass
    assert logger.entries[0]["agentName"] == "astreamer"


async def test_with_agent_concurrent_no_crosstalk():
    import asyncio
    logger = FakeLogger()

    async def call(agent: str):
        client = fake_openai(ok={"usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        w = wrap_openai(client, logger, _tags)
        w.with_agent(agent).chat.completions.create(model="m", messages=[])

    await asyncio.gather(call("a"), call("b"))
    agents = sorted(e["agentName"] for e in logger.entries)
    assert agents == ["a", "b"]


def test_with_agent_anthropic_nonstreaming():
    from spaturzu._anthropic import wrap_anthropic
    from tests.helpers.fake_clients import fake_anthropic
    logger = FakeLogger()
    client = fake_anthropic(ok={"usage": {"input_tokens": 1, "output_tokens": 1}})
    wrapped = wrap_anthropic(client, logger, _tags)
    wrapped.with_agent("writer").messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "hi"}], max_tokens=16,
    )
    assert logger.entries[0]["agentName"] == "writer"
    assert logger.entries[0]["agentPath"] == ["writer"]


def test_with_agent_bedrock_converse():
    from spaturzu._bedrock import wrap_bedrock
    from tests.helpers.fake_clients import fake_bedrock
    logger = FakeLogger()
    client = fake_bedrock(ok={
        "output": {"message": {"content": [{"text": "hi"}]}},
        "usage": {"inputTokens": 1, "outputTokens": 1},
        "ResponseMetadata": {"HTTPStatusCode": 200},
    })
    wrapped = wrap_bedrock(client, logger, _tags)
    wrapped.with_agent("writer").converse(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
    )
    assert logger.entries[0]["agentName"] == "writer"
    assert logger.entries[0]["agentPath"] == ["writer"]


def test_with_agent_bedrock_converse_stream():
    """Bound streaming path: the observer's finally (run when the stream is
    drained) must use the captured frame, not get_current_frame()."""
    from spaturzu._bedrock import wrap_bedrock
    from tests.helpers.fake_clients import fake_bedrock
    logger = FakeLogger()
    client = fake_bedrock(stream=iter([
        {"contentBlockDelta": {"delta": {"text": "hi"}}},
        {"metadata": {"usage": {"inputTokens": 3, "outputTokens": 4}}},
    ]))
    wrapped = wrap_bedrock(client, logger, _tags)
    resp = wrapped.with_agent("streamer").converse_stream(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
    )
    list(resp["stream"])  # drain → observer finally emits the log entry
    assert logger.entries[0]["agentName"] == "streamer"
    assert logger.entries[0]["agentPath"] == ["streamer"]


class _GeminiUsage:
    """Mimic google-genai's typed UsageMetadata object (snake_case attrs)."""

    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


class _GeminiResp:
    def __init__(self, usage: Any) -> None:
        self.usage_metadata = usage


def test_with_agent_gemini_sync():
    from spaturzu._gemini import wrap_gemini
    from tests.helpers.fake_clients import fake_gemini
    logger = FakeLogger()
    client = fake_gemini(
        ok=_GeminiResp(_GeminiUsage(prompt_token_count=1, candidates_token_count=1))
    )
    wrapped = wrap_gemini(client, logger, _tags)
    wrapped.with_agent("writer").models.generate_content(
        model="gemini-2.5-pro", contents="hi",
    )
    assert logger.entries[0]["agentName"] == "writer"
    assert logger.entries[0]["agentPath"] == ["writer"]


async def test_with_agent_gemini_async():
    """Bound async path: the captured frame (derived at call entry) must drive
    attribution even though ``client.aio.models.generate_content`` is awaited."""
    from spaturzu._gemini import wrap_gemini
    from tests.helpers.fake_clients import fake_gemini
    logger = FakeLogger()
    client = fake_gemini(
        aio_ok=_GeminiResp(_GeminiUsage(prompt_token_count=1, candidates_token_count=1))
    )
    wrapped = wrap_gemini(client, logger, _tags)
    await wrapped.with_agent("awriter").aio.models.generate_content(
        model="gemini-2.5-pro", contents="hi",
    )
    assert logger.entries[0]["agentName"] == "awriter"
    assert logger.entries[0]["agentPath"] == ["awriter"]


class _MistralUsage:
    """Mimic mistralai's typed usage object (snake_case attrs)."""

    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _MistralResp:
    def __init__(self, usage: Any) -> None:
        self.usage = usage


def test_with_agent_mistral_complete():
    from spaturzu._mistral import wrap_mistral
    from tests.helpers.fake_clients import fake_mistral
    logger = FakeLogger()
    client = fake_mistral(ok=_MistralResp(_MistralUsage(prompt_tokens=1, completion_tokens=1)))
    wrapped = wrap_mistral(client, logger, _tags)
    wrapped.with_agent("writer").chat.complete(
        model="mistral-large-latest", messages=[{"role": "user", "content": "hi"}],
    )
    assert logger.entries[0]["agentName"] == "writer"
    assert logger.entries[0]["agentPath"] == ["writer"]
