"""Duck-typed minimal LLM clients for Python SDK tests.

Each factory builds an object exposing only the methods the corresponding
wrap intercepts. Plans 3/4/5 will add fake_bedrock / fake_gemini /
fake_mistral here. AsyncOpenAI / AsyncAnthropic shapes are deferred until
the async-wrap branch needs them.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator
from unittest.mock import MagicMock


def fake_openai(*, ok: Any = None, err: BaseException | None = None, stream: Iterator[Any] | None = None) -> Any:
    """Build a fake sync OpenAI client.

    Exactly one of `ok`/`err`/`stream` should be provided.
    """
    create = MagicMock()

    def _create(**params: Any) -> Any:
        if err is not None:
            raise err
        if stream is not None:
            return stream
        return ok

    create.side_effect = _create
    completions = MagicMock()
    completions.create = create
    chat = MagicMock()
    chat.completions = completions
    client = MagicMock()
    client.chat = chat
    client.__create = create  # expose for assertions
    return client


def fake_anthropic(*, ok: Any = None, err: BaseException | None = None, stream: Iterator[Any] | None = None) -> Any:
    create = MagicMock()

    def _create(**params: Any) -> Any:
        if err is not None:
            raise err
        if stream is not None:
            return stream
        return ok

    create.side_effect = _create
    messages = MagicMock()
    messages.create = create
    client = MagicMock()
    client.messages = messages
    client.__create = create
    return client


def as_async_iterable(items: list[Any]) -> AsyncIterator[Any]:
    """Turn a list into a one-shot async iterator (for stream tests)."""

    async def _gen() -> AsyncIterator[Any]:
        for it in items:
            yield it

    return _gen()


def openai_error(status: int, name: str = "APIError") -> Exception:
    cls = type(name, (Exception,), {})
    err = cls(f"fake openai error {status}")
    err.status_code = status  # type: ignore[attr-defined]
    err.status = status  # type: ignore[attr-defined]
    return err


def anthropic_error(status: int, name: str = "APIError") -> Exception:
    return openai_error(status, name)


def fake_bedrock(
    *,
    ok: Any = None,
    err: BaseException | None = None,
    stream: Any = None,
    stream_err: BaseException | None = None,
) -> Any:
    """Build a fake boto3 bedrock-runtime client.

    Exposes `converse` and `converse_stream`. Both return dicts (mirrors
    boto3's runtime shape — no typed responses).
    """
    converse = MagicMock()
    converse_stream = MagicMock()

    def _converse(**params: Any) -> Any:
        if err is not None:
            raise err
        return ok

    def _converse_stream(**params: Any) -> Any:
        if stream_err is not None:
            raise stream_err
        # If caller passed `stream` (an iterator of events), wrap it in
        # the boto3 shape; otherwise return whatever ok was.
        if stream is not None:
            return {"stream": stream, "ResponseMetadata": {"HTTPStatusCode": 200}}
        return ok

    converse.side_effect = _converse
    converse_stream.side_effect = _converse_stream

    client = MagicMock()
    client.converse = converse
    client.converse_stream = converse_stream
    client.__converse = converse
    client.__converse_stream = converse_stream
    return client


def bedrock_error(status: int, name: str = "ThrottlingException") -> Exception:
    """Build a botocore-shaped error.

    boto3 raises ``botocore.exceptions.ClientError`` whose ``response``
    dict carries ``ResponseMetadata.HTTPStatusCode``. We mimic that shape
    on a plain Exception so we don't need to import botocore in tests.

    NOTE: Earlier in Plan 1 we discovered that `err.__class__ = type(name, ...)`
    fails on Python 3.11 due to layout incompatibility. Construct the
    exception via the dynamic class directly.
    """
    cls = type(name, (Exception,), {})
    err = cls(f"fake bedrock error {status}")
    err.response = {"ResponseMetadata": {"HTTPStatusCode": status}}  # type: ignore[attr-defined]
    return err


def fake_gemini(
    *,
    ok: Any = None,
    err: BaseException | None = None,
    stream: Any = None,
    aio_ok: Any = None,
    aio_err: BaseException | None = None,
    aio_stream: Any = None,
) -> Any:
    """Build a fake google-genai client with sync + async surfaces.

    ``client.models.*`` are sync; ``client.aio.models.*`` are async.
    """
    def _sync_gen_content(**params: Any) -> Any:
        if err is not None:
            raise err
        return ok

    def _sync_gen_stream(**params: Any) -> Any:
        if err is not None:
            raise err
        return stream if stream is not None else iter([])

    async def _async_gen_content(**params: Any) -> Any:
        if aio_err is not None:
            raise aio_err
        return aio_ok if aio_ok is not None else ok

    async def _async_gen_stream(**params: Any) -> Any:
        if aio_err is not None:
            raise aio_err
        return aio_stream if aio_stream is not None else (stream or iter([]))

    sync_models = MagicMock()
    sync_models.generate_content = MagicMock(side_effect=_sync_gen_content)
    sync_models.generate_content_stream = MagicMock(side_effect=_sync_gen_stream)

    aio_models = MagicMock()
    aio_models.generate_content = MagicMock(side_effect=_async_gen_content)
    aio_models.generate_content_stream = MagicMock(side_effect=_async_gen_stream)

    aio = MagicMock()
    aio.models = aio_models

    client = MagicMock()
    client.models = sync_models
    client.aio = aio
    client.__sync_models = sync_models
    client.__aio_models = aio_models
    return client


def gemini_error(status: int, name: str = "ApiError") -> Exception:
    """Build a google-genai-shaped error. Note: per Plan 1 finding, use the
    dynamic-class-construction pattern (not `__class__` assignment) to
    avoid Python 3.11 layout incompatibility."""
    cls = type(name, (Exception,), {})
    err = cls(f"fake gemini error {status}")
    err.status = status  # type: ignore[attr-defined]
    return err


def fake_mistral(
    *,
    ok: Any = None,
    err: BaseException | None = None,
    stream: Any = None,
    aio_ok: Any = None,
    aio_err: BaseException | None = None,
    aio_stream: Any = None,
) -> Any:
    """Build a fake mistralai client (sync + async).

    Mistral exposes ``chat.complete`` / ``chat.stream`` (sync) and
    ``chat.complete_async`` / ``chat.stream_async`` (async).
    """

    def _sync_complete(**params: Any) -> Any:
        if err is not None:
            raise err
        return ok

    def _sync_stream(**params: Any) -> Any:
        if err is not None:
            raise err
        return stream if stream is not None else iter([])

    async def _async_complete(**params: Any) -> Any:
        if aio_err is not None:
            raise aio_err
        return aio_ok if aio_ok is not None else ok

    async def _async_stream(**params: Any) -> Any:
        if aio_err is not None:
            raise aio_err
        return aio_stream if aio_stream is not None else (stream or iter([]))

    chat = MagicMock()
    chat.complete = MagicMock(side_effect=_sync_complete)
    chat.stream = MagicMock(side_effect=_sync_stream)
    chat.complete_async = MagicMock(side_effect=_async_complete)
    chat.stream_async = MagicMock(side_effect=_async_stream)
    client = MagicMock()
    client.chat = chat
    return client


def mistral_error(status: int, name: str = "SDKError") -> Exception:
    """Build a mistralai-shaped error. Uses the dynamic-class-construction
    pattern to avoid the Python 3.11 ``__class__`` layout incompatibility
    noted in Plan 1."""
    cls = type(name, (Exception,), {})
    err = cls(f"fake mistral error {status}")
    err.status_code = status  # type: ignore[attr-defined]
    err.status = status  # type: ignore[attr-defined]
    return err
