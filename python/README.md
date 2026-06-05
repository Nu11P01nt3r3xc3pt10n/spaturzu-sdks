# spaturzu (Python)

Per-agent LLM cost attribution + budget enforcement + cross-provider fallback for Python. Wraps your provider client (OpenAI, Anthropic, Bedrock, Gemini, Mistral) and emits a metering row to the spaturzu gateway on every call, with frame-based agent attribution and free-form tags.

```bash
pip install spaturzu                 # core
pip install spaturzu[openai]         # + OpenAI integration
pip install spaturzu[anthropic]      # + Anthropic
pip install spaturzu[bedrock]        # + boto3 for Bedrock Converse
pip install spaturzu[gemini]         # + google-genai
pip install spaturzu[mistral]        # + mistralai
pip install spaturzu[all]            # everything
```

## Quickstart

```python
from spaturzu import spaturzu
from openai import OpenAI

spaturzu = spaturzu(
    base_url="https://spaturzu-api.example.com",
    api_key="...",
    tags={"env": "prod", "region": "us-east-1"},
)

openai = spaturzu.wrap_openai(OpenAI())

with spaturzu.run("researcher"):
    r = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )

# Short-lived processes — flush before exit:
spaturzu.flush()
```

Both sync (`OpenAI`, `Anthropic`, `Mistral`) and async (`AsyncOpenAI`, `AsyncAnthropic`, `client.chat.complete_async`, `client.aio.models.*`) shapes are supported on a single wrap. Python Bedrock is sync-only in v1 (boto3); `aioboto3` support is a future addition.

## Drop-in (one-line) instrumentation

Change only the import — construction and call sites stay the same:

```diff
- from openai import OpenAI
+ from spaturzu.openai import OpenAI
  client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
  client.chat.completions.create(model="gpt-4o-mini", messages=[...])
```

Spaturzu reads its own config from `SPATURZU_API_KEY` / `SPATURZU_BASE_URL`.
Call `spaturzu.configure(...)` once at startup (before constructing any
client) for process-wide tags or an `on_error` handler.

| Swap from | …to | Exports |
|---|---|---|
| `from openai import OpenAI, AsyncOpenAI` | `from spaturzu.openai import …` | `OpenAI`, `AsyncOpenAI` |
| `from anthropic import Anthropic, AsyncAnthropic` | `from spaturzu.anthropic import …` | `Anthropic`, `AsyncAnthropic` |
| `from google.genai import Client` | `from spaturzu.google import Client` | `Client` |
| `from mistralai import Mistral` | `from spaturzu.mistral import Mistral` | `Mistral` |
| `boto3.client("bedrock-runtime", …)` | `from spaturzu.bedrock import BedrockRuntime` | `BedrockRuntime(...)` |

### Agent attribution without a `with` block

```python
client.with_agent("writer").chat.completions.create(...)
```

`.with_agent(name)` tags that call (and nests under any enclosing
`with spaturzu.run("planner"):` as a sub-agent). For multi-call workflows use
the context manager; top-level `run`/`flush` are importable from `spaturzu`:

```python
from spaturzu import run, flush
with run("workflow"):
    client.chat.completions.create(...)
flush()
```

### Budget / fallback on the drop-in path

```python
client = OpenAI(api_key=..., spaturzu={"budget": {"hard_cap": True}})
```

## Supported providers

| Wrap method | Client | Methods intercepted |
|---|---|---|
| `wrap_openai(client)` | `openai` | `chat.completions.create` (sync + async) |
| `wrap_anthropic(client)` | `anthropic` | `messages.create` (sync + async) |
| `wrap_bedrock(client)` | `boto3.client("bedrock-runtime")` | `converse`, `converse_stream` (sync only in v1) |
| `wrap_gemini(client)` | `google.genai.Client` | `models.*` (sync) + `aio.models.*` (async) |
| `wrap_mistral(client)` | `mistralai.Mistral` | `chat.complete`, `chat.stream`, `chat.complete_async`, `chat.stream_async` |

### Bedrock

```python
import boto3
client = boto3.client("bedrock-runtime", region_name="us-east-1")
wrapped = spaturzu.wrap_bedrock(client)

with spaturzu.run("agent"):
    r = wrapped.converse(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[{"role": "user", "content": [{"text": "hello"}]}],
    )
```

### Gemini

```python
from google import genai
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
wrapped = spaturzu.wrap_gemini(client)

with spaturzu.run("agent"):
    # sync
    r = wrapped.models.generate_content(
        model="gemini-2.5-pro",
        contents=[{"role": "user", "parts": [{"text": "hello"}]}],
    )

    # async — via client.aio.models
    r = await wrapped.aio.models.generate_content(
        model="gemini-2.5-pro",
        contents=[{"role": "user", "parts": [{"text": "hello"}]}],
    )
```

### Mistral

```python
from mistralai import Mistral
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
wrapped = spaturzu.wrap_mistral(client)

with spaturzu.run("agent"):
    # sync
    r = wrapped.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": "hello"}],
    )

    # async
    r = await wrapped.chat.complete_async(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": "hello"}],
    )
```

## Agent frames + tags

```python
async with spaturzu.run("research"):
    await openai.chat.completions.create(...)         # agent_path=["research"]

    async with spaturzu.run("synthesize", tags={"phase": "draft"}):
        await anthropic.messages.create(...)          # path=["research","synthesize"]
```

Use `with` for sync code, `async with` for async. Frames propagate via `contextvars.ContextVar` — each `asyncio.Task` gets its own copy, so parallel tasks see independent frames.

## Budget enforcement

```python
openai = spaturzu.wrap_openai(OpenAI(), budget={"hard_cap": True})
# or budget={"hard_cap": True, "on_breach": "warn"}
```

Raises `BudgetExceededError` (importable from `spaturzu`) before the upstream provider is hit.

## Cross-provider fallback

```python
from anthropic import Anthropic
import boto3

openai = spaturzu.wrap_openai(
    OpenAI(),
    fallback=[
        {
            "provider": "anthropic",
            "client": Anthropic(),
            "model": "claude-3-5-haiku-20241022",
        },
        {
            "provider": "bedrock",
            "client": boto3.client("bedrock-runtime"),
            "model": "anthropic.claude-3-5-haiku-20241022-v1:0",
        },
    ],
)
```

All 20 directional pairs (5 providers × 4 other-providers) are supported. Limitations are identical to the Node SDK: non-streaming, text only, no tools, no `response_format`.

**Note on response shape after fallback:** On the happy path, the wrap returns the provider's native typed object (attribute access). When a fallback target serves the call, the response is a **plain dict** in the primary provider's shape — use subscript access (`resp["choices"][0]["message"]["content"]`, etc.) in code paths that may run after a fallback.

## API reference

### `spaturzu(...)`

| Parameter | Type | Default |
|---|---|---|
| `base_url` | `str` | `$SPATURZU_BASE_URL` ?? hosted gateway |
| `api_key` | `str` | `$SPATURZU_API_KEY` |
| `timeout_s` | `float` | `10.0` |
| `backoff_ms` | `list[int]` | `[1000, 2000, 4000, 8000, 16000]` |
| `max_concurrent` | `int` | `50` |
| `on_error` | `(exc, entry) → None` | silent |
| `tags` | `dict[str, str \| int \| float \| bool]` | — |

### `spaturzu.run(name, *, tags=None)` → context manager

Yields a `RunFrame`. Both sync (`with`) and async (`async with`) usage are supported on the same returned object.

### `spaturzu.flush(timeout_s=None)` / `spaturzu.shutdown()`

Block until queued log POSTs settle. `shutdown` also stops BudgetGuard's SSE + polling threads.
