# spaturzu SDKs

Open-source client SDKs for [**spaturzu**](https://spaturzu.superchiu.org) —
per-agent LLM cost attribution, budget enforcement, and cross-provider
fallback. Wrap your existing provider client (OpenAI, Anthropic, Bedrock,
Gemini, Mistral) and every call is metered, attributed to the agent that made
it, and optionally budget-capped — without changing how you call the model.

## Migrate an existing agent in one line

Already calling OpenAI, Anthropic, Bedrock, Gemini, or Mistral? Change a single
import — from `import OpenAI from "openai"` to
`import OpenAI from "@spaturzu/sdk/openai"`. Construction and call sites stay
exactly the same, and you can attribute each call to the agent that made it:

```ts
import OpenAI from "@spaturzu/sdk/openai";

// Swap one import — reads SPATURZU_API_KEY + OPENAI_API_KEY from env.
const openai = new OpenAI();

// Tag any call with the agent that made it — one line, no closure.
await openai.withAgent("support-triage").chat.completions.create({ /* … */ });
```

That's the whole migration — no new objects, no wrappers around your call
sites. **Python is identical:** `from spaturzu.openai import OpenAI`, then
`client.with_agent("support-triage").chat.completions.create(...)`. The same
one-import swap works for every provider — `@spaturzu/sdk/anthropic`,
`/bedrock`, `/google`, `/mistral` (Python: `spaturzu.anthropic`, and so on).

> Running multi-step workflows? Group several calls under one agent with
> `run()` instead — see [What you can do](#what-you-can-do) below.

## SDKs

| SDK | Package | Docs |
|-----|---------|------|
| **TypeScript / Node** | [`@spaturzu/sdk`](./typescript) | [README](./typescript/README.md) |
| **Python** | [`spaturzu`](./python) | [README](./python/README.md) |
| **OpenClaw plugin** ⚠️ *experimental* | [`@spaturzu/openclaw`](./openclaw) | [README](./openclaw/README.md) |

> ⚠️ **`@spaturzu/openclaw` is a work in progress** and not yet ready for
> production use. APIs may change without notice.

Both the TypeScript and Python SDKs treat the underlying provider clients as
**optional dependencies** — install only the ones you actually call.

## Installation

The SDKs are distributed as downloadable artifacts from
**https://spaturzu-sdk.superchiu.org** — they are not on npm / PyPI. Install
straight from the artifact URL; the newest version is always listed on that
page (bump the version in the commands below to match).

**TypeScript / Node** — current version `0.1.3`:

```bash
npm install https://spaturzu-sdk.superchiu.org/sdks/ts/spaturzu-sdk-0.1.3.tgz
# …then whichever provider clients you call (these are normal npm packages):
npm install openai @anthropic-ai/sdk @aws-sdk/client-bedrock-runtime @google/genai @mistralai/mistralai
```

**Python** — current version `0.1.4`:

```bash
WHL=https://spaturzu-sdk.superchiu.org/sdks/python/spaturzu-0.1.4-py3-none-any.whl
pip install "spaturzu @ $WHL"               # core
pip install "spaturzu[openai] @ $WHL"       # with the OpenAI integration
pip install "spaturzu[all] @ $WHL"          # every provider integration
```

Once installed, the package names are unchanged (`@spaturzu/sdk`, `spaturzu`),
so every import in this README works as written. `pnpm` / `yarn` accept the
same tarball URL.

## Documentation

- **Full docs & guides:** https://spaturzu.superchiu.org/docs
- **TypeScript / Node API:** [`typescript/README.md`](./typescript/README.md)
- **Python API:** [`python/README.md`](./python/README.md)
- **OpenClaw plugin** (experimental): [`openclaw/README.md`](./openclaw/README.md)
- **For AI tools / LLMs:** https://spaturzu.superchiu.org/llms.txt

## What you can do

The examples below all build on this one-time setup. Every later snippet
reuses `spaturzu`/`sp`, `openai`, and `messages` from here.

```ts
// TypeScript
import { Spaturzu } from "@spaturzu/sdk";
import OpenAI from "openai";

const spaturzu = new Spaturzu({ apiKey: process.env.SPATURZU_API_KEY });
const openai = spaturzu.wrapOpenAI(new OpenAI());

const messages = [{ role: "user", content: "Summarize the latest sales report." }];
```

```python
# Python
import os
from spaturzu import spaturzu
from openai import OpenAI

sp = spaturzu(api_key=os.environ["SPATURZU_API_KEY"])
openai = sp.wrap_openai(OpenAI())

messages = [{"role": "user", "content": "Summarize the latest sales report."}]
```

> Wrapping is transparent: the wrapped client has the **same methods and
> return types** as the original. You keep calling the provider exactly as
> before — spaturzu just meters each call in the background.

### 1. Attribute cost to a named agent

Wrap a block of work in `run("name", …)` and every model call inside it is
billed to that agent — so the dashboard shows cost *per agent*, not one
undifferentiated total.

```ts
// TypeScript
await spaturzu.run("researcher", async () => {
  await openai.chat.completions.create({ model: "gpt-4o", messages });
});
```

```python
# Python
with sp.run("researcher"):
    openai.chat.completions.create(model="gpt-4o", messages=messages)
```

### 2. Nest sub-agents into a cost tree

Nested `run()` calls share one run id and extend the agent path, so a
multi-step workflow shows up as a tree (`research › synthesize`).

```ts
// TypeScript
await spaturzu.run("research", async () => {
  await openai.chat.completions.create({ model: "gpt-4o", messages }); // path: research

  await spaturzu.run("synthesize", async () => {
    await openai.chat.completions.create({ model: "gpt-4o", messages }); // path: research › synthesize
  });
});
```

```python
# Python
with sp.run("research"):
    openai.chat.completions.create(model="gpt-4o", messages=messages)        # path: research
    with sp.run("synthesize"):
        openai.chat.completions.create(model="gpt-4o", messages=messages)    # path: research › synthesize
```

### 3. Slice cost by team, customer, or environment with tags

Set tags globally on the client, or per-frame on a `run()`. Frame tags merge
with the global ones (inner wins on conflict), so you can break spend down by
any dimension you like.

```ts
// TypeScript
const spaturzu = new Spaturzu({
  apiKey: process.env.SPATURZU_API_KEY,
  tags: { env: "prod", team: "growth" },        // on every call
});

await spaturzu.run("billing-agent", { tags: { customer: "acme" } }, async () => {
  await openai.chat.completions.create({ model: "gpt-4o", messages });
});
```

```python
# Python
sp = spaturzu(
    api_key=os.environ["SPATURZU_API_KEY"],
    tags={"env": "prod", "team": "growth"},       # on every call
)

with sp.run("billing-agent", tags={"customer": "acme"}):
    openai.chat.completions.create(model="gpt-4o", messages=messages)
```

### 4. Stop runaway spend with a hard-cap budget

When an agent's budget is exhausted, the wrapped call raises
`BudgetExceededError` **before** it reaches the provider — so a refused call
costs nothing. (Use `onBreach: "warn"` / `"on_breach": "warn"` to log and
proceed instead of throwing.)

```ts
// TypeScript
import { BudgetExceededError } from "@spaturzu/sdk";

const capped = spaturzu.wrapOpenAI(new OpenAI(), {
  budget: { hardCap: true, onBreach: "throw" },
});

try {
  await capped.chat.completions.create({ model: "gpt-4o", messages });
} catch (err) {
  if (err instanceof BudgetExceededError) {
    // budget hit — the request never left your process, so no tokens were spent
  }
}
```

```python
# Python
from spaturzu import BudgetExceededError

capped = sp.wrap_openai(OpenAI(), budget={"hard_cap": True, "on_breach": "throw"})

try:
    capped.chat.completions.create(model="gpt-4o", messages=messages)
except BudgetExceededError:
    pass  # call never reached OpenAI — no spend
```

### 5. Survive a provider outage with cross-provider fallback

Give a wrap a fallback chain. On a retryable error (429 / 5xx / connection),
spaturzu transparently retries the next provider — and **translates the
response back to your primary provider's shape**, so your code is unchanged.

```ts
// TypeScript
import Anthropic from "@anthropic-ai/sdk";

const resilient = spaturzu.wrapOpenAI(new OpenAI(), {
  fallback: [
    { provider: "anthropic", client: new Anthropic(), model: "claude-3-5-haiku-20241022" },
  ],
});

// If OpenAI is down, this is served by Anthropic — still returns an OpenAI-shaped response.
const r = await resilient.chat.completions.create({ model: "gpt-4o", messages });
```

```python
# Python
from anthropic import Anthropic

resilient = sp.wrap_openai(OpenAI(), fallback=[
    {"provider": "anthropic", "client": Anthropic(), "model": "claude-3-5-haiku-20241022"},
])

r = resilient.chat.completions.create(model="gpt-4o", messages=messages)
```

> All 20 directional provider pairs are supported. v1 fallback is
> non-streaming, text-only (no tools / `response_format`).

### One wrap method per provider

The same five providers, the same shape, in both languages. Streaming and
sync/async calls are metered automatically — no extra configuration.

| Provider | TypeScript | Python |
|---|---|---|
| OpenAI (+ OpenAI-compatible) | `wrapOpenAI` | `wrap_openai` |
| Anthropic | `wrapAnthropic` | `wrap_anthropic` |
| Amazon Bedrock | `wrapBedrock` | `wrap_bedrock` |
| Google Gemini | `wrapGemini` | `wrap_gemini` |
| Mistral | `wrapMistral` | `wrap_mistral` |

> **Short-lived processes** (CLIs, serverless): call `spaturzu.flush()` /
> `sp.flush()` before exit so queued metering rows are sent.

For the complete API — every option, streaming details, and per-provider
notes — see the [TypeScript](./typescript/README.md) and
[Python](./python/README.md) READMEs, or the full docs at
**https://spaturzu.superchiu.org/docs**.

## Repository layout

```
sdks/
├── typescript/   @spaturzu/sdk        — Node/TS SDK (5 providers, 20 fallback pairs)
├── python/       spaturzu             — Python SDK (parity with the TS surface)
└── openclaw/     @spaturzu/openclaw   — OpenClaw metering/budget plugin (WIP)
```

## Development

**TypeScript packages** (managed as a pnpm workspace):

```bash
pnpm install
pnpm build       # build all packages
pnpm test        # run all test suites
pnpm typecheck
```

**Python SDK:**

```bash
cd python
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,all]"
.venv/bin/pytest
```

## License

[MIT](./LICENSE) © Superchiu Ltd

spaturzu is a product of **Superchiu Ltd**.
