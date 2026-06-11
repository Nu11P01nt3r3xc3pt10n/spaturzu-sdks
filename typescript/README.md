# @spaturzu/sdk

Per-agent LLM cost attribution + budget enforcement + cross-provider fallback for Node.js. Wraps your provider client (OpenAI, Anthropic, Bedrock, Gemini, Mistral) and emits a metering row to the spaturzu gateway on every call, with frame-based agent attribution and free-form tags.

Part of [**spaturzu**](https://spaturzu.superchiu.org) — full docs at <https://spaturzu.superchiu.org/docs>.

```bash
# Published on npm as @spaturzu/sdk (ESM-only).
npm install @spaturzu/sdk

# Plus whichever provider clients you use (normal npm packages):
npm install openai @anthropic-ai/sdk @aws-sdk/client-bedrock-runtime @google/genai @mistralai/mistralai
```

The provider clients are **optional peer dependencies** — install only what you use. `pnpm` / `yarn` work the same (`pnpm add @spaturzu/sdk`); the imports below are unchanged.

## Quickstart

```ts
import { Spaturzu } from "@spaturzu/sdk";
import OpenAI from "openai";

const spaturzu = new Spaturzu({
  baseURL: process.env.SPATURZU_BASE_URL,
  apiKey: process.env.SPATURZU_API_KEY,
  tags: { env: "prod", region: "us-east-1" }, // applied to every call
});

const openai = spaturzu.wrapOpenAI(new OpenAI());

await spaturzu.run("researcher", async () => {
  const r = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Hello" }],
  });
});

// Short-lived processes (CLIs, lambdas) — flush before exit:
await spaturzu.flush();
```

## Drop-in (one-line) instrumentation

Change only the import specifier — construction and call sites stay the same:

```diff
- import OpenAI from "openai";
+ import OpenAI from "@spaturzu/sdk/openai";
  const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  await openai.chat.completions.create({ model: "gpt-4o-mini", messages: [...] });
```

Spaturzu reads its own config from `SPATURZU_API_KEY` / `SPATURZU_BASE_URL`.
Call `configure({...})` once at startup (before constructing any client) for
process-wide tags or an `onError` handler.

| Swap from | …to | Imported as |
|---|---|---|
| `openai` | `@spaturzu/sdk/openai` | `import OpenAI from …` |
| `@anthropic-ai/sdk` | `@spaturzu/sdk/anthropic` | `import Anthropic from …` |
| `@aws-sdk/client-bedrock-runtime` | `@spaturzu/sdk/bedrock` | `import { BedrockRuntime } from …` |
| `@google/genai` | `@spaturzu/sdk/google` | `import { GoogleGenAI } from …` |
| `@mistralai/mistralai` | `@spaturzu/sdk/mistral` | `import { Mistral } from …` |

> **Bedrock:** the drop-in is `BedrockRuntime` (the aggregated client with
> `converse` / `converseStream`), not the low-level `BedrockRuntimeClient`.
> The Command form (`client.send(new ConverseCommand(...))`) is not instrumented.

### Agent attribution without a closure

```ts
await openai.withAgent("writer").chat.completions.create({ ... });
```

`.withAgent(name)` tags that call (and nests under any enclosing `run()` as a
sub-agent). For multi-call workflows, use the closure form:

```ts
import { run, flush } from "@spaturzu/sdk";
await run("workflow", async () => { await openai.chat.completions.create({ ... }); });
await flush();
```

### Budget / fallback on the drop-in path

Pass an optional second constructor argument (ignored by the real SDK):

```ts
const openai = new OpenAI({ apiKey }, { budget: { hardCap: true }, fallback: [...] });
```

> ESM-only, same as the main entry. The drop-in subpaths require the
> corresponding provider package to be installed (it's an optional peer dep).

## Supported providers

| Wrap method | Client | Methods intercepted |
|---|---|---|
| `wrapOpenAI(client)` | `openai` | `chat.completions.create` |
| `wrapAnthropic(client)` | `@anthropic-ai/sdk` | `messages.create` |
| `wrapBedrock(client)` | `@aws-sdk/client-bedrock-runtime` | `converse`, `converseStream` |
| `wrapGemini(client)` | `@google/genai` | `models.generateContent`, `models.generateContentStream` |
| `wrapMistral(client)` | `@mistralai/mistralai` | `chat.complete`, `chat.stream` |

All wraps capture non-streaming + streaming usage, support the budget pre-call gate, and work as cross-provider fallback sources and targets.

### Bedrock

```ts
// Use the aggregated `BedrockRuntime` client (it has the `converse` /
// `converseStream` methods). The low-level `BedrockRuntimeClient` uses the
// Command form (`client.send(...)`) and is not instrumented.
import { BedrockRuntime } from "@aws-sdk/client-bedrock-runtime";
const client = new BedrockRuntime({ region: "us-east-1" });
const wrapped = spaturzu.wrapBedrock(client);

await wrapped.converse({
  modelId: "anthropic.claude-3-5-sonnet-20241022-v2:0",
  messages: [{ role: "user", content: [{ text: "hello" }] }],
});
```

Targets the named-method form only. If you use the Command pattern (`client.send(new ConverseCommand(...))`), migrate to the named methods — the Command form is **not** instrumented in v1.

### Gemini

```ts
import { GoogleGenAI } from "@google/genai";
const client = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const wrapped = spaturzu.wrapGemini(client);

await wrapped.models.generateContent({
  model: "gemini-2.5-pro",
  contents: [{ role: "user", parts: [{ text: "hello" }] }],
});
```

### Mistral

```ts
import { Mistral } from "@mistralai/mistralai";
const client = new Mistral({ apiKey: process.env.MISTRAL_API_KEY });
const wrapped = spaturzu.wrapMistral(client);

await wrapped.chat.complete({
  model: "mistral-large-latest",
  messages: [{ role: "user", content: "hello" }],
});
```

Note: Mistral exposes `chat.complete` and `chat.stream` as separate methods, not a `stream: true` flag.

## Agent frames + tags

`spaturzu.run(name, fn)` opens an agent frame; nested calls share `runId` and extend `agentPath`. Tags merge with the parent frame's; the inner scope wins on key conflict.

```ts
await spaturzu.run("research", async () => {
  await openai.chat.completions.create({ ... });   // agentPath=["research"]

  await spaturzu.run("synthesize", { tags: { phase: "draft" } }, async () => {
    await anthropic.messages.create({ ... });      // agentPath=["research","synthesize"], tags.phase="draft"
  });
});
```

## Budget enforcement (hard cap)

```ts
const openai = spaturzu.wrapOpenAI(new OpenAI(), {
  budget: { hardCap: true, onBreach: "throw" },   // 'warn' to log + proceed
});
```

When a configured hard-cap budget is breached, the wrap throws `BudgetExceededError` **before** hitting the upstream provider. No token cost is incurred for refused calls.

## Cross-provider fallback

Declare a fallback chain to handle provider outages:

```ts
import Anthropic from "@anthropic-ai/sdk";
import { BedrockRuntime } from "@aws-sdk/client-bedrock-runtime";

const openai = spaturzu.wrapOpenAI(new OpenAI(), {
  fallback: [
    {
      provider: "anthropic",
      client: new Anthropic(),
      model: "claude-3-5-haiku-20241022",
    },
    {
      provider: "bedrock",
      client: new BedrockRuntime({ region: "us-east-1" }),
      model: "anthropic.claude-3-5-haiku-20241022-v1:0",
    },
  ],
});
```

On a retryable upstream error (429, 5xx, connection-class) the wrap walks the chain. Request/response shapes are translated automatically — your code keeps receiving OpenAI-shaped responses even when an Anthropic or Bedrock fallback served the call. All 20 directional pairs (5 providers × 4 other-providers) are supported.

**v1 fallback limits:** non-streaming, text content only, no tools, no `response_format`. Calls that use any of those quietly skip fallback and surface the original error.

## API reference

### `new Spaturzu(opts)`

| Option | Type | Default |
|---|---|---|
| `baseURL` | `string` | `process.env.SPATURZU_BASE_URL` ?? `https://spaturzu-api.superchiu.org` |
| `apiKey` | `string` | `process.env.SPATURZU_API_KEY` |
| `timeoutMs` | `number` | `10000` |
| `backoffMs` | `number[]` | `[1000, 2000, 4000, 8000, 16000]` |
| `maxConcurrent` | `number` | `50` |
| `onError` | `(err) => void` | silent |
| `tags` | `Record<string, string \| number \| boolean>` | — |

### `spaturzu.run(name, [opts,] fn)`

Open an agent frame. `opts.tags` adds frame-scoped tags. Returns `fn`'s result.

### `spaturzu.flush()` / `spaturzu.shutdown()`

Block until queued log POSTs settle. `shutdown` also stops the BudgetGuard's SSE + polling threads.

## Notes

- The wraps don't import provider SDKs — they duck-type against runtime shapes, so any compatible client (OpenAI-compat proxies, vLLM, Together, Groq) works for `wrapOpenAI`.
- Tokens come from the provider's `usage` field when present; otherwise the SDK estimates via tiktoken locally (rough for non-OpenAI tokenizers — rows are flagged `usageSource: 'tiktoken'`).
- Streaming captures usage from the final chunk for each provider (OpenAI's `include_usage`, Anthropic's `message_delta`, Bedrock's `metadata`, Gemini's last `usageMetadata`, Mistral's last `data.usage`).
- After a fallback target serves a call, the response is in the **primary provider's shape** (translated). Use the property/key access matching the primary, not the fallback target.
