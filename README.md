# spaturzu SDKs

Open-source client SDKs for [**spaturzu**](https://spaturzu.superchiu.org) —
per-agent LLM cost attribution, budget enforcement, and cross-provider
fallback. Wrap your existing provider client (OpenAI, Anthropic, Bedrock,
Gemini, Mistral) and every call is metered, attributed to the agent that made
it, and optionally budget-capped — without changing how you call the model.

```ts
const openai = spaturzu.wrapOpenAI(new OpenAI());

await spaturzu.run("researcher", async () => {
  // every call inside here is metered and attributed to "researcher"
  await openai.chat.completions.create({ model: "gpt-4o", messages });
});
```

## SDKs

| SDK | Package | Install | Docs |
|-----|---------|---------|------|
| **TypeScript / Node** | [`@spaturzu/sdk`](./typescript) | `pnpm add @spaturzu/sdk` | [README](./typescript/README.md) |
| **Python** | [`spaturzu`](./python) | `pip install spaturzu` | [README](./python/README.md) |
| **OpenClaw plugin** ⚠️ *experimental* | [`@spaturzu/openclaw`](./openclaw) | — | [README](./openclaw/README.md) |

> ⚠️ **`@spaturzu/openclaw` is a work in progress** and not yet ready for
> production use. APIs may change without notice.

Both the TypeScript and Python SDKs treat the underlying provider clients as
**optional dependencies** — install only the ones you actually call.

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
