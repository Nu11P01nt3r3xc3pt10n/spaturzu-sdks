# @spaturzu/openclaw

OpenClaw plugin that meters every LLM call and enforces Spaturzu hard-cap
budgets across every channel and agent your assistant runs.

## What it does

- **Metering** — on `llm_output`, posts a Spaturzu `LogEntry`
  (`provider`, `model`, `runId`, `agentName`, `sessionId`, token counts,
  tags) to your Spaturzu gateway. Fire-and-forget; never blocks the host.
- **Budget hard-caps** — on `before_agent_run`, asks Spaturzu whether the
  resolved agent has any breached `hard_cap` budget; if so, returns a
  `cost_limit` block decision and the turn is stopped before any LLM call.
- **Fails open** — if the Spaturzu gateway is unreachable, the gate
  resolves to "pass" so your assistant stays up.

## Install

````bash
npm install @spaturzu/openclaw
# or pnpm/yarn/bun
````

Then enable it in your OpenClaw config:

````jsonc
{
  "plugins": {
    "spaturzu": {
      "enabled": true,
      "hooks": {
        // Required: llm_output exposes conversation content for plugins.
        // This plugin reads event.usage only and never transmits prompt
        // or assistant text — see the Privacy section below.
        "allowConversationAccess": true
      },
      "config": {
        "apiKey": "spa_…",
        "baseUrl": "https://spaturzu-api.superchiu.org",
        "agentPrefix": "openclaw",
        "enforceBudgets": true,
        "tags": { "env": "prod" }
      }
    }
  }
}
````

## Config

| Field             | Type                            | Default                                | Notes |
|-------------------|---------------------------------|----------------------------------------|---|
| `apiKey`          | string (required)               | —                                      | Spaturzu project API key (`spa_…`). |
| `baseUrl`         | URL                             | `https://spaturzu-api.superchiu.org`   | Spaturzu gateway. |
| `agentPrefix`     | string                          | `"openclaw"`                           | Prepended to `ctx.agentId`. `""` disables. |
| `enforceBudgets`  | boolean                         | `true`                                 | Register the `before_agent_run` gate. |
| `tags`            | record<string, string\|num\|bool> | `{}`                                 | Static tags merged into every reported call. |

## Privacy

The plugin reads `event.usage` only and transmits **token counts, never
prompt or assistant text**. The Spaturzu gateway's log schema does not have
a field for message text, so it cannot accidentally exfiltrate one. The
`allowConversationAccess: true` setting is required by OpenClaw to deliver
the `llm_output` event in the first place — not because this plugin uses
the conversation content.

## How it maps OpenClaw → Spaturzu

| OpenClaw                | Spaturzu                                                |
|-------------------------|---------------------------------------------------------|
| `event.runId`           | `runId` (UUIDv5-derived when not already a UUID)        |
| `ctx.agentId`           | `agentName` (with `agentPrefix`)                        |
| `event.sessionId`       | `sessionId`                                             |
| `event.provider/model`  | `provider` / `model`                                    |
| `event.usage.input`     | `promptTokens`                                          |
| `event.usage.output`    | `completionTokens`                                      |
| `event.usage.cacheRead` | `cachedInputTokens`                                     |
| `ctx.channelId`         | `tags.channel`                                          |
| `event.harnessId`       | `tags.harness`                                          |

Every model call from one OpenClaw run collapses into one Spaturzu
`agent_run`.
