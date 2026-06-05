import type {
  PluginHookAgentContext,
  PluginHookBeforeAgentRunEvent,
  PluginHookLlmOutputEvent,
  InputGateDecision,
} from "openclaw/plugin-sdk/plugin-entry";
import type { ReportCallInput } from "@spaturzu/sdk";
import { BudgetExceededError } from "@spaturzu/sdk";
import type { SpaturzuOpenClawConfig } from "./config.js";
import { toAgentName, toRunId, toTags } from "./identity.js";

/** The slice of the Spaturzu instance these handlers need.
 *  Extracted as an interface so tests can inject a fake without
 *  constructing a real Spaturzu. */
export interface SpaturzuLike {
  report(call: ReportCallInput): void;
  checkBudget(
    agentName: string | null,
    opts?: { onBreach?: "throw" | "warn" },
  ): Promise<void>;
}

/** Build the `llm_output` hook handler. Fire-and-forget: it reads token
 *  usage off the event and posts a LogEntry to Spaturzu's gateway via
 *  `sp.report()`. Never throws — the host must not crash if the report
 *  builder or the Spaturzu fake misbehaves. */
export function makeLlmOutputHandler(
  sp: SpaturzuLike,
  cfg: SpaturzuOpenClawConfig,
) {
  return (
    event: PluginHookLlmOutputEvent,
    ctx: PluginHookAgentContext,
  ): void => {
    try {
      const tags = toTags(
        { channelId: ctx.channelId, harnessId: event.harnessId },
        // SpaturzuOpenClawConfig.tags has TagInput shape (string | number |
        // boolean); coerce to string at the boundary like the SDK does
        // internally so `toTags` sees a clean Record<string,string>.
        Object.fromEntries(
          Object.entries(cfg.tags).map(([k, v]) => [k, String(v)]),
        ),
      );
      const agentName = toAgentName(ctx.agentId, cfg.agentPrefix);
      sp.report({
        provider: event.provider,
        model: event.model,
        runId: toRunId(event.runId),
        agentName,
        agentPath: [agentName],
        sessionId: event.sessionId,
        promptTokens: event.usage?.input,
        completionTokens: event.usage?.output,
        cachedInputTokens: event.usage?.cacheRead,
        usageSource: "provider",
        status: 200,
        tags,
      });
    } catch {
      // Never crash the OpenClaw host. The SDK Logger already swallows
      // POST failures; this catch covers report() itself throwing.
    }
  };
}

/** Build the `before_agent_run` hook handler. Resolves the agent name,
 *  calls `sp.checkBudget()`, and translates a `BudgetExceededError` into
 *  a block decision with category "cost_limit". Any other error → pass
 *  (fail-open). */
export function makeBeforeAgentRunHandler(
  sp: SpaturzuLike,
  cfg: SpaturzuOpenClawConfig,
) {
  return async (
    _event: PluginHookBeforeAgentRunEvent,
    ctx: PluginHookAgentContext,
  ): Promise<InputGateDecision> => {
    try {
      await sp.checkBudget(toAgentName(ctx.agentId, cfg.agentPrefix));
      return { outcome: "pass" };
    } catch (err) {
      if (err instanceof BudgetExceededError) {
        return {
          outcome: "block",
          category: "cost_limit",
          reason: `spaturzu hard-cap: ${err.scope}/${err.period} ${err.currentCost} >= ${err.limitCost}`,
          message: "This assistant has reached its configured spend limit.",
        };
      }
      // Unexpected error → fail open. The host should never go dark
      // because the metering plane hiccupped.
      return { outcome: "pass" };
    }
  };
}
