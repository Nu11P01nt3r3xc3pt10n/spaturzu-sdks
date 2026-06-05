import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { Spaturzu } from "@spaturzu/sdk";
import { parseConfig } from "./config.js";
import {
  makeBeforeAgentRunHandler,
  makeLlmOutputHandler,
} from "./handlers.js";

/** OpenClaw plugin that:
 *   - meters every model output via `POST /v1/logs` (`llm_output` hook), and
 *   - gates each turn on Spaturzu hard-cap budgets, returning a "cost_limit"
 *     block decision from `before_agent_run` when a budget is breached.
 *
 *  Reads `apiKey` (required), `baseUrl`, `agentPrefix`, `enforceBudgets`,
 *  and static `tags` from the operator's plugin config. Tears down the
 *  shared Logger + BudgetGuard cleanly on `gateway_stop`.
 *
 *  Note: `llm_output` delivery requires the operator to set
 *  `allowConversationAccess: true` for this plugin in OpenClaw's config.
 *  We read `event.usage` only and never transmit prompt or assistant text. */
export default definePluginEntry({
  id: "spaturzu",
  name: "Spaturzu",
  description:
    "Meter LLM usage and enforce budget hard-caps via Spaturzu.",
  register(api) {
    const cfg = parseConfig(api.pluginConfig);
    const sp = new Spaturzu({
      baseURL: cfg.baseUrl,
      apiKey: cfg.apiKey,
      tags: cfg.tags,
      onError: (err) =>
        api.logger.warn(
          `[spaturzu] background error: ${err instanceof Error ? err.message : String(err)}`,
        ),
    });

    api.on("llm_output", makeLlmOutputHandler(sp, cfg));
    if (cfg.enforceBudgets) {
      api.on("before_agent_run", makeBeforeAgentRunHandler(sp, cfg));
    }
    api.on("gateway_stop", async () => {
      await sp.flush();
      await sp.shutdown();
    });
  },
});
