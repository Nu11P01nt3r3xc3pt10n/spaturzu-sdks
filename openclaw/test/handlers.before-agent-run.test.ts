import { describe, it, expect, vi } from "vitest";
import { BudgetExceededError } from "@spaturzu/sdk";
import {
  makeBeforeAgentRunHandler,
  type SpaturzuLike,
} from "../src/handlers.js";
import { parseConfig } from "../src/config.js";

const EVENT = {
  prompt: "hi",
  messages: [],
};

describe("makeBeforeAgentRunHandler", () => {
  it("returns { outcome: 'pass' } when checkBudget resolves", async () => {
    const sp: SpaturzuLike = {
      report: () => {},
      checkBudget: vi.fn().mockResolvedValue(undefined),
    };
    const cfg = parseConfig({ apiKey: "spa_test" });
    const handler = makeBeforeAgentRunHandler(sp, cfg);

    const result = await handler(EVENT, { agentId: "writer" });
    expect(result).toEqual({ outcome: "pass" });
    expect(sp.checkBudget).toHaveBeenCalledWith("openclaw/writer");
  });

  it("returns a cost_limit block on BudgetExceededError", async () => {
    const err = new BudgetExceededError({
      scope: "agent",
      period: "daily",
      limitCost: "5.00",
      currentCost: "5.50",
      agentName: "openclaw/writer",
    });
    const sp: SpaturzuLike = {
      report: () => {},
      checkBudget: vi.fn().mockRejectedValue(err),
    };
    const cfg = parseConfig({ apiKey: "spa_test" });
    const handler = makeBeforeAgentRunHandler(sp, cfg);

    const result = await handler(EVENT, { agentId: "writer" });
    expect(result).toMatchObject({
      outcome: "block",
      category: "cost_limit",
    });
    if (result && result.outcome === "block") {
      expect(result.reason).toContain("agent/daily");
      expect(result.reason).toContain("5.50");
      expect(result.message).toMatch(/spend limit/);
    }
  });

  it("fails OPEN on unexpected error (returns pass)", async () => {
    const sp: SpaturzuLike = {
      report: () => {},
      checkBudget: vi.fn().mockRejectedValue(new Error("network down")),
    };
    const cfg = parseConfig({ apiKey: "spa_test" });
    const handler = makeBeforeAgentRunHandler(sp, cfg);

    const result = await handler(EVENT, { agentId: "writer" });
    expect(result).toEqual({ outcome: "pass" });
  });

  it("resolves agent name via the configured prefix (empty prefix → bare name)", async () => {
    const sp: SpaturzuLike = {
      report: () => {},
      checkBudget: vi.fn().mockResolvedValue(undefined),
    };
    const cfg = parseConfig({ apiKey: "spa_test", agentPrefix: "" });
    const handler = makeBeforeAgentRunHandler(sp, cfg);
    await handler(EVENT, { agentId: "writer" });
    expect(sp.checkBudget).toHaveBeenCalledWith("writer");
  });
});
