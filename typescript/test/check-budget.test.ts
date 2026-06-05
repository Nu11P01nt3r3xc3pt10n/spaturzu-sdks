import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { Spaturzu, BudgetExceededError } from "../src/index.js";

function spyFetch() {
  return vi.spyOn(globalThis, "fetch") as unknown as ReturnType<typeof vi.fn>;
}

/** Mock that returns the supplied policy for /policy URLs and a never-
 *  resolving SSE stream for /events URLs (so the BudgetGuard's sseLoop
 *  hangs harmlessly until shutdown()). Policy errors are signalled by
 *  passing `null` (the mock then returns 500). */
function policyMock(policy: unknown | null) {
  return (input: unknown) => {
    const url = String(input);
    if (url.includes("/v1/agents/")) {
      if (policy === null) return Promise.resolve(new Response("", { status: 500 }));
      return Promise.resolve(
        new Response(JSON.stringify(policy), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );
    }
    if (url.includes("/v1/projects/_me/events")) {
      // ReadableStream with no enqueue — reader.read() never resolves;
      // shutdown() aborts the fetch signal and ends the loop.
      const body = new ReadableStream<Uint8Array>({ start() {} });
      return Promise.resolve(new Response(body, { status: 200 }));
    }
    return Promise.resolve(new Response("", { status: 200 }));
  };
}

describe("Spaturzu.checkBudget()", () => {
  let fetchSpy: ReturnType<typeof spyFetch>;
  let sp: Spaturzu;
  beforeEach(() => {
    fetchSpy = spyFetch();
  });
  afterEach(async () => {
    await sp.shutdown().catch(() => {});
    vi.restoreAllMocks();
  });

  it("throws BudgetExceededError on a breached hard_cap policy", async () => {
    fetchSpy.mockImplementation(
      policyMock({
        agent: "writer",
        agentId: "a1",
        limits: [
          {
            budgetId: "b1",
            scope: "agent",
            period: "daily",
            limitCost: "5.00",
            currentCost: "5.50",
            pctUsed: 110,
            alertThresholdPct: 80,
            breached: true,
            enforcement: "hard_cap",
          },
        ],
      }) as unknown as typeof fetch,
    );
    sp = new Spaturzu({ baseURL: "https://gw.example", apiKey: "spa_test" });

    await expect(sp.checkBudget("writer")).rejects.toBeInstanceOf(
      BudgetExceededError,
    );
  });

  it("resolves when policy has no breached hard_cap limit", async () => {
    fetchSpy.mockImplementation(
      policyMock({
        agent: "writer",
        agentId: "a1",
        limits: [
          {
            budgetId: "b1",
            scope: "project",
            period: "monthly",
            limitCost: "100.00",
            currentCost: "12.34",
            pctUsed: 12,
            alertThresholdPct: 80,
            breached: false,
            enforcement: "hard_cap",
          },
          {
            budgetId: "b2",
            scope: "agent",
            period: "daily",
            limitCost: "5.00",
            currentCost: "5.50",
            pctUsed: 110,
            alertThresholdPct: 80,
            breached: true,
            enforcement: "alert",
          },
        ],
      }) as unknown as typeof fetch,
    );
    sp = new Spaturzu({ baseURL: "https://gw.example", apiKey: "spa_test" });

    await expect(sp.checkBudget("writer")).resolves.toBeUndefined();
  });

  it("fails OPEN when the policy fetch errors (network outage doesn't block)", async () => {
    fetchSpy.mockImplementation(policyMock(null) as unknown as typeof fetch);
    sp = new Spaturzu({ baseURL: "https://gw.example", apiKey: "spa_test" });

    await expect(sp.checkBudget("writer")).resolves.toBeUndefined();
  });
});

import { __resetDefaultForTests } from "../src/default.js";
import {
  configure,
  checkBudget as topLevelCheckBudget,
} from "../src/index.js";

describe("top-level checkBudget() forwards to default singleton", () => {
  let fetchSpy: ReturnType<typeof spyFetch>;
  beforeEach(() => {
    __resetDefaultForTests();
    fetchSpy = spyFetch();
  });
  afterEach(async () => {
    // Tear down the default singleton's guard SSE.
    const { getDefaultSpaturzu } = await import("../src/default.js");
    await getDefaultSpaturzu().shutdown().catch(() => {});
    __resetDefaultForTests();
    vi.restoreAllMocks();
  });

  it("throws BudgetExceededError on breached hard_cap via the singleton", async () => {
    fetchSpy.mockImplementation(
      policyMock({
        agent: "writer",
        agentId: "a1",
        limits: [
          {
            budgetId: "b1",
            scope: "agent",
            period: "daily",
            limitCost: "1.00",
            currentCost: "2.00",
            pctUsed: 200,
            alertThresholdPct: 80,
            breached: true,
            enforcement: "hard_cap",
          },
        ],
      }) as unknown as typeof fetch,
    );
    configure({ baseURL: "https://gw.example", apiKey: "spa_top" });

    await expect(topLevelCheckBudget("writer")).rejects.toBeInstanceOf(
      BudgetExceededError,
    );
  });
});
