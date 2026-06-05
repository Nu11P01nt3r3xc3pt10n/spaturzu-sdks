import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  BudgetGuard,
  BudgetExceededError,
  type AgentPolicy,
} from "../src/budget.js";

const POLICY_OK: AgentPolicy = {
  agent: "agent-a",
  agentId: "id-a",
  limits: [
    {
      budgetId: "b1",
      scope: "agent",
      period: "monthly",
      limitCost: "100.00",
      currentCost: "10.00",
      pctUsed: 10,
      alertThresholdPct: 80,
      breached: false,
      enforcement: "hard_cap",
    },
  ],
};

const POLICY_BREACHED: AgentPolicy = {
  agent: "agent-a",
  agentId: "id-a",
  limits: [
    {
      budgetId: "b1",
      scope: "agent",
      period: "monthly",
      limitCost: "100.00",
      currentCost: "150.00",
      pctUsed: 150,
      alertThresholdPct: 80,
      breached: true,
      enforcement: "hard_cap",
    },
  ],
};

function mockFetchOnce(spy: ReturnType<typeof vi.fn>, body: unknown) {
  spy.mockResolvedValueOnce(
    new Response(JSON.stringify(body), { status: 200 }),
  );
}

describe("BudgetGuard: preCallCheck", () => {
  let fetchSpy: ReturnType<typeof vi.fn>;
  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, "fetch") as unknown as ReturnType<typeof vi.fn>;
    // BudgetGuard.constructor does NOT auto-start the SSE loop in v1 — the
    // SSE thread spawns only on guard.start(). These tests never call
    // start(), so the /events branch is dead code. Kept here as a placeholder
    // for when Plan 1-follow-up adds explicit SSE tests.
    // Block the SSE loop from interfering — every test that doesn't care
    // about SSE provides a pending fetch for /events that never resolves.
    fetchSpy.mockImplementation((url: string) => {
      if (url.includes("/events"))
        return new Promise<Response>(() => {
          // never resolves; sseLoop awaits it forever
        });
      return Promise.resolve(new Response("{}", { status: 200 }));
    });
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns normally when no policy applies (404)", async () => {
    fetchSpy.mockImplementationOnce((url: string) => {
      if (url.includes("/policy"))
        return Promise.resolve(new Response("not found", { status: 404 }));
      return new Promise<Response>(() => {});
    });
    const guard = new BudgetGuard({
      baseURL: "https://gw.example",
      refreshIntervalMs: 60_000,
    });
    await expect(guard.preCallCheck("agent-a", "throw")).resolves.toBeUndefined();
    guard.stop();
  });

  it("returns normally on a non-breached hard-cap", async () => {
    fetchSpy.mockImplementationOnce((url: string) => {
      if (url.includes("/policy"))
        return Promise.resolve(
          new Response(JSON.stringify(POLICY_OK), { status: 200 }),
        );
      return new Promise<Response>(() => {});
    });
    const guard = new BudgetGuard({ baseURL: "https://gw.example" });
    await expect(guard.preCallCheck("agent-a", "throw")).resolves.toBeUndefined();
    guard.stop();
  });

  it("throws BudgetExceededError on a breached hard-cap", async () => {
    fetchSpy.mockImplementationOnce((url: string) => {
      if (url.includes("/policy"))
        return Promise.resolve(
          new Response(JSON.stringify(POLICY_BREACHED), { status: 200 }),
        );
      return new Promise<Response>(() => {});
    });
    const guard = new BudgetGuard({ baseURL: "https://gw.example" });
    await expect(guard.preCallCheck("agent-a", "throw")).rejects.toBeInstanceOf(
      BudgetExceededError,
    );
    guard.stop();
  });

  it("warns and proceeds on breach when onBreach='warn'", async () => {
    fetchSpy.mockImplementationOnce((url: string) => {
      if (url.includes("/policy"))
        return Promise.resolve(
          new Response(JSON.stringify(POLICY_BREACHED), { status: 200 }),
        );
      return new Promise<Response>(() => {});
    });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const guard = new BudgetGuard({ baseURL: "https://gw.example" });
    await expect(guard.preCallCheck("agent-a", "warn")).resolves.toBeUndefined();
    expect(warn).toHaveBeenCalledOnce();
    expect(warn.mock.calls[0]![0]).toMatch(/hard-cap budget breached/);
    guard.stop();
  });

  it("caches policy across calls — same agent only fetched once", async () => {
    fetchSpy.mockImplementation((url: string) => {
      if (url.includes("/policy"))
        return Promise.resolve(
          new Response(JSON.stringify(POLICY_OK), { status: 200 }),
        );
      return new Promise<Response>(() => {});
    });
    const guard = new BudgetGuard({ baseURL: "https://gw.example" });
    await guard.preCallCheck("agent-a", "throw");
    await guard.preCallCheck("agent-a", "throw");
    await guard.preCallCheck("agent-a", "throw");
    const policyFetches = fetchSpy.mock.calls.filter((c) =>
      String(c[0]).includes("/policy"),
    );
    expect(policyFetches).toHaveLength(1);
    guard.stop();
  });

  it("resolves the untagged sentinel when agentName is null", async () => {
    let capturedUrl: string | null = null;
    fetchSpy.mockImplementation((url: string) => {
      if (url.includes("/policy")) {
        capturedUrl = url;
        return Promise.resolve(
          new Response(JSON.stringify(POLICY_OK), { status: 200 }),
        );
      }
      return new Promise<Response>(() => {});
    });
    const guard = new BudgetGuard({ baseURL: "https://gw.example" });
    await guard.preCallCheck(null, "throw");
    expect(capturedUrl).toContain(encodeURIComponent("(untagged)"));
    guard.stop();
  });
});

describe("BudgetGuard: error containment", () => {
  let fetchSpy: ReturnType<typeof vi.fn>;
  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, "fetch") as unknown as ReturnType<typeof vi.fn>;
    fetchSpy.mockImplementation(() => new Promise<Response>(() => {}));
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("preCallCheck returns normally when policy fetch fails — never crashes the caller", async () => {
    fetchSpy.mockImplementationOnce((url: string) => {
      if (url.includes("/policy"))
        return Promise.resolve(new Response("server error", { status: 500 }));
      return new Promise<Response>(() => {});
    });
    const errors: unknown[] = [];
    const guard = new BudgetGuard({
      baseURL: "https://gw.example",
      onError: (e) => errors.push(e),
    });
    await expect(guard.preCallCheck("agent-a", "throw")).resolves.toBeUndefined();
    expect(errors).toHaveLength(1);
    guard.stop();
  });
});
