// Day 20 — hard-cap budget enforcement (client side).
//
// One `BudgetGuard` lives per spaturzu instance. Both wrappers reach into it
// before delegating to the upstream provider's `create()`. The guard:
//
//   1. Lazily fetches `/v1/agents/:name/policy` on the first wrapped call
//      for each agent name. Result cached in-process.
//   2. Refreshes the cache every 60s as a backstop.
//   3. Opens a single SSE connection to `/v1/projects/_me/events`. On a
//      `budget_breach` event it re-fetches the affected policy so the
//      next call sees the up-to-date `breached` flag.
//   4. On the pre-call check, if any applicable hard-cap budget is
//      breached, either throws `BudgetExceededError` (default) or
//      `console.warn`s and lets the call through (`onBreach: 'warn'`).
//
// Race window: ≤1s drift between agent state and SDK enforcement is
// acceptable per the Day-20 spec. Sources of drift:
//   • Gateway ingest publishes the breach edge AFTER the /v1/logs TX
//     commits — sub-tick once the cost is recorded.
//   • SSE write → SDK receive → cache flip — bounded by network latency.
//   • Backstop polling — only relevant when SSE drops; fresh policy
//     fetched on demand at first call for a new agent name.

export type BudgetOnBreach = "throw" | "warn";

export interface BudgetWrapOptions {
  /** Must be `true` to enable enforcement. Default behaviour (no opt-in)
   *  is identical to pre-Day-20 — the wrap is a pure observer. */
  hardCap: boolean;
  /** What to do when a hard-cap budget is breached. Default `'throw'`. */
  onBreach?: BudgetOnBreach;
}

export interface BudgetGuardOptions {
  baseURL: string;
  apiKey?: string;
  /** Background policy refresh interval. Default 60_000ms. Tests pass
   *  shorter values to keep wall time low. */
  refreshIntervalMs?: number;
  /** SSE backoff between reconnect attempts. Default 5_000ms. */
  sseRetryMs?: number;
  /** Surface fetch/SSE failures somewhere the application can see them.
   *  Defaults to a silent drop — the metering plane must not crash the
   *  customer's app. */
  onError?: (err: unknown) => void;
}

export type PolicyLimit = {
  budgetId: string;
  scope: "project" | "agent";
  period: "daily" | "monthly";
  limitCost: string;
  currentCost: string;
  pctUsed: number;
  alertThresholdPct: number;
  breached: boolean;
  enforcement: "alert" | "hard_cap";
};

export type AgentPolicy = {
  agent: string;
  agentId: string | null;
  limits: PolicyLimit[];
};

export class BudgetExceededError extends Error {
  readonly code = "budget_exceeded";
  readonly scope: "project" | "agent";
  readonly period: "daily" | "monthly";
  readonly limitCost: string;
  readonly currentCost: string;
  readonly agentName: string | null;

  constructor(opts: {
    scope: "project" | "agent";
    period: "daily" | "monthly";
    limitCost: string;
    currentCost: string;
    agentName: string | null;
  }) {
    super(
      `spaturzu: ${opts.scope}/${opts.period} budget exceeded (${opts.currentCost} >= ${opts.limitCost})` +
        (opts.agentName ? ` for agent "${opts.agentName}"` : ""),
    );
    this.name = "BudgetExceededError";
    this.scope = opts.scope;
    this.period = opts.period;
    this.limitCost = opts.limitCost;
    this.currentCost = opts.currentCost;
    this.agentName = opts.agentName;
  }
}

const UNTAGGED_NAME = "(untagged)";

export class BudgetGuard {
  private readonly opts: Required<
    Pick<BudgetGuardOptions, "baseURL" | "refreshIntervalMs" | "sseRetryMs">
  > & {
    apiKey: string | undefined;
    onError: (err: unknown) => void;
  };
  private readonly policies = new Map<string, AgentPolicy>();
  private readonly inflight = new Map<string, Promise<AgentPolicy | null>>();
  private sseAbort: AbortController | null = null;
  private refreshHandle: ReturnType<typeof setInterval> | null = null;
  private started = false;

  constructor(opts: BudgetGuardOptions) {
    this.opts = {
      baseURL: opts.baseURL.replace(/\/+$/, ""),
      apiKey: opts.apiKey,
      refreshIntervalMs: opts.refreshIntervalMs ?? 60_000,
      sseRetryMs: opts.sseRetryMs ?? 5_000,
      onError: opts.onError ?? (() => {}),
    };
  }

  start(): void {
    if (this.started) return;
    this.started = true;
    void this.sseLoop();
    this.refreshHandle = setInterval(() => {
      void this.refreshAll();
    }, this.opts.refreshIntervalMs);
    this.refreshHandle.unref?.();
  }

  stop(): void {
    this.started = false;
    this.sseAbort?.abort();
    this.sseAbort = null;
    if (this.refreshHandle) clearInterval(this.refreshHandle);
    this.refreshHandle = null;
  }

  /** Pre-call gate. Throws or warns according to `onBreach`; returns
   *  normally when nothing is breached or when no policy applies. */
  async preCallCheck(
    agentName: string | null,
    onBreach: BudgetOnBreach,
  ): Promise<void> {
    // Untagged calls still feel project-scope hard caps — we resolve them
    // under the "(untagged)" key which the server treats as "no agent row
    // matched, return project-scope budgets only."
    const name = agentName ?? UNTAGGED_NAME;
    const policy = await this.getPolicy(name);
    if (!policy) return;

    for (const limit of policy.limits) {
      if (limit.enforcement !== "hard_cap") continue;
      if (!limit.breached) continue;
      if (onBreach === "warn") {
        // eslint-disable-next-line no-console
        console.warn(
          `[spaturzu] hard-cap budget breached (${limit.scope}/${limit.period}): ` +
            `${limit.currentCost} >= ${limit.limitCost} — proceeding (onBreach='warn')`,
        );
        return;
      }
      throw new BudgetExceededError({
        scope: limit.scope,
        period: limit.period,
        limitCost: limit.limitCost,
        currentCost: limit.currentCost,
        agentName,
      });
    }
  }

  // ─── policy fetch / cache ────────────────────────────────────────────

  private async getPolicy(name: string): Promise<AgentPolicy | null> {
    const cached = this.policies.get(name);
    if (cached) return cached;
    const inflight = this.inflight.get(name);
    if (inflight) return inflight;
    const p = this.fetchPolicy(name).finally(() => {
      this.inflight.delete(name);
    });
    this.inflight.set(name, p);
    return p;
  }

  private async fetchPolicy(name: string): Promise<AgentPolicy | null> {
    try {
      const url = `${this.opts.baseURL}/v1/agents/${encodeURIComponent(name)}/policy`;
      const headers: Record<string, string> = {};
      if (this.opts.apiKey) headers["x-spaturzu-key"] = this.opts.apiKey;
      const res = await fetch(url, { headers });
      if (!res.ok) {
        this.opts.onError(
          new Error(`policy fetch ${res.status} ${res.statusText}`),
        );
        return null;
      }
      const json = (await res.json()) as AgentPolicy;
      this.policies.set(name, json);
      return json;
    } catch (err) {
      this.opts.onError(err);
      return null;
    }
  }

  private async refreshAll(): Promise<void> {
    for (const name of [...this.policies.keys()]) {
      await this.fetchPolicy(name).catch(() => {});
    }
  }

  // ─── SSE ──────────────────────────────────────────────────────────────

  private async sseLoop(): Promise<void> {
    while (this.started) {
      const ac = new AbortController();
      this.sseAbort = ac;
      try {
        const url = `${this.opts.baseURL}/v1/projects/_me/events`;
        const headers: Record<string, string> = { accept: "text/event-stream" };
        if (this.opts.apiKey) headers["x-spaturzu-key"] = this.opts.apiKey;
        const res = await fetch(url, { headers, signal: ac.signal });
        if (!res.ok || !res.body) {
          this.opts.onError(
            new Error(`events ${res.status} ${res.statusText}`),
          );
          if (this.started) await this.sleep(this.opts.sseRetryMs);
          continue;
        }
        await this.parseSSE(res.body);
      } catch (err) {
        if (ac.signal.aborted) return;
        this.opts.onError(err);
        if (this.started) await this.sleep(this.opts.sseRetryMs);
      }
    }
  }

  private async parseSSE(body: ReadableStream<Uint8Array>): Promise<void> {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) return;
      buffer += decoder.decode(value, { stream: true });
      // SSE messages: \n\n-delimited blocks of lines. We accept the
      // "\r\n\r\n" variant too in case a proxy normalises line endings.
      let boundary = this.nextBoundary(buffer);
      while (boundary !== -1) {
        const raw = buffer.slice(0, boundary.index);
        buffer = buffer.slice(boundary.index + boundary.length);
        this.processSSEMessage(raw);
        boundary = this.nextBoundary(buffer);
      }
    }
  }

  private nextBoundary(
    s: string,
  ): { index: number; length: number } | -1 {
    const lf = s.indexOf("\n\n");
    const crlf = s.indexOf("\r\n\r\n");
    if (lf === -1 && crlf === -1) return -1;
    if (lf === -1) return { index: crlf, length: 4 };
    if (crlf === -1) return { index: lf, length: 2 };
    return lf < crlf
      ? { index: lf, length: 2 }
      : { index: crlf, length: 4 };
  }

  private processSSEMessage(msg: string): void {
    let event = "message";
    let data = "";
    for (const line of msg.split(/\r?\n/)) {
      if (!line || line.startsWith(":")) continue;
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    }
    if (event !== "budget_breach" || !data) return;
    try {
      const payload = JSON.parse(data) as {
        scope: "project" | "agent";
        agentName: string | null;
      };
      // Refresh the affected agent's policy so the next call sees the new
      // breached state. Project-scope breaches affect every agent — refresh
      // them all.
      if (payload.scope === "agent" && payload.agentName) {
        void this.fetchPolicy(payload.agentName).catch(() => {});
      } else {
        void this.refreshAll();
      }
    } catch (err) {
      this.opts.onError(err);
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((r) => {
      const t = setTimeout(r, ms);
      t.unref?.();
    });
  }

  // ─── test helpers ─────────────────────────────────────────────────────

  /** Drop in-memory caches. Used by tests that need a fresh fetch. */
  _resetForTests(): void {
    this.policies.clear();
    this.inflight.clear();
  }
}
