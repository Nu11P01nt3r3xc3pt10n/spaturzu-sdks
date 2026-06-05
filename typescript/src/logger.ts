import type { LogEntry } from "./wire-types.js";

// LogEntry plus the SDK-supplied request id. We send the id over the wire so
// nested agents can reference the parent's request without round-tripping the
// log POST first — fire-and-forget would otherwise force a `.then()` chain on
// every call, defeating the point. Day 7: the same id makes retries
// idempotent on the gateway via UNIQUE(project_id, id) + ON CONFLICT DO
// NOTHING.
export type SdkLogEntry = LogEntry & { id?: string };

// Default exponential-backoff schedule for retries. Each entry is the wait
// before the *next* attempt — so [1s, 2s, 4s, 8s, 16s] yields up to 5 retries
// after the initial send (6 total attempts). Total worst-case retry wall
// time: 1+2+4+8+16 = 31s + per-attempt timeouts.
const DEFAULT_BACKOFF_MS = [1_000, 2_000, 4_000, 8_000, 16_000];

// HTTP-shaped error so the retry loop can distinguish "server said no" from
// "the network died." Network errors (TypeError from fetch, AbortError on
// timeout) bubble up as plain Errors and are treated as retryable.
export class HttpError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "HttpError";
  }
}

// 5xx → retry (server hiccup). 429 → retry (rate-limited; Day 17 will set
// Retry-After). Everything else 4xx → fatal: validation/auth errors won't
// resolve by retrying.
function isRetryable(err: unknown): boolean {
  if (err instanceof HttpError) {
    return err.status >= 500 || err.status === 429;
  }
  return true;
}

// ±25% jitter so a hundred SDKs reconnecting after a gateway blip don't
// thunder back in lockstep on the next attempt boundary.
function jitter(ms: number): number {
  return ms * (0.75 + Math.random() * 0.5);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export interface LoggerOptions {
  /** Per-attempt request timeout. Default 10s. */
  timeoutMs?: number;
  /** Backoff schedule between retries. Length = number of retries; default
   *  matches DEFAULT_BACKOFF_MS. Set `[]` to disable retries. */
  backoffMs?: number[];
  /** Max in-flight log POSTs (counting retries as one slot). Excess calls
   *  queue FIFO and acquire a slot when one frees. Default 50.
   *
   *  Why this cap: an unhinged customer app that fires 10K LLM calls in a
   *  burst would otherwise open 10K concurrent fetches to the gateway, each
   *  with its own 5-step retry chain. The gateway and the local socket
   *  table both fall over. 50 is high enough that legitimate parallel agent
   *  workflows never queue. */
  maxConcurrent?: number;
}

const DEFAULT_MAX_CONCURRENT = 50;

// Fire-and-forget log poster. `log()` returns sync; the HTTP POST runs on the
// background. Failures are swallowed by default so a wonky gateway never
// crashes the customer's app — surface them via `onError` if you want them.
//
// `flush()` is the escape hatch for short-lived processes (CI runs, lambdas,
// CLI scripts): call `await spaturzu.flush()` before exit so in-flight logs
// reach the gateway. Long-running servers can ignore it. Retries are awaited
// transparently — flush() doesn't return until each entry has either
// succeeded or burned through its retry budget.
export class Logger {
  private inflight = new Set<Promise<unknown>>();
  private timeoutMs: number;
  private backoffMs: number[];
  private maxConcurrent: number;
  private active = 0;
  private waiters: Array<() => void> = [];

  constructor(
    private baseURL: string,
    private apiKey: string | undefined,
    private onError: (err: unknown, entry: SdkLogEntry) => void = () => {},
    opts: LoggerOptions = {},
  ) {
    this.timeoutMs = opts.timeoutMs ?? 10_000;
    this.backoffMs = opts.backoffMs ?? DEFAULT_BACKOFF_MS;
    this.maxConcurrent = opts.maxConcurrent ?? DEFAULT_MAX_CONCURRENT;
  }

  log(entry: SdkLogEntry): void {
    const p = this.acquireAndSend(entry).catch((err) => {
      this.onError(err, entry);
    });
    this.inflight.add(p);
    p.finally(() => this.inflight.delete(p));
  }

  private async acquireAndSend(entry: SdkLogEntry): Promise<void> {
    await this.acquireSlot();
    try {
      await this.sendWithRetries(entry);
    } finally {
      this.releaseSlot();
    }
  }

  // FIFO semaphore. A waiter that gets popped is responsible for the
  // active++ — that way `active` only ever counts log entries currently
  // either sending or about to start, never queued ones.
  private acquireSlot(): Promise<void> {
    if (this.active < this.maxConcurrent) {
      this.active++;
      return Promise.resolve();
    }
    return new Promise<void>((resolve) => {
      this.waiters.push(() => {
        this.active++;
        resolve();
      });
    });
  }

  private releaseSlot(): void {
    this.active--;
    const next = this.waiters.shift();
    if (next) next();
  }

  async flush(): Promise<void> {
    // Recurse: a new log POST may be enqueued while we're awaiting current
    // ones (rare, but possible if the customer keeps logging during shutdown).
    while (this.inflight.size > 0) {
      const snapshot = [...this.inflight];
      await Promise.allSettled(snapshot);
    }
  }

  private async sendWithRetries(entry: SdkLogEntry): Promise<void> {
    const maxAttempts = this.backoffMs.length + 1;
    let lastErr: unknown;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        await this.send(entry);
        return;
      } catch (err) {
        lastErr = err;
        if (!isRetryable(err)) throw err;
        if (attempt === maxAttempts) throw err;
        await sleep(jitter(this.backoffMs[attempt - 1]!));
      }
    }
    // Unreachable — the loop either returns on success or throws on the last
    // attempt. Kept to satisfy TS that `lastErr` is referenced.
    throw lastErr;
  }

  private async send(entry: SdkLogEntry): Promise<void> {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), this.timeoutMs);
    try {
      const headers: Record<string, string> = {
        "content-type": "application/json",
      };
      if (this.apiKey) headers["x-spaturzu-key"] = this.apiKey;
      const res = await fetch(`${this.baseURL}/v1/logs`, {
        method: "POST",
        headers,
        body: JSON.stringify(entry),
        signal: ac.signal,
      });
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new HttpError(
          res.status,
          `spaturzu: POST /v1/logs ${res.status}: ${text.slice(0, 200)}`,
        );
      }
    } finally {
      clearTimeout(timer);
    }
  }
}
