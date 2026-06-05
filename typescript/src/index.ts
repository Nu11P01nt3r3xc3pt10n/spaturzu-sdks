// Public surface of @spaturzu/sdk.
//
// Usage:
//   const spaturzu = new spaturzu({ baseURL, apiKey });
//   const openai = spaturzu.wrapOpenAI(new OpenAI());
//   const anthropic = spaturzu.wrapAnthropic(new Anthropic());
//
//   await spaturzu.run("researcher", async () => {
//     await openai.chat.completions.create({ ... });   // tagged: researcher
//     await spaturzu.run("writer", async () => {
//       await anthropic.messages.create({ ... });      // path: ["researcher","writer"]
//     });
//   });
//
//   // For short-lived processes (CLIs, lambdas):
//   await spaturzu.flush();
//
// Wrappers don't mutate the underlying client — they return Proxies. Use the
// returned reference; the original is untouched.

import {
  runInFrame,
  normalizeTags,
  getCurrentFrame,
  mergeTags,
  type TagInput,
} from "./context.js";
import { Logger, type SdkLogEntry } from "./logger.js";
import { wrapOpenAI as wrapOpenAIImpl } from "./openai.js";
import { wrapAnthropic as wrapAnthropicImpl } from "./anthropic.js";
import { wrapBedrock as wrapBedrockImpl } from "./bedrock.js";
import { wrapGemini as wrapGeminiImpl } from "./gemini.js";
import { wrapMistral as wrapMistralImpl } from "./mistral.js";
import { BudgetGuard, type BudgetWrapOptions, type BudgetOnBreach } from "./budget.js";
import type { FallbackTarget } from "./fallback.js";

export interface SpaturzuOptions {
  /** Gateway base URL. Falls back to `SPATURZU_BASE_URL` env, then the
   *  hosted gateway at `https://spaturzu-api.superchiu.org`. */
  baseURL?: string;
  /** Project API key (Day 6+). Sent as `x-spaturzu-key`. Optional in dev. */
  apiKey?: string;
  /** Per-attempt POST timeout. Defaults to 10s. */
  timeoutMs?: number;
  /** Override the retry backoff schedule. Length = number of retries after
   *  the initial send. Default: [1s, 2s, 4s, 8s, 16s]. Set `[]` to disable
   *  retries. Tests use a short array to keep wall time low. */
  backoffMs?: number[];
  /** Max in-flight log POSTs at once. Default 50; excess calls queue FIFO. */
  maxConcurrent?: number;
  /** Called when a log POST fails after all retries. Default: silent (the
   *  customer's app must not crash because the metering plane hiccupped). */
  onError?: (err: unknown) => void;
  /** Process-wide tags merged into every logged call. Use for dimensions
   *  that don't change inside the process: env, deployment, region, version.
   *  Per-run tags (passed to `run()`) override these on key conflict. */
  tags?: TagInput;
}

/** Optional second argument to `spaturzu.run()` for frame-scoped tags. */
export interface RunOptions {
  /** Tags applied to every LLM call inside this frame (and any nested
   *  `run()`s, unless they override the same key). Merged on top of the
   *  parent frame's tags + the spaturzu-level process tags. */
  tags?: TagInput;
}

/** Friendly input for {@link Spaturzu.report}. A subset of the wire
 *  `SdkLogEntry`: required `provider` + `model`, everything else optional.
 *  When called inside a `run()` frame, `runId` / `agentName` / `agentPath`
 *  and frame tags are inherited unless overridden here. */
export interface ReportCallInput {
  provider: string;
  model: string;
  /** Must be a valid UUID if set. */
  runId?: string;
  agentName?: string;
  agentPath?: string[];
  userId?: string;
  sessionId?: string;
  promptTokens?: number;
  completionTokens?: number;
  cachedInputTokens?: number;
  usageSource?: "provider" | "tiktoken";
  /** Defaults to 200. */
  status?: number;
  latencyMs?: number;
  tags?: TagInput;
  metadata?: Record<string, unknown>;
}

export class Spaturzu {
  private logger: Logger;
  private processTags: Record<string, string> | undefined;
  private baseURL: string;
  private apiKey: string | undefined;
  private onError: ((err: unknown) => void) | undefined;
  // Day 20: lazily-constructed BudgetGuard. Shared across all wraps for
  // this spaturzu instance — one policy cache + one SSE connection per
  // process is enough.
  private guard: BudgetGuard | null = null;

  constructor(opts: SpaturzuOptions = {}) {
    const baseURL =
      opts.baseURL ??
      process.env.SPATURZU_BASE_URL ??
      "https://spaturzu-api.superchiu.org";
    const apiKey = opts.apiKey ?? process.env.SPATURZU_API_KEY;
    this.baseURL = baseURL.replace(/\/+$/, "");
    this.apiKey = apiKey;
    this.onError = opts.onError;
    this.logger = new Logger(
      this.baseURL,
      apiKey,
      (err) => opts.onError?.(err),
      {
        timeoutMs: opts.timeoutMs,
        backoffMs: opts.backoffMs,
        maxConcurrent: opts.maxConcurrent,
      },
    );
    this.processTags = normalizeTags(opts.tags);
  }

  private ensureGuard(): BudgetGuard {
    if (!this.guard) {
      this.guard = new BudgetGuard({
        baseURL: this.baseURL,
        apiKey: this.apiKey,
        onError: (err) => this.onError?.(err),
      });
      this.guard.start();
    }
    return this.guard;
  }

  /** Process-wide tag accessor for the wrappers. Returned by reference, so
   *  callers must not mutate. */
  private getProcessTags = (): Record<string, string> | undefined =>
    this.processTags;

  /** Run `fn` inside a fresh agent frame. Nests automatically — a parent
   *  spaturzu.run() and a child spaturzu.run() share `runId` and the child's
   *  agent_path is the parent's plus the child name.
   *
   *  Two-arg form: `spaturzu.run("name", fn)`.
   *  Three-arg form: `spaturzu.run("name", { tags: { team: "search" } }, fn)`
   *  applies frame-scoped tags. */
  run<T>(agentName: string, fn: () => Promise<T>): Promise<T>;
  run<T>(
    agentName: string,
    opts: RunOptions,
    fn: () => Promise<T>,
  ): Promise<T>;
  run<T>(
    agentName: string,
    optsOrFn: RunOptions | (() => Promise<T>),
    fn?: () => Promise<T>,
  ): Promise<T> {
    if (typeof optsOrFn === "function") {
      return runInFrame(agentName, optsOrFn);
    }
    if (!fn) {
      throw new TypeError("spaturzu.run: fn is required when opts are supplied");
    }
    return runInFrame(agentName, fn, normalizeTags(optsOrFn.tags));
  }

  /** Report a completed LLM call without wrapping a provider client.
   *  Builds an SdkLogEntry from the caller's fields, inheriting `runId`,
   *  `agentName`, `agentPath`, and frame tags from the current `run()`
   *  frame when not explicitly provided. Fire-and-forget — failures are
   *  swallowed by the Logger (use `onError` to observe them). */
  report(call: ReportCallInput): void {
    const frame = getCurrentFrame();
    const tags = mergeTags(
      mergeTags(this.getProcessTags(), frame?.tags),
      normalizeTags(call.tags),
    );
    const entry: SdkLogEntry = {
      provider: call.provider,
      model: call.model,
      runId: call.runId ?? frame?.runId,
      parentRequestId: frame?.parentRequestId,
      agentName: call.agentName ?? frame?.agentName,
      agentPath: call.agentPath ?? frame?.agentPath,
      userId: call.userId,
      sessionId: call.sessionId,
      promptTokens: call.promptTokens,
      completionTokens: call.completionTokens,
      cachedInputTokens: call.cachedInputTokens,
      usageSource: call.usageSource,
      status: call.status ?? 200,
      latencyMs: call.latencyMs,
      ...(tags ? { tags } : {}),
      ...(call.metadata ? { metadata: call.metadata } : {}),
    };
    this.logger.log(entry);
  }

  /** Pre-call budget gate without wrapping a provider client. Lazily starts
   *  the shared BudgetGuard (one policy cache + one SSE connection per
   *  process) and runs `preCallCheck`: throws `BudgetExceededError` on a
   *  breached hard-cap policy, resolves otherwise. Fails open on a
   *  policy-fetch error — a metering-plane outage must not block calls. */
  checkBudget(
    agentName: string | null,
    opts?: { onBreach?: BudgetOnBreach },
  ): Promise<void> {
    return this.ensureGuard().preCallCheck(agentName, opts?.onBreach ?? "throw");
  }

  /** Wrap an OpenAI client. Returns a Proxy; the original is untouched.
   *
   *  Day 20: pass `{ budget: { hardCap: true } }` to enforce hard-cap
   *  budgets. The wrapped `create()` throws `BudgetExceededError` before
   *  hitting the provider when any applicable hard-cap budget is breached.
   *  `onBreach: 'warn'` logs to console.warn and lets the call through.
   *
   *  Day 24: pass `fallback: [...]` to declare a cross-provider failover
   *  chain. Each entry is `{ provider, client, model }`. On a retryable
   *  upstream error (429 / 5xx / connection-class), the wrap walks the
   *  chain and translates request/response when crossing providers. v1
   *  scope: non-streaming, no tools/response_format, text content only.
   *  Calls that use any of those features silently skip fallback. */
  wrapOpenAI<T extends { chat: { completions: { create: (params: any, options?: any) => Promise<any> } } }>(
    client: T,
    opts: { budget?: BudgetWrapOptions; fallback?: FallbackTarget[] } = {},
  ): T {
    const budget = opts.budget?.hardCap === true ? opts.budget : null;
    const guard = budget ? this.ensureGuard() : null;
    return wrapOpenAIImpl(
      client,
      this.logger,
      this.getProcessTags,
      guard,
      budget?.onBreach ?? "throw",
      opts.fallback ?? [],
    );
  }

  /** Wrap an Anthropic client. Returns a Proxy; the original is untouched.
   *  See `wrapOpenAI` for the `budget` and `fallback` options. */
  wrapAnthropic<T extends { messages: { create: (params: any, options?: any) => Promise<any> } }>(
    client: T,
    opts: { budget?: BudgetWrapOptions; fallback?: FallbackTarget[] } = {},
  ): T {
    const budget = opts.budget?.hardCap === true ? opts.budget : null;
    const guard = budget ? this.ensureGuard() : null;
    return wrapAnthropicImpl(
      client,
      this.logger,
      this.getProcessTags,
      guard,
      budget?.onBreach ?? "throw",
      opts.fallback ?? [],
    );
  }

  /** Wrap a Bedrock Runtime client. Returns a Proxy; the original is untouched.
   *  Pass the aggregated `BedrockRuntime` client (it has the named methods);
   *  the low-level `BedrockRuntimeClient` has no `converse`/`converseStream`.
   *  Targets the named-method form (`client.converse(input)`,
   *  `client.converseStream(input)`) only — Command-pattern users
   *  (`client.send(new ConverseCommand(...))`) are NOT instrumented in v1.
   *
   *  See `wrapOpenAI` for the `budget` and `fallback` options. */
  wrapBedrock<T extends { converse: (p: any, o?: any) => Promise<any>; converseStream: (p: any, o?: any) => Promise<any> }>(
    client: T,
    opts: { budget?: BudgetWrapOptions; fallback?: FallbackTarget[] } = {},
  ): T {
    const budget = opts.budget?.hardCap === true ? opts.budget : null;
    const guard = budget ? this.ensureGuard() : null;
    return wrapBedrockImpl(
      client,
      this.logger,
      this.getProcessTags,
      guard,
      budget?.onBreach ?? "throw",
      opts.fallback ?? [],
    );
  }

  /** Wrap a Google GenAI client. Intercepts `client.models.generateContent`
   *  and `client.models.generateContentStream`.
   *
   *  See `wrapOpenAI` for the `budget` and `fallback` options. */
  wrapGemini<T extends { models: { generateContent: (p: any, o?: any) => Promise<any>; generateContentStream: (p: any, o?: any) => Promise<any> } }>(
    client: T,
    opts: { budget?: BudgetWrapOptions; fallback?: FallbackTarget[] } = {},
  ): T {
    const budget = opts.budget?.hardCap === true ? opts.budget : null;
    const guard = budget ? this.ensureGuard() : null;
    return wrapGeminiImpl(
      client,
      this.logger,
      this.getProcessTags,
      guard,
      budget?.onBreach ?? "throw",
      opts.fallback ?? [],
    );
  }

  /** Wrap a Mistral client. Intercepts `client.chat.complete` and
   *  `client.chat.stream` (note: separate methods, not a stream:true flag).
   *
   *  See `wrapOpenAI` for the `budget` and `fallback` options. */
  wrapMistral<T extends { chat: { complete: (p: any, o?: any) => Promise<any>; stream: (p: any, o?: any) => Promise<any> } }>(
    client: T,
    opts: { budget?: BudgetWrapOptions; fallback?: FallbackTarget[] } = {},
  ): T {
    const budget = opts.budget?.hardCap === true ? opts.budget : null;
    const guard = budget ? this.ensureGuard() : null;
    return wrapMistralImpl(
      client,
      this.logger,
      this.getProcessTags,
      guard,
      budget?.onBreach ?? "throw",
      opts.fallback ?? [],
    );
  }

  /** Await all in-flight log POSTs. Call before exiting short-lived
   *  processes (CLIs, serverless handlers, CI scripts) so metadata reaches
   *  the gateway. Long-running servers can ignore this. */
  flush(): Promise<void> {
    return this.logger.flush();
  }

  /** Day 20: tear down the BudgetGuard's SSE connection + polling timer.
   *  Call before process exit on short-lived scripts so the open SSE
   *  socket doesn't keep the event loop alive. Long-running servers can
   *  ignore this — the connection cleans up on process termination. */
  shutdown(): Promise<void> {
    this.guard?.stop();
    return this.logger.flush();
  }
}

export { getCurrentFrame } from "./context.js";
export type { RunFrame, TagInput, AgentBinding } from "./context.js";
export type { SdkLogEntry } from "./logger.js";
export {
  BudgetExceededError,
  type BudgetWrapOptions,
  type BudgetOnBreach,
  type AgentPolicy,
  type PolicyLimit,
} from "./budget.js";
export type { FallbackTarget } from "./fallback.js";
export { configure, run, flush, shutdown, report, checkBudget } from "./default.js";
export type { SpaturzuClientOptions } from "./providers/openai.js";
