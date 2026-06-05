import { uuidv7 } from "./uuid.js";
import {
  getCurrentFrame,
  mergeTags,
  setLastRequestId,
  runInFrame,
  normalizeTags,
  type AgentBinding,
  type TagInput,
  type RunFrame,
  type WithAgent,
} from "./context.js";
import type { Logger, SdkLogEntry } from "./logger.js";
import { estimateTokens } from "./tiktoken.js";
import type { BudgetGuard, BudgetOnBreach } from "./budget.js";
import {
  isRetryableUpstreamError,
  tryFallback,
  type FallbackTarget,
} from "./fallback.js";

// Minimal structural type for the openai SDK's chat.completions.create. We
// don't import `openai` directly — it's a peer dep, so users may have any
// version. We rely on duck-typing against the runtime shape we actually use.
interface CreateFn {
  (params: any, options?: any): Promise<any>;
}

type StreamingUsage = {
  prompt_tokens?: number;
  completion_tokens?: number;
  prompt_tokens_details?: { cached_tokens?: number };
};

type ProcessTagsGetter = () => Record<string, string> | undefined;

function buildBaseEntry(opts: {
  requestId: string;
  model: string;
  startedAt: number;
  getProcessTags: ProcessTagsGetter;
  frame: RunFrame | undefined;
}): SdkLogEntry {
  const frame = opts.frame;
  // Process tags are the most general, frame tags are scoped. Frame wins
  // on conflict — the inner scope is the more specific intent.
  const tags = mergeTags(opts.getProcessTags(), frame?.tags);
  return {
    id: opts.requestId,
    provider: "openai",
    model: opts.model,
    runId: frame?.runId,
    parentRequestId: frame?.parentRequestId,
    agentName: frame?.agentName,
    agentPath: frame?.agentPath,
    status: 200,
    ...(tags ? { tags } : {}),
  };
}

function buildSuccess(
  opts: {
    requestId: string;
    model: string;
    startedAt: number;
    getProcessTags: ProcessTagsGetter;
    frame: RunFrame | undefined;
  },
  usage: StreamingUsage | null | undefined,
): SdkLogEntry {
  const base = buildBaseEntry(opts);
  return {
    ...base,
    status: 200,
    latencyMs: Date.now() - opts.startedAt,
    promptTokens: usage?.prompt_tokens,
    completionTokens: usage?.completion_tokens,
    cachedInputTokens: usage?.prompt_tokens_details?.cached_tokens,
    usageSource: "provider",
  };
}

function buildError(
  opts: {
    requestId: string;
    model: string;
    startedAt: number;
    getProcessTags: ProcessTagsGetter;
    frame: RunFrame | undefined;
  },
  err: unknown,
): SdkLogEntry {
  const base = buildBaseEntry(opts);
  // OpenAI's APIError carries `.status`. Fall back to 500 so the column is
  // never null (it's NOT NULL on the wire schema below 599).
  const status =
    typeof (err as { status?: unknown })?.status === "number"
      ? (err as { status: number }).status
      : 500;
  return {
    ...base,
    status: Math.min(599, Math.max(100, status)),
    latencyMs: Date.now() - opts.startedAt,
  };
}

// Day 9 fallback: pull a plain-text rendering of the prompt so we can
// tokenize it when the upstream omits `usage`. Multimodal `content` arrays
// keep only the text parts — image inputs would need provider-specific
// vision pricing and aren't relevant to this estimate.
function extractPromptText(params: any): string {
  const msgs = params?.messages;
  if (!Array.isArray(msgs)) return "";
  const parts: string[] = [];
  for (const m of msgs) {
    const role = typeof m?.role === "string" ? m.role : "";
    const c = m?.content;
    if (typeof c === "string") {
      if (role) parts.push(role);
      parts.push(c);
    } else if (Array.isArray(c)) {
      for (const part of c) {
        if (part?.type === "text" && typeof part.text === "string") {
          if (role) parts.push(role);
          parts.push(part.text);
        }
      }
    }
  }
  return parts.join("\n");
}

// Wrap an OpenAI streaming response. We pass each chunk straight through to
// the consumer and tee it into a usage observer; the final chunk (with
// stream_options.include_usage) carries the totals we need.
//
// Day 9: when the upstream is OpenAI-compatible but doesn't honour
// `include_usage` (Together/Groq/Anyscale/vLLM-self-hosted), no usage is
// ever emitted. We accumulate the assistant's streamed text so the finally
// branch can fall back to tiktoken instead of logging zero tokens.
//
// We intentionally return an AsyncIterable, not the SDK's `Stream` class —
// `for await` works identically. `tee()` / `toReadableStream()` are not
// preserved (rare in agent code; revisit if a customer asks).
async function* observeStream(
  upstream: AsyncIterable<unknown>,
  ctx: {
    requestId: string;
    model: string;
    startedAt: number;
    logger: Logger;
    promptText: string;
    getProcessTags: ProcessTagsGetter;
    frame: RunFrame | undefined;
  },
): AsyncGenerator<unknown, void, unknown> {
  let usage: StreamingUsage | null = null;
  let completionText = "";
  let threw: unknown = null;
  try {
    for await (const chunk of upstream) {
      const c = chunk as
        | {
            usage?: StreamingUsage;
            choices?: Array<{ delta?: { content?: unknown } }>;
          }
        | undefined;
      if (c?.usage) usage = c.usage;
      const choices = c?.choices;
      if (Array.isArray(choices)) {
        for (const ch of choices) {
          const d = ch?.delta?.content;
          if (typeof d === "string") completionText += d;
        }
      }
      yield chunk;
    }
  } catch (err) {
    threw = err;
    throw err;
  } finally {
    if (threw) {
      ctx.logger.log(buildError(ctx, threw));
    } else if (usage) {
      ctx.logger.log(buildSuccess(ctx, usage));
    } else {
      // No usage emitted — tokenize locally. We `await` here so the log
      // entry carries token counts. The cost is paid on the consumer's
      // last-chunk wait (~5 ms), which is invisible against a streaming
      // response that already took seconds. If tiktoken isn't installed,
      // estimateTokens returns null and we fall back to logging without
      // tokens (same as pre-Day-9 behaviour).
      const promptTokens = await estimateTokens(ctx.model, ctx.promptText);
      const completionTokens = await estimateTokens(ctx.model, completionText);
      if (promptTokens != null && completionTokens != null) {
        ctx.logger.log({
          ...buildBaseEntry(ctx),
          status: 200,
          latencyMs: Date.now() - ctx.startedAt,
          promptTokens,
          completionTokens,
          usageSource: "tiktoken",
        });
      } else {
        ctx.logger.log(buildSuccess(ctx, null));
      }
    }
  }
}

export function wrapOpenAI<T extends { chat: { completions: { create: CreateFn } } }>(
  client: T,
  logger: Logger,
  getProcessTags: ProcessTagsGetter = () => undefined,
  guard: BudgetGuard | null = null,
  onBreach: BudgetOnBreach = "throw",
  // Day 24: cross-provider fallback chain. Walked in order on any
  // retryable upstream error (429 / 5xx / connection-class). v1 quietly
  // skips fallback for streaming calls and for calls using features the
  // translator can't safely cross (tools, response_format, non-text
  // content) — original error bubbles in those cases.
  fallbacks: FallbackTarget[] = [],
  agentBinding: AgentBinding | null = null,
): WithAgent<T> {
  const realCreate = client.chat.completions.create.bind(client.chat.completions);

  const coreCreate: CreateFn = async (params, options) => {
    // Day 20 pre-call gate. Runs before the upstream provider is hit — a
    // breached hard-cap throws BudgetExceededError here, never reaching
    // realCreate, so no cost is incurred. No log entry is written for the
    // refused call (it would just show as a zero-token, zero-cost ghost
    // row in the trace view — noise). Note: the hard-cap also gates
    // the fallback path below — we deliberately never bypass enforcement.
    if (guard) {
      const frame = getCurrentFrame();
      await guard.preCallCheck(frame?.agentName ?? null, onBreach);
    }
    const requestId = uuidv7();
    setLastRequestId(requestId);
    const startedAt = Date.now();
    const model: string = params?.model ?? "unknown";
    const isStreaming = params?.stream === true;
    const ctxBase = { requestId, model, startedAt, getProcessTags, frame: getCurrentFrame() };

    if (isStreaming) {
      // Auto-inject include_usage so the upstream emits a final usage chunk.
      // Customer-set stream_options wins for any other field (e.g. they may
      // set their own include_usage=false; we override that — without it we'd
      // have no usage data and the whole point is attribution).
      const enrichedParams = {
        ...params,
        stream_options: {
          ...(params?.stream_options ?? {}),
          include_usage: true,
        },
      };
      // Day 9: capture prompt text once, before the upstream call. Cheap
      // (string concat), and we need it whether the upstream honours
      // include_usage or not — we only know which after streaming completes.
      const promptText = extractPromptText(params);
      let upstream: AsyncIterable<unknown>;
      try {
        upstream = await realCreate(enrichedParams, options);
      } catch (err) {
        logger.log(buildError(ctxBase, err));
        throw err;
      }
      return observeStream(upstream, {
        ...ctxBase,
        logger,
        promptText,
      });
    }

    let primaryErr: unknown;
    try {
      const result = await realCreate(params, options);
      logger.log(buildSuccess(ctxBase, result?.usage));
      return result;
    } catch (err) {
      primaryErr = err;
      logger.log(buildError(ctxBase, err));
    }

    // ─── Day 24 fallback chain ────────────────────────────────────────────
    // Reached only on primary failure. We've already logged the primary's
    // error row above so cost attribution stays accurate even if every
    // fallback also fails.
    if (
      fallbacks.length === 0 ||
      !isRetryableUpstreamError(primaryErr) ||
      (options as { signal?: AbortSignal } | undefined)?.signal?.aborted
    ) {
      throw primaryErr;
    }

    let lastErr: unknown = primaryErr;
    for (const target of fallbacks) {
      const outcome = await tryFallback({
        primaryShape: "openai",
        originalParams: params,
        target,
        options,
      });

      if (outcome.kind === "unsupported") {
        // Translator refused (e.g. streaming or tools). Bubble the original
        // primary error — there's nothing useful we can try.
        throw primaryErr;
      }

      if (outcome.kind === "success") {
        // Log the successful fallback attempt under its own request_id.
        // Same runId / agentName / agentPath (the frame is unchanged),
        // plus a `via: 'fallback'` tag so the trace view can highlight
        // which row was the recovery.
        const fbRequestId = uuidv7();
        setLastRequestId(fbRequestId);
        const frame = getCurrentFrame();
        const fbTags = mergeTags(getProcessTags(), {
          ...(frame?.tags ?? {}),
          via: "fallback",
        });
        const fbEntry: SdkLogEntry = {
          id: fbRequestId,
          provider: target.provider,
          model: target.model,
          runId: frame?.runId,
          parentRequestId: frame?.parentRequestId,
          agentName: frame?.agentName,
          agentPath: frame?.agentPath,
          status: 200,
          latencyMs: outcome.latencyMs,
          promptTokens: outcome.usage.promptTokens,
          completionTokens: outcome.usage.completionTokens,
          cachedInputTokens: outcome.usage.cachedInputTokens,
          usageSource: "provider",
          ...(fbTags ? { tags: fbTags } : {}),
        };
        logger.log(fbEntry);
        return outcome.response;
      }

      // outcome.kind === "error" — log the failed attempt and continue down
      // the chain only if it was retryable too.
      const fbRequestId = uuidv7();
      const frame = getCurrentFrame();
      const fbTags = mergeTags(getProcessTags(), {
        ...(frame?.tags ?? {}),
        via: "fallback",
      });
      logger.log({
        id: fbRequestId,
        provider: target.provider,
        model: target.model,
        runId: frame?.runId,
        parentRequestId: frame?.parentRequestId,
        agentName: frame?.agentName,
        agentPath: frame?.agentPath,
        status: Math.min(599, Math.max(100, outcome.status ?? 500)),
        latencyMs: outcome.latencyMs,
        ...(fbTags ? { tags: fbTags } : {}),
      });
      lastErr = outcome.error;
      if (!isRetryableUpstreamError(outcome.error)) {
        // Auth / bad-request / context-overflow class on the fallback —
        // the next target won't fix that either, surface immediately.
        throw outcome.error;
      }
      if ((options as { signal?: AbortSignal } | undefined)?.signal?.aborted) {
        throw outcome.error;
      }
    }

    throw lastErr;
  };

  // When bound to an agent (via .withAgent()), route each terminal call
  // through runInFrame so the frame is captured at call-time — critical for
  // streaming, whose consumer iterates outside the frame.
  const bindAgent = (fn: CreateFn): CreateFn =>
    agentBinding
      ? (params, options) =>
          runInFrame(agentBinding.agent, () => fn(params, options), agentBinding.tags)
      : fn;
  const wrappedCreate = bindAgent(coreCreate);

  // Use a chain of Proxies so we never mutate the customer's client. Reads
  // for unrelated paths fall through via Reflect.get; only chat.completions.
  // create is replaced.
  const completionsProxy = new Proxy(client.chat.completions, {
    get(target, prop, receiver) {
      if (prop === "create") return wrappedCreate;
      return Reflect.get(target, prop, receiver);
    },
  });

  const chatProxy = new Proxy(client.chat, {
    get(target, prop, receiver) {
      if (prop === "completions") return completionsProxy;
      return Reflect.get(target, prop, receiver);
    },
  });

  return new Proxy(client, {
    get(target, prop, receiver) {
      if (prop === "withAgent") {
        return (agent: string, o?: { tags?: TagInput }) =>
          wrapOpenAI(client, logger, getProcessTags, guard, onBreach, fallbacks, {
            agent,
            tags: normalizeTags(o?.tags),
          });
      }
      if (prop === "chat") return chatProxy;
      return Reflect.get(target, prop, receiver);
    },
  }) as WithAgent<T>;
}
