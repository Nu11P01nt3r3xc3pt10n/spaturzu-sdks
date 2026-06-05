// AWS Bedrock Converse wrap. Mirrors today's openai.ts / anthropic.ts:
// log-entry builders, prompt-text extractor for the tiktoken fallback,
// async-iterable observer for the stream, and a wrap function that
// returns a Proxy chain over the customer's client.
//
// Limitations (spec §2.1):
//   • Converse + ConverseStream only. InvokeModel is v2.
//   • Command pattern (`client.send(new ConverseCommand(...))`) NOT
//     instrumented. Customer's code must use the named-method form
//     `client.converse(input)` / `client.converseStream(input)`.

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

interface ConverseFn {
  (params: any, options?: any): Promise<any>;
}

type BedrockUsage = {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  cacheReadInputTokenCount?: number;
  cacheWriteInputTokenCount?: number;
};

type ProcessTagsGetter = () => Record<string, string> | undefined;

// ─── log-entry builders ────────────────────────────────────────────────────

function buildBaseEntry(opts: {
  requestId: string;
  modelId: string;
  startedAt: number;
  getProcessTags: ProcessTagsGetter;
  frame: RunFrame | undefined;
}): SdkLogEntry {
  const frame = opts.frame;
  const tags = mergeTags(opts.getProcessTags(), frame?.tags);
  return {
    id: opts.requestId,
    provider: "bedrock",
    model: opts.modelId,
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
    modelId: string;
    startedAt: number;
    getProcessTags: ProcessTagsGetter;
    frame: RunFrame | undefined;
  },
  usage: BedrockUsage | null | undefined,
): SdkLogEntry {
  return {
    ...buildBaseEntry(opts),
    status: 200,
    latencyMs: Date.now() - opts.startedAt,
    promptTokens: usage?.inputTokens,
    completionTokens: usage?.outputTokens,
    cachedInputTokens: usage?.cacheReadInputTokenCount,
    usageSource: "provider",
  };
}

function buildError(
  opts: {
    requestId: string;
    modelId: string;
    startedAt: number;
    getProcessTags: ProcessTagsGetter;
    frame: RunFrame | undefined;
  },
  err: unknown,
): SdkLogEntry {
  // AWS SDK v3 errors carry status under `$metadata.httpStatusCode`.
  const meta = (err as { $metadata?: { httpStatusCode?: unknown } })?.$metadata;
  const raw = typeof meta?.httpStatusCode === "number" ? meta.httpStatusCode : null;
  const status = raw ?? 500;
  return {
    ...buildBaseEntry(opts),
    status: Math.min(599, Math.max(100, status)),
    latencyMs: Date.now() - opts.startedAt,
  };
}

// ─── prompt-text extraction (tiktoken fallback) ────────────────────────────

function extractPromptText(params: any): string {
  // Bedrock requires content as an array of content blocks; we keep only
  // the text portions. tiktoken's cl100k_base is a rough approximation
  // for non-OpenAI tokenizers (Bedrock Llama uses SentencePiece, etc.) —
  // the row will be marked usageSource='tiktoken' so consumers can
  // discount accordingly.
  const parts: string[] = [];
  const sys = params?.system;
  if (Array.isArray(sys)) {
    for (const b of sys) {
      if (typeof b?.text === "string") parts.push(b.text);
    }
  }
  const msgs = params?.messages;
  if (Array.isArray(msgs)) {
    for (const m of msgs) {
      const role = typeof m?.role === "string" ? m.role : "";
      if (Array.isArray(m?.content)) {
        for (const b of m.content) {
          if (typeof b?.text === "string") {
            if (role) parts.push(role);
            parts.push(b.text);
          }
        }
      }
    }
  }
  return parts.join("\n");
}

// ─── streaming observation ─────────────────────────────────────────────────

async function* observeStream(
  upstream: AsyncIterable<unknown>,
  ctx: {
    requestId: string;
    modelId: string;
    startedAt: number;
    logger: Logger;
    promptText: string;
    getProcessTags: ProcessTagsGetter;
    frame: RunFrame | undefined;
  },
): AsyncGenerator<unknown, void, unknown> {
  let usage: BedrockUsage | null = null;
  let completionText = "";
  let threw: unknown = null;
  try {
    for await (const event of upstream) {
      // Each event is one of: messageStart, contentBlockStart,
      // contentBlockDelta, contentBlockStop, messageStop, metadata.
      // We only care about contentBlockDelta (text accumulation for
      // the tiktoken fallback) and metadata (authoritative usage).
      const ev = event as {
        contentBlockDelta?: { delta?: { text?: unknown } };
        metadata?: { usage?: BedrockUsage; metrics?: { latencyMs?: number } };
      };
      const dt = ev.contentBlockDelta?.delta?.text;
      if (typeof dt === "string") completionText += dt;
      if (ev.metadata?.usage) usage = ev.metadata.usage;
      yield event;
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
      // No metadata event — tokenize locally. Same fallback pattern as
      // openai.ts; tiktoken is a rough approximation for Bedrock's
      // SentencePiece-based models but better than logging zero.
      const promptTokens = await estimateTokens(ctx.modelId, ctx.promptText);
      const completionTokens = await estimateTokens(ctx.modelId, completionText);
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

// ─── wrap ──────────────────────────────────────────────────────────────────

export function wrapBedrock<
  T extends { converse: ConverseFn; converseStream: ConverseFn },
>(
  client: T,
  logger: Logger,
  getProcessTags: ProcessTagsGetter = () => undefined,
  guard: BudgetGuard | null = null,
  onBreach: BudgetOnBreach = "throw",
  fallbacks: FallbackTarget[] = [],
  agentBinding: AgentBinding | null = null,
): WithAgent<T> {
  const realConverse = client.converse.bind(client);
  const realConverseStream = client.converseStream.bind(client);

  async function gateBudget(): Promise<void> {
    if (guard) {
      const frame = getCurrentFrame();
      await guard.preCallCheck(frame?.agentName ?? null, onBreach);
    }
  }

  const coreConverse: ConverseFn = async (params, options) => {
    await gateBudget();
    const requestId = uuidv7();
    setLastRequestId(requestId);
    const startedAt = Date.now();
    const modelId: string = params?.modelId ?? "unknown";
    const ctxBase = { requestId, modelId, startedAt, getProcessTags, frame: getCurrentFrame() };

    let primaryErr: unknown;
    try {
      const result = await realConverse(params, options);
      logger.log(buildSuccess(ctxBase, result?.usage));
      return result;
    } catch (err) {
      primaryErr = err;
      logger.log(buildError(ctxBase, err));
    }

    // Fallback chain (mirrors openai.ts and anthropic.ts).
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
        primaryShape: "bedrock",
        originalParams: params,
        target,
        options,
      });
      if (outcome.kind === "unsupported") throw primaryErr;
      if (outcome.kind === "success") {
        const fbRequestId = uuidv7();
        setLastRequestId(fbRequestId);
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
          status: 200,
          latencyMs: outcome.latencyMs,
          promptTokens: outcome.usage.promptTokens,
          completionTokens: outcome.usage.completionTokens,
          cachedInputTokens: outcome.usage.cachedInputTokens,
          usageSource: "provider",
          ...(fbTags ? { tags: fbTags } : {}),
        });
        return outcome.response;
      }
      // outcome.kind === "error"
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
      if (!isRetryableUpstreamError(outcome.error)) throw outcome.error;
      if ((options as { signal?: AbortSignal } | undefined)?.signal?.aborted)
        throw outcome.error;
    }
    throw lastErr;
  };

  const coreConverseStream: ConverseFn = async (params, options) => {
    await gateBudget();
    const requestId = uuidv7();
    setLastRequestId(requestId);
    const startedAt = Date.now();
    const modelId: string = params?.modelId ?? "unknown";
    const ctxBase = { requestId, modelId, startedAt, getProcessTags, frame: getCurrentFrame() };
    const promptText = extractPromptText(params);

    let raw: { stream: AsyncIterable<unknown> };
    try {
      raw = await realConverseStream(params, options);
    } catch (err) {
      logger.log(buildError(ctxBase, err));
      throw err;
    }
    // Return a NEW object that mirrors the boto3-shaped { stream } envelope.
    // The customer iterates `result.stream` — we substitute our observer.
    return {
      ...raw,
      stream: observeStream(raw.stream, {
        ...ctxBase,
        logger,
        promptText,
      }),
    };
  };

  // When bound to an agent (via .withAgent()), route each terminal call
  // through runInFrame so the frame is captured at call-time — critical for
  // streaming, whose consumer iterates outside the frame.
  type TermFn = (p: any, o?: any) => Promise<any>;
  const bindAgent = (fn: TermFn): TermFn =>
    agentBinding
      ? (p, o) => runInFrame(agentBinding.agent, () => fn(p, o), agentBinding.tags)
      : fn;
  const wrappedConverse = bindAgent(coreConverse);
  const wrappedConverseStream = bindAgent(coreConverseStream);

  // Proxy chain — substitute only `converse` and `converseStream`; every
  // other method / property on the BedrockRuntimeClient passes through.
  return new Proxy(client, {
    get(target, prop, receiver) {
      if (prop === "withAgent") {
        return (agent: string, o?: { tags?: TagInput }) =>
          wrapBedrock(client, logger, getProcessTags, guard, onBreach, fallbacks, {
            agent,
            tags: normalizeTags(o?.tags),
          });
      }
      if (prop === "converse") return wrappedConverse;
      if (prop === "converseStream") return wrappedConverseStream;
      return Reflect.get(target, prop, receiver);
    },
  }) as WithAgent<T>;
}
