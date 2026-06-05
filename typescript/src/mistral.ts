// Mistral (@mistralai/mistralai) wrap. Mirrors today's openai.ts.
// chat.complete and chat.stream are SEPARATE methods (not a stream:true
// flag), so the wrap intercepts both.
//
// Stream chunks are shaped { data: { choices: [...], usage } } — note
// the data envelope. The observer reads chunk.data.usage, not chunk.usage.

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

interface MistralFn {
  (params: any, options?: any): Promise<any>;
}

type MistralUsage = {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
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
  const tags = mergeTags(opts.getProcessTags(), frame?.tags);
  return {
    id: opts.requestId,
    provider: "mistral",
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
  opts: { requestId: string; model: string; startedAt: number; getProcessTags: ProcessTagsGetter; frame: RunFrame | undefined },
  usage: MistralUsage | null | undefined,
): SdkLogEntry {
  return {
    ...buildBaseEntry(opts),
    status: 200,
    latencyMs: Date.now() - opts.startedAt,
    promptTokens: usage?.prompt_tokens,
    completionTokens: usage?.completion_tokens,
    usageSource: "provider",
  };
}

function buildError(
  opts: { requestId: string; model: string; startedAt: number; getProcessTags: ProcessTagsGetter; frame: RunFrame | undefined },
  err: unknown,
): SdkLogEntry {
  // Mistral SDK errors carry `.statusCode` or `.status` depending on the
  // error class.
  const raw =
    typeof (err as { statusCode?: unknown }).statusCode === "number"
      ? (err as { statusCode: number }).statusCode
      : typeof (err as { status?: unknown }).status === "number"
        ? (err as { status: number }).status
        : 500;
  return {
    ...buildBaseEntry(opts),
    status: Math.min(599, Math.max(100, raw)),
    latencyMs: Date.now() - opts.startedAt,
  };
}

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
    }
  }
  return parts.join("\n");
}

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
  let usage: MistralUsage | null = null;
  let completionText = "";
  let threw: unknown = null;
  try {
    for await (const chunk of upstream) {
      // Mistral chunks: { data: { choices: [{ delta: { content } }], usage } }
      const c = chunk as {
        data?: {
          choices?: Array<{ delta?: { content?: unknown } }>;
          usage?: MistralUsage;
        };
      };
      if (c.data?.usage) usage = c.data.usage;
      const choices = c.data?.choices;
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

export function wrapMistral<
  T extends { chat: { complete: MistralFn; stream: MistralFn } },
>(
  client: T,
  logger: Logger,
  getProcessTags: ProcessTagsGetter = () => undefined,
  guard: BudgetGuard | null = null,
  onBreach: BudgetOnBreach = "throw",
  fallbacks: FallbackTarget[] = [],
  agentBinding: AgentBinding | null = null,
): WithAgent<T> {
  const realComplete = client.chat.complete.bind(client.chat);
  const realStream = client.chat.stream.bind(client.chat);

  async function gateBudget(): Promise<void> {
    if (guard) {
      const frame = getCurrentFrame();
      await guard.preCallCheck(frame?.agentName ?? null, onBreach);
    }
  }

  const coreComplete: MistralFn = async (params, options) => {
    await gateBudget();
    const requestId = uuidv7();
    setLastRequestId(requestId);
    const startedAt = Date.now();
    const model: string = params?.model ?? "unknown";
    const ctxBase = { requestId, model, startedAt, getProcessTags, frame: getCurrentFrame() };

    let primaryErr: unknown;
    try {
      const result = await realComplete(params, options);
      logger.log(buildSuccess(ctxBase, result?.usage));
      return result;
    } catch (err) {
      primaryErr = err;
      logger.log(buildError(ctxBase, err));
    }

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
        primaryShape: "mistral",
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

  const coreStream: MistralFn = async (params, options) => {
    await gateBudget();
    const requestId = uuidv7();
    setLastRequestId(requestId);
    const startedAt = Date.now();
    const model: string = params?.model ?? "unknown";
    const ctxBase = { requestId, model, startedAt, getProcessTags, frame: getCurrentFrame() };
    const promptText = extractPromptText(params);

    let upstream: AsyncIterable<unknown>;
    try {
      upstream = (await realStream(params, options)) as AsyncIterable<unknown>;
    } catch (err) {
      logger.log(buildError(ctxBase, err));
      throw err;
    }
    return observeStream(upstream, { ...ctxBase, logger, promptText });
  };

  // When bound to an agent (via .withAgent()), route each terminal call
  // through runInFrame so the frame is captured at call-time — critical for
  // streaming, whose consumer iterates outside the frame.
  type TermFn = (p: any, o?: any) => Promise<any>;
  const bindAgent = (fn: TermFn): TermFn =>
    agentBinding
      ? (p, o) => runInFrame(agentBinding.agent, () => fn(p, o), agentBinding.tags)
      : fn;
  const wrappedComplete = bindAgent(coreComplete);
  const wrappedStream = bindAgent(coreStream);

  const chatProxy = new Proxy(client.chat, {
    get(target, prop, receiver) {
      if (prop === "complete") return wrappedComplete;
      if (prop === "stream") return wrappedStream;
      return Reflect.get(target, prop, receiver);
    },
  });
  return new Proxy(client, {
    get(target, prop, receiver) {
      if (prop === "withAgent") {
        return (agent: string, o?: { tags?: TagInput }) =>
          wrapMistral(client, logger, getProcessTags, guard, onBreach, fallbacks, {
            agent,
            tags: normalizeTags(o?.tags),
          });
      }
      if (prop === "chat") return chatProxy;
      return Reflect.get(target, prop, receiver);
    },
  }) as WithAgent<T>;
}
