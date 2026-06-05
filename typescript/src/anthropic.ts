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
import type { BudgetGuard, BudgetOnBreach } from "./budget.js";
import type { Logger, SdkLogEntry } from "./logger.js";
import { estimateTokens } from "./tiktoken.js";
import {
  isRetryableUpstreamError,
  tryFallback,
  type FallbackTarget,
} from "./fallback.js";

interface CreateFn {
  (params: any, options?: any): Promise<any>;
}

type AnthropicUsage = {
  input_tokens?: number;
  output_tokens?: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
};

type ProcessTagsGetter = () => Record<string, string> | undefined;

function baseEntry(opts: {
  requestId: string;
  model: string;
  startedAt: number;
  getProcessTags: ProcessTagsGetter;
  frame: RunFrame | undefined;
}): SdkLogEntry {
  const frame = opts.frame;
  // Process tags ⨁ frame tags, frame wins on conflict (more specific).
  const tags = mergeTags(opts.getProcessTags(), frame?.tags);
  return {
    id: opts.requestId,
    provider: "anthropic",
    model: opts.model,
    runId: frame?.runId,
    parentRequestId: frame?.parentRequestId,
    agentName: frame?.agentName,
    agentPath: frame?.agentPath,
    status: 200,
    ...(tags ? { tags } : {}),
  };
}

function success(
  opts: {
    requestId: string;
    model: string;
    startedAt: number;
    getProcessTags: ProcessTagsGetter;
    frame: RunFrame | undefined;
  },
  usage: AnthropicUsage | null | undefined,
): SdkLogEntry {
  // We map cache_read_input_tokens (the portion served from the prompt cache
  // on this call) to cachedInputTokens. cache_creation_input_tokens is the
  // *write* side and isn't a savings vs the uncached rate — leave it out for
  // now; revisit if/when we surface "cache write cost" separately.
  const cached = usage?.cache_read_input_tokens;
  return {
    ...baseEntry(opts),
    status: 200,
    latencyMs: Date.now() - opts.startedAt,
    promptTokens: usage?.input_tokens,
    completionTokens: usage?.output_tokens,
    cachedInputTokens: cached,
    usageSource: "provider",
  };
}

function failure(
  opts: {
    requestId: string;
    model: string;
    startedAt: number;
    getProcessTags: ProcessTagsGetter;
    frame: RunFrame | undefined;
  },
  err: unknown,
): SdkLogEntry {
  const status =
    typeof (err as { status?: unknown })?.status === "number"
      ? (err as { status: number }).status
      : 500;
  return {
    ...baseEntry(opts),
    status: Math.min(599, Math.max(100, status)),
    latencyMs: Date.now() - opts.startedAt,
  };
}

// Day 9 fallback: pull a plain-text rendering of the prompt for tiktoken
// estimation when the upstream omits usage on the stream. Anthropic's
// `messages.create` accepts both string content and content-block arrays;
// we keep only the text blocks.
function extractPromptText(params: any): string {
  const parts: string[] = [];
  const sys = params?.system;
  if (typeof sys === "string") parts.push(sys);
  else if (Array.isArray(sys)) {
    for (const b of sys) {
      if (b?.type === "text" && typeof b.text === "string") parts.push(b.text);
    }
  }
  const msgs = params?.messages;
  if (Array.isArray(msgs)) {
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
  }
  return parts.join("\n");
}

// Anthropic's stream emits typed events. Input usage arrives in `message_start`
// (input_tokens, cache_*); output_tokens accumulates across `message_delta`
// events. We track the last seen value of each so the final log carries
// totals.
//
// Day 9: also accumulate the assistant's streamed text from
// `content_block_delta` events. The native SDK reliably emits usage, but
// proxied/Bedrock-style endpoints may not — this lets us tokenize as a
// fallback without changing the happy path.
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
  let inputTokens: number | undefined;
  let outputTokens: number | undefined;
  let cacheRead: number | undefined;
  let completionText = "";
  let threw: unknown = null;
  try {
    for await (const event of upstream) {
      const ev = event as
        | {
            type?: string;
            message?: { usage?: AnthropicUsage };
            usage?: AnthropicUsage;
            delta?: { type?: string; text?: string };
          }
        | undefined;
      if (ev?.type === "message_start" && ev.message?.usage) {
        inputTokens = ev.message.usage.input_tokens ?? inputTokens;
        cacheRead = ev.message.usage.cache_read_input_tokens ?? cacheRead;
        outputTokens = ev.message.usage.output_tokens ?? outputTokens;
      } else if (ev?.type === "message_delta" && ev.usage) {
        // message_delta.usage.output_tokens is cumulative — overwrite.
        if (typeof ev.usage.output_tokens === "number") {
          outputTokens = ev.usage.output_tokens;
        }
      } else if (
        ev?.type === "content_block_delta" &&
        ev.delta?.type === "text_delta" &&
        typeof ev.delta.text === "string"
      ) {
        completionText += ev.delta.text;
      }
      yield event;
    }
  } catch (err) {
    threw = err;
    throw err;
  } finally {
    if (threw) {
      ctx.logger.log(failure(ctx, threw));
    } else if (inputTokens != null && outputTokens != null) {
      ctx.logger.log(
        success(ctx, {
          input_tokens: inputTokens,
          output_tokens: outputTokens,
          cache_read_input_tokens: cacheRead,
        }),
      );
    } else {
      // Stream ended without usage events — fall back to tiktoken on the
      // captured text. cl100k_base is a rough approximation for Anthropic
      // (see tiktoken.ts), so we mark the row as estimated.
      const promptTokens = await estimateTokens(ctx.model, ctx.promptText);
      const completionTokens = await estimateTokens(ctx.model, completionText);
      if (promptTokens != null && completionTokens != null) {
        ctx.logger.log({
          ...baseEntry(ctx),
          status: 200,
          latencyMs: Date.now() - ctx.startedAt,
          promptTokens,
          completionTokens,
          usageSource: "tiktoken",
        });
      } else {
        // Whatever partial data we have (e.g. input only). Log without a
        // usageSource — gateway will default to 'provider' for compatibility.
        ctx.logger.log(
          success(ctx, {
            input_tokens: inputTokens,
            output_tokens: outputTokens,
            cache_read_input_tokens: cacheRead,
          }),
        );
      }
    }
  }
}

export function wrapAnthropic<
  T extends { messages: { create: CreateFn } },
>(
  client: T,
  logger: Logger,
  getProcessTags: ProcessTagsGetter = () => undefined,
  guard: BudgetGuard | null = null,
  onBreach: BudgetOnBreach = "throw",
  // Day 24: cross-provider fallback chain. See openai.ts for semantics —
  // identical behavior here, mirrored so the wrap surfaces are symmetric.
  fallbacks: FallbackTarget[] = [],
  agentBinding: AgentBinding | null = null,
): WithAgent<T> {
  const realCreate = client.messages.create.bind(client.messages);

  const coreCreate: CreateFn = async (params, options) => {
    // Day 20 pre-call gate — see openai.ts for the rationale; mirror here
    // so both wrappers share enforcement semantics.
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
      const promptText = extractPromptText(params);
      let upstream: AsyncIterable<unknown>;
      try {
        upstream = await realCreate(params, options);
      } catch (err) {
        logger.log(failure(ctxBase, err));
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
      logger.log(success(ctxBase, result?.usage));
      return result;
    } catch (err) {
      primaryErr = err;
      logger.log(failure(ctxBase, err));
    }

    // ─── Day 24 fallback chain (mirror of wrapOpenAI's tail) ──────────────
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
        primaryShape: "anthropic",
        originalParams: params,
        target,
        options,
      });

      if (outcome.kind === "unsupported") {
        throw primaryErr;
      }

      if (outcome.kind === "success") {
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

  const messagesProxy = new Proxy(client.messages, {
    get(target, prop, receiver) {
      if (prop === "create") return wrappedCreate;
      return Reflect.get(target, prop, receiver);
    },
  });

  return new Proxy(client, {
    get(target, prop, receiver) {
      if (prop === "withAgent") {
        return (agent: string, o?: { tags?: TagInput }) =>
          wrapAnthropic(client, logger, getProcessTags, guard, onBreach, fallbacks, {
            agent,
            tags: normalizeTags(o?.tags),
          });
      }
      if (prop === "messages") return messagesProxy;
      return Reflect.get(target, prop, receiver);
    },
  }) as WithAgent<T>;
}
