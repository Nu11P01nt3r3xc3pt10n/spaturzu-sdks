// Gemini (@google/genai) wrap. Mirrors today's openai.ts / anthropic.ts /
// bedrock.ts shape: log-entry builders, prompt-text extractor for the
// tiktoken fallback, async observer, wrap function returning a Proxy.
//
// Notes from spec §2.2:
//   • Intercepts client.models.generateContent / generateContentStream.
//     `@google/genai` is async-by-default; sync/async detection is not
//     needed (everything's a Promise / AsyncIterable).
//   • Stream chunks carry CUMULATIVE usageMetadata; the final chunk's
//     value is the full count. Observer keeps last-seen (overwrite).
//   • Roles: Gemini's content array uses "user" / "model". The wrap
//     does NOT translate roles in the captured prompt-text extractor —
//     it's only for tiktoken fallback, and the role label there is for
//     prompt-shape hashing, not semantic correctness.

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

interface GenerateFn {
  (params: any, options?: any): Promise<any>;
}

type GeminiUsage = {
  promptTokenCount?: number;
  candidatesTokenCount?: number;
  cachedContentTokenCount?: number;
  totalTokenCount?: number;
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
    provider: "gemini",
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
  usage: GeminiUsage | null | undefined,
): SdkLogEntry {
  return {
    ...buildBaseEntry(opts),
    status: 200,
    latencyMs: Date.now() - opts.startedAt,
    promptTokens: usage?.promptTokenCount,
    completionTokens: usage?.candidatesTokenCount,
    cachedInputTokens: usage?.cachedContentTokenCount,
    usageSource: "provider",
  };
}

function buildError(
  opts: { requestId: string; model: string; startedAt: number; getProcessTags: ProcessTagsGetter; frame: RunFrame | undefined },
  err: unknown,
): SdkLogEntry {
  const raw =
    typeof (err as { status?: unknown })?.status === "number"
      ? ((err as { status: number }).status)
      : 500;
  return {
    ...buildBaseEntry(opts),
    status: Math.min(599, Math.max(100, raw)),
    latencyMs: Date.now() - opts.startedAt,
  };
}

function extractPromptText(params: any): string {
  const parts: string[] = [];
  const sys = params?.config?.systemInstruction;
  if (sys?.parts && Array.isArray(sys.parts)) {
    for (const p of sys.parts) {
      if (typeof p?.text === "string") parts.push(p.text);
    }
  } else if (Array.isArray(sys)) {
    for (const p of sys) {
      if (typeof p?.text === "string") parts.push(p.text);
    }
  }
  const contents = params?.contents;
  if (Array.isArray(contents)) {
    for (const c of contents) {
      const role = typeof c?.role === "string" ? c.role : "";
      if (Array.isArray(c?.parts)) {
        for (const p of c.parts) {
          if (typeof p?.text === "string") {
            if (role) parts.push(role);
            parts.push(p.text);
          }
        }
      }
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
  let usage: GeminiUsage | null = null;
  let completionText = "";
  let threw: unknown = null;
  try {
    for await (const chunk of upstream) {
      const c = chunk as {
        usageMetadata?: GeminiUsage;
        candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>;
      };
      // Each chunk carries cumulative usage; the LAST one is authoritative.
      // We just keep overwriting.
      if (c.usageMetadata) usage = c.usageMetadata;
      const cand = c.candidates?.[0];
      if (cand?.content?.parts) {
        for (const p of cand.content.parts) {
          if (typeof p?.text === "string") completionText += p.text;
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

export function wrapGemini<
  T extends {
    models: { generateContent: GenerateFn; generateContentStream: GenerateFn };
  },
>(
  client: T,
  logger: Logger,
  getProcessTags: ProcessTagsGetter = () => undefined,
  guard: BudgetGuard | null = null,
  onBreach: BudgetOnBreach = "throw",
  fallbacks: FallbackTarget[] = [],
  agentBinding: AgentBinding | null = null,
): WithAgent<T> {
  const realGenerate = client.models.generateContent.bind(client.models);
  const realGenerateStream = client.models.generateContentStream.bind(client.models);

  async function gateBudget(): Promise<void> {
    if (guard) {
      const frame = getCurrentFrame();
      await guard.preCallCheck(frame?.agentName ?? null, onBreach);
    }
  }

  const coreGenerate: GenerateFn = async (params, options) => {
    await gateBudget();
    const requestId = uuidv7();
    setLastRequestId(requestId);
    const startedAt = Date.now();
    const model: string = params?.model ?? "unknown";
    const ctxBase = { requestId, model, startedAt, getProcessTags, frame: getCurrentFrame() };

    let primaryErr: unknown;
    try {
      const result = await realGenerate(params, options);
      logger.log(buildSuccess(ctxBase, result?.usageMetadata));
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
        primaryShape: "gemini",
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

  const coreGenerateStream: GenerateFn = async (params, options) => {
    await gateBudget();
    const requestId = uuidv7();
    setLastRequestId(requestId);
    const startedAt = Date.now();
    const model: string = params?.model ?? "unknown";
    const ctxBase = { requestId, model, startedAt, getProcessTags, frame: getCurrentFrame() };
    const promptText = extractPromptText(params);

    let upstream: AsyncIterable<unknown>;
    try {
      upstream = (await realGenerateStream(params, options)) as AsyncIterable<unknown>;
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
  const wrappedGenerate = bindAgent(coreGenerate);
  const wrappedGenerateStream = bindAgent(coreGenerateStream);

  // Proxy chain: models→{generateContent, generateContentStream}.
  const modelsProxy = new Proxy(client.models, {
    get(target, prop, receiver) {
      if (prop === "generateContent") return wrappedGenerate;
      if (prop === "generateContentStream") return wrappedGenerateStream;
      return Reflect.get(target, prop, receiver);
    },
  });
  return new Proxy(client, {
    get(target, prop, receiver) {
      if (prop === "withAgent") {
        return (agent: string, o?: { tags?: TagInput }) =>
          wrapGemini(client, logger, getProcessTags, guard, onBreach, fallbacks, {
            agent,
            tags: normalizeTags(o?.tags),
          });
      }
      if (prop === "models") return modelsProxy;
      return Reflect.get(target, prop, receiver);
    },
  }) as WithAgent<T>;
}
