// Day 24 — SDK-side cross-provider fallback.
//
// When a wrapped call fails with a retryable upstream error (429 / 5xx /
// connection-class), walk a customer-configured fallback chain and try the
// next provider's client. Each attempt is its own log entry against the
// spaturzu gateway, so the trace surfaces "primary failed, fallback served"
// with correct per-call provider attribution (`requests.provider` column).
//
// This file is the pure dispatcher — no Logger, no Proxy plumbing. The
// wraps (openai.ts / anthropic.ts) own the logging and call this to do
// the actual cross-provider call.
//
// Why SDK-side and not gateway-side: with async-logging the customer's
// process already holds every provider's API key. The translation can be
// done locally with zero new auth surface, zero prompt storage on our
// servers. Proxy-mode Day 24 (encrypted keys, gateway routing) is a
// separate effort layered on top of Day 22.

// Direct per-source-provider imports keep the module graph shallow and let
// bundlers prune unused translator files for customers who wrap a single
// provider without cross-shape fallback. Pairs with `sideEffects: false`
// in package.json — the translator files have no module-load side effects,
// so unused exports across the chain are safely droppable. The legacy
// `./translate.js` shim still exists for Plan-1 tests; new code should
// route through this file (which only consumers needing dispatch import).

import {
  chatToAnthropicParams,
  chatResponseToAnthropic,
  chatToBedrockParams,
  chatToGeminiParams,
  chatToMistralParams,
} from "./translate/openai.js";

import {
  anthropicParamsToChat,
  anthropicResponseToChat,
  anthropicToBedrockParams,
  anthropicToGeminiParams,
  anthropicToMistralParams,
} from "./translate/anthropic.js";

import {
  bedrockToChatParams,
  chatResponseFromBedrock,
  bedrockToAnthropicParams,
  anthropicResponseFromBedrock,
  bedrockToGeminiParams,
  bedrockToMistralParams,
} from "./translate/bedrock.js";

import {
  bedrockResponseToChat,
  bedrockResponseToAnthropic,
} from "./translate/bedrock-to-others.js";

import {
  geminiToChatParams,
  chatResponseFromGemini,
  geminiToAnthropicParams,
  anthropicResponseFromGemini,
  geminiToBedrockParams,
  bedrockResponseFromGemini,
  geminiToMistralParams,
} from "./translate/gemini.js";

import {
  geminiResponseToChat,
  geminiResponseToAnthropic,
  geminiResponseToBedrock,
} from "./translate/gemini-to-others.js";

import {
  mistralToChatParams,
  chatResponseFromMistral,
  mistralToAnthropicParams,
  anthropicResponseFromMistral,
  mistralToBedrockParams,
  bedrockResponseFromMistral,
  mistralToGeminiParams,
  geminiResponseFromMistral,
} from "./translate/mistral.js";

import {
  mistralResponseToChat,
  mistralResponseToAnthropic,
  mistralResponseToBedrock,
  mistralResponseToGemini,
} from "./translate/mistral-to-others.js";

import type {
  AnthropicMessageResponse,
  AnthropicMessagesParams,
  BedrockConverseParams,
  BedrockConverseResponse,
  ChatCompletionParams,
  ChatCompletionResponse,
  GeminiParams,
  GeminiResponse,
  MistralChatParams,
  MistralChatResponse,
} from "./translate/index.js";

/** What the wrapped client looks like to the caller. Determines which
 *  translator pair to use when we cross provider boundaries. */
export type ClientShape =
  | "openai"
  | "anthropic"
  | "bedrock"
  | "gemini"
  | "mistral";

export type FallbackTarget = {
  /** Which provider this target speaks. Determines the request/response
   *  translation pair the dispatcher applies. */
  provider: ClientShape;
  /** The customer's SDK instance. Must expose the create-equivalent
   *  method matching `provider`:
   *    'openai'    → `chat.completions.create`
   *    'anthropic' → `messages.create`
   *    'bedrock'   → `converse` / `converseStream`
   *    'gemini'    → `models.generateContent` / `models.generateContentStream`
   *    'mistral'   → `chat.complete` / `chat.stream`
   *  Duck-typed — peer-dep versions stay flexible. */
  client: { [key: string]: unknown };
  /** Model name to use on the fallback provider. The caller's original
   *  `params.model` is preserved in the translated response's `.model`
   *  field so downstream code keeps seeing the model it asked for. */
  model: string;
};

/** Per-attempt return shape. The caller (wrap) logs and decides whether to
 *  move down the chain. */
export type FallbackOutcome =
  | { kind: "success"; response: unknown; usage: NormalisedUsage; latencyMs: number }
  | { kind: "unsupported"; reason: string }
  | { kind: "error"; error: unknown; status: number | null; latencyMs: number };

export type NormalisedUsage = {
  promptTokens: number | undefined;
  completionTokens: number | undefined;
  cachedInputTokens: number | undefined;
};

// Conservative classifier. We fallback on rate-limit, server-side, and
// connection-class errors only. Auth (401/403), bad-request (400), and
// "context overflow" (typically 400 with a body) are caller bugs that
// would just fail again on the next provider — surface them eagerly.
export function isRetryableUpstreamError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const e = err as { status?: unknown; code?: unknown; name?: unknown };

  const status = typeof e.status === "number" ? e.status : null;
  if (status !== null) {
    if (status === 429) return true;
    if (status >= 500 && status < 600) return true;
    // 408 Request Timeout — same family as 5xx in practice.
    if (status === 408) return true;
  }

  const code = typeof e.code === "string" ? e.code : null;
  if (code !== null) {
    if (code === "ETIMEDOUT") return true;
    if (code === "ECONNRESET") return true;
    if (code === "ECONNREFUSED") return true;
    if (code === "ENOTFOUND") return true;
    if (code === "EAI_AGAIN") return true;
  }

  // OpenAI / Anthropic SDK error classes — `name` is the only stable handle
  // we have without importing the peer deps.
  const name = typeof e.name === "string" ? e.name : null;
  if (name !== null) {
    if (name === "APIConnectionError") return true;
    if (name === "APIConnectionTimeoutError") return true;
    if (name === "APITimeoutError") return true;
    if (name === "RateLimitError") return true;
    if (name === "InternalServerError") return true;
    // AWS SDK error class names. The SDK v3 throws errors whose `name`
    // mirrors the AWS exception code (without the trailing "Exception"
    // on some, with it on others — match both).
    if (name === "ThrottlingException") return true;
    if (name === "ServiceUnavailableException") return true;
    if (name === "ModelTimeoutException") return true;
    if (name === "ModelStreamErrorException") return true;
    // @google/genai throws ApiError with .status set; we already match
    // by status above. Adding the class name for defense-in-depth.
    if (name === "ApiError") return true;
  }

  return false;
}

function statusOf(err: unknown): number | null {
  if (!err || typeof err !== "object") return null;
  const s = (err as { status?: unknown }).status;
  return typeof s === "number" ? s : null;
}

function pickSignal(options: unknown): { signal?: AbortSignal } | undefined {
  if (!options || typeof options !== "object") return undefined;
  const sig = (options as { signal?: unknown }).signal;
  if (sig instanceof AbortSignal) return { signal: sig };
  return undefined;
}

function usageFromOpenAi(resp: unknown): NormalisedUsage {
  const r = (resp ?? {}) as { usage?: Record<string, unknown> };
  const u = r.usage ?? {};
  const prompt = typeof u.prompt_tokens === "number" ? u.prompt_tokens : undefined;
  const completion =
    typeof u.completion_tokens === "number" ? u.completion_tokens : undefined;
  const cachedDetails = (u as { prompt_tokens_details?: { cached_tokens?: number } })
    .prompt_tokens_details;
  const cached =
    typeof cachedDetails?.cached_tokens === "number"
      ? cachedDetails.cached_tokens
      : undefined;
  return { promptTokens: prompt, completionTokens: completion, cachedInputTokens: cached };
}

function usageFromAnthropic(resp: unknown): NormalisedUsage {
  const r = (resp ?? {}) as { usage?: Record<string, unknown> };
  const u = r.usage ?? {};
  const input = typeof u.input_tokens === "number" ? u.input_tokens : undefined;
  const output = typeof u.output_tokens === "number" ? u.output_tokens : undefined;
  const cached =
    typeof u.cache_read_input_tokens === "number"
      ? u.cache_read_input_tokens
      : undefined;
  return { promptTokens: input, completionTokens: output, cachedInputTokens: cached };
}

function usageFromBedrock(resp: unknown): NormalisedUsage {
  const r = (resp ?? {}) as { usage?: Record<string, unknown> };
  const u = r.usage ?? {};
  return {
    promptTokens: typeof u.inputTokens === "number" ? u.inputTokens : undefined,
    completionTokens: typeof u.outputTokens === "number" ? u.outputTokens : undefined,
    cachedInputTokens:
      typeof u.cacheReadInputTokenCount === "number"
        ? u.cacheReadInputTokenCount
        : undefined,
  };
}

function usageFromGemini(resp: unknown): NormalisedUsage {
  const r = (resp ?? {}) as { usageMetadata?: Record<string, unknown> };
  const u = r.usageMetadata ?? {};
  return {
    promptTokens: typeof u.promptTokenCount === "number" ? u.promptTokenCount : undefined,
    completionTokens:
      typeof u.candidatesTokenCount === "number" ? u.candidatesTokenCount : undefined,
    cachedInputTokens:
      typeof u.cachedContentTokenCount === "number" ? u.cachedContentTokenCount : undefined,
  };
}

function usageFromMistral(resp: unknown): NormalisedUsage {
  const r = (resp ?? {}) as { usage?: Record<string, unknown> };
  const u = r.usage ?? {};
  return {
    promptTokens: typeof u.prompt_tokens === "number" ? u.prompt_tokens : undefined,
    completionTokens:
      typeof u.completion_tokens === "number" ? u.completion_tokens : undefined,
    cachedInputTokens: undefined, // Mistral has no prompt-cache feature.
  };
}

/** Run a single fallback attempt. Returns one of three outcomes; the caller
 *  decides how to log + whether to walk further down the chain. */
export async function tryFallback(opts: {
  /** Shape the wrapped client speaks. Determines translator pair. */
  primaryShape: ClientShape;
  /** Caller's original (untranslated) params. */
  originalParams: ChatCompletionParams | AnthropicMessagesParams | BedrockConverseParams | GeminiParams | MistralChatParams;
  target: FallbackTarget;
  /** Caller's original options object. Only `signal` is forwarded — other
   *  fields are provider-specific and would confuse the fallback's SDK. */
  options?: unknown;
}): Promise<FallbackOutcome> {
  const { primaryShape, originalParams, target } = opts;

  // ── translate request ─────────────────────────────────────────────────────
  let translatedParams: ChatCompletionParams | AnthropicMessagesParams | BedrockConverseParams | GeminiParams | MistralChatParams | null;
  let translateInbound:
    | "identity"
    | "openai→anthropic"
    | "anthropic→openai"
    | "openai→bedrock"
    | "bedrock→openai"
    | "anthropic→bedrock"
    | "bedrock→anthropic"
    | "openai→gemini"
    | "gemini→openai"
    | "anthropic→gemini"
    | "gemini→anthropic"
    | "bedrock→gemini"
    | "gemini→bedrock"
    | "openai→mistral"
    | "mistral→openai"
    | "anthropic→mistral"
    | "mistral→anthropic"
    | "bedrock→mistral"
    | "mistral→bedrock"
    | "gemini→mistral"
    | "mistral→gemini"
    | "not-yet-implemented";

  // Identity branches: model swap only, no shape translation.
  if (primaryShape === target.provider) {
    translatedParams = {
      ...(originalParams as Record<string, unknown>),
      model: target.model,
    } as ChatCompletionParams | AnthropicMessagesParams | BedrockConverseParams | MistralChatParams;
    translateInbound = "identity";
  }
  // Existing cross-shape pairs (openai ↔ anthropic).
  else if (primaryShape === "openai" && target.provider === "anthropic") {
    translatedParams = chatToAnthropicParams(
      originalParams as ChatCompletionParams,
      target.model,
    );
    translateInbound = "openai→anthropic";
  } else if (primaryShape === "anthropic" && target.provider === "openai") {
    translatedParams = anthropicParamsToChat(
      originalParams as AnthropicMessagesParams,
      target.model,
    );
    translateInbound = "anthropic→openai";
  } else if (primaryShape === "openai" && target.provider === "bedrock") {
    translatedParams = chatToBedrockParams(
      originalParams as ChatCompletionParams,
      target.model,
    );
    translateInbound = "openai→bedrock";
  } else if (primaryShape === "bedrock" && target.provider === "openai") {
    translatedParams = bedrockToChatParams(
      originalParams as BedrockConverseParams,
      target.model,
    );
    translateInbound = "bedrock→openai";
  } else if (primaryShape === "anthropic" && target.provider === "bedrock") {
    translatedParams = anthropicToBedrockParams(
      originalParams as AnthropicMessagesParams,
      target.model,
    );
    translateInbound = "anthropic→bedrock";
  } else if (primaryShape === "bedrock" && target.provider === "anthropic") {
    translatedParams = bedrockToAnthropicParams(
      originalParams as BedrockConverseParams,
      target.model,
    );
    translateInbound = "bedrock→anthropic";
  } else if (primaryShape === "openai" && target.provider === "gemini") {
    translatedParams = chatToGeminiParams(
      originalParams as ChatCompletionParams,
      target.model,
    );
    translateInbound = "openai→gemini";
  } else if (primaryShape === "gemini" && target.provider === "openai") {
    translatedParams = geminiToChatParams(
      originalParams as GeminiParams,
      target.model,
    );
    translateInbound = "gemini→openai";
  } else if (primaryShape === "anthropic" && target.provider === "gemini") {
    translatedParams = anthropicToGeminiParams(
      originalParams as AnthropicMessagesParams,
      target.model,
    );
    translateInbound = "anthropic→gemini";
  } else if (primaryShape === "gemini" && target.provider === "anthropic") {
    translatedParams = geminiToAnthropicParams(
      originalParams as GeminiParams,
      target.model,
    );
    translateInbound = "gemini→anthropic";
  } else if (primaryShape === "bedrock" && target.provider === "gemini") {
    translatedParams = bedrockToGeminiParams(
      originalParams as BedrockConverseParams,
      target.model,
    );
    translateInbound = "bedrock→gemini";
  } else if (primaryShape === "gemini" && target.provider === "bedrock") {
    translatedParams = geminiToBedrockParams(
      originalParams as GeminiParams,
      target.model,
    );
    translateInbound = "gemini→bedrock";
  } else if (primaryShape === "openai" && target.provider === "mistral") {
    translatedParams = chatToMistralParams(
      originalParams as ChatCompletionParams,
      target.model,
    );
    translateInbound = "openai→mistral";
  } else if (primaryShape === "mistral" && target.provider === "openai") {
    translatedParams = mistralToChatParams(
      originalParams as MistralChatParams,
      target.model,
    );
    translateInbound = "mistral→openai";
  } else if (primaryShape === "anthropic" && target.provider === "mistral") {
    translatedParams = anthropicToMistralParams(
      originalParams as AnthropicMessagesParams,
      target.model,
    );
    translateInbound = "anthropic→mistral";
  } else if (primaryShape === "mistral" && target.provider === "anthropic") {
    translatedParams = mistralToAnthropicParams(
      originalParams as MistralChatParams,
      target.model,
    );
    translateInbound = "mistral→anthropic";
  } else if (primaryShape === "bedrock" && target.provider === "mistral") {
    translatedParams = bedrockToMistralParams(
      originalParams as BedrockConverseParams,
      target.model,
    );
    translateInbound = "bedrock→mistral";
  } else if (primaryShape === "mistral" && target.provider === "bedrock") {
    translatedParams = mistralToBedrockParams(
      originalParams as MistralChatParams,
      target.model,
    );
    translateInbound = "mistral→bedrock";
  } else if (primaryShape === "gemini" && target.provider === "mistral") {
    translatedParams = geminiToMistralParams(
      originalParams as GeminiParams,
      target.model,
    );
    translateInbound = "gemini→mistral";
  } else if (primaryShape === "mistral" && target.provider === "gemini") {
    translatedParams = mistralToGeminiParams(
      originalParams as MistralChatParams,
      target.model,
    );
    translateInbound = "mistral→gemini";
  }
  // Defense-in-depth catch-all — unreachable now that the 5×5 matrix is complete.
  else {
    translatedParams = null;
    translateInbound = "not-yet-implemented";
  }

  if (translatedParams === null) {
    return {
      kind: "unsupported",
      reason:
        translateInbound === "not-yet-implemented"
          ? `cross-provider translation for ${primaryShape}→${target.provider} not yet implemented`
          : translateInbound === "openai→anthropic"
            ? "v1 cross-provider translation refuses streaming / tools / non-text content"
            : "v1 cross-provider translation refuses streaming / non-text content",
    };
  }

  // ── invoke target ─────────────────────────────────────────────────────────
  const callerModel =
    (originalParams as { model?: unknown }).model && typeof (originalParams as { model?: unknown }).model === "string"
      ? ((originalParams as { model: string }).model)
      : target.model;
  const fallbackOptions = pickSignal(opts.options);
  const startedAt = Date.now();

  let raw: unknown;
  try {
    if (target.provider === "openai") {
      const createFn = (
        target.client.chat as {
          completions: { create: (p: unknown, o?: unknown) => Promise<unknown> };
        }
      ).completions.create;
      raw = await createFn.call(
        (target.client.chat as { completions: unknown }).completions,
        translatedParams,
        fallbackOptions,
      );
    } else if (target.provider === "anthropic") {
      const createFn = (
        target.client.messages as {
          create: (p: unknown, o?: unknown) => Promise<unknown>;
        }
      ).create;
      raw = await createFn.call(target.client.messages, translatedParams, fallbackOptions);
    } else if (target.provider === "bedrock") {
      const converseFn = target.client.converse as (p: unknown, o?: unknown) => Promise<unknown>;
      raw = await converseFn.call(target.client, translatedParams, fallbackOptions);
    } else if (target.provider === "gemini") {
      const generateFn = (
        target.client.models as { generateContent: (p: unknown, o?: unknown) => Promise<unknown> }
      ).generateContent;
      raw = await generateFn.call(target.client.models, translatedParams, fallbackOptions);
    } else if (target.provider === "mistral") {
      const completeFn = (
        target.client.chat as { complete: (p: unknown, o?: unknown) => Promise<unknown> }
      ).complete;
      raw = await completeFn.call(target.client.chat, translatedParams, fallbackOptions);
    } else {
      // Exhaustive — TypeScript will catch a missing branch at compile time.
      const _exhaustive: never = target.provider;
      throw new Error(`unreachable: unknown provider ${_exhaustive}`);
    }
  } catch (err) {
    return {
      kind: "error",
      error: err,
      status: statusOf(err),
      latencyMs: Date.now() - startedAt,
    };
  }

  // ── translate response ────────────────────────────────────────────────────
  let response: unknown;
  let usage: NormalisedUsage;
  // 5×5 matrix complete — all identity branches handle their provider.
  if (translateInbound === "identity") {
    response = raw;
    if (target.provider === "openai") usage = usageFromOpenAi(raw);
    else if (target.provider === "anthropic") usage = usageFromAnthropic(raw);
    else if (target.provider === "bedrock") usage = usageFromBedrock(raw);
    else if (target.provider === "gemini") usage = usageFromGemini(raw);
    else if (target.provider === "mistral") usage = usageFromMistral(raw);
    else usage = { promptTokens: undefined, completionTokens: undefined, cachedInputTokens: undefined };
  } else if (translateInbound === "openai→anthropic") {
    response = anthropicResponseToChat(raw as AnthropicMessageResponse, callerModel);
    usage = usageFromAnthropic(raw);
  } else if (translateInbound === "anthropic→openai") {
    response = chatResponseToAnthropic(raw as ChatCompletionResponse, callerModel);
    usage = usageFromOpenAi(raw);
  } else if (translateInbound === "openai→bedrock") {
    response = bedrockResponseToChat(raw as BedrockConverseResponse, callerModel);
    usage = usageFromBedrock(raw);
  } else if (translateInbound === "bedrock→openai") {
    response = chatResponseFromBedrock(raw as ChatCompletionResponse, callerModel);
    usage = usageFromOpenAi(raw);
  } else if (translateInbound === "anthropic→bedrock") {
    response = bedrockResponseToAnthropic(raw as BedrockConverseResponse, callerModel);
    usage = usageFromBedrock(raw);
  } else if (translateInbound === "bedrock→anthropic") {
    response = anthropicResponseFromBedrock(raw as AnthropicMessageResponse, callerModel);
    usage = usageFromAnthropic(raw);
  } else if (translateInbound === "openai→gemini") {
    response = geminiResponseToChat(raw as GeminiResponse, callerModel);
    usage = usageFromGemini(raw);
  } else if (translateInbound === "gemini→openai") {
    response = chatResponseFromGemini(raw as ChatCompletionResponse, callerModel);
    usage = usageFromOpenAi(raw);
  } else if (translateInbound === "anthropic→gemini") {
    response = geminiResponseToAnthropic(raw as GeminiResponse, callerModel);
    usage = usageFromGemini(raw);
  } else if (translateInbound === "gemini→anthropic") {
    response = anthropicResponseFromGemini(raw as AnthropicMessageResponse, callerModel);
    usage = usageFromAnthropic(raw);
  } else if (translateInbound === "bedrock→gemini") {
    response = geminiResponseToBedrock(raw as GeminiResponse, callerModel);
    usage = usageFromGemini(raw);
  } else if (translateInbound === "gemini→bedrock") {
    response = bedrockResponseFromGemini(raw as BedrockConverseResponse, callerModel);
    usage = usageFromBedrock(raw);
  } else if (translateInbound === "openai→mistral") {
    response = mistralResponseToChat(raw as MistralChatResponse, callerModel);
    usage = usageFromMistral(raw);
  } else if (translateInbound === "mistral→openai") {
    response = chatResponseFromMistral(raw as ChatCompletionResponse, callerModel);
    usage = usageFromOpenAi(raw);
  } else if (translateInbound === "anthropic→mistral") {
    response = mistralResponseToAnthropic(raw as MistralChatResponse, callerModel);
    usage = usageFromMistral(raw);
  } else if (translateInbound === "mistral→anthropic") {
    response = anthropicResponseFromMistral(raw as AnthropicMessageResponse, callerModel);
    usage = usageFromAnthropic(raw);
  } else if (translateInbound === "bedrock→mistral") {
    response = mistralResponseToBedrock(raw as MistralChatResponse, callerModel);
    usage = usageFromMistral(raw);
  } else if (translateInbound === "mistral→bedrock") {
    response = bedrockResponseFromMistral(raw as BedrockConverseResponse, callerModel);
    usage = usageFromBedrock(raw);
  } else if (translateInbound === "gemini→mistral") {
    response = mistralResponseToGemini(raw as MistralChatResponse, callerModel);
    usage = usageFromMistral(raw);
  } else if (translateInbound === "mistral→gemini") {
    response = geminiResponseFromMistral(raw as GeminiResponse, callerModel);
    usage = usageFromGemini(raw);
  } else {
    response = raw;
    usage = { promptTokens: undefined, completionTokens: undefined, cachedInputTokens: undefined };
  }

  return {
    kind: "success",
    response,
    usage,
    latencyMs: Date.now() - startedAt,
  };
}
