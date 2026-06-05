// Translators where Gemini's generateContent is the SOURCE shape.
// Plan 5 will add geminiToMistralParams (and the inverse mistral
// response translator) to this file.

import type {
  AnthropicMessage,
  AnthropicMessageResponse,
  AnthropicMessagesParams,
  BedrockConverseParams,
  BedrockConverseResponse,
  BedrockMessage,
  ChatCompletionParams,
  ChatCompletionResponse,
  ChatMessage,
  GeminiContent,
  GeminiParams,
  GeminiPart,
  GeminiResponse,
  MistralChatParams,
  MistralMessage,
} from "./index.js";

// ─── helpers ─────────────────────────────────────────────────────────────────

export function flattenGeminiParts(parts: GeminiPart[] | undefined): string {
  if (!Array.isArray(parts)) return "";
  return parts
    .map((p) => (typeof p?.text === "string" ? p.text : ""))
    .filter(Boolean)
    .join("");
}

export function flattenGeminiSystem(
  sys: { parts: GeminiPart[] } | GeminiPart[] | undefined,
): string {
  if (!sys) return "";
  if (Array.isArray(sys)) return flattenGeminiParts(sys);
  if (typeof (sys as { parts?: unknown }).parts !== "undefined") {
    return flattenGeminiParts((sys as { parts: GeminiPart[] }).parts);
  }
  return "";
}

export function geminiHasUnsupportedShape(params: GeminiParams): boolean {
  // No tools/functions in v1 (Gemini's tool API is its own shape).
  // Treat any non-text part as unsupported.
  for (const c of params.contents) {
    for (const p of c.parts ?? []) {
      if (typeof p.text !== "string") return true;
    }
  }
  return false;
}

// ─── gemini → openai ─────────────────────────────────────────────────────────

export function geminiToChatParams(
  params: GeminiParams,
  toModel: string,
): ChatCompletionParams | null {
  if (geminiHasUnsupportedShape(params)) return null;

  const messages: ChatMessage[] = [];
  const sysText = flattenGeminiSystem(params.config?.systemInstruction);
  if (sysText) messages.push({ role: "system", content: sysText });

  for (const c of params.contents) {
    // Gemini uses "model" for assistant; map back.
    const role = c.role === "model" ? "assistant" : "user";
    messages.push({ role, content: flattenGeminiParts(c.parts) });
  }

  const out: ChatCompletionParams = { model: toModel, messages };
  const cfg = params.config ?? {};
  if (typeof cfg.maxOutputTokens === "number") out.max_tokens = cfg.maxOutputTokens;
  if (typeof cfg.temperature === "number") out.temperature = cfg.temperature;
  if (typeof cfg.topP === "number") out.top_p = cfg.topP;
  if (Array.isArray(cfg.stopSequences) && cfg.stopSequences.length > 0)
    out.stop = cfg.stopSequences;
  return out;
}

export function chatResponseFromGemini(
  response: ChatCompletionResponse,
  callerModel: string,
): GeminiResponse {
  const choice = response.choices?.[0];
  const text = choice?.message?.content ?? "";
  const finish = choice?.finish_reason;
  return {
    candidates: [
      {
        content: { role: "model", parts: [{ text }] },
        finishReason: (() => {
          switch (finish) {
            case "length":
              return "MAX_TOKENS";
            case "content_filter":
              return "SAFETY";
            default:
              return "STOP";
          }
        })(),
      },
    ],
    usageMetadata: {
      promptTokenCount: response.usage?.prompt_tokens ?? 0,
      candidatesTokenCount: response.usage?.completion_tokens ?? 0,
      totalTokenCount: response.usage?.total_tokens ?? 0,
    },
  };
}

// ─── gemini → anthropic ──────────────────────────────────────────────────────

export function geminiToAnthropicParams(
  params: GeminiParams,
  toModel: string,
): AnthropicMessagesParams | null {
  if (geminiHasUnsupportedShape(params)) return null;
  const messages: AnthropicMessage[] = [];
  for (const c of params.contents) {
    const role = c.role === "model" ? "assistant" : "user";
    messages.push({ role, content: flattenGeminiParts(c.parts) });
  }
  const out: AnthropicMessagesParams = {
    model: toModel,
    messages,
    max_tokens: params.config?.maxOutputTokens ?? 1024,
  };
  const sysText = flattenGeminiSystem(params.config?.systemInstruction);
  if (sysText) out.system = sysText;
  const cfg = params.config ?? {};
  if (typeof cfg.temperature === "number") out.temperature = cfg.temperature;
  if (typeof cfg.topP === "number") out.top_p = cfg.topP;
  if (Array.isArray(cfg.stopSequences) && cfg.stopSequences.length > 0)
    out.stop_sequences = cfg.stopSequences;
  return out;
}

export function anthropicResponseFromGemini(
  response: AnthropicMessageResponse,
  callerModel: string,
): GeminiResponse {
  const text = (response.content ?? [])
    .map((b) => (typeof b.text === "string" ? b.text : ""))
    .filter(Boolean)
    .join("");
  return {
    candidates: [
      {
        content: { role: "model", parts: [{ text }] },
        finishReason: (() => {
          switch (response.stop_reason) {
            case "max_tokens":
              return "MAX_TOKENS";
            default:
              return "STOP";
          }
        })(),
      },
    ],
    usageMetadata: {
      promptTokenCount: response.usage?.input_tokens ?? 0,
      candidatesTokenCount: response.usage?.output_tokens ?? 0,
      totalTokenCount:
        (response.usage?.input_tokens ?? 0) + (response.usage?.output_tokens ?? 0),
    },
  };
}

// ─── gemini → bedrock ────────────────────────────────────────────────────────

export function geminiToBedrockParams(
  params: GeminiParams,
  toModelId: string,
): BedrockConverseParams | null {
  if (geminiHasUnsupportedShape(params)) return null;
  const messages: BedrockMessage[] = [];
  for (const c of params.contents) {
    const role = c.role === "model" ? "assistant" : "user";
    messages.push({ role, content: [{ text: flattenGeminiParts(c.parts) }] });
  }
  const inferenceConfig: BedrockConverseParams["inferenceConfig"] = {};
  const cfg = params.config ?? {};
  if (typeof cfg.maxOutputTokens === "number") inferenceConfig.maxTokens = cfg.maxOutputTokens;
  if (typeof cfg.temperature === "number") inferenceConfig.temperature = cfg.temperature;
  if (typeof cfg.topP === "number") inferenceConfig.topP = cfg.topP;
  if (Array.isArray(cfg.stopSequences) && cfg.stopSequences.length > 0)
    inferenceConfig.stopSequences = cfg.stopSequences;
  const out: BedrockConverseParams = { modelId: toModelId, messages };
  const sysText = flattenGeminiSystem(cfg.systemInstruction);
  if (sysText) out.system = [{ text: sysText }];
  if (Object.keys(inferenceConfig).length > 0) out.inferenceConfig = inferenceConfig;
  return out;
}

export function bedrockResponseFromGemini(
  response: BedrockConverseResponse,
  callerModel: string,
): GeminiResponse {
  const blocks = response.output?.message?.content ?? [];
  const text = blocks
    .map((b) => (typeof b.text === "string" ? b.text : ""))
    .filter(Boolean)
    .join("");
  return {
    candidates: [
      {
        content: { role: "model", parts: [{ text }] },
        finishReason: (() => {
          switch (response.stopReason) {
            case "max_tokens":
              return "MAX_TOKENS";
            case "content_filtered":
            case "guardrail_intervened":
              return "SAFETY";
            default:
              return "STOP";
          }
        })(),
      },
    ],
    usageMetadata: {
      promptTokenCount: response.usage?.inputTokens ?? 0,
      candidatesTokenCount: response.usage?.outputTokens ?? 0,
      totalTokenCount:
        response.usage?.totalTokens ??
        (response.usage?.inputTokens ?? 0) + (response.usage?.outputTokens ?? 0),
    },
  };
}

/** Gemini generateContent params → Mistral chat.complete params.
 *  Returns null when unsupported. */
export function geminiToMistralParams(
  params: GeminiParams,
  toModel: string,
): MistralChatParams | null {
  if (geminiHasUnsupportedShape(params)) return null;
  const messages: MistralMessage[] = [];
  const sysText = flattenGeminiSystem(params.config?.systemInstruction);
  if (sysText) messages.push({ role: "system", content: sysText });
  for (const c of params.contents) {
    const role = c.role === "model" ? "assistant" : "user";
    messages.push({ role, content: flattenGeminiParts(c.parts) });
  }
  const cfg = params.config ?? {};
  const out: MistralChatParams = { model: toModel, messages };
  if (typeof cfg.maxOutputTokens === "number") out.maxTokens = cfg.maxOutputTokens;
  if (typeof cfg.temperature === "number") out.temperature = cfg.temperature;
  if (typeof cfg.topP === "number") out.topP = cfg.topP;
  if (Array.isArray(cfg.stopSequences) && cfg.stopSequences.length > 0)
    out.stop = cfg.stopSequences;
  return out;
}
