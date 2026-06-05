// Translators where Bedrock's Converse API is the SOURCE shape.
//
// Plans 4/5 will add bedrockToGeminiParams / bedrockToMistralParams (and
// the inverse response translators) to this file. For Plan 3 we only
// ship the bedrock↔openai and bedrock↔anthropic pairs.
//
// All translators here are pure: no I/O, no clock reads beyond
// `Date.now()` for synthesised `created` timestamps on responses.

import type {
  AnthropicMessage,
  AnthropicMessageResponse,
  AnthropicMessagesParams,
  BedrockContentBlock,
  BedrockConverseParams,
  BedrockConverseResponse,
  BedrockMessage,
  ChatCompletionParams,
  ChatCompletionResponse,
  ChatMessage,
  GeminiContent,
  GeminiParams,
  GeminiPart,
  MistralChatParams,
  MistralMessage,
} from "./index.js";

// ─── helpers ─────────────────────────────────────────────────────────────────

export function flattenBedrockContent(blocks: BedrockContentBlock[]): string {
  const parts: string[] = [];
  for (const b of blocks) {
    if (typeof b.text === "string") parts.push(b.text);
  }
  return parts.join("");
}

export function bedrockHasUnsupportedShape(params: BedrockConverseParams): boolean {
  if (params.toolConfig !== undefined && params.toolConfig !== null) return true;
  // additionalModelRequestFields is provider-specific raw JSON — translating
  // it across providers would silently change behavior; refuse.
  if (params.additionalModelRequestFields !== undefined) return true;
  // Non-text content blocks (images, documents, etc.) not in v1.
  for (const m of params.messages) {
    for (const b of m.content) {
      if (typeof (b as { text?: unknown }).text !== "string") return true;
    }
  }
  return false;
}

// ─── bedrock → openai ────────────────────────────────────────────────────────

/** Bedrock Converse params → OpenAI ChatCompletion params.
 *  Returns null when the call uses features v1 doesn't translate. */
export function bedrockToChatParams(
  params: BedrockConverseParams,
  toModel: string,
): ChatCompletionParams | null {
  if (bedrockHasUnsupportedShape(params)) return null;

  const messages: ChatMessage[] = [];

  // Bedrock holds the system prompt at the top level as an array of text
  // blocks; OpenAI expects it as a message in the array.
  if (params.system && params.system.length > 0) {
    const sysText = params.system
      .map((b) => (typeof b.text === "string" ? b.text : ""))
      .filter(Boolean)
      .join("\n\n");
    if (sysText) messages.push({ role: "system", content: sysText });
  }

  for (const m of params.messages) {
    messages.push({ role: m.role, content: flattenBedrockContent(m.content) });
  }

  const out: ChatCompletionParams = { model: toModel, messages };
  const cfg = params.inferenceConfig ?? {};
  if (typeof cfg.maxTokens === "number") out.max_tokens = cfg.maxTokens;
  if (typeof cfg.temperature === "number") out.temperature = cfg.temperature;
  if (typeof cfg.topP === "number") out.top_p = cfg.topP;
  if (Array.isArray(cfg.stopSequences) && cfg.stopSequences.length > 0)
    out.stop = cfg.stopSequences;
  return out;
}

/** OpenAI ChatCompletion response → Bedrock Converse response.
 *  Used when bedrock is the PRIMARY and an openai fallback served the
 *  call — translates the openai response back into a Bedrock shape so
 *  the caller's downstream code sees what it expected. */
export function chatResponseFromBedrock(
  response: ChatCompletionResponse,
  callerModelId: string,
): BedrockConverseResponse {
  const choice = response.choices?.[0];
  const text = choice?.message?.content ?? "";
  return {
    output: {
      message: { role: "assistant", content: [{ text }] },
    },
    stopReason: (() => {
      switch (choice?.finish_reason) {
        case "length":
          return "max_tokens";
        case "tool_calls":
          return "tool_use";
        case "content_filter":
          return "content_filtered";
        default:
          return "end_turn";
      }
    })(),
    usage: {
      inputTokens: response.usage?.prompt_tokens ?? 0,
      outputTokens: response.usage?.completion_tokens ?? 0,
      totalTokens: response.usage?.total_tokens ?? 0,
    },
  };
}

// ─── bedrock → anthropic ─────────────────────────────────────────────────────

/** Bedrock Converse params → Anthropic Messages params. */
export function bedrockToAnthropicParams(
  params: BedrockConverseParams,
  toModel: string,
): AnthropicMessagesParams | null {
  if (bedrockHasUnsupportedShape(params)) return null;

  const messages: AnthropicMessage[] = [];
  for (const m of params.messages) {
    messages.push({ role: m.role, content: flattenBedrockContent(m.content) });
  }

  const out: AnthropicMessagesParams = {
    model: toModel,
    messages,
    max_tokens: params.inferenceConfig?.maxTokens ?? 1024,
  };

  if (params.system && params.system.length > 0) {
    out.system = params.system
      .map((b) => (typeof b.text === "string" ? b.text : ""))
      .filter(Boolean)
      .join("\n\n");
  }
  const cfg = params.inferenceConfig ?? {};
  if (typeof cfg.temperature === "number") out.temperature = cfg.temperature;
  if (typeof cfg.topP === "number") out.top_p = cfg.topP;
  if (Array.isArray(cfg.stopSequences) && cfg.stopSequences.length > 0)
    out.stop_sequences = cfg.stopSequences;

  return out;
}

/** Anthropic Messages response → Bedrock Converse response. */
export function anthropicResponseFromBedrock(
  response: AnthropicMessageResponse,
  callerModelId: string,
): BedrockConverseResponse {
  const text = (response.content ?? [])
    .map((b) => (typeof b.text === "string" ? b.text : ""))
    .filter(Boolean)
    .join("");
  return {
    output: { message: { role: "assistant", content: [{ text }] } },
    stopReason: (() => {
      switch (response.stop_reason) {
        case "max_tokens":
          return "max_tokens";
        case "stop_sequence":
          return "stop_sequence";
        case "tool_use":
          return "tool_use";
        default:
          return "end_turn";
      }
    })(),
    usage: {
      inputTokens: response.usage?.input_tokens ?? 0,
      outputTokens: response.usage?.output_tokens ?? 0,
      totalTokens:
        (response.usage?.input_tokens ?? 0) + (response.usage?.output_tokens ?? 0),
    },
  };
}

/** Bedrock Converse params → Gemini generateContent params.
 *  Returns null when unsupported. */
export function bedrockToGeminiParams(
  params: BedrockConverseParams,
  toModel: string,
): GeminiParams | null {
  if (bedrockHasUnsupportedShape(params)) return null;
  const contents: GeminiContent[] = [];
  for (const m of params.messages) {
    contents.push({
      role: m.role === "assistant" ? "model" : "user",
      parts: [{ text: flattenBedrockContent(m.content) }],
    });
  }
  const config: GeminiParams["config"] = {};
  const cfg = params.inferenceConfig ?? {};
  if (typeof cfg.maxTokens === "number") config.maxOutputTokens = cfg.maxTokens;
  if (typeof cfg.temperature === "number") config.temperature = cfg.temperature;
  if (typeof cfg.topP === "number") config.topP = cfg.topP;
  if (Array.isArray(cfg.stopSequences) && cfg.stopSequences.length > 0)
    config.stopSequences = cfg.stopSequences;
  if (params.system && params.system.length > 0) {
    const sysText = params.system
      .map((b) => (typeof b.text === "string" ? b.text : ""))
      .filter(Boolean)
      .join("\n\n");
    if (sysText) config.systemInstruction = { parts: [{ text: sysText }] };
  }
  const out: GeminiParams = { model: toModel, contents };
  if (Object.keys(config).length > 0) out.config = config;
  return out;
}

/** Bedrock Converse params → Mistral chat.complete params.
 *  Returns null when unsupported. */
export function bedrockToMistralParams(
  params: BedrockConverseParams,
  toModel: string,
): MistralChatParams | null {
  if (bedrockHasUnsupportedShape(params)) return null;
  const messages: MistralMessage[] = [];
  if (params.system && params.system.length > 0) {
    const sysText = params.system
      .map((b) => (typeof b.text === "string" ? b.text : ""))
      .filter(Boolean)
      .join("\n\n");
    if (sysText) messages.push({ role: "system", content: sysText });
  }
  for (const m of params.messages) {
    messages.push({ role: m.role, content: flattenBedrockContent(m.content) });
  }
  const out: MistralChatParams = { model: toModel, messages };
  const cfg = params.inferenceConfig ?? {};
  if (typeof cfg.maxTokens === "number") out.maxTokens = cfg.maxTokens;
  if (typeof cfg.temperature === "number") out.temperature = cfg.temperature;
  if (typeof cfg.topP === "number") out.topP = cfg.topP;
  if (cfg.stopSequences && cfg.stopSequences.length > 0) out.stop = cfg.stopSequences;
  return out;
}
