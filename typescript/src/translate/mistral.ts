// Translators where Mistral's chat.complete is the SOURCE shape.
// Plan 5 completes the 5×5 matrix — all 4 outbound + 4 inverse-response
// translators ship here.

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
  GeminiResponse,
  MistralChatParams,
  MistralChatResponse,
  MistralMessage,
} from "./index.js";

// ─── helpers ─────────────────────────────────────────────────────────────────

export function mistralHasUnsupportedShape(params: MistralChatParams): boolean {
  if (Array.isArray(params.tools) && params.tools.length > 0) return true;
  if (params.responseFormat !== undefined && params.responseFormat !== null) return true;
  for (const m of params.messages) {
    if (m.role === "tool") return true;
  }
  return false;
}

// ─── mistral → openai (near-identity) ────────────────────────────────────────

export function mistralToChatParams(
  params: MistralChatParams,
  toModel: string,
): ChatCompletionParams | null {
  if (mistralHasUnsupportedShape(params)) return null;
  const messages: ChatMessage[] = params.messages
    .filter((m) => m.role !== "tool")
    .map((m) => ({ role: m.role as ChatMessage["role"], content: m.content }));
  const out: ChatCompletionParams = { model: toModel, messages };
  if (typeof params.maxTokens === "number") out.max_tokens = params.maxTokens;
  if (typeof params.temperature === "number") out.temperature = params.temperature;
  if (typeof params.topP === "number") out.top_p = params.topP;
  if (params.stop !== undefined) out.stop = params.stop;
  return out;
}

export function chatResponseFromMistral(
  response: ChatCompletionResponse,
  callerModel: string,
): MistralChatResponse {
  return {
    id: response.id,
    object: "chat.completion",
    created: response.created,
    model: callerModel,
    choices: response.choices.map((c) => ({
      index: c.index,
      message: { role: "assistant", content: c.message.content },
      finish_reason: ((): MistralChatResponse["choices"][number]["finish_reason"] => {
        switch (c.finish_reason) {
          case "length":
            return "length";
          case "tool_calls":
            return "tool_calls";
          default:
            return "stop";
        }
      })(),
    })),
    usage: {
      prompt_tokens: response.usage.prompt_tokens,
      completion_tokens: response.usage.completion_tokens,
      total_tokens: response.usage.total_tokens,
    },
  };
}

// ─── mistral → anthropic ─────────────────────────────────────────────────────

export function mistralToAnthropicParams(
  params: MistralChatParams,
  toModel: string,
): AnthropicMessagesParams | null {
  if (mistralHasUnsupportedShape(params)) return null;
  const systemBuf: string[] = [];
  const messages: AnthropicMessage[] = [];
  for (const m of params.messages) {
    if (m.role === "system") {
      systemBuf.push(m.content);
      continue;
    }
    if (m.role !== "user" && m.role !== "assistant") continue;
    messages.push({ role: m.role, content: m.content });
  }
  const out: AnthropicMessagesParams = {
    model: toModel,
    messages,
    max_tokens: params.maxTokens ?? 1024,
  };
  if (systemBuf.length > 0) out.system = systemBuf.join("\n\n");
  if (typeof params.temperature === "number") out.temperature = params.temperature;
  if (typeof params.topP === "number") out.top_p = params.topP;
  if (typeof params.stop === "string") out.stop_sequences = [params.stop];
  else if (Array.isArray(params.stop) && params.stop.length > 0)
    out.stop_sequences = params.stop;
  return out;
}

export function anthropicResponseFromMistral(
  response: AnthropicMessageResponse,
  callerModel: string,
): MistralChatResponse {
  const text = (response.content ?? [])
    .map((b) => (typeof b.text === "string" ? b.text : ""))
    .filter(Boolean)
    .join("");
  return {
    id: response.id,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: callerModel,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: text },
        finish_reason: response.stop_reason === "max_tokens" ? "length" : "stop",
      },
    ],
    usage: {
      prompt_tokens: response.usage?.input_tokens ?? 0,
      completion_tokens: response.usage?.output_tokens ?? 0,
      total_tokens:
        (response.usage?.input_tokens ?? 0) + (response.usage?.output_tokens ?? 0),
    },
  };
}

// ─── mistral → bedrock ───────────────────────────────────────────────────────

export function mistralToBedrockParams(
  params: MistralChatParams,
  toModelId: string,
): BedrockConverseParams | null {
  if (mistralHasUnsupportedShape(params)) return null;
  const systemBlocks: Array<{ text: string }> = [];
  const messages: BedrockMessage[] = [];
  for (const m of params.messages) {
    if (m.role === "system") {
      systemBlocks.push({ text: m.content });
      continue;
    }
    if (m.role !== "user" && m.role !== "assistant") continue;
    messages.push({ role: m.role, content: [{ text: m.content }] });
  }
  const inferenceConfig: BedrockConverseParams["inferenceConfig"] = {};
  if (typeof params.maxTokens === "number") inferenceConfig.maxTokens = params.maxTokens;
  if (typeof params.temperature === "number") inferenceConfig.temperature = params.temperature;
  if (typeof params.topP === "number") inferenceConfig.topP = params.topP;
  if (typeof params.stop === "string") inferenceConfig.stopSequences = [params.stop];
  else if (Array.isArray(params.stop) && params.stop.length > 0)
    inferenceConfig.stopSequences = params.stop;
  const out: BedrockConverseParams = { modelId: toModelId, messages };
  if (systemBlocks.length > 0) out.system = systemBlocks;
  if (Object.keys(inferenceConfig).length > 0) out.inferenceConfig = inferenceConfig;
  return out;
}

export function bedrockResponseFromMistral(
  response: BedrockConverseResponse,
  callerModel: string,
): MistralChatResponse {
  const blocks = response.output?.message?.content ?? [];
  const text = blocks.map((b) => (typeof b.text === "string" ? b.text : "")).filter(Boolean).join("");
  return {
    id: "",
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: callerModel,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: text },
        finish_reason: response.stopReason === "max_tokens" ? "length" : "stop",
      },
    ],
    usage: {
      prompt_tokens: response.usage?.inputTokens ?? 0,
      completion_tokens: response.usage?.outputTokens ?? 0,
      total_tokens:
        response.usage?.totalTokens ??
        (response.usage?.inputTokens ?? 0) + (response.usage?.outputTokens ?? 0),
    },
  };
}

// ─── mistral → gemini ────────────────────────────────────────────────────────

export function mistralToGeminiParams(
  params: MistralChatParams,
  toModel: string,
): GeminiParams | null {
  if (mistralHasUnsupportedShape(params)) return null;
  const contents: GeminiContent[] = [];
  let systemText = "";
  for (const m of params.messages) {
    if (m.role === "system") {
      systemText = systemText ? `${systemText}\n\n${m.content}` : m.content;
      continue;
    }
    if (m.role !== "user" && m.role !== "assistant") continue;
    contents.push({
      role: m.role === "assistant" ? "model" : "user",
      parts: [{ text: m.content }],
    });
  }
  const config: GeminiParams["config"] = {};
  if (systemText) config.systemInstruction = { parts: [{ text: systemText }] };
  if (typeof params.maxTokens === "number") config.maxOutputTokens = params.maxTokens;
  if (typeof params.temperature === "number") config.temperature = params.temperature;
  if (typeof params.topP === "number") config.topP = params.topP;
  if (typeof params.stop === "string") config.stopSequences = [params.stop];
  else if (Array.isArray(params.stop) && params.stop.length > 0)
    config.stopSequences = params.stop;
  const out: GeminiParams = { model: toModel, contents };
  if (Object.keys(config).length > 0) out.config = config;
  return out;
}

export function geminiResponseFromMistral(
  response: GeminiResponse,
  callerModel: string,
): MistralChatResponse {
  const parts = response.candidates?.[0]?.content?.parts ?? [];
  const text = parts.map((p) => p.text ?? "").filter(Boolean).join("");
  const finish = response.candidates?.[0]?.finishReason;
  const u = response.usageMetadata ?? {};
  return {
    id: "",
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: callerModel,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: text },
        finish_reason: finish === "MAX_TOKENS" ? "length" : "stop",
      },
    ],
    usage: {
      prompt_tokens: u.promptTokenCount ?? 0,
      completion_tokens: u.candidatesTokenCount ?? 0,
      total_tokens: u.totalTokenCount ?? 0,
    },
  };
}
