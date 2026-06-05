// Translators where Anthropic's messages.create is the SOURCE shape.
// Plans 3/4/5 add anthropicToBedrock / anthropicToGemini / anthropicToMistral
// and the inverse response translators back to anthropic shape.

import type {
  AnthropicMessageResponse,
  AnthropicMessagesParams,
  AnthropicTextBlock,
  BedrockConverseParams,
  BedrockMessage,
  BedrockTextBlock,
  ChatCompletionParams,
  ChatCompletionResponse,
  ChatMessage,
  GeminiContent,
  GeminiParams,
  GeminiPart,
  MistralChatParams,
  MistralMessage,
} from "./index.js";

function flattenTextContent(
  content: string | ReadonlyArray<{ type?: unknown; text?: unknown }>,
): string {
  if (typeof content === "string") return content;
  const out: string[] = [];
  for (const part of content) {
    if (part && part.type === "text" && typeof part.text === "string") {
      out.push(part.text);
    }
  }
  return out.join("");
}

export function anthropicHasUnsupportedShape(
  params: AnthropicMessagesParams,
): boolean {
  if (params.stream === true) return true;
  for (const m of params.messages) {
    if (Array.isArray(m.content)) {
      for (const part of m.content) {
        if ((part as { type?: string }).type !== "text") return true;
      }
    }
  }
  return false;
}

function mapStopReasonToFinish(
  reason: AnthropicMessageResponse["stop_reason"],
): ChatCompletionResponse["choices"][number]["finish_reason"] {
  switch (reason) {
    case "max_tokens":
      return "length";
    case "tool_use":
      return "tool_calls";
    case "end_turn":
    case "stop_sequence":
    case null:
    case undefined:
      return "stop";
  }
}

/** Anthropic Messages params → OpenAI ChatCompletion params. */
export function anthropicParamsToChat(
  params: AnthropicMessagesParams,
  toModel: string,
): ChatCompletionParams | null {
  if (anthropicHasUnsupportedShape(params)) return null;

  const messages: ChatMessage[] = [];
  if (params.system !== undefined) {
    messages.push({
      role: "system",
      content: flattenTextContent(
        params.system as string | AnthropicTextBlock[],
      ),
    });
  }
  for (const m of params.messages) {
    messages.push({ role: m.role, content: flattenTextContent(m.content) });
  }

  const out: ChatCompletionParams = {
    model: toModel,
    messages,
    max_tokens: params.max_tokens,
  };
  if (params.temperature !== undefined) out.temperature = params.temperature;
  if (params.top_p !== undefined) out.top_p = params.top_p;
  if (params.stop_sequences && params.stop_sequences.length > 0)
    out.stop = params.stop_sequences;
  return out;
}

/** Anthropic Messages response → OpenAI ChatCompletion response. */
export function anthropicResponseToChat(
  response: AnthropicMessageResponse,
  callerModel: string,
): ChatCompletionResponse {
  return {
    id: response.id ?? "",
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: callerModel,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: flattenTextContent(response.content ?? []),
        },
        finish_reason: mapStopReasonToFinish(response.stop_reason ?? null),
      },
    ],
    usage: {
      prompt_tokens: response.usage?.input_tokens ?? 0,
      completion_tokens: response.usage?.output_tokens ?? 0,
      total_tokens:
        (response.usage?.input_tokens ?? 0) +
        (response.usage?.output_tokens ?? 0),
    },
  };
}

/** Anthropic Messages params → Bedrock Converse params.
 *  Returns null when unsupported. */
export function anthropicToBedrockParams(
  params: AnthropicMessagesParams,
  toModelId: string,
): BedrockConverseParams | null {
  if (anthropicHasUnsupportedShape(params)) return null;

  const messages: BedrockMessage[] = [];
  for (const m of params.messages) {
    messages.push({
      role: m.role,
      content: [{ text: flattenTextContent(m.content) }],
    });
  }

  const inferenceConfig: BedrockConverseParams["inferenceConfig"] = {
    maxTokens: params.max_tokens,
  };
  if (typeof params.temperature === "number")
    inferenceConfig.temperature = params.temperature;
  if (typeof params.top_p === "number") inferenceConfig.topP = params.top_p;
  if (params.stop_sequences && params.stop_sequences.length > 0)
    inferenceConfig.stopSequences = params.stop_sequences;

  const out: BedrockConverseParams = {
    modelId: toModelId,
    messages,
    inferenceConfig,
  };

  if (params.system !== undefined) {
    const sysText =
      typeof params.system === "string"
        ? params.system
        : params.system
            .map((b) => (typeof b.text === "string" ? b.text : ""))
            .filter(Boolean)
            .join("\n\n");
    if (sysText) out.system = [{ text: sysText }];
  }

  return out;
}

/** Anthropic Messages params → Gemini generateContent params.
 *  Returns null when unsupported. */
export function anthropicToGeminiParams(
  params: AnthropicMessagesParams,
  toModel: string,
): GeminiParams | null {
  if (anthropicHasUnsupportedShape(params)) return null;

  const contents: GeminiContent[] = [];
  for (const m of params.messages) {
    contents.push({
      role: m.role === "assistant" ? "model" : "user",
      parts: [{ text: flattenTextContent(m.content) }],
    });
  }
  const config: GeminiParams["config"] = { maxOutputTokens: params.max_tokens };
  if (params.system !== undefined) {
    const sysText =
      typeof params.system === "string"
        ? params.system
        : params.system
            .map((b) => (typeof b.text === "string" ? b.text : ""))
            .filter(Boolean)
            .join("\n\n");
    if (sysText) config.systemInstruction = { parts: [{ text: sysText }] };
  }
  if (typeof params.temperature === "number") config.temperature = params.temperature;
  if (typeof params.top_p === "number") config.topP = params.top_p;
  if (params.stop_sequences && params.stop_sequences.length > 0)
    config.stopSequences = params.stop_sequences;
  return { model: toModel, contents, config };
}

/** Anthropic Messages params → Mistral chat.complete params.
 *  Returns null when unsupported. */
export function anthropicToMistralParams(
  params: AnthropicMessagesParams,
  toModel: string,
): MistralChatParams | null {
  if (anthropicHasUnsupportedShape(params)) return null;
  const messages: MistralMessage[] = [];
  if (params.system !== undefined) {
    const sysText =
      typeof params.system === "string"
        ? params.system
        : (params.system as AnthropicTextBlock[]).map((b) => b.text ?? "").filter(Boolean).join("\n\n");
    if (sysText) messages.push({ role: "system", content: sysText });
  }
  for (const m of params.messages) {
    messages.push({ role: m.role, content: flattenTextContent(m.content) });
  }
  const out: MistralChatParams = { model: toModel, messages, maxTokens: params.max_tokens };
  if (typeof params.temperature === "number") out.temperature = params.temperature;
  if (typeof params.top_p === "number") out.topP = params.top_p;
  if (params.stop_sequences && params.stop_sequences.length > 0) out.stop = params.stop_sequences;
  return out;
}
