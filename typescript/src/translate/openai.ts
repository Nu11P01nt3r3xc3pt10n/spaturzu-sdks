// Translators where OpenAI's chat-completion is the SOURCE shape.
// Plans 3/4/5 add chatToBedrock / chatToGemini / chatToMistral and
// the inverse response translators back to chat shape.

import type {
  AnthropicMessage,
  AnthropicMessageResponse,
  AnthropicMessagesParams,
  BedrockConverseParams,
  BedrockMessage,
  BedrockTextBlock,
  ChatCompletionParams,
  ChatCompletionResponse,
  GeminiContent,
  GeminiParams,
  GeminiPart,
  MistralChatParams,
  MistralMessage,
} from "./index.js";

// ─── helpers ─────────────────────────────────────────────────────────────────

// Accepts both OpenAI and Anthropic content-part arrays. Permissive on
// purpose to dodge the TS union-of-arrays trap.
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

export function chatHasUnsupportedShape(params: ChatCompletionParams): boolean {
  if (params.stream === true) return true;
  if (Array.isArray(params.tools) && params.tools.length > 0) return true;
  if (params.tool_choice !== undefined && params.tool_choice !== null) return true;
  if (params.response_format !== undefined && params.response_format !== null)
    return true;
  for (const m of params.messages) {
    if (m.role === "tool") return true;
    if (Array.isArray(m.content)) {
      for (const part of m.content) {
        if ((part as { type?: string }).type !== "text") return true;
      }
    }
  }
  return false;
}

function mapFinishToStopReason(
  finish: ChatCompletionResponse["choices"][number]["finish_reason"],
): AnthropicMessageResponse["stop_reason"] {
  switch (finish) {
    case "length":
      return "max_tokens";
    case "tool_calls":
      return "tool_use";
    case "stop":
    case "content_filter":
    case null:
    case undefined:
      return "end_turn";
  }
}

// ─── translators ─────────────────────────────────────────────────────────────

/** OpenAI ChatCompletion params → Anthropic Messages params.
 *  Returns null when the call uses features v1 doesn't translate. */
export function chatToAnthropicParams(
  params: ChatCompletionParams,
  toModel: string,
): AnthropicMessagesParams | null {
  if (chatHasUnsupportedShape(params)) return null;

  const systemBuf: string[] = [];
  const messages: AnthropicMessage[] = [];
  for (const m of params.messages) {
    if (m.role === "system") {
      systemBuf.push(flattenTextContent(m.content));
      continue;
    }
    if (m.role !== "user" && m.role !== "assistant") continue;
    messages.push({ role: m.role, content: flattenTextContent(m.content) });
  }

  const out: AnthropicMessagesParams = {
    model: toModel,
    messages,
    max_tokens: params.max_tokens ?? params.max_completion_tokens ?? 1024,
  };
  if (systemBuf.length > 0) out.system = systemBuf.join("\n\n");
  if (params.temperature !== undefined) out.temperature = params.temperature;
  if (params.top_p !== undefined) out.top_p = params.top_p;
  if (typeof params.stop === "string") out.stop_sequences = [params.stop];
  else if (Array.isArray(params.stop) && params.stop.length > 0)
    out.stop_sequences = params.stop;
  return out;
}

/** OpenAI ChatCompletion response → Anthropic Messages response. */
export function chatResponseToAnthropic(
  response: ChatCompletionResponse,
  callerModel: string,
): AnthropicMessageResponse {
  const choice = response.choices?.[0];
  const text = choice?.message?.content ?? "";
  return {
    id: response.id ?? "",
    type: "message",
    role: "assistant",
    content: [{ type: "text", text }],
    model: callerModel,
    stop_reason: mapFinishToStopReason(choice?.finish_reason ?? null),
    usage: {
      input_tokens: response.usage?.prompt_tokens ?? 0,
      output_tokens: response.usage?.completion_tokens ?? 0,
    },
  };
}

/** OpenAI ChatCompletion params → Bedrock Converse params.
 *  Returns null when unsupported (streaming/tools/non-text). */
export function chatToBedrockParams(
  params: ChatCompletionParams,
  toModelId: string,
): BedrockConverseParams | null {
  if (chatHasUnsupportedShape(params)) return null;

  const systemBlocks: BedrockTextBlock[] = [];
  const messages: BedrockMessage[] = [];
  for (const m of params.messages) {
    if (m.role === "system") {
      systemBlocks.push({ text: flattenTextContent(m.content) });
      continue;
    }
    if (m.role !== "user" && m.role !== "assistant") continue;
    messages.push({
      role: m.role,
      content: [{ text: flattenTextContent(m.content) }],
    });
  }

  const inferenceConfig: BedrockConverseParams["inferenceConfig"] = {};
  const maxTokens = params.max_tokens ?? params.max_completion_tokens;
  if (typeof maxTokens === "number") inferenceConfig.maxTokens = maxTokens;
  if (typeof params.temperature === "number")
    inferenceConfig.temperature = params.temperature;
  if (typeof params.top_p === "number") inferenceConfig.topP = params.top_p;
  if (typeof params.stop === "string") inferenceConfig.stopSequences = [params.stop];
  else if (Array.isArray(params.stop) && params.stop.length > 0)
    inferenceConfig.stopSequences = params.stop;

  const out: BedrockConverseParams = { modelId: toModelId, messages };
  if (systemBlocks.length > 0) out.system = systemBlocks;
  if (Object.keys(inferenceConfig).length > 0) out.inferenceConfig = inferenceConfig;
  return out;
}

/** OpenAI ChatCompletion params → Gemini generateContent params.
 *  Returns null when unsupported (streaming/tools/non-text). */
export function chatToGeminiParams(
  params: ChatCompletionParams,
  toModel: string,
): GeminiParams | null {
  if (chatHasUnsupportedShape(params)) return null;

  const contents: GeminiContent[] = [];
  let systemText = "";
  for (const m of params.messages) {
    const text = flattenTextContent(m.content);
    if (m.role === "system") {
      systemText = systemText ? `${systemText}\n\n${text}` : text;
      continue;
    }
    if (m.role !== "user" && m.role !== "assistant") continue;
    contents.push({
      role: m.role === "assistant" ? "model" : "user",
      parts: [{ text }],
    });
  }

  const config: GeminiParams["config"] = {};
  if (systemText) config.systemInstruction = { parts: [{ text: systemText }] };
  const maxTokens = params.max_tokens ?? params.max_completion_tokens;
  if (typeof maxTokens === "number") config.maxOutputTokens = maxTokens;
  if (typeof params.temperature === "number") config.temperature = params.temperature;
  if (typeof params.top_p === "number") config.topP = params.top_p;
  if (typeof params.stop === "string") config.stopSequences = [params.stop];
  else if (Array.isArray(params.stop) && params.stop.length > 0)
    config.stopSequences = params.stop;

  const out: GeminiParams = { model: toModel, contents };
  if (Object.keys(config).length > 0) out.config = config;
  return out;
}

/** OpenAI ChatCompletion params → Mistral chat.complete params.
 *  Returns null when unsupported (streaming/tools/non-text). */
export function chatToMistralParams(
  params: ChatCompletionParams,
  toModel: string,
): MistralChatParams | null {
  if (chatHasUnsupportedShape(params)) return null;
  const messages: MistralMessage[] = params.messages
    .filter((m) => m.role !== "tool")
    .map((m) => ({ role: m.role as MistralMessage["role"], content: flattenTextContent(m.content) }));
  const out: MistralChatParams = { model: toModel, messages };
  const max = params.max_tokens ?? params.max_completion_tokens;
  if (typeof max === "number") out.maxTokens = max;
  if (typeof params.temperature === "number") out.temperature = params.temperature;
  if (typeof params.top_p === "number") out.topP = params.top_p;
  if (params.stop !== undefined) out.stop = params.stop;
  return out;
}
