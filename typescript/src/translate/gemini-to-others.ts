// Gemini-response-to-other-shape adapters. Called by the dispatcher
// when gemini is the TARGET of a fallback.

import type {
  AnthropicMessageResponse,
  BedrockConverseResponse,
  ChatCompletionResponse,
  GeminiResponse,
} from "./index.js";

function flattenGeminiResponseText(resp: GeminiResponse): string {
  const parts = resp.candidates?.[0]?.content?.parts ?? [];
  return parts
    .map((p) => (typeof p.text === "string" ? p.text : ""))
    .filter(Boolean)
    .join("");
}

export function geminiResponseToChat(
  response: GeminiResponse,
  callerModel: string,
): ChatCompletionResponse {
  const text = flattenGeminiResponseText(response);
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
        finish_reason: (() => {
          switch (finish) {
            case "MAX_TOKENS":
              return "length";
            case "SAFETY":
            case "RECITATION":
              return "content_filter";
            default:
              return "stop";
          }
        })(),
      },
    ],
    usage: {
      prompt_tokens: u.promptTokenCount ?? 0,
      completion_tokens: u.candidatesTokenCount ?? 0,
      total_tokens:
        u.totalTokenCount ??
        (u.promptTokenCount ?? 0) + (u.candidatesTokenCount ?? 0),
    },
  };
}

export function geminiResponseToAnthropic(
  response: GeminiResponse,
  callerModel: string,
): AnthropicMessageResponse {
  const text = flattenGeminiResponseText(response);
  const finish = response.candidates?.[0]?.finishReason;
  const u = response.usageMetadata ?? {};
  return {
    id: "",
    type: "message",
    role: "assistant",
    content: [{ type: "text", text }],
    model: callerModel,
    stop_reason: (() => {
      switch (finish) {
        case "MAX_TOKENS":
          return "max_tokens";
        default:
          return "end_turn";
      }
    })(),
    usage: {
      input_tokens: u.promptTokenCount ?? 0,
      output_tokens: u.candidatesTokenCount ?? 0,
    },
  };
}

export function geminiResponseToBedrock(
  response: GeminiResponse,
  callerModelId: string,
): BedrockConverseResponse {
  const text = flattenGeminiResponseText(response);
  const finish = response.candidates?.[0]?.finishReason;
  const u = response.usageMetadata ?? {};
  return {
    output: { message: { role: "assistant", content: [{ text }] } },
    stopReason: (() => {
      switch (finish) {
        case "MAX_TOKENS":
          return "max_tokens";
        case "SAFETY":
        case "RECITATION":
          return "content_filtered";
        default:
          return "end_turn";
      }
    })(),
    usage: {
      inputTokens: u.promptTokenCount ?? 0,
      outputTokens: u.candidatesTokenCount ?? 0,
      totalTokens:
        u.totalTokenCount ??
        (u.promptTokenCount ?? 0) + (u.candidatesTokenCount ?? 0),
    },
  };
}
