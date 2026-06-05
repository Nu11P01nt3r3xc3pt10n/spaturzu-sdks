// Mistral-response-to-other-shape adapters. Called by the dispatcher
// when mistral is the TARGET of a fallback.

import type {
  AnthropicMessageResponse,
  BedrockConverseResponse,
  ChatCompletionResponse,
  GeminiResponse,
  MistralChatResponse,
} from "./index.js";

export function mistralResponseToChat(
  response: MistralChatResponse,
  callerModel: string,
): ChatCompletionResponse {
  // Near-identity: same shape with caller-model preserved.
  return {
    id: response.id,
    object: "chat.completion",
    created: response.created,
    model: callerModel,
    choices: response.choices.map((c) => ({
      index: c.index,
      message: c.message,
      finish_reason: ((): ChatCompletionResponse["choices"][number]["finish_reason"] => {
        switch (c.finish_reason) {
          case "length":
          case "model_length":
            return "length";
          case "tool_calls":
            return "tool_calls";
          default:
            return "stop";
        }
      })(),
    })),
    usage: response.usage,
  };
}

export function mistralResponseToAnthropic(
  response: MistralChatResponse,
  callerModel: string,
): AnthropicMessageResponse {
  const choice = response.choices[0];
  const text = choice?.message.content ?? "";
  return {
    id: response.id,
    type: "message",
    role: "assistant",
    content: [{ type: "text", text }],
    model: callerModel,
    stop_reason: ((): AnthropicMessageResponse["stop_reason"] => {
      switch (choice?.finish_reason) {
        case "length":
        case "model_length":
          return "max_tokens";
        case "tool_calls":
          return "tool_use";
        default:
          return "end_turn";
      }
    })(),
    usage: {
      input_tokens: response.usage.prompt_tokens,
      output_tokens: response.usage.completion_tokens,
    },
  };
}

export function mistralResponseToBedrock(
  response: MistralChatResponse,
  _callerModelId: string,
): BedrockConverseResponse {
  const choice = response.choices[0];
  const text = choice?.message.content ?? "";
  return {
    output: { message: { role: "assistant", content: [{ text }] } },
    stopReason: ((): BedrockConverseResponse["stopReason"] => {
      switch (choice?.finish_reason) {
        case "length":
        case "model_length":
          return "max_tokens";
        case "tool_calls":
          return "tool_use";
        default:
          return "end_turn";
      }
    })(),
    usage: {
      inputTokens: response.usage.prompt_tokens,
      outputTokens: response.usage.completion_tokens,
      totalTokens: response.usage.total_tokens,
    },
  };
}

export function mistralResponseToGemini(
  response: MistralChatResponse,
  _callerModel: string,
): GeminiResponse {
  const choice = response.choices[0];
  const text = choice?.message.content ?? "";
  return {
    candidates: [
      {
        content: { role: "model", parts: [{ text }] },
        finishReason:
          choice?.finish_reason === "length" || choice?.finish_reason === "model_length"
            ? "MAX_TOKENS"
            : "STOP",
      },
    ],
    usageMetadata: {
      promptTokenCount: response.usage.prompt_tokens,
      candidatesTokenCount: response.usage.completion_tokens,
      totalTokenCount: response.usage.total_tokens,
    },
  };
}
