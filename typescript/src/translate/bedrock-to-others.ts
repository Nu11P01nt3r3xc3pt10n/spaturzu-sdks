// Bedrock-response-to-other-shape adapters. Called by the dispatcher
// when bedrock is the TARGET of a fallback.

import type {
  AnthropicMessageResponse,
  BedrockConverseResponse,
  ChatCompletionResponse,
} from "./index.js";

function flattenBedrockText(resp: BedrockConverseResponse): string {
  const blocks = resp.output?.message?.content ?? [];
  return blocks
    .map((b) => (typeof b.text === "string" ? b.text : ""))
    .filter(Boolean)
    .join("");
}

/** Bedrock Converse response → OpenAI ChatCompletion response. */
export function bedrockResponseToChat(
  response: BedrockConverseResponse,
  callerModel: string,
): ChatCompletionResponse {
  const text = flattenBedrockText(response);
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
          switch (response.stopReason) {
            case "max_tokens":
              return "length";
            case "tool_use":
              return "tool_calls";
            case "content_filtered":
            case "guardrail_intervened":
              return "content_filter";
            default:
              return "stop";
          }
        })(),
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

/** Bedrock Converse response → Anthropic Messages response. */
export function bedrockResponseToAnthropic(
  response: BedrockConverseResponse,
  callerModel: string,
): AnthropicMessageResponse {
  const text = flattenBedrockText(response);
  return {
    id: "",
    type: "message",
    role: "assistant",
    content: [{ type: "text", text }],
    model: callerModel,
    stop_reason: (() => {
      switch (response.stopReason) {
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
      input_tokens: response.usage?.inputTokens ?? 0,
      output_tokens: response.usage?.outputTokens ?? 0,
    },
  };
}
