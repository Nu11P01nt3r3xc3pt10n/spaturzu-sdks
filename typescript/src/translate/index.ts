// Cross-provider translation, organised by source provider.
//
// Each source-provider file owns:
//   • params translators FROM this provider TO each target provider
//   • response translators FROM each target provider TO this provider's
//     response shape
//
// 5 providers × 4 other-providers = 20 directional pairs. Plans 3/4/5
// add the remaining 18; openai↔anthropic ship today.

// ─── shared shape types ──────────────────────────────────────────────────────

export type ChatTextPart = { type: "text"; text: string };
export type ChatImagePart = { type: "image_url"; image_url: { url: string } };
export type ChatContentPart = ChatTextPart | ChatImagePart;

export type ChatMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content: string | ChatContentPart[];
  name?: string;
  tool_call_id?: string;
};

export type ChatCompletionParams = {
  model: string;
  messages: ChatMessage[];
  max_tokens?: number;
  max_completion_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string | string[];
  stream?: boolean;
  tools?: unknown[];
  tool_choice?: unknown;
  response_format?: unknown;
};

export type ChatCompletionResponse = {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: { role: "assistant"; content: string };
    finish_reason: "stop" | "length" | "tool_calls" | "content_filter" | null;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
};

export type AnthropicTextBlock = { type: "text"; text: string };

export type AnthropicMessage = {
  role: "user" | "assistant";
  content: string | AnthropicTextBlock[];
};

export type AnthropicMessagesParams = {
  model: string;
  messages: AnthropicMessage[];
  system?: string | AnthropicTextBlock[];
  max_tokens: number;
  temperature?: number;
  top_p?: number;
  stop_sequences?: string[];
  stream?: boolean;
};

export type AnthropicMessageResponse = {
  id: string;
  type: "message";
  role: "assistant";
  content: AnthropicTextBlock[];
  model: string;
  stop_reason: "end_turn" | "max_tokens" | "stop_sequence" | "tool_use" | null;
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
};

// ─── Bedrock Converse shapes ─────────────────────────────────────────────────

export type BedrockTextBlock = { text: string };
export type BedrockContentBlock = BedrockTextBlock; // v1: text only.

export type BedrockMessage = {
  role: "user" | "assistant";
  content: BedrockContentBlock[];
};

export type BedrockConverseParams = {
  modelId: string;
  messages: BedrockMessage[];
  system?: BedrockTextBlock[];
  inferenceConfig?: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    stopSequences?: string[];
  };
  toolConfig?: unknown;
  additionalModelRequestFields?: unknown;
};

export type BedrockConverseResponse = {
  output: { message: { role: "assistant"; content: BedrockContentBlock[] } };
  stopReason:
    | "end_turn"
    | "tool_use"
    | "max_tokens"
    | "stop_sequence"
    | "guardrail_intervened"
    | "content_filtered";
  usage: {
    inputTokens: number;
    outputTokens: number;
    totalTokens?: number;
    cacheReadInputTokenCount?: number;
    cacheWriteInputTokenCount?: number;
  };
  metrics?: { latencyMs?: number };
};

// ─── Gemini (google-genai) shapes ────────────────────────────────────────────


export type GeminiPart = { text: string };
export type GeminiContent = { role: "user" | "model"; parts: GeminiPart[] };

export type GeminiParams = {
  model: string;
  contents: GeminiContent[];
  config?: {
    systemInstruction?: { parts: GeminiPart[] } | GeminiPart[];
    temperature?: number;
    topP?: number;
    maxOutputTokens?: number;
    stopSequences?: string[];
  };
};

export type GeminiUsageMetadata = {
  promptTokenCount?: number;
  candidatesTokenCount?: number;
  cachedContentTokenCount?: number;
  totalTokenCount?: number;
};

export type GeminiResponse = {
  candidates: Array<{
    content: GeminiContent;
    finishReason?: "STOP" | "MAX_TOKENS" | "SAFETY" | "RECITATION" | "OTHER";
  }>;
  usageMetadata?: GeminiUsageMetadata;
};

// ─── Mistral (mistralai) shapes ──────────────────────────────────────────────

// Mistral's chat.complete uses an OpenAI-compatible message shape with
// minor field naming differences. We model the v1 surface only.

export type MistralMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  tool_call_id?: string;
};

export type MistralChatParams = {
  model: string;
  messages: MistralMessage[];
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  stop?: string | string[];
  tools?: unknown[];
  responseFormat?: unknown;
};

export type MistralChatResponse = {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: { role: "assistant"; content: string };
    finish_reason: "stop" | "length" | "tool_calls" | "model_length" | "error" | null;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
};

// ─── per-source re-exports ───────────────────────────────────────────────────

export {
  chatToAnthropicParams,
  chatResponseToAnthropic,
  chatToBedrockParams,
  chatToGeminiParams,
  chatToMistralParams,
} from "./openai.js";

export {
  anthropicParamsToChat,
  anthropicResponseToChat,
  anthropicToBedrockParams,
  anthropicToGeminiParams,
  anthropicToMistralParams,
} from "./anthropic.js";

export {
  bedrockToChatParams,
  chatResponseFromBedrock,
  bedrockToAnthropicParams,
  anthropicResponseFromBedrock,
  bedrockToGeminiParams,
  bedrockToMistralParams,
} from "./bedrock.js";

export {
  bedrockResponseToChat,
  bedrockResponseToAnthropic,
} from "./bedrock-to-others.js";

// gemini-source surfaces
export {
  geminiToChatParams,
  chatResponseFromGemini,
  geminiToAnthropicParams,
  anthropicResponseFromGemini,
  geminiToBedrockParams,
  bedrockResponseFromGemini,
  geminiToMistralParams,
} from "./gemini.js";

export {
  geminiResponseToChat,
  geminiResponseToAnthropic,
  geminiResponseToBedrock,
} from "./gemini-to-others.js";

// mistral-source surfaces (Plan 5 — completes the 5×5 matrix)
export {
  mistralToChatParams,
  chatResponseFromMistral,
  mistralToAnthropicParams,
  anthropicResponseFromMistral,
  mistralToBedrockParams,
  bedrockResponseFromMistral,
  mistralToGeminiParams,
  geminiResponseFromMistral,
} from "./mistral.js";

export {
  mistralResponseToChat,
  mistralResponseToAnthropic,
  mistralResponseToBedrock,
  mistralResponseToGemini,
} from "./mistral-to-others.js";
