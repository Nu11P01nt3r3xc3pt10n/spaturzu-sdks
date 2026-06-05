import { describe, it, expect } from "vitest";
import {
  mistralToChatParams,
  mistralToAnthropicParams,
  mistralToBedrockParams,
  mistralToGeminiParams,
  mistralResponseToChat,
  mistralResponseToAnthropic,
  mistralResponseToBedrock,
  mistralResponseToGemini,
  chatToMistralParams,
  anthropicToMistralParams,
  bedrockToMistralParams,
  geminiToMistralParams,
  chatResponseFromMistral,
  anthropicResponseFromMistral,
  bedrockResponseFromMistral,
  geminiResponseFromMistral,
} from "../../src/translate/index.js";

describe("mistralToChatParams (near-identity)", () => {
  it("preserves message shape; renames maxTokens → max_tokens, topP → top_p", () => {
    const out = mistralToChatParams(
      {
        model: "mistral-large-latest",
        messages: [
          { role: "system", content: "be brief" },
          { role: "user", content: "hi" },
        ],
        maxTokens: 200,
        temperature: 0.7,
        topP: 0.9,
        stop: ["STOP"],
      },
      "gpt-4o",
    );
    expect(out?.messages).toHaveLength(2);
    expect(out?.max_tokens).toBe(200);
    expect(out?.temperature).toBe(0.7);
    expect(out?.top_p).toBe(0.9);
    expect(out?.stop).toEqual(["STOP"]);
  });

  it("returns null when tools present", () => {
    const out = mistralToChatParams(
      {
        model: "mistral-large-latest",
        messages: [{ role: "user", content: "hi" }],
        tools: [{ type: "function" }],
      },
      "gpt-4o",
    );
    expect(out).toBeNull();
  });

  it("returns null when responseFormat present", () => {
    const out = mistralToChatParams(
      {
        model: "x",
        messages: [{ role: "user", content: "hi" }],
        responseFormat: { type: "json_object" },
      },
      "gpt-4o",
    );
    expect(out).toBeNull();
  });
});

describe("mistralToAnthropicParams", () => {
  it("pulls system messages out into top-level system field", () => {
    const out = mistralToAnthropicParams(
      {
        model: "x",
        messages: [
          { role: "system", content: "be brief" },
          { role: "user", content: "hi" },
        ],
        maxTokens: 100,
      },
      "claude-3-5-haiku",
    );
    expect(out?.system).toBe("be brief");
    expect(out?.max_tokens).toBe(100);
    expect(out?.messages).toEqual([{ role: "user", content: "hi" }]);
  });

  it("defaults max_tokens to 1024", () => {
    const out = mistralToAnthropicParams(
      {
        model: "x",
        messages: [{ role: "user", content: "hi" }],
      },
      "claude",
    );
    expect(out?.max_tokens).toBe(1024);
  });
});

describe("mistralToBedrockParams", () => {
  it("converts system to bedrock text-block array", () => {
    const out = mistralToBedrockParams(
      {
        model: "x",
        messages: [
          { role: "system", content: "be brief" },
          { role: "user", content: "hi" },
        ],
        maxTokens: 200,
      },
      "anthropic.claude-3-5-sonnet-20241022-v2:0",
    );
    expect(out?.system).toEqual([{ text: "be brief" }]);
    expect(out?.messages[0]).toEqual({ role: "user", content: [{ text: "hi" }] });
    expect(out?.inferenceConfig?.maxTokens).toBe(200);
  });
});

describe("mistralToGeminiParams", () => {
  it("maps assistant → model role; pulls system to systemInstruction", () => {
    const out = mistralToGeminiParams(
      {
        model: "x",
        messages: [
          { role: "system", content: "be brief" },
          { role: "user", content: "hi" },
          { role: "assistant", content: "hello" },
        ],
        maxTokens: 100,
      },
      "gemini-2.5-pro",
    );
    expect(out?.config?.systemInstruction).toEqual({ parts: [{ text: "be brief" }] });
    expect(out?.contents).toEqual([
      { role: "user", parts: [{ text: "hi" }] },
      { role: "model", parts: [{ text: "hello" }] },
    ]);
    expect(out?.config?.maxOutputTokens).toBe(100);
  });
});

describe("mistralResponseToChat (near-identity)", () => {
  it("returns chat shape with model_length finish mapped to length", () => {
    const out = mistralResponseToChat(
      {
        id: "msg-1",
        object: "chat.completion",
        created: 0,
        model: "mistral-large-latest",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "hi back" },
            finish_reason: "model_length",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
      },
      "gpt-4o",
    );
    expect(out.model).toBe("gpt-4o");
    expect(out.choices[0]!.finish_reason).toBe("length");
    expect(out.usage).toEqual({ prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 });
  });
});

describe("mistralResponseToAnthropic", () => {
  it("wraps content in text block; maps tool_calls → tool_use", () => {
    const out = mistralResponseToAnthropic(
      {
        id: "msg-1",
        object: "chat.completion",
        created: 0,
        model: "mistral",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "tool" },
            finish_reason: "tool_calls",
          },
        ],
        usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
      },
      "claude",
    );
    expect(out.content).toEqual([{ type: "text", text: "tool" }]);
    expect(out.stop_reason).toBe("tool_use");
    expect(out.usage).toEqual({ input_tokens: 3, output_tokens: 1 });
  });
});

describe("mistralResponseToBedrock", () => {
  it("wraps content in bedrock content-block array", () => {
    const out = mistralResponseToBedrock(
      {
        id: "msg-1",
        object: "chat.completion",
        created: 0,
        model: "mistral",
        choices: [
          { index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
      "anthropic.claude-3-5-haiku-20241022-v1:0",
    );
    expect(out.output.message.content).toEqual([{ text: "hi" }]);
    expect(out.stopReason).toBe("end_turn");
    expect(out.usage).toEqual({ inputTokens: 1, outputTokens: 1, totalTokens: 2 });
  });
});

describe("mistralResponseToGemini", () => {
  it("maps to Gemini shape with model role and finishReason MAX_TOKENS", () => {
    const out = mistralResponseToGemini(
      {
        id: "msg-1",
        object: "chat.completion",
        created: 0,
        model: "mistral",
        choices: [
          { index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "length" },
        ],
        usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
      },
      "gemini-2.5-pro",
    );
    expect(out.candidates[0]!.content.parts[0]!.text).toBe("hi");
    expect(out.candidates[0]!.finishReason).toBe("MAX_TOKENS");
    expect(out.usageMetadata).toEqual({
      promptTokenCount: 4,
      candidatesTokenCount: 2,
      totalTokenCount: 6,
    });
  });
});

describe("chatToMistralParams", () => {
  it("renames max_tokens → maxTokens, top_p → topP", () => {
    const out = chatToMistralParams(
      {
        model: "gpt-4o",
        messages: [{ role: "user", content: "hi" }],
        max_tokens: 500,
        temperature: 0.5,
        top_p: 0.9,
      },
      "mistral-large-latest",
    );
    expect(out?.maxTokens).toBe(500);
    expect(out?.temperature).toBe(0.5);
    expect(out?.topP).toBe(0.9);
  });
});

describe("anthropicToMistralParams", () => {
  it("converts system string to system message; uses maxTokens", () => {
    const out = anthropicToMistralParams(
      {
        model: "claude",
        messages: [{ role: "user", content: "hi" }],
        max_tokens: 100,
        system: "be brief",
      },
      "mistral-large-latest",
    );
    expect(out?.messages[0]).toEqual({ role: "system", content: "be brief" });
    expect(out?.maxTokens).toBe(100);
  });
});

describe("bedrockToMistralParams", () => {
  it("converts bedrock system array to flat system message", () => {
    const out = bedrockToMistralParams(
      {
        modelId: "x",
        messages: [{ role: "user", content: [{ text: "hi" }] }],
        system: [{ text: "be brief" }],
        inferenceConfig: { maxTokens: 100 },
      },
      "mistral-large-latest",
    );
    expect(out?.messages[0]).toEqual({ role: "system", content: "be brief" });
    expect(out?.maxTokens).toBe(100);
  });
});

describe("geminiToMistralParams", () => {
  it("maps model role → assistant; pulls systemInstruction", () => {
    const out = geminiToMistralParams(
      {
        model: "gemini-2.5-pro",
        contents: [
          { role: "user", parts: [{ text: "hi" }] },
          { role: "model", parts: [{ text: "hello" }] },
        ],
        config: {
          systemInstruction: { parts: [{ text: "be brief" }] },
          maxOutputTokens: 100,
        },
      },
      "mistral-large-latest",
    );
    expect(out?.messages[0]).toEqual({ role: "system", content: "be brief" });
    expect(out?.messages[1]).toEqual({ role: "user", content: "hi" });
    expect(out?.messages[2]).toEqual({ role: "assistant", content: "hello" });
    expect(out?.maxTokens).toBe(100);
  });
});

describe("response-from-mistral translators", () => {
  it("chatResponseFromMistral preserves caller model + tokens", () => {
    const out = chatResponseFromMistral(
      {
        id: "cmpl-1",
        object: "chat.completion",
        created: 0,
        model: "gpt-4o",
        choices: [
          { index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
      },
      "mistral-large-latest",
    );
    expect(out.model).toBe("mistral-large-latest");
    expect(out.choices[0]!.message.content).toBe("hi");
    expect(out.usage).toEqual({ prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 });
  });
});
