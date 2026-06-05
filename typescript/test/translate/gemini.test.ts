import { describe, it, expect } from "vitest";
import {
  geminiToChatParams,
  geminiToAnthropicParams,
  geminiToBedrockParams,
  geminiResponseToChat,
  geminiResponseToAnthropic,
  geminiResponseToBedrock,
  chatToGeminiParams,
  anthropicToGeminiParams,
  bedrockToGeminiParams,
  chatResponseFromGemini,
  anthropicResponseFromGemini,
  bedrockResponseFromGemini,
} from "../../src/translate/index.js";

describe("geminiToChatParams", () => {
  it("converts contents (with 'model' role) into chat messages", () => {
    const out = geminiToChatParams(
      {
        model: "gemini-2.5-pro",
        contents: [
          { role: "user", parts: [{ text: "hi" }] },
          { role: "model", parts: [{ text: "hello" }] },
        ],
        config: {
          systemInstruction: { parts: [{ text: "be brief" }] },
          maxOutputTokens: 200,
        },
      },
      "gpt-4o",
    );
    expect(out?.messages[0]).toEqual({ role: "system", content: "be brief" });
    expect(out?.messages[1]).toEqual({ role: "user", content: "hi" });
    expect(out?.messages[2]).toEqual({ role: "assistant", content: "hello" });
    expect(out?.max_tokens).toBe(200);
  });
});

describe("geminiToAnthropicParams", () => {
  it("defaults max_tokens to 1024", () => {
    const out = geminiToAnthropicParams(
      {
        model: "gemini-2.5-pro",
        contents: [{ role: "user", parts: [{ text: "hi" }] }],
      },
      "claude-3-5-haiku",
    );
    expect(out?.max_tokens).toBe(1024);
  });
});

describe("geminiResponseToChat", () => {
  it("maps finishReason MAX_TOKENS → length", () => {
    const out = geminiResponseToChat(
      {
        candidates: [
          { content: { role: "model", parts: [{ text: "" }] }, finishReason: "MAX_TOKENS" },
        ],
        usageMetadata: { promptTokenCount: 0, candidatesTokenCount: 0 },
      },
      "gpt-4o",
    );
    expect(out.choices[0]!.finish_reason).toBe("length");
  });

  it("maps finishReason SAFETY → content_filter", () => {
    const out = geminiResponseToChat(
      {
        candidates: [
          { content: { role: "model", parts: [{ text: "" }] }, finishReason: "SAFETY" },
        ],
        usageMetadata: {},
      },
      "gpt",
    );
    expect(out.choices[0]!.finish_reason).toBe("content_filter");
  });
});

describe("chatToGeminiParams", () => {
  it("pulls system messages into config.systemInstruction", () => {
    const out = chatToGeminiParams(
      {
        model: "gpt-4o",
        messages: [
          { role: "system", content: "be brief" },
          { role: "user", content: "hi" },
        ],
        max_tokens: 500,
      },
      "gemini-2.5-pro",
    );
    expect(out?.config?.systemInstruction).toEqual({
      parts: [{ text: "be brief" }],
    });
    expect(out?.contents[0]).toEqual({ role: "user", parts: [{ text: "hi" }] });
    expect(out?.config?.maxOutputTokens).toBe(500);
  });

  it("maps assistant role → model", () => {
    const out = chatToGeminiParams(
      {
        model: "gpt-4o",
        messages: [
          { role: "user", content: "hi" },
          { role: "assistant", content: "hello" },
        ],
      },
      "gemini-2.5-pro",
    );
    expect(out?.contents[1]).toEqual({ role: "model", parts: [{ text: "hello" }] });
  });
});

describe("anthropicToGeminiParams", () => {
  it("converts string system to systemInstruction parts", () => {
    const out = anthropicToGeminiParams(
      {
        model: "claude",
        messages: [{ role: "user", content: "hi" }],
        max_tokens: 100,
        system: "be brief",
      },
      "gemini-2.5-pro",
    );
    expect(out?.config?.systemInstruction).toEqual({
      parts: [{ text: "be brief" }],
    });
    expect(out?.config?.maxOutputTokens).toBe(100);
  });
});

describe("bedrockToGeminiParams", () => {
  it("converts bedrock system array to systemInstruction", () => {
    const out = bedrockToGeminiParams(
      {
        modelId: "x",
        messages: [{ role: "user", content: [{ text: "hi" }] }],
        system: [{ text: "be brief" }],
        inferenceConfig: { maxTokens: 200 },
      },
      "gemini-2.5-pro",
    );
    expect(out?.config?.systemInstruction).toEqual({
      parts: [{ text: "be brief" }],
    });
    expect(out?.config?.maxOutputTokens).toBe(200);
  });

  it("maps bedrock assistant role → gemini model role", () => {
    const out = bedrockToGeminiParams(
      {
        modelId: "x",
        messages: [
          { role: "user", content: [{ text: "hi" }] },
          { role: "assistant", content: [{ text: "hello" }] },
        ],
      },
      "gemini-2.5-pro",
    );
    expect(out?.contents[0]).toEqual({ role: "user", parts: [{ text: "hi" }] });
    expect(out?.contents[1]).toEqual({ role: "model", parts: [{ text: "hello" }] });
  });
});

describe("response-from-gemini translators", () => {
  it("chatResponseFromGemini round-trips usage", () => {
    const out = chatResponseFromGemini(
      {
        id: "cmpl-1",
        object: "chat.completion",
        created: 0,
        model: "gpt-4o",
        choices: [
          { index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 4, completion_tokens: 3, total_tokens: 7 },
      },
      "gemini-2.5-pro",
    );
    expect(out.candidates[0]!.content.parts[0]!.text).toBe("hi");
    expect(out.usageMetadata).toEqual({
      promptTokenCount: 4,
      candidatesTokenCount: 3,
      totalTokenCount: 7,
    });
  });
});
