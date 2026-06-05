import { describe, it, expect } from "vitest";
import {
  bedrockToChatParams,
  bedrockToAnthropicParams,
  bedrockResponseToChat,
  bedrockResponseToAnthropic,
  chatToBedrockParams,
  anthropicToBedrockParams,
  chatResponseFromBedrock,
  anthropicResponseFromBedrock,
} from "../../src/translate/index.js";

describe("bedrockToChatParams", () => {
  it("pulls top-level system out into a system message", () => {
    const out = bedrockToChatParams(
      {
        modelId: "anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages: [{ role: "user", content: [{ text: "hi" }] }],
        system: [{ text: "be brief" }, { text: "use lists" }],
        inferenceConfig: { maxTokens: 200, temperature: 0.7, topP: 0.9, stopSequences: ["STOP"] },
      },
      "gpt-4o",
    );
    expect(out?.messages[0]).toEqual({ role: "system", content: "be brief\n\nuse lists" });
    expect(out?.messages[1]).toEqual({ role: "user", content: "hi" });
    expect(out?.max_tokens).toBe(200);
    expect(out?.temperature).toBe(0.7);
    expect(out?.top_p).toBe(0.9);
    expect(out?.stop).toEqual(["STOP"]);
  });

  it("returns null when toolConfig is present", () => {
    const out = bedrockToChatParams(
      {
        modelId: "anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages: [{ role: "user", content: [{ text: "hi" }] }],
        toolConfig: { tools: [] },
      },
      "gpt-4o",
    );
    expect(out).toBeNull();
  });
});

describe("bedrockToAnthropicParams", () => {
  it("defaults max_tokens to 1024 if not in inferenceConfig", () => {
    const out = bedrockToAnthropicParams(
      {
        modelId: "anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages: [{ role: "user", content: [{ text: "hi" }] }],
      },
      "claude-3-5-haiku",
    );
    expect(out?.max_tokens).toBe(1024);
  });

  it("converts system array → joined string", () => {
    const out = bedrockToAnthropicParams(
      {
        modelId: "x",
        messages: [{ role: "user", content: [{ text: "hi" }] }],
        system: [{ text: "be brief" }, { text: "use lists" }],
      },
      "claude",
    );
    expect(out?.system).toBe("be brief\n\nuse lists");
  });
});

describe("bedrockResponseToChat", () => {
  it("maps usage and stopReason to chat shape", () => {
    const out = bedrockResponseToChat(
      {
        output: { message: { role: "assistant", content: [{ text: "hello" }] } },
        stopReason: "end_turn",
        usage: { inputTokens: 4, outputTokens: 3, totalTokens: 7 },
      },
      "gpt-4o",
    );
    expect(out.model).toBe("gpt-4o");
    expect(out.choices[0]!.message.content).toBe("hello");
    expect(out.choices[0]!.finish_reason).toBe("stop");
    expect(out.usage).toEqual({ prompt_tokens: 4, completion_tokens: 3, total_tokens: 7 });
  });

  it("maps stopReason 'max_tokens' to finish_reason 'length'", () => {
    const out = bedrockResponseToChat(
      {
        output: { message: { role: "assistant", content: [] } },
        stopReason: "max_tokens",
        usage: { inputTokens: 0, outputTokens: 0 },
      },
      "gpt",
    );
    expect(out.choices[0]!.finish_reason).toBe("length");
  });
});

describe("bedrockResponseToAnthropic", () => {
  it("wraps content text in a single text block", () => {
    const out = bedrockResponseToAnthropic(
      {
        output: { message: { role: "assistant", content: [{ text: "hi" }, { text: " there" }] } },
        stopReason: "end_turn",
        usage: { inputTokens: 3, outputTokens: 5 },
      },
      "claude-3-5-haiku",
    );
    expect(out.content).toEqual([{ type: "text", text: "hi there" }]);
    expect(out.usage).toEqual({ input_tokens: 3, output_tokens: 5 });
  });
});

describe("chatToBedrockParams", () => {
  it("pulls system messages out into top-level array", () => {
    const out = chatToBedrockParams(
      {
        model: "gpt-4o",
        messages: [
          { role: "system", content: "be brief" },
          { role: "user", content: "hi" },
        ],
        max_tokens: 500,
      },
      "anthropic.claude-3-5-sonnet-20241022-v2:0",
    );
    expect(out?.modelId).toBe("anthropic.claude-3-5-sonnet-20241022-v2:0");
    expect(out?.system).toEqual([{ text: "be brief" }]);
    expect(out?.messages[0]).toEqual({ role: "user", content: [{ text: "hi" }] });
    expect(out?.inferenceConfig?.maxTokens).toBe(500);
  });

  it("returns null for streaming", () => {
    expect(
      chatToBedrockParams(
        { model: "gpt-4o", messages: [{ role: "user", content: "hi" }], stream: true },
        "anthropic.claude-3-5-haiku-20241022-v1:0",
      ),
    ).toBeNull();
  });
});

describe("anthropicToBedrockParams", () => {
  it("converts string system to single-block array", () => {
    const out = anthropicToBedrockParams(
      {
        model: "claude",
        messages: [{ role: "user", content: "hi" }],
        max_tokens: 100,
        system: "be brief",
      },
      "anthropic.claude-3-5-sonnet-20241022-v2:0",
    );
    expect(out?.system).toEqual([{ text: "be brief" }]);
    expect(out?.inferenceConfig?.maxTokens).toBe(100);
  });
});

describe("chatResponseFromBedrock", () => {
  it("translates a chat-shape response back into bedrock shape", () => {
    const out = chatResponseFromBedrock(
      {
        id: "cmpl-1",
        object: "chat.completion",
        created: 0,
        model: "gpt-4o",
        choices: [
          { index: 0, message: { role: "assistant", content: "hi back" }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
      },
      "anthropic.claude-3-5-sonnet-20241022-v2:0",
    );
    expect(out.output.message.content).toEqual([{ text: "hi back" }]);
    expect(out.stopReason).toBe("end_turn");
    expect(out.usage).toEqual({ inputTokens: 5, outputTokens: 2, totalTokens: 7 });
  });
});

describe("anthropicResponseFromBedrock", () => {
  it("translates an anthropic-shape response back into bedrock shape", () => {
    const out = anthropicResponseFromBedrock(
      {
        id: "msg-1",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "hello" }],
        model: "claude",
        stop_reason: "end_turn",
        usage: { input_tokens: 4, output_tokens: 3 },
      },
      "anthropic.claude-3-5-sonnet-20241022-v2:0",
    );
    expect(out.output.message.content).toEqual([{ text: "hello" }]);
    expect(out.stopReason).toBe("end_turn");
    expect(out.usage).toEqual({ inputTokens: 4, outputTokens: 3, totalTokens: 7 });
  });
});
