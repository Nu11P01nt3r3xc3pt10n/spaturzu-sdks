import { describe, it, expect } from "vitest";
import {
  chatToAnthropicParams,
  anthropicResponseToChat,
  anthropicParamsToChat,
  chatResponseToAnthropic,
} from "../src/translate.js";

describe("chatToAnthropicParams", () => {
  it("pulls system messages out into the top-level system field", () => {
    const out = chatToAnthropicParams(
      {
        model: "gpt-4o",
        messages: [
          { role: "system", content: "be brief" },
          { role: "system", content: "use lists" },
          { role: "user", content: "hi" },
        ],
      },
      "claude-3-5-haiku-20241022",
    );
    expect(out?.system).toBe("be brief\n\nuse lists");
    expect(out?.messages).toEqual([{ role: "user", content: "hi" }]);
  });

  it("defaults max_tokens to 1024 when source didn't set one", () => {
    const out = chatToAnthropicParams(
      { model: "gpt-4o", messages: [{ role: "user", content: "hi" }] },
      "claude",
    );
    expect(out?.max_tokens).toBe(1024);
  });

  it("uses max_completion_tokens as a fallback for max_tokens", () => {
    const out = chatToAnthropicParams(
      {
        model: "gpt-4o",
        messages: [{ role: "user", content: "hi" }],
        max_completion_tokens: 500,
      },
      "claude",
    );
    expect(out?.max_tokens).toBe(500);
  });

  it("converts stop: string into stop_sequences", () => {
    const out = chatToAnthropicParams(
      {
        model: "gpt-4o",
        messages: [{ role: "user", content: "hi" }],
        stop: "STOP",
      },
      "claude",
    );
    expect(out?.stop_sequences).toEqual(["STOP"]);
  });

  it("returns null for streaming params", () => {
    expect(
      chatToAnthropicParams(
        {
          model: "gpt-4o",
          messages: [{ role: "user", content: "hi" }],
          stream: true,
        },
        "claude",
      ),
    ).toBeNull();
  });

  it("returns null for tools", () => {
    expect(
      chatToAnthropicParams(
        {
          model: "gpt-4o",
          messages: [{ role: "user", content: "hi" }],
          tools: [{ type: "function" }],
        },
        "claude",
      ),
    ).toBeNull();
  });

  it("returns null for tool messages", () => {
    expect(
      chatToAnthropicParams(
        {
          model: "gpt-4o",
          messages: [{ role: "tool", content: "x", tool_call_id: "id" }],
        },
        "claude",
      ),
    ).toBeNull();
  });

  it("returns null for non-text content parts", () => {
    expect(
      chatToAnthropicParams(
        {
          model: "gpt-4o",
          messages: [
            {
              role: "user",
              content: [{ type: "image_url", image_url: { url: "x" } }],
            },
          ],
        },
        "claude",
      ),
    ).toBeNull();
  });
});

describe("anthropicResponseToChat", () => {
  it("converts a vanilla response, preserving caller-model", () => {
    const out = anthropicResponseToChat(
      {
        id: "msg-1",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "hello" }],
        model: "claude-3-5-haiku",
        stop_reason: "end_turn",
        usage: { input_tokens: 4, output_tokens: 3 },
      },
      "gpt-4o",
    );
    expect(out.model).toBe("gpt-4o");
    expect(out.object).toBe("chat.completion");
    expect(out.choices[0]!.message.content).toBe("hello");
    expect(out.choices[0]!.finish_reason).toBe("stop");
    expect(out.usage).toEqual({
      prompt_tokens: 4,
      completion_tokens: 3,
      total_tokens: 7,
    });
  });

  it("maps stop_reason 'max_tokens' to finish_reason 'length'", () => {
    const out = anthropicResponseToChat(
      {
        id: "x",
        type: "message",
        role: "assistant",
        content: [],
        model: "claude",
        stop_reason: "max_tokens",
        usage: { input_tokens: 1, output_tokens: 1 },
      },
      "gpt",
    );
    expect(out.choices[0]!.finish_reason).toBe("length");
  });

  it("maps stop_reason 'tool_use' to finish_reason 'tool_calls'", () => {
    const out = anthropicResponseToChat(
      {
        id: "x",
        type: "message",
        role: "assistant",
        content: [],
        model: "claude",
        stop_reason: "tool_use",
        usage: { input_tokens: 1, output_tokens: 1 },
      },
      "gpt",
    );
    expect(out.choices[0]!.finish_reason).toBe("tool_calls");
  });
});

describe("anthropicParamsToChat", () => {
  it("pushes the system field as a system message", () => {
    const out = anthropicParamsToChat(
      {
        model: "claude",
        messages: [{ role: "user", content: "hi" }],
        max_tokens: 100,
        system: "be brief",
      },
      "gpt-4o",
    );
    expect(out?.messages[0]).toEqual({ role: "system", content: "be brief" });
    expect(out?.messages[1]).toEqual({ role: "user", content: "hi" });
    expect(out?.max_tokens).toBe(100);
  });

  it("returns null for streaming params", () => {
    expect(
      anthropicParamsToChat(
        {
          model: "claude",
          messages: [{ role: "user", content: "hi" }],
          max_tokens: 100,
          stream: true,
        },
        "gpt-4o",
      ),
    ).toBeNull();
  });
});

describe("chatResponseToAnthropic", () => {
  it("wraps content in a text block and preserves caller-model", () => {
    const out = chatResponseToAnthropic(
      {
        id: "cmpl-1",
        object: "chat.completion",
        created: 0,
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "hi back" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
      },
      "claude-3-5-haiku",
    );
    expect(out.model).toBe("claude-3-5-haiku");
    expect(out.content).toEqual([{ type: "text", text: "hi back" }]);
    expect(out.stop_reason).toBe("end_turn");
    expect(out.usage).toEqual({ input_tokens: 5, output_tokens: 2 });
  });

  it("maps finish_reason 'length' to stop_reason 'max_tokens'", () => {
    const out = chatResponseToAnthropic(
      {
        id: "x",
        object: "chat.completion",
        created: 0,
        model: "gpt",
        choices: [
          { index: 0, message: { role: "assistant", content: "" }, finish_reason: "length" },
        ],
        usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      },
      "claude",
    );
    expect(out.stop_reason).toBe("max_tokens");
  });
});
