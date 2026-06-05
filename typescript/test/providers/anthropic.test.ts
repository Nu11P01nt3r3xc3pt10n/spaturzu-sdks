import { describe, it, expect } from "vitest";
import RealAnthropic from "@anthropic-ai/sdk";
import AnthropicDefault, { Anthropic } from "../../src/providers/anthropic.js";

describe("@spaturzu/sdk/anthropic drop-in", () => {
  it("exports the wrapped subclass as default and named", () => {
    expect(AnthropicDefault).toBe(Anthropic);
  });
  it("constructs an instrumented client preserving instanceof", () => {
    const client = new Anthropic({ apiKey: "sk-ant-test" });
    expect(client).toBeInstanceOf(RealAnthropic);
    expect(typeof client.withAgent).toBe("function");
    expect(client.messages.create).toBeTypeOf("function");
  });
});
