import { describe, it, expect } from "vitest";
import RealOpenAI from "openai";
import OpenAIDefault, { OpenAI } from "../../src/providers/openai.js";

describe("@spaturzu/sdk/openai drop-in", () => {
  it("exports the wrapped subclass as default and named", () => {
    expect(OpenAIDefault).toBe(OpenAI);
  });

  it("constructs an instrumented client that still passes instanceof", () => {
    const client = new OpenAI({ apiKey: "sk-test" });
    // The proxy must not break the prototype chain — `instanceof` still holds.
    expect(client).toBeInstanceOf(RealOpenAI);
    // .withAgent is typed on the drop-in client (declared) + present at runtime.
    expect(typeof client.withAgent).toBe("function");
    expect(client.chat.completions.create).toBeTypeOf("function");
  });
});
