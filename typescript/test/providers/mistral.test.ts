import { describe, it, expect } from "vitest";
import { Mistral as RealMistral } from "@mistralai/mistralai";
import { Mistral } from "../../src/providers/mistral.js";

describe("@spaturzu/sdk/mistral drop-in", () => {
  it("constructs an instrumented client preserving instanceof", () => {
    const client = new Mistral({ apiKey: "test" });
    expect(client).toBeInstanceOf(RealMistral);
    expect(typeof client.withAgent).toBe("function");
    expect(client.chat.complete).toBeTypeOf("function");
    expect(client.chat.stream).toBeTypeOf("function");
  });
});
