import { describe, it, expect } from "vitest";
import { GoogleGenAI as RealGoogleGenAI } from "@google/genai";
import { GoogleGenAI } from "../../src/providers/google.js";

describe("@spaturzu/sdk/google drop-in", () => {
  it("constructs an instrumented GenAI client preserving instanceof", () => {
    const client = new GoogleGenAI({ apiKey: "test" });
    expect(client).toBeInstanceOf(RealGoogleGenAI);
    expect(typeof client.withAgent).toBe("function");
    expect(client.models.generateContent).toBeTypeOf("function");
  });
});
