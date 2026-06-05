import { describe, it, expect } from "vitest";
import { wrapGemini } from "../src/gemini.js";
import { runInFrame } from "../src/context.js";
import type { Logger } from "../src/logger.js";
import {
  fakeGemini,
  fakeOpenAI,
  asAsyncIterable,
  geminiError,
} from "./helpers/fake-clients.js";
import { fakeLogger } from "./helpers/fake-logger.js";

function asLogger(fake: ReturnType<typeof fakeLogger>): Logger {
  return fake as unknown as Logger;
}

describe("wrapGemini: generateContent", () => {
  it("logs provider=gemini, model, usage from response.usageMetadata", async () => {
    const logger = fakeLogger();
    const client = fakeGemini({
      ok: {
        candidates: [
          { content: { role: "model", parts: [{ text: "hi" }] }, finishReason: "STOP" },
        ],
        usageMetadata: {
          promptTokenCount: 10,
          candidatesTokenCount: 20,
          cachedContentTokenCount: 4,
        },
      },
    });
    const wrapped = wrapGemini(client, asLogger(logger));
    await runInFrame("agent-a", async () => {
      await wrapped.models.generateContent({
        model: "gemini-2.5-pro",
        contents: [{ role: "user", parts: [{ text: "hi" }] }],
      });
    });
    expect(logger.entries[0]).toMatchObject({
      provider: "gemini",
      model: "gemini-2.5-pro",
      status: 200,
      promptTokens: 10,
      completionTokens: 20,
      cachedInputTokens: 4,
      usageSource: "provider",
      agentName: "agent-a",
    });
  });

  it("logs error with status from err.status", async () => {
    const logger = fakeLogger();
    const err = geminiError(429);
    const client = fakeGemini({ err });
    const wrapped = wrapGemini(client, asLogger(logger));
    await expect(
      wrapped.models.generateContent({
        model: "gemini-2.5-pro",
        contents: [{ role: "user", parts: [{ text: "hi" }] }],
      }),
    ).rejects.toBe(err);
    expect(logger.entries[0]).toMatchObject({ status: 429 });
  });
});

describe("wrapGemini: generateContentStream", () => {
  it("keeps last-seen usageMetadata across cumulative chunks", async () => {
    const logger = fakeLogger();
    const stream = asAsyncIterable([
      {
        candidates: [{ content: { role: "model", parts: [{ text: "Hello" }] } }],
        usageMetadata: { promptTokenCount: 3, candidatesTokenCount: 1 },
      },
      {
        candidates: [{ content: { role: "model", parts: [{ text: " world" }] } }],
        usageMetadata: { promptTokenCount: 3, candidatesTokenCount: 3 },
      },
    ]);
    const client = fakeGemini({ ok: undefined }, { stream });
    const wrapped = wrapGemini(client, asLogger(logger));
    const result = await wrapped.models.generateContentStream({
      model: "gemini-2.5-pro",
      contents: [{ role: "user", parts: [{ text: "hi" }] }],
    });
    for await (const _ of result as AsyncIterable<unknown>) {
      // drain
    }
    expect(logger.entries[0]).toMatchObject({
      promptTokens: 3,
      completionTokens: 3, // last-seen, NOT summed
    });
  });
});

describe("wrapGemini: fallback to openai on retryable error", () => {
  it("walks the chain and translates response back to Gemini shape", async () => {
    const logger = fakeLogger();
    const primary = fakeGemini({ err: geminiError(503) });
    const secondary = fakeOpenAI({
      ok: {
        usage: { prompt_tokens: 4, completion_tokens: 6 },
        choices: [{ message: { content: "hi from gpt" }, finish_reason: "stop" }],
      },
    });
    const wrapped = wrapGemini(primary, asLogger(logger), undefined, null, "throw", [
      { provider: "openai", client: secondary, model: "gpt-4o" },
    ]);
    const resp = (await wrapped.models.generateContent({
      model: "gemini-2.5-pro",
      contents: [{ role: "user", parts: [{ text: "hi" }] }],
    })) as { candidates: Array<{ content: { parts: Array<{ text: string }> } }> };
    expect(resp.candidates[0]!.content.parts[0]!.text).toBe("hi from gpt");
    expect(logger.entries).toHaveLength(2);
    expect(logger.entries[0]).toMatchObject({ provider: "gemini", status: 503 });
    expect(logger.entries[1]).toMatchObject({
      provider: "openai",
      tags: expect.objectContaining({ via: "fallback" }),
    });
  });
});
