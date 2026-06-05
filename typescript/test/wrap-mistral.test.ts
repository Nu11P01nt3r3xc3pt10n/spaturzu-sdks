import { describe, it, expect } from "vitest";
import { wrapMistral } from "../src/mistral.js";
import { runInFrame } from "../src/context.js";
import type { Logger } from "../src/logger.js";
import {
  fakeMistral,
  fakeOpenAI,
  asAsyncIterable,
  mistralError,
} from "./helpers/fake-clients.js";
import { fakeLogger } from "./helpers/fake-logger.js";

function asLogger(fake: ReturnType<typeof fakeLogger>): Logger {
  return fake as unknown as Logger;
}

describe("wrapMistral: chat.complete (non-streaming)", () => {
  it("logs provider=mistral, model, usage from response.usage", async () => {
    const logger = fakeLogger();
    const client = fakeMistral({
      ok: {
        id: "cmpl-1",
        object: "chat.completion",
        created: 0,
        model: "mistral-large-latest",
        choices: [
          { index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
      },
    });
    const wrapped = wrapMistral(client, asLogger(logger));
    await runInFrame("agent-a", async () => {
      await wrapped.chat.complete({
        model: "mistral-large-latest",
        messages: [{ role: "user", content: "hi" }],
      });
    });
    expect(logger.entries[0]).toMatchObject({
      provider: "mistral",
      model: "mistral-large-latest",
      status: 200,
      promptTokens: 10,
      completionTokens: 20,
      usageSource: "provider",
      agentName: "agent-a",
    });
    // Mistral has no prompt-cache; cachedInputTokens must not appear.
    expect((logger.entries[0] as any).cachedInputTokens).toBeUndefined();
  });

  it("logs error with status from err.statusCode", async () => {
    const logger = fakeLogger();
    const err = mistralError(429);
    const client = fakeMistral({ err });
    const wrapped = wrapMistral(client, asLogger(logger));
    await expect(
      wrapped.chat.complete({
        model: "mistral-large-latest",
        messages: [{ role: "user", content: "hi" }],
      }),
    ).rejects.toBe(err);
    expect(logger.entries[0]).toMatchObject({ status: 429 });
  });
});

describe("wrapMistral: chat.stream", () => {
  it("reads usage from chunk.data.usage (not chunk.usage)", async () => {
    const logger = fakeLogger();
    const stream = asAsyncIterable([
      { data: { choices: [{ delta: { content: "Hello" } }] } },
      { data: { choices: [{ delta: { content: " world" } }] } },
      { data: { choices: [], usage: { prompt_tokens: 5, completion_tokens: 2 } } },
    ]);
    const client = fakeMistral({ ok: undefined }, { stream });
    const wrapped = wrapMistral(client, asLogger(logger));
    const result = await wrapped.chat.stream({
      model: "mistral-large-latest",
      messages: [{ role: "user", content: "hi" }],
    });
    for await (const _ of result as AsyncIterable<unknown>) {
      // drain
    }
    expect(logger.entries[0]).toMatchObject({
      provider: "mistral",
      promptTokens: 5,
      completionTokens: 2,
      usageSource: "provider",
    });
  });
});

describe("wrapMistral: fallback to openai on retryable error", () => {
  it("walks the chain and translates response back to Mistral shape", async () => {
    const logger = fakeLogger();
    const primary = fakeMistral({ err: mistralError(503) });
    const secondary = fakeOpenAI({
      ok: {
        usage: { prompt_tokens: 4, completion_tokens: 6, total_tokens: 10 },
        choices: [{ message: { role: "assistant", content: "hi from gpt" }, finish_reason: "stop" }],
      },
    });
    const wrapped = wrapMistral(primary, asLogger(logger), undefined, null, "throw", [
      { provider: "openai", client: secondary, model: "gpt-4o" },
    ]);
    const resp = (await wrapped.chat.complete({
      model: "mistral-large-latest",
      messages: [{ role: "user", content: "hi" }],
    })) as { choices: Array<{ message: { content: string } }> };
    expect(resp.choices[0]!.message.content).toBe("hi from gpt");
    expect(logger.entries).toHaveLength(2);
    expect(logger.entries[0]).toMatchObject({ provider: "mistral", status: 503 });
    expect(logger.entries[1]).toMatchObject({
      provider: "openai",
      tags: expect.objectContaining({ via: "fallback" }),
    });
  });
});
