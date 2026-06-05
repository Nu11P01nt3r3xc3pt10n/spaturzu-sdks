import { describe, it, expect } from "vitest";
import { wrapAnthropic } from "../src/anthropic.js";
import { runInFrame } from "../src/context.js";
import type { Logger } from "../src/logger.js";
import {
  fakeAnthropic,
  fakeOpenAI,
  asAsyncIterable,
  anthropicError,
} from "./helpers/fake-clients.js";
import { fakeLogger } from "./helpers/fake-logger.js";

function asLogger(fake: ReturnType<typeof fakeLogger>): Logger {
  return fake as unknown as Logger;
}

describe("wrapAnthropic: non-streaming success", () => {
  it("logs provider=anthropic, model, usage, cache_read mapped to cachedInputTokens", async () => {
    const logger = fakeLogger();
    const client = fakeAnthropic({
      ok: {
        id: "msg-1",
        type: "message",
        usage: {
          input_tokens: 10,
          output_tokens: 5,
          cache_read_input_tokens: 3,
          cache_creation_input_tokens: 2,
        },
        content: [{ type: "text", text: "hi" }],
        stop_reason: "end_turn",
      },
    });
    const wrapped = wrapAnthropic(client, asLogger(logger));
    await runInFrame("agent", async () => {
      await wrapped.messages.create({
        model: "claude-3-5-haiku-20241022",
        messages: [{ role: "user", content: "hi" }],
        max_tokens: 100,
      });
    });
    expect(logger.entries[0]).toMatchObject({
      provider: "anthropic",
      model: "claude-3-5-haiku-20241022",
      status: 200,
      promptTokens: 10,
      completionTokens: 5,
      cachedInputTokens: 3,
      usageSource: "provider",
    });
    // cache_creation is the WRITE side and must NOT appear on the wire.
    expect((logger.entries[0] as any).cacheCreationInputTokens).toBeUndefined();
  });
});

describe("wrapAnthropic: error path", () => {
  it("logs status from err.status and re-throws", async () => {
    const logger = fakeLogger();
    const err = anthropicError(529, "OverloadedError");
    const client = fakeAnthropic({ err });
    const wrapped = wrapAnthropic(client, asLogger(logger));
    await expect(
      wrapped.messages.create({
        model: "claude",
        messages: [{ role: "user", content: "hi" }],
        max_tokens: 10,
      }),
    ).rejects.toBe(err);
    expect(logger.entries[0]).toMatchObject({ status: 529 });
  });
});

describe("wrapAnthropic: streaming", () => {
  it("accumulates input_tokens from message_start and cumulative output_tokens from message_delta", async () => {
    const logger = fakeLogger();
    const client = fakeAnthropic({
      stream: asAsyncIterable([
        {
          type: "message_start",
          message: { usage: { input_tokens: 12, output_tokens: 0, cache_read_input_tokens: 4 } },
        },
        { type: "content_block_delta", delta: { type: "text_delta", text: "Hi" } },
        { type: "content_block_delta", delta: { type: "text_delta", text: " there" } },
        { type: "message_delta", usage: { output_tokens: 8 } },
      ]),
    });
    const wrapped = wrapAnthropic(client, asLogger(logger));
    const stream = await wrapped.messages.create({
      model: "claude",
      messages: [{ role: "user", content: "hi" }],
      max_tokens: 100,
      stream: true,
    });
    for await (const _ of stream as AsyncIterable<any>) {
      // drain
    }
    expect(logger.entries).toHaveLength(1);
    expect(logger.entries[0]).toMatchObject({
      status: 200,
      promptTokens: 12,
      completionTokens: 8,
      cachedInputTokens: 4,
      usageSource: "provider",
    });
  });
});

describe("wrapAnthropic: fallback walking", () => {
  it("walks to openai on retryable primary error", async () => {
    const logger = fakeLogger();
    const primary = fakeAnthropic({ err: anthropicError(503) });
    const secondary = fakeOpenAI({
      ok: {
        id: "cmpl-1",
        object: "chat.completion",
        created: 0,
        model: "gpt-4o",
        choices: [{ index: 0, message: { role: "assistant", content: "hi from gpt" }, finish_reason: "stop" }],
        usage: { prompt_tokens: 2, completion_tokens: 6, total_tokens: 8 },
      },
    });
    const wrapped = wrapAnthropic(primary, asLogger(logger), undefined, null, "throw", [
      { provider: "openai", client: secondary, model: "gpt-4o" },
    ]);
    const resp = (await wrapped.messages.create({
      model: "claude-3-5-haiku-20241022",
      messages: [{ role: "user", content: "hi" }],
      max_tokens: 100,
    })) as { content: Array<{ text: string }> };
    expect(resp.content[0]!.text).toBe("hi from gpt");
    expect(logger.entries).toHaveLength(2);
    expect(logger.entries[1]).toMatchObject({
      provider: "openai",
      model: "gpt-4o",
      promptTokens: 2,
      completionTokens: 6,
    });
  });
});
