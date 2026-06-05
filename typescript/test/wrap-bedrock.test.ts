import { describe, it, expect } from "vitest";
import { wrapBedrock } from "../src/bedrock.js";
import { runInFrame } from "../src/context.js";
import type { Logger } from "../src/logger.js";
import {
  fakeBedrock,
  fakeOpenAI,
  asAsyncIterable,
  bedrockError,
} from "./helpers/fake-clients.js";
import { fakeLogger } from "./helpers/fake-logger.js";

function asLogger(fake: ReturnType<typeof fakeLogger>): Logger {
  return fake as unknown as Logger;
}

describe("wrapBedrock: converse (non-streaming)", () => {
  it("logs provider=bedrock, model=modelId, usage from result.usage", async () => {
    const logger = fakeLogger();
    const client = fakeBedrock({
      ok: {
        output: { message: { role: "assistant", content: [{ text: "hi" }] } },
        stopReason: "end_turn",
        usage: { inputTokens: 10, outputTokens: 20, cacheReadInputTokenCount: 4 },
      },
    });
    const wrapped = wrapBedrock(client, asLogger(logger));
    await runInFrame("agent-a", async () => {
      await wrapped.converse({
        modelId: "anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages: [{ role: "user", content: [{ text: "hi" }] }],
      });
    });
    expect(logger.entries[0]).toMatchObject({
      provider: "bedrock",
      model: "anthropic.claude-3-5-sonnet-20241022-v2:0",
      status: 200,
      promptTokens: 10,
      completionTokens: 20,
      cachedInputTokens: 4,
      usageSource: "provider",
      agentName: "agent-a",
    });
  });

  it("logs error with status from $metadata.httpStatusCode", async () => {
    const logger = fakeLogger();
    const err = bedrockError(429);
    const client = fakeBedrock({ err });
    const wrapped = wrapBedrock(client, asLogger(logger));
    await expect(
      wrapped.converse({
        modelId: "x",
        messages: [{ role: "user", content: [{ text: "hi" }] }],
      }),
    ).rejects.toBe(err);
    expect(logger.entries[0]).toMatchObject({ status: 429 });
  });
});

describe("wrapBedrock: converseStream", () => {
  it("observes the terminal metadata event for usage", async () => {
    const logger = fakeLogger();
    const stream = asAsyncIterable([
      { messageStart: { role: "assistant" } },
      { contentBlockDelta: { delta: { text: "Hello" } } },
      { contentBlockDelta: { delta: { text: " world" } } },
      { messageStop: { stopReason: "end_turn" } },
      {
        metadata: {
          usage: { inputTokens: 5, outputTokens: 2 },
          metrics: { latencyMs: 50 },
        },
      },
    ]);
    const client = fakeBedrock({ ok: { stream } }, { ok: { stream } });
    const wrapped = wrapBedrock(client, asLogger(logger));
    const resp = (await wrapped.converseStream({
      modelId: "x",
      messages: [{ role: "user", content: [{ text: "hi" }] }],
    })) as { stream: AsyncIterable<any> };
    const events: any[] = [];
    for await (const e of resp.stream) events.push(e);
    expect(events).toHaveLength(5);
    expect(logger.entries[0]).toMatchObject({
      provider: "bedrock",
      promptTokens: 5,
      completionTokens: 2,
      usageSource: "provider",
    });
  });
});

describe("wrapBedrock: fallback to openai on retryable error", () => {
  it("walks the chain and logs the openai attempt with via=fallback", async () => {
    const logger = fakeLogger();
    const primary = fakeBedrock({ err: bedrockError(503, "ServiceUnavailableException") });
    const secondary = fakeOpenAI({
      ok: {
        usage: { prompt_tokens: 3, completion_tokens: 5 },
        choices: [{ message: { content: "hi from gpt" }, finish_reason: "stop" }],
      },
    });
    const wrapped = wrapBedrock(primary, asLogger(logger), undefined, null, "throw", [
      { provider: "openai", client: secondary, model: "gpt-4o" },
    ]);
    const resp = (await wrapped.converse({
      modelId: "anthropic.claude-3-5-sonnet-20241022-v2:0",
      messages: [{ role: "user", content: [{ text: "hi" }] }],
    })) as { output: { message: { content: Array<{ text: string }> } } };
    // Response translated back to Bedrock shape (chat_response_from_bedrock).
    expect(resp.output.message.content[0]!.text).toBe("hi from gpt");
    expect(logger.entries).toHaveLength(2);
    expect(logger.entries[0]).toMatchObject({ provider: "bedrock", status: 503 });
    expect(logger.entries[1]).toMatchObject({
      provider: "openai",
      model: "gpt-4o",
      promptTokens: 3,
      completionTokens: 5,
      tags: expect.objectContaining({ via: "fallback" }),
    });
  });
});
