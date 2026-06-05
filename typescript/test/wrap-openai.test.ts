import { describe, it, expect } from "vitest";
import { wrapOpenAI } from "../src/openai.js";
import { runInFrame } from "../src/context.js";
import type { Logger } from "../src/logger.js";
import {
  fakeOpenAI,
  fakeAnthropic,
  asAsyncIterable,
  openaiError,
} from "./helpers/fake-clients.js";
import { fakeLogger } from "./helpers/fake-logger.js";

function asLogger(fake: ReturnType<typeof fakeLogger>): Logger {
  // Structural compatibility — wrap only calls .log()/.flush().
  return fake as unknown as Logger;
}

describe("wrapOpenAI: non-streaming success", () => {
  it("logs provider, model, usage, frame attribution", async () => {
    const logger = fakeLogger();
    const client = fakeOpenAI({
      ok: {
        usage: {
          prompt_tokens: 10,
          completion_tokens: 20,
          prompt_tokens_details: { cached_tokens: 4 },
        },
        choices: [{ message: { content: "hi" }, finish_reason: "stop" }],
      },
    });
    const wrapped = wrapOpenAI(client, asLogger(logger));
    await runInFrame("agent-a", async () => {
      await wrapped.chat.completions.create({
        model: "gpt-4o",
        messages: [{ role: "user", content: "hi" }],
      });
    });
    expect(logger.entries).toHaveLength(1);
    expect(logger.entries[0]).toMatchObject({
      provider: "openai",
      model: "gpt-4o",
      status: 200,
      promptTokens: 10,
      completionTokens: 20,
      cachedInputTokens: 4,
      usageSource: "provider",
      agentName: "agent-a",
      agentPath: ["agent-a"],
    });
    expect(typeof logger.entries[0]!.id).toBe("string");
    expect(typeof logger.entries[0]!.latencyMs).toBe("number");
  });

  it("merges process tags with frame tags (frame wins)", async () => {
    const logger = fakeLogger();
    const client = fakeOpenAI({ ok: { usage: { prompt_tokens: 1, completion_tokens: 1 } } });
    const wrapped = wrapOpenAI(
      client,
      asLogger(logger),
      () => ({ env: "prod", region: "us-east-1" }),
    );
    await runInFrame(
      "agent",
      async () => {
        await wrapped.chat.completions.create({
          model: "gpt-4o",
          messages: [{ role: "user", content: "hi" }],
        });
      },
      { env: "staging" }, // frame tag overrides process tag
    );
    expect(logger.entries[0]!.tags).toEqual({
      env: "staging",
      region: "us-east-1",
    });
  });
});

describe("wrapOpenAI: error path", () => {
  it("logs an error entry with status mapped, then re-throws", async () => {
    const logger = fakeLogger();
    const err = openaiError(429);
    const client = fakeOpenAI({ err });
    const wrapped = wrapOpenAI(client, asLogger(logger));
    await expect(
      runInFrame("agent", async () => {
        await wrapped.chat.completions.create({
          model: "gpt-4o",
          messages: [{ role: "user", content: "hi" }],
        });
      }),
    ).rejects.toBe(err);
    expect(logger.entries).toHaveLength(1);
    expect(logger.entries[0]).toMatchObject({ status: 429 });
    expect(logger.entries[0]!.promptTokens).toBeUndefined();
  });

  it("clamps unknown error status to 500", async () => {
    const logger = fakeLogger();
    const client = fakeOpenAI({ err: new Error("mystery") });
    const wrapped = wrapOpenAI(client, asLogger(logger));
    await expect(
      wrapped.chat.completions.create({
        model: "gpt-4o",
        messages: [{ role: "user", content: "hi" }],
      }),
    ).rejects.toBeDefined();
    expect(logger.entries[0]!.status).toBe(500);
  });
});

describe("wrapOpenAI: streaming", () => {
  it("yields chunks unchanged and logs once on stream end with usage", async () => {
    const logger = fakeLogger();
    const client = fakeOpenAI({
      stream: asAsyncIterable([
        { choices: [{ delta: { content: "Hello" } }] },
        { choices: [{ delta: { content: " world" } }] },
        {
          choices: [],
          usage: { prompt_tokens: 5, completion_tokens: 2 },
        },
      ]),
    });
    const wrapped = wrapOpenAI(client, asLogger(logger));
    const stream = await wrapped.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
      stream: true,
    });
    const chunks: any[] = [];
    for await (const c of stream as AsyncIterable<any>) chunks.push(c);
    expect(chunks).toHaveLength(3);
    expect(logger.entries).toHaveLength(1);
    expect(logger.entries[0]).toMatchObject({
      status: 200,
      promptTokens: 5,
      completionTokens: 2,
      usageSource: "provider",
    });
  });

  it("auto-injects include_usage when the caller didn't set it", async () => {
    const logger = fakeLogger();
    const client = fakeOpenAI({
      stream: asAsyncIterable([{ choices: [], usage: { prompt_tokens: 1, completion_tokens: 1 } }]),
    });
    const wrapped = wrapOpenAI(client, asLogger(logger));
    await wrapped.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
      stream: true,
    });
    const args = client.__create.mock.calls[0]![0];
    expect(args.stream_options).toEqual({ include_usage: true });
  });
});

describe("wrapOpenAI: fallback walking", () => {
  it("walks to an anthropic target on retryable primary error", async () => {
    const logger = fakeLogger();
    const primary = fakeOpenAI({ err: openaiError(503) });
    const secondary = fakeAnthropic({
      ok: {
        id: "msg-1",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "hi from claude" }],
        model: "claude-3-5-haiku",
        stop_reason: "end_turn",
        usage: { input_tokens: 3, output_tokens: 5 },
      },
    });
    const wrapped = wrapOpenAI(primary, asLogger(logger), undefined, null, "throw", [
      { provider: "anthropic", client: secondary, model: "claude-3-5-haiku-20241022" },
    ]);
    const resp = (await wrapped.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
    })) as { choices: Array<{ message: { content: string } }> };
    expect(resp.choices[0]!.message.content).toBe("hi from claude");
    // Two log entries: primary error + fallback success
    expect(logger.entries).toHaveLength(2);
    expect(logger.entries[0]).toMatchObject({ provider: "openai", status: 503 });
    expect(logger.entries[1]).toMatchObject({
      provider: "anthropic",
      model: "claude-3-5-haiku-20241022",
      status: 200,
      promptTokens: 3,
      completionTokens: 5,
      tags: expect.objectContaining({ via: "fallback" }),
    });
  });

  it("does not walk the chain on non-retryable primary error (400)", async () => {
    const logger = fakeLogger();
    const primary = fakeOpenAI({ err: openaiError(400) });
    const secondary = fakeAnthropic({ ok: {} });
    const wrapped = wrapOpenAI(primary, asLogger(logger), undefined, null, "throw", [
      { provider: "anthropic", client: secondary, model: "claude" },
    ]);
    await expect(
      wrapped.chat.completions.create({
        model: "gpt-4o",
        messages: [{ role: "user", content: "hi" }],
      }),
    ).rejects.toBeDefined();
    expect(secondary.__create).not.toHaveBeenCalled();
    expect(logger.entries).toHaveLength(1);
  });

  it("bubbles the primary error when translator refuses (tools)", async () => {
    const logger = fakeLogger();
    const primary = fakeOpenAI({ err: openaiError(503) });
    const secondary = fakeAnthropic({ ok: {} });
    const wrapped = wrapOpenAI(primary, asLogger(logger), undefined, null, "throw", [
      { provider: "anthropic", client: secondary, model: "claude" },
    ]);
    // Note: streaming would not reach the fallback branch (different code
    // path); we use a non-stream call but spike the translator refusal by
    // passing `tools` (also refused).
    await expect(
      wrapped.chat.completions.create({
        model: "gpt-4o",
        messages: [{ role: "user", content: "hi" }],
        tools: [{ type: "function", function: { name: "x" } }],
      }),
    ).rejects.toBeDefined();
    expect(secondary.__create).not.toHaveBeenCalled();
  });
});
