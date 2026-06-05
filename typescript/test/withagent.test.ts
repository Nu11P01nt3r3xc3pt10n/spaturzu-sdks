import { describe, it, expect } from "vitest";
import { wrapOpenAI } from "../src/openai.js";
import { runInFrame } from "../src/context.js";
import type { Logger } from "../src/logger.js";
import { fakeOpenAI, asAsyncIterable } from "./helpers/fake-clients.js";
import { fakeLogger } from "./helpers/fake-logger.js";

const asLogger = (f: ReturnType<typeof fakeLogger>) => f as unknown as Logger;

describe("withAgent: non-streaming, no ambient run()", () => {
  it("tags the call with the agent name and a fresh runId", async () => {
    const logger = fakeLogger();
    const client = fakeOpenAI({ ok: { usage: { prompt_tokens: 1, completion_tokens: 1 } } });
    const wrapped = wrapOpenAI(client, asLogger(logger));
    await wrapped.withAgent("writer").chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(logger.entries).toHaveLength(1);
    expect(logger.entries[0]).toMatchObject({
      provider: "openai",
      agentName: "writer",
      agentPath: ["writer"],
    });
    expect(typeof logger.entries[0]!.runId).toBe("string");
  });
});

describe("withAgent: nests under an ambient run()", () => {
  it("extends the parent agent path", async () => {
    const logger = fakeLogger();
    const client = fakeOpenAI({ ok: { usage: { prompt_tokens: 1, completion_tokens: 1 } } });
    const wrapped = wrapOpenAI(client, asLogger(logger));
    await runInFrame("planner", async () => {
      await wrapped.withAgent("writer").chat.completions.create({
        model: "gpt-4o",
        messages: [{ role: "user", content: "hi" }],
      });
    });
    expect(logger.entries[0]).toMatchObject({ agentPath: ["planner", "writer"] });
  });
});

describe("withAgent: streaming, iterated OUTSIDE any frame", () => {
  it("still attributes to the agent (frame captured at call-time)", async () => {
    const logger = fakeLogger();
    const client = fakeOpenAI({
      stream: asAsyncIterable([
        { choices: [{ delta: { content: "hi" } }] },
        { usage: { prompt_tokens: 3, completion_tokens: 4 } },
      ]),
    });
    const wrapped = wrapOpenAI(client, asLogger(logger));
    const stream = await wrapped.withAgent("streamer").chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
      stream: true,
    });
    for await (const _ of stream) { /* drain */ }
    expect(logger.entries).toHaveLength(1);
    expect(logger.entries[0]).toMatchObject({
      agentName: "streamer",
      agentPath: ["streamer"],
      promptTokens: 3,
      completionTokens: 4,
    });
  });
});

describe("withAgent: concurrent calls do not cross-contaminate", () => {
  it("each interleaved call keeps its own agent", async () => {
    const logger = fakeLogger();
    const client = fakeOpenAI({ ok: { usage: { prompt_tokens: 1, completion_tokens: 1 } } });
    const wrapped = wrapOpenAI(client, asLogger(logger));
    await Promise.all([
      wrapped.withAgent("a").chat.completions.create({ model: "m", messages: [] }),
      wrapped.withAgent("b").chat.completions.create({ model: "m", messages: [] }),
    ]);
    const agents = logger.entries.map((e) => e.agentName).sort();
    expect(agents).toEqual(["a", "b"]);
  });
});
