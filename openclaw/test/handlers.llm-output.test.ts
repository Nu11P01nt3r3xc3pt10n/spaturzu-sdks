import { describe, it, expect, vi } from "vitest";
import { makeLlmOutputHandler, type SpaturzuLike } from "../src/handlers.js";
import { parseConfig } from "../src/config.js";

type LlmOutputEvent = Parameters<
  ReturnType<typeof makeLlmOutputHandler>
>[0];
type AgentCtx = Parameters<ReturnType<typeof makeLlmOutputHandler>>[1];

function fakeSpaturzu(): SpaturzuLike & { calls: unknown[] } {
  const calls: unknown[] = [];
  return {
    calls,
    report: (call) => calls.push(call),
    checkBudget: vi.fn().mockResolvedValue(undefined),
  };
}

describe("makeLlmOutputHandler", () => {
  it("maps an llm_output event onto Spaturzu.report() with the expected shape", () => {
    const sp = fakeSpaturzu();
    const cfg = parseConfig({
      apiKey: "spa_test",
      tags: { env: "dev" },
    });
    const handler = makeLlmOutputHandler(sp, cfg);

    const event: LlmOutputEvent = {
      runId: "openclaw-run-1",
      sessionId: "sess-1",
      provider: "anthropic",
      model: "claude-haiku-4-5",
      resolvedRef: "anthropic/claude-haiku-4-5",
      harnessId: "claude-code",
      assistantTexts: ["hello"],
      usage: { input: 12, output: 34, cacheRead: 7 },
    };
    const ctx: AgentCtx = {
      runId: "openclaw-run-1",
      agentId: "writer",
      sessionId: "sess-1",
      channelId: "discord:guild-1",
    };

    handler(event, ctx);

    expect(sp.calls).toHaveLength(1);
    expect(sp.calls[0]).toMatchObject({
      provider: "anthropic",
      model: "claude-haiku-4-5",
      agentName: "openclaw/writer",
      agentPath: ["openclaw/writer"],
      sessionId: "sess-1",
      promptTokens: 12,
      completionTokens: 34,
      cachedInputTokens: 7,
      usageSource: "provider",
      status: 200,
      tags: {
        env: "dev",
        channel: "discord:guild-1",
        harness: "claude-code",
      },
    });
    // runId is normalized to a UUID (v5 derivation from "openclaw-run-1")
    expect((sp.calls[0] as { runId: string }).runId).toMatch(
      /^[0-9a-f-]{36}$/,
    );
  });

  it("omits token fields when usage is missing", () => {
    const sp = fakeSpaturzu();
    const cfg = parseConfig({ apiKey: "spa_test" });
    const handler = makeLlmOutputHandler(sp, cfg);

    handler(
      {
        runId: "r",
        sessionId: "s",
        provider: "openai",
        model: "gpt-4o",
        assistantTexts: [],
      },
      { agentId: "writer" },
    );

    const entry = sp.calls[0] as Record<string, unknown>;
    expect(entry.promptTokens).toBeUndefined();
    expect(entry.completionTokens).toBeUndefined();
    expect(entry.cachedInputTokens).toBeUndefined();
  });

  it("never throws when sp.report throws (host must not crash)", () => {
    const sp: SpaturzuLike = {
      report: () => {
        throw new Error("boom");
      },
      checkBudget: vi.fn(),
    };
    const cfg = parseConfig({ apiKey: "spa_test" });
    const handler = makeLlmOutputHandler(sp, cfg);
    expect(() =>
      handler(
        {
          runId: "r",
          sessionId: "s",
          provider: "openai",
          model: "gpt-4o",
          assistantTexts: [],
        },
        {},
      ),
    ).not.toThrow();
  });
});
