import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { Spaturzu } from "../src/index.js";

const okResponse = () => new Response("", { status: 200 });

function spyFetch() {
  return vi.spyOn(globalThis, "fetch") as unknown as ReturnType<typeof vi.fn>;
}

describe("Spaturzu.report()", () => {
  let fetchSpy: ReturnType<typeof spyFetch>;
  beforeEach(() => {
    fetchSpy = spyFetch();
    fetchSpy.mockResolvedValue(okResponse());
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("POSTs a wire-shaped log entry built from the caller's fields", async () => {
    const sp = new Spaturzu({
      baseURL: "https://gw.example",
      apiKey: "spa_test",
      tags: { env: "dev" },
    });

    sp.report({
      provider: "anthropic",
      model: "claude-haiku-4-5",
      agentName: "researcher",
      agentPath: ["researcher"],
      promptTokens: 12,
      completionTokens: 34,
      cachedInputTokens: 7,
      sessionId: "sess-1",
      tags: { channel: "discord" },
    });

    await sp.flush();

    expect(fetchSpy).toHaveBeenCalledOnce();
    const [url, init] = fetchSpy.mock.calls[0]!;
    expect(String(url)).toBe("https://gw.example/v1/logs");
    expect((init as RequestInit).method).toBe("POST");
    expect((init as RequestInit).headers).toMatchObject({
      "content-type": "application/json",
      "x-spaturzu-key": "spa_test",
    });
    const body = JSON.parse(String((init as RequestInit).body));
    expect(body).toMatchObject({
      provider: "anthropic",
      model: "claude-haiku-4-5",
      agentName: "researcher",
      agentPath: ["researcher"],
      promptTokens: 12,
      completionTokens: 34,
      cachedInputTokens: 7,
      sessionId: "sess-1",
      status: 200,
      tags: { env: "dev", channel: "discord" },
    });
  });

  it("inherits runId / agentPath / frame-tags from the current run() frame", async () => {
    const sp = new Spaturzu({ baseURL: "https://gw.example" });

    await sp.run("orchestrator", { tags: { team: "search" } }, async () => {
      sp.report({ provider: "openai", model: "gpt-4o" });
    });

    await sp.flush();

    const body = JSON.parse(String((fetchSpy.mock.calls[0]![1] as RequestInit).body));
    expect(body.runId).toMatch(/^[0-9a-f-]{36}$/);
    expect(body.agentName).toBe("orchestrator");
    expect(body.agentPath).toEqual(["orchestrator"]);
    expect(body.tags).toMatchObject({ team: "search" });
  });

  it("explicit caller fields override frame inheritance", async () => {
    const sp = new Spaturzu({ baseURL: "https://gw.example" });

    await sp.run("orchestrator", async () => {
      sp.report({
        provider: "openai",
        model: "gpt-4o",
        agentName: "explicit",
        agentPath: ["explicit"],
        runId: "00000000-0000-4000-8000-000000000001",
      });
    });

    await sp.flush();

    const body = JSON.parse(String((fetchSpy.mock.calls[0]![1] as RequestInit).body));
    expect(body.runId).toBe("00000000-0000-4000-8000-000000000001");
    expect(body.agentName).toBe("explicit");
    expect(body.agentPath).toEqual(["explicit"]);
  });
});

import { __resetDefaultForTests } from "../src/default.js";
import { configure, report as topLevelReport, flush as topLevelFlush } from "../src/index.js";

describe("top-level report() forwards to default singleton", () => {
  let fetchSpy: ReturnType<typeof spyFetch>;
  beforeEach(() => {
    __resetDefaultForTests();
    fetchSpy = spyFetch();
    fetchSpy.mockResolvedValue(okResponse());
  });
  afterEach(() => {
    __resetDefaultForTests();
    vi.restoreAllMocks();
  });

  it("uses configure()'d baseURL/apiKey and posts the entry", async () => {
    configure({ baseURL: "https://gw.example", apiKey: "spa_top" });
    topLevelReport({ provider: "openai", model: "gpt-4o" });
    await topLevelFlush();

    expect(fetchSpy).toHaveBeenCalledOnce();
    const [url, init] = fetchSpy.mock.calls[0]!;
    expect(String(url)).toBe("https://gw.example/v1/logs");
    expect((init as RequestInit).headers).toMatchObject({
      "x-spaturzu-key": "spa_top",
    });
  });
});
