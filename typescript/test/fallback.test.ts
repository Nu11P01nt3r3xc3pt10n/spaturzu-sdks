import { describe, it, expect } from "vitest";
import { isRetryableUpstreamError, tryFallback } from "../src/fallback.js";
import { fakeOpenAI, fakeAnthropic, openaiError } from "./helpers/fake-clients.js";

// isRetryableUpstreamError --------------------------------------------------

describe("isRetryableUpstreamError", () => {
  it("returns true for 429", () => {
    expect(isRetryableUpstreamError({ status: 429 })).toBe(true);
  });
  it("returns true for 5xx", () => {
    expect(isRetryableUpstreamError({ status: 503 })).toBe(true);
    expect(isRetryableUpstreamError({ status: 500 })).toBe(true);
  });
  it("returns true for 408", () => {
    expect(isRetryableUpstreamError({ status: 408 })).toBe(true);
  });
  it("returns false for 4xx other than 408/429", () => {
    expect(isRetryableUpstreamError({ status: 400 })).toBe(false);
    expect(isRetryableUpstreamError({ status: 401 })).toBe(false);
    expect(isRetryableUpstreamError({ status: 422 })).toBe(false);
  });
  it("returns true for connection error codes", () => {
    expect(isRetryableUpstreamError({ code: "ETIMEDOUT" })).toBe(true);
    expect(isRetryableUpstreamError({ code: "ECONNRESET" })).toBe(true);
    expect(isRetryableUpstreamError({ code: "ECONNREFUSED" })).toBe(true);
    expect(isRetryableUpstreamError({ code: "ENOTFOUND" })).toBe(true);
    expect(isRetryableUpstreamError({ code: "EAI_AGAIN" })).toBe(true);
  });
  it("returns true for known SDK error names", () => {
    expect(isRetryableUpstreamError({ name: "APIConnectionError" })).toBe(true);
    expect(isRetryableUpstreamError({ name: "APITimeoutError" })).toBe(true);
    expect(isRetryableUpstreamError({ name: "RateLimitError" })).toBe(true);
    expect(isRetryableUpstreamError({ name: "InternalServerError" })).toBe(true);
  });
  it("returns false for null / non-objects", () => {
    expect(isRetryableUpstreamError(null)).toBe(false);
    expect(isRetryableUpstreamError("string")).toBe(false);
    expect(isRetryableUpstreamError(undefined)).toBe(false);
  });
});

// tryFallback dispatcher (identity + cross-shape) ---------------------------

describe("tryFallback: identity (openai → openai)", () => {
  it("calls the target's chat.completions.create with the same params", async () => {
    const target = fakeOpenAI({
      ok: {
        usage: { prompt_tokens: 5, completion_tokens: 7 },
        choices: [{ message: { content: "hi" }, finish_reason: "stop" }],
      },
    });
    const outcome = await tryFallback({
      primaryShape: "openai",
      originalParams: {
        model: "gpt-4o",
        messages: [{ role: "user", content: "hi" }],
      },
      target: { provider: "openai", client: target, model: "gpt-4o-mini" },
    });
    expect(outcome.kind).toBe("success");
    if (outcome.kind !== "success") throw new Error("unreachable");
    expect(outcome.usage).toEqual({
      promptTokens: 5,
      completionTokens: 7,
      cachedInputTokens: undefined,
    });
    // Identity → model override applied; rest of params unchanged.
    const args = target.__create.mock.calls[0]![0];
    expect(args.model).toBe("gpt-4o-mini");
    expect(args.messages).toEqual([{ role: "user", content: "hi" }]);
  });
});

describe("tryFallback: cross-shape (openai → anthropic)", () => {
  it("translates params, calls messages.create, returns chat-shaped response", async () => {
    const anthroResp = {
      id: "msg-1",
      type: "message",
      role: "assistant",
      content: [{ type: "text", text: "hello" }],
      model: "claude-3-5-haiku-20241022",
      stop_reason: "end_turn",
      usage: { input_tokens: 4, output_tokens: 3 },
    };
    const target = fakeAnthropic({ ok: anthroResp });
    const outcome = await tryFallback({
      primaryShape: "openai",
      originalParams: {
        model: "gpt-4o",
        messages: [
          { role: "system", content: "be brief" },
          { role: "user", content: "hi" },
        ],
      },
      target: { provider: "anthropic", client: target, model: "claude-3-5-haiku-20241022" },
    });
    expect(outcome.kind).toBe("success");
    if (outcome.kind !== "success") throw new Error("unreachable");

    // Anthropic call args: system pulled out, max_tokens defaulted.
    const args = target.__create.mock.calls[0]![0];
    expect(args.system).toBe("be brief");
    expect(args.messages).toEqual([{ role: "user", content: "hi" }]);
    expect(args.max_tokens).toBe(1024);
    expect(args.model).toBe("claude-3-5-haiku-20241022");

    // Returned response is chat-shaped, caller-model preserved.
    const resp = outcome.response as { object: string; model: string; choices: any[] };
    expect(resp.object).toBe("chat.completion");
    expect(resp.model).toBe("gpt-4o");
    expect(resp.choices[0].message.content).toBe("hello");
    expect(outcome.usage).toEqual({
      promptTokens: 4,
      completionTokens: 3,
      cachedInputTokens: undefined,
    });
  });

  it("returns 'unsupported' when source has stream:true", async () => {
    const target = fakeAnthropic({ ok: {} });
    const outcome = await tryFallback({
      primaryShape: "openai",
      originalParams: {
        model: "gpt-4o",
        messages: [{ role: "user", content: "hi" }],
        stream: true,
      },
      target: { provider: "anthropic", client: target, model: "claude" },
    });
    expect(outcome.kind).toBe("unsupported");
    expect(target.__create).not.toHaveBeenCalled();
  });

  it("returns 'unsupported' when source has tools", async () => {
    const target = fakeAnthropic({ ok: {} });
    const outcome = await tryFallback({
      primaryShape: "openai",
      originalParams: {
        model: "gpt-4o",
        messages: [{ role: "user", content: "hi" }],
        tools: [{ type: "function", function: { name: "x" } }],
      },
      target: { provider: "anthropic", client: target, model: "claude" },
    });
    expect(outcome.kind).toBe("unsupported");
    expect(target.__create).not.toHaveBeenCalled();
  });

  it("returns 'error' when the target throws", async () => {
    const err = openaiError(503);
    const target = fakeAnthropic({ err });
    const outcome = await tryFallback({
      primaryShape: "openai",
      originalParams: {
        model: "gpt-4o",
        messages: [{ role: "user", content: "hi" }],
      },
      target: { provider: "anthropic", client: target, model: "claude" },
    });
    expect(outcome.kind).toBe("error");
    if (outcome.kind !== "error") throw new Error("unreachable");
    expect(outcome.error).toBe(err);
    expect(outcome.status).toBe(503);
  });
});
