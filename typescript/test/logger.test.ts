import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { Logger, HttpError } from "../src/logger.js";

// Helpers --------------------------------------------------------------------

const okResponse = () =>
  new Response("", { status: 200 });
const errResponse = (status: number, body = "") =>
  new Response(body, { status });

function spyFetch() {
  // vi.spyOn returns a typed mock; cast to any for ergonomic .mockResolvedValueOnce.
  return vi.spyOn(globalThis, "fetch") as unknown as ReturnType<typeof vi.fn>;
}

const ENTRY = {
  provider: "openai" as const,
  model: "gpt-4o",
  status: 200,
};

// Tests ----------------------------------------------------------------------

describe("Logger: fire-and-forget", () => {
  let fetchSpy: ReturnType<typeof spyFetch>;
  beforeEach(() => {
    fetchSpy = spyFetch();
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("log() returns synchronously even though the POST runs in background", () => {
    fetchSpy.mockResolvedValue(okResponse());
    const logger = new Logger("https://gw.example", undefined, () => {}, {
      backoffMs: [],
    });
    const before = Date.now();
    logger.log(ENTRY);
    const after = Date.now();
    expect(after - before).toBeLessThan(20); // No await on the POST.
  });

  it("flush() awaits inflight POSTs", async () => {
    let resolveFetch!: (v: Response) => void;
    fetchSpy.mockReturnValue(
      new Promise<Response>((r) => {
        resolveFetch = r;
      }),
    );
    const logger = new Logger("https://gw.example", undefined, () => {}, {
      backoffMs: [],
    });
    logger.log(ENTRY);
    let flushed = false;
    const flushPromise = logger.flush().then(() => {
      flushed = true;
    });
    // Microtask flush — flush should NOT resolve yet.
    await Promise.resolve();
    expect(flushed).toBe(false);
    resolveFetch(okResponse());
    await flushPromise;
    expect(flushed).toBe(true);
  });
});

describe("Logger: retries", () => {
  let fetchSpy: ReturnType<typeof spyFetch>;
  beforeEach(() => {
    fetchSpy = spyFetch();
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("retries on 5xx then succeeds", async () => {
    fetchSpy
      .mockResolvedValueOnce(errResponse(503))
      .mockResolvedValueOnce(okResponse());
    const errors: unknown[] = [];
    const logger = new Logger(
      "https://gw.example",
      undefined,
      (err) => errors.push(err),
      { backoffMs: [1] }, // 1ms backoff so the test is fast.
    );
    logger.log(ENTRY);
    await logger.flush();
    expect(fetchSpy).toHaveBeenCalledTimes(2);
    expect(errors).toHaveLength(0);
  });

  it("retries on 429 then succeeds", async () => {
    fetchSpy
      .mockResolvedValueOnce(errResponse(429))
      .mockResolvedValueOnce(okResponse());
    const logger = new Logger("https://gw.example", undefined, () => {}, {
      backoffMs: [1],
    });
    logger.log(ENTRY);
    await logger.flush();
    expect(fetchSpy).toHaveBeenCalledTimes(2);
  });

  it("does NOT retry on 4xx (non-429)", async () => {
    fetchSpy.mockResolvedValue(errResponse(400, "bad request"));
    const errors: unknown[] = [];
    const logger = new Logger(
      "https://gw.example",
      undefined,
      (err) => errors.push(err),
      { backoffMs: [1, 1, 1] },
    );
    logger.log(ENTRY);
    await logger.flush();
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(errors).toHaveLength(1);
    expect(errors[0]).toBeInstanceOf(HttpError);
    expect((errors[0] as HttpError).status).toBe(400);
  });

  it("gives up after exhausting backoffMs", async () => {
    fetchSpy.mockResolvedValue(errResponse(500));
    const errors: unknown[] = [];
    const logger = new Logger(
      "https://gw.example",
      undefined,
      (err) => errors.push(err),
      { backoffMs: [1, 1] }, // 2 retries → 3 attempts total
    );
    logger.log(ENTRY);
    await logger.flush();
    expect(fetchSpy).toHaveBeenCalledTimes(3);
    expect(errors).toHaveLength(1);
  });
});

describe("Logger: headers + body", () => {
  let fetchSpy: ReturnType<typeof spyFetch>;
  beforeEach(() => {
    fetchSpy = spyFetch();
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("sets x-spaturzu-key when apiKey is provided", async () => {
    fetchSpy.mockResolvedValue(okResponse());
    const logger = new Logger("https://gw.example", "secret-key", () => {}, {
      backoffMs: [],
    });
    logger.log(ENTRY);
    await logger.flush();
    const init = fetchSpy.mock.calls[0]![1] as RequestInit;
    expect((init.headers as Record<string, string>)["x-spaturzu-key"]).toBe(
      "secret-key",
    );
    expect((init.headers as Record<string, string>)["content-type"]).toBe(
      "application/json",
    );
  });

  it("omits x-spaturzu-key when apiKey is undefined", async () => {
    fetchSpy.mockResolvedValue(okResponse());
    const logger = new Logger("https://gw.example", undefined, () => {}, {
      backoffMs: [],
    });
    logger.log(ENTRY);
    await logger.flush();
    const init = fetchSpy.mock.calls[0]![1] as RequestInit;
    expect(
      (init.headers as Record<string, string>)["x-spaturzu-key"],
    ).toBeUndefined();
  });

  it("posts the entry as JSON body to /v1/logs", async () => {
    fetchSpy.mockResolvedValue(okResponse());
    const logger = new Logger("https://gw.example", undefined, () => {}, {
      backoffMs: [],
    });
    logger.log(ENTRY);
    await logger.flush();
    const [url, init] = fetchSpy.mock.calls[0]!;
    expect(url).toBe("https://gw.example/v1/logs");
    expect((init as RequestInit).method).toBe("POST");
    expect(JSON.parse((init as RequestInit).body as string)).toEqual(ENTRY);
  });
});

describe("Logger: concurrency cap", () => {
  let fetchSpy: ReturnType<typeof spyFetch>;
  beforeEach(() => {
    fetchSpy = spyFetch();
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("never exceeds maxConcurrent in-flight POSTs", async () => {
    let inflight = 0;
    let peak = 0;
    fetchSpy.mockImplementation(() => {
      inflight += 1;
      peak = Math.max(peak, inflight);
      return new Promise<Response>((resolve) => {
        setTimeout(() => {
          inflight -= 1;
          resolve(okResponse());
        }, 5);
      });
    });
    const logger = new Logger("https://gw.example", undefined, () => {}, {
      backoffMs: [],
      maxConcurrent: 3,
    });
    for (let i = 0; i < 10; i++) logger.log(ENTRY);
    await logger.flush();
    expect(peak).toBeLessThanOrEqual(3);
  });
});
