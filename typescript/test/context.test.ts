import { describe, it, expect } from "vitest";
import {
  runInFrame,
  getCurrentFrame,
  mergeTags,
  setLastRequestId,
  normalizeTags,
} from "../src/context.js";

describe("context: runInFrame at the root", () => {
  it("opens a frame with a fresh runId and single-segment agentPath", async () => {
    let captured: ReturnType<typeof getCurrentFrame>;
    await runInFrame("root", async () => {
      captured = getCurrentFrame();
    });
    expect(captured?.runId).toMatch(
      /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/,
    );
    expect(captured?.agentName).toBe("root");
    expect(captured?.agentPath).toEqual(["root"]);
    expect(captured?.parentRequestId).toBeUndefined();
    expect(captured?.lastRequestId).toBeUndefined();
  });

  it("getCurrentFrame returns undefined outside any frame", () => {
    expect(getCurrentFrame()).toBeUndefined();
  });
});

describe("context: nesting", () => {
  it("inner frame inherits runId and extends agentPath", async () => {
    let outerRunId: string | undefined;
    let inner: ReturnType<typeof getCurrentFrame>;
    await runInFrame("outer", async () => {
      outerRunId = getCurrentFrame()?.runId;
      await runInFrame("inner", async () => {
        inner = getCurrentFrame();
      });
    });
    expect(inner?.runId).toBe(outerRunId);
    expect(inner?.agentPath).toEqual(["outer", "inner"]);
  });

  it("inner frame's parentRequestId = outer frame's lastRequestId at open time", async () => {
    let inner: ReturnType<typeof getCurrentFrame>;
    await runInFrame("outer", async () => {
      setLastRequestId("req-from-outer");
      await runInFrame("inner", async () => {
        inner = getCurrentFrame();
      });
    });
    expect(inner?.parentRequestId).toBe("req-from-outer");
  });

  it("inner frame's parentRequestId is undefined when outer never logged", async () => {
    let inner: ReturnType<typeof getCurrentFrame>;
    await runInFrame("outer", async () => {
      await runInFrame("inner", async () => {
        inner = getCurrentFrame();
      });
    });
    expect(inner?.parentRequestId).toBeUndefined();
  });
});

describe("context: tag merging", () => {
  it("inner tags inherit and override the outer's on conflict", async () => {
    let inner: ReturnType<typeof getCurrentFrame>;
    await runInFrame(
      "outer",
      async () => {
        await runInFrame(
          "inner",
          async () => {
            inner = getCurrentFrame();
          },
          { env: "prod", team: "search" },
        );
      },
      { env: "dev", region: "us-east-1" },
    );
    expect(inner?.tags).toEqual({
      env: "prod",
      region: "us-east-1",
      team: "search",
    });
  });

  it("a frame without tags inherits the parent's tags unchanged", async () => {
    let inner: ReturnType<typeof getCurrentFrame>;
    await runInFrame(
      "outer",
      async () => {
        await runInFrame("inner", async () => {
          inner = getCurrentFrame();
        });
      },
      { env: "dev" },
    );
    expect(inner?.tags).toEqual({ env: "dev" });
  });
});

describe("context: parallel runs see isolated frames", () => {
  it("two siblings under one root each get their own agentName", async () => {
    const seen: string[] = [];
    await runInFrame("root", async () => {
      await Promise.all([
        runInFrame("a", async () => {
          seen.push(getCurrentFrame()!.agentName);
        }),
        runInFrame("b", async () => {
          seen.push(getCurrentFrame()!.agentName);
        }),
      ]);
    });
    expect(seen.sort()).toEqual(["a", "b"]);
  });

  it("setLastRequestId in one sibling does not bleed into the other", async () => {
    let aSawParent: string | undefined;
    let bSawParent: string | undefined;
    await runInFrame("root", async () => {
      await Promise.all([
        runInFrame("a", async () => {
          setLastRequestId("from-a");
          aSawParent = "ignored — checked below";
        }),
        runInFrame("b", async () => {
          // b's own frame.parentRequestId was set at open-time from root's
          // lastRequestId (which is undefined here — a is parallel, not nested).
          bSawParent = getCurrentFrame()?.parentRequestId;
        }),
      ]);
    });
    expect(bSawParent).toBeUndefined();
  });
});

describe("context: mergeTags", () => {
  it("returns undefined when both are empty", () => {
    expect(mergeTags(undefined, undefined)).toBeUndefined();
  });

  it("returns a fresh copy of a when b is undefined", () => {
    const a = { x: "1" };
    const out = mergeTags(a, undefined);
    expect(out).toEqual({ x: "1" });
    expect(out).not.toBe(a);
  });

  it("returns a fresh copy of b when a is undefined", () => {
    const b = { y: "2" };
    const out = mergeTags(undefined, b);
    expect(out).toEqual({ y: "2" });
    expect(out).not.toBe(b);
  });

  it("b wins on key conflict", () => {
    expect(mergeTags({ x: "1", y: "2" }, { x: "3" })).toEqual({
      x: "3",
      y: "2",
    });
  });
});

describe("context: setLastRequestId", () => {
  it("is a no-op outside any frame", () => {
    expect(() => setLastRequestId("any")).not.toThrow();
  });

  it("mutates the current frame's lastRequestId", async () => {
    let captured: ReturnType<typeof getCurrentFrame>;
    await runInFrame("agent", async () => {
      setLastRequestId("req-1");
      captured = getCurrentFrame();
    });
    expect(captured?.lastRequestId).toBe("req-1");
  });
});

describe("normalizeTags", () => {
  it("coerces primitives to strings and drops empty", () => {
    expect(normalizeTags({ env: "prod", retries: 3, internal: true })).toEqual({
      env: "prod",
      retries: "3",
      internal: "true",
    });
    expect(normalizeTags(undefined)).toBeUndefined();
    expect(normalizeTags({})).toBeUndefined();
  });
});
