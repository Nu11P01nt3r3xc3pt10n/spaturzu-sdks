import { describe, it, expect } from "vitest";
import {
  toRunId,
  toAgentName,
  toTags,
  SPATURZU_OPENCLAW_NS,
} from "../src/identity.js";
import { v5 as uuidv5 } from "uuid";

describe("toRunId", () => {
  it("passes through a UUID unchanged", () => {
    const u = "0190a9fa-1234-7abc-9def-0123456789ab";
    expect(toRunId(u)).toBe(u);
  });

  it("derives a deterministic UUIDv5 from a non-UUID input", () => {
    const expected = uuidv5("openclaw-run-42", SPATURZU_OPENCLAW_NS);
    expect(toRunId("openclaw-run-42")).toBe(expected);
    expect(toRunId("openclaw-run-42")).toBe(toRunId("openclaw-run-42"));
  });

  it("returns undefined for undefined input", () => {
    expect(toRunId(undefined)).toBeUndefined();
  });

  it("returns undefined for empty-string input", () => {
    expect(toRunId("")).toBeUndefined();
  });
});

describe("toAgentName", () => {
  it("prefixes when prefix is non-empty", () => {
    expect(toAgentName("writer", "openclaw")).toBe("openclaw/writer");
  });
  it("returns bare name when prefix is empty string", () => {
    expect(toAgentName("writer", "")).toBe("writer");
  });
  it("defaults to 'default' when agentId is missing/blank", () => {
    expect(toAgentName(undefined, "openclaw")).toBe("openclaw/default");
    expect(toAgentName("  ", "openclaw")).toBe("openclaw/default");
  });
});

describe("toTags", () => {
  it("merges static tags with channel + harness, dropping undefined", () => {
    expect(
      toTags(
        { channelId: "discord:guild-1", harnessId: "claude-code" },
        { env: "dev" },
      ),
    ).toEqual({
      env: "dev",
      channel: "discord:guild-1",
      harness: "claude-code",
    });
  });

  it("drops missing optional fields rather than emitting undefined values", () => {
    expect(toTags({}, { env: "dev" })).toEqual({ env: "dev" });
  });

  it("enforces LogEntry tag limits (≤32 keys, key ≤64 chars, value ≤256 chars)", () => {
    const huge: Record<string, string> = {};
    for (let i = 0; i < 50; i++) huge[`k${i}`] = "v";
    const longKey = "k".repeat(80);
    const longVal = "v".repeat(300);
    const out = toTags({}, { ...huge, [longKey]: longVal });
    expect(Object.keys(out).length).toBeLessThanOrEqual(32);
    for (const [k, v] of Object.entries(out)) {
      expect(k.length).toBeLessThanOrEqual(64);
      expect(v.length).toBeLessThanOrEqual(256);
    }
  });

  it("skips empty-string keys (gateway requires key length >= 1)", () => {
    const out = toTags({}, { "": "v", env: "dev" });
    expect(out).toEqual({ env: "dev" });
    expect("" in out).toBe(false);
  });

  it("deduplicates keys that collide after truncation (first wins)", () => {
    const prefix = "a".repeat(65); // both slice to "a".repeat(64)
    const out = toTags({}, {
      [prefix + "x"]: "first",
      [prefix + "y"]: "second",
    });
    const expectedKey = "a".repeat(64);
    expect(Object.keys(out)).toEqual([expectedKey]);
    expect(out[expectedKey]).toBe("first");
  });
});
