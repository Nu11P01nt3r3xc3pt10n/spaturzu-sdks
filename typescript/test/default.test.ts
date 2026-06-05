import { describe, it, expect, beforeEach } from "vitest";
import {
  getDefaultSpaturzu,
  configure,
  __resetDefaultForTests,
} from "../src/default.js";

describe("default Spaturzu singleton", () => {
  beforeEach(() => __resetDefaultForTests());

  it("returns the same instance across calls", () => {
    expect(getDefaultSpaturzu()).toBe(getDefaultSpaturzu());
  });

  it("configure() before first use forwards options to the instance", () => {
    configure({ tags: { env: "test" } });
    // processTags is private; read it structurally to confirm the configured
    // tags actually reached the constructor (normalized to string values).
    const instance = getDefaultSpaturzu() as unknown as {
      processTags?: Record<string, string>;
    };
    expect(instance.processTags).toEqual({ env: "test" });
  });

  it("configure() after first construction throws", () => {
    getDefaultSpaturzu();
    expect(() => configure({ tags: { env: "x" } })).toThrow(/before constructing/);
  });
});
