import { describe, it, expect } from "vitest";
import { parseConfig } from "../src/config.js";

describe("parseConfig", () => {
  it("requires apiKey", () => {
    expect(() => parseConfig({})).toThrow();
    expect(() => parseConfig(undefined)).toThrow();
  });

  it("applies defaults for baseUrl, agentPrefix, enforceBudgets, tags", () => {
    const c = parseConfig({ apiKey: "spa_test" });
    expect(c.apiKey).toBe("spa_test");
    expect(c.baseUrl).toBe("https://spaturzu-api.superchiu.org");
    expect(c.agentPrefix).toBe("openclaw");
    expect(c.enforceBudgets).toBe(true);
    expect(c.tags).toEqual({});
  });

  it("accepts overrides", () => {
    const c = parseConfig({
      apiKey: "spa_test",
      baseUrl: "https://gw.example",
      agentPrefix: "",
      enforceBudgets: false,
      tags: { env: "prod" },
    });
    expect(c.baseUrl).toBe("https://gw.example");
    expect(c.agentPrefix).toBe("");
    expect(c.enforceBudgets).toBe(false);
    expect(c.tags).toEqual({ env: "prod" });
  });

  it("rejects invalid baseUrl", () => {
    expect(() => parseConfig({ apiKey: "spa_test", baseUrl: "not-a-url" })).toThrow();
  });
});
