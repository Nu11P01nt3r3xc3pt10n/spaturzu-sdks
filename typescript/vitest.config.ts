import { defineConfig } from "vitest/config";

// Minimal Vitest config for @spaturzu/sdk.
// - Node environment is the default; the SDK runs in Node (uses AsyncLocalStorage).
// - `globals: false` keeps imports explicit (`import { describe, it, expect } from "vitest"`).
// - Coverage is opt-in via `pnpm test:coverage`; no enforced thresholds in v1.
export default defineConfig({
  test: {
    environment: "node",
    globals: false,
    include: ["test/**/*.test.ts"],
    coverage: {
      provider: "v8",
      reporter: ["text", "html"],
      include: ["src/**/*.ts"],
    },
  },
});
