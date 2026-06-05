// Drop-in replacement for the SDK's Logger: records every entry passed to
// .log() into an in-memory array so tests can assert on the payload. Cast
// the returned object to `Logger` at the call site — the wraps only ever
// invoke `.log(entry)`, so structural compatibility is enough.
import type { SdkLogEntry } from "../../src/logger.js";

export interface FakeLogger {
  entries: SdkLogEntry[];
  log: (entry: SdkLogEntry) => void;
  flush: () => Promise<void>;
}

export function fakeLogger(): FakeLogger {
  const entries: SdkLogEntry[] = [];
  return {
    entries,
    log(entry) {
      entries.push(entry);
    },
    async flush() {
      // No-op; the real Logger awaits inflight POSTs, but the fake is sync.
    },
  };
}
