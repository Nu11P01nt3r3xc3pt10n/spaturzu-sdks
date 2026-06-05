// Wire-format types for the gateway's /v1/logs endpoint.
//
// The SDK is the *producer* of this payload — the gateway owns the
// Zod-backed schema and re-validates on receipt — so this file is a
// plain TypeScript mirror, no runtime cost. Keep field names + optionality
// in lockstep with packages/shared/src/index.ts::LogEntrySchema. If the
// gateway schema gains a field, add it here too; if a field becomes
// required server-side, drop the `?`.

export interface LogEntry {
  /** Client-supplied request UUID. SDK generates locally so parent agents
   *  can reference children before the POST round-trips. */
  id?: string;

  provider: string;
  model: string;

  // Attribution — all optional.
  runId?: string;
  parentRequestId?: string;
  agentName?: string;
  agentPath?: string[];
  userId?: string;
  sessionId?: string;

  // Token usage. Either from the provider's `usage` field or estimated
  // locally via tiktoken when the provider omits it.
  promptTokens?: number;
  completionTokens?: number;
  cachedInputTokens?: number;

  /** Provenance of the token counts. 'provider' = upstream returned usage
   *  (authoritative). 'tiktoken' = SDK tokenized text locally. */
  usageSource?: "provider" | "tiktoken";

  /** HTTP status returned by the provider (or synthesised on timeout). */
  status: number;
  latencyMs?: number;

  /** Free-form string→string tags. Server caps: 32 keys, 64-char keys,
   *  256-char values. */
  tags?: Record<string, string>;

  /** Free-form metadata bag. */
  metadata?: Record<string, unknown>;
}
