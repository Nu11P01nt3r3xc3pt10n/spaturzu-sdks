import { v5 as uuidv5, validate as uuidValidate } from "uuid";

/** Namespace UUID used to derive deterministic v5 run-ids from OpenClaw's
 *  non-UUID run identifiers. Stable forever — rotating it would orphan
 *  historical Spaturzu agent_runs from new ones. */
export const SPATURZU_OPENCLAW_NS = "a3b1f6d2-0c9e-4e7b-9b1a-6d5c4e3f2a10";

/** OpenClaw runIds aren't guaranteed to be UUIDs, but Spaturzu's wire
 *  `runId` must be one. Pass valid UUIDs through; derive a deterministic
 *  UUIDv5 otherwise so every model call inside one OpenClaw run collapses
 *  into a single Spaturzu agent_run. */
export function toRunId(openclawRunId: string | undefined): string | undefined {
  if (!openclawRunId) return undefined;
  return uuidValidate(openclawRunId)
    ? openclawRunId
    : uuidv5(openclawRunId, SPATURZU_OPENCLAW_NS);
}

/** Map ctx.agentId → Spaturzu agentName, optionally prefixed.
 *  Pass `""` as prefix to disable prefixing. Blank/missing agentId
 *  falls back to "default" so calls are never untagged. */
export function toAgentName(
  agentId: string | undefined,
  agentPrefix: string,
): string {
  const trimmed = agentId?.trim();
  const base = trimmed && trimmed.length > 0 ? trimmed : "default";
  return agentPrefix ? `${agentPrefix}/${base}` : base;
}

const MAX_TAG_KEYS = 32;
const MAX_KEY_LEN = 64;
const MAX_VAL_LEN = 256;

/** Build the tag bag posted with each LogEntry: static config tags +
 *  optional channel/harness from the event/ctx, with LogEntry limits
 *  enforced defensively (≤32 keys, key ≤64 chars, value ≤256 chars). */
export function toTags(
  input: { channelId?: string; harnessId?: string },
  staticTags: Record<string, string>,
): Record<string, string> {
  const merged: Record<string, string> = { ...staticTags };
  if (input.channelId) merged.channel = input.channelId;
  if (input.harnessId) merged.harness = input.harnessId;

  const out: Record<string, string> = {};
  let count = 0;
  for (const [k, v] of Object.entries(merged)) {
    if (count >= MAX_TAG_KEYS) break;
    const key = k.slice(0, MAX_KEY_LEN);
    // Skip empty keys (LogEntrySchema requires min(1)) and collisions where
    // two source keys truncate to the same string (silent overwrite +
    // count inflation otherwise). Static-tag-first ordering means the
    // earlier entry wins, which matches the merge precedence.
    if (!key || key in out) continue;
    const val = v.slice(0, MAX_VAL_LEN);
    out[key] = val;
    count++;
  }
  return out;
}
