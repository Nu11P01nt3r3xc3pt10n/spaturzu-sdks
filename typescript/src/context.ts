import { AsyncLocalStorage } from "node:async_hooks";
import { randomUUID } from "node:crypto";

/** Tags arrive from customers as primitives (env="prod", retries=3,
 *  is_internal=true). Coerce to string at the boundary so the wire format
 *  stays Record<string,string> and aggregations group cleanly. */
export type TagInput = Record<string, string | number | boolean>;

export function normalizeTags(
  t: TagInput | undefined,
): Record<string, string> | undefined {
  if (!t) return undefined;
  const out: Record<string, string> = {};
  let any = false;
  for (const [k, v] of Object.entries(t)) {
    if (v == null) continue;
    out[k] = typeof v === "string" ? v : String(v);
    any = true;
  }
  return any ? out : undefined;
}

/** Binds a wrapped client to an agent name (+ optional frame-scoped tags)
 *  so each terminal call runs inside `runInFrame(agent, ...)`. Will be
 *  produced by the `.withAgent()` accessor on a wrapped client (added later
 *  in this feature; see the wrap modules). */
export interface AgentBinding {
  agent: string;
  tags?: Record<string, string>;
}

/** Return type of the `wrapX` functions: the wrapped client `T` plus a
 *  `.withAgent()` accessor that tags each terminal call with an agent name
 *  (and optional frame-scoped tags). Chainable — `.withAgent()` returns
 *  another `WithAgent<T>`. Provider-agnostic, so it lives here rather than
 *  being duplicated per wrap module. */
export type WithAgent<T> = T & {
  withAgent(agent: string, opts?: { tags?: TagInput }): WithAgent<T>;
};

// A RunFrame is the unit of agent attribution that propagates through
// AsyncLocalStorage. It survives `await`, `Promise.all`, `setTimeout`, and
// every other async boundary Node tracks.
//
// `lastRequestId` is intentionally mutable: each wrapped LLM call sets it on
// the *current* frame so that a subsequent `spaturzu.run(...)` nested below
// picks it up as the new frame's `parentRequestId`. Mutation is safe because
// the frame object is private to each call to `als.run(frame, fn)` — sibling
// runs get distinct frames.
export interface RunFrame {
  /** Stable across an entire workflow execution. */
  runId: string;
  /** Logical agent name for the current frame. */
  agentName: string;
  /** Full chain from root to current frame, e.g. ["research","writer"]. */
  agentPath: string[];
  /** ID of the parent agent's most recent request at the moment this frame
   *  was opened. Stays constant for every call made by this agent — points
   *  to "the call that spawned me." */
  parentRequestId?: string;
  /** ID of the most recent LLM call started inside this frame. Mutated by
   *  the wrappers; consumed by `run()` when nesting a child agent. */
  lastRequestId?: string;
  /** Cumulative frame-scoped tags. Inherits from the parent frame (if any)
   *  and extends with whatever was passed to `run()`. The wrappers further
   *  merge in process-wide spaturzu constructor tags at log-build time —
   *  this is just the run-scoped layer. */
  tags?: Record<string, string>;
}

const storage = new AsyncLocalStorage<RunFrame>();

export function getCurrentFrame(): RunFrame | undefined {
  return storage.getStore();
}

/** Run `fn` inside a fresh agent frame. If called inside another frame,
 *  the new frame inherits the runId and extends the agent path; the parent's
 *  lastRequestId becomes the child's parentRequestId.
 *
 *  Tags: the new frame inherits the parent frame's tags and extends them
 *  with `tags` (passed-in keys win on conflict — the inner scope is the
 *  more specific one). Pass `undefined` to inherit cleanly. */
export function runInFrame<T>(
  agentName: string,
  fn: () => Promise<T>,
  tags?: Record<string, string>,
): Promise<T> {
  const parent = getCurrentFrame();
  const mergedTags = mergeTags(parent?.tags, tags);
  const frame: RunFrame = {
    runId: parent?.runId ?? randomUUID(),
    agentName,
    agentPath: parent
      ? [...parent.agentPath, agentName]
      : [agentName],
    parentRequestId: parent?.lastRequestId,
    lastRequestId: undefined,
    tags: mergedTags,
  };
  return storage.run(frame, fn);
}

/** Shallow-merge two tag bags, with `b` winning on key conflict. Returns
 *  `undefined` if both are empty so callers can drop the field entirely
 *  rather than serialising `{}`. */
export function mergeTags(
  a: Record<string, string> | undefined,
  b: Record<string, string> | undefined,
): Record<string, string> | undefined {
  if (!a && !b) return undefined;
  if (!a) return b ? { ...b } : undefined;
  if (!b) return { ...a };
  return { ...a, ...b };
}

/** Set the most-recent-call pointer on the current frame, if any. Called
 *  by wrappers as soon as they generate a request UUID, *before* the
 *  upstream HTTP call — so concurrent nested runs see it immediately. */
export function setLastRequestId(requestId: string): void {
  const frame = getCurrentFrame();
  if (frame) frame.lastRequestId = requestId;
}
