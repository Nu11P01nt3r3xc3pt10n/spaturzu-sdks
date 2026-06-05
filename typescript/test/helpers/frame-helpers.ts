// Tiny utility for setting up frame state without re-typing runInFrame.

import { runInFrame } from "../../src/context.js";

/** Run `fn` inside a fresh frame named `agent`. Returns the value `fn` returns. */
export async function inFrame<T>(
  agent: string,
  fn: () => Promise<T>,
  tags?: Record<string, string>,
): Promise<T> {
  return runInFrame(agent, fn, tags);
}
