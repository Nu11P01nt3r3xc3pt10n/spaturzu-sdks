// Lazy, process-wide default Spaturzu instance backing the drop-in provider
// subpaths (@spaturzu/sdk/openai etc.). `new Spaturzu()` already reads
// SPATURZU_BASE_URL / SPATURZU_API_KEY from env, so the zero-config path
// works with no configure() call.
//
// NOTE: `Spaturzu` is referenced only inside function bodies below — never at
// module top level — so the index.ts <-> default.ts import cycle resolves
// safely under ESM (live bindings are read at call time, not eval time).
import {
  Spaturzu,
  type SpaturzuOptions,
  type RunOptions,
  type ReportCallInput,
} from "./index.js";
import type { BudgetOnBreach } from "./budget.js";

let _default: Spaturzu | null = null;
let _configured: SpaturzuOptions | null = null;

/** Set process-wide spaturzu options (tags, onError, baseURL, timeouts) for
 *  the default instance. MUST be called before the first
 *  @spaturzu/sdk/<provider> client is constructed — throws otherwise so a
 *  mis-ordered call fails loudly instead of silently no-op'ing. */
export function configure(opts: SpaturzuOptions): void {
  if (_default) {
    throw new Error(
      "spaturzu: configure() must be called before constructing any " +
        "@spaturzu/sdk/<provider> client",
    );
  }
  _configured = opts;
}

/** Return the lazily-constructed default Spaturzu instance, creating it on
 *  first call (applying any options from a prior `configure()`) and caching
 *  it thereafter. Safe to call repeatedly — every drop-in client construction
 *  goes through here. */
export function getDefaultSpaturzu(): Spaturzu {
  if (!_default) _default = new Spaturzu(_configured ?? {});
  return _default;
}

/** Top-level frame helper bound to the default instance, so a fully drop-in
 *  user never constructs Spaturzu directly. Mirrors `Spaturzu.run`'s
 *  overloads and forwards to the singleton. */
export function run<T>(agentName: string, fn: () => Promise<T>): Promise<T>;
export function run<T>(
  agentName: string,
  opts: RunOptions,
  fn: () => Promise<T>,
): Promise<T>;
export function run<T>(
  agentName: string,
  optsOrFn: RunOptions | (() => Promise<T>),
  fn?: () => Promise<T>,
): Promise<T> {
  const sp = getDefaultSpaturzu();
  return fn === undefined
    ? sp.run(agentName, optsOrFn as () => Promise<T>)
    : sp.run(agentName, optsOrFn as RunOptions, fn);
}

export const flush = (): Promise<void> => getDefaultSpaturzu().flush();
export const shutdown = (): Promise<void> => getDefaultSpaturzu().shutdown();

export const report = (call: ReportCallInput): void =>
  getDefaultSpaturzu().report(call);

export const checkBudget = (
  agentName: string | null,
  opts?: { onBreach?: BudgetOnBreach },
): Promise<void> => getDefaultSpaturzu().checkBudget(agentName, opts);

/** Test-only: reset the singleton + pending config between cases. */
export function __resetDefaultForTests(): void {
  _default = null;
  _configured = null;
}
