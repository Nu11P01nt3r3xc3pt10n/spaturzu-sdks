// Drop-in replacement for `openai`. Swap only the import specifier:
//   - import OpenAI from "openai";
//   + import OpenAI from "@spaturzu/sdk/openai";
// Construction and call sites are unchanged. Spaturzu metering config is read
// from env (SPATURZU_API_KEY / SPATURZU_BASE_URL) via the default singleton;
// call configure({...}) once at startup for tags/onError.
import RealOpenAI from "openai";
import { getDefaultSpaturzu } from "../default.js";
import type { BudgetWrapOptions } from "../budget.js";
import type { FallbackTarget } from "../fallback.js";
import type { TagInput } from "../context.js";

/** Optional 2nd constructor arg — ignored by the real SDK; consumed here to
 *  reach budget hard-caps and cross-provider fallback on the drop-in path. */
export interface SpaturzuClientOptions {
  budget?: BudgetWrapOptions;
  fallback?: FallbackTarget[];
}

class OpenAI extends RealOpenAI {
  /** Tag each call with an agent (supplied at runtime by the wrap proxy;
   *  declared here so it's visible to TypeScript on the drop-in client). */
  declare withAgent: (agent: string, opts?: { tags?: TagInput }) => this;

  constructor(
    opts?: ConstructorParameters<typeof RealOpenAI>[0],
    sp: SpaturzuClientOptions = {},
  ) {
    super(opts);
    // Returning an object from a constructor replaces the `new` result.
    // wrapOpenAI binds the real methods to the real instance, so private
    // fields and identity stay on `this`, never the Proxy.
    return getDefaultSpaturzu().wrapOpenAI(this, sp) as OpenAI;
  }
}

export default OpenAI;
export { OpenAI };
