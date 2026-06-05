// Drop-in replacement for `@anthropic-ai/sdk`. Swap only the import specifier:
//   - import Anthropic from "@anthropic-ai/sdk";
//   + import Anthropic from "@spaturzu/sdk/anthropic";
import RealAnthropic from "@anthropic-ai/sdk";
import { getDefaultSpaturzu } from "../default.js";
import type { SpaturzuClientOptions } from "./openai.js";
import type { TagInput } from "../context.js";

class Anthropic extends RealAnthropic {
  /** Tag each call with an agent (supplied at runtime by the wrap proxy;
   *  declared here so it's visible to TypeScript on the drop-in client). */
  declare withAgent: (agent: string, opts?: { tags?: TagInput }) => this;

  constructor(
    opts?: ConstructorParameters<typeof RealAnthropic>[0],
    sp: SpaturzuClientOptions = {},
  ) {
    super(opts);
    return getDefaultSpaturzu().wrapAnthropic(this, sp) as Anthropic;
  }
}

export default Anthropic;
export { Anthropic };
