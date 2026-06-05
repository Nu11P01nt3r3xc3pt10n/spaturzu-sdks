// Drop-in replacement for `@mistralai/mistralai`. Swap only the import specifier:
//   - import { Mistral } from "@mistralai/mistralai";
//   + import { Mistral } from "@spaturzu/sdk/mistral";
import { Mistral as RealMistral } from "@mistralai/mistralai";
import { getDefaultSpaturzu } from "../default.js";
import type { SpaturzuClientOptions } from "./openai.js";
import type { TagInput } from "../context.js";

class Mistral extends RealMistral {
  /** Tag each call with an agent (supplied at runtime by the wrap proxy;
   *  declared here so it's visible to TypeScript on the drop-in client). */
  declare withAgent: (agent: string, opts?: { tags?: TagInput }) => this;

  constructor(
    opts?: ConstructorParameters<typeof RealMistral>[0],
    sp: SpaturzuClientOptions = {},
  ) {
    super(opts);
    return getDefaultSpaturzu().wrapMistral(this, sp) as Mistral;
  }
}

export { Mistral };
