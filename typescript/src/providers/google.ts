// Drop-in replacement for `@google/genai`. Swap only the import specifier:
//   - import { GoogleGenAI } from "@google/genai";
//   + import { GoogleGenAI } from "@spaturzu/sdk/google";
import { GoogleGenAI as RealGoogleGenAI } from "@google/genai";
import { getDefaultSpaturzu } from "../default.js";
import type { SpaturzuClientOptions } from "./openai.js";
import type { TagInput } from "../context.js";

class GoogleGenAI extends RealGoogleGenAI {
  /** Tag each call with an agent (supplied at runtime by the wrap proxy;
   *  declared here so it's visible to TypeScript on the drop-in client). */
  declare withAgent: (agent: string, opts?: { tags?: TagInput }) => this;

  constructor(
    // `opts` is required (unlike the other providers' optional arg): the real
    // GoogleGenAI constructor throws without a config object.
    opts: ConstructorParameters<typeof RealGoogleGenAI>[0],
    sp: SpaturzuClientOptions = {},
  ) {
    super(opts);
    return getDefaultSpaturzu().wrapGemini(this, sp) as GoogleGenAI;
  }
}

export { GoogleGenAI };
