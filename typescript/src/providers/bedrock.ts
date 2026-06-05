// Drop-in replacement for `@aws-sdk/client-bedrock-runtime` (Converse API).
// Swap only the import specifier:
//   - import { BedrockRuntime } from "@aws-sdk/client-bedrock-runtime";
//   + import { BedrockRuntime } from "@spaturzu/sdk/bedrock";
//
// NOTE: this targets `BedrockRuntime` — the aggregated client that exposes the
// named methods `converse` / `converseStream` — NOT the low-level
// `BedrockRuntimeClient` (which has no such methods; it uses the Command form
// `client.send(new ConverseCommand(...))`, which `wrapBedrock` does not
// instrument). Use this drop-in only if your code calls `client.converse(...)`.
import {
  BedrockRuntime as RealBedrock,
  type BedrockRuntimeClientConfig,
} from "@aws-sdk/client-bedrock-runtime";
import { getDefaultSpaturzu } from "../default.js";
import type { SpaturzuClientOptions } from "./openai.js";
import type { TagInput } from "../context.js";

class BedrockRuntime extends RealBedrock {
  /** Tag each call with an agent (supplied at runtime by the wrap proxy;
   *  declared here so it's visible to TypeScript on the drop-in client). */
  declare withAgent: (agent: string, opts?: { tags?: TagInput }) => this;

  constructor(
    config: BedrockRuntimeClientConfig,
    sp: SpaturzuClientOptions = {},
  ) {
    super(config);
    return getDefaultSpaturzu().wrapBedrock(this, sp) as BedrockRuntime;
  }
}

export { BedrockRuntime };
