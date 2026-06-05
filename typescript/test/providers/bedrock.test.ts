import { describe, it, expect } from "vitest";
import {
  BedrockRuntime as RealBedrockRuntime,
  BedrockRuntimeClient as RealBedrockRuntimeClient,
} from "@aws-sdk/client-bedrock-runtime";
import { BedrockRuntime } from "../../src/providers/bedrock.js";

describe("@spaturzu/sdk/bedrock drop-in", () => {
  it("constructs an instrumented Converse client preserving instanceof", () => {
    const client = new BedrockRuntime({ region: "us-east-1" });
    // Targets the aggregated BedrockRuntime (which has converse/converseStream).
    expect(client).toBeInstanceOf(RealBedrockRuntime);
    // BedrockRuntime extends BedrockRuntimeClient, so this holds too.
    expect(client).toBeInstanceOf(RealBedrockRuntimeClient);
    expect(typeof client.withAgent).toBe("function");
    expect(client.converse).toBeTypeOf("function");
    expect(client.converseStream).toBeTypeOf("function");
  });
});
