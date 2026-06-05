// Duck-typed minimal LLM clients for unit tests. Each factory builds an
// object exposing only the methods the corresponding wrap intercepts.
// Plans 3/4/5 will add fakeBedrock / fakeGemini / fakeMistral here.

import { vi } from "vitest";

export type FakeResult =
  | { ok: any }
  | { err: any }
  | { stream: AsyncIterable<any> };

/** Build a fake OpenAI client. Pass `{ ok }` for a success response,
 *  `{ err }` to throw, or `{ stream }` for a streaming AsyncIterable. */
export function fakeOpenAI(result: FakeResult) {
  const create = vi.fn(async (params: any) => {
    if ("err" in result) throw result.err;
    if ("stream" in result) {
      // Honour the wrap's behaviour even when the caller didn't set stream:true
      // — the test wrapped a stream-style result deliberately.
      return result.stream;
    }
    return result.ok;
  });
  return {
    chat: { completions: { create } },
    // Track the spy so tests can inspect call args.
    __create: create,
  };
}

/** Build a fake Anthropic client (messages.create only). */
export function fakeAnthropic(result: FakeResult) {
  const create = vi.fn(async (params: any) => {
    if ("err" in result) throw result.err;
    if ("stream" in result) return result.stream;
    return result.ok;
  });
  return {
    messages: { create },
    __create: create,
  };
}

/** Helper: turn an array of chunks into a stream-style AsyncIterable. */
export async function* asAsyncIterable<T>(chunks: T[]): AsyncGenerator<T> {
  for (const c of chunks) yield c;
}

/** Helper: build an OpenAI-shaped error with a `status` property. */
export function openaiError(status: number, name = "APIError"): Error {
  const err = new Error(`fake openai error ${status}`);
  Object.assign(err, { status, name });
  return err;
}

/** Helper: build an Anthropic-shaped error. Same shape as openaiError. */
export function anthropicError(status: number, name = "APIError"): Error {
  return openaiError(status, name);
}

/** Build a fake Bedrock client. Exposes `converse` and `converseStream`
 *  only. `result.ok` is a Converse response shape; `result.stream` is
 *  an AsyncIterable of Converse stream events that the wrap will iterate. */
export function fakeBedrock(
  result: FakeResult,
  streamResult: FakeResult | undefined = undefined,
) {
  const converse = vi.fn(async (params: any) => {
    if ("err" in result) throw result.err;
    if ("stream" in result) {
      // For converse (non-stream), passing `stream` is a test misuse —
      // use the `streamResult` arg of converseStream instead.
      throw new Error("fakeBedrock.converse called with stream result");
    }
    return result.ok;
  });
  const converseStream = vi.fn(async (params: any) => {
    const r = streamResult ?? result;
    if ("err" in r) throw r.err;
    if ("ok" in r) {
      // Bedrock streaming returns { stream: AsyncIterable<event> }.
      // For tests, callers can pass either an `ok` with that shape
      // OR a bare `stream` (we wrap it).
      return r.ok;
    }
    return { stream: r.stream };
  });
  return {
    converse,
    converseStream,
    __converse: converse,
    __converseStream: converseStream,
  };
}

/** Build an AWS-SDK-shaped error. v3 errors carry `name` (the exception
 *  type) and `$metadata.httpStatusCode`. */
export function bedrockError(status: number, name = "ThrottlingException"): Error {
  const err = new Error(`fake bedrock error ${status}`);
  Object.assign(err, {
    name,
    $metadata: { httpStatusCode: status },
  });
  return err;
}

/** Build a fake @google/genai client. Exposes `models.generateContent`
 *  and `models.generateContentStream`. */
export function fakeGemini(
  result: FakeResult,
  streamResult: FakeResult | undefined = undefined,
) {
  const generateContent = vi.fn(async (params: any) => {
    if ("err" in result) throw result.err;
    if ("stream" in result) {
      throw new Error("fakeGemini.generateContent called with stream result");
    }
    return result.ok;
  });
  const generateContentStream = vi.fn(async (params: any) => {
    const r = streamResult ?? result;
    if ("err" in r) throw r.err;
    if ("ok" in r) return r.ok;
    return r.stream;
  });
  return {
    models: { generateContent, generateContentStream },
    __generateContent: generateContent,
    __generateContentStream: generateContentStream,
  };
}

/** Build a Gemini-shaped error. `@google/genai` throws ApiError with `.status`. */
export function geminiError(status: number, name = "ApiError"): Error {
  const err = new Error(`fake gemini error ${status}`);
  Object.assign(err, { name, status });
  return err;
}

/** Build a fake Mistral client. `chat.complete` and `chat.stream` are
 *  separate methods (not a stream:true flag). */
export function fakeMistral(
  result: FakeResult,
  streamResult: FakeResult | undefined = undefined,
) {
  const complete = vi.fn(async (params: any) => {
    if ("err" in result) throw result.err;
    if ("stream" in result) {
      throw new Error("fakeMistral.complete called with stream result");
    }
    return result.ok;
  });
  const stream = vi.fn(async (params: any) => {
    const r = streamResult ?? result;
    if ("err" in r) throw r.err;
    if ("ok" in r) return r.ok;
    return r.stream;
  });
  return {
    chat: { complete, stream },
    __complete: complete,
    __stream: stream,
  };
}

export function mistralError(status: number, name = "SDKError"): Error {
  const err = new Error(`fake mistral error ${status}`);
  Object.assign(err, { name, statusCode: status, status });
  return err;
}
