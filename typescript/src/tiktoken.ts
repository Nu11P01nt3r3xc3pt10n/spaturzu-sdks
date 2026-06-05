// Lazy fallback tokenizer for streaming responses that arrive without a
// `usage` payload. The mainstream-OpenAI / Anthropic streams always emit
// usage; the long tail of OpenAI-compatible endpoints (Together, Groq,
// Anyscale, vLLM-self-hosted, older Anthropic-via-Bedrock) does not. This
// module estimates token counts from captured prompt + completion text so
// those calls still attribute cost.
//
// Why lazy: `@dqbd/tiktoken` is a WASM blob (~3 MB) and an *optional* peer
// dep. Customers who never need the fallback don't pay the install cost,
// and the import only fires the first time we hit the missing-usage branch.
//
// Why a module-level cache: constructing a Tiktoken encoder loads the BPE
// table into WASM memory — cheap on the millisecond scale, but pointless to
// repeat per call. Encoders live for the lifetime of the process; we only
// keep one per encoding name (≤2 in practice).

type TiktokenModule = typeof import("@dqbd/tiktoken");
type TiktokenInstance = ReturnType<TiktokenModule["get_encoding"]>;
type TiktokenEncoding = Parameters<TiktokenModule["get_encoding"]>[0];

let modulePromise: Promise<TiktokenModule | null> | undefined;
const encoderCache = new Map<TiktokenEncoding, TiktokenInstance>();
let warned = false;

// Single-flight: cache the import promise so concurrent first-time callers
// share one resolution. On failure we cache `null` so we don't re-attempt
// the import on every subsequent missing-usage call.
function loadModule(): Promise<TiktokenModule | null> {
  if (modulePromise) return modulePromise;
  modulePromise = import("@dqbd/tiktoken")
    .then((m) => m as TiktokenModule)
    .catch(() => {
      if (!warned) {
        warned = true;
        // Single warning per process so a chatty long-tail provider doesn't
        // spam the customer's logs.
        console.warn(
          "[spaturzu] streaming response had no `usage` payload and " +
            "@dqbd/tiktoken is not installed; token counts will be omitted. " +
            "Install @dqbd/tiktoken (peer dep) to enable estimation.",
        );
      }
      return null;
    });
  return modulePromise;
}

// Pick the closest tiktoken encoding for a model name.
//
// `encoding_for_model()` would be the cleanest API but it throws on unknown
// model names — and the whole point of this fallback is unknown providers.
// Hand-rolled mapping covers the OpenAI families precisely; everything else
// falls through to cl100k_base, which is the de-facto convention for
// "rough estimate" against non-OpenAI tokenizers (Anthropic ~30% lower in
// practice; we surface the row as `tiktoken`-sourced so the dashboard can
// flag the lower confidence).
function pickEncoding(model: string): TiktokenEncoding {
  const m = model.toLowerCase();
  if (
    m.startsWith("gpt-4o") ||
    m.startsWith("gpt-4.1") ||
    m.startsWith("gpt-5") ||
    m.startsWith("o1") ||
    m.startsWith("o3") ||
    m.startsWith("o4")
  ) {
    return "o200k_base";
  }
  // gpt-3.5/gpt-4 (legacy) and almost every Together/Groq/Anyscale model use
  // cl100k_base or a near-equivalent. Anthropic models too: their actual
  // tokenizer differs but cl100k is close enough for cost-trend purposes.
  return "cl100k_base";
}

async function getEncoder(model: string): Promise<TiktokenInstance | null> {
  const mod = await loadModule();
  if (!mod) return null;
  const name = pickEncoding(model);
  let enc = encoderCache.get(name);
  if (enc) return enc;
  try {
    enc = mod.get_encoding(name);
    encoderCache.set(name, enc);
    return enc;
  } catch {
    // Defensive: if WASM init fails for some exotic reason, cache nothing
    // so a future call retries. Returning null keeps the caller on the
    // "no token data" path.
    return null;
  }
}

/** Tokenize `text` using the encoding closest to `model`. Returns null when
 *  the optional peer dep isn't installed or the encoder couldn't be loaded —
 *  callers should treat null as "no estimate available" and log without
 *  token counts. Returns 0 cleanly for empty input. */
export async function estimateTokens(
  model: string,
  text: string,
): Promise<number | null> {
  if (!text) return 0;
  const enc = await getEncoder(model);
  if (!enc) return null;
  // `disallowed_special: "all"` is the default — we pass plain text, not
  // chat-template-encoded text, so the special-token guard is irrelevant.
  return enc.encode(text).length;
}
