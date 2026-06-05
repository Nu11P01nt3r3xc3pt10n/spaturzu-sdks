"""Lazy fallback tokenizer for streams that arrive without a ``usage`` payload.

Mirrors `sdks/typescript/src/tiktoken.ts`. Python's `tiktoken` is the canonical
implementation (Microsoft, native Rust under the hood) so unlike Node we
don't need a single-flight import promise — Python's import system caches
modules globally. We still cache encoders per encoding name to avoid
re-loading the BPE tables.

Optional install:

    pip install spaturzu[tiktoken]

When the package is missing we warn once per process and return ``None``
so callers degrade gracefully (the row gets logged without token counts).
"""

from __future__ import annotations

import sys
from threading import Lock
from typing import Any, Optional

# Cached attempt: True if we've successfully imported tiktoken; False if
# the import failed (don't keep retrying). None until first call.
_module: Optional[Any] = None
_load_attempted = False
_load_lock = Lock()
_warned = False
_encoder_cache: dict[str, Any] = {}


def _load_module() -> Optional[Any]:
    """Single-flight import of the optional ``tiktoken`` package.

    Returns the imported module, or ``None`` if it isn't installed.
    Subsequent calls return the cached result without re-attempting.
    """
    global _module, _load_attempted, _warned
    if _load_attempted:
        return _module
    with _load_lock:
        if _load_attempted:
            return _module
        try:
            import tiktoken  # type: ignore[import-not-found]

            _module = tiktoken
        except ImportError:
            if not _warned:
                _warned = True
                # One warning per process so a chatty long-tail provider
                # doesn't spam the customer's logs.
                print(
                    "[spaturzu] streaming response had no `usage` payload "
                    "and `tiktoken` is not installed; token counts will be "
                    "omitted. Install spaturzu[tiktoken] to enable estimation.",
                    file=sys.stderr,
                )
            _module = None
        _load_attempted = True
    return _module


def _pick_encoding(model: str) -> str:
    """Pick the closest tiktoken encoding for a model name.

    ``encoding_for_model`` would be cleaner but throws on unknown model
    names — and the whole point of this fallback is unknown providers.
    Hand-rolled prefix mapping covers the OpenAI families precisely;
    everything else falls through to ``cl100k_base``, the de-facto
    convention for "rough estimate" against non-OpenAI tokenizers
    (Anthropic ~30% lower in practice; the row is flagged as
    ``tiktoken``-sourced so the dashboard can mark lower confidence).
    """
    m = model.lower()
    if (
        m.startswith("gpt-4o")
        or m.startswith("gpt-4.1")
        or m.startswith("gpt-5")
        or m.startswith("o1")
        or m.startswith("o3")
        or m.startswith("o4")
    ):
        return "o200k_base"
    return "cl100k_base"


def _get_encoder(model: str) -> Optional[Any]:
    mod = _load_module()
    if mod is None:
        return None
    name = _pick_encoding(model)
    enc = _encoder_cache.get(name)
    if enc is not None:
        return enc
    try:
        enc = mod.get_encoding(name)
    except Exception:
        # Defensive: if some exotic encoding name fails to load, leave the
        # cache untouched so a future call retries cleanly. Returning None
        # keeps the caller on the "no token data" path.
        return None
    _encoder_cache[name] = enc
    return enc


def estimate_tokens(model: str, text: str) -> Optional[int]:
    """Tokenize ``text`` using the encoding closest to ``model``.

    Returns ``None`` when ``tiktoken`` isn't installed or the encoder
    failed to load — callers should treat ``None`` as "no estimate
    available" and log without token counts. Returns ``0`` cleanly for
    empty input.
    """
    if not text:
        return 0
    enc = _get_encoder(model)
    if enc is None:
        return None
    return len(enc.encode(text))
