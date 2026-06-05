"""Drop-in replacement for ``openai``. Swap only the import::

    - from openai import OpenAI
    + from spaturzu.openai import OpenAI

Construction and call sites are unchanged. Spaturzu metering config is read
from env (SPATURZU_API_KEY / SPATURZU_BASE_URL) via the default singleton;
call ``spaturzu.configure(...)`` once at startup for tags / on_error.

Pass ``spaturzu={"budget": {...}, "fallback": [...]}`` (keyword-only) to reach
budget hard-caps / cross-provider fallback on the drop-in path.
"""
from __future__ import annotations
from typing import Any, Optional

from openai import OpenAI as _RealOpenAI, AsyncOpenAI as _RealAsyncOpenAI
from . import get_default_spaturzu


def OpenAI(*args: Any, spaturzu: Optional[dict] = None, **kwargs: Any) -> Any:
    return get_default_spaturzu().wrap_openai(
        _RealOpenAI(*args, **kwargs), **(spaturzu or {})
    )


def AsyncOpenAI(*args: Any, spaturzu: Optional[dict] = None, **kwargs: Any) -> Any:
    return get_default_spaturzu().wrap_openai(
        _RealAsyncOpenAI(*args, **kwargs), **(spaturzu or {})
    )
