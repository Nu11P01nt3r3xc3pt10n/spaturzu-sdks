"""Drop-in replacement for ``google.genai``. Swap only the import::

    - from google.genai import Client
    + from spaturzu.google import Client

One client covers sync (.models.*) and async (.aio.models.*) calls.
"""
from __future__ import annotations
from typing import Any, Optional

from google.genai import Client as _RealClient
from . import get_default_spaturzu


def Client(*args: Any, spaturzu: Optional[dict] = None, **kwargs: Any) -> Any:
    return get_default_spaturzu().wrap_gemini(
        _RealClient(*args, **kwargs), **(spaturzu or {})
    )
