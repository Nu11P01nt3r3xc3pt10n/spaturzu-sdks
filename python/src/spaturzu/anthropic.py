"""Drop-in replacement for ``anthropic``. Swap only the import::

    - from anthropic import Anthropic
    + from spaturzu.anthropic import Anthropic
"""
from __future__ import annotations
from typing import Any, Optional

from anthropic import Anthropic as _RealAnthropic, AsyncAnthropic as _RealAsyncAnthropic
from . import get_default_spaturzu


def Anthropic(*args: Any, spaturzu: Optional[dict] = None, **kwargs: Any) -> Any:
    return get_default_spaturzu().wrap_anthropic(
        _RealAnthropic(*args, **kwargs), **(spaturzu or {})
    )


def AsyncAnthropic(*args: Any, spaturzu: Optional[dict] = None, **kwargs: Any) -> Any:
    return get_default_spaturzu().wrap_anthropic(
        _RealAsyncAnthropic(*args, **kwargs), **(spaturzu or {})
    )
