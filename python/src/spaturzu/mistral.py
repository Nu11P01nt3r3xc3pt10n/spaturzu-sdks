"""Drop-in replacement for ``mistralai``. Swap only the import::

    - from mistralai import Mistral
    + from spaturzu.mistral import Mistral
"""
from __future__ import annotations
from typing import Any, Optional

# The Mistral class is exported at the package root in the mainline SDK
# (`from mistralai import Mistral`), but some installed layouts expose it only
# under `mistralai.client`. Try the documented path first, fall back so the
# drop-in works across versions.
try:  # pragma: no cover - import-path shim
    from mistralai import Mistral as _RealMistral
except ImportError:  # pragma: no cover
    from mistralai.client import Mistral as _RealMistral
from . import get_default_spaturzu


def Mistral(*args: Any, spaturzu: Optional[dict] = None, **kwargs: Any) -> Any:
    return get_default_spaturzu().wrap_mistral(
        _RealMistral(*args, **kwargs), **(spaturzu or {})
    )
