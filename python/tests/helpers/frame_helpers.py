"""Tiny utility for setting up frame state in tests.

Wraps ``spaturzu._context.open_frame`` / ``close_frame`` as a context
manager so tests look like::

    with in_frame("agent", tags={"env": "prod"}) as frame:
        ...
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from spaturzu._context import RunFrame, open_frame, close_frame


@contextmanager
def in_frame(
    agent_name: str, tags: Optional[dict[str, str]] = None
) -> Iterator[RunFrame]:
    frame, token = open_frame(agent_name, tags)
    try:
        yield frame
    finally:
        close_frame(token)
