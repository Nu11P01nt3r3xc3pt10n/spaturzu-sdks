"""Drop-in Logger replacement: captures every entry in-memory."""

from __future__ import annotations

from typing import Any


class FakeLogger:
    """Structural compatibility with spaturzu._logger.Logger.

    The wraps only ever call ``.log(entry)``; ``flush`` / ``close`` are
    no-ops here. Tests inspect ``.entries`` to assert on the payload.
    """

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def log(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)

    def flush(self, timeout_s: float | None = None) -> None:  # pragma: no cover
        pass

    def close(self) -> None:  # pragma: no cover
        pass
