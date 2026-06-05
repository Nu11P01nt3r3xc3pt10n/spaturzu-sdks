"""Frame propagation for per-agent attribution.

Mirrors `sdks/typescript/src/context.ts`: each call to ``spaturzu.run("name")``
opens a fresh frame, pushes it onto a ``contextvars.ContextVar`` (the Python
equivalent of Node's ``AsyncLocalStorage``), and pops it on exit. Frames
inherit ``run_id`` and extend ``agent_path``; child frame tags merge on top
of the parent's, so an inner ``run("nested", tags={"team": "ml"})`` overrides
the outer ``team`` value while keeping the rest.

Why a ContextVar (not threading.local): an asyncio Task gets its own context
copy, so concurrent tasks see independent frames after ``.set()`` in one
task. ``threading.local`` would only isolate threads, not coroutines.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4


@dataclass
class RunFrame:
    """Per-agent attribution state propagated through ``ContextVar``.

    ``last_request_id`` is intentionally mutable: each wrapped call sets it
    on the *current* frame so that a subsequent ``run()`` nested below
    captures it as the new frame's ``parent_request_id``. Mutation is safe
    because the frame is private to each ``ContextVar.set`` scope; sibling
    runs get distinct frames.
    """

    run_id: str
    agent_name: str
    agent_path: list[str]
    parent_request_id: Optional[str] = None
    last_request_id: Optional[str] = None
    tags: Optional[dict[str, str]] = None


# Sentinel default: frames are explicit, not inherited from process state.
_frame_var: ContextVar[Optional[RunFrame]] = ContextVar(
    "spaturzu_frame", default=None
)


def get_current_frame() -> Optional[RunFrame]:
    """Return the active frame, or ``None`` outside any ``spaturzu.run`` block."""
    return _frame_var.get()


def set_last_request_id(request_id: str) -> None:
    """Mark the most-recent request id on the current frame.

    Wrappers call this *before* the upstream HTTP call so that a concurrent
    nested ``run()`` reads it as ``parent_request_id``. No-op outside a frame.
    """
    frame = _frame_var.get()
    if frame is not None:
        frame.last_request_id = request_id


def merge_tags(
    a: Optional[dict[str, str]], b: Optional[dict[str, str]]
) -> Optional[dict[str, str]]:
    """Shallow-merge two tag bags; ``b`` wins on key conflict.

    Returns ``None`` if both are empty so callers can drop the field
    entirely instead of serialising ``{}``.
    """
    if not a and not b:
        return None
    if not a:
        return dict(b) if b else None
    if not b:
        return dict(a)
    out = dict(a)
    out.update(b)
    return out


def normalise_tags(
    tags: Optional[dict] = None,
) -> Optional[dict[str, str]]:
    """Coerce primitive tag values to strings; drop None/empty.

    Mirrors the Node `normalizeTags`. Returns None when nothing remains so
    callers can omit the field rather than serialise an empty dict.
    """
    if not tags:
        return None
    out: dict[str, str] = {}
    for k, v in tags.items():
        if v is None:
            continue
        out[k] = v if isinstance(v, str) else str(v)
    return out or None


@dataclass
class AgentBinding:
    """Binds a wrapped client to an agent name (+ optional frame-scoped tags)
    so each terminal call runs against a derived child frame. Produced by the
    `.with_agent()` accessor on a wrapped client."""

    agent: str
    tags: Optional[dict[str, str]] = None


def derive_frame(
    parent: Optional[RunFrame],
    agent_name: str,
    tags: Optional[dict[str, str]],
) -> RunFrame:
    """Build a child RunFrame from `parent` (run_id inherited or new,
    agent_path extended, parent_request_id from the parent's last_request_id,
    tags merged). Shared by `open_frame` and the `.with_agent()` capture path."""
    merged = merge_tags(parent.tags if parent else None, tags)
    return RunFrame(
        run_id=parent.run_id if parent else str(uuid4()),
        agent_name=agent_name,
        agent_path=([*parent.agent_path, agent_name] if parent else [agent_name]),
        parent_request_id=parent.last_request_id if parent else None,
        last_request_id=None,
        tags=merged,
    )


def open_frame(
    agent_name: str, tags: Optional[dict[str, str]] = None
) -> tuple[RunFrame, Token]:
    """Push a new frame onto the ContextVar.

    Returns ``(frame, token)``; the token must be passed to
    ``close_frame`` to restore the previous frame. Caller is responsible
    for pairing — the public API wraps this in a context manager so user
    code never has to.
    """
    parent = _frame_var.get()
    frame = derive_frame(parent, agent_name, tags)
    token = _frame_var.set(frame)
    return frame, token


def close_frame(token: Token) -> None:
    """Restore the parent frame using the token from ``open_frame``."""
    _frame_var.reset(token)
