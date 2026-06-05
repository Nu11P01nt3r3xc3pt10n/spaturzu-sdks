"""Unit tests for spaturzu._context (mirror of test/context.test.ts)."""

from __future__ import annotations

import asyncio
import re
import pytest

from spaturzu._context import (
    RunFrame,
    AgentBinding,
    close_frame,
    derive_frame,
    get_current_frame,
    merge_tags,
    normalise_tags,
    open_frame,
    set_last_request_id,
)


UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def test_open_frame_root_has_fresh_runid_and_single_path() -> None:
    frame, token = open_frame("root")
    try:
        assert UUID_RE.match(frame.run_id)
        assert frame.agent_name == "root"
        assert frame.agent_path == ["root"]
        assert frame.parent_request_id is None
        assert frame.last_request_id is None
        assert get_current_frame() is frame
    finally:
        close_frame(token)
    assert get_current_frame() is None


def test_get_current_frame_returns_none_outside_any_frame() -> None:
    assert get_current_frame() is None


def test_nested_frame_inherits_run_id_and_extends_agent_path() -> None:
    outer, outer_tok = open_frame("outer")
    try:
        inner, inner_tok = open_frame("inner")
        try:
            assert inner.run_id == outer.run_id
            assert inner.agent_path == ["outer", "inner"]
        finally:
            close_frame(inner_tok)
    finally:
        close_frame(outer_tok)


def test_nested_frame_parent_request_id_picks_up_outer_last_request_id() -> None:
    _, outer_tok = open_frame("outer")
    try:
        set_last_request_id("req-from-outer")
        inner, inner_tok = open_frame("inner")
        try:
            assert inner.parent_request_id == "req-from-outer"
        finally:
            close_frame(inner_tok)
    finally:
        close_frame(outer_tok)


def test_inner_tags_inherit_and_override_outer() -> None:
    _, outer_tok = open_frame("outer", {"env": "dev", "region": "us-east-1"})
    try:
        inner, inner_tok = open_frame("inner", {"env": "prod", "team": "search"})
        try:
            assert inner.tags == {"env": "prod", "region": "us-east-1", "team": "search"}
        finally:
            close_frame(inner_tok)
    finally:
        close_frame(outer_tok)


def test_set_last_request_id_is_noop_outside_frame() -> None:
    # Must not raise.
    set_last_request_id("any")


def test_set_last_request_id_mutates_current_frame() -> None:
    frame, token = open_frame("agent")
    try:
        set_last_request_id("req-1")
        assert frame.last_request_id == "req-1"
        assert get_current_frame() is frame
        assert get_current_frame().last_request_id == "req-1"  # type: ignore[union-attr]
    finally:
        close_frame(token)


def test_merge_tags_returns_none_when_both_empty() -> None:
    assert merge_tags(None, None) is None
    assert merge_tags({}, {}) is None


def test_merge_tags_returns_fresh_copy_of_a_when_b_is_none() -> None:
    a = {"x": "1"}
    out = merge_tags(a, None)
    assert out == {"x": "1"}
    assert out is not a


def test_merge_tags_b_wins_on_conflict() -> None:
    assert merge_tags({"x": "1", "y": "2"}, {"x": "3"}) == {"x": "3", "y": "2"}


@pytest.mark.asyncio
async def test_parallel_asyncio_tasks_see_isolated_frames() -> None:
    seen: list[str] = []

    async def in_sub(name: str) -> None:
        _, tok = open_frame(name)
        try:
            cur = get_current_frame()
            assert cur is not None
            seen.append(cur.agent_name)
        finally:
            close_frame(tok)

    _, tok = open_frame("root")
    try:
        await asyncio.gather(in_sub("a"), in_sub("b"))
    finally:
        close_frame(tok)
    assert sorted(seen) == ["a", "b"]


def test_normalise_tags_coerces_and_drops_empty():
    assert normalise_tags({"env": "prod", "n": 3, "b": True}) == {
        "env": "prod", "n": "3", "b": "True",
    }
    assert normalise_tags(None) is None
    assert normalise_tags({}) is None


def test_derive_frame_root_and_child():
    root = derive_frame(None, "writer", None)
    assert root.agent_name == "writer"
    assert root.agent_path == ["writer"]
    assert isinstance(root.run_id, str)

    parent = RunFrame(run_id="r1", agent_name="planner", agent_path=["planner"],
                      last_request_id="req-1", tags={"env": "prod"})
    child = derive_frame(parent, "writer", {"team": "ml"})
    assert child.run_id == "r1"
    assert child.agent_path == ["planner", "writer"]
    assert child.parent_request_id == "req-1"
    assert child.tags == {"env": "prod", "team": "ml"}


def test_agent_binding_fields():
    b = AgentBinding("writer", {"k": "v"})
    assert b.agent == "writer" and b.tags == {"k": "v"}
