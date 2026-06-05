"""Unit tests for spaturzu._budget pre-call gate.

The Python BudgetGuard runs an SSE thread + a polling thread; tests stub
``httpx.Client.get`` / ``httpx.Client.post`` / ``httpx.Client.stream`` so
neither network nor the background threads escape the test process.

Key implementation notes discovered from reading _budget.py:
- _sse_loop creates a NEW httpx.Client per attempt (inside a `with` block),
  so we patch httpx.Client.stream on the class.
- _parse_sse calls response.iter_text(), not iter_lines(); stub implements that.
- BudgetGuard.__init__ does NOT call start(); tests omit start() since
  pre_call_check only uses the synchronous _get_policy/_fetch_policy path.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from spaturzu._budget import AgentPolicy, BudgetExceededError, BudgetGuard


POLICY_OK_JSON: dict[str, Any] = {
    "agent": "agent-a",
    "agentId": "id-a",
    "limits": [
        {
            "budgetId": "b1",
            "scope": "agent",
            "period": "monthly",
            "limitCost": "100.00",
            "currentCost": "10.00",
            "pctUsed": 10,
            "alertThresholdPct": 80,
            "breached": False,
            "enforcement": "hard_cap",
        }
    ],
}


def _breached(json_doc: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(json_doc))
    out["limits"][0]["breached"] = True
    out["limits"][0]["currentCost"] = "150.00"
    return out


class _FakeResponse:
    def __init__(self, status: int, body: Any) -> None:
        self.status_code = status
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self) -> Any:
        return self._body

    @property
    def is_success(self) -> bool:
        return self.status_code < 400

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("status", request=None, response=None)  # type: ignore[arg-type]


def _patch_get(monkeypatch, behaviour) -> None:
    def fake_get(self: httpx.Client, url: str, headers: dict[str, str] | None = None) -> _FakeResponse:  # type: ignore[override]
        return behaviour(url)

    monkeypatch.setattr(httpx.Client, "get", fake_get, raising=False)


def _stub_sse_stream(monkeypatch) -> None:
    """Defensive SSE stub.

    BudgetGuard.__init__ does NOT auto-start the SSE thread; the thread
    spawns only via guard.start(). The current pre_call_check tests never
    call start(), so this stub is inert today. Kept for when a future
    test exercises the SSE path explicitly.
    """

    class _StreamCtx:
        def __enter__(self):
            class _Resp:
                status_code = 200

                def iter_text(self):
                    # Block indefinitely — tests do not exercise SSE delivery.
                    while True:
                        import time as _t

                        _t.sleep(60)

            return _Resp()

        def __exit__(self, *a):  # noqa: D401
            return False

    def fake_stream(
        self: httpx.Client,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> _StreamCtx:  # type: ignore[override]
        return _StreamCtx()

    monkeypatch.setattr(httpx.Client, "stream", fake_stream, raising=False)


def test_pre_call_check_returns_normally_when_no_policy(monkeypatch) -> None:
    _stub_sse_stream(monkeypatch)
    _patch_get(monkeypatch, lambda url: _FakeResponse(404, "not found"))
    guard = BudgetGuard(base_url="https://gw.example")
    # No raise.
    guard.pre_call_check("agent-a", "throw")
    guard.close()


def test_pre_call_check_passes_for_unbreached_hard_cap(monkeypatch) -> None:
    _stub_sse_stream(monkeypatch)
    _patch_get(monkeypatch, lambda url: _FakeResponse(200, POLICY_OK_JSON))
    guard = BudgetGuard(base_url="https://gw.example")
    guard.pre_call_check("agent-a", "throw")
    guard.close()


def test_pre_call_check_throws_on_breached_hard_cap(monkeypatch) -> None:
    _stub_sse_stream(monkeypatch)
    _patch_get(monkeypatch, lambda url: _FakeResponse(200, _breached(POLICY_OK_JSON)))
    guard = BudgetGuard(base_url="https://gw.example")
    with pytest.raises(BudgetExceededError):
        guard.pre_call_check("agent-a", "throw")
    guard.close()


def test_pre_call_check_warns_and_proceeds_on_breach(monkeypatch, caplog) -> None:
    _stub_sse_stream(monkeypatch)
    _patch_get(monkeypatch, lambda url: _FakeResponse(200, _breached(POLICY_OK_JSON)))
    guard = BudgetGuard(base_url="https://gw.example")
    with caplog.at_level("WARNING", logger="spaturzu.budget"):
        guard.pre_call_check("agent-a", "warn")
    assert any("hard-cap" in rec.message.lower() for rec in caplog.records)
    guard.close()


def test_pre_call_check_caches_policy(monkeypatch) -> None:
    _stub_sse_stream(monkeypatch)
    calls = {"n": 0}

    def behaviour(url: str) -> _FakeResponse:
        if "/policy" in url:
            calls["n"] += 1
        return _FakeResponse(200, POLICY_OK_JSON)

    _patch_get(monkeypatch, behaviour)
    guard = BudgetGuard(base_url="https://gw.example")
    guard.pre_call_check("agent-a", "throw")
    guard.pre_call_check("agent-a", "throw")
    guard.pre_call_check("agent-a", "throw")
    assert calls["n"] == 1
    guard.close()


def test_pre_call_check_uses_untagged_sentinel_when_agent_name_is_none(monkeypatch) -> None:
    _stub_sse_stream(monkeypatch)
    captured = {"url": None}

    def behaviour(url: str) -> _FakeResponse:
        captured["url"] = url
        return _FakeResponse(200, POLICY_OK_JSON)

    _patch_get(monkeypatch, behaviour)
    guard = BudgetGuard(base_url="https://gw.example")
    guard.pre_call_check(None, "throw")
    assert captured["url"] is not None
    assert "%28untagged%29" in captured["url"]  # URL-encoded "(untagged)"
    guard.close()
