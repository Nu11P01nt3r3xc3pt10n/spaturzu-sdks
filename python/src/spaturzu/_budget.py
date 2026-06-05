"""Day 20 — hard-cap budget enforcement (Python side).

Mirrors `sdks/typescript/src/budget.ts`. The gateway routes are language-
agnostic; only the client-side cache + SSE consumer differs.

Concurrency model: threads, mirroring `_logger.py`. One daemon thread runs
the SSE reader; another runs 60s polling. Both use `httpx.Client` (sync)
because the existing dependency surface is sync-only and a sync HTTP call
inside a daemon thread doesn't block the customer's event loop.

The `pre_call_check` itself is a SYNC method. For async users this means
the FIRST call for each new agent name briefly blocks the event loop on
the policy fetch (~50–200ms LAN). Subsequent calls hit the in-memory
cache (sub-µs). Same trade-off the TS implementation makes.

Race window: ≤1s drift between server state and SDK enforcement is the
spec budget. Sources of drift:

* Gateway ingest publishes the breach edge AFTER the /v1/logs TX commits.
* SSE delivery → background thread updates the cache → next pre-check
  reads the new value. Sub-100ms LAN.
* The 60s poll is a backstop; it doesn't drive the drift budget.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.parse import quote

import httpx


_log = logging.getLogger("spaturzu.budget")


# Marker name the server treats as "no agent row matched, return only
# project-scope budgets." The TS SDK uses the same literal.
UNTAGGED_AGENT_NAME = "(untagged)"


@dataclass
class PolicyLimit:
    budget_id: str
    scope: str  # 'project' | 'agent'
    period: str  # 'daily' | 'monthly'
    limit_cost: str
    current_cost: str
    pct_used: float
    alert_threshold_pct: int
    breached: bool
    enforcement: str  # 'alert' | 'hard_cap'

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "PolicyLimit":
        return cls(
            budget_id=str(d.get("budgetId", "")),
            scope=str(d.get("scope", "")),
            period=str(d.get("period", "")),
            limit_cost=str(d.get("limitCost", "0")),
            current_cost=str(d.get("currentCost", "0")),
            pct_used=float(d.get("pctUsed", 0.0)),
            alert_threshold_pct=int(d.get("alertThresholdPct", 80)),
            breached=bool(d.get("breached", False)),
            enforcement=str(d.get("enforcement", "alert")),
        )


@dataclass
class AgentPolicy:
    agent: str
    agent_id: Optional[str]
    limits: list[PolicyLimit]

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "AgentPolicy":
        return cls(
            agent=str(d.get("agent", "")),
            agent_id=d.get("agentId"),
            limits=[PolicyLimit.from_json(x) for x in (d.get("limits") or [])],
        )


class BudgetExceededError(Exception):
    """Raised from a wrapped ``create`` when a hard-cap budget is breached.

    Mirrors the TS ``BudgetExceededError``. The wrapped call does NOT hit
    the upstream provider — no token cost is incurred.
    """

    code = "budget_exceeded"

    def __init__(
        self,
        *,
        scope: str,
        period: str,
        limit_cost: str,
        current_cost: str,
        agent_name: Optional[str],
    ) -> None:
        msg = (
            f"spaturzu: {scope}/{period} budget exceeded "
            f"({current_cost} >= {limit_cost})"
        )
        if agent_name:
            msg += f" for agent {agent_name!r}"
        super().__init__(msg)
        self.scope = scope
        self.period = period
        self.limit_cost = limit_cost
        self.current_cost = current_cost
        self.agent_name = agent_name


class BudgetGuard:
    """Cache + SSE listener for per-agent budget policy.

    One instance is created lazily by ``spaturzu`` when a customer opts into
    hard-cap enforcement on any wrap. Threads start on construction and
    stop on ``close()``; both are daemonised so the process can exit
    without explicit cleanup.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str] = None,
        refresh_interval_s: float = 60.0,
        sse_retry_s: float = 5.0,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._refresh_interval_s = refresh_interval_s
        self._sse_retry_s = sse_retry_s
        self._on_error = on_error or (lambda _e: None)

        self._policies: dict[str, AgentPolicy] = {}
        # RLock because _process_sse_message may want to re-enter via
        # _fetch_policy which also touches the dict.
        self._lock = threading.RLock()

        # `httpx.Client` is reused across policy fetches. The SSE stream
        # opens its own short-lived client so an abrupt close from the
        # server doesn't poison the shared pool.
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)
        )
        self._stop_evt = threading.Event()
        self._sse_thread: Optional[threading.Thread] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._started = False

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._stop_evt.clear()
        self._sse_thread = threading.Thread(
            target=self._sse_loop, name="spaturzu-sse", daemon=True
        )
        self._sse_thread.start()
        self._poll_thread = threading.Thread(
            target=self._poll_loop, name="spaturzu-poll", daemon=True
        )
        self._poll_thread.start()

    def close(self) -> None:
        if not self._started:
            return
        self._started = False
        self._stop_evt.set()
        try:
            self._client.close()
        except Exception:
            pass
        # Don't join — daemon threads will exit on next wake. Joining would
        # add up to `refresh_interval_s` of shutdown latency.

    # ── public pre-call gate ──────────────────────────────────────────

    def pre_call_check(
        self, agent_name: Optional[str], on_breach: str = "throw"
    ) -> None:
        """Refuse the call when an applicable hard-cap budget is breached.

        Raises ``BudgetExceededError`` on ``on_breach='throw'`` (default).
        Logs a warning and returns on ``on_breach='warn'``. Returns
        without action when no policy applies or nothing is breached.
        """
        name = agent_name or UNTAGGED_AGENT_NAME
        policy = self._get_policy(name)
        if policy is None:
            return
        for limit in policy.limits:
            if limit.enforcement != "hard_cap":
                continue
            if not limit.breached:
                continue
            if on_breach == "warn":
                _log.warning(
                    "[spaturzu] hard-cap budget breached (%s/%s): "
                    "%s >= %s — proceeding (on_breach='warn')",
                    limit.scope,
                    limit.period,
                    limit.current_cost,
                    limit.limit_cost,
                )
                return
            raise BudgetExceededError(
                scope=limit.scope,
                period=limit.period,
                limit_cost=limit.limit_cost,
                current_cost=limit.current_cost,
                agent_name=agent_name,
            )

    # ── policy fetch / cache ─────────────────────────────────────────

    def _get_policy(self, name: str) -> Optional[AgentPolicy]:
        with self._lock:
            cached = self._policies.get(name)
        if cached is not None:
            return cached
        return self._fetch_policy(name)

    def _fetch_policy(self, name: str) -> Optional[AgentPolicy]:
        try:
            url = f"{self._base_url}/v1/agents/{quote(name, safe='')}/policy"
            headers: dict[str, str] = {}
            if self._api_key:
                headers["x-spaturzu-key"] = self._api_key
            r = self._client.get(url, headers=headers)
            if r.status_code != 200:
                self._on_error(
                    RuntimeError(f"policy fetch {r.status_code}: {r.text[:200]}")
                )
                return None
            policy = AgentPolicy.from_json(r.json())
            with self._lock:
                self._policies[name] = policy
            return policy
        except Exception as err:  # noqa: BLE001 — surfaced via on_error
            self._on_error(err)
            return None

    def _refresh_all(self) -> None:
        with self._lock:
            names = list(self._policies.keys())
        for name in names:
            if self._stop_evt.is_set():
                return
            self._fetch_policy(name)

    # ── background loops ─────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while not self._stop_evt.wait(self._refresh_interval_s):
            try:
                self._refresh_all()
            except Exception as err:
                self._on_error(err)

    def _sse_loop(self) -> None:
        url = f"{self._base_url}/v1/projects/_me/events"
        headers: dict[str, str] = {"accept": "text/event-stream"}
        if self._api_key:
            headers["x-spaturzu-key"] = self._api_key

        while not self._stop_evt.is_set():
            try:
                # Per-attempt client — abrupt server close cleans up
                # without affecting other in-flight fetches. `read=None`
                # so the long-lived stream doesn't time out on idle.
                with httpx.Client(
                    timeout=httpx.Timeout(
                        connect=5.0, read=None, write=10.0, pool=5.0
                    )
                ) as client:
                    with client.stream("GET", url, headers=headers) as resp:
                        if resp.status_code != 200:
                            self._on_error(
                                RuntimeError(f"events {resp.status_code}")
                            )
                        else:
                            self._parse_sse(resp)
            except Exception as err:  # noqa: BLE001
                if self._stop_evt.is_set():
                    return
                self._on_error(err)
            if self._stop_evt.wait(self._sse_retry_s):
                return

    def _parse_sse(self, response: httpx.Response) -> None:
        buffer = ""
        for chunk in response.iter_text():
            if self._stop_evt.is_set():
                return
            buffer += chunk
            while True:
                # Accept both \n\n and \r\n\r\n message boundaries.
                lf = buffer.find("\n\n")
                crlf = buffer.find("\r\n\r\n")
                if lf == -1 and crlf == -1:
                    break
                if lf == -1:
                    idx, length = crlf, 4
                elif crlf == -1:
                    idx, length = lf, 2
                else:
                    idx, length = (lf, 2) if lf < crlf else (crlf, 4)
                msg = buffer[:idx]
                buffer = buffer[idx + length :]
                self._process_sse_message(msg)

    def _process_sse_message(self, msg: str) -> None:
        event = "message"
        data = ""
        for line in msg.splitlines():
            if not line or line.startswith(":"):
                continue
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        if event != "budget_breach" or not data:
            return
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as err:
            self._on_error(err)
            return
        # Refresh just the affected agent for agent-scope breaches.
        # Project-scope breaches affect every agent — refresh all known.
        scope = payload.get("scope")
        agent_name = payload.get("agentName")
        if scope == "agent" and isinstance(agent_name, str):
            self._fetch_policy(agent_name)
        else:
            self._refresh_all()

    # ── test helper ──────────────────────────────────────────────────

    def _reset_for_tests(self) -> None:
        with self._lock:
            self._policies.clear()
