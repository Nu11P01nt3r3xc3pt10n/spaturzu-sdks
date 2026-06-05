"""End-to-end smoke for the Python SDK.

Equivalent of `scripts/sdk-smoke.ts`. What it proves:

1. ``contextvars``-driven frames propagate through nested ``spaturzu.run``
   in both sync and async code, with streaming + non-streaming calls.
2. The OpenAI wrapper auto-injects ``stream_options.include_usage`` and
   pulls the final usage chunk on streaming responses.
3. The Anthropic wrapper handles ``message_start`` (input + cache_read)
   and the cumulative ``message_delta`` ``output_tokens`` events.
4. The fire-and-forget logger reaches the gateway; ``flush()`` is durable.
5. The gateway's ``/v1/runs/<id>`` reflects a 4-call parent-child tree
   with the right ``agent_path``, ``parent_request_id``, costs.

Mechanics: spins up two stdlib fake-provider servers on random ports
(no real API keys required), wires real ``openai`` / ``anthropic`` SDK
clients to them, and asserts against the live gateway at ``$GATEWAY``
(default ``http://localhost:3000``). API key is provided via
``SMOKE_PY_KEY`` env or — failing that — minted on the fly via
``pnpm key:create smoke-py --quiet`` shelled out from the repo root.

Usage (from the repo root)::

    pnpm smoke:py
    GATEWAY=https://api.staging.spaturzu.dev pnpm smoke:py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# Real provider SDKs — installed via the [openai] / [anthropic] extras.
import openai as _openai_module
from openai import OpenAI, AsyncOpenAI
from anthropic import Anthropic

from spaturzu import spaturzu


# ─── tiny CLI aesthetics ────────────────────────────────────────────────


def _dim(s: str) -> str:
    return f"\x1b[2m{s}\x1b[0m"


def _green(s: str) -> str:
    return f"\x1b[32m{s}\x1b[0m"


def _red(s: str) -> str:
    return f"\x1b[31m{s}\x1b[0m"


def hdr(label: str) -> None:
    print(f"\n{_dim('▸')} {label}")


def ok(label: str) -> None:
    print(f"  {_green('✓')} {label}")


def fail(label: str) -> None:
    print(f"  {_red('✗')} {label}", file=sys.stderr)
    sys.exit(1)


# ─── repo root + key minting ────────────────────────────────────────────


def find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for cand in [here, *here.parents]:
        if (cand / "pnpm-workspace.yaml").exists():
            return cand
    raise RuntimeError(
        "could not locate repo root (no pnpm-workspace.yaml on any parent)"
    )


def mint_key(project_name: str, repo_root: Path) -> str:
    """Shell out to ``pnpm exec tsx scripts/create-key.ts`` in --quiet mode.

    The Node script prints the raw ``spa_…`` key on stdout; everything
    else goes to stderr. We capture stdout for the key.
    """
    proc = subprocess.run(
        [
            "pnpm",
            "exec",
            "tsx",
            "scripts/create-key.ts",
            project_name,
            "--quiet",
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        fail(f"pnpm key:create exited {proc.returncode}")
    return proc.stdout.strip()


# ─── fake OpenAI server ─────────────────────────────────────────────────


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # silence stderr
        return

    def do_POST(self) -> None:
        if not self.path.endswith("/chat/completions"):
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("content-length", "0") or "0")
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        body = json.loads(raw)
        if body.get("stream") is True:
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            # `Connection: close` is critical with Python's http.server (HTTP/1.0,
            # no chunked encoding, no Content-Length): httpx — used by both the
            # openai and anthropic Python SDKs — relies on EOF to terminate the
            # stream, and a stray `keep-alive` makes it block forever waiting
            # for more events.
            self.send_header("connection", "close")
            self.end_headers()
            base = {
                "id": "chatcmpl-fake",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": body.get("model", "unknown"),
            }
            chunks = [
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "hi"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    **base,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "stop"}
                    ],
                },
                # Final usage chunk (only emitted because the SDK auto-injected
                # ``include_usage``).
                {
                    **base,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 800,
                        "completion_tokens": 250,
                        "total_tokens": 1050,
                        "prompt_tokens_details": {"cached_tokens": 100},
                    },
                },
            ]
            for c in chunks:
                self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {
                    "id": "chatcmpl-fake-nonstream",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": body.get("model", "unknown"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "hi back"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1000,
                        "completion_tokens": 500,
                        "total_tokens": 1500,
                        "prompt_tokens_details": {"cached_tokens": 200},
                    },
                }
            ).encode()
        )


# ─── fake Anthropic server ──────────────────────────────────────────────


class _FakeAnthropicHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send(self, event: str, data: dict[str, Any]) -> None:
        self.wfile.write(
            f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()
        )

    def do_POST(self) -> None:
        if not self.path.endswith("/messages"):
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("content-length", "0") or "0")
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        body = json.loads(raw)
        model = body.get("model", "unknown")
        if body.get("stream") is True:
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            # See _FakeOpenAIHandler comment — Python http.server + httpx
            # demands explicit `Connection: close` for SSE EOF semantics.
            self.send_header("connection", "close")
            self.end_headers()
            self._send(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_fake",
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": 2500,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "output_tokens": 0,
                        },
                    },
                },
            )
            self._send(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            self._send(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "ok"},
                },
            )
            self._send(
                "content_block_stop",
                {"type": "content_block_stop", "index": 0},
            )
            self._send(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 1200},
                },
            )
            self._send("message_stop", {"type": "message_stop"})
            return
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {
                    "id": "msg_fake_nonstream",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 1500,
                        "output_tokens": 700,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                }
            ).encode()
        )


def _start_fake_server(handler_cls: type) -> tuple[ThreadingHTTPServer, str, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}", thread


# ─── HTTP helper for /v1/runs ───────────────────────────────────────────


def _poll_run_aggregates(
    gateway: str,
    run_id: str,
    api_key: str,
    *,
    expected: int,
    budget_ms: int,
) -> dict[str, Any]:
    """Day 15: agent_runs aggregates are eventually-consistent.

    Keep fetching `/v1/runs/<id>` until `run.requestCount` reaches the
    expected value, or the budget runs out. The worker drains the
    aggregate inbox every ~1.5s.
    """
    deadline = time.time() + (budget_ms / 1000.0)
    trace: dict[str, Any] = {}
    while time.time() < deadline:
        trace = http_get_json(
            f"{gateway}/v1/runs/{run_id}",
            {"x-spaturzu-key": api_key},
        )
        count = trace.get("run", {}).get("requestCount", 0)
        if count >= expected:
            return trace
        time.sleep(0.5)
    fail(
        f"aggregator did not drain run {run_id} within {budget_ms}ms "
        f"(expected requestCount>={expected}, got "
        f"{trace.get('run', {}).get('requestCount', '?')}); is the workers "
        "process running?"
    )


def http_get_json(url: str, headers: dict[str, str]) -> dict[str, Any]:
    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"GET {url} {e.code}: {body[:200]}") from e
    except URLError as e:
        raise RuntimeError(f"GET {url} network error: {e}") from e


# ─── async leg (separate to validate the AsyncOpenAI path) ──────────────


async def _async_leg(
    spaturzu: spaturzu, fake_oai_url: str
) -> str:
    """One async streaming call inside a fresh frame. Returns the runId so
    we can verify the row landed."""
    aclient = spaturzu.wrap_openai(
        AsyncOpenAI(api_key="test-key", base_url=fake_oai_url + "/v1")
    )
    captured: dict[str, str] = {}
    async with spaturzu.run("async-leg"):
        from spaturzu import get_current_frame

        f = get_current_frame()
        if f is None:
            fail("async leg: no frame inside async with")
        captured["run_id"] = f.run_id
        stream = await aclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        chunks = 0
        async for _ in stream:
            chunks += 1
        if chunks < 3:
            fail(f"async openai stream chunks = {chunks} (expected ≥3)")
        ok(f"async openai stream consumed ({chunks} chunks)")
    return captured["run_id"]


# ─── main ───────────────────────────────────────────────────────────────


def main() -> None:
    gateway = os.environ.get("GATEWAY", "http://localhost:3000").rstrip("/")

    hdr(f"preflight: {gateway}")
    health = http_get_json(f"{gateway}/healthz", {})
    if health.get("ok") is not True:
        fail(f"/healthz .ok != true: {health}")
    ok("GET /healthz .ok = true")

    repo_root = find_repo_root()

    api_key = os.environ.get("SMOKE_PY_KEY")
    if not api_key:
        hdr("minting api key for project 'smoke-py'")
        api_key = mint_key("smoke-py", repo_root)
    ok(f"key ready ({api_key[:8]}…)")

    # Fakes.
    oai_server, oai_url, _ = _start_fake_server(_FakeOpenAIHandler)
    anth_server, anth_url, _ = _start_fake_server(_FakeAnthropicHandler)
    hdr(f"fake OpenAI at {oai_url}, fake Anthropic at {anth_url}")

    spaturzu = spaturzu(
        base_url=gateway,
        api_key=api_key,
        on_error=lambda err, _entry: print(
            "spaturzu log error:", err, file=sys.stderr
        ),
    )

    # Real SDK clients pointed at the fakes.
    openai_client = spaturzu.wrap_openai(
        OpenAI(api_key="test-key", base_url=oai_url + "/v1")
    )
    anthropic_client = spaturzu.wrap_anthropic(
        Anthropic(api_key="test-key", base_url=anth_url + "/")
    )

    captured_run_id: str = ""

    hdr("running nested spaturzu.run() with mixed providers + streaming (sync)")
    with spaturzu.run("researcher"):
        from spaturzu import get_current_frame

        frame = get_current_frame()
        if frame is None:
            fail("captured frame was None")
        captured_run_id = frame.run_id

        # 1: researcher → openai non-stream
        r1 = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
        if getattr(r1, "object", None) != "chat.completion":
            fail("openai response shape")
        ok("openai non-stream call returned")

        # 2: writer → anthropic streaming
        with spaturzu.run("writer"):
            stream = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            count = 0
            for _ in stream:
                count += 1
            if count < 4:
                fail(f"anthropic stream chunks = {count}")
            ok(f"anthropic stream consumed ({count} events)")

            # 3: still in writer → openai streaming
            ostream = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            oc = 0
            for _ in ostream:
                oc += 1
            if oc < 3:
                fail(f"openai stream chunks = {oc}")
            ok(f"openai stream consumed ({oc} chunks)")

        # 4: back in researcher → anthropic non-stream
        r4 = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "again"}],
        )
        if getattr(r4, "type", None) != "message":
            fail("anthropic response shape")
        ok("anthropic non-stream call returned")

    # Async leg — separate run, separate runId.
    hdr("running async leg with AsyncOpenAI")
    async_run_id = asyncio.run(_async_leg(spaturzu, oai_url))

    hdr("flushing logger")
    spaturzu.flush(timeout_s=30.0)
    ok("spaturzu.flush() resolved")

    # Day 15: agent_runs aggregates are eventually-consistent — the
    # gateway enqueues deltas into agent_aggregate_inbox; the worker
    # drains every ~1.5s. Poll until requestCount catches up.
    hdr(f"GET /v1/runs/{captured_run_id} (poll for aggregates)")
    trace = _poll_run_aggregates(
        gateway, captured_run_id, api_key, expected=4, budget_ms=10_000
    )
    run = trace["run"]
    calls = trace["calls"]
    ok(f"aggregator caught up: requestCount={run['requestCount']}")

    if run["rootAgent"] != "researcher":
        fail(f"run.rootAgent = {run['rootAgent']} (expected researcher)")
    ok(f"run.rootAgent = {run['rootAgent']}")
    if run["requestCount"] != 4:
        fail(f"run.requestCount = {run['requestCount']} (expected 4)")
    ok(f"run.requestCount = {run['requestCount']}")
    expected_in = 1000 + 2500 + 800 + 1500
    expected_out = 500 + 1200 + 250 + 700
    if run["totalTokensInput"] != expected_in:
        fail(f"run.totalTokensInput = {run['totalTokensInput']} (expected {expected_in})")
    ok(f"run.totalTokensInput = {run['totalTokensInput']}")
    if run["totalTokensOutput"] != expected_out:
        fail(
            f"run.totalTokensOutput = {run['totalTokensOutput']} "
            f"(expected {expected_out})"
        )
    ok(f"run.totalTokensOutput = {run['totalTokensOutput']}")

    if len(calls) != 4:
        fail(f"calls.length = {len(calls)} (expected 4)")
    ok(f"calls.length = {len(calls)}")

    c1, c2, c3, c4 = calls
    expectations = [
        (c1, "researcher", ["researcher"], "gpt-4o-mini", 1000, 500, 200),
        (c2, "writer", ["researcher", "writer"], "claude-3-5-sonnet-20241022", 2500, 1200, None),
        (c3, "writer", ["researcher", "writer"], "gpt-4o-mini", 800, 250, 100),
        (c4, "researcher", ["researcher"], "claude-3-5-sonnet-20241022", 1500, 700, None),
    ]
    for idx, (c, name, path, model, pt, ct, cached) in enumerate(expectations, 1):
        if c["agentName"] != name:
            fail(f"c{idx}.agentName = {c['agentName']} (expected {name})")
        if c["agentPath"] != path:
            fail(f"c{idx}.agentPath = {c['agentPath']} (expected {path})")
        if c["model"] != model:
            fail(f"c{idx}.model = {c['model']} (expected {model})")
        if c["promptTokens"] != pt:
            fail(f"c{idx}.promptTokens = {c['promptTokens']} (expected {pt})")
        if c["completionTokens"] != ct:
            fail(
                f"c{idx}.completionTokens = {c['completionTokens']} (expected {ct})"
            )
        if cached is not None and c["cachedInputTokens"] != cached:
            fail(
                f"c{idx}.cachedInputTokens = {c['cachedInputTokens']} (expected {cached})"
            )
        ok(
            f"c{idx}: {name} / {model} / {pt}/{ct}"
            + (f" cached={cached}" if cached is not None else "")
        )

    # Parent-request-id: writer's two calls share the parent (researcher's c1).
    if c1["parentRequestId"] is not None:
        fail(f"c1.parentRequestId = {c1['parentRequestId']} (expected null)")
    if c4["parentRequestId"] is not None:
        fail(f"c4.parentRequestId = {c4['parentRequestId']} (expected null)")
    if not c2["parentRequestId"] or c2["parentRequestId"] != c3["parentRequestId"]:
        fail("c2 and c3 must share parentRequestId")
    ok("writer calls share parentRequestId (= researcher's c1)")

    for c in calls:
        if c["totalCost"] is None:
            fail(f"{c['agentName']} totalCost is null")
    ok("all four calls priced (cost_events written)")

    # Async leg row check — same eventual-consistency caveat as above.
    hdr(f"GET /v1/runs/{async_run_id}  (async leg, poll for aggregates)")
    async_trace = _poll_run_aggregates(
        gateway, async_run_id, api_key, expected=1, budget_ms=10_000
    )
    if async_trace["run"]["requestCount"] != 1:
        fail(
            f"async leg requestCount = {async_trace['run']['requestCount']} "
            "(expected 1)"
        )
    ac = async_trace["calls"][0]
    if ac["agentName"] != "async-leg":
        fail(f"async leg agentName = {ac['agentName']}")
    if ac["promptTokens"] != 800 or ac["completionTokens"] != 250:
        fail(
            f"async leg tokens = {ac['promptTokens']}/{ac['completionTokens']}"
        )
    ok("async streaming leg attributed correctly")

    # Cleanup the fake servers; the daemon threads exit on process end.
    oai_server.shutdown()
    anth_server.shutdown()

    # ── Day 20 — hard-cap enforcement, SDK side ──────────────────────────
    # Self-contained: a stub gateway server returns a "breached" policy for
    # an agent name and 201's any /v1/logs POST. Verifies that the wrapped
    # ``create`` raises BudgetExceededError before the OpenAI client is
    # invoked. The gateway-side end-to-end (breach detection + SSE push +
    # /v1/agents/:name/policy) is already covered by scripts/hard-cap-smoke.ts.
    _run_hard_cap_section()

    # ── Day 24 — cross-provider fallback, SDK side ───────────────────────
    # Self-contained: stub OpenAI/Anthropic clients (no real provider keys).
    # Uses the live gateway from the same `smoke-py` project. Verifies that
    # a primary 429 triggers the translated fallback, the response keeps the
    # caller's shape, and a non-retryable 401 bypasses the chain entirely.
    _run_fallback_section(gateway, api_key)

    print(f"\n{_green('✓')} python sdk smoke passed (run={captured_run_id})")


def _run_hard_cap_section() -> None:
    from spaturzu import BudgetExceededError

    hdr("Day 20 — hard-cap SDK wrap (against stub gateway)")

    stub_hits = {"create": 0, "logs": 0, "policy": 0, "events": 0}

    class _StubGateway(BaseHTTPRequestHandler):
        # Suppress per-request log spam.
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
            return

        def _json(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            # Tell httpx to not keep the connection alive — matches the
            # `Connection: close` pattern recorded in the feedback memory
            # for httpx + http.server interop.
            self.send_header("connection", "close")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/healthz":
                self._json(200, {"ok": True})
                return
            if self.path.startswith("/v1/agents/"):
                stub_hits["policy"] += 1
                # Always return a breached project-scope hard-cap.
                self._json(
                    200,
                    {
                        "agent": "hard-cap-py-spender",
                        "agentId": None,
                        "limits": [
                            {
                                "budgetId": "00000000-0000-0000-0000-0000000000aa",
                                "scope": "project",
                                "period": "monthly",
                                "limitCost": "0.0001000000",
                                "currentCost": "1.2345000000",
                                "pctUsed": 1234.5,
                                "alertThresholdPct": 80,
                                "breached": True,
                                "enforcement": "hard_cap",
                            }
                        ],
                    },
                )
                return
            if self.path.startswith("/v1/projects/") and self.path.endswith(
                "/events"
            ):
                # Open an empty SSE stream + immediately close. The guard's
                # reconnect loop will retry; we don't care — the test
                # doesn't rely on push events.
                stub_hits["events"] += 1
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("cache-control", "no-cache")
                self.send_header("connection", "close")
                self.end_headers()
                self.wfile.write(b": connected\n\n")
                return
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("content-length", "0") or 0)
            if length:
                self.rfile.read(length)
            if self.path == "/v1/logs":
                stub_hits["logs"] += 1
                self._json(201, {"requestId": "00000000-0000-0000-0000-000000000000"})
                return
            self._json(404, {"error": "not found"})

    server, base, thread = _start_fake_server(_StubGateway)

    # Real OpenAI client; baseURL points to a closed port. If our pre-call
    # gate doesn't run, the request would error on connect — which would
    # surface as a non-BudgetExceededError failure below. Cleaner signal
    # than mocking the OpenAI client.
    openai_client = OpenAI(
        api_key="fake", base_url="http://127.0.0.1:1/v1"
    )

    spaturzu = spaturzu(base_url=base, api_key="fake-key")
    try:
        wrapped = spaturzu.wrap_openai(
            openai_client, budget={"hard_cap": True, "on_breach": "throw"}
        )

        caught: BaseException | None = None
        with spaturzu.run("hard-cap-py-spender"):
            try:
                wrapped.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": "should never reach upstream"}
                    ],
                )
            except BaseException as err:  # noqa: BLE001 — surfaced below
                caught = err

        if caught is None:
            fail("wrap.create() did not throw")
        if not isinstance(caught, BudgetExceededError):
            fail(
                f"expected BudgetExceededError, got "
                f"{type(caught).__name__}: {caught}"
            )
        ok(f"threw BudgetExceededError: {caught}")
        if getattr(caught, "scope", None) != "project":
            fail(f"err.scope = {getattr(caught, 'scope', None)} (expected 'project')")
        if getattr(caught, "agent_name", None) != "hard-cap-py-spender":
            fail(
                f"err.agent_name = {getattr(caught, 'agent_name', None)}"
            )
        if stub_hits["policy"] < 1:
            fail("policy endpoint was not hit (cache stayed cold?)")
        ok(f"policy endpoint hit {stub_hits['policy']}× (cache populated)")
        if stub_hits["logs"] != 0:
            fail(
                f"logs endpoint hit {stub_hits['logs']}× — pre-check did not "
                "block the call"
            )
        ok("logs endpoint saw 0 calls — pre-check refused before fetch")

        # on_breach='warn' lets the call through. The OpenAI client will
        # then try to reach :1 and fail — assert that's the failure mode
        # (not BudgetExceededError).
        wrapped_warn = spaturzu.wrap_openai(
            openai_client, budget={"hard_cap": True, "on_breach": "warn"}
        )
        warn_caught: BaseException | None = None
        with spaturzu.run("hard-cap-py-spender"):
            try:
                wrapped_warn.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "let me through"}],
                )
            except BaseException as err:  # noqa: BLE001
                warn_caught = err
        if isinstance(warn_caught, BudgetExceededError):
            fail("on_breach='warn' incorrectly raised BudgetExceededError")
        ok(
            "on_breach='warn' let the call through "
            f"(raised {type(warn_caught).__name__ if warn_caught else 'nothing'})"
        )
    finally:
        spaturzu.shutdown()
        server.shutdown()


def _run_fallback_section(gateway: str, api_key: str) -> None:
    """Day-24 SDK-side cross-provider fallback.

    Three scenarios against stub clients (no real API keys needed):

      A. Sync wrap: primary OpenAI raises a fake 429; the Anthropic
         fallback returns a fake message. Assert the response keeps
         OpenAI ChatCompletion shape (with translated content + usage)
         and that the caller's model is echoed back.
      B. Async wrap: same as A but with async stubs to exercise
         ``try_fallback_async`` + ``_await_then_log_async`` chain.
      C. Non-retryable 401 on the primary bypasses the chain entirely —
         the fallback stub is never invoked.

    Uses the live gateway from the same ``smoke-py`` project — each
    attempt POSTs to ``/v1/logs`` like a normal call. The primary
    failure logs as ``status=4xx provider=openai``; the successful
    fallback logs as ``status=200 provider=anthropic tags.via=fallback``.
    """

    hdr("Day 24 — cross-provider fallback (sync + async)")

    class _FakeRateLimit(Exception):
        status_code = 429
        # OpenAI's SDK names this class RateLimitError; the dispatcher
        # also accepts that as a retryable signal via class name.
        def __init__(self, msg: str = "fake 429") -> None:
            super().__init__(msg)

    _FakeRateLimit.__name__ = "RateLimitError"

    class _FakeAuthError(Exception):
        status_code = 401

        def __init__(self) -> None:
            super().__init__("fake auth error")

    # Sync stubs ---------------------------------------------------------
    class _FailingOpenAiCompletions:
        def __init__(self, err: Exception) -> None:
            self._err = err
            self.hits = 0

        def create(self, **_kwargs: Any) -> Any:
            self.hits += 1
            raise self._err

    class _FailingOpenAiChat:
        def __init__(self, completions: Any) -> None:
            self.completions = completions

    class _FailingOpenAiClient:
        def __init__(self, err: Exception) -> None:
            self.chat = _FailingOpenAiChat(_FailingOpenAiCompletions(err))

    class _AnthropicOkMessages:
        def __init__(self, text: str) -> None:
            self._text = text
            self.hits = 0

        def create(self, **kwargs: Any) -> Any:
            self.hits += 1
            return SimpleNamespace(
                id="msg_stub",
                type="message",
                role="assistant",
                model=kwargs.get("model", "claude-haiku-4-5"),
                content=[SimpleNamespace(type="text", text=self._text)],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=13, output_tokens=19),
            )

    class _AnthropicOkClient:
        def __init__(self, text: str) -> None:
            self.messages = _AnthropicOkMessages(text)

    # ── Scenario A: sync wrap, primary 429 → Anthropic fallback ─────────
    spaturzu = spaturzu(base_url=gateway, api_key=api_key)
    try:
        primary = _FailingOpenAiClient(_FakeRateLimit())
        anthropic_ok = _AnthropicOkClient("hi from anthropic fallback")
        wrapped = spaturzu.wrap_openai(
            primary,
            fallback=[
                {
                    "provider": "anthropic",
                    "client": anthropic_ok,
                    "model": "claude-haiku-4-5",
                }
            ],
        )

        with spaturzu.run("py-fallback-sync"):
            response = wrapped.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "concise"},
                    {"role": "user", "content": "say hi"},
                ],
            )

        if primary.chat.completions.hits != 1:
            fail(f"sync: expected 1 primary hit, got {primary.chat.completions.hits}")
        if anthropic_ok.messages.hits != 1:
            fail(
                f"sync: expected 1 fallback hit, got {anthropic_ok.messages.hits}"
            )
        if response.choices[0].message.content != "hi from anthropic fallback":
            fail("sync: translated content mismatch")
        if response.model != "gpt-4o-mini":
            fail(f"sync: caller model not preserved (got {response.model})")
        if response.usage.prompt_tokens != 13 or response.usage.completion_tokens != 19:
            fail("sync: usage translation mismatch")
        ok("sync OpenAI 429 → Anthropic fallback, response shape preserved")

        # ── Scenario B: async wrap, primary 429 → Anthropic fallback ────
        class _FailingAsyncCompletions:
            def __init__(self, err: Exception) -> None:
                self._err = err
                self.hits = 0

            async def create(self, **_kwargs: Any) -> Any:
                self.hits += 1
                raise self._err

        class _FailingAsyncOpenAiChat:
            def __init__(self, completions: Any) -> None:
                self.completions = completions

        class _FailingAsyncOpenAiClient:
            def __init__(self, err: Exception) -> None:
                self.chat = _FailingAsyncOpenAiChat(
                    _FailingAsyncCompletions(err)
                )

        class _AsyncAnthropicOkMessages:
            def __init__(self, text: str) -> None:
                self._text = text
                self.hits = 0

            async def create(self, **kwargs: Any) -> Any:
                self.hits += 1
                return SimpleNamespace(
                    id="msg_async_stub",
                    type="message",
                    role="assistant",
                    model=kwargs.get("model", "claude-haiku-4-5"),
                    content=[SimpleNamespace(type="text", text=self._text)],
                    stop_reason="end_turn",
                    usage=SimpleNamespace(input_tokens=21, output_tokens=33),
                )

        class _AsyncAnthropicOkClient:
            def __init__(self, text: str) -> None:
                self.messages = _AsyncAnthropicOkMessages(text)

        primary_async = _FailingAsyncOpenAiClient(_FakeRateLimit())
        anthropic_async = _AsyncAnthropicOkClient("hi from async anthropic")
        wrapped_async = spaturzu.wrap_openai(
            primary_async,
            fallback=[
                {
                    "provider": "anthropic",
                    "client": anthropic_async,
                    "model": "claude-haiku-4-5",
                }
            ],
        )

        async def _async_call() -> Any:
            with spaturzu.run("py-fallback-async"):
                return await wrapped_async.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "say hi"}],
                )

        async_response = asyncio.run(_async_call())
        if primary_async.chat.completions.hits != 1:
            fail("async: primary not attempted")
        if anthropic_async.messages.hits != 1:
            fail("async: fallback not attempted")
        if async_response.choices[0].message.content != "hi from async anthropic":
            fail("async: translated content mismatch")
        if async_response.usage.prompt_tokens != 21:
            fail("async: usage translation mismatch")
        ok("async OpenAI 429 → Anthropic fallback, response shape preserved")

        # ── Scenario C: non-retryable 401 bypasses chain ────────────────
        primary_auth = _FailingOpenAiClient(_FakeAuthError())
        anthropic_unwanted = _AnthropicOkClient("should never get here")
        wrapped_auth = spaturzu.wrap_openai(
            primary_auth,
            fallback=[
                {
                    "provider": "anthropic",
                    "client": anthropic_unwanted,
                    "model": "claude-haiku-4-5",
                }
            ],
        )
        caught: Optional[BaseException] = None
        try:
            with spaturzu.run("py-fallback-auth"):
                wrapped_auth.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "x"}],
                )
        except BaseException as err:  # noqa: BLE001
            caught = err
        if not isinstance(caught, _FakeAuthError):
            fail(f"auth: expected FakeAuthError, got {type(caught).__name__}")
        if anthropic_unwanted.messages.hits != 0:
            fail("auth: fallback was attempted but shouldn't have been")
        ok("non-retryable 401 bypasses fallback chain")

        # Flush so the gateway sees the fallback log entries before this
        # process exits — useful for inspecting the trace afterwards.
        spaturzu.flush(timeout_s=10.0)
    finally:
        spaturzu.shutdown()


if __name__ == "__main__":
    main()
