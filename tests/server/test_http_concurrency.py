"""Concurrency of the shared HTTP daemon: one slow tool call must not stop
the daemon from answering everyone else.

The tools are synchronous (`GraphifyMeshServer.call_tool` does blocking file
and network I/O), so the adapter has to get them off the event loop. These
tests drive the real ASGI app over httpx, with the lifespan entered by hand
(`httpx.ASGITransport` does not run it), and make the interleaving
deterministic with a `threading.Event` instead of a sleep race: the slow call
cannot finish until the fast call has already answered.
"""

from __future__ import annotations

import asyncio
import threading

import httpx
import pytest

from graphify_mesh.server.http_app import build_http_app

HEADERS_OK = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


def _auth(config) -> dict:
    return {**HEADERS_OK, "Authorization": f"Bearer {config.http_token}"}


def _tools_call(request_id: int, name: str, arguments: dict) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


class _GatedMesh:
    """Stands in for `GraphifyMeshServer` at the one method the adapter calls.

    `search` blocks until `released` is set; `project_map` sets it. A single
    thread of execution therefore cannot complete `search` at all: only a
    dispatcher that runs the two calls concurrently makes progress.
    """

    def __init__(self, real) -> None:
        self._real = real
        self.config = real.config
        self.released = threading.Event()
        self.fast_answered_while_slow_in_flight = False
        self._slow_in_flight = threading.Event()

    def tool_schemas(self) -> list[dict]:
        return self._real.tool_schemas()

    def call_tool(self, name: str, arguments: dict) -> dict:
        if name == "search":
            self._slow_in_flight.set()
            if not self.released.wait(timeout=10):
                return {
                    "content": [{"type": "text", "text": "slow: never released"}],
                    "isError": True,
                }
            return {"content": [{"type": "text", "text": "slow: done"}], "isError": False}
        self.fast_answered_while_slow_in_flight = self._slow_in_flight.is_set()
        self.released.set()
        return {"content": [{"type": "text", "text": "fast: done"}], "isError": False}


@pytest.mark.anyio
async def test_a_slow_tool_call_does_not_block_other_clients(mesh_server, http_config):
    gated = _GatedMesh(mesh_server)
    app = build_http_app(gated, http_config)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:19744") as c:
            headers = _auth(http_config)

            async def post(payload: dict) -> httpx.Response:
                return await c.post(http_config.http_path, headers=headers, json=payload)

            slow = asyncio.create_task(post(_tools_call(1, "search", {"q": "x"})))
            await asyncio.sleep(0.05)  # let the slow call reach the mesh
            fast = asyncio.create_task(post(_tools_call(2, "project_map", {"repo": "known-repo"})))

            done, pending = await asyncio.wait({slow, fast}, timeout=8)
            for task in pending:
                task.cancel()
            assert not pending, "the slow tool call blocked the event loop"

    assert gated.fast_answered_while_slow_in_flight is True
    assert "slow: done" in slow.result().text


@pytest.mark.anyio
async def test_tool_dispatch_worker_pool_is_bounded(mesh_server, http_config):
    """The pool that gets tool calls off the loop must be bounded, or a burst
    of slow calls spawns unbounded threads in the shared daemon."""
    from graphify_mesh.server import sdk_app

    assert isinstance(sdk_app.TOOL_WORKER_LIMIT, int)
    assert 0 < sdk_app.TOOL_WORKER_LIMIT <= 64


def test_serve_http_never_bounds_on_open_connections(mesh_server, http_config, monkeypatch):
    """uvicorn's `limit_concurrency` must not be used here. It answers HTTP
    503 (`app = service_unavailable`) once `len(self.connections)` reaches the
    limit, so on a daemon holding many long-lived keep-alive clients it would
    reject a client that has nothing in flight at all."""
    import uvicorn

    from graphify_mesh.server import http_app

    captured: dict = {}

    def fake_run(app, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    http_app.serve_http(mesh_server, http_config)

    assert "limit_concurrency" not in captured
    assert not hasattr(http_app, "MAX_CONCURRENT_REQUESTS")
    assert isinstance(http_app.BODY_BUFFER_SLOTS, int)
    assert 0 < http_app.BODY_BUFFER_SLOTS <= 64


@pytest.mark.anyio
async def test_body_buffer_slot_is_not_held_across_the_downstream_call(
    mesh_server, http_config, monkeypatch
):
    """The risk in bounding body buffering with a semaphore: if the slot is
    held across `await self.app(...)`, it stops bounding memory and starts
    capping real request concurrency.

    Driven with ONE slot and the gated mesh: request A's tool call cannot
    finish until request B's tool call has run, and B cannot even have its
    body read while A holds the only slot. Both answering proves the slot is
    released at the handoff, not at the end of the call.
    """
    monkeypatch.setattr("graphify_mesh.server.http_app.BODY_BUFFER_SLOTS", 1)
    gated = _GatedMesh(mesh_server)
    app = build_http_app(gated, http_config)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:19744") as c:
            headers = _auth(http_config)

            async def post(payload: dict) -> httpx.Response:
                return await c.post(http_config.http_path, headers=headers, json=payload)

            slow = asyncio.create_task(post(_tools_call(1, "search", {"q": "x"})))
            await asyncio.sleep(0.05)
            fast = asyncio.create_task(post(_tools_call(2, "project_map", {"repo": "known-repo"})))

            done, pending = await asyncio.wait({slow, fast}, timeout=8)
            for task in pending:
                task.cancel()
            assert not pending, "the body-buffer slot was held across the downstream call"

    assert "slow: done" in slow.result().text
    assert fast.result().status_code == 200


@pytest.mark.anyio
async def test_over_the_slot_bound_requests_queue_and_none_are_rejected(
    mesh_server, http_config, monkeypatch
):
    """Bounding body buffering must never turn into a rejection: with one
    slot and 12 simultaneous requests, all 12 answer 200 and none gets a 503
    or a 413."""
    monkeypatch.setattr("graphify_mesh.server.http_app.BODY_BUFFER_SLOTS", 1)
    app = build_http_app(mesh_server, http_config)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:19744") as c:
            headers = _auth(http_config)
            payloads = [_tools_call(i, "project_map", {"repo": "known-repo"}) for i in range(12)]
            responses = await asyncio.gather(
                *(c.post(http_config.http_path, headers=headers, json=p) for p in payloads)
            )

    assert [r.status_code for r in responses] == [200] * 12
