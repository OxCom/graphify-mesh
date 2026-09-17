"""Streamable-HTTP transport: one shared daemon for every local agent.

Stateless by design — `cwd` travels in each tool call, so there is no
per-session state worth keeping, and a client that disappears leaves nothing
behind. The bearer token is mandatory: unlike the unix socket this replaces,
a loopback TCP port is reachable by every local user and process, and what it
serves is the merged graph of every repository in registry.json.
"""

from __future__ import annotations

import contextlib
import hmac
import json
import logging
from collections import deque
from collections.abc import AsyncIterator, Callable

import anyio
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from graphify_mesh.server.config import ServerConfig
from graphify_mesh.server.frames import (
    MAX_MESSAGE_BYTES,
    PARSE_ERROR,
    envelope_error,
    error_frame_text,
    frame_id,
)
from graphify_mesh.server.sdk_app import build_sdk_server
from graphify_mesh.server.server import GraphifyMeshServer

log = logging.getLogger("graphify_mesh.server.http_app")

# A bound exists because a request being read holds up to `MAX_MESSAGE_BYTES`
# of body in this one shared process, so unbounded concurrent reads make
# inbound memory unbounded too. It bounds body buffering specifically, and it
# QUEUES rather than rejecting: making a local agent's request wait a few
# milliseconds is better than failing it. Not uvicorn's `limit_concurrency`,
# which answers 503 once the number of open CONNECTIONS reaches the limit —
# on a daemon serving many long-lived keep-alive clients that rejects the
# 33rd client with nothing in flight at all.
BODY_BUFFER_SLOTS = 8


def _security_settings(config: ServerConfig) -> TransportSecuritySettings:
    """DNS-rebinding protection. A public bind (already gated by
    `ConfigError` in `ServerConfig.from_env`) accepts any Host, matching the
    upstream `graphify` server (`serve.py:_build_http_app`); a loopback or
    specific bind restricts Host to that address plus the localhost
    aliases, each with and without the port."""
    if config.http_host in ("0.0.0.0", "::", ""):  # noqa: S104
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    allowed = {config.http_host, "localhost", "127.0.0.1"}
    allowed |= {f"{host}:{config.http_port}" for host in list(allowed)}
    return TransportSecuritySettings(allowed_hosts=sorted(allowed))


class _MCPASGIApp:
    """Raw-ASGI wrapper around the streamable-HTTP session manager, so
    Starlette treats this as the ASGI app for the mount path directly with
    no request/response wrapping that would break the streaming path."""

    def __init__(self, manager: StreamableHTTPSessionManager) -> None:
        self._manager = manager

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self._manager.handle_request(scope, receive, send)


class _TokenAuthMiddleware:
    """Mandatory bearer-token gate, ahead of the session manager.

    Raw ASGI, not Starlette's `BaseHTTPMiddleware`: that middleware buffers
    the response and breaks the Streamable HTTP stream. Never logs the
    token, the Authorization header, or the request body — a rejection logs
    only the remote address and the fact of rejection.
    """

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self._expected = token.encode("utf-8")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        provided: bytes | None = None
        scheme, _, candidate = headers.get(b"authorization", b"").partition(b" ")
        if scheme.lower() == b"bearer" and candidate:
            provided = candidate.strip()

        if provided is None or not hmac.compare_digest(provided, self._expected):
            client = scope.get("client")
            remote = client[0] if client else "unknown"
            log.warning("rejected unauthorized request from %s", remote)
            body = b'{"error": "unauthorized"}'
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode("ascii")),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        await self.app(scope, receive, send)


async def _send_json(send: Send, status: int, body: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class _FrameGuardMiddleware:
    """Size cap and envelope check for POST bodies, between the token gate
    and the session manager.

    Two things the SDK does differently from `stdio_guard`, both visible to a
    client that got past the token gate:

    * its body limit is its own default rather than this package's number,
      so the two transports stated different caps;
    * its parse and envelope errors carry `str(JSONDecodeError)` or
      `str(ValidationError)` — validation details and excerpts of the input —
      straight to the client, where stdio answers generically.

    So the body is read here, under `MAX_MESSAGE_BYTES` counted on bytes
    actually received (a chunked client has no `Content-Length` to check),
    checked with the same `frames.envelope_error` stdio uses, and replayed
    downstream untouched when it is well-formed. The detail goes to the log;
    the client gets a fixed message.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        # One semaphore per app, so a test app and the daemon never share one.
        self._slots = anyio.Semaphore(BODY_BUFFER_SLOTS)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > MAX_MESSAGE_BYTES:
            # Nothing is buffered on this path, so it takes no slot.
            await self._too_large(scope, send)
            return

        await self._slots.acquire()
        released = False

        def release_slot() -> None:
            # The slot covers the span where THIS middleware holds body
            # bytes, and nothing more: it is handed back as soon as the body
            # reaches the app below, never held across the downstream call,
            # which would cap real request concurrency at `BODY_BUFFER_SLOTS`
            # instead of bounding memory.
            nonlocal released
            if not released:
                released = True
                self._slots.release()

        try:
            await self._guarded(scope, receive, send, release_slot)
        finally:
            release_slot()

    async def _guarded(
        self, scope: Scope, receive: Receive, send: Send, release_slot: Callable[[], None]
    ) -> None:
        body = bytearray()
        body_complete = False
        trailing: Message | None = None
        while True:
            message = await receive()
            if message["type"] != "http.request":
                trailing = dict(message)
                break
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > MAX_MESSAGE_BYTES:
                # Answer now; the rest of the body is never read, so an
                # over-limit client cannot make this process buffer it.
                await self._too_large(scope, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                body_complete = True
                break

        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.warning("rejected unparseable request body from %s: %s", _remote(scope), exc)
            await _send_json(send, 400, error_frame_text(*PARSE_ERROR).encode("utf-8"))
            return

        coded = envelope_error(parsed)
        if coded is not None:
            log.warning(
                "rejected malformed JSON-RPC frame from %s (code %d)", _remote(scope), coded[0]
            )
            frame = error_frame_text(*coded, frame_id(parsed))
            await _send_json(send, 400, frame.encode("utf-8"))
            return

        # Both references are dropped before dispatching downstream: the
        # parsed object would otherwise stay live for the whole call (several
        # times the byte size of the body, as Python objects), and `body` is
        # handed to `_replayed_receive` as-is rather than copied, so exactly
        # one buffer of these bytes exists here at any moment.
        del parsed
        replay = _replayed_receive(body, body_complete, trailing, receive, release_slot)
        del body
        await self.app(scope, replay, send)

    async def _too_large(self, scope: Scope, send: Send) -> None:
        log.warning(
            "rejected oversized request body from %s (> %d bytes)",
            _remote(scope),
            MAX_MESSAGE_BYTES,
        )
        await _send_json(send, 413, b'{"error": "request body too large"}')


def _remote(scope: Scope) -> str:
    client = scope.get("client")
    return client[0] if client else "unknown"


def _replayed_receive(
    body: bytearray,
    body_complete: bool,
    trailing: Message | None,
    receive: Receive,
    on_handoff: Callable[[], None],
) -> Receive:
    """Hands the already-read body to the app below exactly once, then falls
    back to the real `receive` (for the disconnect message).

    `on_handoff` runs when the body message has been delivered: at that point
    the app below owns these bytes and this layer holds none, which is what
    the body-buffer slot was reserving. The buffer is passed through rather
    than copied to `bytes`; the only consumer is the SDK's
    `RequestBodyLimitMiddleware`, which extends its own bytearray from it and
    never requires an immutable `bytes`.
    """
    queued: deque[Message] = deque(
        [{"type": "http.request", "body": body, "more_body": not body_complete}]
    )
    if trailing is not None:
        queued.append(trailing)

    async def replay() -> Message:
        if queued:
            message = queued.popleft()
            if message["type"] == "http.request":
                on_handoff()
            return message
        return await receive()

    return replay


def build_http_app(mesh: GraphifyMeshServer, config: ServerConfig) -> Starlette:
    """The ASGI app for the shared streamable-HTTP daemon. No server
    started — for in-process tests and for `serve_http` to run under
    uvicorn."""
    if not config.http_token:
        raise ValueError("http transport requires config.http_token")

    sdk = build_sdk_server(mesh)
    security = _security_settings(config)

    # Stateless: no sessions, no SSE, no session reaper. `cwd` travels in
    # each tool call instead.
    manager = StreamableHTTPSessionManager(
        app=sdk,
        json_response=True,
        stateless=True,
        security_settings=security,
        # The same number stdio caps a line at. Left to its default, the
        # HTTP bound would be the SDK's, which is both a different number
        # and one that can change between SDK releases. `session_idle_timeout`
        # is deliberately NOT passed: the SDK documents it as unused in
        # stateless mode, and a parameter this adapter does not need is one
        # more thing an SDK release can move.
        max_request_body_size=MAX_MESSAGE_BYTES,
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        # The session manager owns an anyio task group that must wrap the
        # whole server lifetime, so enter it here rather than per-request.
        async with manager.run():
            yield

    return Starlette(
        routes=[Route(config.http_path, endpoint=_MCPASGIApp(manager))],
        # Order matters: the token gate stays outermost, so an
        # unauthenticated request is answered before this process reads,
        # buffers or parses a single byte of its body.
        middleware=[
            Middleware(_TokenAuthMiddleware, token=config.http_token),
            Middleware(_FrameGuardMiddleware),
        ],
        lifespan=lifespan,
    )


def serve_http(mesh: GraphifyMeshServer, config: ServerConfig) -> None:
    """Blocking streamable-HTTP transport on the configured host and port."""
    import uvicorn

    app = build_http_app(mesh, config)
    log.info(
        "graphify-mesh http daemon: host=%s port=%s path=%s token required",
        config.http_host,
        config.http_port,
        config.http_path,
    )
    uvicorn.run(app, host=config.http_host, port=config.http_port)
