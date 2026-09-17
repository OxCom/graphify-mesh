"""Registers this package's 5 tools on the MCP SDK's low-level Server.

Both transports (stdio and streamable HTTP) go through here, so
`GraphifyMeshServer.tool_schemas` / `.call_tool` stay the single source of
truth for what the tools are and what they answer. The dict-shaped result
those methods already return — including `isError: true` for a scope or
generation failure — is mapped onto SDK content without changing its
meaning: a tool failure must stay a readable tool result, never a
transport-level error.
"""

from __future__ import annotations

import logging
from typing import Any

import anyio
import anyio.to_thread
import mcp.types as types
from mcp.server.lowlevel import Server

from graphify_mesh.server.server import SERVER_NAME, GraphifyMeshServer

log = logging.getLogger("graphify_mesh.server.sdk_app")

# `GraphifyMeshServer.call_tool` is synchronous and does blocking work (disk
# reads of a generation, an embedding HTTP call with its own timeout). Running
# it inline in the async handler would hold the event loop for that whole
# duration, which in the shared HTTP daemon stalls every other client's tool
# call, its initialization, and even the 401 for an unauthenticated request.
# Dispatch it to a thread instead, through a limiter of this package's own so
# a burst of slow calls can neither spawn unbounded threads nor consume
# anyio's shared default thread tokens.
TOOL_WORKER_LIMIT = 8


def _to_blocks(result: dict) -> list[types.ContentBlock]:
    return [
        types.TextContent(type="text", text=item.get("text", ""))
        for item in result.get("content", [])
    ]


def build_sdk_server(mesh: GraphifyMeshServer) -> Server:
    # `anyio.CapacityLimiter()` needs a running async context, so it cannot be
    # built at import time; build it on first use inside the loop, where the
    # single-threaded event loop makes the check-and-set race-free.
    limiter_holder: dict[str, anyio.CapacityLimiter] = {}

    def tool_limiter() -> anyio.CapacityLimiter:
        limiter = limiter_holder.get("limiter")
        if limiter is None:
            limiter = anyio.CapacityLimiter(TOOL_WORKER_LIMIT)
            limiter_holder["limiter"] = limiter
        return limiter

    async def list_tools(
        ctx: Any, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        # Same net as `call_tool` below, same reason: the SDK's own
        # request dispatch catches an uncaught handler exception and
        # returns `ErrorData(message=str(err))` verbatim to the client —
        # there is no sanitizing layer above this one for a request
        # handler. `mesh.tool_schemas()` takes no arguments today, but this
        # keeps the two handlers in this file consistent rather than
        # sanitizing one half of the adapter and not the other.
        try:
            schemas = mesh.tool_schemas()
        except Exception:
            log.exception("mesh.tool_schemas raised unexpectedly")
            # `from None` only drops the traceback chain; `str(err)` of
            # whatever we raise is what the SDK puts in the client-visible
            # message, so the replacement exception must carry no detail.
            raise RuntimeError("internal error") from None
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=schema["name"],
                    description=schema["description"],
                    input_schema=schema["inputSchema"],
                )
                for schema in schemas
            ]
        )

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        # Narrow net: `mesh.call_tool` already catches everything it can
        # anticipate and returns `isError: true` with a safe message, so this
        # branch should be unreachable in practice. It exists because the
        # SDK's own fallback puts the raw exception text in the
        # client-visible result — this catches first so a bug here never
        # leaks a traceback, a path, or an exception class name.
        name = params.name
        arguments = params.arguments or {}
        try:
            result = await anyio.to_thread.run_sync(
                mesh.call_tool, name, arguments, limiter=tool_limiter()
            )
        except Exception:
            log.exception("mesh.call_tool raised unexpectedly for tool %r", name)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="internal error")], is_error=True
            )
        return types.CallToolResult(
            content=_to_blocks(result), is_error=bool(result.get("isError"))
        )

    return Server(SERVER_NAME, on_list_tools=list_tools, on_call_tool=call_tool)
