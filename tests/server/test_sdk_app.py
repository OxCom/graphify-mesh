from __future__ import annotations

import json

import mcp.types as types
import pytest

from graphify_mesh.server.sdk_app import build_sdk_server


@pytest.mark.anyio
async def test_list_tools_matches_the_native_schemas(mesh_server):
    sdk = build_sdk_server(mesh_server)

    tools = await _list_tools(sdk)
    assert [t.name for t in tools] == [s["name"] for s in mesh_server.tool_schemas()]
    for tool, schema in zip(tools, mesh_server.tool_schemas(), strict=True):
        assert tool.inputSchema == schema["inputSchema"]
        assert tool.description == schema["description"]


@pytest.mark.anyio
async def test_call_tool_passes_arguments_through(mesh_server):
    sdk = build_sdk_server(mesh_server)
    blocks, is_error = await _call_tool(sdk, "project_map", {"repo": "known-repo"})
    assert is_error is False
    payload = json.loads(blocks[0].text)
    assert payload["repo"] == "known-repo"


@pytest.mark.anyio
async def test_unknown_tool_is_an_error_result_not_a_crash(mesh_server):
    sdk = build_sdk_server(mesh_server)
    blocks, is_error = await _call_tool(sdk, "no_such_tool", {})
    assert is_error is True
    assert "unknown tool" in blocks[0].text


@pytest.mark.anyio
async def test_tool_error_reaches_the_client_as_is_error(mesh_server):
    sdk = build_sdk_server(mesh_server)
    # k above MAX_K is a ToolError today; it must stay a readable tool error.
    blocks, is_error = await _call_tool(sdk, "search", {"q": "x", "k": 10_000})
    assert is_error is True
    assert blocks[0].text
    assert "Traceback" not in blocks[0].text


@pytest.mark.anyio
async def test_call_tool_exception_stays_generic(mesh_server, monkeypatch):
    """`build_sdk_server`'s `call_tool` wraps `mesh.call_tool(...)` in its own
    try/except: without it, `mcp.server.lowlevel.Server.call_tool()`'s own
    fallback (`except Exception as e: return self._make_error_result(str(e))`)
    would put the raw exception text in the client-visible result."""

    def boom(name: str, arguments: dict) -> dict:
        raise RuntimeError("secret /etc/passwd internal detail")

    monkeypatch.setattr(mesh_server, "call_tool", boom)
    sdk = build_sdk_server(mesh_server)
    blocks, is_error = await _call_tool(sdk, "search", {"q": "x"})
    assert is_error is True
    assert blocks[0].text == "internal error"
    assert "RuntimeError" not in blocks[0].text
    assert "Traceback" not in blocks[0].text
    assert "/etc/passwd" not in blocks[0].text


# The only place that knows how to reach the SDK's registered request
# handlers directly (Task 1 probe): `Server.request_handlers` keys by
# request type, and the `list_tools`/`call_tool` handlers take that
# request object (or `None` for the no-arg `list_tools` style).
async def _list_tools(sdk):
    handler = sdk.request_handlers[types.ListToolsRequest]
    result = await handler(None)
    return result.root.tools


async def _call_tool(sdk, name: str, arguments: dict):
    handler = sdk.request_handlers[types.CallToolRequest]
    request = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    result = await handler(request)
    call_result = result.root
    return call_result.content, call_result.isError
