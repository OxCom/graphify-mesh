"""End-to-end stdio transport test: launches the REAL `graphify_mesh.server.server` module
entrypoint as a subprocess (synthetic fixture mesh root only — never a real
a real project), talks newline-delimited JSON-RPC 2.0 to it exactly like a
real MCP client would, and confirms:
  * `initialize` / `tools/list` / `tools/call` all work over the real pipe
    (not just via `GraphifyMeshServer.call_tool` in-process, which the other
    tests use — the JSON-RPC dispatch itself is the SDK's, not this
    package's code, so it is only pinned here).
  * the process exits cleanly and promptly when stdin is closed (WS6).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"


def _write_registry(mesh_root: Path) -> None:
    registry_path = mesh_root / "bin" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps({"repos": [], "disabled": [], "external_roots": []}), encoding="utf-8"
    )


def _spawn(mesh_root: Path) -> subprocess.Popen:
    env = {
        "GRAPHIFY_MESH_ROOT": str(mesh_root),
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(SRC_DIR),
    }
    return subprocess.Popen(
        [sys.executable, "-m", "graphify_mesh.server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        bufsize=1,
    )


def _spawn_script(mesh_root: Path, script: str) -> subprocess.Popen:
    """Like `_spawn`, but runs an inline `-c` script instead of the real
    `graphify_mesh.server.server` entrypoint. Used only to install a
    deliberately raising handler above the tool layer (see
    `_RAISING_TOOL_SCHEMAS_SCRIPT` / `_RAISING_PROGRESS_NOTIFICATION_SCRIPT`)
    without touching production code — the real entrypoint has no hook for
    this, on purpose."""
    env = {
        "GRAPHIFY_MESH_ROOT": str(mesh_root),
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(SRC_DIR),
    }
    return subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        bufsize=1,
    )


def _send(proc: subprocess.Popen, message: dict) -> dict:
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    assert line, "server produced no response (check stderr for a crash)"
    return json.loads(line)


def test_stdio_initialize_and_tools_list_over_real_subprocess(tmp_path):
    _write_registry(tmp_path)
    proc = _spawn(tmp_path)
    try:
        init_resp = _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"},
                },
            },
        )
        assert init_resp["result"]["serverInfo"]["name"] == "graphify-mesh"
        proc.stdin.write(
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        )
        proc.stdin.flush()

        list_resp = _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = {t["name"] for t in list_resp["result"]["tools"]}
        assert names == {"search", "cross_project", "find_similar", "project_map", "context_pack"}
    finally:
        proc.stdin.close()
        proc.wait(timeout=5)


def test_stdio_tool_call_degrades_gracefully_with_no_published_generation(tmp_path):
    _write_registry(tmp_path)
    proc = _spawn(tmp_path)
    try:
        _initialize(proc)
        resp = _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "search", "arguments": {"q": "anything", "scope": "all"}},
            },
        )
        assert resp["result"]["isError"] is True
    finally:
        proc.stdin.close()
        proc.wait(timeout=5)


def test_process_exits_cleanly_on_stdin_close(tmp_path):
    """WS6: 'companion server must exit cleanly on stdin close' — the
    observed leak was 10 stale `graphify.serve` processes that never did
    this. Verify exit code 0 and a bounded wait, not a hang."""
    _write_registry(tmp_path)
    proc = _spawn(tmp_path)
    proc.stdin.close()
    returncode = proc.wait(timeout=5)
    assert returncode == 0


# --- transport-level protections after the move onto mcp.server.stdio ------
#
# `stdio_guard._CappedLineReader` (not the SDK) intercepts an unparseable or
# non-object/batch line before it ever reaches `mcp.server.stdio`'s own
# reader, and answers it directly with the coded JSON-RPC error the spec
# requires (`-32700`/`-32600`, `id: null`) — see that module's docstring for
# why the SDK itself cannot be relied on for this (it degrades a parse
# failure to an untargeted, uncoded `notifications/message` instead).


def _send_raw(proc: subprocess.Popen, raw_line: str) -> str:
    proc.stdin.write(raw_line + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    assert line, "server produced no response line (check stderr for a crash)"
    return line


def _initialize(proc: subprocess.Popen) -> None:
    _send(
        proc,
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        },
    )
    proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
    proc.stdin.flush()


def _assert_no_leak(text: str) -> None:
    assert "Traceback" not in text
    assert str(SRC_DIR) not in text
    assert "RuntimeError" not in text
    assert "ValueError" not in text


def test_unparseable_json_line_gets_parse_error(tmp_path):
    _write_registry(tmp_path)
    proc = _spawn(tmp_path)
    try:
        response_line = _send_raw(proc, "{this is not json")
        response = json.loads(response_line)
        assert response == {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": "parse error"},
        }
        _assert_no_leak(response_line)

        # transport keeps serving: a well-formed request right after still works
        _initialize(proc)
        list_resp = _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = {t["name"] for t in list_resp["result"]["tools"]}
        assert names == {"search", "cross_project", "find_similar", "project_map", "context_pack"}
    finally:
        proc.stdin.close()
        proc.wait(timeout=5)


def test_batch_array_gets_invalid_request(tmp_path):
    _write_registry(tmp_path)
    proc = _spawn(tmp_path)
    try:
        response_line = _send_raw(proc, json.dumps([{"jsonrpc": "2.0", "id": 1, "method": "ping"}]))
        response = json.loads(response_line)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] is None
        assert response["error"]["code"] == -32600
        _assert_no_leak(response_line)

        _initialize(proc)
        list_resp = _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert list_resp["result"]["tools"]
    finally:
        proc.stdin.close()
        proc.wait(timeout=5)


def test_bare_non_object_payload_gets_invalid_request(tmp_path):
    """A single scalar (not an object, not a batch array) is equally not a
    valid JSON-RPC frame — same `-32600` path as a batch array."""
    _write_registry(tmp_path)
    proc = _spawn(tmp_path)
    try:
        response_line = _send_raw(proc, "42")
        response = json.loads(response_line)
        assert response["id"] is None
        assert response["error"]["code"] == -32600
        _assert_no_leak(response_line)

        _initialize(proc)
        list_resp = _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert list_resp["result"]["tools"]
    finally:
        proc.stdin.close()
        proc.wait(timeout=5)


def test_notification_without_id_gets_no_response_over_real_subprocess(tmp_path):
    _write_registry(tmp_path)
    proc = _spawn(tmp_path)
    try:
        _initialize(proc)
        # An id-less message is a notification per JSON-RPC 2.0: no response
        # may be written for it. Send one, then an id'd request: the next
        # line off the pipe must belong to the id'd request, not the
        # notification.
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/whatever"}) + "\n")
        proc.stdin.flush()
        list_resp = _send(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
        assert list_resp.get("id") == 3
        assert "error" not in list_resp
    finally:
        proc.stdin.close()
        proc.wait(timeout=5)


def test_unknown_method_returns_json_rpc_error_over_real_subprocess(tmp_path):
    """Moved from the deleted `GraphifyMeshServer.handle_message` dispatcher
    (tests/server/test_server.py::test_unknown_method_returns_json_rpc_error):
    method routing is now the SDK's. mcp 1.x's `mcp.server.lowlevel.Server`
    reported an unrecognized method as -32602 (invalid params); mcp 2.x
    reports the spec-correct -32601 (method not found) instead — verified
    against a real subprocess, not assumed."""
    _write_registry(tmp_path)
    proc = _spawn(tmp_path)
    try:
        _initialize(proc)
        response = _send(proc, {"jsonrpc": "2.0", "id": 5, "method": "bogus/method"})
        assert response["error"]["code"] == -32601
    finally:
        proc.stdin.close()
        proc.wait(timeout=5)


def test_tool_call_exception_stays_generic_over_real_subprocess(tmp_path):
    """`GraphifyMeshServer.call_tool`'s own try/except (server.py, unchanged
    by the stdio-transport move) is what keeps a tool-handler exception from
    leaking — not the SDK. Drive it with a k value large enough to trip the
    server's own ToolError path, then check the response body."""
    _write_registry(tmp_path)
    proc = _spawn(tmp_path)
    try:
        _initialize(proc)
        resp = _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "search", "arguments": {"q": "x", "scope": "all", "k": 10_000}},
            },
        )
        result = resp["result"]
        assert result["isError"] is True
        _assert_no_leak(json.dumps(result))
    finally:
        proc.stdin.close()
        proc.wait(timeout=5)


# --- coverage restored from the deleted protocol.serve tests: a handler
# raising ABOVE the tool layer (inside the SDK's own request/notification
# dispatch, not inside GraphifyMeshServer.call_tool) must still answer with
# an error and keep the transport loop serving, and must never respond at
# all to a raising notification. Neither lever exists in production code —
# these scripts build the real `GraphifyMeshServer` / `build_sdk_server` and
# monkeypatch only inside the test's own subprocess.

_TOOL_SCHEMAS_SENTINEL = "sekrit-tool-schemas-boom-9f3c2a"

_RAISING_TOOL_SCHEMAS_SCRIPT = f"""
from graphify_mesh.server.config import ServerConfig
from graphify_mesh.server.server import GraphifyMeshServer, serve_stdio


def _raise():
    raise RuntimeError({_TOOL_SCHEMAS_SENTINEL!r})


mesh = GraphifyMeshServer(ServerConfig.from_env())
mesh.tool_schemas = _raise  # lever: list_tools's own net (sdk_app.py) must catch this
serve_stdio(mesh)
"""

_RAISING_PROGRESS_NOTIFICATION_SCRIPT = """
import anyio
import mcp.types as types
from mcp.server.stdio import stdio_server

from graphify_mesh.server.config import ServerConfig
from graphify_mesh.server.sdk_app import build_sdk_server
from graphify_mesh.server.server import GraphifyMeshServer
from graphify_mesh.server.stdio_guard import capped_stdin

mesh = GraphifyMeshServer(ServerConfig.from_env())
sdk = build_sdk_server(mesh)


# `sdk.progress_notification()` (decorator registration) no longer exists on
# mcp 2.x's `Server` — notification handlers register by method string via
# `add_notification_handler`, and take `(ctx, params)` instead of the old
# unpacked-keyword shape.
async def _raise(ctx, params: types.ProgressNotificationParams) -> None:
    raise RuntimeError("boom: notification handler failed")


sdk.add_notification_handler("notifications/progress", types.ProgressNotificationParams, _raise)


async def _run():
    async with stdio_server(stdin=capped_stdin()) as (read_stream, write_stream):
        await sdk.run(read_stream, write_stream, sdk.create_initialization_options())


anyio.run(_run)
"""


def test_request_handler_exception_above_the_tool_layer_gets_error_and_loop_survives(tmp_path):
    """Lever: monkeypatch `GraphifyMeshServer.tool_schemas` to raise, so the
    exception happens inside the SDK's `list_tools` request handler
    (`mcp.server.lowlevel.Server._handle_request`) — above `call_tool`'s own
    try/except, which `test_tool_call_exception_stays_generic_over_real_subprocess`
    already covers and which is a narrower guarantee than this one. Restores
    the coverage the deleted
    `test_serve_survives_raising_handler_and_answers_requests_with_32603`
    (old hand-rolled `protocol.serve`) gave.

    `sdk_app.build_sdk_server`'s `list_tools` handler now carries the same
    net as `call_tool`: it logs the traceback to stderr and re-raises a
    generic `RuntimeError("internal error")` with `from None`, so the SDK's
    `_handle_request` — which otherwise returns `ErrorData(message=str(err))`
    verbatim for an uncaught handler exception — stringifies the generic
    replacement, not the original. This test raises with a distinctive
    sentinel and asserts that sentinel never reaches the client, pinning the
    net rather than just describing the SDK's raw behavior.
    """
    _write_registry(tmp_path)
    proc = _spawn_script(tmp_path, _RAISING_TOOL_SCHEMAS_SCRIPT)
    try:
        _initialize(proc)
        response_line = _send_raw(
            proc, json.dumps({"jsonrpc": "2.0", "id": 9, "method": "tools/list"})
        )
        response = json.loads(response_line)
        assert response.get("id") == 9
        assert "error" in response
        assert "result" not in response
        assert response["error"]["message"] == "internal error"
        assert _TOOL_SCHEMAS_SENTINEL not in response_line
        _assert_no_leak(response_line)

        # Loop survived: a normal request right after still gets answered.
        ping_resp = _send(proc, {"jsonrpc": "2.0", "id": 10, "method": "ping"})
        assert ping_resp == {"jsonrpc": "2.0", "id": 10, "result": {}}
    finally:
        proc.stdin.close()
        proc.wait(timeout=5)


def test_notification_handler_exception_still_gets_no_response(tmp_path):
    """Lever: a `progress_notification` handler registered on the real
    `build_sdk_server(mesh)` output that raises. `notifications/progress`
    has no `id`, so JSON-RPC 2.0 forbids any response for it regardless of
    what the handler does — `mcp.server.lowlevel.Server._handle_notification`
    catches the exception itself and never calls `respond`. Restores the
    coverage the deleted `test_serve_never_responds_to_raising_notification`
    (old hand-rolled `protocol.serve`) gave."""
    _write_registry(tmp_path)
    proc = _spawn_script(tmp_path, _RAISING_PROGRESS_NOTIFICATION_SCRIPT)
    try:
        _initialize(proc)
        proc.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/progress",
                    "params": {"progressToken": "t", "progress": 1},
                }
            )
            + "\n"
        )
        proc.stdin.flush()

        # No response was written for the notification: the next line off
        # the pipe belongs to this id'd request, not to the notification.
        list_resp = _send(proc, {"jsonrpc": "2.0", "id": 11, "method": "tools/list"})
        assert list_resp.get("id") == 11
        assert "error" not in list_resp
    finally:
        proc.stdin.close()
        proc.wait(timeout=5)
