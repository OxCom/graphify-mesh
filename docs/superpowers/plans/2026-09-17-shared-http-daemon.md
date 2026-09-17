# Shared HTTP Daemon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve every local AI agent from one `graphify-mesh` process reachable by URL, while keeping the existing per-session stdio mode as the default.

**Architecture:** Both transports run through the `mcp` SDK against the unchanged `GraphifyMeshServer.tool_schemas` / `.call_tool` pair, so the tool surface cannot drift between them. The HTTP mode is a stateless streamable-HTTP ASGI app behind a mandatory bearer token on loopback. `GenerationStore` gains a read/write lock so reads run in parallel while a generation reload is serialized.

**Tech Stack:** Python 3.11+, `mcp` SDK (low-level `Server`), starlette, uvicorn, pytest, Docker for all test and lint runs.

**Spec:** `docs/superpowers/specs/2026-09-17-shared-http-daemon-design.md`

## Global Constraints

- Every test and lint run happens inside the Docker image, never on the host: `docker build -f Dockerfile.test -t graphify-mesh-test .` then `docker run --rm -v "$PWD":/app graphify-mesh-test <cmd>`.
- Lint and type gates that must stay green: `ruff check .`, `ruff format --check .`, `mypy src/`.
- **Git is read-only for the implementing agents.** Every "commit" step prints the command for the operator to run; no agent runs `git add`, `git commit`, or any other mutating git command.
- Default transport is `stdio`. A bare `graphify-mesh-server` must behave exactly as it does today.
- Default HTTP port is `19744`. Default host is `127.0.0.1`. Default path is `/mcp`.
- HTTP mode without a token must refuse to start. Empty or whitespace-only token counts as absent.
- Structural invariants of the package are untouched: no cross-repo edges in the structural graph, every generation rebuilt from empty, `graphify global add` never called.
- Tool argument and result shapes stay as they are today. `cwd` stays an optional per-call argument validated only against registered roots.
- Review gates: after each task a `caveman:cavecrew-reviewer` agent on Haiku reviews the diff. Before the host switch (Task 11), Codex reviews the full change set through an `Agent` wrapping `mcp__agents-bridge__ask_codex`.
- Serena covers `.py` files in this project; agents editing Python use Serena's editing tools, and the main thread runs `get_diagnostics_for_file` on every touched Python file before a task is called done.

---

### Task 1: Pin the SDK dependency and probe its real surface

The whole plan rests on what the installed `mcp` version actually offers. Nothing else starts until that is recorded as a test.

**Files:**
- Modify: `pyproject.toml` (dependencies list, test extra)
- Modify: `Dockerfile.test` (no content change expected; rebuild required)
- Test: `tests/server/test_sdk_surface.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: the verified import paths and call signatures every later task uses —
  `mcp.server.lowlevel.Server`, `mcp.server.stdio.stdio_server`,
  `mcp.server.streamable_http_manager.StreamableHTTPSessionManager`,
  `mcp.server.transport_security.TransportSecuritySettings`, `mcp.types`.
  Also produces the recorded answer to: does the installed version let a
  `call_tool` handler return `types.CallToolResult` directly (so `isError` can be
  set explicitly), or must an error be raised as an exception? Task 3 branches on
  that recorded answer.

- [ ] **Step 1: Add the dependencies**

In `pyproject.toml`, under `[project] dependencies`, append with a comment in the style of the existing entries:

```toml
    # mcp: the official MCP SDK. Both transports (stdio and streamable HTTP)
    #   run through its low-level Server, so the 5 tools have exactly one
    #   registration path and cannot drift between modes. Replaced this
    #   package's own newline-delimited JSON-RPC transport (server/protocol.py)
    #   — see docs/superpowers/specs/2026-09-17-shared-http-daemon-design.md.
    "mcp>=1.12,<2",
    # starlette + uvicorn: ASGI app and server for the shared HTTP daemon
    #   (graphify_mesh.server.http_app). Required, not optional: the daemon is
    #   the mode that makes one instance serve every agent.
    "starlette>=0.37,<1",
    "uvicorn>=0.30,<1",
```

In the `test` extra, add the in-process ASGI client used by the HTTP tests:

```toml
test = ["pytest>=9.0.3,<10", "pytest-cov>=5,<8", "httpx>=0.27,<1"]
```

- [ ] **Step 2: Rebuild the test image**

```bash
docker build -f Dockerfile.test -t graphify-mesh-test .
```

Expected: build succeeds and the new packages install. If the corporate index blocks one of them, stop and report — do not vendor or download anything by hand.

- [ ] **Step 3: Write the surface-probe test**

```python
# tests/server/test_sdk_surface.py
"""Pins the mcp SDK surface this package builds on.

Both transports are wired against the low-level Server, so an SDK upgrade
that moves or renames any of these is a build break we want to see as a
failing test here rather than as a mystery at runtime.
"""

from __future__ import annotations

import inspect


def test_lowlevel_server_and_transports_import():
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.server.transport_security import TransportSecuritySettings

    assert callable(Server)
    assert callable(stdio_server)
    assert callable(StreamableHTTPSessionManager)
    assert callable(TransportSecuritySettings)


def test_session_manager_accepts_stateless_and_json_response():
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    params = inspect.signature(StreamableHTTPSessionManager.__init__).parameters
    assert "stateless" in params
    assert "json_response" in params
    assert "security_settings" in params


def test_types_expose_tool_textcontent_and_calltoolresult():
    import mcp.types as types

    assert hasattr(types, "Tool")
    assert hasattr(types, "TextContent")
    # Recorded fact for Task 3: when CallToolResult exists, the call_tool
    # handler can set isError explicitly instead of signalling via exception.
    assert hasattr(types, "CallToolResult")
```

- [ ] **Step 4: Run the probe**

```bash
docker run --rm -v "$PWD":/app graphify-mesh-test python -m pytest tests/server/test_sdk_surface.py -v
```

Expected: PASS. If `test_types_expose_tool_textcontent_and_calltoolresult` fails on `CallToolResult`, record that in the task report — Task 3 then uses the raise-an-exception path instead.

- [ ] **Step 5: Record the installed versions in the report**

```bash
docker run --rm -v "$PWD":/app graphify-mesh-test python -c "import mcp, starlette, uvicorn; print(mcp.__version__ if hasattr(mcp,'__version__') else 'mcp ?'); print(starlette.__version__); print(uvicorn.__version__)"
```

Report the three versions and whether `CallToolResult` is available. Later tasks read this.

- [ ] **Step 6: Commit (operator runs this)**

```bash
git add pyproject.toml tests/server/test_sdk_surface.py && git commit -m "build: depend on the mcp SDK, starlette and uvicorn"
```

---

### Task 2: Transport configuration in `ServerConfig`

**Files:**
- Modify: `src/graphify_mesh/server/config.py`
- Test: `tests/server/test_config_transport.py` (create)

**Interfaces:**
- Consumes: `ServerConfig` as it exists (`mesh_root`, `registry_path`, `from_env`).
- Produces:
  - `ServerConfig.transport: str` — `"stdio"` or `"http"`
  - `ServerConfig.http_host: str`, `.http_port: int`, `.http_path: str`
  - `ServerConfig.http_token: str | None`
  - `ServerConfig.allow_public_bind: bool`
  - `ServerConfig.from_env(..., transport=None, http_host=None, http_port=None, http_path=None, allow_public_bind=None)` — explicit arguments win over environment, environment wins over the defaults.
  - `class ConfigError(ValueError)` in the same module, raised for an invalid transport, port or path, and for HTTP mode with no token.
  - `DEFAULT_HTTP_PORT = 19744`, `DEFAULT_HTTP_HOST = "127.0.0.1"`, `DEFAULT_HTTP_PATH = "/mcp"` module constants.

- [ ] **Step 1: Write the failing tests**

```python
# tests/server/test_config_transport.py
from __future__ import annotations

import pytest

from graphify_mesh.server.config import (
    DEFAULT_HTTP_HOST,
    DEFAULT_HTTP_PATH,
    DEFAULT_HTTP_PORT,
    ConfigError,
    ServerConfig,
)


def test_defaults_are_stdio(monkeypatch, tmp_path):
    for var in (
        "GRAPHIFY_MESH_TRANSPORT",
        "GRAPHIFY_MESH_HTTP_HOST",
        "GRAPHIFY_MESH_HTTP_PORT",
        "GRAPHIFY_MESH_HTTP_PATH",
        "GRAPHIFY_MESH_HTTP_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    config = ServerConfig.from_env(mesh_root=tmp_path)
    assert config.transport == "stdio"
    assert config.http_host == DEFAULT_HTTP_HOST
    assert config.http_port == DEFAULT_HTTP_PORT
    assert config.http_path == DEFAULT_HTTP_PATH
    assert config.http_token is None


def test_env_selects_http_and_port(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_TRANSPORT", "http")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_PORT", "20001")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "s3cret")
    config = ServerConfig.from_env(mesh_root=tmp_path)
    assert config.transport == "http"
    assert config.http_port == 20001
    assert config.http_token == "s3cret"


def test_explicit_argument_beats_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_TRANSPORT", "http")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_PORT", "20001")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "s3cret")
    config = ServerConfig.from_env(mesh_root=tmp_path, http_port=20002)
    assert config.http_port == 20002


def test_http_without_token_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_TRANSPORT", "http")
    monkeypatch.delenv("GRAPHIFY_MESH_HTTP_TOKEN", raising=False)
    with pytest.raises(ConfigError, match="GRAPHIFY_MESH_HTTP_TOKEN"):
        ServerConfig.from_env(mesh_root=tmp_path)


def test_blank_token_counts_as_absent(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_TRANSPORT", "http")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "   ")
    with pytest.raises(ConfigError, match="GRAPHIFY_MESH_HTTP_TOKEN"):
        ServerConfig.from_env(mesh_root=tmp_path)


def test_stdio_ignores_a_missing_token(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_TRANSPORT", "stdio")
    monkeypatch.delenv("GRAPHIFY_MESH_HTTP_TOKEN", raising=False)
    assert ServerConfig.from_env(mesh_root=tmp_path).transport == "stdio"


@pytest.mark.parametrize("value", ["sse", "HTTP2", ""])
def test_unknown_transport_is_refused(monkeypatch, tmp_path, value):
    monkeypatch.setenv("GRAPHIFY_MESH_TRANSPORT", value)
    with pytest.raises(ConfigError):
        ServerConfig.from_env(mesh_root=tmp_path)


@pytest.mark.parametrize("value", ["0", "-1", "65536", "abc", "80.5"])
def test_invalid_port_is_refused(monkeypatch, tmp_path, value):
    monkeypatch.setenv("GRAPHIFY_MESH_TRANSPORT", "http")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "s3cret")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_PORT", value)
    with pytest.raises(ConfigError):
        ServerConfig.from_env(mesh_root=tmp_path)


def test_path_must_start_with_slash(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_TRANSPORT", "http")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "s3cret")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_PATH", "mcp")
    with pytest.raises(ConfigError, match="path"):
        ServerConfig.from_env(mesh_root=tmp_path)


def test_public_bind_requires_opt_in(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_TRANSPORT", "http")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "s3cret")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_HOST", "0.0.0.0")
    with pytest.raises(ConfigError, match="allow-public-bind"):
        ServerConfig.from_env(mesh_root=tmp_path)
    config = ServerConfig.from_env(mesh_root=tmp_path, allow_public_bind=True)
    assert config.http_host == "0.0.0.0"
```

- [ ] **Step 2: Run them to see them fail**

```bash
docker run --rm -v "$PWD":/app graphify-mesh-test python -m pytest tests/server/test_config_transport.py -v
```

Expected: FAIL on `ImportError: cannot import name 'DEFAULT_HTTP_HOST'`.

- [ ] **Step 3: Implement the fields**

Add to `src/graphify_mesh/server/config.py`:

```python
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 19744
DEFAULT_HTTP_PATH = "/mcp"

_PUBLIC_BIND_HOSTS = frozenset({"0.0.0.0", "::", ""})


class ConfigError(ValueError):
    """Invalid transport configuration. Raised at startup, never per request."""
```

Extend the dataclass with the five fields (defaults matching the constants, `http_token: str | None = None`, `allow_public_bind: bool = False`) and resolve them in `from_env` with the flag-over-environment-over-default order. The token check runs only when the resolved transport is `http`, and reads:

```python
if resolved_transport == "http":
    token = (os.environ.get("GRAPHIFY_MESH_HTTP_TOKEN") or "").strip()
    if not token:
        raise ConfigError(
            "HTTP transport requires a bearer token: set GRAPHIFY_MESH_HTTP_TOKEN "
            "(a loopback TCP port is reachable by every local process, unlike the "
            "unix socket it replaces)"
        )
```

The public-bind check raises `ConfigError` naming `--allow-public-bind` when the host is in `_PUBLIC_BIND_HOSTS` and the opt-in is false.

- [ ] **Step 4: Run the tests to green**

```bash
docker run --rm -v "$PWD":/app graphify-mesh-test python -m pytest tests/server/test_config_transport.py -v
docker run --rm -v "$PWD":/app graphify-mesh-test python -m pytest tests/ -q
docker run --rm -v "$PWD":/app graphify-mesh-test mypy src/
docker run --rm -v "$PWD":/app graphify-mesh-test ruff check .
```

Expected: all PASS, no new mypy or ruff findings.

- [ ] **Step 5: Commit (operator runs this)**

```bash
git add src/graphify_mesh/server/config.py tests/server/test_config_transport.py && git commit -m "feat(server): transport, bind and token configuration"
```

---

### Task 3: SDK adapter — one registration path for the 5 tools

**Files:**
- Create: `src/graphify_mesh/server/sdk_app.py`
- Test: `tests/server/test_sdk_app.py` (create)

**Interfaces:**
- Consumes: `GraphifyMeshServer.tool_schemas() -> list[dict]` and `.call_tool(name, arguments) -> dict` with keys `content` (list of `{"type": "text", "text": str}`) and `isError` (bool), both unchanged; `SERVER_NAME`, `SERVER_VERSION` from `server.server`.
- Produces: `build_sdk_server(mesh: GraphifyMeshServer) -> mcp.server.lowlevel.Server` — an SDK server whose `list_tools` returns the same 5 tools with the same input schemas, and whose `call_tool` maps the existing dict result onto SDK content, preserving `isError`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/server/test_sdk_app.py
from __future__ import annotations

import json

import pytest

from graphify_mesh.server.sdk_app import build_sdk_server


@pytest.mark.anyio
async def test_list_tools_matches_the_native_schemas(mesh_server):
    sdk = build_sdk_server(mesh_server)
    handler = sdk.request_handlers  # exercised through the helpers below
    assert handler is not None

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
```

Add the two async helpers `_list_tools` / `_call_tool` at the bottom of the file, invoking the handlers the SDK registered (the exact accessor comes from the Task 1 probe; keep the helpers as the only place that knows it). Add a `mesh_server` fixture to `tests/server/conftest.py` that builds a `GraphifyMeshServer` over the existing fixture generation used by `tests/server/test_server.py`, reusing that file's fixture rather than inventing a second fixture tree.

- [ ] **Step 2: Run them to see them fail**

```bash
docker run --rm -v "$PWD":/app graphify-mesh-test python -m pytest tests/server/test_sdk_app.py -v
```

Expected: FAIL with `ModuleNotFoundError: graphify_mesh.server.sdk_app`.

- [ ] **Step 3: Implement the adapter**

```python
# src/graphify_mesh/server/sdk_app.py
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

from typing import Any

import mcp.types as types
from mcp.server.lowlevel import Server

from graphify_mesh.server.server import SERVER_NAME, GraphifyMeshServer


def _to_blocks(result: dict) -> list[types.ContentBlock]:
    return [
        types.TextContent(type="text", text=item.get("text", ""))
        for item in result.get("content", [])
    ]


def build_sdk_server(mesh: GraphifyMeshServer) -> Server:
    sdk: Server = Server(SERVER_NAME)

    @sdk.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=schema["name"],
                description=schema["description"],
                inputSchema=schema["inputSchema"],
            )
            for schema in mesh.tool_schemas()
        ]

    @sdk.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        result = mesh.call_tool(name, arguments or {})
        return types.CallToolResult(content=_to_blocks(result), isError=bool(result.get("isError")))

    return sdk
```

If the Task 1 probe recorded that the installed SDK does not accept a returned `CallToolResult`, use this handler body instead — the SDK then turns the raised exception into an `isError` result carrying the message:

```python
    @sdk.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.ContentBlock]:
        result = mesh.call_tool(name, arguments or {})
        blocks = _to_blocks(result)
        if result.get("isError"):
            raise ToolResultError(blocks[0].text if blocks else "tool error")
        return blocks
```

with `class ToolResultError(Exception)` defined in this module. Whichever branch is used, the test asserting `isError is True` for an over-large `k` must pass, and no traceback text may reach the client.

`mesh.call_tool` is synchronous and holds no event loop state, so calling it directly from the async handler is correct. It performs file reads; if a later measurement shows those blocking the loop under concurrency, moving the call to `anyio.to_thread.run_sync` is the follow-up — not part of this task.

- [ ] **Step 4: Run the tests to green**

```bash
docker run --rm -v "$PWD":/app graphify-mesh-test python -m pytest tests/server/test_sdk_app.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit (operator runs this)**

```bash
git add src/graphify_mesh/server/sdk_app.py tests/server/test_sdk_app.py tests/server/conftest.py && git commit -m "feat(server): register the 5 tools on the MCP SDK server"
```

---

### Task 4: stdio through the SDK, with the four `protocol.py` protections kept

**Files:**
- Create: `src/graphify_mesh/server/stdio_guard.py`
- Modify: `src/graphify_mesh/server/server.py` (transport call in `main`, drop the `protocol` import, keep `handle_message` untouched for now)
- Delete: `src/graphify_mesh/server/protocol.py`
- Modify: `tests/server/test_protocol.py` → rewrite as `tests/server/test_stdio_transport.py`
- Modify: `tests/server/test_stdio_e2e.py`, `tests/server/test_hardening.py`
- Test: `tests/server/test_stdio_transport.py` (create; replaces `test_protocol.py`)

**Interfaces:**
- Consumes: `build_sdk_server` from Task 3.
- Produces:
  - `graphify_mesh.server.stdio_guard.MAX_LINE_BYTES = 10 * 1024 * 1024`
  - `graphify_mesh.server.stdio_guard.capped_stdin(stream=None)` — an async-iterable text stream wrapper that yields complete lines, drains and drops any line over `MAX_LINE_BYTES` with a stderr warning instead of buffering it, and stops at EOF. Passed to `stdio_server(stdin=...)`.
  - `graphify_mesh.server.server.serve_stdio(mesh) -> None` — blocking, returns on stdin EOF.

**Why this task is not just a deletion:** `protocol.py` pins four behaviors. Each one either is provably provided by the SDK — demonstrate it with a test — or keeps a guard here:

1. unparseable JSON answers `-32700` rather than hanging the client;
2. a non-object payload, including a batch array, answers `-32600`;
3. a line over `MAX_LINE_BYTES` is drained, never buffered whole;
4. a handler exception logs the traceback to stderr and returns a generic `-32603` with no exception text, paths or stack frames, and notifications get no response at all.

Item 3 is the one the SDK's `stdio_server` demonstrably does not do — its reader is `async for line in stdin`, unbounded — hence `stdio_guard`.

- [ ] **Step 1: Write the failing transport tests**

```python
# tests/server/test_stdio_transport.py
from __future__ import annotations

import json

from graphify_mesh.server.stdio_guard import MAX_LINE_BYTES, capped_stdin


def test_oversized_line_is_dropped_not_buffered(caplog):
    huge = "x" * (MAX_LINE_BYTES + 10)
    stream = _text_stream(
        f"{huge}\n" + json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}) + "\n"
    )
    lines = list(_drain(capped_stdin(stream)))
    assert len(lines) == 1
    assert json.loads(lines[0])["method"] == "ping"
    assert any("oversized" in r.message for r in caplog.records)


def test_eof_ends_the_stream():
    assert list(_drain(capped_stdin(_text_stream("")))) == []


def test_blank_lines_are_skipped():
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    lines = list(_drain(capped_stdin(_text_stream(f"\n\n{payload}\n"))))
    assert len(lines) == 1
```

`_text_stream` wraps a string in `io.StringIO`; `_drain` runs the async iterator to a list with `anyio.run`. Write both helpers at the bottom of the file.

Then the end-to-end protections, in `tests/server/test_stdio_e2e.py`, driven by launching the server module as a subprocess exactly as that file does today: send a broken line, a batch array, a notification, and a normal `tools/list`, and assert `-32700`, `-32600`, no response for the notification, and a well-formed tool list. Assert on every error path that the response body contains no `Traceback`, no absolute path from the repository, and no exception class name.

- [ ] **Step 2: Run them to see them fail**

```bash
docker run --rm -v "$PWD":/app graphify-mesh-test python -m pytest tests/server/test_stdio_transport.py tests/server/test_stdio_e2e.py -v
```

Expected: FAIL with `ModuleNotFoundError: graphify_mesh.server.stdio_guard`.

- [ ] **Step 3: Implement the guard and the stdio entry point**

`stdio_guard.py` reuses the drain logic that `protocol.py:_drain_line` / `_read_lines` already got right — move it, do not reinvent it — and wraps the result in the async-iterable shape `stdio_server(stdin=...)` expects (`anyio.wrap_file` over a file-like object whose `readline` is size-capped is the least-code route; the exact wrapper type comes from the Task 1 probe).

In `server.py`:

```python
def serve_stdio(mesh: GraphifyMeshServer) -> None:
    """Blocking stdio transport. Returns on stdin EOF — the clean-exit
    contract that keeps closed sessions from leaking resident processes."""
    import anyio
    from mcp.server.stdio import stdio_server

    from graphify_mesh.server.sdk_app import build_sdk_server
    from graphify_mesh.server.stdio_guard import capped_stdin

    sdk = build_sdk_server(mesh)

    async def _run() -> None:
        async with stdio_server(stdin=capped_stdin()) as (read_stream, write_stream):
            await sdk.run(read_stream, write_stream, sdk.create_initialization_options())

    anyio.run(_run)
```

- [ ] **Step 4: Delete the old transport**

Remove `src/graphify_mesh/server/protocol.py` and `tests/server/test_protocol.py`. `handle_message` in `server.py` still references `protocol.result_response` / `protocol.error_response`; Task 9 removes `handle_message` itself, so for this task inline the two tiny helpers into `server.py` as module-level functions with the same names and behavior rather than leaving a dangling import.

- [ ] **Step 5: Run everything**

```bash
docker run --rm -v "$PWD":/app graphify-mesh-test python -m pytest tests/ -q
docker run --rm -v "$PWD":/app graphify-mesh-test mypy src/
docker run --rm -v "$PWD":/app graphify-mesh-test ruff check .
```

Expected: PASS. Any test that imported `graphify_mesh.server.protocol` is updated, not skipped.

- [ ] **Step 6: Commit (operator runs this)**

```bash
git add -A src/graphify_mesh/server tests/server && git commit -m "refactor(server): run stdio through the SDK, keep the line-size guard"
```

---

### Task 5: Read/write lock in `GenerationStore`

**Files:**
- Create: `src/graphify_mesh/server/rwlock.py`
- Modify: `src/graphify_mesh/server/store.py:433-461` (`GenerationStore.__init__`, `ensure_fresh`) and the `generation` property at `:605`
- Test: `tests/server/test_store_concurrency.py` (create)

**Interfaces:**
- Consumes: `GenerationStore` as it is.
- Produces:
  - `graphify_mesh.server.rwlock.ReadWriteLock` with `read()` and `write()` context managers, writer-preferring so a stream of readers cannot starve a reload.
  - Unchanged public behavior of `GenerationStore.ensure_fresh()` and `.generation`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/server/test_store_concurrency.py
from __future__ import annotations

import threading

from graphify_mesh.server.rwlock import ReadWriteLock


def test_readers_run_in_parallel():
    lock = ReadWriteLock()
    both_inside = threading.Barrier(2, timeout=5)

    def reader():
        with lock.read():
            both_inside.wait()  # raises BrokenBarrierError if readers serialize

    threads = [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive()


def test_writer_excludes_readers():
    lock = ReadWriteLock()
    order: list[str] = []
    writer_holding = threading.Event()
    release_writer = threading.Event()

    def writer():
        with lock.write():
            order.append("writer-in")
            writer_holding.set()
            release_writer.wait(timeout=5)
            order.append("writer-out")

    def reader():
        writer_holding.wait(timeout=5)
        with lock.read():
            order.append("reader-in")

    w = threading.Thread(target=writer)
    r = threading.Thread(target=reader)
    w.start()
    r.start()
    writer_holding.wait(timeout=5)
    release_writer.set()
    w.join(timeout=5)
    r.join(timeout=5)
    assert order == ["writer-in", "writer-out", "reader-in"]


def test_reload_happens_once_under_concurrent_readers(store_with_two_generations):
    store, publish_next = store_with_two_generations
    store.ensure_fresh()
    publish_next()

    reload_calls: list[str] = []
    original = store._try_reload

    def counting_reload(target, mtime):
        reload_calls.append(target)
        return original(target, mtime)

    store._try_reload = counting_reload  # type: ignore[method-assign]

    errors: list[BaseException] = []

    def hit():
        try:
            assert store.generation is not None
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=hit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert errors == []
    assert len(reload_calls) == 1


def test_readers_never_see_a_half_built_generation(store_with_two_generations):
    store, publish_next = store_with_two_generations
    store.ensure_fresh()
    publish_next()

    seen: list[int] = []

    def hit():
        gen = store.generation
        # build_indexes() ran before the generation was published to readers
        seen.append(len(gen.node_by_id))

    threads = [threading.Thread(target=hit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert seen and all(count == seen[0] for count in seen)
    assert seen[0] > 0
```

Add a `store_with_two_generations` fixture to `tests/server/conftest.py`: it builds a mesh root with one published generation, returns the `GenerationStore` plus a `publish_next()` callable that writes a second generation directory and flips `current` to it, reusing the generation-building helpers already in `tests/server/test_store.py` rather than duplicating them.

- [ ] **Step 2: Run them to see them fail**

```bash
docker run --rm -v "$PWD":/app graphify-mesh-test python -m pytest tests/server/test_store_concurrency.py -v
```

Expected: FAIL with `ModuleNotFoundError: graphify_mesh.server.rwlock`.

- [ ] **Step 3: Implement the lock**

```python
# src/graphify_mesh/server/rwlock.py
"""Writer-preferring read/write lock.

The shared daemon answers many reads against one loaded generation while a
sync publish occasionally replaces it. Reads must not serialize behind each
other, and the swap must not run while a read is in flight. Writer-preferring
matters because reads are frequent: a reader-preferring lock would let a
steady stream of queries postpone the reload indefinitely, and the daemon
would keep serving a generation the pipeline already replaced.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class ReadWriteLock:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        with self._cond:
            while self._writer or self._waiting_writers:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        with self._cond:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers:
                    self._cond.wait()
                self._writer = True
            finally:
                self._waiting_writers -= 1
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()
```

- [ ] **Step 4: Wire it into the store**

`__init__` gains `self._lock = ReadWriteLock()`. `ensure_fresh` becomes:

```python
    def ensure_fresh(self) -> None:
        target, mtime = self._stat_signature()
        if target is None:
            with self._lock.write():
                if self._generation is None:
                    self.degraded = ["no_generation_published"]
            return
        with self._lock.read():
            if target == self._current_target and mtime == self._manifest_mtime:
                return  # unchanged, nothing to do
        with self._lock.write():
            # Re-check under the write lock: another thread may have reloaded
            # this exact generation while this one waited for the lock.
            if target == self._current_target and mtime == self._manifest_mtime:
                return
            self._try_reload(target, mtime)
```

`_try_reload` is unchanged and now only ever runs under the write lock — say so in its docstring. The `generation` property keeps calling `ensure_fresh()` and then reads `self._generation` under `self._lock.read()`.

- [ ] **Step 5: Run the tests to green**

```bash
docker run --rm -v "$PWD":/app graphify-mesh-test python -m pytest tests/server/test_store_concurrency.py tests/server/test_store.py -v
docker run --rm -v "$PWD":/app graphify-mesh-test python -m pytest tests/ -q
```

Expected: PASS.

- [ ] **Step 6: Commit (operator runs this)**

```bash
git add src/graphify_mesh/server/rwlock.py src/graphify_mesh/server/store.py tests/server/test_store_concurrency.py tests/server/conftest.py && git commit -m "feat(server): parallel reads, serialized generation reload"
```

---

### Task 6: Lock the two read-path caches

**Files:**
- Modify: `src/graphify_mesh/server/scope.py:52-80` (`_registry_cache`, `load_registry_entries`)
- Modify: `src/graphify_mesh/server/similar.py:49-70` (`_per_generation_cache`)
- Test: `tests/server/test_cache_concurrency.py` (create)

**Interfaces:**
- Consumes: `load_registry_entries(registry_path)` and the two memoized index getters, signatures unchanged.
- Produces: the same functions, safe to call from many threads, each cache built once per key.

- [ ] **Step 1: Write the failing tests**

```python
# tests/server/test_cache_concurrency.py
from __future__ import annotations

import threading

from graphify_mesh.server import scope as scope_mod


def test_registry_is_parsed_once_under_concurrent_readers(registry_file, monkeypatch):
    parses: list[str] = []
    original = scope_mod.json.loads

    def counting_loads(text, *args, **kwargs):
        parses.append("x")
        return original(text, *args, **kwargs)

    monkeypatch.setattr(scope_mod.json, "loads", counting_loads)
    scope_mod._registry_cache.clear()

    results: list[int] = []

    def hit():
        results.append(len(scope_mod.load_registry_entries(registry_file)))

    threads = [threading.Thread(target=hit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert results and all(n == results[0] for n in results)
    assert len(parses) == 1


def test_similar_index_is_built_once_under_concurrent_readers(generation_fixture):
    from graphify_mesh.server import similar as similar_mod

    builds: list[str] = []

    def counting_build(generation):
        builds.append("x")
        return {}

    getter = similar_mod._per_generation_cache(counting_build)

    def hit():
        getter(generation_fixture)

    threads = [threading.Thread(target=hit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(builds) == 1
```

`registry_file` writes a small valid `registry.json` into `tmp_path`; `generation_fixture` returns a loaded `Generation`. Both go in `tests/server/conftest.py`, reusing the existing fixtures where `tests/server/test_scope.py` and `test_similar.py` already build these.

- [ ] **Step 2: Run them to see them fail**

```bash
docker run --rm -v "$PWD":/app graphify-mesh-test python -m pytest tests/server/test_cache_concurrency.py -v
```

Expected: FAIL — the parse or build count exceeds 1 on at least one run. If a run passes by luck, raise the thread count to 32 and confirm it fails before implementing; a test that cannot fail is not a test.

- [ ] **Step 3: Implement the locks**

In `scope.py`, add `_registry_cache_lock = threading.Lock()` next to the cache. The fast path — signature hit — stays outside the lock. The parse-and-store path takes the lock, re-checks the signature under it, then parses and stores. Keep the existing rule that a missing or unreadable registry is never cached.

In `similar.py`, `_per_generation_cache` gets a `threading.Lock` per closure:

```python
cache: dict[int, _T] = {}
lock = threading.Lock()


def get(generation: Generation) -> _T:
    cache_key = id(generation)
    hit = cache.get(cache_key)
    if hit is not None:
        return hit
    with lock:
        hit = cache.get(cache_key)
        if hit is not None:
            return hit
        value = build_fn(generation)
        cache[cache_key] = value
        weakref.finalize(generation, cache.pop, cache_key, None)
        return value
```

The `weakref.finalize` eviction and the `id()` keying stay exactly as they are — they are what keeps a recycled `id()` from serving a stale index.

- [ ] **Step 4: Run the tests to green**

```bash
docker run --rm -v "$PWD":/app graphify-mesh-test python -m pytest tests/server/test_cache_concurrency.py tests/server/test_scope.py tests/server/test_similar.py -v
docker run --rm -v "$PWD":/app graphify-mesh-test python -m pytest tests/ -q
```

Expected: PASS.

- [ ] **Step 5: Commit (operator runs this)**

```bash
git add src/graphify_mesh/server/scope.py src/graphify_mesh/server/similar.py tests/server/test_cache_concurrency.py tests/server/conftest.py && git commit -m "fix(server): make the read-path caches thread-safe"
```

---

### Task 7: The HTTP app — stateless, token-gated, loopback

**Files:**
- Create: `src/graphify_mesh/server/http_app.py`
- Test: `tests/server/test_http_app.py` (create)

**Interfaces:**
- Consumes: `build_sdk_server` (Task 3), `ServerConfig` transport fields (Task 2).
- Produces:
  - `build_http_app(mesh: GraphifyMeshServer, config: ServerConfig) -> starlette.applications.Starlette` — the ASGI app, no server started, for in-process tests.
  - `serve_http(mesh: GraphifyMeshServer, config: ServerConfig) -> None` — blocking `uvicorn.run` on the configured host and port.

- [ ] **Step 1: Write the failing tests**

```python
# tests/server/test_http_app.py
from __future__ import annotations

import json

import httpx
import pytest

from graphify_mesh.server.http_app import build_http_app

HEADERS_OK = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


@pytest.fixture
def client(mesh_server, http_config):
    app = build_http_app(mesh_server, http_config)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:19744")


@pytest.mark.anyio
async def test_missing_token_is_401(client, http_config):
    async with client as c:
        response = await c.post(http_config.http_path, headers=HEADERS_OK, json=_initialize())
    assert response.status_code == 401
    assert "token" not in response.text.lower() or http_config.http_token not in response.text


@pytest.mark.anyio
async def test_wrong_token_is_401(client, http_config):
    async with client as c:
        response = await c.post(
            http_config.http_path,
            headers={**HEADERS_OK, "Authorization": "Bearer nope"},
            json=_initialize(),
        )
    assert response.status_code == 401


@pytest.mark.anyio
async def test_correct_token_lists_the_five_tools(client, http_config, mesh_server):
    async with client as c:
        await c.post(http_config.http_path, headers=_auth(http_config), json=_initialize())
        response = await c.post(
            http_config.http_path, headers=_auth(http_config), json=_tools_list()
        )
    assert response.status_code == 200
    names = [t["name"] for t in _result(response)["tools"]]
    assert names == [s["name"] for s in mesh_server.tool_schemas()]


@pytest.mark.anyio
async def test_foreign_host_header_is_rejected(client, http_config):
    async with client as c:
        response = await c.post(
            http_config.http_path,
            headers={**_auth(http_config), "Host": "evil.example.com"},
            json=_initialize(),
        )
    assert response.status_code in (400, 401, 403, 421)


@pytest.mark.anyio
async def test_tool_call_carries_cwd_through(client, http_config, registered_repo_root):
    async with client as c:
        await c.post(http_config.http_path, headers=_auth(http_config), json=_initialize())
        response = await c.post(
            http_config.http_path,
            headers=_auth(http_config),
            json=_tools_call("search", {"q": "anything", "cwd": str(registered_repo_root)}),
        )
    assert response.status_code == 200
    assert _result(response)["isError"] is False


@pytest.mark.anyio
async def test_no_session_id_is_issued_in_stateless_mode(client, http_config):
    async with client as c:
        response = await c.post(
            http_config.http_path, headers=_auth(http_config), json=_initialize()
        )
    assert "mcp-session-id" not in {k.lower() for k in response.headers}


def test_token_is_never_logged(caplog, mesh_server, http_config):
    build_http_app(mesh_server, http_config)
    assert http_config.http_token not in caplog.text
```

Write the small `_initialize()`, `_tools_list()`, `_tools_call(name, args)`, `_auth(config)` and `_result(response)` helpers at the bottom of the file; `_result` parses the JSON body and returns `payload["result"]`, tolerating an SSE-framed body by taking the last `data:` line. Add `http_config` (a `ServerConfig` with `transport="http"`, `http_token="test-token"`) and `registered_repo_root` fixtures to `tests/server/conftest.py`.

- [ ] **Step 2: Run them to see them fail**

```bash
docker run --rm -v "$PWD":/app graphify-mesh-test python -m pytest tests/server/test_http_app.py -v
```

Expected: FAIL with `ModuleNotFoundError: graphify_mesh.server.http_app`.

- [ ] **Step 3: Implement the app**

```python
# src/graphify_mesh/server/http_app.py
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
import logging
from collections.abc import AsyncIterator

from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.routing import Route

from graphify_mesh.server.config import ServerConfig
from graphify_mesh.server.sdk_app import build_sdk_server
from graphify_mesh.server.server import GraphifyMeshServer

log = logging.getLogger("graphify_mesh.server.http_app")
```

The app is assembled as:

- `build_sdk_server(mesh)` for the tool surface.
- `TransportSecuritySettings(allowed_hosts=[...])` listing the bind host, `localhost`, `127.0.0.1`, each with and without the port. A public bind (already gated by `ConfigError` in Task 2) disables the check, matching what the upstream `graphify` server does at `serve.py:2340`.
- `StreamableHTTPSessionManager(app=sdk, json_response=True, stateless=True, security_settings=security)`.
- A token middleware that reads `Authorization: Bearer <token>`, compares with `hmac.compare_digest`, and returns `401 {"error": "unauthorized"}` on absent or wrong. It logs the rejection with the remote address and no header content. Never log the request body.
- A lifespan entering `manager.run()`, since the manager owns a task group that must wrap the whole server lifetime.
- `Route(config.http_path, endpoint=...)` for the manager's ASGI app.

`serve_http` logs one line naming host, port, path and "token required", then calls `uvicorn.run(app, host=config.http_host, port=config.http_port)`.

- [ ] **Step 4: Run the tests to green**

```bash
docker run --rm -v "$PWD":/app graphify-mesh-test python -m pytest tests/server/test_http_app.py -v
docker run --rm -v "$PWD":/app graphify-mesh-test mypy src/
```

Expected: PASS.

- [ ] **Step 5: Commit (operator runs this)**

```bash
git add src/graphify_mesh/server/http_app.py tests/server/test_http_app.py tests/server/conftest.py && git commit -m "feat(server): stateless streamable-HTTP daemon behind a mandatory token"
```

---

### Task 8: CLI wiring and `cwd` wording

**Files:**
- Modify: `src/graphify_mesh/server/server.py` (`main`, `build_server`, the two `cwd` tool descriptions at `:325-420`, and the `_validate_cwd` docstring at `:110`)
- Modify: `src/graphify_mesh/server/server.py` — delete `handle_message` and the inlined `result_response`/`error_response` helpers from Task 4 once nothing references them
- Test: `tests/server/test_cli.py` (create)

**Interfaces:**
- Consumes: `serve_stdio` (Task 4), `serve_http` (Task 7), `ServerConfig.from_env` (Task 2).
- Produces: `main(argv: list[str] | None = None) -> int` accepting `--transport {stdio,http}`, `--host`, `--port`, `--path`, `--allow-public-bind`; returns 0 on a clean stdio exit, 2 on a `ConfigError`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/server/test_cli.py
from __future__ import annotations

import pytest

from graphify_mesh.server.server import main


def test_no_arguments_runs_stdio(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_ROOT", str(tmp_path))
    monkeypatch.delenv("GRAPHIFY_MESH_TRANSPORT", raising=False)
    calls: list[str] = []
    monkeypatch.setattr(
        "graphify_mesh.server.server.serve_stdio", lambda mesh: calls.append("stdio")
    )
    assert main([]) == 0
    assert calls == ["stdio"]


def test_transport_http_starts_the_daemon(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_ROOT", str(tmp_path))
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "s3cret")
    seen: list[tuple[str, int, str]] = []

    def fake_serve_http(mesh, config):
        seen.append((config.http_host, config.http_port, config.http_path))

    monkeypatch.setattr("graphify_mesh.server.server.serve_http", fake_serve_http)
    assert main(["--transport", "http", "--port", "20005"]) == 0
    assert seen == [("127.0.0.1", 20005, "/mcp")]


def test_http_without_token_exits_2_and_says_why(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GRAPHIFY_MESH_ROOT", str(tmp_path))
    monkeypatch.delenv("GRAPHIFY_MESH_HTTP_TOKEN", raising=False)
    assert main(["--transport", "http"]) == 2
    assert "GRAPHIFY_MESH_HTTP_TOKEN" in capsys.readouterr().err


def test_public_bind_without_opt_in_exits_2(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GRAPHIFY_MESH_ROOT", str(tmp_path))
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "s3cret")
    assert main(["--transport", "http", "--host", "0.0.0.0"]) == 2
    assert "allow-public-bind" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["search", "context_pack"])
def test_cwd_description_no_longer_claims_a_proxy_injects_it(mesh_server, name):
    schema = next(s for s in mesh_server.tool_schemas() if s["name"] == name)
    description = schema["inputSchema"]["properties"]["cwd"]["description"]
    assert "proxy" not in description.lower()
    assert "absolute" in description.lower()
```

- [ ] **Step 2: Run them to see them fail**

```bash
docker run --rm -v "$PWD":/app graphify-mesh-test python -m pytest tests/server/test_cli.py -v
```

Expected: FAIL — `main` takes no flags today and `serve_stdio` / `serve_http` are not referenced from it.

- [ ] **Step 3: Implement `main`**

```python
def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="graphify-mesh-server",
        description=(
            "MCP server for the merged global graph. Default transport is stdio "
            "(one process per client session). --transport http runs one shared "
            "daemon for every local agent; it requires GRAPHIFY_MESH_HTTP_TOKEN."
        ),
    )
    parser.add_argument("--transport", choices=("stdio", "http"), default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--path", default=None)
    parser.add_argument("--allow-public-bind", action="store_true", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        config = ServerConfig.from_env(
            transport=args.transport,
            http_host=args.host,
            http_port=args.port,
            http_path=args.path,
            allow_public_bind=args.allow_public_bind,
        )
    except ConfigError as exc:
        print(f"graphify-mesh-server: {exc}", file=sys.stderr)
        return 2

    mesh = GraphifyMeshServer(config)
    if config.transport == "http":
        serve_http(mesh, config)
    else:
        serve_stdio(mesh)
    return 0
```

`build_server()` keeps working for callers that want a default-configured server.

- [ ] **Step 4: Reword the `cwd` argument**

In both `search` and `context_pack` schemas, replace "injected by the session proxy" with text that tells the caller what to do, because with the shared daemon nothing injects it automatically:

```text
Absolute path of the project directory this call is about, used to resolve
scope='current'. Pass it on every call — one session can move between
projects. Omit it only with an explicit scope ('all' or 'repo:<id>'); a
directory outside registry.json is refused.
```

Update `_validate_cwd`'s docstring the same way: the per-call argument is the caller's contract, not a proxy's.

- [ ] **Step 5: Delete the dead dispatcher**

With both transports on the SDK, `handle_message` and the inlined `result_response` / `error_response` helpers have no callers. Confirm with `mcp__serena__find_referencing_symbols`, then remove them and the tests that exercised the raw dispatcher directly, keeping every behavioral assertion by moving it to the stdio end-to-end test from Task 4.

- [ ] **Step 6: Run everything**

```bash
docker run --rm -v "$PWD":/app graphify-mesh-test python -m pytest tests/ -q
docker run --rm -v "$PWD":/app graphify-mesh-test mypy src/
docker run --rm -v "$PWD":/app graphify-mesh-test ruff check .
docker run --rm -v "$PWD":/app graphify-mesh-test ruff format --check .
```

Expected: PASS.

- [ ] **Step 7: Commit (operator runs this)**

```bash
git add src/graphify_mesh/server/server.py tests/server && git commit -m "feat(server): transport flags, per-call cwd contract, drop the hand-rolled dispatcher"
```

---

### Task 9: Transport parity test

**Files:**
- Test: `tests/server/test_transport_parity.py` (create)

**Interfaces:**
- Consumes: `serve_stdio` via a subprocess, `build_http_app` in process, both fixtures from earlier tasks.
- Produces: nothing new — this is the gate that the two modes answer identically.

- [ ] **Step 1: Write the test**

```python
# tests/server/test_transport_parity.py
"""The two transports must answer the same request identically.

One registration path (sdk_app.build_sdk_server) is what makes this true;
this test is what keeps it true when someone adds a tool to one path only.
"""

from __future__ import annotations

import pytest


@pytest.mark.anyio
async def test_tools_list_is_identical(stdio_client, http_client):
    stdio_tools = await stdio_client.tools_list()
    http_tools = await http_client.tools_list()
    assert stdio_tools == http_tools


@pytest.mark.anyio
async def test_project_map_result_is_identical(stdio_client, http_client, registered_repo_id):
    args = {"repo": registered_repo_id}
    assert await stdio_client.call("project_map", args) == await http_client.call(
        "project_map", args
    )


@pytest.mark.anyio
async def test_scope_failure_is_identical(stdio_client, http_client):
    args = {"q": "x", "cwd": "/definitely/not/registered"}
    stdio_result = await stdio_client.call("search", args)
    http_result = await http_client.call("search", args)
    assert stdio_result["isError"] is True
    assert stdio_result == http_result
```

Add `stdio_client` and `http_client` fixtures to `tests/server/conftest.py`, each exposing the same two methods (`tools_list()`, `call(name, arguments)`) over its transport, so the test body stays transport-blind.

- [ ] **Step 2: Run it**

```bash
docker run --rm -v "$PWD":/app graphify-mesh-test python -m pytest tests/server/test_transport_parity.py -v
```

Expected: PASS. A difference here means the two transports diverged — fix the divergence, never the assertion.

- [ ] **Step 3: Commit (operator runs this)**

```bash
git add tests/server/test_transport_parity.py tests/server/conftest.py && git commit -m "test(server): pin stdio and HTTP to identical answers"
```

---

### Task 10: Documentation

**Files:**
- Modify: `CLAUDE.md` (the MCP server section: "stdio JSON-RPC 2.0, stdlib only (no `mcp` SDK), one process per client session")
- Modify: `docs/mcp-server.md` (transports, headers, token, `cwd` contract)
- Modify: `docs/configuration.md` (the five new variables and flags, `ServerConfig` fields)
- Modify: `docs/architecture.md` (the server half: two transports, the read/write lock)
- Modify: `docs/setup.md` (how to run the shared daemon)
- Modify: `README.md` if it repeats the stdio-only claim

**Interfaces:**
- Consumes: the behavior shipped by Tasks 1-9.
- Produces: documentation that no longer states anything untrue. The three claims that must go: "stdlib only (no `mcp` SDK)", "one process per client session" as the only mode, and "injected by the session proxy" for `cwd`.

- [ ] **Step 1: Rewrite the claims that stopped being true**

Not a patch around the edges: each of the three statements above is now false and gets replaced with what the code does. `docs/configuration.md` gains the variable table exactly as it appears in the spec, including the default port 19744 and the "blank counts as absent" token rule.

- [ ] **Step 2: Document the operator side**

`docs/setup.md` gets the shared-daemon walkthrough: the env file with port and token, the systemd unit shape, the client entry shape (`type: "http"`, URL, `Authorization` header), and the rollback (restore the unit's `ExecStart` and the client entry; stdio never stopped working).

- [ ] **Step 3: Check the docs against the code**

```bash
docker run --rm -v "$PWD":/app graphify-mesh-test python -m pytest tests/ -q
grep -rn "stdlib only\|one process per client session\|session proxy" CLAUDE.md docs/ README.md src/
```

Expected: the grep returns nothing outside historical spec and plan files under `docs/superpowers/`.

- [ ] **Step 4: Sync the vault**

Run the `project-docs` skill in Update mode for `project_id: agentscopex.graphify-mesh`. The documented meanings that changed: transport modes, configuration surface, the `cwd` contract, and the concurrency model. "Nothing changed" is not an available answer for this task.

- [ ] **Step 5: Commit (operator runs this)**

```bash
git add CLAUDE.md README.md docs/ && git commit -m "docs: two transports, shared daemon, per-call cwd"
```

---

### Task 11: Host switch (operator-facing, after Codex review)

Runs only after Tasks 1-10 are green and Codex has reviewed the whole change set.

**Files (all outside the repository):**
- Modify: `~/.config/claude-mcp/graphify-mesh.env`
- Modify: `~/.config/systemd/user/claude-mcp-graphify-mesh.service`
- Create: `~/.claude/hooks/graphify-mesh-cwd.py`
- Modify: `~/.claude/settings.json` (PreToolUse entry for `mcp__graphify-mesh__*`)
- Modify: `~/.claude.json` (the `graphify-mesh` server entry)
- Delete: `~/.claude/mcp-proxies/graphify-mesh-filter.mjs`

**Interfaces:**
- Consumes: `graphify-mesh-server --transport http` and the variables from Task 2.
- Produces: a running daemon on `127.0.0.1:19744/mcp` that every Claude Code session reaches with no per-session process.

- [ ] **Step 1: Install the runtime dependencies (operator runs this)**

The daemon runs from the host install, so the host needs the three new packages. Tests stay in Docker; this is the runtime only:

```bash
pip install --user --upgrade 'mcp>=1.12,<2' 'starlette>=0.37,<1' 'uvicorn>=0.30,<1'
```

- [ ] **Step 2: Generate the token and record it**

```bash
umask 077
printf 'GRAPHIFY_MESH_TRANSPORT=http\nGRAPHIFY_MESH_HTTP_PORT=19744\nGRAPHIFY_MESH_HTTP_TOKEN=%s\n' "$(openssl rand -hex 32)" >> ~/.config/claude-mcp/graphify-mesh.env
chmod 600 ~/.config/claude-mcp/graphify-mesh.env
```

Append, never overwrite: that file already carries values the unit reads.

- [ ] **Step 3: Write the `cwd` hook**

`~/.claude/hooks/graphify-mesh-cwd.py` reads the PreToolUse payload on stdin and emits `updatedInput` with `cwd` set to the session's project directory **only when the tool input has no `cwd` key**. A model-supplied value is never replaced — overwriting it would break switching projects mid-session, which is the whole reason `cwd` is per call. On any parse failure it emits nothing and exits 0, so a hook bug can never block a tool call.

- [ ] **Step 4: Register the hook**

Add a `PreToolUse` entry for `mcp__graphify-mesh__*` in `~/.claude/settings.json` next to the two that already match that prefix. Keep the existing ones.

- [ ] **Step 5: Point the unit at the daemon**

In `~/.config/systemd/user/claude-mcp-graphify-mesh.service`, replace the node broker `ExecStart` and its two `MCP_BROKER_*` variables with:

```ini
ExecStart=%h/.local/bin/graphify-mesh-server --transport http
```

Everything else in the unit stays: the `GRAPHIFY_MESH_*` and Ollama variables, `EnvironmentFile`, the sandboxing directives, `Restart=always`, `RestartSec=2`, `StartLimitIntervalSec=0`.

- [ ] **Step 6: Reload and restart**

```bash
systemctl --user daemon-reload
systemctl --user restart claude-mcp-graphify-mesh.service
systemctl --user status claude-mcp-graphify-mesh.service --no-pager
journalctl --user -u claude-mcp-graphify-mesh.service -n 30 --no-pager
```

Expected: active, and one log line naming host, port, path and that a token is required. Open sessions lose graphify-mesh tools for the few seconds of the restart.

- [ ] **Step 7: Verify with a live request**

```bash
TOKEN=$(sed -n 's/^GRAPHIFY_MESH_HTTP_TOKEN=//p' ~/.config/claude-mcp/graphify-mesh.env | tail -1)
curl -sS -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:19744/mcp \
  -H 'Accept: application/json, text/event-stream' -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}'
curl -sS -X POST http://127.0.0.1:19744/mcp \
  -H "Authorization: Bearer $TOKEN" -H 'Accept: application/json, text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' | head -c 400
```

Expected: `401` for the unauthenticated call, and the five tool names for the authenticated one. The token is read from the env file, never pasted into a command line that lands in shell history — `$TOKEN` above keeps it out of the literal command.

- [ ] **Step 8: Switch the client**

Replace the `graphify-mesh` entry in `~/.claude.json` with:

```json
"graphify-mesh": {
  "type": "http",
  "url": "http://127.0.0.1:19744/mcp",
  "headers": { "Authorization": "Bearer <token from the env file>" }
}
```

Then delete `~/.claude/mcp-proxies/graphify-mesh-filter.mjs`. The shared `mcp-proxies/lib/` files stay — swarmvault, serena and agentmemory still use them.

- [ ] **Step 9: Verify from a new session**

Open a new Claude Code window in a registered project, call `mcp__graphify-mesh__search` with no `cwd`, and confirm a scoped result — that proves the hook fills it. Then call it with an explicit `cwd` for a second registered project and confirm the scope follows the argument. Finally check that no per-session process exists:

```bash
pgrep -af "graphify-mesh" | grep -v "transport http" || echo "no per-session backend"
```

- [ ] **Step 10: Record the rollback**

Write the two-line rollback into `docs/setup.md` if it is not already there: restore the unit's `ExecStart`, restore the `.claude.json` entry and `graphify-mesh-filter.mjs` from git history, `daemon-reload`, `restart`.

---

## Review gates

- **Per task:** dispatch `caveman:cavecrew-reviewer` on Haiku with the task's diff and its "Interfaces" block. A finding at `blocker` or `major` is fixed before the next task starts.
- **Before Task 11:** dispatch an `Agent` whose prompt instructs it to call `mcp__agents-bridge__ask_codex` with the full change set (Tasks 1-10) and relay the result verbatim, then `await_agent` until it returns. Every run started is awaited or cancelled. Codex findings are triaged in the main thread; correctness and security findings are fixed before the host switch.
- **Diagnostics:** the main thread runs `mcp__serena__get_diagnostics_for_file` on every touched `.py` file at the end of each task, because the editing agents have no diagnostics tool of their own.

## Self-review notes

Spec coverage check, section by section: mode selection (Task 2, 8), tool wiring (Task 3), port choice (Task 2 default, Task 11 deployment), `cwd` contract (Task 8 wording, Task 11 hook), concurrency (Task 5 store, Task 6 caches), security (Task 2 token rule, Task 7 middleware and Host check), the four `protocol.py` protections (Task 4), files list (Tasks 1-10 individually, Task 11 for host files), testing (each task plus Task 9 parity), migration and rollback (Task 11 steps 5-10), risks (Task 1 probes risk 1, Task 4 addresses the loss risk, Task 11 covers the host install).

Naming is consistent across tasks: `build_sdk_server`, `build_http_app`, `serve_http`, `serve_stdio`, `capped_stdin`, `MAX_LINE_BYTES`, `ReadWriteLock`, `ConfigError`, `DEFAULT_HTTP_PORT`.

Known open item, deliberately left to Task 1 rather than guessed here: whether the installed SDK accepts a returned `types.CallToolResult` in a `call_tool` handler. Task 3 carries both implementations and picks by the recorded fact.
