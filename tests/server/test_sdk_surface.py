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


def test_session_manager_accepts_the_request_body_cap():
    """`build_http_app` passes `max_request_body_size`, so an SDK without it
    raises `TypeError` before the port opens."""
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    params = inspect.signature(StreamableHTTPSessionManager.__init__).parameters
    assert "max_request_body_size" in params


def test_session_idle_timeout_is_not_relied_on():
    """It is documented as unused in stateless mode, and this package no
    longer passes it — the adapter must not gain a dependency on a parameter
    it does not need."""
    import inspect as _inspect

    from graphify_mesh.server import http_app

    source = _inspect.getsource(http_app)
    assert "session_idle_timeout=" not in source


def test_declared_mcp_floor_is_a_verified_version():
    """The floor must not admit an SDK nobody ran the suite against. Every
    parameter this package passes is verified against the installed version
    above, so the declared floor has to be at least that generation."""
    import importlib.metadata as md
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject.open("rb") as handle:
        deps = tomllib.load(handle)["project"]["dependencies"]

    mcp_specs = [d for d in deps if d.split(">")[0].split("=")[0].strip() == "mcp"]
    assert len(mcp_specs) == 1
    floor_text = mcp_specs[0].split(">=")[1].split(",")[0].strip()
    floor = tuple(int(part) for part in floor_text.split("."))
    installed = tuple(int(part) for part in md.version("mcp").split(".")[: len(floor)])

    assert floor >= (1, 30), f"mcp floor {floor_text} predates the verified surface"
    assert installed >= floor, f"installed mcp {md.version('mcp')} is below the declared floor"
