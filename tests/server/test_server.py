from __future__ import annotations

import json
from pathlib import Path

from conftest import (
    build_generation,
    fake_embed_query_fn,
    make_link,
    make_node,
    registry_repo,
    write_registry,
)

from graphify_mesh.server.config import ServerConfig
from graphify_mesh.server.server import GraphifyMeshServer


def _server(tmp_path: Path, cwd: Path, embed_query_fn=None) -> GraphifyMeshServer:
    config = ServerConfig.from_env(
        mesh_root=tmp_path, registry_path=tmp_path / "bin" / "registry.json"
    )
    server = GraphifyMeshServer(
        config, cwd=cwd, embed_query_fn=embed_query_fn or fake_embed_query_fn()
    )
    return server


# --- JSON-RPC dispatch (initialize, tools/list, notifications, unknown
# method) is now the SDK's `mcp.server.lowlevel.Server`, not this module's
# code, and is exercised over a real subprocess in test_stdio_e2e.py:
# test_stdio_initialize_and_tools_list_over_real_subprocess,
# test_notification_without_id_gets_no_response_over_real_subprocess, and
# test_unknown_method_returns_json_rpc_error_over_real_subprocess.


def test_search_degrades_gracefully_when_no_generation_published(tmp_path):
    write_registry(tmp_path / "bin" / "registry.json", [registry_repo("acme.repo", tmp_path)])
    server = _server(tmp_path, tmp_path)
    result = server.call_tool("search", {"q": "widget"})
    assert result["isError"] is True
    assert "no consistent published generation" in result["content"][0]["text"]


def test_search_scope_fail_closed_at_tool_layer(tmp_path, monkeypatch):
    registry_path = tmp_path / "bin" / "registry.json"
    registered_root = tmp_path / "registered"
    registered_root.mkdir(parents=True)
    write_registry(registry_path, [registry_repo("acme.registered", registered_root)])

    unregistered_cwd = tmp_path / "unregistered"
    unregistered_cwd.mkdir(parents=True)
    server = _server(tmp_path, unregistered_cwd)
    # Bypass the store's real generation load requirement by monkeypatching
    # `_generation` to a trivial synthetic one, so this test isolates the
    # SCOPE fail-closed behavior from generation-load concerns.
    monkeypatch.setattr(
        server, "_generation", lambda: build_generation([make_node("acme.registered", "X", "x.py")])
    )

    result = server.call_tool("search", {"q": "x"})
    assert result["isError"] is True
    assert "cannot resolve implicit scope" in result["content"][0]["text"]


def test_search_tool_end_to_end_with_synthetic_generation(tmp_path, monkeypatch):
    registry_path = tmp_path / "bin" / "registry.json"
    root = tmp_path / "acme"
    root.mkdir(parents=True)
    write_registry(registry_path, [registry_repo("acme.repo", root)])

    node = make_node("acme.repo", "OrderService", "src/order.py", node_id="n1", line=7)
    generation = build_generation([node])
    server = _server(tmp_path, root)
    monkeypatch.setattr(server, "_generation", lambda: generation)

    result = server.call_tool("search", {"q": "OrderService"})
    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    assert payload["scope_mode"] == "repo"
    assert any(h["citation"] == "[acme.repo:src/order.py:7]" for h in payload["hits"])


def test_cross_project_tool_end_to_end(tmp_path, monkeypatch):
    write_registry(tmp_path / "bin" / "registry.json", [registry_repo("acme.repo", tmp_path)])
    node = make_node("acme.repo", "SharedThing", "src/shared.py", node_id="n1", line=3)
    generation = build_generation([node])
    server = _server(tmp_path, tmp_path)
    monkeypatch.setattr(server, "_generation", lambda: generation)

    result = server.call_tool("cross_project", {"q": "SharedThing"})
    payload = json.loads(result["content"][0]["text"])
    assert payload["hits"]


def test_find_similar_tool_end_to_end(tmp_path, monkeypatch):
    # find_similar serves registered, enabled repos only, seed node included,
    # so the registry has to name the repo the synthetic generation carries.
    write_registry(tmp_path / "bin" / "registry.json", [registry_repo("acme.repo", tmp_path)])
    seed = make_node("acme.repo", "Gateway", "src/gw.py", node_id="seed")
    neighbor = make_node("acme.repo", "GatewayHelper", "src/gwh.py", node_id="nb")
    generation = build_generation([seed, neighbor], links=[make_link("seed", "nb")])
    server = _server(tmp_path, tmp_path)
    monkeypatch.setattr(server, "_generation", lambda: generation)
    result = server.call_tool("find_similar", {"node": "Gateway"})
    payload = json.loads(result["content"][0]["text"])
    assert payload["resolved"] is True


def test_project_map_tool_end_to_end(tmp_path, monkeypatch):
    write_registry(tmp_path / "bin" / "registry.json", [registry_repo("acme.repo", tmp_path)])
    node = make_node("acme.repo", "Widget", "src/widget.py", node_id="n1")
    generation = build_generation([node])
    server = _server(tmp_path, tmp_path)
    monkeypatch.setattr(server, "_generation", lambda: generation)
    result = server.call_tool("project_map", {"repo": "acme.repo"})
    payload = json.loads(result["content"][0]["text"])
    assert payload["resolved"] is True
    assert payload["node_count"] == 1


def test_context_pack_tool_end_to_end(tmp_path, monkeypatch):
    write_registry(tmp_path / "bin" / "registry.json", [registry_repo("acme.repo", tmp_path)])
    node = make_node("acme.repo", "OrderGoal", "src/order_goal.py", node_id="n1", line=5)
    generation = build_generation([node])
    server = _server(tmp_path, tmp_path)
    monkeypatch.setattr(server, "_generation", lambda: generation)

    result = server.call_tool("context_pack", {"goal": "order goal", "token_budget": 5000})
    payload = json.loads(result["content"][0]["text"])
    assert payload["cards"]


def test_unknown_tool_name_returns_is_error():
    server = _server(Path("/tmp"), Path("/tmp"))
    result = server.call_tool("not-a-real-tool", {})
    assert result["isError"] is True


def test_search_per_call_cwd_resolves_current_scope(tmp_path, monkeypatch):
    """The shared-daemon case: the process cwd matches no registered root,
    and the session's own cwd arrives as a `cwd` argument."""
    registry_path = tmp_path / "bin" / "registry.json"
    root = tmp_path / "acme"
    root.mkdir(parents=True)
    write_registry(registry_path, [registry_repo("acme.repo", root)])

    node = make_node("acme.repo", "OrderService", "src/order.py", node_id="n1", line=7)
    daemon_cwd = tmp_path / "daemon-home"
    daemon_cwd.mkdir(parents=True)
    server = _server(tmp_path, daemon_cwd)
    monkeypatch.setattr(server, "_generation", lambda: build_generation([node]))

    result = server.call_tool("search", {"q": "OrderService", "cwd": str(root / "src")})
    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    assert payload["scope_mode"] == "repo"
    assert any(h["repo"] == "acme.repo" for h in payload["hits"])


def test_per_call_cwd_outside_registry_still_fails_closed(tmp_path, monkeypatch):
    registry_path = tmp_path / "bin" / "registry.json"
    root = tmp_path / "acme"
    root.mkdir(parents=True)
    write_registry(registry_path, [registry_repo("acme.repo", root)])

    server = _server(tmp_path, root)  # process cwd WOULD resolve; the call's cwd must not
    monkeypatch.setattr(
        server, "_generation", lambda: build_generation([make_node("acme.repo", "X", "x.py")])
    )

    result = server.call_tool("search", {"q": "x", "cwd": str(tmp_path)})
    assert result["isError"] is True
    assert "cannot resolve implicit scope" in result["content"][0]["text"]


def test_relative_per_call_cwd_is_rejected(tmp_path, monkeypatch):
    write_registry(tmp_path / "bin" / "registry.json", [registry_repo("acme.repo", tmp_path)])
    server = _server(tmp_path, tmp_path)
    monkeypatch.setattr(
        server, "_generation", lambda: build_generation([make_node("acme.repo", "X", "x.py")])
    )

    result = server.call_tool("search", {"q": "x", "cwd": "relative/dir"})
    assert result["isError"] is True
    assert "absolute path" in result["content"][0]["text"]


def test_context_pack_honors_per_call_cwd(tmp_path, monkeypatch):
    registry_path = tmp_path / "bin" / "registry.json"
    root = tmp_path / "acme"
    root.mkdir(parents=True)
    write_registry(registry_path, [registry_repo("acme.repo", root)])

    node = make_node("acme.repo", "OrderService", "src/order.py", node_id="n1", line=7)
    daemon_cwd = tmp_path / "daemon-home"
    daemon_cwd.mkdir(parents=True)
    server = _server(tmp_path, daemon_cwd)
    monkeypatch.setattr(server, "_generation", lambda: build_generation([node]))

    result = server.call_tool(
        "context_pack",
        {
            "goal": "order service",
            "token_budget": 5000,
            "cwd": str(root),
        },
    )
    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    assert payload["cards"]
    assert {c["repo"] for c in payload["cards"]} == {"acme.repo"}


def test_per_call_cwd_traversal_out_of_a_registered_root_fails_closed(tmp_path, monkeypatch):
    """`.../acme/../outside` resolves outside every root, so it must NOT
    inherit the registered ancestor it textually starts with."""
    registry_path = tmp_path / "bin" / "registry.json"
    root = tmp_path / "acme"
    root.mkdir(parents=True)
    write_registry(registry_path, [registry_repo("acme.repo", root)])
    server = _server(tmp_path, root)
    monkeypatch.setattr(
        server, "_generation", lambda: build_generation([make_node("acme.repo", "X", "x.py")])
    )

    result = server.call_tool("search", {"q": "x", "cwd": f"{root}/../outside"})
    assert result["isError"] is True
    assert "cannot resolve implicit scope" in result["content"][0]["text"]


def test_per_call_cwd_symlink_out_of_every_root_fails_closed(tmp_path, monkeypatch):
    registry_path = tmp_path / "bin" / "registry.json"
    root = tmp_path / "acme"
    root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir(parents=True)
    write_registry(registry_path, [registry_repo("acme.repo", root)])

    link = root / "escape"
    link.symlink_to(outside, target_is_directory=True)

    server = _server(tmp_path, root)
    monkeypatch.setattr(
        server, "_generation", lambda: build_generation([make_node("acme.repo", "X", "x.py")])
    )

    result = server.call_tool("search", {"q": "x", "cwd": str(link)})
    assert result["isError"] is True
    assert "cannot resolve implicit scope" in result["content"][0]["text"]


def test_non_string_and_empty_per_call_cwd_are_rejected(tmp_path, monkeypatch):
    write_registry(tmp_path / "bin" / "registry.json", [registry_repo("acme.repo", tmp_path)])
    server = _server(tmp_path, tmp_path)
    monkeypatch.setattr(
        server, "_generation", lambda: build_generation([make_node("acme.repo", "X", "x.py")])
    )

    for bad in ("", 7, ["/tmp"], {"path": "/tmp"}):
        result = server.call_tool("search", {"q": "x", "cwd": bad})
        assert result["isError"] is True, bad
        assert "non-empty string" in result["content"][0]["text"], bad


def test_omitted_per_call_cwd_still_uses_the_process_cwd(tmp_path, monkeypatch):
    """The one-process-per-session deployment keeps working unchanged."""
    registry_path = tmp_path / "bin" / "registry.json"
    root = tmp_path / "acme"
    root.mkdir(parents=True)
    write_registry(registry_path, [registry_repo("acme.repo", root)])
    server = _server(tmp_path, root)
    monkeypatch.setattr(
        server, "_generation", lambda: build_generation([make_node("acme.repo", "X", "x.py")])
    )

    result = server.call_tool("search", {"q": "x"})
    assert result["isError"] is False
    assert json.loads(result["content"][0]["text"])["scope_mode"] == "repo"


def test_per_call_cwd_selects_the_most_specific_of_nested_roots(tmp_path, monkeypatch):
    registry_path = tmp_path / "bin" / "registry.json"
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    write_registry(
        registry_path,
        [registry_repo("acme.outer", outer), registry_repo("acme.inner", inner)],
    )
    nodes = [
        make_node("acme.outer", "SharedName", "outer.py", node_id="n1"),
        make_node("acme.inner", "SharedName", "inner.py", node_id="n2"),
    ]
    server = _server(tmp_path, tmp_path)
    monkeypatch.setattr(server, "_generation", lambda: build_generation(nodes))

    result = server.call_tool("search", {"q": "SharedName", "cwd": str(inner)})
    payload = json.loads(result["content"][0]["text"])
    assert {h["repo"] for h in payload["hits"]} == {"acme.inner"}
