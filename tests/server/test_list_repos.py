"""`list_repos`: registered-repo discovery with per-repo generation facts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import (
    build_generation,
    fake_embed_query_fn,
    make_node,
    registry_repo,
    write_registry,
)

from graphify_mesh.server.config import ServerConfig
from graphify_mesh.server.server import GraphifyMeshServer, ToolError
from graphify_mesh.sync.registry import registry_hash

A = "a.repo"
B = "b.repo"
C = "c.repo"


def _mesh(
    tmp_path: Path,
    *,
    disabled: list[str] | None = None,
    manifest_extra: dict | None = None,
    nodes: list[dict] | None = None,
    transport: str = "stdio",
    cwd: Path | None = None,
    with_generation: bool = True,
) -> GraphifyMeshServer:
    roots = {A: tmp_path / "a", B: tmp_path / "b", C: tmp_path / "c"}
    for root in roots.values():
        root.mkdir(exist_ok=True)
    registry_path = tmp_path / "bin" / "registry.json"
    write_registry(
        registry_path,
        [registry_repo(r, root) for r, root in roots.items()],
        disabled=disabled,
    )
    config = ServerConfig(
        mesh_root=tmp_path,
        registry_path=registry_path,
        transport=transport,
        http_token="t" if transport == "http" else None,
    )
    mesh = GraphifyMeshServer(config, cwd=cwd or roots[A], embed_query_fn=fake_embed_query_fn())
    if with_generation:
        generation = build_generation(
            nodes
            if nodes is not None
            else [
                make_node(A, "One", "src/a.py", node_id="a1"),
                make_node(A, "Two", "src/a2.py", node_id="a2"),
            ],
            manifest_extra=manifest_extra
            if manifest_extra is not None
            else {"repo_input_hashes": {A: "h", B: "h", C: "h"}},
        )
        mesh._optional_generation = lambda: generation  # type: ignore[method-assign]
    return mesh


def _call(mesh: GraphifyMeshServer, arguments: dict | None = None) -> dict:
    response = mesh.call_tool("list_repos", arguments or {})
    assert response["isError"] is False
    return json.loads(response["content"][0]["text"])


def _by_id(result: dict) -> dict[str, dict]:
    return {repo["repo_id"]: repo for repo in result["repos"]}


def test_enabled_only_and_sorted(tmp_path):
    result = _mesh(tmp_path, disabled=[B]).tool_list_repos({})
    assert [repo["repo_id"] for repo in result["repos"]] == [A, C]
    assert result["repos"][0]["root"] == str(tmp_path / "a")
    outside = _mesh(tmp_path, disabled=[B]).tool_list_repos({"cwd": str(tmp_path / "b")})
    assert outside["current"] is None


def test_nested_roots_pick_inner_repo(tmp_path):
    mesh = _mesh(tmp_path)
    inner = tmp_path / "a" / "inner"
    inner.mkdir()
    write_registry(
        mesh.config.registry_path,
        [
            registry_repo(A, tmp_path / "a"),
            registry_repo("inner.repo", inner),
        ],
    )
    assert mesh.tool_list_repos({"cwd": str(inner / "src")})["current"] == "inner.repo"


def test_unmatched_cwd_gives_no_current(tmp_path):
    result = _mesh(tmp_path).tool_list_repos({"cwd": "/nonexistent/elsewhere"})
    assert result["current"] is None


@pytest.mark.parametrize("bad", ["relative/dir", 5, ""])
def test_bad_cwd_is_an_error(tmp_path, bad):
    mesh = _mesh(tmp_path)
    with pytest.raises(ToolError, match="cwd"):
        mesh.tool_list_repos({"cwd": bad})
    assert mesh.call_tool("list_repos", {"cwd": bad})["isError"] is True


def test_http_without_cwd_has_no_current(tmp_path):
    result = _mesh(tmp_path, transport="http").tool_list_repos({})
    assert result["current"] is None


def test_stdio_without_cwd_uses_process_cwd(tmp_path):
    mesh = _mesh(tmp_path, cwd=tmp_path / "b" / "deep")
    assert mesh.tool_list_repos({})["current"] == B


def test_no_generation_real_path(tmp_path):
    mesh = _mesh(tmp_path, with_generation=False)
    result = _call(mesh)
    assert [repo["repo_id"] for repo in result["repos"]] == [A, B, C]
    for repo in result["repos"]:
        assert repo["in_generation"] is None
        assert repo["node_count"] is None
        assert repo["stale"] is None
    assert result["generation"] is None
    assert "no_generation_published" in result["degraded"]


def test_zero_node_repo_is_in_generation(tmp_path):
    repos = _by_id(_mesh(tmp_path).tool_list_repos({}))
    assert repos[B]["in_generation"] is True
    assert repos[B]["node_count"] == 0
    assert repos[A]["node_count"] == 2


def test_repo_missing_from_hashes_is_not_in_generation(tmp_path):
    mesh = _mesh(tmp_path, manifest_extra={"repo_input_hashes": {A: "h"}})
    repos = _by_id(mesh.tool_list_repos({}))
    assert repos[A]["in_generation"] is True
    assert repos[B]["in_generation"] is False


def test_stale_repos_flagged(tmp_path):
    mesh = _mesh(
        tmp_path,
        manifest_extra={"repo_input_hashes": {A: "h", B: "h", C: "h"}, "stale_repos": [B]},
    )
    repos = _by_id(mesh.tool_list_repos({}))
    assert repos[B]["stale"] is True
    assert repos[A]["stale"] is False


def test_generation_summary(tmp_path):
    mesh = _mesh(
        tmp_path,
        manifest_extra={"repo_input_hashes": {}, "created_at": "2026-10-20T00:00:00Z"},
    )
    assert mesh.tool_list_repos({})["generation"] == {
        "id": "gen-test-1",
        "created_at": "2026-10-20T00:00:00Z",
    }


def test_registry_drift_marker(tmp_path):
    drifted = _mesh(tmp_path, manifest_extra={"registry_hash": "stale-hash"})
    assert "registry_changed_since_generation" in _call(drifted)["degraded"]

    live = registry_hash(tmp_path / "bin" / "registry.json")
    same = _mesh(tmp_path, manifest_extra={"registry_hash": live})
    assert "registry_changed_since_generation" not in _call(same)["degraded"]


def test_no_registry_internals_and_schema_listed(tmp_path):
    mesh = _mesh(tmp_path)
    response = mesh.call_tool("list_repos", {})
    assert "collection_path" not in response["content"][0]["text"]
    schema = {tool["name"]: tool for tool in mesh.tool_schemas()}["list_repos"]
    assert "cwd" in schema["inputSchema"]["properties"]
