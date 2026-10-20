"""`cross_project(exclude_repos=...)`: subtracts repos from the search set."""

from __future__ import annotations

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

A = "a.repo"
B = "b.repo"


def _mesh(tmp_path: Path) -> GraphifyMeshServer:
    roots = {A: tmp_path / "a", B: tmp_path / "b"}
    for root in roots.values():
        root.mkdir()
    registry_path = tmp_path / "bin" / "registry.json"
    write_registry(registry_path, [registry_repo(r, root) for r, root in roots.items()])
    generation = build_generation(
        [
            make_node(A, "OrderCalculator", "src/a.py", node_id="a1"),
            make_node(B, "OrderCalculatorTwin", "src/b.py", node_id="b1"),
        ],
        overlay_edges=[],
    )
    config = ServerConfig(mesh_root=tmp_path, registry_path=registry_path, transport="stdio")
    mesh = GraphifyMeshServer(config, cwd=roots[A], embed_query_fn=fake_embed_query_fn())
    mesh._generation = lambda: generation  # type: ignore[method-assign]
    return mesh


def _repos(result: dict) -> set[str]:
    return {hit["repo"] for hit in result["hits"]}


def test_baseline_returns_both_repos(tmp_path):
    result = _mesh(tmp_path).tool_cross_project({"q": "OrderCalculator", "k": 10})
    assert _repos(result) == {A, B}


def test_exclude_removes_repo(tmp_path):
    result = _mesh(tmp_path).tool_cross_project(
        {"q": "OrderCalculator", "k": 10, "exclude_repos": [A]}
    )
    assert _repos(result) == {B}


def test_exclude_unknown_id_is_an_error(tmp_path):
    with pytest.raises(ToolError, match="exclude_repos"):
        _mesh(tmp_path).tool_cross_project({"q": "x", "exclude_repos": ["nope.repo"]})


def test_exclude_every_repo_is_an_error(tmp_path):
    with pytest.raises(ToolError, match="leaves no repo"):
        _mesh(tmp_path).tool_cross_project({"q": "x", "exclude_repos": [A, B]})


def test_repos_minus_exclude_leaving_nothing_is_an_error(tmp_path):
    with pytest.raises(ToolError, match="leaves no repo"):
        _mesh(tmp_path).tool_cross_project({"q": "x", "repos": [A], "exclude_repos": [A]})


def test_repos_with_exclude_is_set_difference(tmp_path):
    result = _mesh(tmp_path).tool_cross_project(
        {"q": "OrderCalculator", "k": 10, "repos": [A, B], "exclude_repos": [A]}
    )
    assert _repos(result) == {B}


@pytest.mark.parametrize("bad", ["a.repo", [1], {"a": 1}])
def test_exclude_bad_type_is_an_error(tmp_path, bad):
    with pytest.raises(ToolError, match="exclude_repos"):
        _mesh(tmp_path).tool_cross_project({"q": "x", "exclude_repos": bad})


def test_schema_advertises_exclude_repos_and_degraded_note(tmp_path):
    tools = {t["name"]: t for t in _mesh(tmp_path).tool_schemas()}
    tool = tools["cross_project"]
    assert "exclude_repos" in tool["inputSchema"]["properties"]
    assert "embeddings_unavailable" in tool["description"]
