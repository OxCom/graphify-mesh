"""A repo the registry disables must not be served, however it reaches a tool.

The published generation still carries every repo that was enabled when it was
built, so `registry.json` is the only thing that says a repo is out. Each tool
used to consult it differently: `cross_project` and `scope='all'` resolved to
"no filter at all", `project_map` accepted any registered repo_id, and
`find_similar` never read the registry. These tests pin one answer for all four.

The second half pins the fail-closed `cwd` rule for the shared HTTP daemon.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import (
    build_generation,
    fake_embed_query_fn,
    key_for,
    make_node,
    registry_repo,
    write_registry,
)

from graphify_mesh.server.config import ServerConfig
from graphify_mesh.server.server import GraphifyMeshServer, ToolError

LIVE = "live.repo"
DEAD = "dead.repo"


def _two_repo_mesh(tmp_path: Path, transport: str = "stdio") -> GraphifyMeshServer:
    """One enabled repo and one disabled repo, both present in the published
    generation with a `similar_approach` edge between them — the shape a mesh
    has right after a repo is disabled but before the next sync republishes."""
    live_root, dead_root = tmp_path / "live", tmp_path / "dead"
    live_root.mkdir()
    dead_root.mkdir()
    registry_path = tmp_path / "bin" / "registry.json"
    write_registry(
        registry_path,
        [registry_repo(LIVE, live_root), registry_repo(DEAD, dead_root, enabled=False)],
    )

    live_node = make_node(LIVE, "OrderCalculator", "src/order.py", node_id="live1")
    dead_node = make_node(DEAD, "OrderCalculatorClone", "src/clone.py", node_id="dead1")
    generation = build_generation(
        [live_node, dead_node],
        overlay_edges=[
            {
                "type": "similar_approach",
                "source": {
                    "repo": LIVE,
                    "source_file": "src/order.py",
                    "qualified_label": "OrderCalculator",
                },
                "target": {
                    "repo": DEAD,
                    "source_file": "src/clone.py",
                    "qualified_label": "OrderCalculatorClone",
                },
                "confidence": 0.9,
                "provenance": "ANN_COSINE",
            }
        ],
    )
    config = ServerConfig(
        mesh_root=tmp_path,
        registry_path=registry_path,
        transport=transport,
        http_token="test-token" if transport == "http" else None,  # noqa: S106
    )
    mesh = GraphifyMeshServer(config, cwd=live_root, embed_query_fn=fake_embed_query_fn())
    mesh._generation = lambda: generation  # type: ignore[method-assign]
    mesh._dead_key = key_for(DEAD, dead_node)  # type: ignore[attr-defined]
    return mesh


def _repos_in(result: dict) -> set[str]:
    return {hit["repo"] for hit in result["hits"]}


def test_cross_project_without_a_repo_list_excludes_a_disabled_repo(tmp_path):
    mesh = _two_repo_mesh(tmp_path)
    result = mesh.tool_cross_project({"q": "OrderCalculator", "k": 10})
    assert DEAD not in _repos_in(result)


def test_cross_project_rejects_a_disabled_repo_named_explicitly(tmp_path):
    mesh = _two_repo_mesh(tmp_path)
    with pytest.raises(ToolError):
        mesh.tool_cross_project({"q": "OrderCalculator", "repos": [DEAD]})


def test_scope_all_search_excludes_a_disabled_repo(tmp_path):
    mesh = _two_repo_mesh(tmp_path)
    result = mesh.tool_search({"q": "OrderCalculator", "scope": "all", "k": 10})
    assert result["scope_mode"] == "all"
    assert DEAD not in _repos_in(result)


def test_context_pack_scope_all_excludes_a_disabled_repo(tmp_path):
    mesh = _two_repo_mesh(tmp_path)
    result = mesh.tool_context_pack({"goal": "OrderCalculator", "scope": "all"})
    assert DEAD not in {card["repo"] for card in result["cards"]}


def test_project_map_refuses_a_disabled_repo(tmp_path):
    mesh = _two_repo_mesh(tmp_path)
    result = mesh.tool_project_map({"repo": DEAD})
    assert result["resolved"] is False
    assert result["degraded"] == ["repo_not_registered"]

    assert mesh.tool_project_map({"repo": LIVE})["resolved"] is True


def test_find_similar_drops_hits_from_a_disabled_repo(tmp_path):
    """The overlay edge points straight at the disabled repo, so the enabled-repo
    set handed to `similar.py` is the only thing that can drop it. `resolved`
    stays True: the QUERY node was found, which is what it reports."""
    mesh = _two_repo_mesh(tmp_path)
    result = mesh.tool_find_similar({"node": "OrderCalculator", "k": 10})
    assert result["resolved"] is True
    assert result["hits"] == []
    assert "disabled_repo_hits_filtered" in result["degraded"]


def test_find_similar_does_not_mark_degraded_when_nothing_was_filtered(tmp_path):
    live_root = tmp_path / "live"
    live_root.mkdir()
    registry_path = tmp_path / "bin" / "registry.json"
    write_registry(registry_path, [registry_repo(LIVE, live_root)])
    seed = make_node(LIVE, "PaymentGateway", "src/gateway.py", node_id="s1")
    neighbor = make_node(LIVE, "PaymentAdapter", "src/adapter.py", node_id="s2")
    generation = build_generation(
        [seed, neighbor], links=[{"source": "s1", "target": "s2", "confidence": "EXTRACTED"}]
    )
    config = ServerConfig(mesh_root=tmp_path, registry_path=registry_path)
    mesh = GraphifyMeshServer(config, cwd=live_root, embed_query_fn=fake_embed_query_fn())
    mesh._generation = lambda: generation  # type: ignore[method-assign]

    result = mesh.tool_find_similar({"node": "PaymentGateway", "k": 10})
    assert {hit["repo"] for hit in result["hits"]} == {LIVE}
    assert "disabled_repo_hits_filtered" not in result["degraded"]


def _seed_with_disabled_top_neighbor(tmp_path: Path) -> GraphifyMeshServer:
    """The seed has two neighbors: a disabled-repo one scoring 0.9 through the
    overlay, and an enabled same-repo structural one scoring lower. With k=1
    the disabled one used to win the single slot and then get dropped, leaving
    the caller with nothing."""
    live_root, dead_root = tmp_path / "live", tmp_path / "dead"
    live_root.mkdir()
    dead_root.mkdir()
    registry_path = tmp_path / "bin" / "registry.json"
    write_registry(
        registry_path,
        [registry_repo(LIVE, live_root), registry_repo(DEAD, dead_root, enabled=False)],
    )

    seed = make_node(LIVE, "OrderCalculator", "src/order.py", node_id="live1")
    runner_up = make_node(LIVE, "OrderTotals", "src/totals.py", node_id="live2")
    dead_node = make_node(DEAD, "OrderCalculatorClone", "src/clone.py", node_id="dead1")
    generation = build_generation(
        [seed, runner_up, dead_node],
        links=[{"source": "live1", "target": "live2", "confidence": "EXTRACTED"}],
        overlay_edges=[
            {
                "type": "similar_approach",
                "source": {
                    "repo": LIVE,
                    "source_file": "src/order.py",
                    "qualified_label": "OrderCalculator",
                },
                "target": {
                    "repo": DEAD,
                    "source_file": "src/clone.py",
                    "qualified_label": "OrderCalculatorClone",
                },
                "confidence": 0.9,
                "provenance": "ANN_COSINE",
            }
        ],
    )
    config = ServerConfig(mesh_root=tmp_path, registry_path=registry_path)
    mesh = GraphifyMeshServer(config, cwd=live_root, embed_query_fn=fake_embed_query_fn())
    mesh._generation = lambda: generation  # type: ignore[method-assign]
    return mesh


def test_find_similar_k1_returns_the_enabled_runner_up(tmp_path):
    mesh = _seed_with_disabled_top_neighbor(tmp_path)
    result = mesh.tool_find_similar({"node": "OrderCalculator", "k": 1})
    assert result["resolved"] is True
    assert [hit["label"] for hit in result["hits"]] == ["OrderTotals"]
    assert "disabled_repo_hits_filtered" in result["degraded"]


def test_find_similar_keeps_resolved_true_when_every_neighbor_was_filtered(tmp_path):
    mesh = _two_repo_mesh(tmp_path)
    result = mesh.tool_find_similar({"node": "OrderCalculator", "k": 10})
    assert result["resolved"] is True
    assert result["hits"] == []


def test_find_similar_fallback_survives_filtering_and_skips_disabled_matches(tmp_path):
    """A trivial node with neither an overlay edge nor same-repo neighbors must
    still reach the label+community fallback after filtering empties the
    candidate set, and the fallback itself must skip disabled repos."""
    # All three nodes share label and path, so the alias lookup resolves the
    # seed by key order: these repo_ids put the seed's repo first.
    seed_repo, twin_repo, dead_repo = "a.seed", "b.twin", "c.dead"
    roots = {name: tmp_path / name for name in (seed_repo, twin_repo, dead_repo)}
    for root in roots.values():
        root.mkdir()
    registry_path = tmp_path / "bin" / "registry.json"
    write_registry(
        registry_path,
        [
            registry_repo(seed_repo, roots[seed_repo]),
            registry_repo(twin_repo, roots[twin_repo]),
            registry_repo(dead_repo, roots[dead_repo], enabled=False),
        ],
    )

    seed = make_node(seed_repo, "Widget", "src/w.py", node_id="l1", community_name="ui")
    twin_enabled = make_node(twin_repo, "Widget", "src/w.py", node_id="o1", community_name="ui")
    twin_disabled = make_node(dead_repo, "Widget", "src/w.py", node_id="d1", community_name="ui")
    generation = build_generation([seed, twin_enabled, twin_disabled])
    config = ServerConfig(mesh_root=tmp_path, registry_path=registry_path)
    mesh = GraphifyMeshServer(config, cwd=roots[seed_repo], embed_query_fn=fake_embed_query_fn())
    mesh._generation = lambda: generation  # type: ignore[method-assign]

    result = mesh.tool_find_similar({"node": "Widget", "k": 5})
    assert result["resolved"] is True
    assert _repos_in(result) == {twin_repo}
    assert "similarity_fallback_exact_match" in result["degraded"]
    assert "disabled_repo_hits_filtered" in result["degraded"]


def test_find_similar_refuses_a_seed_node_in_a_disabled_repo(tmp_path):
    """Fail closed on the seed, the same answer `project_map` gives for a
    disabled repo_id: a disabled repo is not a queryable surface."""
    mesh = _two_repo_mesh(tmp_path)
    result = mesh.tool_find_similar({"node": "OrderCalculatorClone", "k": 10})
    assert result["resolved"] is False
    assert result["hits"] == []
    assert "node_repo_disabled" in result["degraded"]


# --- implicit scope under the shared HTTP daemon --------------------------


@pytest.mark.parametrize("tool", ["tool_search", "tool_context_pack"])
@pytest.mark.parametrize("scope", [None, "", "current"])
def test_http_transport_refuses_implicit_scope_without_cwd(tmp_path, tool, scope):
    """The daemon's own directory is not the caller's. It can itself sit
    inside a registered project, and then the fallback answered with THAT
    project's results instead of failing closed."""
    mesh = _two_repo_mesh(tmp_path, transport="http")
    arguments: dict = {"q": "OrderCalculator", "goal": "OrderCalculator"}
    if scope is not None:
        arguments["scope"] = scope

    with pytest.raises(ToolError) as excinfo:
        getattr(mesh, tool)(arguments)
    assert "'cwd'" in str(excinfo.value)


def test_http_transport_accepts_an_explicit_scope_without_cwd(tmp_path):
    mesh = _two_repo_mesh(tmp_path, transport="http")
    result = mesh.tool_search({"q": "OrderCalculator", "scope": "all"})
    assert result["scope_mode"] == "all"


def test_http_transport_accepts_implicit_scope_with_cwd(tmp_path):
    mesh = _two_repo_mesh(tmp_path, transport="http")
    result = mesh.tool_search({"q": "OrderCalculator", "cwd": str(tmp_path / "live")})
    assert result["scope_mode"] == "repo"


def test_stdio_transport_keeps_the_process_cwd_fallback(tmp_path):
    """One process per client session is what makes the process directory the
    caller's, so stdio keeps the fallback the HTTP daemon must refuse."""
    mesh = _two_repo_mesh(tmp_path, transport="stdio")
    result = mesh.tool_search({"q": "OrderCalculator"})
    assert result["scope_mode"] == "repo"
    assert _repos_in(result) <= {LIVE}


def test_http_refusal_reaches_the_client_as_a_tool_error_not_a_crash(tmp_path):
    mesh = _two_repo_mesh(tmp_path, transport="http")
    response = mesh.call_tool("search", {"q": "OrderCalculator"})
    assert response["isError"] is True
    assert "'cwd'" in response["content"][0]["text"]
    with pytest.raises(json.JSONDecodeError):
        json.loads(response["content"][0]["text"])  # a message, not a result payload
