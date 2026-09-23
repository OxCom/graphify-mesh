"""`neighbors` payload bounds and edge-confidence handling.

Review of e3997cc (2026-09-23) found two ways a result overstates itself:

1. Every seed matching an ambiguous label is emitted before MAX_TRAVERSAL_NODES
   is applied, so 2001 seeds plus one node came back `truncated: false,
   complete: true` despite the documented 2000-node payload cap.
2. Only INFERRED edges are filtered. An AMBIGUOUS edge is followed by default
   and the result still reports `reliability: "exact"`.
"""

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

from graphify_mesh.server import traverse
from graphify_mesh.server.config import ServerConfig
from graphify_mesh.server.server import GraphifyMeshServer

REPO = "repo.a"


def _call(tmp_path: Path, nodes, links, **arguments) -> dict:
    registry_path = tmp_path / "bin" / "registry.json"
    write_registry(registry_path, [registry_repo(REPO, tmp_path / "a")])
    config = ServerConfig.from_env(mesh_root=tmp_path, registry_path=registry_path)
    server = GraphifyMeshServer(config, cwd=tmp_path, embed_query_fn=fake_embed_query_fn())
    generation = build_generation(nodes, links)
    server._generation = lambda: generation  # type: ignore[method-assign]
    result = server.call_tool("neighbors", {"repo": REPO, **arguments})
    assert result["isError"] is False, result["content"][0]["text"]
    return json.loads(result["content"][0]["text"])


class TestSeedsCountAgainstTheCap:
    def test_more_seeds_than_the_cap_is_not_complete(self, tmp_path, monkeypatch):
        monkeypatch.setattr(traverse, "MAX_TRAVERSAL_NODES", 2)
        handlers = [make_node(REPO, "Handler", f"src/h{i}.py", node_id=f"h{i}") for i in range(3)]
        child = make_node(REPO, "Child", "src/child.py", node_id="child")
        links = [make_link("child", "h0", relation="inherits")]

        payload = _call(
            tmp_path, [*handlers, child], links, node="Handler", relation="inherits", direction="in"
        )

        assert payload["truncated"] is True
        assert payload["complete"] is False
        assert len(payload["seeds"]) + payload["count"] <= 2

    def test_seeds_plus_nodes_stay_within_the_cap(self, tmp_path, monkeypatch):
        monkeypatch.setattr(traverse, "MAX_TRAVERSAL_NODES", 3)
        handlers = [make_node(REPO, "Handler", f"src/h{i}.py", node_id=f"h{i}") for i in range(2)]
        children = [make_node(REPO, f"C{i}", f"src/c{i}.py", node_id=f"c{i}") for i in range(2)]
        links = [make_link(f"c{i}", "h0", relation="inherits") for i in range(2)]

        payload = _call(
            tmp_path,
            [*handlers, *children],
            links,
            node="Handler",
            relation="inherits",
            direction="in",
        )

        assert len(payload["seeds"]) + payload["count"] <= 3
        assert payload["complete"] is False


class TestAmbiguousEdgesAreNotExact:
    NODES = [
        make_node(REPO, "Base", "src/base.py", node_id="base"),
        make_node(REPO, "Mid", "src/mid.py", node_id="mid"),
    ]
    LINKS = [make_link("mid", "base", confidence="AMBIGUOUS", relation="inherits")]

    def test_ambiguous_edge_is_not_followed_by_default(self, tmp_path):
        payload = _call(
            tmp_path, self.NODES, self.LINKS, node="Base", relation="inherits", direction="in"
        )
        assert payload["nodes"] == []
        assert payload["reliability"] == "exact"

    def test_opted_in_ambiguous_edge_makes_the_result_partial(self, tmp_path):
        payload = _call(
            tmp_path,
            self.NODES,
            self.LINKS,
            node="Base",
            relation="inherits",
            direction="in",
            include_inferred=True,
        )
        assert [n["label"] for n in payload["nodes"]] == ["Mid"]
        assert payload["nodes"][0]["via"][0]["confidence"] == "AMBIGUOUS"
        assert payload["reliability"] == "partial"
