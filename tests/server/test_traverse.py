"""`neighbors` tool: complete, deterministic traversal over chosen relations
inside one registered repo, wired through `GraphifyMeshServer.call_tool`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import (
    build_generation,
    fake_embed_query_fn,
    key_for,
    make_link,
    make_node,
    registry_repo,
    write_registry,
)

from graphify_mesh.server import traverse
from graphify_mesh.server.config import ServerConfig
from graphify_mesh.server.server import GraphifyMeshServer

REPO = "repo.a"
OTHER = "repo.b"

# inherits: source = child, target = parent.
#   Base <- Mid <- Leaf1 <- Deep
#               <- Leaf2
BASE = make_node(REPO, "Base", "src/base.py", node_id="base", line=3)
MID = make_node(REPO, "Mid", "src/mid.py", node_id="mid")
LEAF1 = make_node(REPO, "Leaf1", "src/leaf1.py", node_id="leaf1")
LEAF2 = make_node(REPO, "Leaf2", "src/leaf2.py", node_id="leaf2")
DEEP = make_node(REPO, "Deep", "src/deep.py", node_id="deep")
UTIL = make_node(REPO, "util", "src/util.py", node_id="util")
# Same label as BASE in another repo, with a (forbidden-shaped) edge into BASE.
FOREIGN_BASE = make_node(OTHER, "Base", "src/base.py", node_id="b_base")
FOREIGN_CHILD = make_node(OTHER, "ForeignChild", "src/fc.py", node_id="b_child")

NODES = [BASE, MID, LEAF1, LEAF2, DEEP, UTIL, FOREIGN_BASE, FOREIGN_CHILD]
LINKS = [
    make_link("mid", "base", relation="inherits"),
    make_link("leaf1", "mid", relation="inherits"),
    make_link("leaf2", "mid", relation="inherits"),
    make_link("deep", "leaf1", relation="inherits"),
    make_link("b_child", "base", relation="inherits"),
    make_link("b_child", "b_base", relation="inherits"),
    make_link("mid", "util", relation="imports"),
    make_link("base", "base", relation="inherits"),  # self-loop
]


def _server(tmp_path: Path, generation, repos=None, disabled=None) -> GraphifyMeshServer:
    registry_path = tmp_path / "bin" / "registry.json"
    entries = repos or [registry_repo(REPO, tmp_path / "a"), registry_repo(OTHER, tmp_path / "b")]
    write_registry(registry_path, entries, disabled=disabled)
    config = ServerConfig.from_env(mesh_root=tmp_path, registry_path=registry_path)
    server = GraphifyMeshServer(config, cwd=tmp_path, embed_query_fn=fake_embed_query_fn())
    server._generation = lambda: generation  # type: ignore[method-assign]
    return server


def _call(server: GraphifyMeshServer, **arguments) -> dict:
    result = server.call_tool("neighbors", arguments)
    assert result["isError"] is False, result["content"][0]["text"]
    return json.loads(result["content"][0]["text"])


def _error(server: GraphifyMeshServer, **arguments) -> str:
    result = server.call_tool("neighbors", arguments)
    assert result["isError"] is True
    return result["content"][0]["text"]


@pytest.fixture()
def server(tmp_path):
    return _server(tmp_path, build_generation(NODES, LINKS))


def _labels(payload: dict) -> list[str]:
    return [n["label"] for n in payload["nodes"]]


def test_transitive_subtypes_over_inherits_in(server):
    payload = _call(server, node="Base", repo=REPO, relation="inherits", direction="in", depth=5)
    assert payload["resolved"] is True
    by_label = {n["label"]: n for n in payload["nodes"]}
    assert set(by_label) == {"Mid", "Leaf1", "Leaf2", "Deep"}
    assert {k: v["depth"] for k, v in by_label.items()} == {
        "Mid": 1,
        "Leaf1": 2,
        "Leaf2": 2,
        "Deep": 3,
    }
    assert by_label["Deep"]["via"] == [
        {"relation": "inherits", "from": key_for(REPO, LEAF1), "confidence": "EXTRACTED"}
    ]
    assert payload["complete"] is True
    assert payload["truncated"] is False
    assert payload["frontier_exhausted"] is True
    assert payload["reliability"] == "exact"
    assert payload["count"] == 4
    assert payload["relations"] == ["inherits"]
    assert payload["generation_id"] == "gen-test-1"
    assert payload["seeds"] == [
        {
            "key": key_for(REPO, BASE),
            "label": "Base",
            "source_file": "src/base.py",
            "citation": f"[{REPO}:src/base.py:3]",
        }
    ]
    # sorted by (depth, key); seed and the self-loop never reappear
    assert [(n["depth"], n["key"]) for n in payload["nodes"]] == sorted(
        (n["depth"], n["key"]) for n in payload["nodes"]
    )
    assert "Base" not in _labels(payload)
    assert any("not a ranked top-k" in note for note in payload["notes"])


def test_depth_one_limits_to_direct_children(server):
    payload = _call(server, node="Base", repo=REPO, relation="inherits", direction="in")
    assert _labels(payload) == ["Mid"]
    assert payload["depth"] == 1
    assert payload["frontier_exhausted"] is False
    assert payload["complete"] is True


def test_direction_out_gives_parents(server):
    payload = _call(server, node="Deep", repo=REPO, relation="inherits", direction="out", depth=16)
    assert {n["label"]: n["depth"] for n in payload["nodes"]} == {
        "Leaf1": 1,
        "Mid": 2,
        "Base": 3,
    }


def test_direction_both(server):
    payload = _call(
        server, node="Mid", repo=REPO, relation=["inherits", "imports"], direction="both"
    )
    assert set(_labels(payload)) == {"Base", "Leaf1", "Leaf2", "util"}
    assert payload["relations"] == ["imports", "inherits"]


def test_repo_isolation(server):
    payload = _call(server, node="Base", repo=REPO, relation="inherits", direction="in", depth=3)
    assert [s["key"] for s in payload["seeds"]] == [key_for(REPO, BASE)]
    assert "ForeignChild" not in _labels(payload)
    other = _call(server, node="Base", repo=OTHER, relation="inherits", direction="in")
    assert [s["key"] for s in other["seeds"]] == [key_for(OTHER, FOREIGN_BASE)]
    assert _labels(other) == ["ForeignChild"]


def test_label_ambiguity_gives_multiple_seeds(tmp_path):
    a1 = make_node(REPO, "Handler", "src/a.py", node_id="h1")
    a2 = make_node(REPO, "Handler", "src/b.py", node_id="h2")
    c1 = make_node(REPO, "C1", "src/c1.py", node_id="c1")
    c2 = make_node(REPO, "C2", "src/c2.py", node_id="c2")
    links = [make_link("c1", "h1", relation="inherits"), make_link("c2", "h2", relation="inherits")]
    server = _server(tmp_path, build_generation([a1, a2, c1, c2], links))
    payload = _call(server, node="Handler", repo=REPO, relation="inherits", direction="in")
    assert [s["key"] for s in payload["seeds"]] == sorted([key_for(REPO, a1), key_for(REPO, a2)])
    assert set(_labels(payload)) == {"C1", "C2"}


def test_case_insensitive_label_fallback(server):
    payload = _call(server, node="BASE", repo=REPO, relation="inherits", direction="in")
    assert [s["label"] for s in payload["seeds"]] == ["Base"]


def test_key_and_node_id_resolution(server):
    by_key = _call(server, node=key_for(REPO, MID), repo=REPO, relation="inherits", direction="in")
    by_id = _call(server, node="mid", repo=REPO, relation="inherits", direction="in")
    assert by_key["seeds"] == by_id["seeds"]
    assert set(_labels(by_key)) == {"Leaf1", "Leaf2"}


def test_node_not_found(server):
    payload = _call(server, node="Nope", repo=REPO, relation="inherits", direction="in")
    assert payload["resolved"] is False
    assert payload["degraded"] == ["node_not_found"]
    assert payload["nodes"] == []


def test_inferred_skipped_by_default_and_included_with_flag(tmp_path):
    nodes = [BASE, MID]
    links = [make_link("mid", "base", confidence="INFERRED", relation="inherits")]
    server = _server(tmp_path, build_generation(nodes, links))
    default = _call(server, node="Base", repo=REPO, relation="inherits", direction="in")
    assert default["nodes"] == []
    assert default["reliability"] == "exact"
    included = _call(
        server,
        node="Base",
        repo=REPO,
        relation="inherits",
        direction="in",
        include_inferred=True,
    )
    assert _labels(included) == ["Mid"]
    assert included["nodes"][0]["via"][0]["confidence"] == "INFERRED"
    assert included["reliability"] == "partial"
    assert traverse.NOTE_INFERRED in included["notes"]


def test_calls_relation_is_partial_with_di_note(tmp_path):
    caller = make_node(REPO, "caller", "src/x.py", node_id="caller")
    callee = make_node(REPO, "callee", "src/y.py", node_id="callee")
    server = _server(
        tmp_path,
        build_generation([caller, callee], [make_link("caller", "callee", relation="calls")]),
    )
    payload = _call(server, node="callee", repo=REPO, relation="calls", direction="in")
    assert _labels(payload) == ["caller"]
    assert payload["reliability"] == "partial"
    assert traverse.NOTE_PARTIAL_RELATIONS in payload["notes"]
    assert payload["complete"] is True


def test_bare_method_name_resolves_to_method_label(tmp_path):
    method = make_node(REPO, ".reminderSubmitWeek()", "src/emails.py", node_id="m")
    caller = make_node(REPO, ".run()", "src/job.py", node_id="c")
    server = _server(
        tmp_path,
        build_generation([method, caller], [make_link("c", "m", relation="calls")]),
    )
    payload = _call(server, node="reminderSubmitWeek", repo=REPO, relation="calls", direction="in")
    assert [s["label"] for s in payload["seeds"]] == [".reminderSubmitWeek()"]
    assert _labels(payload) == [".run()"]


def test_unknown_relation_is_error_listing_known(server):
    text = _error(server, node="Base", repo=REPO, relation="inherts", direction="in")
    assert "inherts" in text
    assert "imports, inherits" in text


def test_unregistered_and_disabled_repo_fail_closed(tmp_path):
    generation = build_generation(NODES, LINKS)
    server = _server(
        tmp_path,
        generation,
        repos=[registry_repo(REPO, tmp_path / "a"), registry_repo(OTHER, tmp_path / "b", False)],
    )
    for repo in ("repo.unknown", OTHER):
        payload = _call(server, node="Base", repo=repo, relation="inherits", direction="in")
        assert payload["resolved"] is False
        assert payload["degraded"] == ["repo_not_registered"]
        assert payload["complete"] is False
        assert payload["nodes"] == []


def test_registered_repo_absent_from_generation(tmp_path):
    server = _server(
        tmp_path,
        build_generation(NODES, LINKS),
        repos=[registry_repo("repo.new", tmp_path / "n")],
    )
    payload = _call(server, node="Base", repo="repo.new", relation="inherits", direction="in")
    assert payload["resolved"] is False
    assert payload["degraded"] == ["repo_not_in_current_generation"]


def test_truncation_at_cap(server, monkeypatch):
    # the cap counts the single Base seed too, leaving room for two nodes
    monkeypatch.setattr(traverse, "MAX_TRAVERSAL_NODES", 3)
    payload = _call(server, node="Base", repo=REPO, relation="inherits", direction="in", depth=5)
    assert payload["count"] == 2
    assert payload["truncated"] is True
    assert payload["complete"] is False
    assert payload["frontier_exhausted"] is False
    # deterministic cut: Mid (depth 1) plus the lower key of the depth-2 pair
    assert _labels(payload)[0] == "Mid"
    assert payload["nodes"][1]["key"] == min(key_for(REPO, LEAF1), key_for(REPO, LEAF2))
    assert traverse.NOTE_TRUNCATED in payload["notes"]
    assert traverse.NOTE_COMPLETE_SET not in payload["notes"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"depth": True},
        {"depth": 0},
        {"depth": traverse.MAX_DEPTH + 1},
        {"depth": "2"},
        {"direction": "up"},
        {"direction": None},
        {"relation": []},
        {"relation": [1]},
        {"relation": ""},
        {"relation": ["x" * (traverse.MAX_RELATION_LENGTH + 1)]},
        {"relation": ["inherits"] * (traverse.MAX_RELATIONS + 1)},
        {"relation": None},
        {"node": ""},
        {"node": 5},
        {"repo": ""},
        {"include_inferred": "yes"},
    ],
)
def test_bad_arguments_are_tool_errors(server, overrides):
    arguments = {"node": "Base", "repo": REPO, "relation": "inherits", "direction": "in"}
    arguments.update(overrides)
    _error(server, **arguments)


def test_neighbors_listed_in_tool_schemas(server):
    schema = next(s for s in server.tool_schemas() if s["name"] == "neighbors")
    assert schema["inputSchema"]["required"] == ["node", "repo", "relation", "direction"]
    assert "top-k" in schema["description"]


def test_generation_relations_index():
    generation = build_generation(NODES, LINKS)
    assert generation.relations == frozenset({"inherits", "imports"})
