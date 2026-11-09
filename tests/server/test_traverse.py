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


def test_calls_note_without_resolver_edge_has_no_resolver_note(tmp_path):
    caller = make_node(REPO, "caller", "src/x.py", node_id="caller")
    callee = make_node(REPO, "callee", "src/y.py", node_id="callee")
    server = _server(
        tmp_path,
        build_generation([caller, callee], [make_link("caller", "callee", relation="calls")]),
    )
    payload = _call(server, node="callee", repo=REPO, relation="calls", direction="in")
    assert not any("resolver" in note for note in payload["notes"])
    assert not any("typed properties" in note or "~20%" in note for note in payload["notes"])


def test_calls_note_names_resolver_when_edge_carries_it(tmp_path):
    caller = make_node(REPO, "caller", "src/x.py", node_id="caller")
    callee = make_node(REPO, "callee", "src/y.py", node_id="callee")
    link = make_link("caller", "callee", relation="calls")
    link["_resolver"] = "some-resolver"
    server = _server(tmp_path, build_generation([caller, callee], [link]))
    payload = _call(server, node="callee", repo=REPO, relation="calls", direction="in")
    assert traverse.NOTE_PARTIAL_RELATIONS in payload["notes"]
    assert traverse.NOTE_RESOLVER.format(name="some-resolver") in payload["notes"]


def test_depth_bounded_note_replaces_complete_set_when_frontier_not_exhausted(server):
    payload = _call(server, node="Base", repo=REPO, relation="inherits", direction="in")
    assert payload["frontier_exhausted"] is False
    assert traverse.NOTE_COMPLETE_SET not in payload["notes"]
    assert (
        traverse.NOTE_DEPTH_BOUNDED.format(depth=1, max_depth=traverse.MAX_DEPTH)
        in payload["notes"]
    )


def test_complete_set_note_when_frontier_exhausted(server):
    payload = _call(server, node="Base", repo=REPO, relation="inherits", direction="in", depth=10)
    assert payload["frontier_exhausted"] is True
    assert traverse.NOTE_COMPLETE_SET in payload["notes"]


def test_unresolved_seed_emits_no_completeness_note(server):
    payload = _call(server, node="NoSuchNode", repo=REPO, relation="inherits", direction="in")
    assert payload["resolved"] is False
    assert traverse.NOTE_COMPLETE_SET not in payload["notes"]
    assert not any("Exact for the nodes within depth" in note for note in payload["notes"])


def test_resolver_on_edge_cut_by_node_cap_gives_no_note(tmp_path, monkeypatch):
    monkeypatch.setattr(traverse, "MAX_TRAVERSAL_NODES", 2)
    callee = make_node(REPO, "callee", "src/y.py", node_id="callee")
    a = make_node(REPO, "a", "src/a.py", node_id="a")
    z = make_node(REPO, "z", "src/z.py", node_id="z")
    link_a = make_link("a", "callee", relation="calls")
    link_z = make_link("z", "callee", relation="calls")
    link_z["_resolver"] = "cut-resolver"
    server = _server(tmp_path, build_generation([callee, a, z], [link_a, link_z]))
    payload = _call(server, node="callee", repo=REPO, relation="calls", direction="in")
    assert _labels(payload) == ["a"]
    assert payload["truncated"] is True
    assert not any("cut-resolver" in note for note in payload["notes"])


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


def _scored(src, dst, confidence, score, relation="inherits", also_seen=None):
    link = make_link(src, dst, confidence=confidence, relation=relation)
    link["confidence_score"] = score
    if also_seen is not None:
        link["also_seen"] = also_seen
    return link


def test_via_carries_confidence_score_and_also_seen_count(tmp_path):
    links = [
        _scored("mid", "base", "INFERRED", 0.75, also_seen=[{"x": 1}, {"x": 2}]),
        # Parallel link, same via key, lower score: max score is kept.
        _scored("mid", "base", "INFERRED", 0.5),
        _scored("leaf1", "base", "EXTRACTED", 1.0, also_seen=[]),
        _scored("leaf2", "base", "EXTRACTED", float("nan")),
    ]
    server = _server(tmp_path, build_generation([BASE, MID, LEAF1, LEAF2], links))
    payload = _call(
        server, node="Base", repo=REPO, relation="inherits", direction="in", include_inferred=True
    )
    by_label = {n["label"]: n for n in payload["nodes"]}
    base_key = key_for(REPO, BASE)
    assert by_label["Mid"]["via"] == [
        {
            "relation": "inherits",
            "from": base_key,
            "confidence": "INFERRED",
            "confidence_score": 0.75,
            "also_seen": 2,
        }
    ]
    # Empty also_seen is omitted; a non-finite score is omitted.
    assert by_label["Leaf1"]["via"] == [
        {
            "relation": "inherits",
            "from": base_key,
            "confidence": "EXTRACTED",
            "confidence_score": 1.0,
        }
    ]
    assert by_label["Leaf2"]["via"] == [
        {"relation": "inherits", "from": base_key, "confidence": "EXTRACTED"}
    ]
    assert payload["min_confidence_score"] is None


def test_via_is_independent_of_parallel_link_order(tmp_path):
    a = _scored("mid", "base", "INFERRED", 0.4, also_seen=[1])
    b = _scored("mid", "base", "INFERRED", 0.9)
    payloads = []
    for links in ([a, b], [b, a]):
        server = _server(tmp_path, build_generation([BASE, MID], links))
        payloads.append(
            _call(
                server,
                node="Base",
                repo=REPO,
                relation="inherits",
                direction="in",
                include_inferred=True,
            )["nodes"]
        )
    assert payloads[0] == payloads[1]
    assert payloads[0][0]["via"][0]["confidence_score"] == 0.9
    assert payloads[0][0]["via"][0]["also_seen"] == 1


def test_min_confidence_score_filters_inferred_keeps_extracted(tmp_path):
    links = [
        _scored("mid", "base", "INFERRED", 0.4),
        _scored("leaf1", "base", "INFERRED", 0.8),
        _scored("leaf2", "base", "EXTRACTED", 0.1),
        make_link("deep", "base", confidence="AMBIGUOUS", relation="inherits"),  # unscored
    ]
    server = _server(tmp_path, build_generation([BASE, MID, LEAF1, LEAF2, DEEP], links))
    arguments = {
        "node": "Base",
        "repo": REPO,
        "relation": "inherits",
        "direction": "in",
        "include_inferred": True,
    }
    unfiltered = _call(server, **arguments)
    assert _labels(unfiltered) == ["Deep", "Leaf1", "Leaf2", "Mid"]
    filtered = _call(server, **arguments, min_confidence_score=0.5)
    assert _labels(filtered) == ["Leaf1", "Leaf2"]
    assert filtered["min_confidence_score"] == 0.5
    # Boundary is inclusive.
    assert "Mid" in _labels(_call(server, **arguments, min_confidence_score=0.4))
    assert any("below 0.5" in note for note in filtered["notes"])
    assert not any("below" in note for note in unfiltered["notes"])
    assert filtered["include_inferred"] is True
    # A floor without include_inferred is an error.
    no_inferred = dict(arguments, include_inferred=False)
    _error(server, **no_inferred, min_confidence_score=0.0)


@pytest.mark.parametrize(("value", "echoed"), [(None, None), (0, 0.0), (1, 1.0), (0.5, 0.5)])
def test_min_confidence_score_accepted_values_echoed(server, value, echoed):
    payload = _call(
        server,
        node="Base",
        repo=REPO,
        relation="inherits",
        direction="in",
        include_inferred=True,
        min_confidence_score=value,
    )
    assert payload["min_confidence_score"] == echoed


@pytest.mark.parametrize("value", [-0.1, 1.5, "x", float("nan"), float("inf"), True, [0.5]])
def test_min_confidence_score_invalid_values_rejected(server, value):
    _error(
        server,
        node="Base",
        repo=REPO,
        relation="inherits",
        direction="in",
        include_inferred=True,
        min_confidence_score=value,
    )


def test_min_confidence_score_in_schema(server):
    schema = next(s for s in server.tool_schemas() if s["name"] == "neighbors")
    prop = schema["inputSchema"]["properties"]["min_confidence_score"]
    assert prop["type"] == ["number", "null"]
    assert (prop["minimum"], prop["maximum"]) == (0, 1)
