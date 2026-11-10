"""Purpose-named wrappers (`subclasses`, `importers`), namespaced-name
resolution with near-match suggestions in `neighbors`, the MCP server
instructions, and test-code demotion in `search`."""

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

from graphify_mesh.server import ranking, retrieval
from graphify_mesh.server.config import ServerConfig
from graphify_mesh.server.sdk_app import build_sdk_server
from graphify_mesh.server.server import SERVER_INSTRUCTIONS, GraphifyMeshServer

REPO = "app.hub"

NODES = [
    make_node(REPO, "AbstractForm", "src/Form/AbstractForm.php", node_id="form"),
    # Same short name in another namespace: FQCN must pick the right one.
    make_node(REPO, "AbstractForm", "src/Legacy/AbstractForm.php", node_id="legacy"),
    make_node(REPO, "TeamForm", "src/Form/TeamForm.php", node_id="team"),
    make_node(REPO, "LeadTeamForm", "src/Form/LeadTeamForm.php", node_id="lead"),
    make_node(REPO, "LegacyForm", "src/Legacy/LegacyForm.php", node_id="legacy_child"),
    make_node(REPO, "Listable", "src/Form/Listable.php", node_id="iface"),
    make_node(REPO, ".submit()", "src/Form/TeamForm.php", node_id="submit"),
    make_node(REPO, "consumer.php", "src/consumer.php", node_id="consumer"),
]
LINKS = [
    make_link("team", "form", relation="inherits"),
    make_link("lead", "team", relation="inherits"),
    make_link("legacy_child", "legacy", relation="inherits"),
    make_link("team", "iface", relation="implements"),
    make_link("team", "submit", relation="method"),
    make_link("consumer", "form", relation="imports"),
]


def _server(tmp_path: Path, nodes=NODES, links=LINKS) -> GraphifyMeshServer:
    registry_path = tmp_path / "bin" / "registry.json"
    write_registry(registry_path, [registry_repo(REPO, tmp_path / "hub")])
    config = ServerConfig.from_env(mesh_root=tmp_path, registry_path=registry_path)
    server = GraphifyMeshServer(config, cwd=tmp_path, embed_query_fn=fake_embed_query_fn())
    generation = build_generation(nodes, links)
    server._generation = lambda: generation  # type: ignore[method-assign]
    return server


def _call(server, tool, **arguments) -> dict:
    result = server.call_tool(tool, arguments)
    assert result["isError"] is False, result["content"][0]["text"]
    return json.loads(result["content"][0]["text"])


def _labels(payload) -> set[str]:
    return {node["label"] for node in payload["nodes"]}


def test_subclasses_is_transitive_over_inherits_and_implements(tmp_path):
    payload = _call(_server(tmp_path), "subclasses", node="Listable", repo=REPO)
    assert payload["relations"] == ["implements", "inherits"]
    assert payload["direction"] == "in" and payload["depth"] == 16
    assert _labels(payload) == {"TeamForm", "LeadTeamForm"}
    assert payload["complete"] and payload["frontier_exhausted"]
    assert payload["reliability"] == "exact"


def test_fqcn_resolves_to_the_namespaced_class_only(tmp_path):
    server = _server(tmp_path)
    for node in ("App\\Form\\AbstractForm", "\\App\\Form\\AbstractForm"):
        payload = _call(server, "subclasses", node=node, repo=REPO)
        assert [seed["source_file"] for seed in payload["seeds"]] == ["src/Form/AbstractForm.php"]
        assert _labels(payload) == {"TeamForm", "LeadTeamForm"}


def test_fqcn_through_neighbors_too(tmp_path):
    payload = _call(
        _server(tmp_path),
        "neighbors",
        node="App\\Form\\AbstractForm",
        repo=REPO,
        relation="inherits",
        direction="in",
        depth=16,
    )
    assert payload["resolved"] is True
    assert _labels(payload) == {"TeamForm", "LeadTeamForm"}


def test_class_method_form_resolves_to_the_method_node(tmp_path):
    payload = _call(
        _server(tmp_path),
        "neighbors",
        node="App\\Form\\TeamForm::submit",
        repo=REPO,
        relation="method",
        direction="in",
    )
    assert [seed["label"] for seed in payload["seeds"]] == [".submit()"]
    assert _labels(payload) == {"TeamForm"}


def test_miss_returns_near_labels(tmp_path):
    payload = _call(_server(tmp_path), "subclasses", node="App\\Form\\AbstractFrom", repo=REPO)
    assert payload["resolved"] is False
    assert payload["degraded"] == ["node_not_found"]
    assert "AbstractForm" in payload["suggestions"]


def test_importers_are_direct_by_default(tmp_path):
    payload = _call(_server(tmp_path), "importers", node="App\\Form\\AbstractForm", repo=REPO)
    assert payload["relations"] == ["imports"]
    assert _labels(payload) == {"consumer.php"}


def test_purpose_tools_fail_closed_on_unregistered_repo(tmp_path):
    payload = _call(_server(tmp_path), "subclasses", node="AbstractForm", repo="nope")
    assert payload["degraded"] == ["repo_not_registered"]


def test_existing_tools_are_kept_and_new_ones_listed(tmp_path):
    names = [schema["name"] for schema in _server(tmp_path).tool_schemas()]
    assert names == [
        "search",
        "cross_project",
        "find_similar",
        "project_map",
        "context_pack",
        "neighbors",
        "list_repos",
        "subclasses",
        "importers",
        "route_handler",
        "controller_outline",
    ]


def test_server_sends_instructions_naming_every_purpose_tool(tmp_path):
    options = build_sdk_server(_server(tmp_path)).create_initialization_options()
    assert options.instructions == SERVER_INSTRUCTIONS
    for name in ("subclasses", "importers", "route_handler", "controller_outline", "search"):
        assert f"({name})" in SERVER_INSTRUCTIONS
    assert "grep those" in SERVER_INSTRUCTIONS


def test_is_test_path_and_query_detection():
    for path in (
        "tests/php/Form/TeamsTest.php",
        "src/Form/TeamsListingCriteriaTest.php",
        "assets/src/__tests__/a.ts",
        "pkg/x_test.go",
        "test_x.py",
        "web/a.spec.tsx",
    ):
        assert ranking.is_test_path(path), path
    for path in ("src/Form/TeamsListingCriteria.php", "src/Testimonial.php", "", "src/latest.py"):
        assert not ranking.is_test_path(path), path
    assert ranking.query_mentions_tests("teams listing test")
    assert ranking.query_mentions_tests("phpunit TeamsTest")
    assert not ranking.query_mentions_tests("my team sub teams listing filter")


def _team_listing_generation():
    """Ten test methods that repeat the query words ahead of the one source
    class, the shape of the cem.hub probe that lost the source file."""
    nodes = [
        make_node(
            REPO, f".testTeamsListingFilter{i}()", f"tests/Form/T{i}Test.php", node_id=f"t{i}"
        )
        for i in range(10)
    ]
    # Matches one query word of three; every test method matches all three.
    nodes.append(make_node(REPO, "ListingCriteria", "src/Form/ListingCriteria.php"))
    return build_generation(nodes, [])


def test_search_ranks_source_above_tests_unless_asked(monkeypatch):
    generation = _team_listing_generation()
    repos = frozenset({REPO})
    query = "teams listing filter"
    plain = retrieval.rank(query, generation, repos, 3, fake_embed_query_fn())
    assert plain.hits[0].source_file == "src/Form/ListingCriteria.php"
    # Same query with demotion off (what a query naming tests gets): the
    # test methods, which match every query word, win.
    monkeypatch.setattr(ranking, "query_mentions_tests", lambda query: True)
    asked = retrieval.rank(query, generation, repos, 3, fake_embed_query_fn())
    assert ranking.is_test_path(asked.hits[0].source_file), [h.source_file for h in asked.hits]


def test_test_penalty_is_multiplicative_and_opt_in():
    paths = {"t": "tests/a_test.py", "s": "src/a.py"}
    assert ranking.apply_penalties("t", 1.0, {}, paths) == 1.0
    assert ranking.apply_penalties("t", 1.0, {}, paths, demote_tests=True) == (
        ranking.TEST_PATH_PENALTY_FACTOR
    )
    assert ranking.apply_penalties("s", 1.0, {}, paths, demote_tests=True) == 1.0
