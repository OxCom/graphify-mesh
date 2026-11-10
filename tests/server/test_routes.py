"""`route_handler` and `controller_outline`: route resolution by full path,
route name, placeholder pattern and prefix, near-match suggestions, and the
outline's guard/check lists and coverage, through `GraphifyMeshServer.call_tool`."""

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

from graphify_mesh.server import routes
from graphify_mesh.server.config import ServerConfig
from graphify_mesh.server.server import GraphifyMeshServer

REPO = "app.hub"
FRONT = "src/Controller/Front/TeamController.php"
REST = "src/Controller/Rest/TagsController.php"


def _route(node_id, path, name, methods, source_file, line):
    return make_node(
        REPO,
        path,
        source_file,
        node_id=node_id,
        source_location=f"L{line}",
        _kind="http_route",
        route_path=path,
        route_name=name,
        http_methods=methods,
        route_complete=True,
    )


def _link(src, dst, relation, **extra):
    link = make_link(src, dst, relation=relation)
    link.update(extra)
    return link


NODES = [
    make_node(REPO, "TeamController", FRONT, node_id="c_team"),
    make_node(REPO, ".teams()", FRONT, node_id="m_teams", source_location="L20"),
    make_node(REPO, ".members()", FRONT, node_id="m_members", source_location="L30"),
    make_node(REPO, "TagsController", REST, node_id="c_tags"),
    make_node(REPO, ".tags()", REST, node_id="m_tags", source_location="L10"),
    make_node(REPO, ".summary()", REST, node_id="m_summary", source_location="L40"),
    make_node(REPO, ".delete()", REST, node_id="m_delete", source_location="L50"),
    make_node(REPO, "ROLE_HR", "", node_id="g_hr", _kind="access_guard", guard_literal=True),
    make_node(REPO, "ROLE_LEAD", "", node_id="g_lead", _kind="access_guard", guard_literal=True),
    make_node(REPO, "canEdit", REST, node_id="k_edit", _kind="access_guard", guard_literal=False),
    # Shared node first seen on an #[IsGranted]; the edge says access_control.
    make_node(REPO, "roles: [ROLE_USER]", "", node_id="g_ac", _kind="access_guard"),
    _route("r_teams", "/teams/time-tracker", "teams-dashboard", ["GET"], FRONT, 20),
    _route("r_members", "/teams/{team}/members", "teams-members", ["GET"], FRONT, 30),
    _route("r_tags", "/api/users/{me}/tags", "api-tags", ["GET"], REST, 10),
    _route("r_summary", "/api/users/{me}/tags/summary", "api-tags-summary", ["GET"], REST, 40),
    _route("r_delete", "/api/users/{me}/tags/{tag}", "api-tags-delete", ["DELETE"], REST, 50),
]
LINKS = [
    _link("r_teams", "m_teams", "handled_by"),
    _link("r_members", "m_members", "handled_by"),
    _link("r_tags", "m_tags", "handled_by"),
    _link("r_summary", "m_summary", "handled_by"),
    _link("r_delete", "m_delete", "handled_by"),
    _link("c_team", "m_teams", "method"),
    _link("c_team", "m_members", "method"),
    _link("c_tags", "m_tags", "method"),
    _link("c_tags", "m_summary", "method"),
    _link("c_tags", "m_delete", "method"),
    _link("c_team", "g_lead", "guarded_by", guard_scope="class", source_file=FRONT),
    _link("m_delete", "g_hr", "guarded_by", guard_scope="method", source_file=REST),
    _link(
        "m_delete",
        "k_edit",
        "checks_access",
        conditional=None,
        check_call="canEdit",
        guard_attribute="canEdit",
        source_file=REST,
    ),
    _link("r_summary", "g_ac", "guarded_by", guard_attribute="access_control", source_file=REST),
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


def _paths(payload):
    return [route["path"] for route in payload["routes"]]


def test_route_by_full_path(tmp_path):
    payload = _call(_server(tmp_path), "route_handler", route="/teams/time-tracker", repo=REPO)
    assert payload["resolved"] is True and payload["match"] == routes.MATCH_PATH
    handler = payload["routes"][0]["handlers"][0]
    assert handler["label"] == ".teams()" and handler["class"] == "TeamController"
    assert payload["routes"][0]["http_methods"] == ["GET"]


def test_route_by_name(tmp_path):
    payload = _call(_server(tmp_path), "route_handler", route="api-tags-delete", repo=REPO)
    assert payload["match"] == routes.MATCH_NAME
    assert _paths(payload) == ["/api/users/{me}/tags/{tag}"]


def test_route_placeholder_names_need_not_match(tmp_path):
    payload = _call(_server(tmp_path), "route_handler", route="/teams/{id}/members", repo=REPO)
    assert payload["match"] == routes.MATCH_PATTERN
    assert _paths(payload) == ["/teams/{team}/members"]


def test_query_placeholder_does_not_match_a_literal_segment(tmp_path):
    payload = _call(_server(tmp_path), "route_handler", route="/api/users/{u}/tags/{t}", repo=REPO)
    assert _paths(payload) == ["/api/users/{me}/tags/{tag}"]


def test_concrete_value_matches_a_route_placeholder(tmp_path):
    payload = _call(_server(tmp_path), "route_handler", route="/teams/42/members", repo=REPO)
    assert _paths(payload) == ["/teams/{team}/members"]


def test_route_by_prefix_returns_every_route_under_it(tmp_path):
    payload = _call(_server(tmp_path), "route_handler", route="/api/users/{x}", repo=REPO)
    assert payload["match"] == routes.MATCH_PREFIX
    assert sorted(_paths(payload)) == [
        "/api/users/{me}/tags",
        "/api/users/{me}/tags/summary",
        "/api/users/{me}/tags/{tag}",
    ]


def test_unindexed_child_route_falls_back_to_longest_ancestor(tmp_path):
    """The stacked-route case: `/teams/time-tracker/date/{date}` is not in
    the index, its parent route is."""
    payload = _call(
        _server(tmp_path), "route_handler", route="/teams/time-tracker/date/{date}", repo=REPO
    )
    assert payload["resolved"] is True and payload["match"] == routes.MATCH_ANCESTOR
    assert _paths(payload) == ["/teams/time-tracker"]
    assert payload["notes"][0] == routes.NOTE_ANCESTOR


def test_exact_route_wins_over_its_ancestor_prefix(tmp_path):
    """Once the stacked route is indexed, the exact path resolves to it and
    the parent route is not returned."""
    child = _route("r_date", "/teams/time-tracker/date/{date}", "teams-date", ["GET"], FRONT, 25)
    nodes = [*NODES, child, make_node(REPO, ".date()", FRONT, node_id="m_date")]
    links = [*LINKS, _link("r_date", "m_date", "handled_by"), _link("c_team", "m_date", "method")]
    payload = _call(
        _server(tmp_path, nodes=nodes, links=links),
        "route_handler",
        route="/teams/time-tracker/date/{date}",
        repo=REPO,
    )
    assert payload["match"] == routes.MATCH_PATH
    assert _paths(payload) == ["/teams/time-tracker/date/{date}"]
    assert payload["routes"][0]["handlers"][0]["label"] == ".date()"
    assert routes.NOTE_ANCESTOR not in payload["notes"]


def test_no_match_returns_closest_routes(tmp_path):
    payload = _call(_server(tmp_path), "route_handler", route="api-tag-delet", repo=REPO)
    assert payload["resolved"] is False
    assert "node_not_found" in payload["degraded"]
    assert "api-tags-delete" in payload["suggestions"]


def test_unregistered_repo_fails_closed(tmp_path):
    payload = _call(_server(tmp_path), "route_handler", route="/teams", repo="nope")
    assert payload["resolved"] is False and payload["degraded"] == ["repo_not_registered"]


def test_outline_by_file_lists_routes_in_source_order_with_guards(tmp_path):
    payload = _call(_server(tmp_path), "controller_outline", target=REST, repo=REPO)
    assert payload["resolved"] is True and payload["match"] == "file"
    (controller,) = payload["controllers"]
    assert [r["path"] for r in controller["routes"]] == [
        "/api/users/{me}/tags",
        "/api/users/{me}/tags/summary",
        "/api/users/{me}/tags/{tag}",
    ]
    delete = controller["routes"][2]["handlers"][0]
    assert [g["guard"] for g in delete["guards"]] == ["ROLE_HR"]
    assert delete["access_checks"] == [
        {
            "check": "canEdit",
            "call": "canEdit",
            "kind": "pattern",
            "conditional": None,
            "citation": f"[{REPO}:{REST}:?]",
        }
    ]
    assert controller["routes"][1]["guards"][0]["attribute"] == "access_control"
    assert payload["coverage"] == {
        "routes": 3,
        "guarded_or_framework_checked": 2,
        "pattern_check_only": 0,
        "no_indexed_check": 1,
    }
    assert routes.NOTE_NOT_MODELLED in payload["notes"]


def test_pattern_only_check_is_not_counted_as_checked(tmp_path):
    """`canEdit()` is inferred to be an access check from its name only, so a
    route whose sole check is that call lands in its own bucket. A framework
    `isGranted` call (kind derived from `check_call` when the edge has no
    `check_kind`) counts as checked."""
    links = [
        link
        for link in LINKS
        if not (link["relation"] == "guarded_by" and link["source"] in ("m_delete", "r_summary"))
    ]
    links.append(_link("m_tags", "g_hr", "checks_access", check_call="isGranted", source_file=REST))
    payload = _call(_server(tmp_path, links=links), "controller_outline", target=REST, repo=REPO)
    kinds = {
        route["path"]: [c["kind"] for h in route["handlers"] for c in h["access_checks"]]
        for route in payload["controllers"][0]["routes"]
    }
    assert kinds == {
        "/api/users/{me}/tags": ["framework"],
        "/api/users/{me}/tags/summary": [],
        "/api/users/{me}/tags/{tag}": ["pattern"],
    }
    assert payload["coverage"] == {
        "routes": 3,
        "guarded_or_framework_checked": 1,
        "pattern_check_only": 1,
        "no_indexed_check": 1,
    }


def test_class_level_guard_applies_to_every_action(tmp_path):
    payload = _call(_server(tmp_path), "controller_outline", target="TeamController", repo=REPO)
    assert payload["match"] == "class"
    (controller,) = payload["controllers"]
    assert controller["classes"] == [
        {"label": "TeamController", "guards": controller["routes"][0]["handlers"][0]["guards"]}
    ]
    for route in controller["routes"]:
        assert [(g["guard"], g["scope"]) for g in route["handlers"][0]["guards"]] == [
            ("ROLE_LEAD", "class")
        ]


def test_outline_resolves_fqcn_suffix_directory_and_absolute_path(tmp_path):
    server = _server(tmp_path)
    fqcn = _call(
        server, "controller_outline", target="App\\Controller\\Rest\\TagsController", repo=REPO
    )
    assert [c["source_file"] for c in fqcn["controllers"]] == [REST]
    suffix = _call(server, "controller_outline", target="Rest/TagsController.php", repo=REPO)
    assert suffix["match"] == "file_suffix"
    directory = _call(server, "controller_outline", target="src/Controller", repo=REPO)
    assert directory["match"] == "directory" and directory["route_count"] == 5
    absolute = _call(server, "controller_outline", target=str(tmp_path / "hub" / REST), repo=REPO)
    assert absolute["match"] == "file"


def test_outline_miss_suggests_controller_files(tmp_path):
    payload = _call(_server(tmp_path), "controller_outline", target="TagController.php", repo=REPO)
    assert payload["resolved"] is False
    assert REST in payload["suggestions"]


def test_missing_relations_are_reported_as_unknown_not_absent(tmp_path):
    links = [link for link in LINKS if link["relation"] not in ("guarded_by", "checks_access")]
    payload = _call(_server(tmp_path, links=links), "controller_outline", target=REST, repo=REPO)
    assert {"guards_not_indexed", "access_checks_not_indexed"} <= set(payload["degraded"])
    assert routes.NOTE_GUARDS_NOT_INDEXED in payload["notes"]
    assert routes.NOTE_CHECKS_NOT_INDEXED in payload["notes"]
