"""`route_handler` and `controller_outline`: route resolution by full path,
route name, placeholder pattern and prefix, near-match suggestions, and the
outline's guard/check lists and coverage, through `GraphifyMeshServer.call_tool`."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
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
from graphify_mesh.server.store import Generation

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


# --- goldens --------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"
# Goldens built from real repositories stay outside the repo; point this at
# their directory to run them.
EXTERNAL_GOLDENS_ENV = "GRAPHIFY_MESH_ROUTE_GOLDENS"


def _fixture_generation(fixture: dict) -> Generation:
    manifest = fixture["manifest"]
    generation = Generation(
        generation_id=manifest["generation_id"],
        manifest=manifest,
        graph=fixture["graph"],
        overlay={"edges": []},
        lexical={},
        embeddings={},
    )
    generation.build_indexes()
    return generation


def _with_legacy_note(payload: dict) -> dict:
    return {**payload, "notes": [*payload["notes"], routes.NOTE_COVERAGE_LEGACY]}


def test_legacy_golden_pins_head_output():
    """A synthetic graph without route_schema (attribute routes, a routes.yaml
    route handled in src/C.php, a {param} route, guards, checks, a directory
    outline). The expected output was produced by routes.py at HEAD f0063f3;
    the only allowed change is the appended legacy coverage note."""
    fixture = json.loads((FIXTURES / "routes_legacy_graph.json").read_text(encoding="utf-8"))
    golden = json.loads((FIXTURES / "routes_legacy_golden.json").read_text(encoding="utf-8"))
    generation = _fixture_generation(fixture)
    repo, root = fixture["repo"], fixture["repo_root"]
    for query, expected in golden["route_handler"].items():
        payload = routes.route_handler(query, repo, generation).as_dict()
        assert "route_schema" not in payload
        assert payload == _with_legacy_note(expected), query
    for target, expected in golden["controller_outline"].items():
        payload = routes.controller_outline(target, repo, generation, repo_root=root).as_dict()
        assert payload == _with_legacy_note(expected), target


def test_v2_golden_pins_current_output():
    """A synthetic schema-2 graph (symfony attribute routes in two controllers
    of one directory, a laravel routes file with a closure, an unresolved
    route, `{id}`, `{id?}` and empty templates, every guard_kind, framework and
    pattern checks). The expected output was produced by routes.py and checked
    by hand against the Route contract v2 rules in docs/mcp-server.md."""
    fixture = json.loads((FIXTURES / "routes_v2_graph.json").read_text(encoding="utf-8"))
    golden = json.loads((FIXTURES / "routes_v2_golden.json").read_text(encoding="utf-8"))
    generation = _fixture_generation(fixture)
    repo, root = fixture["repo"], fixture["repo_root"]
    assert set(golden["route_handler"]) == set(fixture["queries"]["route_handler"])
    assert set(golden["controller_outline"]) == set(fixture["queries"]["controller_outline"])
    for query, expected in golden["route_handler"].items():
        payload = routes.route_handler(query, repo, generation, repo_root=root).as_dict()
        assert payload == expected, query
    for target, expected in golden["controller_outline"].items():
        payload = routes.controller_outline(target, repo, generation, repo_root=root).as_dict()
        assert payload == expected, target
    # No single outline target spans app/, routes/ and src/, so the whole
    # repo is read card by card.
    cards = [
        routes._route_card(nid, repo, generation, set(), root)
        for nid in routes.route_ids(repo, generation)
    ]
    assert routes.coverage(cards) == golden["whole"]


V2_COVERAGE_KEYS = {
    "routes",
    "authorization_covered",
    "authentication_only",
    "public_declared",
    "undecided",
    "undecided_reasons",
    "framework_check_only",
    "pattern_check_only",
    "no_indexed_check",
}


def test_v2_miss_reports_v2_coverage_keys():
    """A result with no routes on a repo with v2 route nodes still reports
    the v2 buckets, all zero; a repo without v2 routes keeps the legacy keys."""
    fixture = json.loads((FIXTURES / "routes_v2_graph.json").read_text(encoding="utf-8"))
    generation = _fixture_generation(fixture)
    repo, root = fixture["repo"], fixture["repo_root"]
    payloads = [
        routes.route_handler("missing.route", repo, generation, repo_root=root).as_dict(),
        routes.controller_outline("Missing.php", repo, generation, repo_root=root).as_dict(),
    ]
    for payload in payloads:
        assert payload["route_schema"] == routes.ROUTE_SCHEMA_V2
        assert set(payload["coverage"]) == V2_COVERAGE_KEYS
        assert payload["coverage"]["undecided_reasons"] == {}
        assert all(
            value == 0 for key, value in payload["coverage"].items() if key != "undecided_reasons"
        )
    legacy = json.loads((FIXTURES / "routes_legacy_graph.json").read_text(encoding="utf-8"))
    payload = routes.route_handler("missing.route", legacy["repo"], _fixture_generation(legacy))
    assert set(payload.as_dict()["coverage"]) == {
        "routes",
        "guarded_or_framework_checked",
        "pattern_check_only",
        "no_indexed_check",
    }


def test_v2_symfony_note_follows_the_routes_in_the_result():
    """The Symfony note is chosen by the providers of the routes a result
    returns, not by the repo: a laravel-only result and a miss in a mixed
    repo carry no Symfony note."""
    fixture = json.loads((FIXTURES / "routes_v2_graph.json").read_text(encoding="utf-8"))
    generation = _fixture_generation(fixture)
    repo, root = fixture["repo"], fixture["repo_root"]

    def handler(query):
        return routes.route_handler(query, repo, generation, repo_root=root).as_dict()

    def outline(target):
        return routes.controller_outline(target, repo, generation, repo_root=root).as_dict()

    assert routes.NOTE_NOT_MODELLED_SYMFONY not in handler("posts.index")["notes"]
    assert routes.NOTE_NOT_MODELLED_SYMFONY not in outline(POSTS)["notes"]
    assert routes.NOTE_NOT_MODELLED_SYMFONY not in handler("missing.route")["notes"]
    assert routes.NOTE_NOT_MODELLED_SYMFONY not in outline("Missing.php")["notes"]
    assert routes.NOTE_NOT_MODELLED_SYMFONY in handler("/admin/users")["notes"]
    assert routes.NOTE_NOT_MODELLED_SYMFONY in outline("src/Controller/Admin")["notes"]


def _external_goldens() -> list[Path]:
    directory = os.environ.get(EXTERNAL_GOLDENS_ENV)
    if not directory or not Path(directory).is_dir():
        return []
    return sorted(Path(directory).glob("*.route-golden.json"))


def test_external_route_goldens():
    """Goldens from real repositories, withheld from the repo. Each file holds
    its own repo id, root, graph, expected payloads and whole-repo coverage."""
    paths = _external_goldens()
    if not paths:
        pytest.skip(f"no *.route-golden.json under ${EXTERNAL_GOLDENS_ENV}")
    for path in paths:
        fixture = json.loads(path.read_text(encoding="utf-8"))
        generation = _fixture_generation(fixture)
        repo, root = fixture["repo"], fixture["repo_root"]
        for query, expected in fixture["outputs"]["route_handler"].items():
            payload = routes.route_handler(query, repo, generation).as_dict()
            assert payload == expected, f"{path.name}: {query}"
        for target, expected in fixture["outputs"]["controller_outline"].items():
            payload = routes.controller_outline(target, repo, generation, repo_root=root)
            assert payload.as_dict() == expected, f"{path.name}: {target}"
        whole = routes.controller_outline(
            fixture["whole"]["target"], repo, generation, repo_root=root
        ).as_dict()
        assert whole["coverage"] == fixture["whole"]["coverage"], path.name


def test_legacy_route_belongs_to_its_declaring_file(tmp_path):
    """Without route_schema a route stays with its own source_file, as at
    HEAD, even when its handler lives in another file."""
    nodes = [
        _route("r_y", "/legacy/{id}", "legacy", ["GET"], "config/routes.yaml", 3),
        make_node(REPO, ".legacy()", "src/C.php", node_id="m_y"),
        make_node(REPO, "C", "src/C.php", node_id="c_y"),
    ]
    links = [_link("r_y", "m_y", "handled_by"), _link("c_y", "m_y", "method")]
    server = _server(tmp_path, nodes=nodes, links=links)
    declared = _call(server, "controller_outline", target="config/routes.yaml", repo=REPO)
    assert declared["resolved"] is True and declared["route_count"] == 1
    handler = _call(server, "controller_outline", target="src/C.php", repo=REPO)
    assert handler["resolved"] is False


# --- route contract v2 ---------------------------------------------------------

LARA = "app.lara"
LAM = "app.lam"
ROUTES_FILE = "routes/web.php"
POSTS = "app/Http/Controllers/PostController.php"
PAGES = "app/Http/Controllers/PageController.php"
LAM_CONFIG = "module/User/config/module.config.php"
LAM_CTRL = "module/User/src/Controller/UserController.php"


def _v2_route(repo, node_id, path, template, name, methods, declared_in, line, **extra):
    fields = {
        "provider": "laravel",
        "route_template": template,
        "effective_guards": [],
        "authorization_state": "not_proven",
        "authorization_reason": "",
        **extra,
    }
    return make_node(
        repo,
        path,
        declared_in,
        node_id=node_id,
        line=line,
        _kind="http_route",
        route_schema=2,
        route_path=path,
        route_name=name,
        http_methods=methods,
        route_complete=True,
        declared_in=declared_in,
        **fields,
    )


def _v2_guard(repo, node_id, label, kind, provider="laravel"):
    return make_node(
        repo,
        label,
        ROUTES_FILE,
        node_id=node_id,
        _kind="access_guard",
        provider=provider,
        guard_kind=kind,
        guard_key=label,
        guard_literal=True,
    )


AUTHN = {"effective_guards": ["l_g_auth"]}
AUTHZ = {"effective_guards": ["l_g_can"], "authorization_state": "covered"}
PUBLIC = {"effective_guards": ["l_g_public"]}
UNDECIDED = {
    "authorization_state": "undecided",
    "authorization_reason": "access-control-undecided",
    "route_template_reason": "route-template-unparsed",
}
LAMINAS = {"provider": "laminas"}

V2_NODES = [
    make_node(LARA, "PostController", POSTS, node_id="l_c_post"),
    make_node(LARA, "PageController", PAGES, node_id="l_c_page"),
    *[
        make_node(LARA, f".{name}()", file, node_id=f"l_m_{name}", line=line)
        for name, file, line in [
            ("index", POSTS, 10),
            ("store", POSTS, 20),
            ("show", POSTS, 30),
            ("feed", POSTS, 40),
            ("admin", POSTS, 50),
            ("drafts", POSTS, 60),
            ("about", PAGES, 5),
        ]
    ],
    _v2_guard(LARA, "l_g_auth", "auth", "authentication"),
    _v2_guard(LARA, "l_g_can", "can:create,App\\Models\\Post", "authorization"),
    _v2_guard(LARA, "l_g_public", "guest", "public"),
    make_node(LARA, "Gate::authorize", POSTS, node_id="l_k_gate", _kind="access_guard"),
    make_node(LARA, "canDraft", POSTS, node_id="l_k_draft", _kind="access_guard"),
    *[
        _v2_route(LARA, f"l_r_{name}", path, template, route_name, methods, ROUTES_FILE, line, **x)
        for name, path, template, route_name, methods, line, x in [
            ("index", "/posts", "/posts", "posts.index", ["GET"], 3, AUTHN),
            ("store", "/posts", "/posts", "posts.store", ["POST"], 4, AUTHZ),
            ("show", "/posts/{post?}", "/posts/{post?}", "posts.show", ["GET"], 5, {}),
            ("feed", "/feed", "/feed", "feed", ["GET"], 6, PUBLIC),
            ("admin", "/admin/{any}", "", "admin", ["GET"], 7, UNDECIDED),
            ("drafts", "/drafts", "/drafts", "drafts", ["GET"], 8, {}),
            ("about", "/about", "/about", "about", ["GET"], 9, {}),
        ]
    ],
    make_node(LAM, "UserController", LAM_CTRL, node_id="m_c_user"),
    make_node(LAM, ".show()", LAM_CTRL, node_id="m_m_show", line=12),
    _v2_route(
        LAM, "m_r_user", "/user[/:id]", "/user/{id?}", "user", ["GET"], LAM_CONFIG, 20, **LAMINAS
    ),
]


def _route_guard(route_id: str, guard_id: str, kind: str) -> dict:
    extra = {"provider": "laravel", "guard_kind": kind, "_via": "middleware"}
    return _link(route_id, guard_id, "guarded_by", guard_scope="route", **extra)


def _v2_check(method_id: str, target_id: str, call: str, kind: str) -> dict:
    extra = {"provider": "laravel", "guard_kind": "authorization", "conditional": False}
    return _link(method_id, target_id, "checks_access", check_call=call, check_kind=kind, **extra)


V2_LINKS = [
    *[
        _link(f"l_r_{name}", f"l_m_{name}", "handled_by", provider="laravel", _via="route_file")
        for name in ("index", "store", "show", "feed", "admin", "drafts", "about")
    ],
    *[
        _link("l_c_post", f"l_m_{name}", "method")
        for name in ("index", "store", "show", "feed", "admin", "drafts")
    ],
    _link("l_c_page", "l_m_about", "method"),
    _route_guard("l_r_index", "l_g_auth", "authentication"),
    _route_guard("l_r_store", "l_g_can", "authorization"),
    _route_guard("l_r_feed", "l_g_public", "public"),
    _v2_check("l_m_show", "l_k_gate", "Gate::authorize", "framework"),
    _v2_check("l_m_drafts", "l_k_draft", "canDraft", "pattern"),
    _link("m_r_user", "m_m_show", "handled_by", provider="laminas", _via="config"),
    _link("m_c_user", "m_m_show", "method"),
]


def _mesh_server(tmp_path: Path, nodes, links) -> GraphifyMeshServer:
    registry_path = tmp_path / "bin" / "registry.json"
    write_registry(
        registry_path,
        [registry_repo(repo, tmp_path / repo) for repo in (REPO, LARA, LAM)],
    )
    config = ServerConfig.from_env(mesh_root=tmp_path, registry_path=registry_path)
    server = GraphifyMeshServer(config, cwd=tmp_path, embed_query_fn=fake_embed_query_fn())
    generation = build_generation(nodes, links)
    server._generation = lambda: generation  # type: ignore[method-assign]
    return server


def _v2_server(tmp_path: Path) -> GraphifyMeshServer:
    return _mesh_server(tmp_path, V2_NODES, V2_LINKS)


def test_v2_routes_file_route_resolves_to_its_controller_file_and_class(tmp_path):
    server = _v2_server(tmp_path)
    by_file = _call(server, "controller_outline", target=POSTS, repo=LARA)
    assert by_file["match"] == "file" and by_file["route_schema"] == 2
    (controller,) = by_file["controllers"]
    assert controller["source_file"] == POSTS
    assert [c["label"] for c in controller["classes"]] == ["PostController"]
    assert [r["route_name"] for r in controller["routes"]] == [
        "posts.index",
        "posts.store",
        "posts.show",
        "feed",
        "admin",
        "drafts",
    ]
    assert {r["declared_in"] for r in controller["routes"]} == {ROUTES_FILE}
    assert {r["source_file"] for r in controller["routes"]} == {ROUTES_FILE}
    handler = controller["routes"][0]["handlers"][0]
    assert handler["class"] == "PostController" and handler["via"] == "route_file"
    by_class = _call(server, "controller_outline", target="PostController", repo=LARA)
    assert by_class["match"] == "class"
    assert [c["source_file"] for c in by_class["controllers"]] == [POSTS]
    routes_file = _call(server, "controller_outline", target=ROUTES_FILE, repo=LARA)
    assert routes_file["resolved"] is False


def test_v2_coverage_buckets(tmp_path):
    payload = _call(_v2_server(tmp_path), "controller_outline", target=POSTS, repo=LARA)
    assert payload["coverage"] == {
        "routes": 6,
        "authorization_covered": 1,
        "authentication_only": 1,
        "public_declared": 1,
        "undecided": 1,
        "framework_check_only": 1,
        "pattern_check_only": 1,
        "no_indexed_check": 0,
        "undecided_reasons": {"access-control-undecided": 1},
    }
    cards = payload["controllers"][0]["routes"]
    states = {r["route_name"]: r["authorization_state"] for r in cards}
    assert states == {
        "posts.index": "no_proven_authorization",
        "posts.store": "covered",
        "posts.show": "no_proven_authorization",
        "feed": "no_proven_authorization",
        "admin": "undecided",
        "drafts": "no_proven_authorization",
    }
    admin = payload["controllers"][0]["routes"][4]
    assert admin["authorization_reason"] == "access-control-undecided"
    index = payload["controllers"][0]["routes"][0]
    assert index["effective_guards"] == [
        {"guard": "auth", "kind": "authentication", "key": "auth", "provider": "laravel"}
    ]
    assert index["guards"][0]["kind"] == "authentication"
    assert index["guards"][0]["via"] == "middleware"
    whole = _call(_v2_server(tmp_path), "controller_outline", target="app/Http", repo=LARA)
    assert whole["coverage"]["routes"] == 7 and whole["coverage"]["no_indexed_check"] == 1


def test_v2_wording_is_framework_neutral(tmp_path):
    payload = _call(_v2_server(tmp_path), "controller_outline", target=POSTS, repo=LARA)
    assert routes.NOTE_SOURCE_V2 in payload["notes"]
    assert routes.NOTE_COVERAGE_V2 in payload["notes"]
    assert routes.NOTE_NOT_MODELLED not in payload["notes"]
    assert routes.NOTE_NOT_MODELLED_SYMFONY not in payload["notes"]
    notes = " ".join(payload["notes"]).lower()
    for word in ("access_control", "firewall", "voter", "isgranted"):
        assert word not in notes
    text = json.dumps(payload["coverage"]) + json.dumps(payload["controllers"])
    assert "uncovered" not in text and "unprotected" not in text


def test_v2_symfony_provider_keeps_its_note(tmp_path):
    nodes = [
        {**node, "provider": "symfony"} if node.get("_kind") == "http_route" else node
        for node in V2_NODES
    ]
    server = _mesh_server(tmp_path, nodes, V2_LINKS)
    payload = _call(server, "route_handler", route="/feed", repo=LARA)
    assert routes.NOTE_NOT_MODELLED_SYMFONY in payload["notes"]


def test_v2_check_kind_comes_from_the_edge_only(tmp_path):
    """Without the schema an `isGranted` call with no `check_kind` is derived
    as framework; on v2 the producer states the kind, and a missing one is not
    upgraded from the call name."""
    links = [
        *V2_LINKS,
        _link("l_m_about", "l_k_gate", "checks_access", check_call="isGranted", source_file=PAGES),
    ]
    server = _mesh_server(tmp_path, V2_NODES, links)
    payload = _call(server, "route_handler", route="/about", repo=LARA)
    (check,) = payload["routes"][0]["handlers"][0]["access_checks"]
    assert check["kind"] == "pattern"
    assert payload["coverage"]["pattern_check_only"] == 1


def test_v2_optional_template_segment(tmp_path):
    server = _v2_server(tmp_path)
    for query in ("/user", "/user/7"):
        payload = _call(server, "route_handler", route=query, repo=LAM)
        assert payload["match"] == routes.MATCH_PATTERN, query
        assert _paths(payload) == ["/user[/:id]"], query
    exact = _call(server, "route_handler", route="/user/{id?}", repo=LAM)
    assert exact["match"] == routes.MATCH_PATH and _paths(exact) == ["/user[/:id]"]
    assert exact["routes"][0]["route_template"] == "/user/{id?}"
    outline = _call(server, "controller_outline", target="UserController", repo=LAM)
    assert [c["source_file"] for c in outline["controllers"]] == [LAM_CTRL]
    assert outline["controllers"][0]["routes"][0]["declared_in"] == LAM_CONFIG
    laravel = _call(server, "route_handler", route="/posts/7", repo=LARA)
    assert laravel["match"] == routes.MATCH_PATTERN
    assert [r["route_name"] for r in laravel["routes"]] == ["posts.show"]


def test_v2_empty_template_matches_only_the_exact_path(tmp_path):
    server = _v2_server(tmp_path)
    exact = _call(server, "route_handler", route="/admin/{any}", repo=LARA)
    assert exact["match"] == routes.MATCH_PATH
    assert exact["routes"][0]["route_template"] == ""
    assert exact["routes"][0]["route_template_reason"] == "route-template-unparsed"
    miss = _call(server, "route_handler", route="/admin/x", repo=LARA)
    assert miss["resolved"] is False
    feed = _call(server, "route_handler", route="/feed", repo=LARA)
    assert feed["routes"][0]["route_template_reason"] == ""


def test_v2_framework_check_of_unknown_kind_is_not_an_authorization_check(tmp_path):
    """A producer that cannot resolve an `isGranted` subject (an enum value)
    marks the edge `guard_kind: unknown`; the route then keeps the bucket its
    guards give it rather than `framework_check_only`."""
    links = [
        *V2_LINKS,
        _link(
            "l_m_index",
            "l_k_gate",
            "checks_access",
            check_call="isGranted",
            check_kind="framework",
            guard_kind="unknown",
        ),
    ]
    server = _mesh_server(tmp_path, V2_NODES, links)
    payload = _call(server, "route_handler", route="posts.index", repo=LARA)
    (check,) = payload["routes"][0]["handlers"][0]["access_checks"]
    assert check["kind"] == "framework" and check["guard_kind"] == "unknown"
    assert payload["coverage"]["authentication_only"] == 1
    assert payload["coverage"]["framework_check_only"] == 0


def test_mixed_generation_reads_each_repo_by_its_own_schema(tmp_path):
    """One generation holding a v2 repo and an old one: the old repo's output
    equals what an old-only generation gives."""
    old_only = _call(_server(tmp_path), "controller_outline", target=REST, repo=REPO)
    mixed_server = _mesh_server(tmp_path, [*NODES, *V2_NODES], [*LINKS, *V2_LINKS])
    mixed_old = _call(mixed_server, "controller_outline", target=REST, repo=REPO)
    assert mixed_old == old_only
    assert "route_schema" not in mixed_old
    assert mixed_old["notes"][-1] == routes.NOTE_COVERAGE_LEGACY
    mixed_new = _call(mixed_server, "controller_outline", target=POSTS, repo=LARA)
    assert mixed_new["route_schema"] == 2
    assert "authorization_covered" in mixed_new["coverage"]


def _r2(node_id: str, path: str, template: str | None = None, **extra) -> dict:
    """A v2 route in LARA named after its id, declared in the routes file."""
    route_template = path if template is None else template
    declared_in = extra.pop("declared_in", ROUTES_FILE)
    route = _v2_route(
        LARA, node_id, path, route_template, node_id, ["GET"], ROUTES_FILE, 1, **extra
    )
    route["declared_in"] = declared_in
    return route


def _handled(route_id: str, method_id: str, source_file: str = POSTS) -> list:
    """A handler method node plus the handled_by edge to it."""
    return [
        make_node(LARA, f".{method_id}()", source_file, node_id=method_id),
        _link(route_id, method_id, "handled_by", provider="laravel", _via="route_file"),
    ]


def _split(items: list) -> tuple[list, list]:
    nodes = [item for item in items if "id" in item]
    links = [item for item in items if "relation" in item]
    return nodes, links


def _bucket_of(payload: dict) -> str:
    buckets = {k: v for k, v in payload["coverage"].items() if isinstance(v, int) and k != "routes"}
    (bucket,) = [k for k, v in buckets.items() if v == 1]
    return bucket


def test_v2_wrong_typed_fields_are_treated_as_absent(tmp_path):
    bad_kind = make_node(LARA, "auth", ROUTES_FILE, node_id="g_bad", _kind="access_guard")
    bad_kind["guard_kind"] = ["authentication"]
    items = [
        bad_kind,
        _r2("r_list_state", "/a", authorization_state=["covered"]),
        _r2("r_dict_guards", "/b", effective_guards=[{"id": "g"}]),
        _r2("r_str_guards", "/c", effective_guards="abc"),
        _r2("r_list_kind", "/d", effective_guards=["g_bad"]),
        *_handled("r_list_state", "m_a"),
        *_handled("r_dict_guards", "m_b"),
        *_handled("r_str_guards", "m_c"),
        *_handled("r_list_kind", "m_d"),
        _link("m_d", "g_bad", "checks_access", check_kind="framework", guard_kind=["x"]),
    ]
    server = _mesh_server(tmp_path, *_split(items))
    state = _call(server, "route_handler", route="/a", repo=LARA)["routes"][0]
    assert state["authorization_state"] == "undecided"
    assert state["authorization_reason"] == routes.REASON_STATE_MISSING
    for path in ("/b", "/c"):
        card = _call(server, "route_handler", route=path, repo=LARA)["routes"][0]
        assert card["effective_guards"] == [], path
    payload = _call(server, "route_handler", route="/d", repo=LARA)
    (guard,) = payload["routes"][0]["effective_guards"]
    assert guard["kind"] is None
    (check,) = payload["routes"][0]["handlers"][0]["access_checks"]
    assert check["guard_kind"] is None
    assert "route_field_invalid" in payload["degraded"]


def test_v2_framework_checks_never_land_in_pattern_check_only(tmp_path):
    """`pattern_check_only` means a pattern check and nothing else. A framework
    check of kind authentication counts as authentication; one of unknown kind
    falls through to the guard buckets, and with no guard is undecided rather
    than no_indexed_check."""
    items = [
        make_node(LARA, "check", POSTS, node_id="k", _kind="access_guard"),
        _v2_guard(LARA, "g_pub", "guest", "public"),
        _r2("r_authn", "/authn"),
        _r2("r_unknown", "/unknown"),
        _r2("r_unknown_pub", "/unknown-pub", effective_guards=["g_pub"]),
        _r2("r_pattern", "/pattern"),
        *_handled("r_authn", "m_authn"),
        *_handled("r_unknown", "m_unknown"),
        *_handled("r_unknown_pub", "m_unknown_pub"),
        *_handled("r_pattern", "m_pattern"),
        _v2_check("m_authn", "k", "isGranted", "framework") | {"guard_kind": "authentication"},
        _v2_check("m_unknown", "k", "isGranted", "framework") | {"guard_kind": "unknown"},
        _v2_check("m_unknown_pub", "k", "isGranted", "framework") | {"guard_kind": "unknown"},
        _v2_check("m_pattern", "k", "canX", "pattern") | {"guard_kind": "unknown"},
    ]
    server = _mesh_server(tmp_path, *_split(items))
    expected = {
        "/authn": "authentication_only",
        "/unknown": "undecided",
        "/unknown-pub": "public_declared",
        "/pattern": "pattern_check_only",
    }
    for path, bucket in expected.items():
        payload = _call(server, "route_handler", route=path, repo=LARA)
        assert _bucket_of(payload) == bucket, path
    unknown = _call(server, "route_handler", route="/unknown", repo=LARA)
    assert unknown["coverage"]["undecided_reasons"] == {routes.REASON_CHECK_KIND_UNKNOWN: 1}
    assert routes.MARK_INCONSISTENT not in unknown["degraded"]


def test_v2_root_query_matches_an_optional_template(tmp_path):
    items = [_r2("r_slug", "/{slug?}"), *_handled("r_slug", "m_slug")]
    payload = _call(_mesh_server(tmp_path, *_split(items)), "route_handler", route="/", repo=LARA)
    assert payload["match"] == routes.MATCH_PATTERN
    assert _paths(payload) == ["/{slug?}"]


def test_v2_prefix_listing_keeps_routes_with_an_empty_template(tmp_path):
    items = [
        _r2("r_any", "/admin/{any}", template=""),
        _r2("r_users", "/admin/users"),
        *_handled("r_any", "m_any"),
        *_handled("r_users", "m_users"),
    ]
    server = _mesh_server(tmp_path, *_split(items))
    payload = _call(server, "route_handler", route="/admin", repo=LARA)
    assert payload["match"] == routes.MATCH_PREFIX
    assert sorted(_paths(payload)) == ["/admin/users", "/admin/{any}"]
    # The wildcard tier ran first and could not read /admin/{any}.
    assert payload["complete"] is False
    assert routes.MARK_TEMPLATE_MISSING in payload["degraded"]


def test_schema_is_decided_per_route_node(tmp_path):
    """One repo holding a v2 route and an unversioned one: the unversioned
    card keeps the legacy reading, including the isGranted kind fallback."""
    legacy = make_node(
        LARA,
        "/old",
        POSTS,
        node_id="r_old",
        _kind="http_route",
        route_path="/old",
        route_name="old",
        http_methods=["GET"],
    )
    items = [
        legacy,
        make_node(LARA, "check", POSTS, node_id="k", _kind="access_guard"),
        _v2_guard(LARA, "g_authz", "can:edit", "authorization"),
        _r2("r_new", "/new", **_covered_by()),
        *_handled("r_new", "m_new"),
        *_handled("r_old", "m_old"),
        _link("m_old", "k", "checks_access", check_call="isGranted"),
    ]
    server = _mesh_server(tmp_path, *_split(items))
    old = _call(server, "route_handler", route="/old", repo=LARA)
    card = old["routes"][0]
    assert "authorization_state" not in card
    assert card["handlers"][0]["access_checks"][0]["kind"] == "framework"
    assert old["coverage"]["guarded_or_framework_checked"] == 1
    outline = _call(server, "controller_outline", target=POSTS, repo=LARA)
    assert outline["coverage"]["routes"] == 2
    assert outline["coverage"]["authorization_covered"] == 1
    assert outline["coverage"]["legacy"] == {
        "routes": 1,
        "guarded_or_framework_checked": 1,
        "pattern_check_only": 0,
        "no_indexed_check": 0,
    }


COVERED = {"authorization_state": "covered"}


def test_v2_inconsistent_producer_data_is_undecided(tmp_path):
    other = make_node(LAM, "auth", LAM_CONFIG, node_id="g_other", _kind="access_guard")
    other["guard_kind"] = "authorization"
    items = [
        other,
        _v2_guard(LARA, "g_authz", "can:edit", "authorization"),
        make_node(LARA, "not a guard", POSTS, node_id="n_plain"),
        _r2("r_claims_open", "/claims-open", effective_guards=["g_authz"]),
        _r2("r_missing", "/missing", effective_guards=["no_such_node"]),
        _r2("r_other_repo", "/other-repo", effective_guards=["g_other"]),
        _r2("r_not_guard", "/not-guard", effective_guards=["n_plain"], **COVERED),
        *_handled("r_claims_open", "m_1"),
        *_handled("r_missing", "m_2"),
        *_handled("r_other_repo", "m_3"),
        *_handled("r_not_guard", "m_4"),
    ]
    server = _mesh_server(tmp_path, *_split(items))
    claims = _call(server, "route_handler", route="/claims-open", repo=LARA)
    assert claims["coverage"]["undecided_reasons"] == {routes.REASON_INCONSISTENT: 1}
    for path in ("/missing", "/other-repo", "/not-guard"):
        payload = _call(server, "route_handler", route=path, repo=LARA)
        assert payload["routes"][0]["effective_guards"] == [], path
        assert payload["coverage"]["undecided_reasons"] == {routes.REASON_GUARD_MISSING: 1}, path
        assert "effective_guard_unresolved" in payload["degraded"], path


def test_v2_route_with_handlers_in_two_files_is_counted_once(tmp_path):
    items = [
        _v2_guard(LARA, "g_authz", "can:edit", "authorization"),
        _r2("r_two", "/two", **_covered_by()),
        *_handled("r_two", "m_post", POSTS),
        *_handled("r_two", "m_page", PAGES),
    ]
    server = _mesh_server(tmp_path, *_split(items))
    payload = _call(server, "controller_outline", target="app/Http/Controllers", repo=LARA)
    assert [c["source_file"] for c in payload["controllers"]] == [PAGES, POSTS]
    assert payload["route_count"] == 1
    assert payload["coverage"]["routes"] == 1
    assert payload["coverage"]["authorization_covered"] == 1


def test_v2_declared_in_is_normalized_against_the_repo_root(tmp_path):
    root = tmp_path / LARA
    items = [
        _r2("r_inside", "/inside", declared_in=str(root / ROUTES_FILE)),
        _r2("r_outside", "/outside", declared_in="/srv/private/checkout/routes/web.php"),
        _r2("r_escape", "/escape", declared_in="../other/routes/web.php"),
        *_handled("r_inside", "m_in"),
        *_handled("r_outside", "m_out"),
        *_handled("r_escape", "m_esc"),
    ]
    generation = build_generation(*_split(items))
    outline = routes.controller_outline(POSTS, LARA, generation, repo_root=str(root)).as_dict()
    declared = {r["path"]: r["declared_in"] for r in outline["controllers"][0]["routes"]}
    assert declared == {"/inside": ROUTES_FILE, "/outside": "", "/escape": ""}
    assert "declared_in_outside_repo" in outline["degraded"]
    assert "/srv/private" not in json.dumps(outline)
    # route_handler gets no repo root from the server: an absolute path is withheld.
    handler = routes.route_handler("/inside", LARA, generation).as_dict()
    assert handler["routes"][0]["declared_in"] == ""
    assert "declared_in_outside_repo" in handler["degraded"]


def test_v2_closure_and_unresolved_routes_stay_under_their_declaring_file(tmp_path):
    items = [
        _v2_guard(LARA, "g_authz", "can:edit", "authorization"),
        _r2("r_closure", "/hello", handler_kind="closure", **_covered_by()),
        _r2("r_unresolved", "/gone", handler_kind="unresolved"),
        _r2("r_method", "/posts-x", handler_kind="method"),
        *_handled("r_method", "m_x"),
    ]
    server = _mesh_server(tmp_path, *_split(items))
    payload = _call(server, "controller_outline", target=ROUTES_FILE, repo=LARA)
    assert payload["resolved"] is True
    cards = {r["path"]: r for r in payload["controllers"][0]["routes"]}
    assert set(cards) == {"/hello", "/gone"}
    assert cards["/hello"]["handler_kind"] == "closure" and cards["/hello"]["handlers"] == []
    assert cards["/gone"]["handler_kind"] == "unresolved"
    assert payload["coverage"]["authorization_covered"] == 1
    assert payload["coverage"]["no_indexed_check"] == 1
    method = _call(server, "route_handler", route="/posts-x", repo=LARA)["routes"][0]
    assert method["handler_kind"] == "method"


def test_v2_handler_kind_disagreeing_with_edges_is_undecided(tmp_path):
    items = [
        _r2("r_no_edge", "/no-edge", handler_kind="method", authorization_state="covered"),
        _r2("r_has_edge", "/has-edge", handler_kind="closure", authorization_state="covered"),
        *_handled("r_has_edge", "m_h"),
    ]
    server = _mesh_server(tmp_path, *_split(items))
    for path in ("/no-edge", "/has-edge"):
        payload = _call(server, "route_handler", route=path, repo=LARA)
        assert payload["coverage"]["undecided_reasons"] == {routes.REASON_INCONSISTENT: 1}, path


def test_v2_handler_kind_is_derived_when_absent(tmp_path):
    items = [_r2("r_edge", "/edge"), _r2("r_none", "/none"), *_handled("r_edge", "m_e")]
    for item in items:
        item.pop("handler_kind", None)
    server = _mesh_server(tmp_path, *_split(items))
    edge = _call(server, "route_handler", route="/edge", repo=LARA)["routes"][0]
    none = _call(server, "route_handler", route="/none", repo=LARA)["routes"][0]
    assert edge["handler_kind"] == "method" and none["handler_kind"] == "unresolved"


def test_v2_inconsistent_producer_data_sets_a_degraded_marker(tmp_path):
    items = [
        _v2_guard(LARA, "g_authz", "can:edit", "authorization"),
        _r2("r_claims_open", "/claims-open", effective_guards=["g_authz"]),
        _r2("r_no_edge", "/no-edge", handler_kind="method", **COVERED),
        _r2("r_fine", "/fine", **_covered_by()),
        *_handled("r_claims_open", "m_1"),
        *_handled("r_fine", "m_2"),
    ]
    server = _mesh_server(tmp_path, *_split(items))
    for path in ("/claims-open", "/no-edge"):
        payload = _call(server, "route_handler", route=path, repo=LARA)
        assert routes.MARK_INCONSISTENT in payload["degraded"], path
    fine = _call(server, "route_handler", route="/fine", repo=LARA)
    assert routes.MARK_INCONSISTENT not in fine["degraded"]


def test_v2_out_of_enum_values_count_as_absent_and_are_flagged(tmp_path):
    items = [
        _r2("r_state", "/state", authorization_state="maybe"),
        _v2_guard(LARA, "g_authz", "can:edit", "authorization"),
        _r2("r_kind", "/kind", handler_kind="lambda", **_covered_by()),
        *_handled("r_state", "m_s"),
        *_handled("r_kind", "m_k"),
    ]
    server = _mesh_server(tmp_path, *_split(items))
    state = _call(server, "route_handler", route="/state", repo=LARA)
    assert state["routes"][0]["authorization_state"] == "undecided"
    assert "route_field_invalid" in state["degraded"]
    kind = _call(server, "route_handler", route="/kind", repo=LARA)
    assert kind["routes"][0]["handler_kind"] == "method"
    assert kind["coverage"]["authorization_covered"] == 1
    assert "route_field_invalid" in kind["degraded"]


def test_v2_out_of_enum_guard_kind_counts_as_absent_and_is_flagged(tmp_path):
    foo = _v2_check("m_f", "k", "isGranted", "framework")
    foo["guard_kind"] = "foo"
    unknown = _v2_check("m_u", "k", "isGranted", "framework")
    unknown["guard_kind"] = "unknown"
    items = [
        make_node(LARA, "check", POSTS, node_id="k", _kind="access_guard"),
        _r2("r_f", "/f"),
        _r2("r_u", "/u"),
        *_handled("r_f", "m_f"),
        *_handled("r_u", "m_u"),
        foo,
        unknown,
    ]
    server = _mesh_server(tmp_path, *_split(items))
    bad = _call(server, "route_handler", route="/f", repo=LARA)
    assert _bucket_of(bad) == "undecided"
    assert bad["coverage"]["undecided_reasons"] == {routes.REASON_INCONSISTENT: 1}
    assert routes.MARK_FIELD_INVALID in bad["degraded"]
    kept = _call(server, "route_handler", route="/u", repo=LARA)
    assert _bucket_of(kept) == "undecided"
    assert kept["coverage"]["undecided_reasons"] == {routes.REASON_CHECK_KIND_UNKNOWN: 1}
    assert routes.MARK_FIELD_INVALID not in kept["degraded"]


def test_v2_route_source_file_is_normalized_like_declared_in(tmp_path):
    """A v2 route node's own source_file (= declared_in) is emitted only as a
    repo-relative path, in the card, its citation and the outline file list."""
    root = tmp_path / LARA
    inside = _r2("r_in", "/in", handler_kind="closure", declared_in=str(root / ROUTES_FILE))
    inside["source_file"] = str(root / ROUTES_FILE)
    outside = _r2("r_out", "/out", handler_kind="closure", declared_in="/srv/private/web.php")
    outside["source_file"] = "/srv/private/web.php"
    generation = build_generation([inside, outside], [])
    outline = routes.controller_outline(ROUTES_FILE, LARA, generation, repo_root=str(root))
    payload = outline.as_dict()
    (card,) = payload["controllers"][0]["routes"]
    assert card["source_file"] == ROUTES_FILE and card["declared_in"] == ROUTES_FILE
    assert card["citation"] == f"[{LARA}:{ROUTES_FILE}:1]"
    for query in ("/in", "/out"):
        handler = routes.route_handler(query, LARA, generation).as_dict()
        assert handler["routes"][0]["source_file"] == "", query
        assert handler["routes"][0]["key"] is None, query
        assert "declared_in_outside_repo" in handler["degraded"], query
        assert "/srv/private" not in json.dumps(handler) and str(root) not in json.dumps(handler)
    whole = routes.controller_outline("/", LARA, generation, repo_root=str(root)).as_dict()
    assert "/srv/private" not in json.dumps(whole)


def test_handler_shared_with_a_v2_route_has_its_edges_read_as_v2(tmp_path):
    """An unversioned route whose handler also serves a v2 route reads that
    handler's edges as v2: a check without check_kind stays a pattern check."""
    legacy = make_node(
        LARA,
        "/old",
        POSTS,
        node_id="r_old",
        _kind="http_route",
        route_path="/old",
        route_name="old",
        http_methods=["GET"],
    )
    lone = {**legacy, "id": "r_lone", "label": "/lone", "route_path": "/lone", "route_name": "lone"}
    items = [
        legacy,
        lone,
        make_node(LARA, "check", POSTS, node_id="k", _kind="access_guard"),
        _r2("r_new", "/new", **COVERED),
        *_handled("r_new", "m_shared"),
        _link("r_old", "m_shared", "handled_by"),
        *_handled("r_lone", "m_lone"),
        _link("m_shared", "k", "checks_access", check_call="isGranted"),
        _link("m_lone", "k", "checks_access", check_call="isGranted"),
    ]
    server = _mesh_server(tmp_path, *_split(items))
    old = _call(server, "route_handler", route="/old", repo=LARA)
    assert "authorization_state" not in old["routes"][0]
    assert old["routes"][0]["handlers"][0]["access_checks"][0]["kind"] == "pattern"
    lone_card = _call(server, "route_handler", route="/lone", repo=LARA)["routes"][0]
    assert lone_card["handlers"][0]["access_checks"][0]["kind"] == "framework"


AUTHZ_EDGE = {"guard_kind": "authorization", "provider": "laravel"}


def _covered_by(guard_id: str = "g_authz") -> dict:
    return {"effective_guards": [guard_id], **COVERED}


def test_v2_http_methods_of_the_wrong_type_are_flagged(tmp_path):
    as_int, as_str = _r2("r_int", "/int"), _r2("r_str", "/str")
    as_int["http_methods"], as_str["http_methods"] = 5, "GET"
    items = [as_int, as_str, *_handled("r_int", "m_i"), *_handled("r_str", "m_s")]
    server = _mesh_server(tmp_path, *_split(items))
    for path in ("/int", "/str"):
        payload = _call(server, "route_handler", route=path, repo=LARA)
        assert payload["routes"][0]["http_methods"] == [], path
        assert routes.MARK_FIELD_INVALID in payload["degraded"], path


def test_v2_handler_guard_and_check_paths_are_repo_relative(tmp_path):
    """Every path a v2 card emits goes through repo_path: the handler card,
    guard and check citations, and the outline's controller files."""
    root = tmp_path / LARA
    inside, outside = str(root / POSTS), "/srv/private/checkout/PageController.php"
    items = [
        _v2_guard(LARA, "g_authz", "can:edit", "authorization"),
        make_node(LARA, "check", POSTS, node_id="k", _kind="access_guard"),
        make_node(LARA, "PostController", inside, node_id="c_in"),
        _r2("r_in", "/in", **_covered_by()),
        _r2("r_out", "/out", **_covered_by()),
        make_node(LARA, ".a()", inside, node_id="m_in", line=7),
        make_node(LARA, ".b()", outside, node_id="m_out", line=9),
        _link("r_in", "m_in", "handled_by", provider="laravel"),
        _link("r_out", "m_out", "handled_by", provider="laravel"),
        _link("c_in", "m_in", "method"),
        _link(
            "m_in", "g_authz", "guarded_by", guard_scope="method", source_file=inside, **AUTHZ_EDGE
        ),
        _link(
            "c_in", "g_authz", "guarded_by", guard_scope="class", source_file=outside, **AUTHZ_EDGE
        ),
        _link(
            "r_in",
            "g_authz",
            "guarded_by",
            guard_scope="route",
            source_file=str(root / ROUTES_FILE),
            **AUTHZ_EDGE,
        ),
        _v2_check("m_in", "k", "Gate::authorize", "framework") | {"source_file": inside},
    ]
    generation = build_generation(*_split(items))
    outline = routes.controller_outline("app", LARA, generation, repo_root=str(root)).as_dict()
    assert [c["source_file"] for c in outline["controllers"]] == [POSTS]
    (card,) = outline["controllers"][0]["routes"]
    (handler,) = card["handlers"]
    assert handler["source_file"] == POSTS and handler["key"] is None
    assert handler["citation"] == f"[{LARA}:{POSTS}:7]"
    citations = {g["scope"]: g["citation"] for g in handler["guards"]}
    assert citations == {"method": f"[{LARA}:{POSTS}:?]", "class": f"[{LARA}::?]"}
    assert card["guards"][0]["citation"] == f"[{LARA}:{ROUTES_FILE}:?]"
    assert handler["access_checks"][0]["citation"] == f"[{LARA}:{POSTS}:?]"
    assert routes.MARK_OUTSIDE_REPO in outline["degraded"]
    text = json.dumps(outline)
    assert str(root) not in text and "/srv/private" not in text
    for query in ("/in", "/out"):
        payload = routes.route_handler(query, LARA, generation).as_dict()
        assert payload["routes"][0]["handlers"][0]["source_file"] == "", query
        assert routes.MARK_OUTSIDE_REPO in payload["degraded"], query
        text = json.dumps(payload)
        assert str(root) not in text and "/srv/private" not in text, query


@pytest.mark.parametrize(
    "value",
    [
        r"C:\Users\me\proj\routes\web.php",
        "C:/Users/me/proj/routes/web.php",
        "c:routes/web.php",
        r"\\server\share\routes\web.php",
        r"\routes\web.php",
    ],
)
def test_repo_path_rejects_windows_host_paths(value):
    markers: set[str] = set()
    assert routes.repo_path(value, "/repo", markers) == ""
    assert markers == {routes.MARK_OUTSIDE_REPO}


def test_v2_covered_without_an_authorization_guard_is_inconsistent(tmp_path):
    items = [
        _v2_guard(LARA, "g_auth", "auth", "authentication"),
        _v2_guard(LARA, "g_pub", "guest", "public"),
        _v2_guard(LARA, "g_authz", "can:edit", "authorization"),
        _r2("r_none", "/none", **COVERED),
        _r2("r_authn", "/authn", **_covered_by("g_auth")),
        _r2("r_pub", "/pub", effective_guards=["g_pub", "g_auth"], **COVERED),
        _r2("r_ok", "/ok", effective_guards=["g_auth", "g_authz"], **COVERED),
        *_handled("r_none", "m_1"),
        *_handled("r_authn", "m_2"),
        *_handled("r_pub", "m_3"),
        *_handled("r_ok", "m_4"),
    ]
    server = _mesh_server(tmp_path, *_split(items))
    for path in ("/none", "/authn", "/pub"):
        payload = _call(server, "route_handler", route=path, repo=LARA)
        assert payload["coverage"]["undecided_reasons"] == {routes.REASON_INCONSISTENT: 1}, path
        assert routes.MARK_INCONSISTENT in payload["degraded"], path
    ok = _call(server, "route_handler", route="/ok", repo=LARA)
    assert _bucket_of(ok) == "authorization_covered"
    assert routes.MARK_INCONSISTENT not in ok["degraded"]


def test_v2_not_proven_with_an_authorization_edge_guard_is_inconsistent(tmp_path):
    absent = _r2("r_absent", "/absent")
    absent.pop("effective_guards")
    items = [
        _v2_guard(LARA, "g_authz", "can:edit", "authorization"),
        _v2_guard(LARA, "g_auth", "auth", "authentication"),
        make_node(LARA, "PostController", POSTS, node_id="c_x"),
        _r2("r_route", "/route"),
        _r2("r_method", "/method"),
        _r2("r_class", "/class"),
        absent,
        _r2("r_authn", "/authn"),
        *_handled("r_route", "m_route"),
        *_handled("r_method", "m_method"),
        *_handled("r_class", "m_class"),
        *_handled("r_absent", "m_absent"),
        *_handled("r_authn", "m_authn"),
        _link("c_x", "m_class", "method"),
        _link("r_route", "g_authz", "guarded_by", guard_scope="route", **AUTHZ_EDGE),
        _link("m_method", "g_authz", "guarded_by", guard_scope="method", **AUTHZ_EDGE),
        _link("c_x", "g_authz", "guarded_by", guard_scope="class", **AUTHZ_EDGE),
        _link("m_absent", "g_authz", "guarded_by", **AUTHZ_EDGE),
        _link("r_authn", "g_auth", "guarded_by", guard_kind="authentication"),
    ]
    server = _mesh_server(tmp_path, *_split(items))
    for path in ("/route", "/method", "/class", "/absent"):
        payload = _call(server, "route_handler", route=path, repo=LARA)
        assert payload["coverage"]["undecided_reasons"] == {routes.REASON_INCONSISTENT: 1}, path
        assert routes.MARK_INCONSISTENT in payload["degraded"], path
    authn = _call(server, "route_handler", route="/authn", repo=LARA)
    assert authn["routes"][0]["mesh_reason"] == ""


def test_v2_empty_template_skipped_by_the_wildcard_tier_is_incomplete(tmp_path):
    items = [
        _r2("r_any", "/admin/{any}", template=""),
        _r2("r_id", "/admin/{id}"),
        *_handled("r_any", "m_any"),
        *_handled("r_id", "m_id"),
    ]
    server = _mesh_server(tmp_path, *_split(items))
    pattern = _call(server, "route_handler", route="/admin/7", repo=LARA)
    assert pattern["match"] == routes.MATCH_PATTERN and _paths(pattern) == ["/admin/{id}"]
    assert pattern["complete"] is False
    assert routes.MARK_TEMPLATE_MISSING in pattern["degraded"]
    exact = _call(server, "route_handler", route="/admin/{any}", repo=LARA)
    assert exact["complete"] is True
    assert routes.MARK_TEMPLATE_MISSING not in exact["degraded"]


def test_v2_outline_keeps_the_marker_of_a_route_listed_under_no_file(tmp_path):
    root = tmp_path / LARA
    inside = _r2("r_in", "/in", handler_kind="closure")
    outside = _r2("r_out", "/out", handler_kind="closure", declared_in="/srv/private/web.php")
    outside["source_file"] = "/srv/private/web.php"
    generation = build_generation([inside, outside], [])
    outline = routes.controller_outline(ROUTES_FILE, LARA, generation, repo_root=str(root))
    payload = outline.as_dict()
    assert [r["path"] for r in payload["controllers"][0]["routes"]] == ["/in"]
    assert routes.MARK_OUTSIDE_REPO in payload["degraded"]


def test_v2_framework_check_without_guard_kind_is_inconsistent(tmp_path):
    """The v2 contract always states guard_kind on a checks_access edge; a
    framework check without it is producer-inconsistent, never no_indexed_check."""
    no_kind = _v2_check("m_a", "k", "isGranted", "framework")
    no_kind.pop("guard_kind")
    items = [
        make_node(LARA, "check", POSTS, node_id="k", _kind="access_guard"),
        _r2("r_a", "/a"),
        *_handled("r_a", "m_a"),
        no_kind,
    ]
    payload = _call(_mesh_server(tmp_path, *_split(items)), "route_handler", route="/a", repo=LARA)
    assert _bucket_of(payload) == "undecided"
    assert payload["coverage"]["undecided_reasons"] == {routes.REASON_INCONSISTENT: 1}
    assert routes.MARK_INCONSISTENT in payload["degraded"]
