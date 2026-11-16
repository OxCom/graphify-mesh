"""`route_handler` and `controller_outline` tools: HTTP routes of one
registered repo, read off the route nodes an extractor attached to the graph.

graphify-php emits one node per route (`_kind: "http_route"`, label = path,
plus `route_path`, `route_name`, `http_methods`), a `handled_by` edge from
the route to its handler method, and a `guarded_by` edge from a handler
method (method-level `#[IsGranted]`) or its controller class (class-level,
which Symfony applies to every action of the class) to an
`_kind: "access_guard"` node. The handler's class is the source of the
`method` edge that attaches the handler to it.

Both tools answer with the complete set for what they resolved, never a
ranked top-k, and stay inside `repo`. When the generation has no
`guarded_by` relation at all, `guards` lists are empty because nothing was
indexed, not because the routes are open; `degraded` then carries
`guards_not_indexed` so a caller cannot read the empty list as "no guard".

Route contract v2 (`route_schema: 2` on every route node) is framework
neutral: the producer states `provider`, a normalized `route_template`, the
`declared_in` file, the `effective_guards` and an `authorization_state` per
route, and `guard_kind`/`check_kind` on the edges. A repo whose route nodes
carry no `route_schema` keeps the name-based reading below unchanged.
"""

from __future__ import annotations

import difflib
import posixpath
import re
from dataclasses import dataclass, field

from graphify_mesh.server import ranking
from graphify_mesh.server.citation import citation
from graphify_mesh.server.store import Generation
from graphify_mesh.sync.embedding import node_line

ROUTE_KIND = "http_route"
GUARD_KIND = "access_guard"
REL_HANDLED_BY = "handled_by"
REL_GUARDED_BY = "guarded_by"
# In-body checks (`$this->isGranted(...)`, `$this->can<X>()`) that branch on
# access rather than deny it outright; edge field `conditional`.
REL_CHECKS_ACCESS = "checks_access"
# `check_kind` on a checks_access edge; derived from `check_call` when absent.
CHECK_KIND_FRAMEWORK = "framework"
CHECK_KIND_PATTERN = "pattern"
FRAMEWORK_CHECK_CALLS = frozenset({"isGranted", "denyAccessUnlessGranted"})
REL_METHOD = "method"

# Route contract v2: `route_schema` on a route node; absent on older graphs.
ROUTE_SCHEMA_V2 = 2
PROVIDER_SYMFONY = "symfony"
GUARD_KIND_AUTHORIZATION = "authorization"
GUARD_KIND_AUTHENTICATION = "authentication"
GUARD_KIND_PUBLIC = "public"
GUARD_KINDS = frozenset(
    {
        GUARD_KIND_AUTHORIZATION,
        GUARD_KIND_AUTHENTICATION,
        GUARD_KIND_PUBLIC,
        "middleware",
        "unknown",
    }
)
AUTH_COVERED = "covered"
AUTH_NOT_PROVEN = "not_proven"
AUTH_UNDECIDED = "undecided"
# How `authorization_state` is shown: "not proven" never reads as "open".
AUTH_STATE_LABELS = {
    AUTH_COVERED: "covered",
    AUTH_NOT_PROVEN: "no_proven_authorization",
    AUTH_UNDECIDED: "undecided",
}
REASON_STATE_MISSING = "authorization-state-missing"
# Mesh reasons that put a route in `undecided` whatever its producer state.
REASON_INCONSISTENT = "producer-inconsistent"
REASON_GUARD_MISSING = "guard-missing"
# A not-proven route whose only indexed check is a framework check of a kind
# that is neither authorization nor authentication.
REASON_CHECK_KIND_UNKNOWN = "access-check-kind-unknown"
HANDLER_KINDS = frozenset({"method", "closure", "unresolved"})
HANDLER_METHOD = "method"
HANDLER_UNRESOLVED = "unresolved"
# `degraded` markers for v2 data the reader had to discard.
MARK_FIELD_INVALID = "route_field_invalid"
MARK_GUARD_UNRESOLVED = "effective_guard_unresolved"
MARK_OUTSIDE_REPO = "declared_in_outside_repo"
MARK_INCONSISTENT = "route_producer_inconsistent"
MARK_TEMPLATE_MISSING = "route_template_missing"
# Drive letter (`C:\x`, `C:/x`, `c:x`), backslash-rooted (`\x`) or UNC (`\\host\share`).
_WINDOWS_ROOTED = re.compile(r"^(?:[A-Za-z]:|\\)")

# Bound on routes reported by one call; past it `truncated` is set.
MAX_ROUTES = 500
MAX_SUGGESTIONS = 8

MATCH_PATH = "path"
MATCH_NAME = "route_name"
MATCH_PATTERN = "path_pattern"
MATCH_PREFIX = "path_prefix"
MATCH_ANCESTOR = "ancestor_prefix"

NOTE_SOURCE = (
    "Routes come from attribute extraction (#[Route]); routes declared in YAML/XML config or "
    "registered at runtime are not indexed."
)
NOTE_ANCESTOR = (
    "No indexed route matches the full path; these are the longest indexed routes whose path "
    "is a prefix of it. The exact route may be missing from the index: confirm in source."
)
NOTE_GUARDS_NOT_INDEXED = (
    "This generation carries no guarded_by edges, so guards are unknown, not absent."
)
NOTE_CHECKS_NOT_INDEXED = (
    "This generation carries no checks_access edges, so in-body access checks are unknown, "
    "not absent."
)
NOTE_NOT_MODELLED = (
    "Not modelled: access_control rules that could not be proved to match a route, "
    "firewalls, voters' own logic, and checks in called services, helpers or listeners. "
    "An empty guards/access_checks list means nothing was indexed for that route, not that "
    "it has no access control."
)
NOTE_CLASS_GUARD = (
    "A guard with scope 'class' comes from the controller class and applies to every action in it."
)
NOTE_COVERAGE_LEGACY = (
    "coverage.guarded_or_framework_checked counts a route with any guard, including "
    "authentication-only guards (IS_AUTHENTICATED_*) and access_control rules with `roles: []`, "
    "so it is not a count of routes with authorization."
)
NOTE_SOURCE_V2 = (
    "Routes come from what the extractor indexed for each provider (attributes, route files, "
    "route config); routes registered at runtime are not indexed."
)
NOTE_NOT_MODELLED_V2 = (
    "Not modelled: guards that could not be proved to apply to a route, and checks in called "
    "services, helpers, listeners or middleware bodies. An empty guards/access_checks list means "
    "nothing was indexed for that route, not that it has no access control."
)
NOTE_NOT_MODELLED_SYMFONY = (
    "Symfony: access_control rules that could not be proved to match a route, firewalls and "
    "voters' own logic are not modelled."
)
NOTE_COVERAGE_V2 = (
    "coverage reads each route's authorization_state from the extractor: "
    "no_proven_authorization means no authorization guard was proved, not that the route is open."
)


@dataclass
class RouteResult:
    resolved: bool
    repo: str
    query: str
    match: str | None = None
    generation_id: str | None = None
    routes: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    truncated: bool = False
    notes: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    route_schema: int | None = None
    # A route the matching tier could not read was left out.
    incomplete: bool = False

    def as_dict(self) -> dict:
        payload = {
            "resolved": self.resolved,
            "repo": self.repo,
            "query": self.query,
            "match": self.match,
            "generation_id": self.generation_id,
            "routes": self.routes,
            "count": len(self.routes),
            "coverage": coverage(self.routes, self.route_schema is not None),
            "truncated": self.truncated,
            "complete": self.resolved and not self.truncated and not self.incomplete,
            "suggestions": self.suggestions,
            "notes": self.notes,
            "degraded": self.degraded,
        }
        if self.route_schema is not None:
            payload["route_schema"] = self.route_schema
        return payload


@dataclass
class OutlineResult:
    resolved: bool
    repo: str
    target: str
    match: str | None = None
    generation_id: str | None = None
    controllers: list[dict] = field(default_factory=list)
    route_count: int = 0
    suggestions: list[str] = field(default_factory=list)
    truncated: bool = False
    notes: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    route_schema: int | None = None
    # One card per route, for coverage: a route whose handlers sit in two
    # files is listed under both controllers but counted once.
    unique_routes: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        payload = {
            "resolved": self.resolved,
            "repo": self.repo,
            "target": self.target,
            "match": self.match,
            "generation_id": self.generation_id,
            "controllers": self.controllers,
            "route_count": self.route_count,
            "coverage": coverage(self.unique_routes, self.route_schema is not None),
            "truncated": self.truncated,
            "complete": self.resolved and not self.truncated,
            "suggestions": self.suggestions,
            "notes": self.notes,
            "degraded": self.degraded,
        }
        if self.route_schema is not None:
            payload["route_schema"] = self.route_schema
        return payload


# --- graph reads -------------------------------------------------------


def route_ids(repo: str, generation: Generation) -> list[str]:
    ids = [
        nid
        for nid in generation.nodes_by_repo.get(repo, [])
        if generation.node_by_id[nid].get("_kind") == ROUTE_KIND
    ]
    return sorted(ids, key=lambda nid: _route_sort_key(nid, generation))


def route_path(node: dict) -> str:
    """`route_path` when the extractor composed one, else the label."""
    return str(node.get("route_path") or node.get("label") or "")


def is_v2_route(node: dict) -> bool:
    """Route contract v2 is decided per route node: an integer `route_schema`
    of 2 or more. A node without it keeps the name-based reading."""
    value = node.get("route_schema")
    return isinstance(value, int) and not isinstance(value, bool) and value >= ROUTE_SCHEMA_V2


def route_schema(ids: list[str], generation: Generation) -> int | None:
    """ROUTE_SCHEMA_V2 when any of the repo's route nodes is v2, else None.
    Only reported in the payload; every card is read by its own node."""
    if any(is_v2_route(generation.node_by_id[nid]) for nid in ids):
        return ROUTE_SCHEMA_V2
    return None


def _v2_str(container: dict, key: str, markers: set[str]) -> str | None:
    """A v2 string field; any other type counts as absent and is flagged."""
    value = container.get(key)
    if value is None or isinstance(value, str):
        return value
    markers.add(MARK_FIELD_INVALID)
    return None


def _v2_str_list(container: dict, key: str, markers: set[str]) -> list[str] | None:
    """A v2 list-of-strings field; anything else counts as absent and is flagged."""
    value = container.get(key)
    if value is None:
        return None
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    markers.add(MARK_FIELD_INVALID)
    return None


def _v2_guard_kind(container: dict, markers: set[str]) -> str | None:
    """A v2 `guard_kind`; a value outside GUARD_KINDS counts as absent and is flagged."""
    value = _v2_str(container, "guard_kind", markers)
    if value is None or value in GUARD_KINDS:
        return value
    markers.add(MARK_FIELD_INVALID)
    return None


def repo_path(value: str | None, repo_root: str | None, markers: set[str]) -> str:
    """`value` as a repo-relative path. An absolute path inside `repo_root` is
    made relative; one outside it, or with no root to check against, a Windows
    host path, and a relative path escaping the repo, become "" so no host
    path is emitted."""
    if not value:
        return ""
    path = value
    if _WINDOWS_ROOTED.match(path):
        markers.add(MARK_OUTSIDE_REPO)
        return ""
    if posixpath.isabs(path):
        root = (repo_root or "").rstrip("/")
        if not root or not path.startswith(root + "/"):
            markers.add(MARK_OUTSIDE_REPO)
            return ""
        path = path[len(root) + 1 :]
    normalized = posixpath.normpath(path)
    if normalized in (".", "..") or normalized.startswith("../"):
        markers.add(MARK_OUTSIDE_REPO)
        return ""
    return normalized


def _is_host_path(value: str) -> bool:
    """A path a node key would embed verbatim: absolute or Windows-rooted."""
    return posixpath.isabs(value) or bool(_WINDOWS_ROOTED.match(value))


def _providers(ids: list[str], generation: Generation) -> set[str]:
    providers = set()
    for nid in ids:
        node = generation.node_by_id[nid]
        if is_v2_route(node) and isinstance(node.get("provider"), str) and node["provider"]:
            providers.add(node["provider"])
    return providers


def controller_files(
    route_id: str,
    generation: Generation,
    repo_root: str | None = None,
    markers: set[str] | None = None,
) -> list[str]:
    """Files a route is listed under. A v2 route belongs to its handler
    methods' files, so a route declared in a routes file (Laravel
    `routes/web.php`, CI4 `app/Config/Routes.php`, Laminas `module.config.php`)
    belongs to its controller; a v2 route without a handler (closure,
    unresolved) stays with `declared_in`. An unversioned route stays with its
    own `source_file`, as before contract v2. `markers` collects
    `declared_in_outside_repo` for v2 paths withheld by `repo_path`."""
    marks = markers if markers is not None else set()
    node = generation.node_by_id[route_id]
    own = str(node.get("source_file") or "")
    if not is_v2_route(node):
        return [own]
    files = sorted(
        {
            repo_path(
                str(generation.node_by_id.get(handler_id, {}).get("source_file") or ""),
                repo_root,
                marks,
            )
            for handler_id, _link in _out_edges(route_id, REL_HANDLED_BY, generation)
        }
        - {""}
    )
    if files:
        return files
    return [_v2_declared_in(node, repo_root, marks)]


def _v2_declared_in(node: dict, repo_root: str | None, markers: set[str]) -> str:
    """Repo-relative file a v2 route is declared in: `declared_in`, else its
    `source_file`; "" when neither lies inside the repo."""
    declared = _v2_str(node, "declared_in", markers) or str(node.get("source_file") or "")
    return repo_path(declared, repo_root, markers)


def _handler_is_v2(handler_id: str, generation: Generation) -> bool:
    """A handler is v2 when a v2 route is `handled_by` it."""
    return any(
        link.get("relation") == REL_HANDLED_BY
        and link.get("target") == handler_id
        and is_v2_route(generation.node_by_id.get(neighbor_id, {}))
        for neighbor_id, link in generation.adjacency.get(handler_id, [])
    )


def _route_sort_key(nid: str, generation: Generation) -> tuple:
    node = generation.node_by_id[nid]
    return (
        str(node.get("source_file") or ""),
        node_line(node) or 0,
        route_path(node),
        nid,
    )


def _out_edges(node_id: str, relation: str, generation: Generation) -> list[tuple[str, dict]]:
    found = [
        (neighbor_id, link)
        for neighbor_id, link in generation.adjacency.get(node_id, [])
        if link.get("relation") == relation
        and link.get("source") == node_id
        and ranking.edge_followable(link, include_inferred=False)
    ]
    return sorted(found, key=lambda pair: pair[0])


def _owner_class(method_id: str, generation: Generation) -> str | None:
    for neighbor_id, link in generation.adjacency.get(method_id, []):
        if link.get("relation") == REL_METHOD and link.get("target") == method_id:
            return neighbor_id
    return None


def _card(
    node_id: str,
    repo: str,
    generation: Generation,
    v2: bool = False,
    repo_root: str | None = None,
    markers: set[str] | None = None,
) -> dict:
    """Node card. On v2 data the source file goes through `repo_path`, and the
    key, which embeds the raw source file, is withheld for a host path."""
    node = generation.node_by_id.get(node_id, {})
    source_file = node.get("source_file") or ""
    key = generation.key_by_node_id.get(node_id)
    if v2:
        raw = str(source_file)
        source_file = repo_path(raw, repo_root, markers if markers is not None else set())
        if _is_host_path(raw):
            key = None
    return {
        "key": key,
        "label": node.get("label", ""),
        "source_file": source_file,
        "citation": citation(repo, source_file, node),
    }


def _edge_file(link: dict, v2: bool, repo_root: str | None, markers: set[str]) -> str:
    """The file an edge cites; repo-relative through `repo_path` on v2 data."""
    if not v2:
        return link.get("source_file") or ""
    return repo_path(str(link.get("source_file") or ""), repo_root, markers)


def _guards(
    owner_id: str,
    scope: str,
    repo: str,
    generation: Generation,
    v2: bool = False,
    markers: set[str] | None = None,
    repo_root: str | None = None,
) -> list[dict]:
    """`guarded_by` targets of `owner_id`: a handler method (#[IsGranted] on
    the method), its class (#[IsGranted] on the class), or a route node
    (a security.yaml access_control rule proved to match the route). The
    mechanism is read off the edge first: shared literal guard nodes keep the
    attribute of whichever declaration was seen first. On a v2 route the
    edge also gives `guard_kind`, `provider` and `_via`."""
    marks = markers if markers is not None else set()
    guards = []
    for guard_id, link in _out_edges(owner_id, REL_GUARDED_BY, generation):
        node = generation.node_by_id.get(guard_id, {})
        guard = {
            "guard": node.get("label", ""),
            "scope": link.get("guard_scope") or scope,
            "attribute": link.get("guard_attribute") or node.get("guard_attribute"),
            "literal": node.get("guard_literal"),
            "subject": node.get("guard_subject") or "",
            "citation": citation(repo, _edge_file(link, v2, repo_root, marks), link),
        }
        if v2:
            guard["kind"] = _v2_guard_kind(link, marks) or _v2_guard_kind(node, marks)
            guard["provider"] = _v2_str(link, "provider", marks) or _v2_str(node, "provider", marks)
            guard["via"] = _v2_str(link, "_via", marks)
            guard["key"] = _v2_str(node, "guard_key", marks)
        guards.append(guard)
    return guards


def _access_checks(
    owner_id: str,
    repo: str,
    generation: Generation,
    v2: bool = False,
    markers: set[str] | None = None,
    repo_root: str | None = None,
) -> list[dict]:
    """`checks_access` targets of a handler: direct `$this->isGranted()`,
    `denyAccessUnlessGranted()` or a configured `can*`-style call in its body.
    `conditional` is true for a bool check, false for a throwing one, null when
    the callee decides. `kind` is `framework` for a framework access call and
    `pattern` for a name-matched helper, whose effect the index cannot see. On
    a v2 route `kind` comes from the edge only; without the schema a missing
    `check_kind` is derived from the Symfony call names."""
    marks = markers if markers is not None else set()
    checks = []
    for target_id, link in _out_edges(owner_id, REL_CHECKS_ACCESS, generation):
        node = generation.node_by_id.get(target_id, {})
        call = link.get("check_call")
        if v2:
            stated = _v2_str(link, "check_kind", marks)
            kind = CHECK_KIND_FRAMEWORK if stated == CHECK_KIND_FRAMEWORK else CHECK_KIND_PATTERN
        else:
            kind = link.get("check_kind") or (
                CHECK_KIND_FRAMEWORK if call in FRAMEWORK_CHECK_CALLS else CHECK_KIND_PATTERN
            )
        check = {
            "check": node.get("label", ""),
            "call": call,
            "kind": kind,
            "conditional": link.get("conditional"),
            "citation": citation(repo, _edge_file(link, v2, repo_root, marks), link),
        }
        if v2:
            check["guard_kind"] = _v2_guard_kind(link, marks)
            check["provider"] = _v2_str(link, "provider", marks)
        checks.append(check)
    return checks


@dataclass(frozen=True)
class _EdgeEvidence:
    """What a v2 route's edges hold, checked against its producer fields."""

    has_handler: bool
    # A guarded_by edge of kind authorization on the route, handler or class.
    authorization_guard: bool
    # A framework checks_access edge with no (or an invalid) guard_kind.
    framework_check_without_kind: bool


def _route_card(
    route_id: str,
    repo: str,
    generation: Generation,
    markers: set[str] | None = None,
    repo_root: str | None = None,
) -> dict:
    """One route with its guards and handlers. Each card is read by its own
    node's schema; `markers` collects `degraded` markers for discarded v2 data."""
    marks = markers if markers is not None else set()
    node = generation.node_by_id[route_id]
    v2 = is_v2_route(node)
    card = _card(route_id, repo, generation)
    card.pop("label")
    if v2:
        # Same file as declared_in, so it gets the same host-path guard.
        card["source_file"] = _v2_declared_in(node, repo_root, marks)
        card["citation"] = citation(repo, card["source_file"], node)
        # The node key embeds the raw source_file; withheld rather than rewritten.
        if _is_host_path(str(node.get("source_file") or "")):
            card["key"] = None
    if v2:
        methods = _v2_str_list(node, "http_methods", marks) or []
    else:
        methods = list(node.get("http_methods") or [])
    card = {
        "path": route_path(node),
        "route_name": node.get("route_name"),
        "http_methods": methods,
        "route_complete": node.get("route_complete"),
        **card,
        "guards": _guards(route_id, "route", repo, generation, v2, marks, repo_root),
        "handlers": [],
    }
    for handler_id, link in _out_edges(route_id, REL_HANDLED_BY, generation):
        # Edges are v2 when their route or their handler is.
        edges_v2 = v2 or _handler_is_v2(handler_id, generation)
        handler = _card(handler_id, repo, generation, edges_v2, repo_root, marks)
        guards = _guards(handler_id, "method", repo, generation, edges_v2, marks, repo_root)
        class_id = _owner_class(handler_id, generation)
        handler["class"] = None
        if class_id is not None:
            handler["class"] = generation.node_by_id.get(class_id, {}).get("label")
            guards += _guards(class_id, "class", repo, generation, edges_v2, marks, repo_root)
        handler["guards"] = guards
        handler["access_checks"] = _access_checks(
            handler_id, repo, generation, edges_v2, marks, repo_root
        )
        if v2:
            handler["via"] = _v2_str(link, "_via", marks)
            handler["provider"] = _v2_str(link, "provider", marks)
        card["handlers"].append(handler)
    if v2:
        handlers = card["handlers"]
        edge_guards = [*card["guards"], *(g for h in handlers for g in h["guards"])]
        evidence = _EdgeEvidence(
            has_handler=bool(handlers),
            authorization_guard=any(g["kind"] == GUARD_KIND_AUTHORIZATION for g in edge_guards),
            framework_check_without_kind=any(
                check["kind"] == CHECK_KIND_FRAMEWORK and check["guard_kind"] is None
                for h in handlers
                for check in h["access_checks"]
            ),
        )
        card.update(_route_v2_fields(node, generation, evidence, marks, repo_root))
    return card


def _route_v2_fields(
    node: dict,
    generation: Generation,
    evidence: _EdgeEvidence,
    markers: set[str],
    repo_root: str | None,
) -> dict:
    """Route contract v2 fields of a route node. `effective_guards` holds
    access_guard node ids, resolved here to their label and kind; the ids
    themselves are opaque and never parsed. `mesh_reason` is set when the
    producer data contradicts itself, which overrides the stated state in
    `coverage`."""
    state = _v2_str(node, "authorization_state", markers)
    reason = _v2_str(node, "authorization_reason", markers) or ""
    if state not in AUTH_STATE_LABELS:
        if state is not None:
            markers.add(MARK_FIELD_INVALID)
        state, reason = AUTH_UNDECIDED, reason or REASON_STATE_MISSING
    effective, missing = [], False
    for guard_id in _v2_str_list(node, "effective_guards", markers) or []:
        guard = generation.node_by_id.get(guard_id)
        if (
            guard is None
            or guard.get("repo") != node.get("repo")
            or guard.get("_kind") != GUARD_KIND
        ):
            missing = True
            continue
        effective.append(
            {
                "guard": guard.get("label", ""),
                "kind": _v2_guard_kind(guard, markers),
                "key": _v2_str(guard, "guard_key", markers),
                "provider": _v2_str(guard, "provider", markers),
            }
        )
    if missing:
        markers.add(MARK_GUARD_UNRESOLVED)
    has_handler = evidence.has_handler
    handler_kind = _v2_str(node, "handler_kind", markers)
    stated_kind = handler_kind in HANDLER_KINDS
    if not stated_kind:
        if handler_kind is not None:
            markers.add(MARK_FIELD_INVALID)
        handler_kind = HANDLER_METHOD if has_handler else HANDLER_UNRESOLVED
    authorized = any(guard["kind"] == GUARD_KIND_AUTHORIZATION for guard in effective)
    mesh_reason = ""
    if missing:
        mesh_reason = REASON_GUARD_MISSING
    elif (
        (stated_kind and (handler_kind == HANDLER_METHOD) != has_handler)
        # Covered needs an authorization guard among the effective guards.
        or (state == AUTH_COVERED and not authorized)
        or (state == AUTH_NOT_PROVEN and authorized)
        # An authorization guard on the route, handler or class edges that
        # the producer left out of an empty effective_guards.
        or (state == AUTH_NOT_PROVEN and not effective and evidence.authorization_guard)
        # The contract always states guard_kind on a v2 checks_access edge.
        or evidence.framework_check_without_kind
    ):
        mesh_reason = REASON_INCONSISTENT
    if mesh_reason == REASON_INCONSISTENT:
        markers.add(MARK_INCONSISTENT)
    return {
        "provider": _v2_str(node, "provider", markers),
        "handler_kind": handler_kind,
        "route_template": _v2_str(node, "route_template", markers) or "",
        # Why the template is empty or partial (`route-template-unparsed`,
        # `route-template-requirement-unread`, ...); "" when nothing applies.
        "route_template_reason": _v2_str(node, "route_template_reason", markers) or "",
        "declared_in": _v2_declared_in(node, repo_root, markers),
        "effective_guards": effective,
        "authorization_state": AUTH_STATE_LABELS[state],
        "authorization_reason": reason if state == AUTH_UNDECIDED else "",
        "mesh_reason": mesh_reason,
    }


def _base_notes(
    generation: Generation,
    has_v2: bool = False,
    has_legacy: bool = True,
    providers: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    if has_v2:
        notes = [NOTE_SOURCE_V2, NOTE_NOT_MODELLED_V2]
        if PROVIDER_SYMFONY in (providers or set()):
            notes.append(NOTE_NOT_MODELLED_SYMFONY)
    else:
        notes = [NOTE_SOURCE, NOTE_NOT_MODELLED]
    degraded: list[str] = []
    if REL_GUARDED_BY in generation.relations:
        notes.append(NOTE_CLASS_GUARD)
    else:
        notes.append(NOTE_GUARDS_NOT_INDEXED)
        degraded.append("guards_not_indexed")
    if REL_CHECKS_ACCESS not in generation.relations:
        notes.append(NOTE_CHECKS_NOT_INDEXED)
        degraded.append("access_checks_not_indexed")
    if has_v2:
        notes.append(NOTE_COVERAGE_V2)
    if has_legacy:
        notes.append(NOTE_COVERAGE_LEGACY)
    return notes, degraded


def _repo_notes(
    ids: list[str], generation: Generation, result_ids: list[str]
) -> tuple[list[str], list[str]]:
    """Schema notes follow the repo's route nodes; the provider notes follow
    the routes in the result (`result_ids`), so a miss gets none."""
    v2 = [is_v2_route(generation.node_by_id[nid]) for nid in ids]
    return _base_notes(generation, any(v2), not all(v2), _providers(result_ids, generation))


def coverage(route_cards: list[dict], v2_repo: bool = False) -> dict:
    """Coverage of unversioned cards in the three legacy buckets, of v2 cards
    in the v2 buckets (`_coverage_v2`). A mix reports the v2 buckets with the
    legacy ones nested under `legacy`; `routes` always counts every card.
    With no cards, a repo with any v2 route node gets the v2 buckets at zero."""
    v2 = [card for card in route_cards if "authorization_state" in card]
    legacy = [card for card in route_cards if "authorization_state" not in card]
    if not route_cards and v2_repo:
        return _coverage_v2([])
    if not v2:
        return _coverage_legacy(legacy)
    result = {**_coverage_v2(v2), "routes": len(route_cards)}
    if legacy:
        result["legacy"] = _coverage_legacy(legacy)
    return result


def _coverage_legacy(route_cards: list[dict]) -> dict:
    """Routes in three exclusive buckets by what the index holds: a guard (on
    the route, its handler or the handler's class) or a framework check
    (`isGranted`, `denyAccessUnlessGranted`); only pattern-matched helper calls,
    which do not count as checked because the index cannot see what they do;
    or nothing. `no_indexed_check` is NOT "unprotected": see NOTE_NOT_MODELLED."""
    checked = pattern_only = 0
    for card in route_cards:
        handlers = card["handlers"]
        kinds = {check["kind"] for h in handlers for check in h["access_checks"]}
        if card["guards"] or any(h["guards"] for h in handlers) or CHECK_KIND_FRAMEWORK in kinds:
            checked += 1
        elif kinds:
            pattern_only += 1
    return {
        "routes": len(route_cards),
        "guarded_or_framework_checked": checked,
        "pattern_check_only": pattern_only,
        "no_indexed_check": len(route_cards) - checked - pattern_only,
    }


def _coverage_bucket(card: dict) -> str:
    """First matching bucket. A `mesh_reason` puts the route in `undecided`;
    otherwise the producer's `authorization_state` decides
    `authorization_covered` and `undecided`, and a route with no proven
    authorization is split by what else the index holds, strongest first.
    A framework check of unknown kind yields to the guard buckets and is
    otherwise `undecided`."""
    if card["mesh_reason"]:
        return "undecided"
    state = card["authorization_state"]
    if state == AUTH_STATE_LABELS[AUTH_COVERED]:
        return "authorization_covered"
    if state == AUTH_STATE_LABELS[AUTH_UNDECIDED]:
        return "undecided"
    checks = [check for h in card["handlers"] for check in h["access_checks"]]
    framework = {
        check.get("guard_kind") for check in checks if check["kind"] == CHECK_KIND_FRAMEWORK
    }
    kinds = {guard["kind"] for guard in card["effective_guards"]}
    if GUARD_KIND_AUTHORIZATION in framework:
        return "framework_check_only"
    if GUARD_KIND_AUTHENTICATION in kinds or GUARD_KIND_AUTHENTICATION in framework:
        return "authentication_only"
    if GUARD_KIND_PUBLIC in kinds:
        return "public_declared"
    # An indexed framework check of another kind: never `no_indexed_check`.
    if any(check["kind"] == CHECK_KIND_FRAMEWORK for check in checks):
        return "undecided"
    if any(check["kind"] == CHECK_KIND_PATTERN for check in checks):
        return "pattern_check_only"
    return "no_indexed_check"


def _coverage_v2(route_cards: list[dict]) -> dict:
    buckets: dict[str, int] = dict.fromkeys(
        (
            "authorization_covered",
            "authentication_only",
            "public_declared",
            "undecided",
            "framework_check_only",
            "pattern_check_only",
            "no_indexed_check",
        ),
        0,
    )
    reasons: dict[str, int] = {}
    for card in route_cards:
        bucket = _coverage_bucket(card)
        buckets[bucket] += 1
        if bucket == "undecided":
            reason = card["mesh_reason"] or card["authorization_reason"]
            if not reason:
                not_proven = card["authorization_state"] == AUTH_STATE_LABELS[AUTH_NOT_PROVEN]
                reason = REASON_CHECK_KIND_UNKNOWN if not_proven else REASON_STATE_MISSING
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "routes": len(route_cards),
        **buckets,
        "undecided_reasons": dict(sorted(reasons.items())),
    }


# --- route_handler -------------------------------------------------------


def _segments(path: str) -> list[str]:
    return [segment for segment in path.strip().split("/") if segment]


def _is_param(segment: str) -> bool:
    return segment.startswith("{") and segment.endswith("}")


def _segments_match(query: list[str], route: list[str], optional: bool = False) -> bool:
    """Same length, every segment equal or a route `{placeholder}`, which
    matches any query segment (its name is ignored, so `{date}` matches
    `{day}` and a concrete `2026-01-05`). A query placeholder matches only a
    route placeholder, never a literal segment like `summary`. With
    `optional`, trailing `{name?}` route segments may be left out."""
    required = len(route)
    if optional:
        while required and _is_optional(route[required - 1]):
            required -= 1
    if not required <= len(query) <= len(route):
        return False
    return all(q == r or _is_param(r) for q, r in zip(query, route[: len(query)], strict=True))


def _is_optional(segment: str) -> bool:
    return _is_param(segment) and segment.endswith("?}")


def _match_segments(node: dict, for_prefix: bool = False) -> list[str] | None:
    """Segments a path query is matched against: the normalized
    `route_template` on a v2 route, else the route path. A v2 route whose
    template is empty takes no part in the wildcard tier (only the exact
    `route_path` compare applies), which `route_handler` reports as
    `route_template_missing`, but is listed by the prefix tiers through its
    `route_path`."""
    if not is_v2_route(node):
        return _segments(route_path(node))
    template = node.get("route_template")
    if isinstance(template, str) and template:
        return _segments(template)
    return _segments(route_path(node)) if for_prefix else None


def resolve_routes(
    arg: str, ids: list[str], generation: Generation
) -> tuple[str | None, list[str]]:
    """(match kind, route ids) for the first tier that matches: full path, route
    name, path with placeholders as wildcards, routes under the path as a
    prefix, then the longest indexed routes that are a prefix of the path.
    On a v2 route the full-path tier also accepts the template, and the
    wildcard tiers read the template, where `{name?}` may be left out."""
    nodes = {nid: generation.node_by_id[nid] for nid in ids}
    wanted = arg.strip()
    trimmed = "/" + "/".join(_segments(wanted)) if wanted.startswith("/") else wanted

    def same_path(node: dict) -> bool:
        if route_path(node).rstrip("/") == trimmed.rstrip("/"):
            return True
        template = node.get("route_template") if is_v2_route(node) else None
        if not isinstance(template, str) or not template:
            return False
        return template.rstrip("/") == trimmed.rstrip("/")

    exact = [nid for nid, node in nodes.items() if same_path(node)]
    if exact and trimmed:
        return MATCH_PATH, exact
    named = [nid for nid, node in nodes.items() if node.get("route_name") == wanted]
    if named:
        return MATCH_NAME, named
    if not wanted.startswith("/"):
        return None, []
    query = _segments(wanted)
    pattern = []
    for nid, node in nodes.items():
        segments = _match_segments(node)
        # The bare root only reaches v2 templates: `/{slug?}` matches "/".
        if segments is None or (not query and not is_v2_route(node)):
            continue
        if _segments_match(query, segments, is_v2_route(node)):
            pattern.append(nid)
    if pattern:
        return MATCH_PATTERN, pattern
    if not query:
        return None, []
    prefix_segments = {nid: _match_segments(node, for_prefix=True) for nid, node in nodes.items()}
    under = [
        nid
        for nid, segments in prefix_segments.items()
        if segments is not None
        and len(segments) > len(query)
        and _segments_match(query, segments[: len(query)])
    ]
    if under:
        return MATCH_PREFIX, under
    best = 0
    ancestors: list[str] = []
    for nid, segments in prefix_segments.items():
        if not segments or len(segments) >= len(query):
            continue
        if not _segments_match(query[: len(segments)], segments):
            continue
        if len(segments) > best:
            best, ancestors = len(segments), [nid]
        elif len(segments) == best:
            ancestors.append(nid)
    if ancestors:
        return MATCH_ANCESTOR, ancestors
    return None, []


def _suggestions(arg: str, ids: list[str], generation: Generation) -> list[str]:
    labels = sorted(
        {route_path(generation.node_by_id[nid]) for nid in ids}
        | {str(generation.node_by_id[nid].get("route_name") or "") for nid in ids}
    )
    labels = [label for label in labels if label]
    return difflib.get_close_matches(arg.strip(), labels, n=MAX_SUGGESTIONS, cutoff=0.3)


def route_handler(
    arg: str, repo: str, generation: Generation, repo_root: str | None = None
) -> RouteResult:
    result = RouteResult(
        resolved=False, repo=repo, query=arg, generation_id=generation.generation_id
    )
    if not generation.nodes_by_repo.get(repo):
        result.degraded = ["repo_not_in_current_generation"]
        return result
    ids = route_ids(repo, generation)
    if not ids:
        result.degraded = ["no_routes_indexed"]
        return result
    result.route_schema = route_schema(ids, generation)
    match, matched = resolve_routes(arg, ids, generation)
    matched = sorted(matched, key=lambda nid: _route_sort_key(nid, generation))
    if len(matched) > MAX_ROUTES:
        matched, result.truncated = matched[:MAX_ROUTES], True
    notes, degraded = _repo_notes(ids, generation, matched)
    markers: set[str] = set()
    if _template_skipped(arg, match, ids, generation):
        result.incomplete = True
        markers.add(MARK_TEMPLATE_MISSING)
    result.notes, result.degraded = notes, degraded
    if not matched:
        result.degraded = sorted({*degraded, *markers, "node_not_found"})
        result.suggestions = _suggestions(arg, ids, generation)
        return result
    result.resolved = True
    result.match = match
    if match == MATCH_ANCESTOR:
        result.notes.insert(0, NOTE_ANCESTOR)
        result.suggestions = _suggestions(arg, ids, generation)
    result.routes = [_route_card(nid, repo, generation, markers, repo_root) for nid in matched]
    if markers:
        result.degraded = sorted({*result.degraded, *markers})
    return result


def _template_skipped(arg: str, match: str | None, ids: list[str], generation: Generation) -> bool:
    """True when the wildcard tier ran (no full-path or name match for a path
    query) while a v2 route of the repo has no template it could match on."""
    if match in (MATCH_PATH, MATCH_NAME) or not arg.strip().startswith("/"):
        return False
    return any(
        is_v2_route(generation.node_by_id[nid])
        and _match_segments(generation.node_by_id[nid]) is None
        for nid in ids
    )


# --- controller_outline ----------------------------------------------------


def _relative_target(target: str, repo_root: str | None) -> str:
    value = target.strip().replace("\\\\", "\\")
    if repo_root and value.startswith(repo_root.rstrip("/") + "/"):
        value = value[len(repo_root.rstrip("/")) + 1 :]
    return value.strip("/") if "/" in value else value


def resolve_controller_files(
    target: str, ids: list[str], repo: str, generation: Generation, repo_root: str | None
) -> tuple[str | None, list[str]]:
    """(match kind, controller files) for a file path, a path suffix, a
    directory, or a class name (short, FQCN, or leading backslash). Files are
    those `controller_files` lists each route under."""
    files = sorted(
        {path for nid in ids for path in controller_files(nid, generation, repo_root)} - {""}
    )
    value = _relative_target(target, repo_root)
    if not value:
        return None, []
    if value in files:
        return "file", [value]
    suffix = [path for path in files if path.endswith("/" + value)]
    if suffix:
        return "file_suffix", suffix
    prefix = value.rstrip("/") + "/"
    under = [path for path in files if path.startswith(prefix) or ("/" + prefix) in path]
    if under:
        return "directory", under
    short = value.lstrip("\\").rsplit("\\", 1)[-1].split("::", 1)[0]
    namespace = "/".join(part for part in value.lstrip("\\").split("\\")[1:])
    classes = []
    for path in files:
        stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if stem != short:
            continue
        if namespace and not path.rsplit(".", 1)[0].endswith(namespace):
            continue
        classes.append(path)
    if classes:
        return "class", classes
    return None, []


def controller_outline(
    target: str, repo: str, generation: Generation, repo_root: str | None = None
) -> OutlineResult:
    result = OutlineResult(
        resolved=False, repo=repo, target=target, generation_id=generation.generation_id
    )
    if not generation.nodes_by_repo.get(repo):
        result.degraded = ["repo_not_in_current_generation"]
        return result
    ids = route_ids(repo, generation)
    if not ids:
        result.degraded = ["no_routes_indexed"]
        return result
    result.route_schema = route_schema(ids, generation)
    # Paths withheld while listing files would otherwise vanish without a marker.
    markers: set[str] = set()
    files_by_route = {nid: controller_files(nid, generation, repo_root, markers) for nid in ids}
    match, files = resolve_controller_files(target, ids, repo, generation, repo_root)
    if not files:
        notes, degraded = _repo_notes(ids, generation, [])
        result.notes = notes
        result.degraded = sorted({*degraded, *markers, "node_not_found"})
        all_files = sorted({path for paths in files_by_route.values() for path in paths})
        name = target.strip().rstrip("/").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        by_name = {path.rsplit("/", 1)[-1]: path for path in all_files}
        close = difflib.get_close_matches(name, list(by_name), n=MAX_SUGGESTIONS, cutoff=0.4)
        result.suggestions = [by_name[item] for item in close]
        return result
    result.resolved, result.match = True, match
    budget = MAX_ROUTES
    cards_by_route: dict[str, dict] = {}
    for path in files:
        routes = []
        for nid in ids:
            if path not in files_by_route[nid]:
                continue
            # A route already listed under another file costs no budget.
            if nid not in cards_by_route:
                if budget == 0:
                    result.truncated = True
                    continue
                budget -= 1
                cards_by_route[nid] = _route_card(nid, repo, generation, markers, repo_root)
            routes.append(nid)
        cards = [cards_by_route[nid] for nid in routes]
        classes: dict[str, dict] = {}
        for card in cards:
            for handler in card["handlers"]:
                name = handler.get("class")
                if name and name not in classes:
                    class_guards = [g for g in handler["guards"] if g["scope"] == "class"]
                    classes[name] = {"label": name, "guards": class_guards}
        result.controllers.append(
            {"source_file": path, "classes": list(classes.values()), "routes": cards}
        )
        if result.truncated:
            break
    result.unique_routes = list(cards_by_route.values())
    result.route_count = len(result.unique_routes)
    result.notes, result.degraded = _repo_notes(ids, generation, list(cards_by_route))
    if markers:
        result.degraded = sorted({*result.degraded, *markers})
    return result
