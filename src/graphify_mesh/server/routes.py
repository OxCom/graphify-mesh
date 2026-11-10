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
"""

from __future__ import annotations

import difflib
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

    def as_dict(self) -> dict:
        return {
            "resolved": self.resolved,
            "repo": self.repo,
            "query": self.query,
            "match": self.match,
            "generation_id": self.generation_id,
            "routes": self.routes,
            "count": len(self.routes),
            "coverage": coverage(self.routes),
            "truncated": self.truncated,
            "complete": self.resolved and not self.truncated,
            "suggestions": self.suggestions,
            "notes": self.notes,
            "degraded": self.degraded,
        }


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

    def as_dict(self) -> dict:
        return {
            "resolved": self.resolved,
            "repo": self.repo,
            "target": self.target,
            "match": self.match,
            "generation_id": self.generation_id,
            "controllers": self.controllers,
            "route_count": self.route_count,
            "coverage": coverage([card for c in self.controllers for card in c["routes"]]),
            "truncated": self.truncated,
            "complete": self.resolved and not self.truncated,
            "suggestions": self.suggestions,
            "notes": self.notes,
            "degraded": self.degraded,
        }


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


def _card(node_id: str, repo: str, generation: Generation) -> dict:
    node = generation.node_by_id.get(node_id, {})
    source_file = node.get("source_file") or ""
    return {
        "key": generation.key_by_node_id.get(node_id),
        "label": node.get("label", ""),
        "source_file": source_file,
        "citation": citation(repo, source_file, node),
    }


def _guards(owner_id: str, scope: str, repo: str, generation: Generation) -> list[dict]:
    """`guarded_by` targets of `owner_id`: a handler method (#[IsGranted] on
    the method), its class (#[IsGranted] on the class), or a route node
    (a security.yaml access_control rule proved to match the route). The
    mechanism is read off the edge first: shared literal guard nodes keep the
    attribute of whichever declaration was seen first."""
    guards = []
    for guard_id, link in _out_edges(owner_id, REL_GUARDED_BY, generation):
        node = generation.node_by_id.get(guard_id, {})
        guards.append(
            {
                "guard": node.get("label", ""),
                "scope": link.get("guard_scope") or scope,
                "attribute": link.get("guard_attribute") or node.get("guard_attribute"),
                "literal": node.get("guard_literal"),
                "subject": node.get("guard_subject") or "",
                "citation": citation(repo, link.get("source_file") or "", link),
            }
        )
    return guards


def _access_checks(owner_id: str, repo: str, generation: Generation) -> list[dict]:
    """`checks_access` targets of a handler: direct `$this->isGranted()`,
    `denyAccessUnlessGranted()` or a configured `can*`-style call in its body.
    `conditional` is true for a bool check, false for a throwing one, null when
    the callee decides. `kind` is `framework` for the Symfony calls and
    `pattern` for a name-matched helper, whose effect the index cannot see."""
    checks = []
    for target_id, link in _out_edges(owner_id, REL_CHECKS_ACCESS, generation):
        node = generation.node_by_id.get(target_id, {})
        call = link.get("check_call")
        kind = link.get("check_kind") or (
            CHECK_KIND_FRAMEWORK if call in FRAMEWORK_CHECK_CALLS else CHECK_KIND_PATTERN
        )
        checks.append(
            {
                "check": node.get("label", ""),
                "call": call,
                "kind": kind,
                "conditional": link.get("conditional"),
                "citation": citation(repo, link.get("source_file") or "", link),
            }
        )
    return checks


def _route_card(route_id: str, repo: str, generation: Generation) -> dict:
    node = generation.node_by_id[route_id]
    card = _card(route_id, repo, generation)
    card.pop("label")
    card = {
        "path": route_path(node),
        "route_name": node.get("route_name"),
        "http_methods": list(node.get("http_methods") or []),
        "route_complete": node.get("route_complete"),
        **card,
        "guards": _guards(route_id, "route", repo, generation),
        "handlers": [],
    }
    for handler_id, _link in _out_edges(route_id, REL_HANDLED_BY, generation):
        handler = _card(handler_id, repo, generation)
        guards = _guards(handler_id, "method", repo, generation)
        class_id = _owner_class(handler_id, generation)
        handler["class"] = None
        if class_id is not None:
            handler["class"] = generation.node_by_id.get(class_id, {}).get("label")
            guards += _guards(class_id, "class", repo, generation)
        handler["guards"] = guards
        handler["access_checks"] = _access_checks(handler_id, repo, generation)
        card["handlers"].append(handler)
    return card


def _base_notes(generation: Generation) -> tuple[list[str], list[str]]:
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
    return notes, degraded


def coverage(route_cards: list[dict]) -> dict:
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


# --- route_handler -------------------------------------------------------


def _segments(path: str) -> list[str]:
    return [segment for segment in path.strip().split("/") if segment]


def _is_param(segment: str) -> bool:
    return segment.startswith("{") and segment.endswith("}")


def _segments_match(query: list[str], route: list[str]) -> bool:
    """Same length, every segment equal or a route `{placeholder}`, which
    matches any query segment (its name is ignored, so `{date}` matches
    `{day}` and a concrete `2026-01-05`). A query placeholder matches only a
    route placeholder, never a literal segment like `summary`."""
    if len(query) != len(route):
        return False
    return all(q == r or _is_param(r) for q, r in zip(query, route, strict=True))


def resolve_routes(
    arg: str, ids: list[str], generation: Generation
) -> tuple[str | None, list[str]]:
    """(match kind, route ids) for the first tier that matches: full path, route
    name, path with placeholders as wildcards, routes under the path as a
    prefix, then the longest indexed routes that are a prefix of the path."""
    nodes = {nid: generation.node_by_id[nid] for nid in ids}
    wanted = arg.strip()
    trimmed = "/" + "/".join(_segments(wanted)) if wanted.startswith("/") else wanted
    exact = [
        nid for nid, node in nodes.items() if route_path(node).rstrip("/") == trimmed.rstrip("/")
    ]
    if exact and trimmed:
        return MATCH_PATH, exact
    named = [nid for nid, node in nodes.items() if node.get("route_name") == wanted]
    if named:
        return MATCH_NAME, named
    query = _segments(wanted)
    if not query or not wanted.startswith("/"):
        return None, []
    pattern = [
        nid for nid, node in nodes.items() if _segments_match(query, _segments(route_path(node)))
    ]
    if pattern:
        return MATCH_PATTERN, pattern
    under = [
        nid
        for nid, node in nodes.items()
        if len(_segments(route_path(node))) > len(query)
        and _segments_match(query, _segments(route_path(node))[: len(query)])
    ]
    if under:
        return MATCH_PREFIX, under
    best = 0
    ancestors: list[str] = []
    for nid, node in nodes.items():
        segments = _segments(route_path(node))
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


def route_handler(arg: str, repo: str, generation: Generation) -> RouteResult:
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
    match, matched = resolve_routes(arg, ids, generation)
    notes, degraded = _base_notes(generation)
    result.notes, result.degraded = notes, degraded
    if not matched:
        result.degraded = sorted({*degraded, "node_not_found"})
        result.suggestions = _suggestions(arg, ids, generation)
        return result
    result.resolved = True
    result.match = match
    if match == MATCH_ANCESTOR:
        result.notes.insert(0, NOTE_ANCESTOR)
        result.suggestions = _suggestions(arg, ids, generation)
    matched = sorted(matched, key=lambda nid: _route_sort_key(nid, generation))
    if len(matched) > MAX_ROUTES:
        matched, result.truncated = matched[:MAX_ROUTES], True
    result.routes = [_route_card(nid, repo, generation) for nid in matched]
    return result


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
    directory, or a class name (short, FQCN, or leading backslash)."""
    files = sorted({str(generation.node_by_id[nid].get("source_file") or "") for nid in ids} - {""})
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
    match, files = resolve_controller_files(target, ids, repo, generation, repo_root)
    notes, degraded = _base_notes(generation)
    result.notes, result.degraded = notes, degraded
    if not files:
        result.degraded = sorted({*degraded, "node_not_found"})
        all_files = sorted(
            {str(generation.node_by_id[nid].get("source_file") or "") for nid in ids}
        )
        name = target.strip().rstrip("/").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        by_name = {path.rsplit("/", 1)[-1]: path for path in all_files}
        close = difflib.get_close_matches(name, list(by_name), n=MAX_SUGGESTIONS, cutoff=0.4)
        result.suggestions = [by_name[item] for item in close]
        return result
    result.resolved, result.match = True, match
    budget = MAX_ROUTES
    for path in files:
        routes = [nid for nid in ids if generation.node_by_id[nid].get("source_file") == path]
        if len(routes) > budget:
            routes, result.truncated = routes[:budget], True
        budget -= len(routes)
        cards = [_route_card(nid, repo, generation) for nid in routes]
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
        result.route_count += len(cards)
        if result.truncated:
            break
    return result
