"""`neighbors` tool: exact, complete traversal over chosen relation types
inside ONE registered repo of the current generation.

`search` / `context_pack` answer "what is relevant" with a ranked top-k, so
they cannot answer "which classes extend X" or "who imports this module":
a top-k cut silently drops the tail, and the caller cannot tell a short
answer from a truncated one. This module follows EVERY indexed edge of the
requested relations out to the requested depth, and says so in the payload
(`complete`, `truncated`, `reliability`, `notes`) so an answer built on it
can state exactly what it rests on.

Traversal stays inside `repo`: the structural graph carries no cross-repo
edges (invariant 1 in `docs/architecture.md`), and a neighbor whose node
belongs to another repo is skipped rather than followed. Ordering is
deterministic — every frontier and result list is sorted by durable key,
never dict/set iteration order. Argument validation lives in `server.py`
with the other `_validate_*` helpers; this module assumes validated input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from graphify_mesh.server import ranking
from graphify_mesh.server.citation import citation
from graphify_mesh.server.store import Generation

# Depth ceiling for one call. Hierarchies and import chains deeper than this
# are rare; the ceiling bounds per-call work for a hostile `depth`.
MAX_DEPTH = 16
DEFAULT_DEPTH = 1

# Hard cap on reported nodes. Past this the result is marked `truncated` and
# `complete` is false — the payload stays bounded and never claims a
# completeness it does not have.
MAX_TRAVERSAL_NODES = 2000

# Bounds on the `relation` argument: count per call and length per name.
MAX_RELATIONS = 16
MAX_RELATION_LENGTH = 64

DIRECTION_IN = "in"
DIRECTION_OUT = "out"
DIRECTION_BOTH = "both"
DIRECTIONS = (DIRECTION_IN, DIRECTION_OUT, DIRECTION_BOTH)

RELIABILITY_EXACT = "exact"
RELIABILITY_PARTIAL = "partial"

# Relations the upstream extractor resolves only partly: method calls made
# through typed properties or dependency injection produce no `calls` edge
# (measured: 554 of 2797 PHP methods in cem.hub carry one, ~20%). A missing
# edge of these relations is therefore not evidence of a missing call.
PARTIAL_RELATIONS = frozenset({"calls", "indirect_call"})

NOTE_COMPLETE_SET = (
    "Result is the complete set of nodes reachable over the INDEXED edges of the "
    "requested relations in the published generation, not a ranked top-k."
)
NOTE_PARTIAL_RELATIONS = (
    "calls/indirect_call edges are incomplete: method calls through typed properties "
    "or dependency injection are not resolved by the extractor (measured ~20% coverage "
    "of PHP methods in cem.hub). Absence of an edge is not evidence of no call; impact "
    "answers must not rest on these relations."
)
NOTE_INFERRED = (
    "INFERRED-confidence edges were followed; they are extractor guesses, not resolved references."
)
NOTE_UNKEYED_SKIPPED = (
    "Nodes without a source_file or label have no durable key and were neither "
    "reported nor traversed through; see skipped_unkeyed."
)


@dataclass
class NeighborsResult:
    resolved: bool
    repo: str
    relations: list[str]
    direction: str
    depth: int
    include_inferred: bool
    generation_id: str | None = None
    seeds: list[dict] = field(default_factory=list)
    nodes: list[dict] = field(default_factory=list)
    truncated: bool = False
    frontier_exhausted: bool = False
    skipped_unkeyed: int = 0
    degraded: list[str] = field(default_factory=list)

    @property
    def reliability(self) -> str:
        if self.include_inferred or PARTIAL_RELATIONS.intersection(self.relations):
            return RELIABILITY_PARTIAL
        return RELIABILITY_EXACT

    @property
    def notes(self) -> list[str]:
        notes = [NOTE_COMPLETE_SET]
        if PARTIAL_RELATIONS.intersection(self.relations):
            notes.append(NOTE_PARTIAL_RELATIONS)
        if self.include_inferred:
            notes.append(NOTE_INFERRED)
        if self.skipped_unkeyed:
            notes.append(NOTE_UNKEYED_SKIPPED)
        return notes

    def as_dict(self) -> dict:
        return {
            "resolved": self.resolved,
            "repo": self.repo,
            "relations": self.relations,
            "direction": self.direction,
            "depth": self.depth,
            "generation_id": self.generation_id,
            "seeds": self.seeds,
            "nodes": self.nodes,
            "count": len(self.nodes),
            "truncated": self.truncated,
            "frontier_exhausted": self.frontier_exhausted,
            "complete": self.resolved and not self.truncated,
            "reliability": self.reliability,
            "skipped_unkeyed": self.skipped_unkeyed,
            "notes": self.notes,
            "degraded": self.degraded,
        }


def _resolve_seeds(node_arg: str, repo: str, generation: Generation) -> list[str]:
    """Seed node ids for `node_arg`, restricted to `repo`: a durable key, then
    a graph node id, then an exact label, then a case-insensitive label /
    `norm_label` match, then a bare method name matched as `.name()`. Every
    node matching at the first tier that matches anything becomes a seed."""
    for candidate in (generation.node_id_by_key.get(node_arg), node_arg):
        if candidate is None:
            continue
        node = generation.node_by_id.get(candidate)
        if node is not None and node.get("repo") == repo:
            return [candidate]
    repo_node_ids = generation.nodes_by_repo.get(repo, [])
    exact = [nid for nid in repo_node_ids if generation.node_by_id[nid].get("label") == node_arg]
    if exact:
        return exact
    folded = node_arg.lower()
    matched = [
        nid
        for nid in repo_node_ids
        if str(generation.node_by_id[nid].get("label") or "").lower() == folded
        or generation.node_by_id[nid].get("norm_label") == folded
    ]
    if matched:
        return matched
    # graphify labels methods `.name()`, so a bare method name like
    # `reminderSubmitWeek` would otherwise never resolve.
    method_label = f".{node_arg}()"
    return [nid for nid in repo_node_ids if generation.node_by_id[nid].get("label") == method_label]


def _node_card(node_id: str, key: str, repo: str, generation: Generation) -> dict:
    node = generation.node_by_id.get(node_id, {})
    source_file = node.get("source_file") or ""
    return {
        "key": key,
        "label": node.get("label", ""),
        "source_file": source_file,
        "citation": citation(repo, source_file, node),
    }


def _edge_matches(
    link: dict, current: str, direction: str, relations: frozenset[str], include_inferred: bool
) -> bool:
    if link.get("relation") not in relations:
        return False
    confidence = link.get("confidence", ranking.CONFIDENCE_EXTRACTED)
    if confidence == ranking.CONFIDENCE_INFERRED and not include_inferred:
        return False
    # Adjacency is built undirected (each link listed under both endpoints),
    # so orientation is read back off the link itself.
    if direction == DIRECTION_OUT:
        return link.get("source") == current
    if direction == DIRECTION_IN:
        return link.get("target") == current
    return True


def neighbors(
    node_arg: str,
    repo: str,
    relations: list[str],
    direction: str,
    depth: int,
    include_inferred: bool,
    generation: Generation,
) -> NeighborsResult:
    """Breadth-first traversal from every seed matching `node_arg` in `repo`.
    `relations` must already be sorted, deduplicated and known to exist in
    `generation.relations` (the server checks both)."""
    result = NeighborsResult(
        resolved=False,
        repo=repo,
        relations=relations,
        direction=direction,
        depth=depth,
        include_inferred=include_inferred,
        generation_id=generation.generation_id,
    )
    if not generation.nodes_by_repo.get(repo):
        result.degraded = ["repo_not_in_current_generation"]
        return result

    seed_ids = [
        nid
        for nid in _resolve_seeds(node_arg, repo, generation)
        if nid in generation.key_by_node_id
    ]
    if not seed_ids:
        result.degraded = ["node_not_found"]
        return result
    result.resolved = True

    key_of = generation.key_by_node_id
    seed_ids.sort(key=lambda nid: key_of[nid])
    result.seeds = [_node_card(nid, key_of[nid], repo, generation) for nid in seed_ids]

    relation_set = frozenset(relations)
    visited: set[str] = set(seed_ids)
    unkeyed: set[str] = set()
    frontier = seed_ids
    for level in range(1, depth + 1):
        # node_id -> {(from_key, relation, confidence)}: every matching edge
        # from the previous level into a newly reached node.
        reached: dict[str, set[tuple[str, str, str]]] = {}
        for current in frontier:
            for neighbor_id, link in generation.adjacency.get(current, []):
                if neighbor_id in visited:
                    continue
                if not _edge_matches(link, current, direction, relation_set, include_inferred):
                    continue
                neighbor = generation.node_by_id.get(neighbor_id)
                if neighbor is None or neighbor.get("repo") != repo:
                    continue
                if neighbor_id not in key_of:
                    unkeyed.add(neighbor_id)
                    continue
                confidence = link.get("confidence", ranking.CONFIDENCE_EXTRACTED)
                reached.setdefault(neighbor_id, set()).add(
                    (key_of[current], link["relation"], confidence)
                )
        if not reached:
            result.frontier_exhausted = True
            break
        next_ids = sorted(reached, key=lambda nid: key_of[nid])
        room = MAX_TRAVERSAL_NODES - len(result.nodes)
        if len(next_ids) > room:
            next_ids = next_ids[:room]
            result.truncated = True
        for nid in next_ids:
            card = _node_card(nid, key_of[nid], repo, generation)
            card["depth"] = level
            card["via"] = [
                {"relation": rel, "from": from_key, "confidence": conf}
                for from_key, rel, conf in sorted(reached[nid])
            ]
            result.nodes.append(card)
        visited.update(next_ids)
        if result.truncated:
            break
        frontier = next_ids
    result.skipped_unkeyed = len(unkeyed)
    return result
