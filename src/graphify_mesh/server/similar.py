"""`find_similar` tool implementation (WS5 deliverable 5).

Reads the WS4 `cross-project-overlay.json` `similar_approach` edges (the
real ANN cosine-similarity output, see `graphify_mesh.sync.overlay_similar`) for
cross-repo candidates, adds same-repo structural neighbors when
`cross_repo_only=False`, and — for a node with NEITHER an overlay edge NOR
same-repo neighbors (the documented trivial/unembedded case) — falls back to
the exact label+community match `graphify_mesh.sync.overlay_similar` already
implements at build time. This module does not reimplement that scoring
algorithm: it calls `overlay_similar.normalize_label` (the same public
normalization wrapper the build-time scorer uses) and applies the identical
match rule (same normalized label AND same `community_name`) directly over
the published merged graph, since the build-time function's own signature
expects per-repo raw graphs that are no longer available at query time (only
the merged, already-repo-attributed graph is published) — the ALGORITHM is
shared, only the input data shape differs.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from graphify_mesh.server import lexical_read, ranking
from graphify_mesh.server.retrieval import Hit, _hit_from_key
from graphify_mesh.server.store import Generation
from graphify_mesh.sync.lexical_index import normalize_alias_query
from graphify_mesh.sync.overlay_refs import LogicalRef
from graphify_mesh.sync.overlay_similar import normalize_label

FALLBACK_SCORE = 0.4
STRUCTURAL_NEIGHBOR_SCORE = 0.5
FALLBACK_PROVENANCE = "PLACEHOLDER_STRUCTURAL_MATCH"
STRUCTURAL_PROVENANCE = "STRUCTURAL_NEIGHBOR"


@dataclass
class SimilarResult:
    resolved: bool
    hits: list[Hit] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)


_T = TypeVar("_T")


def _per_generation_cache(build_fn: Callable[[Generation], _T]) -> Callable[[Generation], _T]:
    """Lazy per-generation memoization WITHOUT touching `store.py`: the
    cache lives in this module, keyed on the Generation instance's identity
    (`id()`), with a `weakref.finalize` hook evicting the entry as soon as
    the generation object is garbage-collected — so a recycled `id()` from
    a later generation can never observe a stale index, and dropped
    generations don't pin their indexes in memory. (`Generation` is a
    non-frozen `eq=True` dataclass, hence unhashable — a plain
    `WeakKeyDictionary` keyed on the instance is not an option.)"""
    import threading

    cache: dict[int, _T] = {}
    lock = threading.Lock()

    def get(generation: Generation) -> _T:
        cache_key = id(generation)
        hit = cache.get(cache_key)
        if hit is not None:
            return hit
        with lock:
            hit = cache.get(cache_key)
            if hit is not None:
                return hit
            value = build_fn(generation)
            cache[cache_key] = value
            weakref.finalize(generation, cache.pop, cache_key, None)
            return value

    return get


def _build_similar_edge_index(generation: Generation) -> dict[str, list[tuple[str, float, str]]]:
    """One pass over the overlay edge list per generation: logical-ref key
    -> [(other_key, confidence, provenance)] for `similar_approach` edges,
    indexed under BOTH endpoints (a self-edge is indexed once, matching the
    previous per-call scan's `if src == key ... elif tgt == key` behavior).
    Per-key list order is overlay edge-list order, same as the old scan."""
    index: dict[str, list[tuple[str, float, str]]] = {}
    for edge in generation.overlay.get("edges", []):
        if not isinstance(edge, dict):
            continue
        if edge.get("type") != "similar_approach":
            continue
        try:
            src_key = LogicalRef.from_dict(edge["source"]).to_key()
            tgt_key = LogicalRef.from_dict(edge["target"]).to_key()
        except (KeyError, TypeError):
            continue
        confidence = float(edge.get("confidence", 0.0))
        provenance = edge.get("provenance", "")
        index.setdefault(src_key, []).append((tgt_key, confidence, provenance))
        if tgt_key == src_key:
            continue
        index.setdefault(tgt_key, []).append((src_key, confidence, provenance))
    return index


_similar_edge_index = _per_generation_cache(_build_similar_edge_index)


def _build_label_community_index(
    generation: Generation,
) -> dict[tuple[str, str], list[tuple[str, str | None, str]]]:
    """One pass over all nodes per generation: (community_name,
    normalized_label) -> [(node_id, repo, logical-ref key)] for every node
    that has a durable key. Backs `fallback_exact_match` so each fallback
    call is a dict lookup instead of an O(all nodes) normalize-per-node
    scan."""
    index: dict[tuple[str, str], list[tuple[str, str | None, str]]] = {}
    for node_id, node in generation.node_by_id.items():
        community = node.get("community_name")
        if not community:
            continue
        node_key = generation.key_by_node_id.get(node_id)
        if not node_key:
            continue
        norm = normalize_label(node.get("label", ""))
        index.setdefault((community, norm), []).append((node_id, node.get("repo"), node_key))
    return index


_label_community_index = _per_generation_cache(_build_label_community_index)


def resolve_key(query: str, generation: Generation) -> str | None:
    """Resolve a query through version-aware exact-alias lookup."""
    norm = normalize_alias_query(query.strip()) if query else ""
    keys = sorted(key for _, key in lexical_read.alias_refs(generation.lexical, norm))
    if not keys:
        return None
    return keys[0]


def overlay_similar_pairs(key: str, generation: Generation) -> list[tuple[str, float, str]]:
    return list(_similar_edge_index(generation).get(key, []))


def same_repo_structural_neighbors(key: str, generation: Generation) -> list[str]:
    node_id = generation.node_id_by_key.get(key)
    if node_id is None:
        return []
    node = generation.node_by_id.get(node_id, {})
    repo = node.get("repo")
    neighbors = []
    for neighbor_id, edge in generation.adjacency.get(node_id, []):
        if edge.get("confidence", ranking.CONFIDENCE_EXTRACTED) == ranking.CONFIDENCE_INFERRED:
            continue
        neighbor_node = generation.node_by_id.get(neighbor_id, {})
        if neighbor_node.get("repo") != repo:
            continue
        if generation.degree(neighbor_id) > ranking.HUB_DEGREE_THRESHOLD:
            continue
        neighbor_key = generation.key_by_node_id.get(neighbor_id)
        if neighbor_key:
            neighbors.append(neighbor_key)
    return neighbors


def fallback_exact_match(
    key: str, generation: Generation, cross_repo_only: bool, top_k: int
) -> list[str]:
    """Documented fallback for trivial/unembedded nodes (deliverable 7):
    exact normalized-label + same-community_name match, mirroring
    `graphify_mesh.sync.overlay_similar`'s build-time placeholder scorer exactly
    (same normalization, same match rule), applied over the published
    merged graph instead of per-repo raw graphs."""
    node_id = generation.node_id_by_key.get(key)
    node = generation.node_by_id.get(node_id, {}) if node_id else {}
    label, community, repo = node.get("label"), node.get("community_name"), node.get("repo")
    if not label or not community:
        return []
    target_norm = normalize_label(label)

    matches = []
    for other_id, other_repo, other_key in _label_community_index(generation).get(
        (community, target_norm), []
    ):
        if other_id == node_id:
            continue
        if cross_repo_only and other_repo == repo:
            continue
        matches.append(other_key)
    return sorted(matches)[:top_k]


def find_similar(
    query: str, generation: Generation, k: int, cross_repo_only: bool = False
) -> SimilarResult:
    k = max(1, min(k, ranking.MAX_K))
    resolved_key = resolve_key(query, generation)
    if resolved_key is None:
        return SimilarResult(resolved=False, degraded=["node_not_found"])

    candidates: dict[str, tuple[float, str]] = {}
    for other_key, score, provenance in overlay_similar_pairs(resolved_key, generation):
        best = candidates.get(other_key)
        if best is None or score > best[0]:
            candidates[other_key] = (score, provenance)

    if not cross_repo_only:
        for other_key in same_repo_structural_neighbors(resolved_key, generation):
            candidates.setdefault(other_key, (STRUCTURAL_NEIGHBOR_SCORE, STRUCTURAL_PROVENANCE))

    degraded: list[str] = []
    if not candidates:
        fallback_keys = fallback_exact_match(resolved_key, generation, cross_repo_only, k)
        if fallback_keys:
            degraded.append("similarity_fallback_exact_match")
        for other_key in fallback_keys:
            candidates.setdefault(other_key, (FALLBACK_SCORE, FALLBACK_PROVENANCE))

    ranked = sorted(candidates.items(), key=lambda kv: (-kv[1][0], kv[0]))[:k]
    hits = []
    for other_key, (score, _provenance) in ranked:
        hit = _hit_from_key(other_key, generation, score, "similar")
        if hit:
            hits.append(hit)
    return SimilarResult(resolved=True, hits=hits, degraded=degraded)
