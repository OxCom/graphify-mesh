"""WS5 candidate generation + fusion orchestration.

Scope filtering happens INSIDE each `*_candidates` function, before any
ranking/scoring is computed — never as a post-filter of an already-ranked
list (see `scope.py` module docstring for why that ordering matters).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from graphify_mesh.server import lexical_read, ranking
from graphify_mesh.server.store import Generation
from graphify_mesh.sync.lexical_index import normalize_alias_query, tokenize_text
from graphify_mesh.sync.vectors import RepoVectors

EmbedQueryFn = Callable[[str], list[float] | None]

# Score carried by exact-alias bypass hits. It must dominate every fused
# score (RRF sums stay below 1.0 even with all three retrievers agreeing)
# while remaining a finite float: the score is serialized with `json.dumps`,
# and `float("inf")` would emit the bare token `Infinity`, which strict JSON
# parsers reject.
EXACT_MATCH_SCORE = 1_000_000.0


@dataclass
class Hit:
    key: str
    repo: str
    source_file: str
    label: str
    node_id: str
    degree: int
    score: float
    match_type: str  # "exact" | "fused"
    deprecated: bool


@dataclass
class RankedResult:
    hits: list[Hit] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)


def exact_alias_hits(query: str, lexical: dict, repo_filter: frozenset[str] | None) -> list[str]:
    """Exact FQCN/label bypass candidates (WS5 fusion contract): looked up
    through the version-aware lexical reader, normalized with the SAME
    normalization the index was built with (`normalize_alias_query`)."""
    if not query or not query.strip():
        return []
    norm = normalize_alias_query(query.strip())
    refs = lexical_read.alias_refs(lexical, norm)
    keys = {key for repo, key in refs if repo_filter is None or repo in repo_filter}
    return sorted(keys)


def lexical_candidates(
    query: str,
    lexical: dict,
    repo_filter: frozenset[str] | None,
    depth: int = ranking.CANDIDATE_DEPTH_LEXICAL,
) -> list[str]:
    """Rank lexical candidates using version-aware postings and document metadata."""
    tokens = tokenize_text(query)
    if not tokens:
        return []
    total_docs = max(lexical_read.document_count(lexical), 1)
    field_boosts = lexical.get("field_boosts", {})

    scores: dict[str, float] = {}
    for term in tokens:
        triples = lexical_read.term_postings(lexical, term)
        df = lexical_read.term_doc_freq(lexical, term) or 1
        idf = math.log(1 + (total_docs / df))
        for repo, key, field_name in triples:
            if repo_filter is not None and repo not in repo_filter:
                continue
            if key is None:
                continue
            weight = field_boosts.get(field_name, 1.0)
            scores[key] = scores.get(key, 0.0) + weight * idf

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ranked[:depth]]


def _repo_scores(repo_vectors: RepoVectors, normalized_query: np.ndarray) -> np.ndarray:
    """Per-repo cosine scores against an already-normalized query vector.
    Mixed-dimension shard vs query carries no comparable signal (same rule
    `cosine_similarity` applied per key) — every key in that repo scores 0
    rather than being dropped from the pool, so it still tie-breaks by key
    like the old per-key comparison did."""
    if repo_vectors.dim != normalized_query.shape[0]:
        return np.zeros(len(repo_vectors), dtype=np.float32)
    return repo_vectors.normalized() @ normalized_query


def vector_candidates(
    query: str,
    embeddings: dict[str, RepoVectors],
    repo_filter: frozenset[str] | None,
    embed_query_fn: EmbedQueryFn,
    depth: int = ranking.CANDIDATE_DEPTH_VECTOR,
) -> tuple[list[str], bool]:
    """Returns (ranked keys, degraded). `degraded=True` means the vector
    retriever contributed nothing this call (no embeddings published this
    generation, or the query-embed call failed) — the caller renormalizes
    fusion over whichever OTHER retrievers succeeded and must surface
    `"embeddings_unavailable"` in the response's `degraded` field.

    Scoring is matrix cosine similarity per repo: each `RepoVectors`
    matrix's L2-normalized copy is cached on the instance
    (`RepoVectors.normalized()`, computed once per process no matter how
    many queries hit it this session) and multiplied against the
    L2-normalized query vector in one `@` call — equivalent to calling
    `cosine_similarity` per key, just batched over the whole repo instead
    of a Python-level loop.
    """
    if not embeddings:
        return [], True
    vector = embed_query_fn(query)
    if not vector:
        return [], True

    query_vec = np.asarray(vector, dtype=np.float32)
    query_norm = float(np.linalg.norm(query_vec))
    # Mirror `cosine_similarity`'s own zero-vector rule (score 0 rather
    # than raising/NaN) via a safe divisor instead of skipping the whole
    # query — a literal all-zero query vector should still tie-break by
    # key, exactly like the old per-key comparison did.
    safe_query_norm = query_norm if query_norm != 0.0 else 1.0
    normalized_query = query_vec / safe_query_norm

    # Per repo, pull only the top-`depth` scores via np.argpartition instead
    # of materializing (key, score) for EVERY embedded node across all repos
    # and full-sorting that one giant Python list per query. Boundary ties
    # are kept (everything scoring >= the depth-th largest value), so the
    # merged shortlist provably contains every key the old global sort could
    # have selected — the final sort below then applies the exact same
    # tie-break semantics (score desc, then key) on <= depth x n_repos
    # candidates (plus ties).
    scored: list[tuple[str, float]] = []
    for repo_id, repo_vectors in embeddings.items():
        if repo_filter is not None and repo_id not in repo_filter:
            continue
        if len(repo_vectors) == 0:
            continue
        scores = _repo_scores(repo_vectors, normalized_query)
        n_rows = scores.shape[0]
        if depth >= n_rows:
            top_idx = np.arange(n_rows)
        else:
            partition = np.argpartition(scores, -depth)
            threshold = scores[partition[-depth]]
            top_idx = np.nonzero(scores >= threshold)[0]
        for idx in top_idx:
            scored.append((repo_vectors.keys[int(idx)], float(scores[int(idx)])))
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in scored[:depth]], False


def structural_candidates(
    seed_keys: list[str],
    generation: Generation,
    repo_filter: frozenset[str] | None,
    include_inferred: bool = False,
    depth: int = ranking.CANDIDATE_DEPTH_STRUCTURAL,
) -> list[str]:
    """1-hop neighbors of `seed_keys` in the structural graph. Non-EXTRACTED
    (INFERRED/AMBIGUOUS) edges are excluded by default (only EXTRACTED-confidence
    edges count as traversal candidates unless the caller opts in) — baseline
    systemic failure #4 (INFERRED edge pollution)."""
    proximity: dict[str, int] = {}
    for seed_key in seed_keys:
        seed_id = generation.node_id_by_key.get(seed_key)
        if seed_id is None:
            continue
        for neighbor_id, edge in generation.adjacency.get(seed_id, []):
            if not ranking.edge_followable(edge, include_inferred):
                continue
            neighbor_key = generation.key_by_node_id.get(neighbor_id)
            if neighbor_key is None:
                continue
            neighbor_node = generation.node_by_id.get(neighbor_id, {})
            if repo_filter is not None and neighbor_node.get("repo") not in repo_filter:
                continue
            proximity[neighbor_key] = proximity.get(neighbor_key, 0) + 1

    ranked = sorted(proximity.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ranked[:depth]]


def _hit_from_key(key: str, generation: Generation, score: float, match_type: str) -> Hit | None:
    node_id = generation.node_id_by_key.get(key)
    if node_id is None:
        return None
    node = generation.node_by_id.get(node_id, {})
    source_file = node.get("source_file") or ""
    return Hit(
        key=key,
        repo=node.get("repo", ""),
        source_file=source_file,
        label=node.get("label", ""),
        node_id=node_id,
        degree=generation.degree(node_id),
        score=score,
        match_type=match_type,
        deprecated=ranking.DEPRECATED_PATH_MARKER in source_file,
    )


def rank(
    query: str,
    generation: Generation,
    repo_filter: frozenset[str] | None,
    k: int,
    embed_query_fn: EmbedQueryFn,
    include_inferred: bool = False,
) -> RankedResult:
    """Top-level WS5 search orchestration: exact-alias bypass, then
    lexical+vector+structural candidate generation, RRF fusion, hub/
    DEPRECATED penalties, MMR diversification, deterministic tie-break."""
    k = max(1, min(k, ranking.MAX_K))
    # Store-level degraded markers (reload rejections, embeddings stamp
    # problems, ...) are merged into every tool response by the server layer
    # (`GraphifyMeshServer.call_tool`), not read out of the manifest here.
    degraded: list[str] = []

    exact_keys = exact_alias_hits(query, generation.lexical, repo_filter)
    exact_hits = [
        h
        for h in (_hit_from_key(k_, generation, EXACT_MATCH_SCORE, "exact") for k_ in exact_keys)
        if h
    ]
    exact_hits.sort(key=lambda h: h.key)
    # An alias shared by many nodes must not blow past the caller's `k`;
    # truncating after the sort keeps which hits survive deterministic.
    del exact_hits[k:]
    selected_keys = {h.key for h in exact_hits}
    remaining_slots = max(0, k - len(exact_hits))

    fused_hits: list[Hit] = []
    if remaining_slots > 0:
        lexical_ranked = lexical_candidates(query, generation.lexical, repo_filter)
        vector_ranked, vec_degraded = vector_candidates(
            query, generation.embeddings, repo_filter, embed_query_fn
        )
        if vec_degraded and "embeddings_unavailable" not in degraded:
            degraded.append("embeddings_unavailable")

        seed_for_structural = (exact_keys + lexical_ranked)[:10]
        structural_ranked = structural_candidates(
            seed_for_structural, generation, repo_filter, include_inferred=include_inferred
        )

        rankings = {
            "lexical": lexical_ranked,
            "vector": vector_ranked,
            "structural": structural_ranked,
        }
        fused_scores = ranking.fuse_rankings(rankings)
        for excluded in selected_keys:
            fused_scores.pop(excluded, None)

        degree_by_key = {
            key: generation.degree(generation.node_id_by_key[key])
            for key in fused_scores
            if key in generation.node_id_by_key
        }
        path_by_key = {
            key: generation.node_by_id.get(generation.node_id_by_key.get(key, ""), {}).get(
                "source_file", ""
            )
            for key in fused_scores
            if key in generation.node_id_by_key
        }
        penalized = [
            (key, ranking.apply_penalties(key, score, degree_by_key, path_by_key))
            for key, score in fused_scores.items()
        ]
        diversified_keys = ranking.mmr_select(penalized, path_by_key, remaining_slots)
        final_scores = dict(penalized)
        for key in diversified_keys:
            hit = _hit_from_key(key, generation, final_scores.get(key, 0.0), "fused")
            if hit:
                fused_hits.append(hit)

    return RankedResult(hits=exact_hits + fused_hits, degraded=sorted(set(degraded)))
