"""WS5 candidate generation + fusion orchestration.

Scope filtering happens INSIDE each `*_candidates` function, before any
ranking/scoring is computed — never as a post-filter of an already-ranked
list (see `scope.py` module docstring for why that ordering matters).
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np

from graphify_mesh.server import anchors, lexical_read, ranking
from graphify_mesh.server.store import Generation
from graphify_mesh.sync.lexical_index import normalize_alias_query, tokenize_text
from graphify_mesh.sync.vectors import RepoVectors

EmbedQueryFn = Callable[[str], list[float] | None]
IsTestFn = Callable[[str], bool]
_T = TypeVar("_T")

# Score carried by exact-alias bypass hits. It must dominate every fused
# score (RRF sums stay below 1.0 even with all three retrievers agreeing)
# while remaining a finite float: the score is serialized with `json.dumps`,
# and `float("inf")` would emit the bare token `Infinity`, which strict JSON
# parsers reject.
EXACT_MATCH_SCORE = 1_000_000.0


# When test code is demoted, each retriever fills its candidate depth with
# source nodes from its whole score order, however many test nodes outrank
# them (test methods repeat the domain words in their names and can fill a
# list), and keeps this many of its best test nodes behind them.
TEST_CANDIDATE_DEPTH = 10
# One result slot in this many is kept for the best test hit when tests are
# demoted, so a test stays reachable when source hits alone could fill `k`.
# The slots sit at fixed positions of the whole result (10th, 20th, ...).
TEST_RESULT_SLOT_EVERY = 10
# A test takes a reserved slot only when it scores at least this fraction of
# the hit it pushes down. Derivation: per retriever that ranked that hit, the
# hit contributes at most rrf(0), and the first test at least rrf(depth), since
# at most `depth` source keys sit ahead of it after the source-first move; then
# the test-path penalty applies. A test found by every retriever that found the
# displaced hit clears it; one below it was missed by one of them.
TEST_SLOT_MIN_SCORE_RATIO = (
    ranking.TEST_PATH_PENALTY_FACTOR
    * ranking.rrf_contribution(
        max(
            ranking.CANDIDATE_DEPTH_LEXICAL,
            ranking.CANDIDATE_DEPTH_VECTOR,
            ranking.CANDIDATE_DEPTH_STRUCTURAL,
        )
    )
    / ranking.rrf_contribution(0)
)

# How many top query matches seed the structural (1-hop neighbour) retriever.
STRUCTURAL_SEED_COUNT = 10


@dataclass
class Hit:
    key: str
    repo: str
    source_file: str
    label: str
    node_id: str
    degree: int
    score: float
    match_type: str  # "exact" | "anchor" | "fused"
    deprecated: bool
    stale: bool = False


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
    is_test: IsTestFn | None = None,
    test_depth: int = 0,
) -> list[str]:
    """Rank lexical candidates using version-aware postings and document metadata.

    A term scores field boost x idf once per (key, field) it matched.

    With `is_test`, the result is the best `depth` non-test keys of the whole
    score order, then the best `test_depth` test keys."""
    tokens = tokenize_text(query)
    if not tokens:
        return []
    total_docs = max(lexical_read.document_count(lexical), 1)
    field_boosts = lexical.get("field_boosts", {})

    scores: dict[str, float] = {}
    for term in tokens:
        df = lexical_read.term_doc_freq(lexical, term)
        if df == 0:
            continue
        idf = math.log(1 + (total_docs / df))
        # Several graph nodes can share one logical key (one label declared on
        # several interfaces of a file); each brings its own posting. Count a
        # term once per field per key, or that label's score is multiplied.
        best: dict[tuple[str, str], float] = {}
        for repo, key, field_name in lexical_read.term_postings(lexical, term):
            if repo_filter is not None and repo not in repo_filter:
                continue
            if key is None:
                continue
            weight = field_boosts.get(field_name, 1.0)
            if weight > best.get((key, field_name), 0.0):
                best[(key, field_name)] = weight
        for (key, _field_name), weight in best.items():
            scores[key] = scores.get(key, 0.0) + idf * weight

    ranked = [k for k, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
    if is_test is not None:
        return _partition_tests(ranked, is_test, depth, test_depth)
    return ranked[:depth]


def _repo_scores(repo_vectors: RepoVectors, normalized_query: np.ndarray) -> np.ndarray:
    """Per-repo cosine scores against an already-normalized query vector.
    Mixed-dimension shard vs query carries no comparable signal (same rule
    `cosine_similarity` applied per key) — every key in that repo scores 0
    rather than being dropped from the pool, so it still tie-breaks by key
    like the old per-key comparison did."""
    if repo_vectors.dim != normalized_query.shape[0]:
        return np.zeros(len(repo_vectors), dtype=np.float32)
    return repo_vectors.normalized() @ normalized_query


def _embeddings_degraded_marker(
    embeddings: dict[str, RepoVectors], embed_query_fn: EmbedQueryFn
) -> str:
    """The `degraded` entry for a call where the vector retriever contributed
    nothing.

    Two very different situations used to collapse onto the same bare
    `"embeddings_unavailable"` string: a generation that published no vectors
    at all (a build/config problem) and a query-time embed call that failed
    (a backend problem). A client could not tell them apart, which is the
    whole reason a wedged backend looked like a misconfiguration for a day.

    The bare marker is kept for the not-published case — `store` already
    reports it that way for the same condition, and it stays the stable
    string those checks match. A query-time failure instead reports
    `"embeddings_unavailable:<reason>"`, taking the reason from the
    embedder when it exposes one; a plain callable (the `EmbedQueryFn`
    contract is only `str -> list[float] | None`) has no reason to give, so
    it falls back to the bare marker rather than inventing one.
    """
    if not embeddings:
        return "embeddings_unavailable"
    reason_fn = getattr(embed_query_fn, "failure_reason", None)
    if not callable(reason_fn):
        return "embeddings_unavailable"
    return f"embeddings_unavailable:{reason_fn()}"


def vector_candidates(
    query: str,
    embeddings: dict[str, RepoVectors],
    repo_filter: frozenset[str] | None,
    embed_query_fn: EmbedQueryFn,
    depth: int = ranking.CANDIDATE_DEPTH_VECTOR,
    demote_tests: bool = False,
    test_depth: int = 0,
) -> tuple[list[str], bool]:
    """Returns (ranked keys, degraded). `degraded=True` means the vector
    retriever contributed nothing this call (no embeddings published this
    generation, or the query-embed call failed) — the caller renormalizes
    fusion over whichever OTHER retrievers succeeded and must surface the
    marker `_embeddings_degraded_marker` builds in the response's `degraded`
    field, which distinguishes those two causes.

    Scoring is matrix cosine similarity per repo: each `RepoVectors`
    matrix's L2-normalized copy is cached on the instance
    (`RepoVectors.normalized()`, computed once per process no matter how
    many queries hit it this session) and multiplied against the
    L2-normalized query vector in one `@` call — equivalent to calling
    `cosine_similarity` per key, just batched over the whole repo instead
    of a Python-level loop.

    With `demote_tests`, the result is the best `depth` non-test keys, then
    the best `test_depth` test keys (test path read from the key).
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
    # candidates (plus ties). With test demotion the source rows and the test
    # rows are shortlisted separately, so tests cannot use up the depth.
    scored: list[tuple[str, float, bool]] = []
    for repo_id, repo_vectors in embeddings.items():
        if repo_filter is not None and repo_id not in repo_filter:
            continue
        if len(repo_vectors) == 0:
            continue
        scores = _repo_scores(repo_vectors, normalized_query)
        groups: tuple[tuple[np.ndarray | None, int, bool], ...] = ((None, depth, False),)
        if demote_tests:
            source_rows, test_rows = _source_and_test_rows(repo_vectors)
            groups = ((source_rows, depth, False), (test_rows, test_depth, True))
        keys = repo_vectors.keys
        for rows, limit, is_test_row in groups:
            for idx in _top_rows(scores, rows, limit):
                scored.append((keys[int(idx)], float(scores[int(idx)]), is_test_row))
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    if demote_tests:
        kept = _partition_tests(scored, lambda item: item[2], depth, test_depth)
        return [key for key, _score, _test in kept], False
    return [key for key, _score, _test in scored[:depth]], False


def _top_rows(scores: np.ndarray, rows: np.ndarray | None, depth: int) -> np.ndarray:
    """Row indexes of the `depth` best scores among `rows` (all rows when None),
    plus every row tied with the depth-th best, so a (score, key) sort of the
    result selects what a full sort would."""
    candidates = scores if rows is None else scores[rows]
    if depth <= 0 or candidates.shape[0] == 0:
        return np.empty(0, dtype=np.intp)
    if depth >= candidates.shape[0]:
        picked = np.arange(candidates.shape[0])
    else:
        threshold = candidates[np.argpartition(candidates, -depth)[-depth]]
        picked = np.flatnonzero(candidates >= threshold)
    return picked if rows is None else rows[picked]


# Row indexes of the source keys and of the test keys of each loaded
# `RepoVectors`, built on the first test-demoting query that reads the repo and
# dropped with it, so later queries never classify rows again.
_TEST_SPLITS: dict[int, tuple[weakref.ref, np.ndarray, np.ndarray]] = {}


def _source_and_test_rows(repo_vectors: RepoVectors) -> tuple[np.ndarray, np.ndarray]:
    cache_id = id(repo_vectors)
    cached = _TEST_SPLITS.get(cache_id)
    if cached is not None and cached[0]() is repo_vectors:
        return cached[1], cached[2]
    by_path: dict[str, bool] = {}
    mask = np.fromiter(
        (_key_is_test(key, by_path) for key in repo_vectors.keys),
        dtype=bool,
        count=len(repo_vectors.keys),
    )
    source_rows, test_rows = np.flatnonzero(~mask), np.flatnonzero(mask)
    ref = weakref.ref(repo_vectors, lambda _ref: _TEST_SPLITS.pop(cache_id, None))
    _TEST_SPLITS[cache_id] = (ref, source_rows, test_rows)
    return source_rows, test_rows


def _key_is_test(key: str, by_path: dict[str, bool]) -> bool:
    """`is_test_path` of the source file inside a logical key
    (`repo\\x1fsource_file\\x1flabel`, the node's own `source_file`)."""
    parts = key.split("\x1f", 2)
    path = parts[1] if len(parts) > 1 else ""
    if path not in by_path:
        by_path[path] = ranking.is_test_path(path)
    return by_path[path]


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
        stale=key in generation.stale_keys,
    )


def _partition_tests(
    ordered: Iterable[_T], is_test: Callable[[_T], bool], depth: int, test_depth: int
) -> list[_T]:
    """The first `depth` source items of `ordered`, then its first `test_depth`
    test items; stops reading once both are full."""
    source: list[_T] = []
    tests: list[_T] = []
    for key in ordered:
        if is_test(key):
            if len(tests) < test_depth:
                tests.append(key)
        elif len(source) < depth:
            source.append(key)
        if len(source) >= depth and len(tests) >= test_depth:
            break
    return source + tests


def _test_key_fn(generation: Generation) -> IsTestFn:
    def is_test(key: str) -> bool:
        node = generation.node_by_id.get(generation.node_id_by_key.get(key, ""), {})
        return ranking.is_test_path(node.get("source_file") or "")

    return is_test


def _source_first(keys: list[str], generation: Generation) -> list[str]:
    """`keys` with every test-path node moved after the source nodes, each
    group in its original order."""
    source: list[str] = []
    tests: list[str] = []
    for key in keys:
        node = generation.node_by_id.get(generation.node_id_by_key.get(key, ""), {})
        (tests if ranking.is_test_path(node.get("source_file") or "") else source).append(key)
    return source + tests


def _reserve_test_slots(
    selected: list[str],
    scored: list[tuple[str, float]],
    is_test: IsTestFn,
    slots: list[int],
    exclude: set[str],
) -> list[str]:
    """`selected` with a test hit placed at each index of `slots` that has no
    test at or before it yet: the best-scoring test not placed so far, when it
    scores at least `TEST_SLOT_MIN_SCORE_RATIO` of the hit it pushes down.
    Demotion orders tests last; this keeps one reachable per slot. Every
    choice reads only the result before it, so for a prefix-stable `selected`
    the top-n of a larger k equals the result for k=n."""
    score_of = dict(scored)
    spare = [
        key
        for key, _score in sorted(scored, key=lambda kv: (-kv[1], kv[0]))
        if key not in exclude and is_test(key)
    ]
    slot_set = set(slots)
    result: list[str] = []
    used: set[str] = set()
    next_selected = 0
    next_spare = 0
    slots_due = 0
    tests_placed = 0
    while len(result) < len(selected):
        while next_selected < len(selected) and selected[next_selected] in used:
            next_selected += 1
        pushed_down = selected[next_selected] if next_selected < len(selected) else None
        if len(result) in slot_set:
            slots_due += 1
            while next_spare < len(spare) and spare[next_spare] in used:
                next_spare += 1
            if tests_placed < slots_due and next_spare < len(spare):
                candidate = spare[next_spare]
                floor = TEST_SLOT_MIN_SCORE_RATIO * score_of.get(pushed_down or "", 0.0)
                if score_of[candidate] >= floor:
                    result.append(candidate)
                    used.add(candidate)
                    tests_placed += 1
                    continue
        if pushed_down is None:
            break
        result.append(pushed_down)
        used.add(pushed_down)
        if is_test(pushed_down):
            tests_placed += 1
    return result


def rank(
    query: str,
    generation: Generation,
    repo_filter: frozenset[str] | None,
    k: int,
    embed_query_fn: EmbedQueryFn,
    include_inferred: bool = False,
) -> RankedResult:
    """Top-level WS5 search orchestration: exact-alias bypass, symbol
    anchors (`anchors.py`), then lexical+vector+structural candidate
    generation, RRF fusion, hub/DEPRECATED/stale-semantic penalties, MMR diversification,
    deterministic tie-break."""
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

    anchor_hits: list[Hit] = []
    anchor_key_set: set[str] = set()
    if remaining_slots > 0:
        for key in anchors.anchor_keys(query, generation, repo_filter, exclude=selected_keys):
            hit = _hit_from_key(key, generation, anchors.ANCHOR_MATCH_SCORE, "anchor")
            if hit:
                anchor_hits.append(hit)
        del anchor_hits[remaining_slots:]
        # Anchors stay in the fused pool so MMR diversifies against them exactly
        # as when they rank there on their own; they are dropped after selection.
        anchor_key_set = {h.key for h in anchor_hits}
        remaining_slots -= len(anchor_hits)

    fused_hits: list[Hit] = []
    if remaining_slots > 0:
        demote_tests = not ranking.query_mentions_tests(query)
        # Tests move behind source BEFORE each list is cut to its depth: cut
        # first, and test methods (which repeat the domain words in their
        # names) can fill the whole list and drop the source node entirely.
        is_test = _test_key_fn(generation) if demote_tests else None
        test_depth = TEST_CANDIDATE_DEPTH if demote_tests else 0
        lexical_ranked = lexical_candidates(
            query, generation.lexical, repo_filter, is_test=is_test, test_depth=test_depth
        )
        vector_ranked, vec_degraded = vector_candidates(
            query,
            generation.embeddings,
            repo_filter,
            embed_query_fn,
            demote_tests=demote_tests,
            test_depth=test_depth,
        )
        if vec_degraded:
            vec_marker = _embeddings_degraded_marker(generation.embeddings, embed_query_fn)
            if vec_marker not in degraded:
                degraded.append(vec_marker)

        # Seeded from the query matches of both retrievers, not lexical alone, so
        # a node only the vector side found can still pull in its neighbours.
        seed_scores = ranking.fuse_rankings({"lexical": lexical_ranked, "vector": vector_ranked})
        seed_order = sorted(seed_scores, key=lambda key_: (-seed_scores[key_], key_))
        seed_for_structural = (exact_keys + seed_order)[:STRUCTURAL_SEED_COUNT]
        structural_ranked = structural_candidates(
            seed_for_structural, generation, repo_filter, include_inferred=include_inferred
        )

        rankings = {
            "lexical": lexical_ranked,
            "vector": vector_ranked,
            "structural": structural_ranked,
        }
        if demote_tests:
            # RRF scores by position, so a test node holding rank 3 pushes every
            # source node behind it down a slot; the per-score penalty below
            # cannot undo that. Tests keep their relative order, after source.
            rankings = {name: _source_first(keys, generation) for name, keys in rankings.items()}
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
            (
                key,
                ranking.apply_penalties(
                    key,
                    score,
                    degree_by_key,
                    path_by_key,
                    stale_keys=generation.stale_keys,
                    demote_tests=demote_tests,
                ),
            )
            for key, score in fused_scores.items()
        ]
        diversified_keys = ranking.mmr_select(
            penalized, path_by_key, remaining_slots + len(anchor_key_set)
        )
        diversified_keys = [key for key in diversified_keys if key not in anchor_key_set][
            :remaining_slots
        ]
        final_scores = dict(penalized)
        if is_test is not None:
            # Slot positions count from the top of the whole result, so the
            # exact and anchor hits ahead of the fused ones do not move them.
            ahead = len(exact_hits) + len(anchor_hits)
            slots = [
                position - ahead
                for position in range(TEST_RESULT_SLOT_EVERY - 1, k, TEST_RESULT_SLOT_EVERY)
                if position >= ahead
            ]
            diversified_keys = _reserve_test_slots(
                diversified_keys, penalized, is_test, slots, exclude=anchor_key_set
            )
        for key in diversified_keys:
            hit = _hit_from_key(key, generation, final_scores.get(key, 0.0), "fused")
            if hit:
                fused_hits.append(hit)

    return RankedResult(hits=exact_hits + anchor_hits + fused_hits, degraded=sorted(set(degraded)))
