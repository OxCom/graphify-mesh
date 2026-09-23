"""Symbol-anchored ranking for `search`.

A multi-token query often names one symbol (`export`, `AbstractCronJob`) plus
words describing its context. Each query token that resolves through the
exact-alias table to a RARE alias (few members) yields anchor candidates. A
candidate is scored by the idf of the OTHER query tokens found in its own
lexical fields and in its depth-1 EXTRACTED neighbours' fields. Only a
candidate that clears an absolute floor AND dominates the next one is pinned
right after the exact-alias hits; otherwise ranking is left untouched.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, field

from graphify_mesh.server import lexical_read, ranking
from graphify_mesh.server.store import Generation
from graphify_mesh.sync.lexical_index import normalize_alias_query, tokenize_text

# An alias form yields candidates only with 1..N members after the repo
# filter. Rarity is the alias member count, NOT the token's document
# frequency: a symbol name such as `export` can have the highest df in the
# query while its `.export()` alias still names only a handful of methods.
ANCHOR_MAX_ALIAS_MEMBERS = 10

# Absolute score floor a pinned anchor must reach, in idf units.
ANCHOR_MIN_SCORE = 10.0

# A pinned anchor must score at least this multiple of the next candidate.
ANCHOR_DOMINANCE = 2.0

ANCHOR_MAX_PINS = 2

# Score carried by anchor hits: below `retrieval.EXACT_MATCH_SCORE` so exact
# hits stay first, above every fused score (RRF sums stay below 1.0).
ANCHOR_MATCH_SCORE = 500_000.0

# Query tokens are caller-chosen, so the per-generation df memo is bounded;
# past the cap it is cleared and refilled.
_DOC_FREQ_CACHE_MAX = 4096


@dataclass
class _GenerationCache:
    generation_id: str
    # Identity check on top of the id: a weakref, so a replaced generation is
    # never kept alive by this cache.
    generation: weakref.ref[Generation]
    doc_ids_by_key: dict[str, list[int]] | None
    doc_freq: dict[str, int] = field(default_factory=dict)


# Only the most recent generation's cache is kept. Concurrent readers may
# race on this assignment; the loser just rebuilds, the content is the same,
# so no lock is needed.
_cache: _GenerationCache | None = None


def _generation_cache(generation: Generation) -> _GenerationCache:
    global _cache
    cache = _cache
    if (
        cache is None
        or cache.generation_id != generation.generation_id
        or cache.generation() is not generation
    ):
        cache = _GenerationCache(
            generation_id=generation.generation_id,
            generation=weakref.ref(generation),
            doc_ids_by_key=lexical_read.doc_ids_by_key(generation.lexical),
        )
        _cache = cache
    return cache


def _doc_freq(token: str, generation: Generation, cache: _GenerationCache) -> int:
    df = cache.doc_freq.get(token)
    if df is None:
        df = lexical_read.term_doc_freq(generation.lexical, token)
        if len(cache.doc_freq) >= _DOC_FREQ_CACHE_MAX:
            cache.doc_freq.clear()
        cache.doc_freq[token] = df
    return df


def _alias_members(
    form: str, generation: Generation, repo_filter: frozenset[str] | None
) -> set[str]:
    refs = lexical_read.alias_refs(generation.lexical, normalize_alias_query(form))
    return {key for repo, key in refs if repo_filter is None or repo in repo_filter}


def _neighbor_keys(
    key: str, generation: Generation, repo_filter: frozenset[str] | None
) -> set[str]:
    node_id = generation.node_id_by_key.get(key)
    if node_id is None:
        return set()
    keys: set[str] = set()
    for neighbor_id, edge in generation.adjacency.get(node_id, []):
        if not ranking.edge_followable(edge, include_inferred=False):
            continue
        neighbor_key = generation.key_by_node_id.get(neighbor_id)
        if neighbor_key is None:
            continue
        neighbor_node = generation.node_by_id.get(neighbor_id, {})
        if repo_filter is not None and neighbor_node.get("repo") not in repo_filter:
            continue
        keys.add(neighbor_key)
    return keys


def _full_token_keys(
    token: str, generation: Generation, repo_filter: frozenset[str] | None
) -> set[str]:
    """Every key `token` has a posting for (v2 path)."""
    return {
        key
        for repo, key, _field in lexical_read.term_postings(generation.lexical, token)
        if key is not None and (repo_filter is None or repo in repo_filter)
    }


def _probe_token_keys(
    token: str, generation: Generation, key_docs: dict[str, list[int]], doc_ids: set[int]
) -> set[str]:
    """The probe keys `token` has a posting for (v3 path). Probe keys are
    already repo-filtered, so no posting-level filter is needed."""
    hits = lexical_read.term_doc_hits(generation.lexical, token, doc_ids)
    return {key for key, ids in key_docs.items() if not hits.isdisjoint(ids)}


def _candidate_scores(
    tokens: list[str],
    via: dict[str, set[str]],
    generation: Generation,
    repo_filter: frozenset[str] | None,
) -> list[tuple[str, float]]:
    """Candidates with their idf score, best first. Only candidates and
    their neighbours are ever tested for membership, so on v3 the posting
    lookup is restricted to those keys instead of every posting."""
    cache = _generation_cache(generation)
    neighbors_by_key = {key: _neighbor_keys(key, generation, repo_filter) for key in via}
    needed = [t for t in tokens if any(t not in via_tokens for via_tokens in via.values())]

    total_docs = lexical_read.document_count(generation.lexical)
    token_idf: dict[str, float] = {}
    for token in needed:
        df = _doc_freq(token, generation, cache)
        token_idf[token] = math.log(total_docs / df) if df > 0 and total_docs > 0 else 0.0

    token_keys: dict[str, set[str]] = {}
    if cache.doc_ids_by_key is None:
        for token in needed:
            token_keys[token] = _full_token_keys(token, generation, repo_filter)
    else:
        probe = set(via).union(*neighbors_by_key.values())
        key_docs = {key: cache.doc_ids_by_key[key] for key in probe if key in cache.doc_ids_by_key}
        doc_ids = {doc_id for ids in key_docs.values() for doc_id in ids}
        for token in needed:
            token_keys[token] = _probe_token_keys(token, generation, key_docs, doc_ids)

    scores: list[tuple[str, float]] = []
    for key, via_tokens in via.items():
        neighbors = neighbors_by_key[key]
        score = 0.0
        for token in tokens:
            if token in via_tokens:
                continue
            keys = token_keys[token]
            if key in keys:
                score += token_idf[token]
            if not neighbors.isdisjoint(keys):
                score += token_idf[token]
        scores.append((key, score))
    scores.sort(key=lambda kv: (-kv[1], kv[0]))
    return scores


def anchor_keys(
    query: str,
    generation: Generation,
    repo_filter: frozenset[str] | None,
    exclude: set[str],
) -> list[str]:
    """Keys to pin right after the exact-alias hits, best first; `[]` when
    no candidate is both above `ANCHOR_MIN_SCORE` and dominant."""
    tokens = list(dict.fromkeys(tokenize_text(query)))
    if len(tokens) < 2:
        return []

    via: dict[str, set[str]] = {}
    for token in tokens:
        for form in (token, f".{token}()"):
            members = _alias_members(form, generation, repo_filter)
            if not 1 <= len(members) <= ANCHOR_MAX_ALIAS_MEMBERS:
                continue
            for key in members:
                if key in exclude or key not in generation.node_id_by_key:
                    continue
                via.setdefault(key, set()).add(token)
    if not via:
        return []

    scores = _candidate_scores(tokens, via, generation, repo_filter)

    pinned: list[str] = []
    for index, (key, score) in enumerate(scores):
        if len(pinned) >= ANCHOR_MAX_PINS:
            break
        next_score = scores[index + 1][1] if index + 1 < len(scores) else 0.0
        if score < ANCHOR_MIN_SCORE or score < ANCHOR_DOMINANCE * next_score:
            break
        pinned.append(key)
    return pinned
