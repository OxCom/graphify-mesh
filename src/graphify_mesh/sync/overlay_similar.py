"""WS4 ship-order item 2: `similar_approach` overlay edges — now backed by
the WS3 embedding index for real ANN mutual top-k cosine similarity.

Two scoring paths, both cross-repo only (same-repo pairs are never a
cross-project relation):

  1. Real path (`embedding_vectors_by_repo` has vectors for a node): mutual
     top-k ANN cosine similarity over the embedding index
     (`embed_similarity.mutual_top_k_pairs`), threshold + top_k as named,
     documented-untuned config (C7).

  2. Fallback path (a node has no vector at all — either the WS3 embed stage
     is degraded/not run this generation, or the node was skipped by WS3's
     trivial-accessor heuristic, `embedding.is_trivial_node`): the original
     WS4 placeholder scorer — exact normalized-label + same-`community_name`
     match. This is the documented `find_similar` fallback for
     trivial/unembedded nodes (deliverable 7): no vector lookup is attempted
     for them, ever; a structural exact-match is used instead so they are
     never silently left with zero similarity candidates.

Per-pair cap: in addition to the per-node `top_k` cap, no more than
`MAX_EDGES_PER_REPO_PAIR` similar_approach edges are ever emitted between
the same two repos in one generation, so two large/near-duplicate repos
can't flood the overlay with edges (WS4 ship-order item 2: "per-pair cap").
"""

from __future__ import annotations

import logging

from graphify_mesh.sync.embed_similarity import (
    CANDIDATE_FLOOR,
    CODE_EXTENSIONS,
    CROSS_LANGUAGE_THRESHOLD,
    IDENTITY_FLOOR,
    IDENTITY_RELATIVE,
    LANGUAGE_FAMILIES,
    MIN_CALIBRATION_COPIES,
    SIMILARITY_THRESHOLD_DEFAULT,
    mutual_top_k_pairs,
)
from graphify_mesh.sync.embedding import key_to_ref, node_key
from graphify_mesh.sync.overlay_refs import LogicalRef, OverlayEdge
from graphify_mesh.sync.vectors import RepoVectors

log = logging.getLogger(__name__)

PLACEHOLDER_PROVENANCE = "PLACEHOLDER_STRUCTURAL_MATCH"
PLACEHOLDER_CONFIDENCE = 0.4
EMBEDDING_PROVENANCE = "EXTRACTED_EMBEDDING_ANN_COSINE"

# WS4 ship-order item 2: cap on similar_approach edges between any single
# pair of repos, independent of the per-node top_k cap.
# A safety valve, not a budget. Raised from 50 to 200 and then to 2000 on 2026-09-30,
# once measurement showed the thing it was protecting against is prevented earlier and
# by something else.
#
# What bounds the population is the mutual top-k filter: each node keeps at most `top_k`
# candidates and a pair survives only if each side is in the other's list. With the
# same-path rule below removing copies, the entire live corpus yields 5536 cross-path
# pairs over 23 repos and 37 connected repo pairs, the largest single pair holding 1008.
# Against that, a cap of 200 discarded 30% of all edges and cut as high as cosine 0.8968;
# 500 still discarded 9%. Proportional forms were measured and are worse, not better:
# a cap of 2%/5%/10% of the smaller repo discarded 73%/60%/34% and cut higher still,
# because a large repo's analogues are not proportional to its node count.
#
# So the value below sits roughly twice the observed maximum and is expected never to
# bind. If it ever does, the log line says so — that is a signal to re-measure, not a
# number to raise quietly.
MAX_EDGES_PER_REPO_PAIR = 2000


def _normalize_label(label: str) -> str:
    return label.strip().lower()


def normalize_label(label: str) -> str:
    """Public wrapper around `_normalize_label` so callers outside this
    module (WS5's `graphify-mesh` MCP server, `find_similar`'s trivial/unembedded
    fallback path) reuse the IDENTICAL normalization this module's own
    exact-match scorer uses, rather than re-deriving it."""
    return _normalize_label(label)


def _fallback_exact_match_pairs(
    graphs_by_repo: dict[str, dict],
    excluded_keys: set[str],
    top_k: int,
) -> list[OverlayEdge]:
    """Original WS4 placeholder scorer, restricted to nodes NOT already
    covered by the real embedding path (`excluded_keys`). See module
    docstring path 2. Exact label+community match, mutual top-k trivially
    satisfied since candidacy here is symmetric and unranked."""
    buckets: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    for repo_id, graph_data in graphs_by_repo.items():
        for node in graph_data.get("nodes", []):
            if not isinstance(node, dict):
                continue
            label = node.get("label")
            community_name = node.get("community_name")
            source_file = node.get("source_file")
            if not label or not community_name or not source_file:
                continue
            key = node_key(repo_id, node)
            if key is not None and key in excluded_keys:
                continue
            bucket_key = (_normalize_label(label), community_name)
            buckets.setdefault(bucket_key, []).append((repo_id, source_file, label))

    edges: list[OverlayEdge] = []
    per_node_emitted: dict[tuple[str, str, str], int] = {}
    per_repo_pair_emitted: dict[frozenset, int] = {}
    for (norm_label, community_name), members in buckets.items():
        distinct_repos = {m[0] for m in members}
        if len(distinct_repos) < 2:
            continue
        seen_pairs: set[frozenset] = set()
        for i, (repo_a, file_a, label_a) in enumerate(members):
            for repo_b, file_b, label_b in members[i + 1 :]:
                if repo_a == repo_b:
                    continue
                node_a_key = (repo_a, file_a, label_a)
                node_b_key = (repo_b, file_b, label_b)
                pair_key = frozenset((node_a_key, node_b_key))
                if pair_key in seen_pairs:
                    continue
                if (
                    per_node_emitted.get(node_a_key, 0) >= top_k
                    or per_node_emitted.get(node_b_key, 0) >= top_k
                ):
                    continue
                repo_pair_key = frozenset((repo_a, repo_b))
                if per_repo_pair_emitted.get(repo_pair_key, 0) >= MAX_EDGES_PER_REPO_PAIR:
                    continue
                seen_pairs.add(pair_key)
                per_node_emitted[node_a_key] = per_node_emitted.get(node_a_key, 0) + 1
                per_node_emitted[node_b_key] = per_node_emitted.get(node_b_key, 0) + 1
                per_repo_pair_emitted[repo_pair_key] = (
                    per_repo_pair_emitted.get(repo_pair_key, 0) + 1
                )
                edges.append(
                    OverlayEdge(
                        type="similar_approach",
                        source=LogicalRef(repo=repo_a, source_file=file_a, qualified_label=label_a),
                        target=LogicalRef(repo=repo_b, source_file=file_b, qualified_label=label_b),
                        provenance=PLACEHOLDER_PROVENANCE,
                        confidence=PLACEHOLDER_CONFIDENCE,
                        evidence=(
                            "placeholder fallback scorer (no embedding available): "
                            "exact label+community match "
                            f"({norm_label!r} in {community_name!r})"
                        ),
                    )
                )
    return edges


def _extension(source_file: str | None) -> str:
    """The file's extension, lowercased, or the empty string."""
    if not source_file or "." not in source_file.rsplit("/", 1)[-1]:
        return ""
    return "." + source_file.rsplit(".", 1)[-1].lower()


def _same_language(ext_a: str, ext_b: str) -> bool:
    """Whether two extensions are the same language for threshold purposes."""
    if ext_a == ext_b:
        return True
    return any(ext_a in family and ext_b in family for family in LANGUAGE_FAMILIES)


def _calibrated_line(
    admit: dict[tuple, float],
    repo_pair_key: frozenset,
    ext_a: str,
    ext_b: str,
    fallback: float,
) -> float:
    """The admission line for a same-language pair, from the calibration it has.

    Three corrections over reading `admit[(repo_pair_key, ext_a)]` directly:

    * The two sides can carry different extensions inside one family (`.ts` against
      `.tsx`). Taking the first key's extension made the line depend on which repo id
      sorted first, so the same pair of files was judged differently depending on the
      repository names. The stricter of the two lines is used instead — deterministic,
      and it never admits on the weaker side's scale.
    * A `(pair, extension)` with too few copies used to fall straight back to the
      caller's absolute value, with all of its language bias. Measured: live
      `launcher<->styleguide` has 5 `.ts` copies, so its 368 `.ts` analogues were judged
      at 0.82 while the `.tsx` beside them was calibrated at 0.696. Any calibrated line
      from the same family in the same repo pair is a far better estimate than the
      constant, so the strictest of those is used before falling back.
    * The constant remains the last resort, for a pair with no family calibration at all.
    """
    lines = [admit[(repo_pair_key, ext)] for ext in (ext_a, ext_b) if (repo_pair_key, ext) in admit]
    if lines:
        return max(lines)
    family = next((f for f in LANGUAGE_FAMILIES if ext_a in f or ext_b in f), frozenset())
    family_lines = [
        line
        for (pair_key, ext), line in admit.items()
        if pair_key == repo_pair_key and ext in family
    ]
    if family_lines:
        return max(family_lines)
    return fallback


def _embedding_pairs(
    embedding_vectors_by_repo: dict[str, RepoVectors],
    top_k: int,
    threshold: float,
    embedding_model: str,
) -> tuple[list[OverlayEdge], set[str]]:
    total = sum(len(rv) for rv in embedding_vectors_by_repo.values())
    all_keys = {key for rv in embedding_vectors_by_repo.values() for key in rv.keys}
    if total < 2:
        return [], all_keys

    # Candidates are generated down to the floor, not to `threshold`: the admission
    # threshold is computed per repo pair below and can sit under the caller's value.
    pairs = mutual_top_k_pairs(
        embedding_vectors_by_repo, top_k=top_k, threshold=min(threshold, CANDIDATE_FLOOR)
    )

    # First pass: each repo pair's identity level, from the copies it holds. A same-path
    # pair is the identical file in both repos, so its score is what "the same code" means
    # for this language mix — the scale the analogue threshold is then expressed in.
    # Keyed by (repo pair, extension): a copy shares a path, so it shares an extension,
    # and that is the unit the language bias actually lives in. Calibrating per repo pair
    # alone applies the scale of whatever the repos share most — usually PHP or JSON
    # config — to analogues of every other language in the same repos.
    copies_by_pair: dict[tuple, list[float]] = {}
    for key_a, key_b, score in pairs:
        ref_a = key_to_ref(key_a)
        ref_b = key_to_ref(key_b)
        if ref_a.source_file and ref_a.source_file == ref_b.source_file:
            copies_by_pair.setdefault(
                (frozenset((ref_a.repo, ref_b.repo)), _extension(ref_a.source_file)), []
            ).append(score)

    admit: dict[tuple, float] = {}
    for calibration_key, scores in copies_by_pair.items():
        if len(scores) < MIN_CALIBRATION_COPIES:
            continue
        scores.sort()
        identity = scores[len(scores) // 2]
        repo_pair_key = calibration_key[0]
        if identity < IDENTITY_FLOOR:
            # No discriminative power for this pair: copies and unrelated code score
            # alike, so any threshold derived from it would admit noise.
            admit[calibration_key] = 2.0
            log.info(
                "similar_approach: %s (%s) identity level %.4f below "
                "IDENTITY_FLOOR=%.2f — no analogue edges emitted for it",
                "<->".join(sorted(repo_pair_key)),
                calibration_key[1] or "-",
                identity,
                IDENTITY_FLOOR,
            )
        else:
            admit[calibration_key] = identity * IDENTITY_RELATIVE

    edges: list[OverlayEdge] = []
    per_repo_pair_emitted: dict[frozenset, int] = {}
    for key_a, key_b, score in pairs:
        ref_a = key_to_ref(key_a)
        ref_b = key_to_ref(key_b)
        # Same file, two repos: a copy, not an approach. `similar_approach` answers
        # "where else is this done", and "at the identical path in the other repo" is
        # not an answer to that — the reader already knows a shared file is shared.
        # Measured on the live generation before this rule: every one of the 50 edges
        # for cem.hub<->cryengine.hub was a same-path pair scoring 0.9929 to 0.9989,
        # and 88% of all cem.hub<->cem.messages pairs above threshold were same-path.
        # They crowded out every genuine analogue, e.g. AbstractClient.ts <->
        # AbstractApiClient.ts at 0.9925 and QaApiAuthenticator <-> SystemAuthenticator
        # at 0.9211, which appear only once copies stop competing for the budget.
        if ref_a.source_file and ref_a.source_file == ref_b.source_file:
            continue
        repo_pair_key = frozenset((ref_a.repo, ref_b.repo))
        ext_a = _extension(ref_a.source_file)
        ext_b = _extension(ref_b.source_file)
        # An approach lives in code. Config and documentation resembling each other is
        # not a finding, and it spends the same budget as one.
        if ext_a not in CODE_EXTENSIONS or ext_b not in CODE_EXTENSIONS:
            continue
        if _same_language(ext_a, ext_b):
            applied = _calibrated_line(admit, repo_pair_key, ext_a, ext_b, threshold)
        else:
            applied = CROSS_LANGUAGE_THRESHOLD
        if score < applied:
            continue
        emitted = per_repo_pair_emitted.get(repo_pair_key, 0)
        if emitted >= MAX_EDGES_PER_REPO_PAIR:
            if emitted == MAX_EDGES_PER_REPO_PAIR:
                log.warning(
                    "similar_approach: repo pair %s hit MAX_EDGES_PER_REPO_PAIR=%d — "
                    "further analogues are being dropped, starting at confidence %.4f. "
                    "The cap is a safety valve sized about twice the measured maximum; "
                    "reaching it means the corpus changed shape, so re-measure before "
                    "raising it",
                    "<->".join(sorted(repo_pair_key)),
                    MAX_EDGES_PER_REPO_PAIR,
                    score,
                )
                per_repo_pair_emitted[repo_pair_key] = emitted + 1
            continue
        per_repo_pair_emitted[repo_pair_key] = emitted + 1
        edges.append(
            OverlayEdge(
                type="similar_approach",
                source=ref_a,
                target=ref_b,
                provenance=EMBEDDING_PROVENANCE,
                confidence=round(score, 4),
                evidence=(
                    f"embedding ANN cosine similarity={score:.4f} model={embedding_model!r} "
                    f"threshold={applied:.4f} (mutual top-{top_k})"
                ),
            )
        )
    return edges, all_keys


def compute_similar_approach_edges(
    graphs_by_repo: dict[str, dict],
    embedding_vectors_by_repo: dict[str, RepoVectors] | None = None,
    top_k: int = 5,
    threshold: float = SIMILARITY_THRESHOLD_DEFAULT,
    embedding_model: str = "unknown",
) -> list[OverlayEdge]:
    """Mutual top-k cross-project similarity (WS4 item 2), backed by the WS3
    embedding index when available, falling back to the exact-match
    placeholder scorer for nodes without a vector (see module docstring).

    Args:
        graphs_by_repo: repo_id -> that repo's raw per-repo graph.json data.
        embedding_vectors_by_repo: repo_id -> RepoVectors, produced by
            `embedding.run_embedding_stage` this generation. `None` or empty
            means no embedding index is available at all this run (e.g. the
            embed stage was degraded) — every node falls back to the
            exact-match scorer, reproducing the pre-WS3 placeholder
            behavior exactly.
        top_k: cap on edges emitted per node (both scoring paths).
        threshold: cosine similarity cutoff for the embedding path (C7:
            named, documented-untuned default; see embed_similarity.py).
    """
    embedding_edges: list[OverlayEdge] = []
    embedded_keys: set[str] = set()
    if embedding_vectors_by_repo:
        embedding_edges, embedded_keys = _embedding_pairs(
            embedding_vectors_by_repo, top_k, threshold, embedding_model
        )

    fallback_edges = _fallback_exact_match_pairs(graphs_by_repo, embedded_keys, top_k)
    return embedding_edges + fallback_edges
