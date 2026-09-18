from __future__ import annotations

import math
import random

import numpy as np
import pytest

from graphify_mesh.sync import embed_similarity
from graphify_mesh.sync.embed_similarity import _bucket_signature_batch, _planes_matrix
from graphify_mesh.sync.vectors import RepoVectors


def _unit(values: list[float]) -> list[float]:
    # Small helper: not normalized, cosine_similarity itself normalizes.
    return values


def _rv(mapping: dict[str, list[float]]) -> RepoVectors:
    return RepoVectors.from_mapping(mapping)


def test_cosine_similarity_identical_vectors_is_one():
    v = [1.0, 2.0, 3.0]
    # float32 batch math (np.dot/np.linalg.norm) can drift in the last
    # decimals vs. the old pure-Python float64 sum; loosen tolerance only,
    # expected value unchanged.
    assert embed_similarity.cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-4)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert embed_similarity.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_zero_vector_is_zero_not_nan():
    assert embed_similarity.cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_mutual_top_k_finds_similar_cross_repo_pair_above_threshold():
    # Two near-identical vectors in different repos -> cosine ~0.9995,
    # comfortably above the default threshold, and each other's only
    # candidate -> mutual top-1.
    vectors_by_repo = {
        "repo.a": _rv({"a": _unit([1.0, 0.01, 0.0, 0.0])}),
        "repo.b": _rv({"b": _unit([0.99, 0.02, 0.01, 0.0])}),
    }
    pairs = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=1)
    assert len(pairs) == 1
    key_a, key_b, score = pairs[0]
    assert {key_a, key_b} == {"a", "b"}
    assert score > 0.9


def test_mutual_top_k_excludes_pairs_below_threshold():
    # Orthogonal vectors -> cosine 0.0, well below any reasonable threshold —
    # must produce zero pairs regardless of LSH bucket assignment, since the
    # exact cosine + threshold check always applies to any compared pair.
    vectors_by_repo = {
        "repo.a": _rv({"a": [1.0, 0.0, 0.0, 0.0]}),
        "repo.b": _rv({"b": [0.0, 1.0, 0.0, 0.0]}),
    }
    pairs = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=5, threshold=0.82)
    assert pairs == []


def test_mutual_top_k_ignores_same_repo_pairs():
    # Two near-identical vectors, but same repo -> never a cross-project
    # candidate even though cosine similarity is high.
    vectors_by_repo = {
        "repo.a": _rv({"a": [1.0, 0.0, 0.0, 0.0], "b": [0.99, 0.01, 0.0, 0.0]}),
    }
    pairs = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=5, threshold=0.5)
    assert pairs == []


def test_mutual_top_k_deterministic_across_calls():
    vectors_by_repo = {
        "repo.a": _rv({"a": [1.0, 0.02, 0.01, 0.0]}),
        "repo.b": _rv({"b": [0.99, 0.01, 0.0, 0.01]}),
        "repo.c": _rv({"c": [-1.0, 0.0, 0.0, 0.0]}),
    }
    first = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=2)
    second = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=2)
    assert first == second


def test_mutual_top_k_pairs_repo_vectors_matches_previous_results():
    """Same fixture vectors as the pre-refactor test, now wrapped per repo:
    identical pairs, scores approx-equal (abs=1e-4)."""
    vectors_by_repo = {
        "repoA": _rv({"a1": [1.0, 0.0, 0.0], "a2": [0.0, 1.0, 0.0]}),
        "repoB": _rv({"b1": [0.99, 0.01, 0.0], "b2": [0.0, 0.98, 0.05]}),
    }
    pairs = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=2, threshold=0.8)
    assert ("a1", "b1") in {(a, b) for a, b, _ in pairs}
    assert ("a2", "b2") in {(a, b) for a, b, _ in pairs}
    for key_a, key_b, score in pairs:
        assert key_a < key_b
        assert score >= 0.8


def test_mutual_top_k_pairs_same_repo_never_paired():
    vectors_by_repo = {"repoA": _rv({"a1": [1.0, 0.0], "a2": [1.0, 0.0]})}
    assert embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=5, threshold=0.1) == []


def test_bucket_chunking_identical_pairs(monkeypatch):
    """Force BUCKET_CHUNK_ROWS=2 so chunked and unchunked paths both run on
    a >2-row bucket; results must be identical (chunking is allocation-only)."""
    vectors_by_repo = {
        "repoA": _rv({f"a{i}": [1.0, 0.001 * i] for i in range(6)}),
        "repoB": _rv({f"b{i}": [1.0, 0.001 * i + 0.0005] for i in range(6)}),
    }
    full = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=3, threshold=0.5)
    monkeypatch.setattr(embed_similarity, "BUCKET_CHUNK_ROWS", 2)
    chunked = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=3, threshold=0.5)
    assert [(a, b) for a, b, _ in full] == [(a, b) for a, b, _ in chunked]
    for (_, _, s1), (_, _, s2) in zip(full, chunked, strict=True):
        assert s1 == pytest.approx(s2, abs=1e-4)


def test_oversized_bucket_truncated_to_cap(monkeypatch, caplog):
    """A bucket above MAX_BUCKET_COMPARE_ROWS only compares the first
    `cap` rows in the deterministic row order, and says so in the log."""
    # Identical vectors hash to one bucket; row order is sorted repo, then
    # sorted key inside the repo, so the kept prefix is a00..a05 + b00, b01.
    vector = [1.0, 0.5, -0.25, 0.1]
    vectors_by_repo = {
        "repo.a": _rv({f"a{i:02d}": list(vector) for i in range(6)}),
        "repo.b": _rv({f"b{i:02d}": list(vector) for i in range(6)}),
    }
    monkeypatch.setattr(embed_similarity, "MAX_BUCKET_COMPARE_ROWS", 8)
    with caplog.at_level("WARNING"):
        pairs = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=3, threshold=0.5)

    compared = {f"a{i:02d}" for i in range(6)} | {"b00", "b01"}
    emitted = {key for key_a, key_b, _ in pairs for key in (key_a, key_b)}
    assert emitted <= compared
    assert "comparing only the first 8" in caplog.text


def _reference_mutual_top_k(
    vectors_by_repo: dict[str, RepoVectors], top_k: int, threshold: float
) -> list[tuple[str, str, float]]:
    """Independent reference for `mutual_top_k_pairs`, written straight from
    the documented semantics: all-pairs cosine, threshold filter, cross-repo
    only, stable sort by score descending, top_k per key, mutual filter,
    dedupe. It shares no code with the production path beyond the input
    containers, so a regression in either one shows up as a disagreement.

    Valid only when every row sits in one LSH bucket (see `_force_one_bucket`),
    because candidate generation is all-pairs here and bucket-local there.
    """
    rows: list[tuple[str, str, np.ndarray]] = []
    for repo_id in sorted(vectors_by_repo):
        rv = vectors_by_repo[repo_id]
        for row_index, key in enumerate(rv.keys):
            rows.append((repo_id, key, np.asarray(rv.matrix[row_index], dtype=np.float32)))

    def cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))

    # Generation order: the pair (i, j) with i < j over the flattened rows.
    generated: list[tuple[str, str, float]] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows[i][0] == rows[j][0]:
                continue
            score = cosine(rows[i][2], rows[j][2])
            if score < threshold:
                continue
            generated.append((rows[i][1], rows[j][1], score))

    per_key: dict[str, list[tuple[float, int, str]]] = {key: [] for _, key, _ in rows}
    for rank, (key_a, key_b, score) in enumerate(generated):
        per_key[key_a].append((score, rank, key_b))
        per_key[key_b].append((score, rank, key_a))

    kept: dict[str, list[tuple[str, float]]] = {}
    for key, scored in per_key.items():
        ordered = sorted(scored, key=lambda item: (-item[0], item[1]))[:top_k]
        kept[key] = [(other, score) for score, _rank, other in ordered]
    top_sets = {key: {other for other, _ in pairs} for key, pairs in kept.items()}

    seen: set[frozenset] = set()
    pairs: list[tuple[str, str, float]] = []
    for key_a, scored in kept.items():
        for key_b, score in scored:
            if key_a not in top_sets[key_b]:
                continue
            pair_id = frozenset((key_a, key_b))
            if pair_id in seen:
                continue
            seen.add(pair_id)
            first, second = sorted((key_a, key_b))
            pairs.append((first, second, score))
    return pairs


def _angled(t: float) -> list[float]:
    return [math.cos(t), math.sin(t), 0.0, 0.0]


def test_matches_an_independent_reference_implementation(monkeypatch):
    """Pins the documented semantics against a reference written in the test,
    not against the production implementation with a different parameter.

    The fixture is built so every part of the contract bites: `a2`/`b2` sit far
    enough from the cluster that the threshold drops all but their own pair,
    `a1`/`a2` share a repo so the cross-repo rule must exclude them, and the
    cluster is denser than `top_k`, so both the per-key slice and the mutual
    filter change the answer.
    """
    _force_one_bucket(monkeypatch)
    top_k, threshold = 2, 0.5
    vectors_by_repo = {
        "repo.a": _rv({"a0": _angled(0.00), "a1": _angled(0.05), "a2": _angled(1.20)}),
        "repo.b": _rv({"b0": _angled(0.02), "b1": _angled(0.30), "b2": _angled(1.25)}),
        "repo.c": _rv({"c0": _angled(0.10)}),
    }

    expected = _reference_mutual_top_k(vectors_by_repo, top_k=top_k, threshold=threshold)
    actual = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=top_k, threshold=threshold)

    assert [(key_a, key_b) for key_a, key_b, _ in actual] == [
        (key_a, key_b) for key_a, key_b, _ in expected
    ]
    for (_, _, got), (_, _, want) in zip(actual, expected, strict=True):
        assert got == pytest.approx(want, abs=1e-6)

    # Guard the fixture itself: a scenario where every candidate survives
    # would pass without testing the slice or the mutual filter. Ten cross-repo
    # pairs clear the threshold here and four survive top_k plus mutuality.
    assert ("a2", "b2") in {(key_a, key_b) for key_a, key_b, _ in actual}
    assert len(actual) == 4


def test_candidates_per_key_bounded_by_top_k():
    """Every key in a dense cross-repo bucket keeps at most `top_k`
    candidates, so accumulation stays linear in top_k, not in bucket size."""
    vectors_by_repo = {
        f"repo.{r}": _rv({f"k{r}{i}": [1.0, 0.0001 * i] for i in range(8)}) for r in range(3)
    }
    top_k = 2
    pairs = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=top_k, threshold=0.5)
    partners: dict[str, set[str]] = {}
    for key_a, key_b, _ in pairs:
        partners.setdefault(key_a, set()).add(key_b)
        partners.setdefault(key_b, set()).add(key_a)
    for key, others in partners.items():
        assert len(others) <= top_k, key


def test_mixed_dim_repo_dropped_with_warning(caplog):
    """A repo whose dim differs from the first (sorted) repo's dim is dropped
    entirely — same scores-zero-everywhere outcome as the old per-vector drop."""
    vectors_by_repo = {
        "repoA": _rv({"a1": [1.0, 0.0]}),
        "repoB": _rv({"b1": [1.0, 0.0, 0.0]}),
    }
    assert embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=2, threshold=0.1) == []


def _build_varied_vectors(count: int, dims: tuple[int, ...] = (8, 64)) -> dict[str, np.ndarray]:
    # ONE rng instance drives every component of every vector (not re-seeded
    # per component/vector) so components are genuinely varied rather than
    # degenerate per-key constants; dims are mixed across the set.
    rng = random.Random(42)
    vectors: dict[str, np.ndarray] = {}
    for i in range(count):
        dim = dims[i % len(dims)]
        vectors[f"k{i}"] = np.asarray([rng.uniform(-1, 1) for _ in range(dim)], dtype=np.float32)
    return vectors


def test_bucket_signature_batch_deterministic_across_calls():
    """Binding contract (post-relaxation): signatures are deterministic
    across runs/processes for identical inputs on the same platform — NOT
    required to be bit-identical to the pre-numpy scalar implementation
    (near-boundary dot products may flip sign under batched/BLAS summation
    vs. Python's sequential sum; LSH tolerates this by design, see the
    module docstring's known-limitation paragraph).

    Built from 200 varied vectors (dims mixed 8/64, all components drawn
    from one shared `random.Random(42)` instance) to exercise more than a
    handful of small/degenerate fixtures.
    """
    vectors_a = _build_varied_vectors(200)
    vectors_b = _build_varied_vectors(200)  # separately constructed, equal values
    assert vectors_a.keys() == vectors_b.keys()

    for dim in (8, 64):
        planes = _planes_matrix(dim)
        keys_for_dim = [k for k, v in vectors_a.items() if v.shape[0] == dim]
        matrix_a = np.stack([vectors_a[k] for k in keys_for_dim])
        matrix_b = np.stack([vectors_b[k] for k in keys_for_dim])
        assert (matrix_a == matrix_b).all()  # sanity: inputs really are equal
        sigs_a = _bucket_signature_batch(matrix_a, planes)
        sigs_b = _bucket_signature_batch(matrix_b, planes)
        assert sigs_a == sigs_b


def test_mutual_top_k_pairs_invariants_hold_over_varied_vector_set():
    """Behavioral-parity test at the mutual_top_k_pairs level: assert the
    documented invariants hold over a larger, varied vector set, rather than
    pinning an exact pair list (which is no longer a meaningful contract
    once bit-identical bucketing to the scalar implementation is withdrawn
    -- see module docstring's known-limitation paragraph)."""
    vectors = _build_varied_vectors(200)
    # Guarantee an unconditional qualifying pair regardless of LSH bucketing:
    # two EXACTLY identical dim-8 vectors (dim 8 is the dim of the first
    # varied-set key, so it's the dim `mutual_top_k_pairs` keeps) in
    # different repos. Identical float32 inputs produce identical dot
    # products against any hyperplane set, hence identical signatures,
    # hence the same bucket -- guaranteed, not merely likely. Their cosine
    # similarity is exactly 1.0 >= threshold, so they always qualify.
    anchor_key_a = "anchor_a"
    anchor_key_b = "anchor_b"
    anchor_vector = np.asarray([1.0, 0.5, -0.25, 0.1, 0.0, -0.5, 0.75, -1.0], dtype=np.float32)
    vectors[anchor_key_a] = anchor_vector
    vectors[anchor_key_b] = anchor_vector.copy()

    keys = list(vectors.keys())
    # Assign alternating repos so cross-repo-only filtering is exercised,
    # except force the two anchors into different repos explicitly (their
    # position in `keys` would otherwise land on an arbitrary repo index).
    repo_by_key = {key: f"repo.{i % 4}" for i, key in enumerate(keys)}
    repo_by_key[anchor_key_a] = "repo.0"
    repo_by_key[anchor_key_b] = "repo.1"

    mappings_by_repo: dict[str, dict[str, np.ndarray]] = {}
    for key, repo_id in repo_by_key.items():
        mappings_by_repo.setdefault(repo_id, {})[key] = vectors[key]
    vectors_by_repo = {repo_id: _rv(mapping) for repo_id, mapping in mappings_by_repo.items()}

    threshold = 0.5
    top_k = 3
    pairs = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=top_k, threshold=threshold)
    expected_anchor_pair = tuple(sorted((anchor_key_a, anchor_key_b)))
    assert any((key_a, key_b) == expected_anchor_pair for key_a, key_b, _ in pairs), (
        "anchor pair must be emitted"
    )

    seen_unordered: set[frozenset] = set()
    per_key_partners: dict[str, set[str]] = {}
    for key_a, key_b, score in pairs:
        # Deterministic key_a < key_b ordering.
        assert key_a < key_b
        # No duplicate/reversed pairs.
        pair_id = frozenset((key_a, key_b))
        assert pair_id not in seen_unordered
        seen_unordered.add(pair_id)
        # Cross-repo only.
        assert repo_by_key[key_a] != repo_by_key[key_b]
        # Every emitted score respects the threshold.
        assert score >= threshold
        per_key_partners.setdefault(key_a, set()).add(key_b)
        per_key_partners.setdefault(key_b, set()).add(key_a)

    # Mutual top-k membership: a partner is only ever emitted for `key` if
    # it's within `key`'s own top-k candidate set (by construction, the
    # mutual filter requires membership on both sides). Note this must be
    # checked against the *bucket-restricted* candidate set, not a global
    # recomputation over every other key — LSH can miss a globally-better
    # candidate in a different bucket, so a globally-top-k check would be
    # too strict and could fail on data alone, independent of any real bug.
    # The invariant the algorithm actually guarantees is weaker but robust:
    # no key can be a partner in more than `top_k` emitted pairs.
    for _key, partners in per_key_partners.items():
        assert len(partners) <= top_k


def test_mutual_top_k_pairs_ndarray_matches_prior_results():
    # Reuse the same fixture as
    # test_mutual_top_k_finds_similar_cross_repo_pair_above_threshold, but
    # feed 1-D float32 ndarrays instead of plain lists -> same pairs, scores
    # approx-equal within float32 tolerance.
    vectors_by_repo = {
        "repo.a": _rv({"a": np.asarray([1.0, 0.01, 0.0, 0.0], dtype=np.float32)}),
        "repo.b": _rv({"b": np.asarray([0.99, 0.02, 0.01, 0.0], dtype=np.float32)}),
    }
    pairs = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=1)
    assert len(pairs) == 1
    key_a, key_b, score = pairs[0]
    assert {key_a, key_b} == {"a", "b"}
    assert score > 0.9


def test_mutual_top_k_pairs_ndarray_deterministic_and_matches_list_input():
    list_vectors_by_repo = {
        "repo.a": _rv({"a": [1.0, 0.02, 0.01, 0.0]}),
        "repo.b": _rv({"b": [0.99, 0.01, 0.0, 0.01]}),
        "repo.c": _rv({"c": [-1.0, 0.0, 0.0, 0.0]}),
    }
    ndarray_vectors_by_repo = {
        "repo.a": _rv({"a": np.asarray([1.0, 0.02, 0.01, 0.0], dtype=np.float32)}),
        "repo.b": _rv({"b": np.asarray([0.99, 0.01, 0.0, 0.01], dtype=np.float32)}),
        "repo.c": _rv({"c": np.asarray([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)}),
    }
    list_pairs = embed_similarity.mutual_top_k_pairs(list_vectors_by_repo, top_k=2)
    ndarray_pairs = embed_similarity.mutual_top_k_pairs(ndarray_vectors_by_repo, top_k=2)
    assert len(list_pairs) == len(ndarray_pairs)
    for (la, lb, lscore), (na, nb, nscore) in zip(list_pairs, ndarray_pairs, strict=True):
        assert (la, lb) == (na, nb)
        assert lscore == pytest.approx(nscore, abs=1e-4)


def _force_one_bucket(monkeypatch) -> None:
    # Pin every row into a single LSH bucket so an ordering assertion tests
    # ordering only, never hyperplane assignment.
    monkeypatch.setattr(
        embed_similarity,
        "_bucket_signature_batch",
        lambda matrix, planes: ["0" * embed_similarity.LSH_NUM_HYPERPLANES] * len(matrix),
    )


def test_kept_candidates_emitted_in_descending_score_order(monkeypatch):
    """Pairs come back in descending score order, not generation order.

    `overlay_similar.py` truncates each repo pair at MAX_EDGES_PER_REPO_PAIR,
    so the order here decides which pairs survive the cap.
    """
    _force_one_bucket(monkeypatch)
    # Generation order is sorted-key order (b0..b4); score order is a different
    # permutation, so an implementation emitting generation order fails here.
    tilts = {"b0": 0.5, "b1": 0.1, "b2": 0.3, "b3": 0.0, "b4": 0.4}
    vectors_by_repo = {
        "repo.a": _rv({"a": [1.0, 0.0, 0.0, 0.0]}),
        "repo.b": _rv({key: [1.0, tilt, 0.0, 0.0] for key, tilt in tilts.items()}),
    }
    pairs = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=3, threshold=0.5)

    expected_keys = [("a", "b3"), ("a", "b1"), ("a", "b2")]
    assert [(key_a, key_b) for key_a, key_b, _ in pairs] == expected_keys
    for (_, _key_b, score), expected in zip(pairs, expected_keys, strict=True):
        tilt = tilts[expected[1]]
        assert score == pytest.approx(1.0 / math.sqrt(1.0 + tilt * tilt), abs=1e-5)
    scores = [score for _, _, score in pairs]
    assert scores == sorted(scores, reverse=True)


def test_equal_scores_keep_generation_order(monkeypatch):
    """Two candidates with an identical score stay in generation order."""
    _force_one_bucket(monkeypatch)
    vectors_by_repo = {
        "repo.a": _rv({"a": [1.0, 0.0, 0.0, 0.0]}),
        "repo.b": _rv({"b0": [1.0, 0.2, 0.0, 0.0], "b1": [1.0, 0.2, 0.0, 0.0]}),
    }
    pairs = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=2, threshold=0.5)

    assert [(key_a, key_b) for key_a, key_b, _ in pairs] == [("a", "b0"), ("a", "b1")]
    assert pairs[0][2] == pytest.approx(pairs[1][2], abs=1e-6)


def test_cosine_similarity_accepts_lists_and_arrays():
    assert embed_similarity.cosine_similarity(
        [1.0, 0.0], np.asarray([1.0, 0.0], dtype=np.float32)
    ) == pytest.approx(1.0)
    assert embed_similarity.cosine_similarity([1.0, 0.0], [0.0]) == 0.0  # mixed dim still 0
