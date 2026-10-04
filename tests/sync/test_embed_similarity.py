from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pytest

from graphify_mesh.sync import embed_similarity, overlay_similar
from graphify_mesh.sync.embed_similarity import _bucket_signature_batch, _planes_matrix
from graphify_mesh.sync.vectors import RepoVectors

CORPUS_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "similarity"


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
    `cap` rows in the deterministic row order, and says so in the log.

    This exercises the LSH path, which a corpus this small no longer takes by
    default: since 2026-09-30 anything at or below EXACT_MAX_ROWS is compared
    exactly, so the cap is forced down here to reach the bucketing code.
    """
    # Identical vectors hash to one bucket; row order is sorted repo, then
    # sorted key inside the repo, so the kept prefix is a00..a05 + b00, b01.
    vector = [1.0, 0.5, -0.25, 0.1]
    vectors_by_repo = {
        "repo.a": _rv({f"a{i:02d}": list(vector) for i in range(6)}),
        "repo.b": _rv({f"b{i:02d}": list(vector) for i in range(6)}),
    }
    monkeypatch.setattr(embed_similarity, "EXACT_MAX_ROWS", 4)
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


def test_pairs_ordered_by_score_across_seed_keys(monkeypatch):
    """Descending score ACROSS seed keys, not just within one.

    The test above uses a single seed key, so it passes on an implementation that
    sorts inside each key's list and then emits the keys in insertion order. That
    is what shipped until 2026-09-30, and insertion order is path order: on the
    real corpus the MAX_EDGES_PER_REPO_PAIR budget went entirely to `.docker/...`,
    `.gitlab-ci.yml` and `composer.json`, and no PHP or TS symbol ever received a
    similar_approach edge. Here the alphabetically FIRST seed key carries the
    LOWEST score, so key order and score order disagree.
    """
    _force_one_bucket(monkeypatch)
    # a1 pairs weakest, a3 strongest; a tilt of 0 is a perfect match.
    tilts = {"a1": 0.5, "a2": 0.3, "a3": 0.0}
    vectors_by_repo = {
        "repo.a": _rv({key: [1.0, tilt, 0.0, 0.0] for key, tilt in tilts.items()}),
        "repo.b": _rv(
            {"b1": [1.0, 0.5, 0.0, 0.0], "b2": [1.0, 0.3, 0.0, 0.0], "b3": [1.0, 0.0, 0.0, 0.0]}
        ),
    }
    pairs = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=1, threshold=0.5)

    scores = [score for _, _, score in pairs]
    assert scores == sorted(scores, reverse=True), (
        "pairs must be globally descending: overlay_similar truncates this list "
        "at MAX_EDGES_PER_REPO_PAIR, so whatever comes first is what survives"
    )
    assert pairs, "the fixture must produce at least one mutual pair"
    assert pairs[0][0] == "a3" or pairs[0][1] == "a3", (
        "the strongest pair involves a3, which sorts last by key"
    )


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


# --- Synthetic corpus (tests/fixtures/similarity/) -------------------------
#
# Every test above synthesizes a handful of vectors and checks mechanics. Four
# defects shipped through that suite and were found only against the live
# corpus, because each one is a property of a realistic score distribution:
# order across seed keys, LSH recall at the working threshold, copies
# outranking analogues, and a threshold that meant different things per
# language. The fixture keeps the shape of the live corpus and none of its
# content; `expected.json` holds the values the generator measured on the
# files as written, so a behaviour change fails against a recorded number.

_SEP = "\x1f"


def _path_of(key: str) -> str:
    return key.split(_SEP)[1]


def _repo_of(key: str) -> str:
    return key.split(_SEP)[0]


def _load_corpus() -> dict[str, RepoVectors]:
    # Mirror of `build_corpus.load_fixture`: the generator verifies its
    # expectations through the same steps, so the two must stay in step.
    manifest = json.loads((CORPUS_DIR / "corpus.json").read_text(encoding="utf-8"))
    return {
        repo_id: RepoVectors.from_rows(
            list(keys), np.load(CORPUS_DIR / f"{repo_id}.npy").astype(np.float32)
        )
        for repo_id, keys in manifest["keys"].items()
    }


@pytest.fixture(scope="module")
def corpus() -> dict[str, RepoVectors]:
    return _load_corpus()


@pytest.fixture(scope="module")
def expected() -> dict:
    return json.loads((CORPUS_DIR / "expected.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def corpus_pairs(corpus, expected) -> list[tuple[str, str, float]]:
    """Mutual pairs at the default threshold: what the pair-level tests reason about."""
    return embed_similarity.mutual_top_k_pairs(corpus, top_k=expected["top_k"])


@pytest.fixture(scope="module")
def corpus_candidates(corpus, expected) -> list[tuple[str, str, float]]:
    """Mutual pairs down to CANDIDATE_FLOOR: what overlay_similar calibrates from."""
    return embed_similarity.mutual_top_k_pairs(
        corpus,
        top_k=expected["top_k"],
        threshold=min(expected["threshold"], overlay_similar.CANDIDATE_FLOOR),
    )


@pytest.fixture(scope="module")
def corpus_edges(corpus, expected):
    # No graphs: the placeholder fallback has nothing to add, every edge is an
    # embedding edge.
    return overlay_similar.compute_similar_approach_edges({}, corpus, top_k=expected["top_k"])


def _edges_between(edges, repos: list[str]):
    wanted = set(repos)
    return [edge for edge in edges if {edge.source.repo, edge.target.repo} == wanted]


def _pairs_between(pairs, repos: list[str]):
    wanted = set(repos)
    return [p for p in pairs if {_repo_of(p[0]), _repo_of(p[1])} == wanted]


def _copies(pairs):
    return [p for p in pairs if _path_of(p[0]) == _path_of(p[1])]


def _kind(key_a: str, key_b: str) -> tuple[str, str]:
    """overlay_similar's grouping of a candidate pair.

    ("copy", ext) calibrates ext's line; ("analogue", "ext_a/ext_b") is a
    same-language code pair with sorted extensions; ("cross", "") a cross-language
    code pair; ("noncode", "") has a side outside CODE_EXTENSIONS and is never emitted.
    """
    ext_a = overlay_similar._extension(_path_of(key_a))
    ext_b = overlay_similar._extension(_path_of(key_b))
    if _path_of(key_a) == _path_of(key_b):
        return ("copy", ext_a)
    if ext_a not in overlay_similar.CODE_EXTENSIONS or ext_b not in overlay_similar.CODE_EXTENSIONS:
        return ("noncode", "")
    if overlay_similar._same_language(ext_a, ext_b):
        return ("analogue", "/".join(sorted((ext_a, ext_b))))
    return ("cross", "")


def _groups(candidates, repos: list[str]) -> dict[tuple[str, str], list]:
    grouped: dict[tuple[str, str], list] = {}
    for pair in _pairs_between(candidates, repos):
        grouped.setdefault(_kind(pair[0], pair[1]), []).append(pair)
    return grouped


def _line_of(copies: list[float]) -> float | None:
    """A (repo pair, extension) group's line: None when uncalibrated, 2.0 when floored."""
    if len(copies) < overlay_similar.MIN_CALIBRATION_COPIES:
        return None
    ordered = sorted(copies)
    identity = ordered[len(ordered) // 2]
    if identity < overlay_similar.IDENTITY_FLOOR:
        return 2.0
    return identity * overlay_similar.IDENTITY_RELATIVE


def _lines_for(groups) -> dict[str, float]:
    lines = {}
    for (kind, ext), members in groups.items():
        if kind == "copy":
            line = _line_of([s for _, _, s in members])
            if line is not None:
                lines[ext] = line
    return lines


def _applied(lines: dict[str, float], ext_a: str, ext_b: str, threshold: float):
    """overlay_similar._calibrated_line restated: (applied threshold, how).

    A floored group carries 2.0 and the stricter-of rule makes it win: declined."""
    own = [lines[ext] for ext in (ext_a, ext_b) if ext in lines]
    family = next(
        (f for f in overlay_similar.LANGUAGE_FAMILIES if ext_a in f or ext_b in f), frozenset()
    )
    pooled = [line for ext, line in lines.items() if ext in family]
    if own:
        line, how = max(own), "calibrated"
    elif pooled:
        line, how = max(pooled), "family"
    else:
        return threshold, "constant"
    return line, ("declined" if line >= 2.0 else how)


def _applied_threshold(edge) -> float:
    match = re.search(r"threshold=(\d\.\d{4})", edge.evidence)
    assert match, edge.evidence
    return float(match.group(1))


def _edge_kind(edge) -> tuple[str, str]:
    return _kind(edge.source.to_key(), edge.target.to_key())


def _edge_keys(edges) -> set[tuple[str, str]]:
    return {(edge.source.to_key(), edge.target.to_key()) for edge in edges}


def _situation_edges(corpus_edges, situation, pair_label: str):
    return [
        e
        for e in _edges_between(corpus_edges, situation["repos"])
        if _edge_kind(e) == ("analogue", pair_label)
    ]


def _lsh_pairs(monkeypatch, corpus, top_k):
    # Force the bucketed path for a corpus that the exact path would otherwise take.
    monkeypatch.setattr(embed_similarity, "EXACT_MAX_ROWS", 0)
    return embed_similarity.mutual_top_k_pairs(corpus, top_k=top_k)


def _true_mutual_top_k(corpus, top_k, threshold) -> set[tuple[str, str]]:
    """Reference from the full cosine matrix, no bucketing and no heaps."""
    keys = [key for repo_id in sorted(corpus) for key in corpus[repo_id].keys]
    matrix = np.concatenate(
        [np.asarray(corpus[repo_id].normalized()) for repo_id in sorted(corpus)]
    )
    cosines = matrix @ matrix.T
    repos = np.array([_repo_of(key) for key in keys])
    cosines[repos[:, None] == repos[None, :]] = -1.0
    cosines[cosines < threshold] = -1.0
    top: list[set[int]] = []
    for row in cosines:
        order = np.argsort(-row, kind="stable")[:top_k]
        top.append({int(j) for j in order if row[j] >= threshold})
    return {
        tuple(sorted((keys[i], keys[j])))
        for i, partners in enumerate(top)
        for j in partners
        if i in top[j]
    }


def test_corpus_matches_its_recorded_shape(corpus, expected, corpus_pairs, corpus_candidates):
    """Guards the fixture itself: a regenerated corpus whose expected.json was not
    regenerated with it, or a constant that moved, fails here first."""
    assert sum(len(rv) for rv in corpus.values()) == expected["rows"]
    assert {rv.dim for rv in corpus.values()} == {expected["dim"]}
    assert len(corpus_pairs) == expected["pairs_total"]
    assert len(corpus_candidates) == expected["candidates_total"]
    assert expected["threshold"] == embed_similarity.SIMILARITY_THRESHOLD_DEFAULT
    assert expected["candidate_floor"] == overlay_similar.CANDIDATE_FLOOR
    assert expected["identity_relative"] == overlay_similar.IDENTITY_RELATIVE
    assert expected["identity_floor"] == overlay_similar.IDENTITY_FLOOR
    assert expected["min_calibration_copies"] == overlay_similar.MIN_CALIBRATION_COPIES
    assert expected["cross_language_threshold"] == overlay_similar.CROSS_LANGUAGE_THRESHOLD
    assert expected["language_families"] == [
        sorted(family) for family in overlay_similar.LANGUAGE_FAMILIES
    ]
    assert expected["code_extensions"] == sorted(overlay_similar.CODE_EXTENSIONS)


def test_corpus_pairs_hold_the_documented_invariants(corpus_pairs, expected):
    """Guards the output contract on a realistic score distribution: ordered keys,
    no duplicate pair, cross-repo only, threshold respected, at most top_k partners."""
    seen: set[frozenset] = set()
    partners: dict[str, set[str]] = {}
    for key_a, key_b, score in corpus_pairs:
        assert key_a < key_b
        assert _repo_of(key_a) != _repo_of(key_b)
        assert score >= expected["threshold"]
        pair_id = frozenset((key_a, key_b))
        assert pair_id not in seen
        seen.add(pair_id)
        partners.setdefault(key_a, set()).add(key_b)
        partners.setdefault(key_b, set()).add(key_a)
    assert max(len(others) for others in partners.values()) <= expected["top_k"]


def test_exact_path_is_the_true_mutual_top_k(corpus, corpus_pairs, expected):
    """Guards LSH recall: the default path must return exactly the mutual top-k
    of the full cosine matrix, which bucketed candidate generation cannot."""
    reference = _true_mutual_top_k(corpus, expected["top_k"], expected["threshold"])
    assert {(key_a, key_b) for key_a, key_b, _ in corpus_pairs} == reference


def test_exact_candidates_superset_of_lsh_candidates(monkeypatch, corpus, corpus_pairs, expected):
    """Guards the docstring claim that LSH only narrows which pairs are compared:
    with heaps that cannot truncate, exact finds every pair LSH finds and more."""
    rows = expected["rows"]
    exact = {(a, b) for a, b, _ in embed_similarity.mutual_top_k_pairs(corpus, top_k=rows)}
    lsh = {(a, b) for a, b, _ in _lsh_pairs(monkeypatch, corpus, top_k=rows)}
    assert exact > lsh

    # At the working top_k the superset does NOT hold: a pair LSH reports can be
    # "mutual" only because the stronger competitors were never compared. Every
    # such pair fails the true mutual test, so it is a false positive of the
    # bucketed path, not a pair the exact path lost.
    lsh_at_k = {(a, b) for a, b, _ in _lsh_pairs(monkeypatch, corpus, top_k=expected["top_k"])}
    exact_at_k = {(a, b) for a, b, _ in corpus_pairs}
    reference = _true_mutual_top_k(corpus, expected["top_k"], expected["threshold"])
    assert (lsh_at_k - exact_at_k).isdisjoint(reference)
    assert lsh_at_k - exact_at_k, "the corpus is built to show the false-mutual case"


def test_exact_path_finds_pair_lsh_provably_misses(monkeypatch, corpus, corpus_pairs, expected):
    """Guards LSH recall with one concrete pair: the two vectors hash to different
    buckets under the module's own planes, so the bucketed path never compares them."""
    key_a, key_b, score = expected["lsh_missed_pair"]
    planes = embed_similarity._planes_matrix(expected["dim"])
    signature = {}
    for repo_id in (_repo_of(key_a), _repo_of(key_b)):
        rv = corpus[repo_id]
        sigs = embed_similarity._bucket_signature_batch(rv.matrix, planes)
        signature.update(zip(rv.keys, sigs, strict=True))
    assert signature[key_a] != signature[key_b]
    assert _path_of(key_a) != _path_of(key_b), "the example is an analogue, not a copy"

    exact = {(a, b): s for a, b, s in corpus_pairs}
    assert exact[(key_a, key_b)] == pytest.approx(score, abs=1e-5)

    lsh = {(a, b) for a, b, _ in _lsh_pairs(monkeypatch, corpus, top_k=expected["top_k"])}
    assert (key_a, key_b) not in lsh

    # Recorded recall loss: every pair split across buckets is lost, and that is
    # most of them at 12 hyperplanes and this threshold.
    lost = set(exact) - lsh
    assert len(lost) == expected["pairs_split_across_lsh_buckets"]
    assert len(lost) > len(exact) / 2


def test_corpus_pairs_globally_descending_across_seed_keys(corpus_pairs, corpus_edges):
    """Guards pair order: downstream consumers read this list as a ranking, so score
    must win over key order across every seed key, in the pairs and in the edges."""
    scores = [score for _, _, score in corpus_pairs]
    assert scores == sorted(scores, reverse=True)
    by_key = sorted(corpus_pairs)
    assert [(a, b) for a, b, _ in by_key] != [(a, b) for a, b, _ in corpus_pairs], (
        "the corpus must make key order and score order differ"
    )
    assert by_key[0][2] < corpus_pairs[0][2]

    confidences = [edge.confidence for edge in corpus_edges]
    assert confidences == sorted(confidences, reverse=True)


def test_same_path_pairs_excluded_and_analogues_take_their_place(
    corpus_candidates, corpus_edges, expected
):
    """Guards the copies-crowd-out-analogues defect: a file at the same path in
    both repos never becomes an edge, and every analogue above its admission line
    is emitted although dozens of copies outrank the weakest of them."""
    assert all(edge.source.source_file != edge.target.source_file for edge in corpus_edges)
    emitted = _edge_keys(corpus_edges)

    situation = expected["situations"]["php_php_large"]
    recorded = situation["emission"][".php/.php"]
    groups = _groups(corpus_candidates, situation["repos"])
    copies = groups[("copy", ".php")]
    analogues = groups[("analogue", ".php/.php")]
    assert len(copies) == situation["calibration"][".php"]["copies"]
    assert len(analogues) == recorded["analogues"]
    assert emitted.isdisjoint((a, b) for a, b, _ in copies)
    admission = recorded["admission_threshold"]
    admitted = [(a, b) for a, b, s in analogues if s >= admission]
    assert emitted >= set(admitted)
    assert len(admitted) == recorded["emitted"]

    key_a, key_b, score = recorded["weakest_emitted_pair"]
    outranking = sum(1 for _, _, s in copies if s > score)
    assert outranking == recorded["copies_ranked_above_weakest_emitted"]
    assert outranking >= 25, "copies must be numerous enough to have taken a budget"
    assert (key_a, key_b) in emitted


def test_admission_is_calibrated_per_repo_pair_and_extension(
    corpus_candidates, corpus_edges, expected
):
    """Guards per-(repo pair, extension) calibration: every extension's line is
    identity * IDENTITY_RELATIVE of ITS copies, every emission group is admitted
    at the line the rule gives it, and in the mixed situation ts analogues are
    admitted below the php line while php analogues in the same band are rejected."""
    threshold = expected["threshold"]
    checked = 0
    for name, situation in expected["situations"].items():
        groups = _groups(corpus_candidates, situation["repos"])
        lines = _lines_for(groups)
        for ext, recorded in situation["calibration"].items():
            copies = [s for _, _, s in groups[("copy", ext)]]
            assert len(copies) == recorded["copies"], (name, ext)
            assert _line_of(copies) == pytest.approx(recorded["line"], abs=1e-6), (name, ext)
        for pair_label, recorded in situation["emission"].items():
            ext_a, ext_b = pair_label.split("/")
            applied, how = _applied(lines, ext_a, ext_b, threshold)
            assert how == recorded["admission"], (name, pair_label)
            assert applied == pytest.approx(recorded["admission_threshold"], abs=1e-6)
            analogues = groups[("analogue", pair_label)]
            edges = _situation_edges(corpus_edges, situation, pair_label)
            assert len(edges) == recorded["emitted"] == sum(1 for p in analogues if p[2] >= applied)
            if how != "calibrated" or not edges:
                continue
            checked += 1
            assert min(e.confidence for e in edges) >= round(applied, 4) - 1e-4
            rejected = [(a, b) for a, b, s in analogues if s < applied]
            assert _edge_keys(edges).isdisjoint(rejected), (name, pair_label)
    assert checked >= 6, "calibration must be exercised on several groups"

    mixed = expected["situations"]["mixed_extensions"]
    php_line = mixed["calibration"][".php"]["line"]
    ts_line = mixed["calibration"][".ts"]["line"]
    assert php_line - ts_line >= 0.04, (php_line, ts_line)
    groups = _groups(corpus_candidates, mixed["repos"])
    emitted = _edge_keys(_edges_between(corpus_edges, mixed["repos"]))
    ts_in_band = [(a, b) for a, b, s in groups[("analogue", ".ts/.ts")] if ts_line <= s < php_line]
    php_in_band = [
        (a, b) for a, b, s in groups[("analogue", ".php/.php")] if ts_line <= s < php_line
    ]
    assert ts_in_band and php_in_band, "the band between the two lines must be populated"
    assert set(ts_in_band) <= emitted
    assert emitted.isdisjoint(php_in_band)


def test_non_code_pairs_are_never_emitted_while_their_copies_still_calibrate(
    corpus_candidates, corpus_edges, expected
):
    """Guards CODE_EXTENSIONS: shared .yml and .json files still give a repo pair
    an identity level (a line, or a decline), and .json/.json analogues are never
    edges whatever that level says."""
    yml = expected["situations"]["cross_language"]["calibration"][".yml"]
    assert yml["code"] is False and yml["line"] not in (None, 2.0)
    low = expected["situations"]["low_identity"]
    json_group = low["calibration"][".json"]
    assert json_group["code"] is False
    assert json_group["copies"] >= overlay_similar.MIN_CALIBRATION_COPIES
    assert json_group["identity_level"] is not None
    groups = _groups(corpus_candidates, low["repos"])
    noncode = groups[("noncode", "")]
    assert len(noncode) == low["noncode_pairs"]["candidates"] >= 20
    assert max(s for _, _, s in noncode) > overlay_similar.CANDIDATE_FLOOR + 0.05
    assert _edge_keys(corpus_edges).isdisjoint((a, b) for a, b, _ in noncode)
    for edge in corpus_edges:
        for ref in (edge.source, edge.target):
            assert overlay_similar._extension(ref.source_file) in overlay_similar.CODE_EXTENSIONS


def test_admission_is_independent_of_repo_id_order(corpus, corpus_edges, expected):
    """Guards the stricter-of-two rule: renaming the repos so their sort order
    reverses must not change the admitted set, which the old first-key rule broke
    for .ts/.tsx pairs."""
    ids = sorted(corpus)
    rename = {old: f"z{len(ids) - i}.{old}" for i, old in enumerate(ids)}
    assert [rename[r] for r in ids] == sorted((rename[r] for r in ids), reverse=True)
    reversed_corpus = {}
    for repo_id, rv in corpus.items():
        keys = [rename[repo_id] + key[len(repo_id) :] for key in rv.keys]
        reversed_corpus[rename[repo_id]] = RepoVectors.from_rows(keys, rv.matrix)
    edges = overlay_similar.compute_similar_approach_edges(
        {}, reversed_corpus, top_k=expected["top_k"]
    )
    back = {v: k for k, v in rename.items()}

    def original(ref):
        return f"{back[ref.repo]}{_SEP}{ref.source_file}{_SEP}{ref.qualified_label}"

    mapped = {tuple(sorted((original(e.source), original(e.target)))) for e in edges}
    assert mapped == _edge_keys(corpus_edges)
    mixed_family = [e for e in corpus_edges if _edge_kind(e) == ("analogue", ".ts/.tsx")]
    assert len(mixed_family) >= 5, "the fixture must hold .ts/.tsx edges"


def test_cross_language_admission_is_flat_and_family_rule_calibrates_ts_variants(
    corpus_candidates, corpus_edges, expected
):
    """Guards the cross-language rule: code pairs across languages are admitted at
    the flat CROSS_LANGUAGE_THRESHOLD, while .ts/.tsx/.js/.vue count as one language
    and take the stricter of their two calibrated lines.

    Why the constant is as low as 0.74 and not 0.80: different syntax puts a
    ceiling on the cosine between two implementations of the same thing, so the
    value is spread through 0.70-0.80 rather than concentrated above it. Live,
    only 7 code pairs sit at 0.80 or above against 44 in [0.75, 0.80) and 176 in
    [0.70, 0.75), and the first band read by name is almost entirely real; 0.80
    was measured and rejected because it would keep seven pairs for the whole mesh.
    """
    flat = overlay_similar.CROSS_LANGUAGE_THRESHOLD
    assert overlay_similar._same_language(".ts", ".tsx")
    assert overlay_similar._same_language(".js", ".vue")
    assert overlay_similar._same_language(".mjs", ".cts")
    assert not overlay_similar._same_language(".php", ".ts")
    assert not overlay_similar._same_language(".mdx", ".tsx")

    cross_edges = [e for e in corpus_edges if _edge_kind(e) == ("cross", "")]
    assert cross_edges
    assert min(e.confidence for e in cross_edges) >= round(flat, 4)
    assert all(_applied_threshold(e) == round(flat, 4) for e in cross_edges)
    for name, situation in expected["situations"].items():
        recorded = situation["cross_language"]
        candidates = _groups(corpus_candidates, situation["repos"]).get(("cross", ""), [])
        assert len(candidates) == recorded["candidates"], name
        edges = [
            e
            for e in _edges_between(corpus_edges, situation["repos"])
            if _edge_kind(e) == ("cross", "")
        ]
        assert len(edges) == recorded["emitted"] == sum(1 for p in candidates if p[2] >= flat), name
        if recorded["first_rejected_score"] is not None:
            assert recorded["first_rejected_score"] < flat

    # Same family, different extension: the stricter of the two lines, never the
    # flat constant. The mixed situation calibrates both .ts and .tsx.
    mixed = expected["situations"]["mixed_extensions"]
    ts_line = mixed["calibration"][".ts"]["line"]
    tsx_line = mixed["calibration"][".tsx"]["line"]
    assert ts_line is not None and tsx_line is not None
    recorded = mixed["emission"][".ts/.tsx"]
    assert recorded["admission"] == "calibrated"
    assert recorded["admission_threshold"] == pytest.approx(max(ts_line, tsx_line), abs=1e-6)
    family_edges = [
        e
        for e in corpus_edges
        if _edge_kind(e)[0] == "analogue"
        and "/" in _edge_kind(e)[1]
        and _edge_kind(e)[1].split("/")[0] != _edge_kind(e)[1].split("/")[1]
    ]
    assert family_edges
    assert all(_applied_threshold(e) != round(flat, 4) for e in family_edges)


def test_uncalibrated_group_takes_family_line_before_the_constant(
    monkeypatch, corpus, corpus_candidates, corpus_edges, expected
):
    """Guards the fallback order: a (repo pair, extension) with too few copies is
    judged by the strictest line of its language family in that repo pair — a
    declined family declines it too — and the caller's constant is reached only
    when the family has no calibration at all."""
    threshold = expected["threshold"]
    tsx = expected["situations"]["tsx_tsx"]
    family = tsx["emission"][".ts/.ts"]
    assert family["admission"] == "family"
    groups = _groups(corpus_candidates, tsx["repos"])
    assert len(groups.get(("copy", ".ts"), [])) < overlay_similar.MIN_CALIBRATION_COPIES
    assert family["admission_threshold"] == pytest.approx(
        tsx["calibration"][".tsx"]["line"], abs=1e-6
    )
    assert family["admission_threshold"] != threshold
    edges = _situation_edges(corpus_edges, tsx, ".ts/.ts")
    assert len(edges) == family["emitted"] > 0
    assert family["emitted"] >= family["emitted_at_default_threshold"]
    assert all(_applied_threshold(e) == round(family["admission_threshold"], 4) for e in edges)

    low = expected["situations"]["low_identity"]
    declined_family = low["emission"][".ts/.ts"]
    assert declined_family["admission"] == "declined"
    assert declined_family["analogues"] >= 20
    assert _situation_edges(corpus_edges, low, ".ts/.ts") == []

    constant = [
        (name, label, recorded)
        for name, pair in expected["other_repo_pairs"].items()
        for label, recorded in pair["emission"].items()
        if recorded["admission"] == "constant" and recorded["emitted"]
    ]
    assert constant, "the fixture must hold a repo pair with no family calibration"
    for name, label, recorded in constant:
        pair = expected["other_repo_pairs"][name]
        lines = _lines_for(_groups(corpus_candidates, pair["repos"]))
        ext_a, ext_b = label.split("/")
        assert _applied(lines, ext_a, ext_b, threshold) == (threshold, "constant")
        edges = _situation_edges(corpus_edges, pair, label)
        assert len(edges) == recorded["emitted"]
        assert all(_applied_threshold(e) == round(threshold, 4) for e in edges)

    # With calibration impossible everywhere, every same-language edge is at the constant.
    monkeypatch.setattr(overlay_similar, "MIN_CALIBRATION_COPIES", 10**6)
    fallback = overlay_similar.compute_similar_approach_edges({}, corpus, top_k=expected["top_k"])
    same_language = [e for e in fallback if _edge_kind(e)[0] == "analogue"]
    assert same_language
    assert all(_applied_threshold(e) == round(threshold, 4) for e in same_language)
    assert min(e.confidence for e in same_language) >= round(threshold, 4)
    assert len([e for e in fallback if _edge_kind(e)[0] == "cross"]) == len(
        [e for e in corpus_edges if _edge_kind(e)[0] == "cross"]
    )
    for name, situation in expected["situations"].items():
        for label, recorded in situation["emission"].items():
            edges = _situation_edges(fallback, situation, label)
            assert len(edges) == recorded["emitted_at_default_threshold"], (name, label)


def test_evidence_string_carries_the_applied_threshold(corpus_candidates, corpus_edges, expected):
    """Guards the evidence text: each edge names the line it was admitted at — the
    group's calibrated or family line, the flat cross-language constant, or the
    caller's constant only for a group with no calibration at all."""
    threshold = expected["threshold"]
    seen = set()
    lines_by_pair: dict[tuple, dict[str, float]] = {}
    for edge in corpus_edges:
        applied = _applied_threshold(edge)
        assert edge.confidence >= applied
        seen.add(applied)
        kind, label = _edge_kind(edge)
        if kind == "cross":
            assert applied == round(overlay_similar.CROSS_LANGUAGE_THRESHOLD, 4)
            continue
        repo_pair = tuple(sorted((edge.source.repo, edge.target.repo)))
        if repo_pair not in lines_by_pair:
            lines_by_pair[repo_pair] = _lines_for(_groups(corpus_candidates, list(repo_pair)))
        ext_a, ext_b = label.split("/")
        expected_line, _ = _applied(lines_by_pair[repo_pair], ext_a, ext_b, threshold)
        assert applied == round(expected_line, 4), edge.evidence
    assert round(overlay_similar.CROSS_LANGUAGE_THRESHOLD, 4) in seen
    assert (
        len(seen - {round(overlay_similar.CROSS_LANGUAGE_THRESHOLD, 4), round(threshold, 4)}) >= 4
    )


def _share_by_extension_pair(candidates, admission_for) -> dict[str, tuple[int, int]]:
    grouped: dict[tuple, list] = {}
    for key_a, key_b, score in candidates:
        repo_pair = tuple(sorted((_repo_of(key_a), _repo_of(key_b))))
        grouped.setdefault((repo_pair, *_kind(key_a, key_b)), []).append((key_a, key_b, score))
    lines_by_pair: dict[tuple, dict[str, float]] = {}
    for (repo_pair, kind, ext), members in grouped.items():
        if kind == "copy":
            line = _line_of([s for _, _, s in members])
            if line is not None:
                lines_by_pair.setdefault(repo_pair, {})[ext] = line
    shares: dict[str, list[int]] = {}
    for (repo_pair, kind, _label), members in grouped.items():
        if kind not in ("analogue", "cross"):
            continue
        for key_a, key_b, score in members:
            ext_a = overlay_similar._extension(_path_of(key_a))
            ext_b = overlay_similar._extension(_path_of(key_b))
            applied = admission_for(kind, lines_by_pair.get(repo_pair, {}), ext_a, ext_b)
            entry = shares.setdefault("/".join(sorted((ext_a, ext_b))), [0, 0])
            entry[0] += 1
            entry[1] += int(score >= applied)
    return {label: (available, admitted) for label, (available, admitted) in shares.items()}


def test_calibration_removes_the_language_bias_in_admitted_share(corpus_candidates, expected):
    """Guards the point of the change: at the absolute threshold .php/.ts admitted
    under 5% of its analogues and .ts/.tsx under 3%; calibrated, the code pairs
    .php/.php, .tsx/.tsx, .ts/.ts, .ts/.tsx and .php/.ts each admit at least 10%
    and sit within a factor of 4 of each other (live reference: 16.7 / 13.5 /
    16.4 / 6.0 / 17.0 percent, against 11.7 / 10.8 / 0.1 / 0.4 / 0.0 before)."""
    threshold = expected["threshold"]

    def calibrated(kind, lines, ext_a, ext_b):
        if kind == "cross":
            return overlay_similar.CROSS_LANGUAGE_THRESHOLD
        return _applied(lines, ext_a, ext_b, threshold)[0]

    after = _share_by_extension_pair(corpus_candidates, calibrated)
    before = _share_by_extension_pair(corpus_candidates, lambda *_: threshold)
    for label, (available, admitted) in after.items():
        recorded = expected["share_by_extension_pair"][label]
        assert (available, admitted) == (recorded["available"], recorded["admitted"]), label

    code_pairs = (".php/.php", ".tsx/.tsx", ".ts/.ts", ".ts/.tsx", ".php/.ts")
    before_share = {label: before[label][1] / before[label][0] for label in code_pairs}
    after_share = {label: after[label][1] / after[label][0] for label in code_pairs}
    assert before_share[".php/.ts"] < 0.05, before_share
    assert before_share[".ts/.tsx"] < 0.03, before_share
    for label, value in after_share.items():
        assert value >= 0.10, (label, value)
    assert max(after_share.values()) / min(after_share.values()) <= 4.0, after_share


def test_group_below_identity_floor_is_declined_at_the_shipped_floor(
    corpus, corpus_edges, expected, caplog
):
    """Guards the floor on a real case: the low_identity pair's .tsx and .json
    copies sit at identity ~0.76, under IDENTITY_FLOOR, so every group of that
    pair — its family-judged .ts analogues included — emits nothing, and each
    floored (pair, extension) is logged once by name."""
    low = expected["situations"]["low_identity"]
    floored = {ext for ext, recorded in low["calibration"].items() if recorded["line"] == 2.0}
    assert {".tsx", ".json"} <= floored
    assert all(
        recorded["identity_level"] < overlay_similar.IDENTITY_FLOOR
        for ext, recorded in low["calibration"].items()
        if ext in floored
    )
    assert low["emitted"] == 0
    assert all(r["admission"] == "declined" for r in low["emission"].values())
    assert sum(r["analogues"] for r in low["emission"].values()) >= 80
    assert _edges_between(corpus_edges, low["repos"]) == []

    with caplog.at_level("INFO", logger=overlay_similar.__name__):
        overlay_similar.compute_similar_approach_edges({}, corpus, top_k=expected["top_k"])
    messages = [r.getMessage() for r in caplog.records if "IDENTITY_FLOOR" in r.getMessage()]
    name = "<->".join(sorted(low["repos"]))
    for ext in floored:
        assert sum(1 for m in messages if name in m and f"({ext})" in m) == 1, ext
    calibrated_elsewhere = {
        (s["repos"][0], s["repos"][1], ext)
        for s in expected["situations"].values()
        for ext, r in s["calibration"].items()
        if r["line"] not in (None, 2.0)
    }
    for repo_a, repo_b, ext in calibrated_elsewhere:
        assert not any(f"{repo_a}<->{repo_b}" in m and f"({ext})" in m for m in messages)


def test_identity_floor_sits_between_recorded_groups(expected):
    """Guards the most consequential constant: the highest declined identity and
    the lowest calibrated identity in the fixture bracket IDENTITY_FLOOR within
    0.05 on each side, so moving it either way flips a recorded group."""
    declined, calibrated = [], []
    for situation in expected["situations"].values():
        for ext, recorded in situation["calibration"].items():
            if recorded["line"] is None:
                continue
            (declined if recorded["line"] == 2.0 else calibrated).append(
                (recorded["identity_level"], ext)
            )
    floor = overlay_similar.IDENTITY_FLOOR
    highest_declined = max(declined)
    lowest_calibrated = min(calibrated)
    assert highest_declined[0] < floor <= lowest_calibrated[0], (
        highest_declined,
        lowest_calibrated,
    )
    assert floor - highest_declined[0] <= 0.05, highest_declined
    assert lowest_calibrated[0] - floor <= 0.05, lowest_calibrated
    assert expected["identity_floor"] == floor


def test_every_group_is_declined_when_the_floor_exceeds_every_identity(
    monkeypatch, corpus, corpus_candidates, expected, caplog
):
    """Guards the floor's reach: raised over every fixture identity level, it
    declines every calibrated and family-judged group, logs each once, and leaves
    only groups with no family calibration at all and cross-language pairs."""
    monkeypatch.setattr(overlay_similar, "IDENTITY_FLOOR", 0.95)
    with caplog.at_level("INFO", logger=overlay_similar.__name__):
        edges = overlay_similar.compute_similar_approach_edges({}, corpus, top_k=expected["top_k"])

    floored = [r.getMessage() for r in caplog.records if "IDENTITY_FLOOR" in r.getMessage()]
    groups_seen = 0
    for situation in expected["situations"].values():
        name = "<->".join(sorted(situation["repos"]))
        for ext, recorded in situation["calibration"].items():
            if recorded["line"] is None:
                continue
            groups_seen += 1
            assert sum(1 for m in floored if name in m and f"({ext})" in m) == 1, (name, ext)
        for label, recorded in situation["emission"].items():
            if recorded["admission"] != "constant":
                assert _situation_edges(edges, situation, label) == [], (name, label)
    assert groups_seen >= 7

    for edge in edges:
        kind, label = _edge_kind(edge)
        if kind == "cross":
            continue
        repo_pair = tuple(sorted((edge.source.repo, edge.target.repo)))
        lines = _lines_for(_groups(corpus_candidates, list(repo_pair)))
        ext_a, ext_b = label.split("/")
        assert _applied(lines, ext_a, ext_b, expected["threshold"])[1] == "constant", edge.evidence


def test_candidate_floor_leaves_room_below_every_admission(
    corpus_candidates, corpus_edges, expected
):
    """Guards the floor-equals-admission defect: no calibrated line sits at
    CANDIDATE_FLOOR, so a group just above IDENTITY_FLOOR still rejects part of
    its candidates instead of admitting 98% of them as it did at a 0.69 floor."""
    floor = overlay_similar.CANDIDATE_FLOOR
    lowest = overlay_similar.IDENTITY_FLOOR * overlay_similar.IDENTITY_RELATIVE
    assert lowest - floor >= 0.05, (lowest, floor)
    assert min(e.confidence for e in corpus_edges) > floor + 0.05

    lines = []
    near_floor_checked = 0
    for name, situation in expected["situations"].items():
        groups = _groups(corpus_candidates, situation["repos"])
        for ext, recorded in situation["calibration"].items():
            if recorded["line"] in (None, 2.0):
                continue
            lines.append(recorded["line"])
            assert recorded["line"] - floor >= 0.05, (name, ext, recorded["line"])
            if recorded["identity_level"] >= overlay_similar.IDENTITY_FLOOR + 0.03:
                continue
            analogues = groups.get(("analogue", f"{ext}/{ext}"), [])
            if len(analogues) < 20:
                continue
            near_floor_checked += 1
            rejected = sum(1 for _, _, s in analogues if s < recorded["line"])
            assert rejected >= len(analogues) / 10, (name, ext, rejected, len(analogues))
            assert situation["emission"][f"{ext}/{ext}"]["admitted_share"] < 0.9, (name, ext)
    assert near_floor_checked >= 1, "a near-floor code group must be present"
    assert min(lines) < overlay_similar.IDENTITY_FLOOR


def test_cap_never_binds_on_the_fixture_and_every_admitted_analogue_is_emitted(
    corpus, corpus_edges, expected, caplog
):
    """Guards silent loss of analogues: at the shipped MAX_EDGES_PER_REPO_PAIR no
    fixture repo pair is truncated and no cap warning is logged."""
    with caplog.at_level("WARNING", logger=overlay_similar.__name__):
        edges = overlay_similar.compute_similar_approach_edges({}, corpus, top_k=expected["top_k"])
    assert _edge_keys(edges) == _edge_keys(corpus_edges)
    assert "MAX_EDGES_PER_REPO_PAIR" not in caplog.text
    for name, situation in expected["situations"].items():
        assert len(_edges_between(corpus_edges, situation["repos"])) == situation["emitted"], name
        assert situation["emitted"] < overlay_similar.MAX_EDGES_PER_REPO_PAIR


def test_cap_binds_when_lowered_and_drops_the_weakest_pairs(
    monkeypatch, corpus, corpus_edges, expected, caplog
):
    """Guards the safety valve: a lowered cap keeps exactly the highest-scoring
    pairs per repo pair and warns once per repo pair, naming it and the score
    where dropping started."""
    cap = 10
    monkeypatch.setattr(overlay_similar, "MAX_EDGES_PER_REPO_PAIR", cap)
    with caplog.at_level("WARNING", logger=overlay_similar.__name__):
        capped = overlay_similar.compute_similar_approach_edges({}, corpus, top_k=expected["top_k"])

    def by_repo_pair(edges):
        grouped: dict[frozenset, list] = {}
        for edge in edges:
            grouped.setdefault(frozenset((edge.source.repo, edge.target.repo)), []).append(edge)
        return grouped

    full = by_repo_pair(corpus_edges)
    truncated = by_repo_pair(capped)
    assert truncated.keys() == full.keys()
    bound = {pair for pair, edges in full.items() if len(edges) > cap}
    assert len(bound) >= 3, "the fixture must hold several repo pairs above the lowered cap"
    for pair, edges in full.items():
        kept = truncated[pair]
        assert kept == edges[: min(cap, len(edges))]
        if pair in bound:
            assert len(kept) == cap
            assert min(e.confidence for e in kept) >= max(e.confidence for e in edges[cap:])

    warnings = [r for r in caplog.records if "MAX_EDGES_PER_REPO_PAIR" in r.getMessage()]
    assert len(warnings) == len(bound)
    for pair in bound:
        name = "<->".join(sorted(pair))
        matching = [r.getMessage() for r in warnings if name in r.getMessage()]
        assert len(matching) == 1, name
        first_dropped = full[pair][cap].confidence
        assert f"{first_dropped:.4f}" in matching[0]


def test_php_repo_vs_tsx_repo_shares_config_copies_and_code_pairs_are_cross_language(
    corpus_candidates, corpus_edges, expected
):
    """Guards the cross-language situation: a php repo and a tsx repo share only
    config copies, so no code line is calibrated, and its code pairs are judged by
    the flat constant alone — which none of them reaches, so the pair ends with
    zero edges although it has cross-language candidates."""
    situation = expected["situations"]["cross_language"]
    groups = _groups(corpus_candidates, situation["repos"])
    assert not any(kind == "analogue" for kind, _ in groups)
    assert situation["emission"] == {}
    assert all(not entry["code"] for entry in situation["calibration"].values())
    assert situation["calibration"]
    cross = groups[("cross", "")]
    recorded = situation["cross_language"]
    assert len(cross) == recorded["candidates"] >= 10
    assert max(s for _, _, s in cross) == pytest.approx(recorded["best_score"], abs=1e-6)
    assert recorded["best_score"] < overlay_similar.CROSS_LANGUAGE_THRESHOLD
    assert recorded["emitted"] == situation["emitted"] == 0
    assert _edges_between(corpus_edges, situation["repos"]) == []


def test_score_distribution_differs_by_language(corpus_edges, expected):
    """Guards the gap calibration exists for: identical php scores higher than
    identical tsx, so php's identity level and admission line sit above tsx's and
    php analogues reach higher scores; one absolute threshold would read the two
    scales as one."""
    php = expected["situations"]["php_php_large"]
    tsx = expected["situations"]["tsx_tsx"]
    php_edges = _situation_edges(corpus_edges, php, ".php/.php")
    tsx_edges = _situation_edges(corpus_edges, tsx, ".tsx/.tsx")
    php_best = max(e.confidence for e in php_edges)
    tsx_best = max(e.confidence for e in tsx_edges)

    assert php_best == round(php["emission"][".php/.php"]["best_analogue_score"], 4)
    assert tsx_best == round(tsx["emission"][".tsx/.tsx"]["best_analogue_score"], 4)
    assert tsx_best < php_best
    assert (
        php["calibration"][".php"]["identity_level"] - tsx["calibration"][".tsx"]["identity_level"]
        >= 0.03
    )
    assert php["calibration"][".php"]["line"] - tsx["calibration"][".tsx"]["line"] >= 0.03
    assert min(e.confidence for e in tsx_edges) < min(e.confidence for e in php_edges)
