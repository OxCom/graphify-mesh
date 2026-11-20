"""Per-repo-pair candidate cache for the overlay exact path (stage caches, Task 3).

The equivalence tests compare against `_legacy_exact_pairs`, the exact path as
it was before the cache: one global generation counter as the heap tie-break.
The fixture is built from exactly representable vectors so equal scores are
real ties, there are more tied candidates than top_k across several repos, and
the repo that changes size sorts first, so every cached pair's global offsets
shift between the two runs.
"""

from __future__ import annotations

import dataclasses
import gc
import heapq
import weakref

import numpy as np
import pytest

from graphify_mesh.sync import embed_similarity, overlay_similar
from graphify_mesh.sync.embed_similarity import PairCandidates
from graphify_mesh.sync.overlay_pair_cache import OVERLAY_CACHE_SUBDIR, OverlayPairCache
from graphify_mesh.sync.stage_cache import PendingCacheWrites, read_entry, write_entry
from graphify_mesh.sync.vectors import RepoVectors

TOP_K = 2
THRESHOLD = 0.5

E0 = [1.0, 0.0, 0.0, 0.0]
E1 = [0.0, 1.0, 0.0, 0.0]
DIAG = [1.0, 1.0, 0.0, 0.0]
E2 = [0.0, 0.0, 1.0, 0.0]


def _corpus(grow_a: bool = False, change_c: bool = False) -> dict[str, RepoVectors]:
    repo_a = {"a/k1": E0, "a/k2": E0, "a/k3": E1}
    if grow_a:
        # Sorts before every other row of repo.a, so every row after it shifts too.
        repo_a["a/k0"] = E0
    repo_c = {"c/k0": E0, "c/k1": DIAG, "c/k2": E1}
    if change_c:
        repo_c["c/k1"] = E2
    return {
        "repo.a": RepoVectors.from_mapping(repo_a),
        "repo.b": RepoVectors.from_mapping({"b/k0": E0, "b/k1": E0, "b/k2": DIAG, "b/k3": E1}),
        "repo.c": RepoVectors.from_mapping(repo_c),
        "repo.d": RepoVectors.from_mapping({"d/k0": E0, "d/k1": E1, "d/k2": DIAG, "d/k3": E2}),
    }


def _legacy_exact_pairs(
    vectors_by_repo: dict[str, RepoVectors], top_k: int, threshold: float
) -> list[tuple[str, str, float]]:
    """The pre-cache exact path, verbatim but for the setup lines."""
    repo_ids = sorted(repo_id for repo_id, rv in vectors_by_repo.items() if len(rv))
    keys = [key for repo_id in repo_ids for key in vectors_by_repo[repo_id].keys]
    offsets: dict[str, int] = {}
    cursor = 0
    for repo_id in repo_ids:
        offsets[repo_id] = cursor
        cursor += len(vectors_by_repo[repo_id].keys)
    chunk = embed_similarity.BUCKET_CHUNK_ROWS

    candidates: dict[str, list[tuple[float, int, str]]] = {key: [] for key in keys}
    next_rank = 0
    for index_a, repo_a in enumerate(repo_ids):
        rows_a = vectors_by_repo[repo_a].normalized()
        count_a = len(vectors_by_repo[repo_a].keys)
        base_a = offsets[repo_a]
        for repo_b in repo_ids[index_a + 1 :]:
            rows_b = vectors_by_repo[repo_b].normalized()
            count_b = len(vectors_by_repo[repo_b].keys)
            base_b = offsets[repo_b]
            for start_a in range(0, count_a, chunk):
                end_a = min(start_a + chunk, count_a)
                block_a = rows_a[start_a:end_a]
                for start_b in range(0, count_b, chunk):
                    end_b = min(start_b + chunk, count_b)
                    sims = block_a @ rows_b[start_b:end_b].T
                    hit_rows, hit_cols = np.nonzero(sims >= threshold)
                    for local_row, local_col in zip(hit_rows, hit_cols, strict=True):
                        key_a = keys[base_a + start_a + int(local_row)]
                        key_b = keys[base_b + start_b + int(local_col)]
                        score = float(sims[local_row, local_col])
                        rank = next_rank
                        next_rank += 1
                        for owner_key, other_key in ((key_a, key_b), (key_b, key_a)):
                            heap = candidates[owner_key]
                            entry = (score, -rank, other_key)
                            if len(heap) < top_k:
                                heapq.heappush(heap, entry)
                            elif heap and entry > heap[0]:
                                heapq.heapreplace(heap, entry)
    return embed_similarity._mutual_from_candidates(candidates)


def _run(vectors_by_repo, cache=None):
    return embed_similarity.mutual_top_k_pairs(
        vectors_by_repo, top_k=TOP_K, threshold=THRESHOLD, pair_cache=cache
    )


def _cache(tmp_path) -> OverlayPairCache:
    return OverlayPairCache(tmp_path, PendingCacheWrites(tmp_path))


def _cache_files(tmp_path) -> set[str]:
    cache_dir = tmp_path / OVERLAY_CACHE_SUBDIR
    return {path.name for path in cache_dir.iterdir()} if cache_dir.is_dir() else set()


@pytest.fixture(params=[1024, 2], ids=["one-chunk", "multi-chunk"])
def chunk_rows(request, monkeypatch):
    monkeypatch.setattr(embed_similarity, "BUCKET_CHUNK_ROWS", request.param)
    return request.param


def test_fixture_has_real_ties_beyond_top_k():
    # Guards the fixture itself: without exact ties the tie-break is never exercised.
    vectors = _corpus()
    owner = vectors["repo.a"].normalized()[0]
    tied = [
        key
        for repo_id in ("repo.b", "repo.c", "repo.d")
        for key, row in zip(vectors[repo_id].keys, vectors[repo_id].normalized()[:], strict=True)
        if float(np.dot(owner, row)) == 1.0
    ]
    assert len(tied) > TOP_K
    assert len({key.split("/")[0] for key in tied}) >= 2


def test_cache_on_and_off_match_legacy_across_row_count_change(tmp_path, chunk_rows):
    first = _corpus()
    cache = _cache(tmp_path)
    assert _run(first, cache) == _legacy_exact_pairs(first, TOP_K, THRESHOLD)
    cache.pending.flush()

    second = _corpus(grow_a=True)
    warm = _cache(tmp_path)
    cached = _run(second, warm)
    uncached = _run(second)
    legacy = _legacy_exact_pairs(second, TOP_K, THRESHOLD)

    assert warm.hits == 3, "pairs without repo.a must come from the cache"
    assert cached == uncached == legacy
    assert cached, "fixture must produce pairs"


def test_changed_repo_recomputes_only_its_pairs(tmp_path, monkeypatch):
    computed: list[tuple[int, int]] = []
    real = embed_similarity._pair_candidates

    def counting(rv_a, rv_b, top_k, threshold):
        computed.append((len(rv_a), len(rv_b)))
        return real(rv_a, rv_b, top_k, threshold)

    monkeypatch.setattr(embed_similarity, "_pair_candidates", counting)

    cold = _cache(tmp_path)
    _run(_corpus(), cold)
    assert (cold.hits, cold.misses, len(computed)) == (0, 6, 6)
    cold.pending.flush()

    computed.clear()
    warm = _cache(tmp_path)
    changed = _corpus(change_c=True)
    result = _run(changed, warm)
    assert (warm.hits, warm.misses, len(computed)) == (3, 3, 3)
    assert result == _legacy_exact_pairs(changed, TOP_K, THRESHOLD)

    # Pairs touching repo.c were re-keyed: flush adds 3, GC drops the 3 stale ones.
    warm.pending.flush()
    assert len(_cache_files(tmp_path)) == 9
    assert warm.collect_garbage() == 3
    assert _cache_files(tmp_path) == {f"{key}.bin" for key in warm.used_keys}


def test_cache_off_reads_and_writes_nothing(tmp_path):
    vectors = _corpus()
    assert _run(vectors) == _legacy_exact_pairs(vectors, TOP_K, THRESHOLD)
    assert _cache_files(tmp_path) == set()


def test_lsh_path_uses_no_pair_cache(tmp_path, monkeypatch, caplog):
    seeded = _cache(tmp_path)
    _run(_corpus(), seeded)
    seeded.pending.flush()
    before = _cache_files(tmp_path)

    monkeypatch.setattr(embed_similarity, "EXACT_MAX_ROWS", 1)
    lsh = _cache(tmp_path)
    with caplog.at_level("INFO"):
        _run(_corpus(grow_a=True), lsh)

    assert "overlay: LSH path, pair cache not used" in caplog.text
    assert (len(lsh.pending), lsh.hits, lsh.misses, lsh.consulted) == (0, 0, 0, False)
    assert lsh.collect_garbage() == 0
    assert _cache_files(tmp_path) == before


def test_corrupt_entry_is_a_recompute(tmp_path):
    cold = _cache(tmp_path)
    vectors = _corpus()
    expected = _run(vectors, cold)
    cold.pending.flush()
    cache_dir = tmp_path / OVERLAY_CACHE_SUBDIR
    for path in cache_dir.iterdir():
        path.write_bytes(b"garbage")

    warm = _cache(tmp_path)
    assert _run(vectors, warm) == expected
    assert (warm.hits, warm.misses) == (0, 6)


def test_readable_entry_that_does_not_fit_counts_as_a_miss(tmp_path):
    cold = _cache(tmp_path)
    vectors = _corpus()
    expected = _run(vectors, cold)
    cold.pending.flush()
    # A well-formed entry for 99 rows: no pair in the fixture has that many.
    wrong = dataclasses.replace(
        embed_similarity._pair_candidates(vectors["repo.a"], vectors["repo.b"], TOP_K, THRESHOLD),
        count_a=99,
    ).to_bytes()
    cache_dir = tmp_path / OVERLAY_CACHE_SUBDIR
    for path in list(cache_dir.iterdir()):
        write_entry(cache_dir, path.name.removesuffix(".bin"), wrong)

    warm = _cache(tmp_path)
    assert _run(vectors, warm) == expected
    assert (warm.hits, warm.misses) == (0, 6)


def test_entry_that_does_not_fit_is_recomputed():
    pair = embed_similarity._pair_candidates(
        _corpus()["repo.a"], _corpus()["repo.b"], TOP_K, THRESHOLD
    )
    restored = PairCandidates.from_bytes(pair.to_bytes())
    assert restored is not None
    assert restored.fits(3, 4, TOP_K)
    assert not restored.fits(4, 4, TOP_K)
    assert not restored.fits(3, 4, 1)
    for name in ("a_row", "a_other", "a_score", "b_row", "b_other", "b_score"):
        assert np.array_equal(getattr(restored, name), getattr(pair, name))
        assert getattr(restored, name).dtype == getattr(pair, name).dtype
    assert PairCandidates.from_bytes(b"not an npz") is None


def test_key_depends_on_every_input(tmp_path):
    vectors = _corpus()
    base = _cache(tmp_path).bind(vectors, TOP_K, 0.6)._key("repo.a", "repo.b")
    assert base != _cache(tmp_path).bind(vectors, TOP_K + 1, 0.6)._key("repo.a", "repo.b")
    assert base != _cache(tmp_path).bind(vectors, TOP_K, 0.61)._key("repo.a", "repo.b")
    assert base != _cache(tmp_path).bind(_corpus(grow_a=True), TOP_K, 0.6)._key("repo.a", "repo.b")
    # Identity comes from the vectors alone, so an equal corpus gives the same key.
    assert base == _cache(tmp_path).bind(_corpus(), TOP_K, 0.6)._key("repo.a", "repo.b")


class _CountingPairCache:
    """Holds only payload bytes, so any PairCandidates still alive is the caller's."""

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], bytes] = {}
        self.refs: list[weakref.ref] = []
        self.max_alive = 0

    def bind(self, vectors_by_repo, top_k, threshold):
        return self

    def _check(self) -> None:
        gc.collect()
        alive = sum(1 for ref in self.refs if ref() is not None)
        self.max_alive = max(self.max_alive, alive)

    def read(self, repo_a, repo_b):
        self._check()
        payload = self.store.get((repo_a, repo_b))
        if payload is None:
            return None
        pair = PairCandidates.from_bytes(payload)
        self.refs.append(weakref.ref(pair))
        return pair

    def stage(self, repo_a, repo_b, candidates):
        self.refs.append(weakref.ref(candidates))
        self._check()
        self.store[(repo_a, repo_b)] = candidates.to_bytes()


def test_only_one_pair_alive_at_a_time():
    stub = _CountingPairCache()
    vectors = _corpus()
    cold = _run(vectors, stub)
    assert len(stub.store) == 6
    warm = _run(vectors, stub)
    gc.collect()

    assert cold == warm
    # At a read, the previous pair is gone; at a stage, only the pair being staged lives.
    assert stub.max_alive == 1
    assert all(ref() is None for ref in stub.refs)


def test_cache_entries_land_only_on_flush(tmp_path):
    cache = _cache(tmp_path)
    _run(_corpus(), cache)
    assert _cache_files(tmp_path) == set()
    assert len(cache.pending) == 6
    spool = cache.pending.staging_dir
    assert spool is not None and len(list(spool.iterdir())) == 6
    cache.pending.flush()
    assert not spool.exists()
    for key in cache.used_keys:
        assert read_entry(tmp_path / OVERLAY_CACHE_SUBDIR, key) is not None


def _durable(vectors_by_repo: dict[str, RepoVectors]) -> dict[str, RepoVectors]:
    # Durable keys with distinct code paths, so calibration and the code filter apply.
    out = {}
    for repo_id, rv in vectors_by_repo.items():
        mapping = {}
        for key, row in zip(rv.keys, rv.matrix, strict=True):
            name = key.replace("/", "_")
            mapping[f"{repo_id}\x1fsrc/{name}.py\x1f{name}"] = row
        out[repo_id] = RepoVectors.from_mapping(mapping)
    return out


def test_overlay_edges_identical_with_cache_on_and_off(tmp_path):
    def edges(vectors, cache):
        return overlay_similar.compute_similar_approach_edges(
            {}, embedding_vectors_by_repo=vectors, top_k=TOP_K, pair_cache=cache
        )

    cold = _cache(tmp_path)
    edges(_durable(_corpus()), cold)
    cold.pending.flush()

    grown = _durable(_corpus(grow_a=True))
    warm = _cache(tmp_path)
    cached = edges(grown, warm)
    uncached = edges(grown, None)

    assert warm.hits == 3
    assert cached
    assert [repr(edge) for edge in cached] == [repr(edge) for edge in uncached]


def test_entry_that_does_not_fit_is_recomputed_in_the_run():
    vectors = _corpus()
    # No repo in the fixture has 99 rows, so this entry fits no pair.
    wrong = dataclasses.replace(
        embed_similarity._pair_candidates(vectors["repo.a"], vectors["repo.b"], TOP_K, THRESHOLD),
        count_a=99,
    )

    class _Stale:
        staged = 0

        def bind(self, vectors_by_repo, top_k, threshold):
            return self

        def read(self, repo_a, repo_b):
            return wrong

        def stage(self, repo_a, repo_b, candidates):
            self.staged += 1

    stale = _Stale()
    assert _run(vectors, stale) == _legacy_exact_pairs(vectors, TOP_K, THRESHOLD)
    assert stale.staged == 6
