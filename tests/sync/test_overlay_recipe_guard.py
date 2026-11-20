"""Pins the source of the code that turns two repos' vectors into cached pair
candidates and their cache key. With the stage cache on, a repo pair whose
vectors did not change is served from disk, so a change here is invisible
until PAIR_CANDIDATES_RECIPE is bumped."""

from __future__ import annotations

import hashlib
import inspect

from graphify_mesh.sync import embed_similarity, embedding, overlay_pair_cache, vectors
from graphify_mesh.sync.vectors import RepoVectors

GUARDED = (
    # Candidate computation, per-row reduction and the payload round trip.
    embed_similarity._pair_candidates,
    embed_similarity._top_k_per_row,
    embed_similarity._concat_top_k,
    embed_similarity.PairCandidates,
    embed_similarity._push_candidates,
    # The normalized rows every score is computed from.
    RepoVectors.normalized,
    vectors._NormalizedRows,
    # The key and the per-repo identity it is built on.
    overlay_pair_cache.pair_key,
    embedding.shard_matrix_sha256,
)
GUARDED_VALUES = (embed_similarity.PAIR_CANDIDATES_RECIPE,)
PINNED_SOURCE_SHA256 = "c3a62393cc235a265177186f0c958732867162c43d72f4c85ed561291d3f1bb3"


def _source_sha256() -> str:
    h = hashlib.sha256()
    for obj in GUARDED:
        h.update(inspect.getsource(obj).encode("utf-8"))
        h.update(b"\0")
    for value in GUARDED_VALUES:
        h.update(repr(value).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def test_overlay_pair_candidates_source_is_pinned():
    actual = _source_sha256()
    assert actual == PINNED_SOURCE_SHA256, (
        "code that shapes cached overlay pair candidates or their key changed. "
        "A comment- or docstring-only change: update PINNED_SOURCE_SHA256 only. "
        "A semantic change: bump PAIR_CANDIDATES_RECIPE in embed_similarity.py "
        "(invalidates every cached pair) and update PINNED_SOURCE_SHA256. "
        f"New value: {actual!r}"
    )
