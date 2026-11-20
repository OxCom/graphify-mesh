"""Per-repo-pair candidate cache for the overlay `similar_approach` exact path.

An entry holds one repo pair's `PairCandidates` under
`<cache_dir>/overlay/<key>.bin`. The key is derived from both repos' matrix
identity (`shard_matrix_sha256` of the vectors themselves), top_k, the
effective threshold and the code recipe, so an entry is either valid for this
run or never looked up. Writes are spooled through `PendingCacheWrites` and
land only after publish.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

from graphify_mesh.sync import embed_similarity
from graphify_mesh.sync.embed_similarity import PairCandidates
from graphify_mesh.sync.embedding import shard_matrix_sha256
from graphify_mesh.sync.stage_cache import (
    CACHE_FORMAT,
    PendingCacheWrites,
    gc_entries,
    read_entry,
)
from graphify_mesh.sync.vectors import RepoVectors

log = logging.getLogger(__name__)

OVERLAY_CACHE_SUBDIR = "overlay"


def pair_key(matrix_sha256_a: str, matrix_sha256_b: str, top_k: int, threshold: float) -> str:
    # Chunk size and numpy version are in the key: BLAS block shape can move the
    # last bit of a score, and a cached score must equal the one a cold run computes.
    parts = (
        embed_similarity.PAIR_CANDIDATES_RECIPE,
        str(CACHE_FORMAT),
        matrix_sha256_a,
        matrix_sha256_b,
        str(int(top_k)),
        repr(float(threshold)),
        str(embed_similarity.BUCKET_CHUNK_ROWS),
        np.__version__,
    )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


class _BoundPairCache:
    def __init__(
        self,
        owner: OverlayPairCache,
        sha_by_repo: dict[str, str],
        rows_by_repo: dict[str, int],
        top_k: int,
        threshold: float,
    ) -> None:
        self._owner = owner
        self._sha_by_repo = sha_by_repo
        self._rows_by_repo = rows_by_repo
        self._top_k = top_k
        self._threshold = threshold

    def _key(self, repo_a: str, repo_b: str) -> str:
        return pair_key(
            self._sha_by_repo[repo_a], self._sha_by_repo[repo_b], self._top_k, self._threshold
        )

    def read(self, repo_a: str, repo_b: str) -> PairCandidates | None:
        key = self._key(repo_a, repo_b)
        self._owner.used_keys.add(key)
        payload = read_entry(self._owner.dir, key)
        if payload is None:
            self._owner.misses += 1
            return None
        candidates = PairCandidates.from_bytes(payload)
        if candidates is None:
            log.debug("overlay: pair cache entry %s unreadable, recomputing", key)
            self._owner.misses += 1
            return None
        # The caller re-checks fits() too; checking here keeps hits honest.
        if not candidates.fits(self._rows_by_repo[repo_a], self._rows_by_repo[repo_b], self._top_k):
            log.debug("overlay: pair cache entry %s does not fit, recomputing", key)
            self._owner.misses += 1
            return None
        self._owner.hits += 1
        return candidates

    def stage(self, repo_a: str, repo_b: str, candidates: PairCandidates) -> None:
        key = self._key(repo_a, repo_b)
        self._owner.used_keys.add(key)
        payload = candidates.to_bytes()
        self._owner.staged_bytes += len(payload)
        self._owner.pending.add(self._owner.dir, key, payload)


class OverlayPairCache:
    """Cache source handed to `mutual_top_k_pairs`.

    Each repo's identity is hashed from the very vectors handed to overlay
    (keys + matrix), never taken from shard meta, so a key cannot describe rows
    other than the ones compared. After publish and `pending.flush()`, call
    `collect_garbage()` to drop entries this run did not use.
    """

    def __init__(self, cache_dir: Path, pending: PendingCacheWrites) -> None:
        self.dir = cache_dir / OVERLAY_CACHE_SUBDIR
        self.pending = pending
        self.used_keys: set[str] = set()
        self.consulted = False
        self.hits = 0
        self.misses = 0
        self.staged_bytes = 0

    def bind(
        self, vectors_by_repo: dict[str, RepoVectors], top_k: int, threshold: float
    ) -> _BoundPairCache:
        sha_by_repo = {
            repo_id: shard_matrix_sha256(vectors) for repo_id, vectors in vectors_by_repo.items()
        }
        rows_by_repo = {repo_id: len(vectors.keys) for repo_id, vectors in vectors_by_repo.items()}
        self.consulted = True
        return _BoundPairCache(self, sha_by_repo, rows_by_repo, top_k, threshold)

    def collect_garbage(self) -> int:
        """Remove entries this run did not read or stage. A run that never
        reached the exact path (LSH, no vectors) keeps the cache as it is."""
        if not self.consulted:
            return 0
        return gc_entries(self.dir, self.used_keys)
