"""Per-repo partial cache for the lexical index.

An entry holds one repo's `RepoLexicalPartial` (local doc ids, JSON payload)
under `<cache_dir>/lexical/<key>.bin`. The key covers everything a partial is
derived from: the repo id and source root, `graph.json` and `manifest.json`
bytes, the accepted source digest, a stat of every graphed source file (a
snippet can come from a file outside the source digest), the tokenizer inputs
and the code recipe. Writes are spooled through `PendingCacheWrites` and land
only after publish.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import unicodedata
from functools import lru_cache
from pathlib import Path

from graphify_mesh.sync.embedding import _source_file_stat_lines
from graphify_mesh.sync.lexical_index import (
    LEXICAL_PARTIAL_RECIPE,
    LEXICAL_SCHEMA_VERSION,
    TOKENIZER_VERSION,
    ManifestRead,
    RepoLexicalPartial,
)
from graphify_mesh.sync.stage_cache import (
    CACHE_FORMAT,
    PendingCacheWrites,
    gc_entries,
    read_entry,
)

log = logging.getLogger(__name__)

LEXICAL_CACHE_SUBDIR = "lexical"
_ABSENT = b"absent"


@lru_cache(maxsize=1)
def _graphify_version() -> str:
    # Tokenization strips diacritics through graphify's own helper.
    try:
        return importlib.metadata.version("graphifyy")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def partial_key(
    repo_id: str,
    graph_sha: bytes | None,
    manifest: ManifestRead,
    source_digest: str | None,
    graph_data: dict,
    source_root: Path | None,
) -> str | None:
    """None when an input is unknown or unreadable: that repo is not cached.
    `graph_sha` is the sha256 of the bytes `graph_data` was parsed from, and
    `manifest` is the same read `build_repo_partial` parses."""
    if graph_sha is None or source_digest is None or not manifest.readable:
        return None
    manifest_sha = _ABSENT if manifest.raw is None else hashlib.sha256(manifest.raw).digest()
    header = (
        LEXICAL_PARTIAL_RECIPE,
        str(CACHE_FORMAT),
        str(LEXICAL_SCHEMA_VERSION),
        TOKENIZER_VERSION,
        _graphify_version(),
        unicodedata.unidata_version,
        repo_id,
        source_digest,
        str(source_root),
    )
    h = hashlib.sha256("\x00".join(header).encode("utf-8", "surrogatepass"))
    h.update(b"\x00graph\x00" + graph_sha)
    h.update(b"\x00manifest\x00" + manifest_sha + b"\x00")
    for line in _source_file_stat_lines(graph_data, source_root):
        h.update(line)
    return h.hexdigest()


class LexicalPartialCache:
    """Cache source handed to `build_lexical_index`.

    After publish and `pending.flush()`, call `collect_garbage()` to drop
    entries this run did not use.
    """

    def __init__(self, cache_dir: Path, pending: PendingCacheWrites) -> None:
        self.dir = cache_dir / LEXICAL_CACHE_SUBDIR
        self.pending = pending
        self.used_keys: set[str] = set()
        self.consulted = False
        self.hits = 0
        self.misses = 0
        self.staged_bytes = 0

    def key(
        self,
        repo_id: str,
        graph_sha: bytes | None,
        manifest: ManifestRead,
        source_digest: str | None,
        graph_data: dict,
        source_root: Path | None,
    ) -> str | None:
        return partial_key(repo_id, graph_sha, manifest, source_digest, graph_data, source_root)

    def read(self, key: str, repo_id: str) -> RepoLexicalPartial | None:
        self.consulted = True
        self.used_keys.add(key)
        payload = read_entry(self.dir, key)
        if payload is None:
            self.misses += 1
            return None
        partial = RepoLexicalPartial.from_bytes(payload)
        del payload
        if partial is None or partial.repo_id != repo_id:
            log.debug("lexical: partial cache entry %s unreadable, recomputing", key)
            self.misses += 1
            return None
        self.hits += 1
        return partial

    def stage(self, key: str, partial: RepoLexicalPartial) -> None:
        self.used_keys.add(key)
        # A cache problem is never a run failure: an unserializable partial is
        # simply not cached.
        try:
            payload = partial.to_bytes()
        except (TypeError, ValueError) as exc:
            log.warning("lexical: could not serialize partial %s: %s", key, exc)
            return
        self.staged_bytes += len(payload)
        self.pending.add(self.dir, key, payload)

    def collect_garbage(self) -> int:
        """Remove entries this run did not read or stage. A run that computed
        no key (no graph paths or digests) keeps the cache as it is."""
        if not self.consulted:
            return 0
        return gc_entries(self.dir, self.used_keys)
