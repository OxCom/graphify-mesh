"""Per-repo cache for the overlay API scan (`provides_api` / `consumes_api` facts).

An entry holds one repo's providers and consumer candidates under
`<cache_dir>/overlay_api/<key>.bin`. The key covers the repo id and source
root, the repo's live source digest (`SourceDigest.attempt_id`, which hashes
every file the scan reads: see `overlay_api._walk_source_files`), the
fingerprint of graphify's ignore rules from the same walk, the graphify
version and the extractor recipe. A repo with no digest from this run is not
cached. A hit therefore skips the walk and every read. Writes are spooled
through `PendingCacheWrites` and land only after publish.

The key describes the tree as the digest walk saw it, earlier in the run. A
scan is stored only when its facts provably come from that tree: the scan
completed with no walk, stat or read failure; every digest marker is a content
hash; the scan read exactly the digest's code files with a scanned suffix; and
the content marker of the bytes it read equals the digest's marker for each
file. A file edited between the two walks, or an ignore state that diverged
(git failing in one walk only), therefore never lands in the cache. The
pipeline passes no digest at all when content digests are off.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from graphify_mesh.sync.lexical_partial_cache import _graphify_version
from graphify_mesh.sync.overlay_api import (
    API_FACTS_RECIPE,
    ApiFacts,
    ApiProvider,
    ApiScan,
    scanned_suffix,
)
from graphify_mesh.sync.stage_cache import (
    CACHE_FORMAT,
    PendingCacheWrites,
    gc_entries,
    read_entry,
)
from graphify_mesh.sync.state import SourceDigest

log = logging.getLogger(__name__)

# Not `overlay/`: gc_entries removes every key a cache did not use, so two
# caches must never share a directory.
API_CACHE_SUBDIR = "overlay_api"


@dataclass(frozen=True)
class ApiSourceId:
    """One repo's digest walk this run: its attempt_id and ignore_hash (the
    key), and the change markers of its code files with a scanned suffix."""

    attempt_id: str
    ignore_hash: str
    api_markers: Mapping[str, str]
    mtime_markers: bool

    @classmethod
    def from_digest(cls, digest: SourceDigest) -> ApiSourceId:
        return cls(
            attempt_id=digest.attempt_id,
            ignore_hash=digest.ignore_hash,
            api_markers={
                rel: marker
                for rel, marker in digest.code_markers.items()
                if scanned_suffix(rel) is not None
            },
            mtime_markers=digest.mtime_markers,
        )


def facts_key(repo_id: str, root: Path, source_attempt_id: str, ignore_hash: str) -> str:
    parts = (
        API_FACTS_RECIPE,
        str(CACHE_FORMAT),
        _graphify_version(),
        repo_id,
        str(root),
        source_attempt_id,
        ignore_hash,
    )
    return hashlib.sha256("\x00".join(parts).encode("utf-8", "surrogatepass")).hexdigest()


def _encode(repo_id: str, facts: ApiFacts) -> bytes:
    providers, candidates = facts
    payload = {
        "repo_id": repo_id,
        "providers": [
            [p.repo, p.http_method, p.path_template, p.source_file, p.qualified_label]
            for p in providers
        ],
        "candidates": [list(c) for c in candidates],
    }
    # ensure_ascii keeps surrogate-escaped file names round-trippable.
    return json.dumps(payload, separators=(",", ":")).encode("ascii")


def _decode(payload: bytes, repo_id: str) -> ApiFacts | None:
    try:
        data = json.loads(payload)
        if data["repo_id"] != repo_id:
            return None
        providers = [ApiProvider(*(str(f) for f in row)) for row in data["providers"]]
        candidates = [
            (str(row[0]), str(row[1]), str(row[2]), str(row[3])) for row in data["candidates"]
        ]
    except (ValueError, KeyError, TypeError, IndexError):
        return None
    if any(len(row) != 4 for row in data["candidates"]):
        return None
    return providers, candidates


class ApiFactsCache:
    """Cache source handed to `overlay_api.collect_api_facts`.

    `source_ids_by_repo` maps a repo id to its digest walk from this run under
    CURRENT_CLASSIFICATION; a repo missing from it is neither read nor written.
    A scan that cannot be matched to its digest is not stored (`skipped`).
    After publish and `pending.flush()`, call `collect_garbage()` to drop
    entries this run did not use.
    """

    def __init__(
        self,
        cache_dir: Path,
        pending: PendingCacheWrites,
        source_ids_by_repo: dict[str, ApiSourceId],
    ) -> None:
        self.dir = cache_dir / API_CACHE_SUBDIR
        self.pending = pending
        self._source_ids = source_ids_by_repo
        self.used_keys: set[str] = set()
        self.consulted = False
        self.hits = 0
        self.misses = 0
        self.skipped = 0
        self.staged_bytes = 0

    def _key(self, repo_id: str, root: Path) -> str | None:
        ids = self._source_ids.get(repo_id)
        if ids is None:
            return None
        return facts_key(repo_id, root, ids.attempt_id, ids.ignore_hash)

    def lookup(self, repo_id: str, root: Path) -> ApiFacts | None:
        key = self._key(repo_id, root)
        if key is None:
            return None
        self.consulted = True
        self.used_keys.add(key)
        payload = read_entry(self.dir, key)
        facts = _decode(payload, repo_id) if payload is not None else None
        if facts is None:
            if payload is not None:
                log.debug("overlay: API facts entry %s unreadable, recomputing", key)
            self.misses += 1
            return None
        self.hits += 1
        return facts

    def _untrusted(self, repo_id: str, scan: ApiScan) -> str | None:
        """Why this scan may not describe the tree the key names, or None."""
        ids = self._source_ids[repo_id]
        if not scan.complete:
            return "a walk, stat or read failed during the scan"
        if ids.mtime_markers:
            return "the source digest holds mtime markers, not content hashes"
        if scan.markers.keys() != ids.api_markers.keys():
            missing = len(ids.api_markers.keys() - scan.markers.keys())
            extra = len(scan.markers.keys() - ids.api_markers.keys())
            return (
                f"the scan read other files than the digest hashed "
                f"({missing} not read, {extra} not in the digest)"
            )
        changed = sum(1 for rel, marker in scan.markers.items() if ids.api_markers[rel] != marker)
        if changed:
            return f"{changed} file(s) changed between the digest walk and the scan"
        return None

    def store(self, repo_id: str, root: Path, facts: ApiFacts, scan: ApiScan) -> None:
        key = self._key(repo_id, root)
        if key is None:
            return
        self.used_keys.add(key)
        reason = self._untrusted(repo_id, scan)
        if reason is not None:
            self.skipped += 1
            log.info("overlay: API facts of %s not cached: %s", repo_id, reason)
            return
        payload = _encode(repo_id, facts)
        self.staged_bytes += len(payload)
        self.pending.add(self.dir, key, payload)

    def collect_garbage(self) -> int:
        """Remove entries this run did not read or stage. A run that keyed no
        repo keeps the cache as it is."""
        if not self.consulted:
            return 0
        return gc_entries(self.dir, self.used_keys)
