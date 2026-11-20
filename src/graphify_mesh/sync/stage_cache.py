"""Content-addressed stage cache entries under `<global_dir>/cache/<stage>/`.

An entry is `<key>.bin`: magic, CACHE_FORMAT, payload length and a sha256
prefix, then the payload. A missing, foreign-format, truncated, corrupt or
unreadable entry reads as None so the caller recomputes; a cache problem is
never a run failure.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

CACHE_FORMAT = 1

_MAGIC = b"GMSC"
_PREFIX = _MAGIC + CACHE_FORMAT.to_bytes(4, "big")
_LEN_BYTES = 8
_DIGEST_BYTES = 16
_HEADER_SIZE = len(_PREFIX) + _LEN_BYTES + _DIGEST_BYTES
_SUFFIX = ".bin"
_STAGING_PREFIX = ".staging-"
# No slash and no leading dot: a key must stay a single file name inside dir.
_KEY_RE = re.compile(r"[0-9A-Za-z_-][0-9A-Za-z._-]*")


def _entry_path(cache_dir: Path, key: str) -> Path | None:
    if not _KEY_RE.fullmatch(key):
        return None
    return cache_dir / f"{key}{_SUFFIX}"


def _digest(payload: bytes) -> bytes:
    return hashlib.sha256(payload).digest()[:_DIGEST_BYTES]


def read_entry(dir: Path, key: str) -> bytes | None:
    path = _entry_path(dir, key)
    if path is None:
        log.debug("stage-cache: invalid key %r ignored", key)
        return None
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        log.debug("stage-cache: unreadable entry %s: %s", path, exc)
        return None
    if len(raw) < _HEADER_SIZE or not raw.startswith(_PREFIX):
        log.debug("stage-cache: foreign or corrupt entry %s ignored", path)
        return None
    offset = len(_PREFIX)
    length = int.from_bytes(raw[offset : offset + _LEN_BYTES], "big")
    digest = raw[offset + _LEN_BYTES : _HEADER_SIZE]
    payload = raw[_HEADER_SIZE:]
    if len(payload) != length or _digest(payload) != digest:
        log.debug("stage-cache: truncated or corrupt entry %s ignored", path)
        return None
    return payload


def write_entry(dir: Path, key: str, payload: bytes) -> None:
    """Atomic: a crash mid-write must not leave a half entry a reader accepts."""
    path = _entry_path(dir, key)
    if path is None:
        raise ValueError(f"invalid cache key: {key!r}")
    dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dir), prefix=".cache-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(_PREFIX)
            fh.write(len(payload).to_bytes(_LEN_BYTES, "big"))
            fh.write(_digest(payload))
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def gc_entries(dir: Path, keep_keys: set[str]) -> int:
    """Remove entries whose key is not in keep_keys; return how many were removed.

    Stale `.tmp` leftovers from a crashed write are collected too.
    """
    if not dir.is_dir():
        return 0
    removed = 0
    for path in dir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if name.endswith(_SUFFIX) and name[: -len(_SUFFIX)] in keep_keys:
            continue
        if not (name.endswith(_SUFFIX) or name.endswith(".tmp")):
            continue
        try:
            path.unlink()
            removed += 1
        except OSError as exc:
            log.debug("stage-cache: could not remove %s: %s", path, exc)
    return removed


class PendingCacheWrites:
    """Cache writes made during a run, spooled to disk at once and moved into
    place only after publish.

    Payloads never stay in RAM: each one is written, framed and atomic, into a
    private `<cache_dir>/.staging-<pid>-<random>/` (same filesystem, so `flush`
    is a rename). A run that never reaches flip_current (dry run, blocked
    publish, crash) calls `discard()` and leaves the cache untouched. With no
    `cache_dir` (cache off, dry run) `add` drops the payload.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir
        self._staging: Path | None = None
        self._items: list[tuple[Path, str, Path]] = []

    def add(self, dir: Path, key: str, payload: bytes) -> None:
        if self._cache_dir is None:
            return
        if _entry_path(dir, key) is None:
            log.warning("stage-cache: invalid key %r not staged", key)
            return
        # A cache problem is never a run failure: an entry that cannot be spooled
        # is simply not cached.
        try:
            if self._staging is None:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                self._staging = Path(
                    tempfile.mkdtemp(prefix=f"{_STAGING_PREFIX}{os.getpid()}-", dir=self._cache_dir)
                )
            spool_key = f"p{len(self._items)}"
            write_entry(self._staging, spool_key, payload)
        except Exception as exc:
            log.warning("stage-cache: could not stage %s/%s: %s", dir, key, exc)
            return
        self._items.append((dir, key, self._staging / f"{spool_key}{_SUFFIX}"))

    def __len__(self) -> int:
        return len(self._items)

    @property
    def staging_dir(self) -> Path | None:
        return self._staging

    def flush(self) -> None:
        items, self._items = self._items, []
        for dir, key, spooled in items:
            # Runs after flip_current: any failure here must not fail a published run.
            try:
                _promote(spooled, dir, key)
            except Exception as exc:
                log.warning("stage-cache: could not write %s/%s: %s", dir, key, exc)
        self.discard()

    def discard(self) -> None:
        """Drop everything staged and remove the spool dir. Safe to call twice."""
        self._items = []
        staging, self._staging = self._staging, None
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def _promote(spooled: Path, dir: Path, key: str) -> None:
    """Move one spooled entry into place; it is already framed and fsynced."""
    path = _entry_path(dir, key)
    if path is None:
        raise ValueError(f"invalid cache key: {key!r}")
    dir.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(spooled, path)
    except OSError:
        # Another filesystem than the spool dir: fall back to a framed copy.
        payload = read_entry(spooled.parent, spooled.name[: -len(_SUFFIX)])
        if payload is None:
            raise
        write_entry(dir, key, payload)


def sweep_staging_dirs(cache_dir: Path) -> list[str]:
    """Remove every `.staging-*` dir under cache_dir. Call only while holding the
    transaction lock: no other run can own one then, so all of them are stale."""
    if not cache_dir.is_dir():
        return []
    removed: list[str] = []
    for path in sorted(cache_dir.iterdir()):
        if path.name.startswith(_STAGING_PREFIX) and path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)
    return removed
