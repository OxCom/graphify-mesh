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
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

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


_COPY_CHUNK = 1 << 20


def _write_framed(
    dir: Path, key: str, src: BinaryIO, length: int, expected_digest: bytes | None = None
) -> None:
    """Write `length` bytes from `src` as entry `key`, atomically, without
    holding the payload in RAM: the digest slot is filled after the stream.
    Same framing as `write_entry`. With `expected_digest`, a payload that does
    not match it raises before anything is renamed into place."""
    path = _entry_path(dir, key)
    if path is None:
        raise ValueError(f"invalid cache key: {key!r}")
    dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dir), prefix=".cache-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(_PREFIX)
            fh.write(length.to_bytes(_LEN_BYTES, "big"))
            fh.write(b"\x00" * _DIGEST_BYTES)
            hasher = hashlib.sha256()
            remaining = length
            while remaining:
                chunk = src.read(min(_COPY_CHUNK, remaining))
                if not chunk:
                    raise ValueError(f"source ended {remaining} byte(s) early")
                hasher.update(chunk)
                fh.write(chunk)
                remaining -= len(chunk)
            if src.read(1):
                raise ValueError("source grew while it was copied")
            digest = hasher.digest()[:_DIGEST_BYTES]
            if expected_digest is not None and digest != expected_digest:
                raise ValueError("payload does not match its recorded digest")
            fh.seek(len(_PREFIX) + _LEN_BYTES)
            fh.write(digest)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def write_entry_from_file(dir: Path, key: str, source: Path) -> None:
    """`write_entry` for a payload that lives in a file, streamed."""
    with source.open("rb") as src:
        _write_framed(dir, key, src, os.fstat(src.fileno()).st_size)


def _open_entry(path: Path) -> tuple[BinaryIO, int, bytes] | None:
    """Open a framed entry positioned at its payload: (file, length, digest).
    None (file closed) for a foreign, truncated or oversized entry."""
    fh = path.open("rb")
    try:
        header = fh.read(_HEADER_SIZE)
        if len(header) < _HEADER_SIZE or not header.startswith(_PREFIX):
            fh.close()
            return None
        offset = len(_PREFIX)
        length = int.from_bytes(header[offset : offset + _LEN_BYTES], "big")
        if os.fstat(fh.fileno()).st_size != _HEADER_SIZE + length:
            fh.close()
            return None
        return fh, length, header[offset + _LEN_BYTES : _HEADER_SIZE]
    except BaseException:
        fh.close()
        raise


def copy_entry_payload(dir: Path, key: str, dest: Path) -> bool:
    """Stream entry `key`'s payload into `dest`, verifying its digest on the
    way; the payload is never held in RAM. False, with `dest` untouched, for a
    missing, corrupt or unreadable entry: the caller recomputes."""
    path = _entry_path(dir, key)
    if path is None:
        log.debug("stage-cache: invalid key %r ignored", key)
        return False
    part = dest.with_name(dest.name + ".part")
    try:
        opened = _open_entry(path)
        if opened is None:
            log.debug("stage-cache: foreign or corrupt entry %s ignored", path)
            return False
        src, length, digest = opened
        with src, part.open("wb") as out:
            hasher = hashlib.sha256()
            remaining = length
            while remaining:
                chunk = src.read(min(_COPY_CHUNK, remaining))
                if not chunk:
                    break
                hasher.update(chunk)
                out.write(chunk)
                remaining -= len(chunk)
        if remaining or hasher.digest()[:_DIGEST_BYTES] != digest:
            log.debug("stage-cache: truncated or corrupt entry %s ignored", path)
            part.unlink(missing_ok=True)
            return False
        os.replace(part, dest)
        return True
    except FileNotFoundError:
        part.unlink(missing_ok=True)
        return False
    except OSError as exc:
        log.debug("stage-cache: unreadable entry %s: %s", path, exc)
        part.unlink(missing_ok=True)
        return False


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
        self._spool(dir, key, lambda staging, spool_key: write_entry(staging, spool_key, payload))

    def add_file(self, dir: Path, key: str, source: Path) -> None:
        """`add` for a payload held in a file: streamed, never read into RAM.
        The file may be removed once this returns."""
        self._spool(
            dir, key, lambda staging, spool_key: write_entry_from_file(staging, spool_key, source)
        )

    def _spool(self, dir: Path, key: str, write: Callable[[Path, str], None]) -> None:
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
            write(self._staging, spool_key)
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
        # Another filesystem than the spool dir: fall back to a framed copy,
        # streamed so a large entry is never held in RAM.
        opened = _open_entry(spooled)
        if opened is None:
            raise
        src, length, digest = opened
        with src:
            _write_framed(dir, key, src, length, expected_digest=digest)


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
