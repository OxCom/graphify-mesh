"""Streamed stage-cache entries: same framing as write_entry/read_entry, so
every existing reader stays compatible, and no payload is held in RAM."""

from __future__ import annotations

import os

import pytest

from graphify_mesh.sync import stage_cache
from graphify_mesh.sync.stage_cache import (
    PendingCacheWrites,
    copy_entry_payload,
    read_entry,
    write_entry,
    write_entry_from_file,
)

_PAYLOAD = bytes(range(256)) * 9000  # > one copy chunk


@pytest.fixture
def small_chunks(monkeypatch):
    monkeypatch.setattr(stage_cache, "_COPY_CHUNK", 4096)


def test_streamed_write_has_write_entry_framing(tmp_path, small_chunks):
    source = tmp_path / "payload"
    source.write_bytes(_PAYLOAD)
    write_entry(tmp_path / "a", "k", _PAYLOAD)
    write_entry_from_file(tmp_path / "b", "k", source)
    assert (tmp_path / "a" / "k.bin").read_bytes() == (tmp_path / "b" / "k.bin").read_bytes()
    assert read_entry(tmp_path / "b", "k") == _PAYLOAD


def test_copy_entry_payload_round_trip(tmp_path, small_chunks):
    write_entry(tmp_path, "k", _PAYLOAD)
    dest = tmp_path / "out.json"
    assert copy_entry_payload(tmp_path, "k", dest)
    assert dest.read_bytes() == _PAYLOAD


@pytest.mark.parametrize("damage", ["flip", "truncate", "extend", "foreign"])
def test_copy_entry_payload_rejects_bad_entries(tmp_path, small_chunks, damage):
    write_entry(tmp_path, "k", _PAYLOAD)
    path = tmp_path / "k.bin"
    raw = bytearray(path.read_bytes())
    if damage == "flip":
        raw[-1] ^= 0xFF
    elif damage == "truncate":
        del raw[-10:]
    elif damage == "extend":
        raw += b"x"
    else:
        raw[:4] = b"XXXX"
    path.write_bytes(bytes(raw))
    dest = tmp_path / "out.json"
    dest.write_bytes(b"previous")
    assert not copy_entry_payload(tmp_path, "k", dest)
    assert dest.read_bytes() == b"previous"
    assert not (tmp_path / "out.json.part").exists()


def test_copy_entry_payload_missing_entry(tmp_path):
    assert not copy_entry_payload(tmp_path, "absent", tmp_path / "out.json")
    assert not copy_entry_payload(tmp_path, "../escape", tmp_path / "out.json")


def test_add_file_spools_and_flushes(tmp_path, small_chunks):
    cache_dir = tmp_path / "cache"
    source = tmp_path / "merged.json"
    source.write_bytes(_PAYLOAD)
    pending = PendingCacheWrites(cache_dir)
    pending.add_file(cache_dir / "merge", "k", source)
    source.unlink()
    assert len(pending) == 1
    pending.flush()
    assert read_entry(cache_dir / "merge", "k") == _PAYLOAD
    assert not list(cache_dir.glob(".staging-*"))


def test_add_file_without_cache_dir_drops(tmp_path):
    source = tmp_path / "merged.json"
    source.write_bytes(_PAYLOAD)
    pending = PendingCacheWrites(None)
    pending.add_file(tmp_path / "merge", "k", source)
    assert len(pending) == 0


def test_add_file_unreadable_source_is_not_cached(tmp_path):
    pending = PendingCacheWrites(tmp_path / "cache")
    pending.add_file(tmp_path / "cache" / "merge", "k", tmp_path / "missing")
    assert len(pending) == 0
    pending.discard()


def test_promote_cross_filesystem_fallback_streams(tmp_path, small_chunks, monkeypatch):
    cache_dir = tmp_path / "cache"
    pending = PendingCacheWrites(cache_dir)
    pending.add(cache_dir / "merge", "k", _PAYLOAD)
    real_replace = os.replace

    def exdev(src, dst):
        if str(src).endswith(".bin") and ".staging-" in str(src):
            raise OSError(18, "Invalid cross-device link")
        return real_replace(src, dst)

    monkeypatch.setattr(stage_cache.os, "replace", exdev)
    pending.flush()
    assert read_entry(cache_dir / "merge", "k") == _PAYLOAD


def test_promote_fallback_refuses_corrupt_spool(tmp_path, small_chunks, monkeypatch):
    staging = tmp_path / "spool"
    write_entry(staging, "p0", _PAYLOAD)
    spooled = staging / "p0.bin"
    raw = bytearray(spooled.read_bytes())
    raw[-1] ^= 0xFF
    spooled.write_bytes(bytes(raw))
    real_replace = os.replace

    def exdev(src, dst):
        if src == spooled:
            raise OSError(18, "Invalid cross-device link")
        return real_replace(src, dst)

    monkeypatch.setattr(stage_cache.os, "replace", exdev)
    with pytest.raises(ValueError):
        stage_cache._promote(spooled, tmp_path / "dest", "k")
    assert not (tmp_path / "dest" / "k.bin").exists()
    assert not list((tmp_path / "dest").glob("*.tmp"))
