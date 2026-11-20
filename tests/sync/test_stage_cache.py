from __future__ import annotations

import os

import pytest

from graphify_mesh.sync import pipeline, stage_cache
from graphify_mesh.sync.pipeline import run
from graphify_mesh.sync.stage_cache import (
    CACHE_FORMAT,
    PendingCacheWrites,
    gc_entries,
    read_entry,
    sweep_staging_dirs,
    write_entry,
)


def test_round_trip(tmp_path):
    write_entry(tmp_path / "s", "abc123", b"payload")
    assert read_entry(tmp_path / "s", "abc123") == b"payload"
    assert (tmp_path / "s" / "abc123.bin").is_file()
    assert not list((tmp_path / "s").glob("*.tmp"))


def test_missing_entry_is_none(tmp_path):
    assert read_entry(tmp_path, "nope") is None


def test_corrupt_and_foreign_format_read_as_none(tmp_path):
    (tmp_path / "bad.bin").write_bytes(b"garbage")
    (tmp_path / "empty.bin").write_bytes(b"")
    write_entry(tmp_path, "old", b"x")
    raw = (tmp_path / "old.bin").read_bytes()
    foreign = raw[:4] + (CACHE_FORMAT + 1).to_bytes(4, "big") + raw[8:]
    (tmp_path / "old.bin").write_bytes(foreign)
    assert read_entry(tmp_path, "bad") is None
    assert read_entry(tmp_path, "empty") is None
    assert read_entry(tmp_path, "old") is None


def test_truncated_and_bitflipped_entries_read_as_none(tmp_path):
    write_entry(tmp_path, "t", b"0123456789")
    raw = (tmp_path / "t.bin").read_bytes()
    (tmp_path / "t.bin").write_bytes(raw[:-3])
    assert read_entry(tmp_path, "t") is None
    flipped = bytearray(raw)
    flipped[-1] ^= 0x01
    (tmp_path / "t.bin").write_bytes(bytes(flipped))
    assert read_entry(tmp_path, "t") is None
    (tmp_path / "t.bin").write_bytes(raw[:10])
    assert read_entry(tmp_path, "t") is None
    (tmp_path / "t.bin").write_bytes(raw)
    assert read_entry(tmp_path, "t") == b"0123456789"


@pytest.mark.parametrize("key", ["../x", "a/b", ".hidden", "", "x\0y"])
def test_invalid_keys_rejected(tmp_path, key):
    cache = tmp_path / "c"
    with pytest.raises(ValueError):
        write_entry(cache, key, b"v")
    assert read_entry(cache, key) is None
    assert not (tmp_path / "x.bin").exists()
    assert not cache.exists()


def test_flush_swallows_non_oserror_and_continues(tmp_path, monkeypatch):
    real = stage_cache._promote

    def flaky(spooled, dir, key):
        if key == "bad":
            raise RuntimeError("boom")
        real(spooled, dir, key)

    monkeypatch.setattr(stage_cache, "_promote", flaky)
    pending = PendingCacheWrites(tmp_path)
    pending.add(tmp_path / "s", "bad", b"1")
    pending.add(tmp_path / "s", "good", b"2")
    pending.flush()
    assert read_entry(tmp_path / "s", "good") == b"2"
    assert read_entry(tmp_path / "s", "bad") is None
    assert _spool_dirs(tmp_path) == []


def test_unreadable_entry_is_none(tmp_path):
    (tmp_path / "dir.bin").mkdir()
    assert read_entry(tmp_path, "dir") is None


def test_gc_keeps_exactly_keep_keys(tmp_path):
    for key in ("a", "b", "c"):
        write_entry(tmp_path, key, key.encode())
    (tmp_path / ".cache-x.tmp").write_bytes(b"leftover")
    assert gc_entries(tmp_path, {"a", "c", "zzz"}) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.bin", "c.bin"]
    assert gc_entries(tmp_path / "missing", {"a"}) == 0


def _spool_dirs(cache_dir):
    return sorted(p.name for p in cache_dir.glob(".staging-*"))


def test_pending_writes_are_spooled_to_disk_and_moved_on_flush(tmp_path):
    pending = PendingCacheWrites(tmp_path)
    pending.add(tmp_path / "s", "k", b"v")
    assert read_entry(tmp_path / "s", "k") is None
    # The payload is on disk, framed, in the run's private spool dir, not in RAM.
    spool = pending.staging_dir
    assert spool is not None and spool.parent == tmp_path
    assert spool.name.startswith(f".staging-{os.getpid()}-")
    assert read_entry(spool, "p0") == b"v"
    assert not hasattr(pending, "_payloads")
    assert all(not isinstance(item, bytes) for entry in pending._items for item in entry)

    pending.flush()
    assert read_entry(tmp_path / "s", "k") == b"v"
    assert _spool_dirs(tmp_path) == []
    pending.flush()


def test_flush_swallows_write_failure(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(stage_cache, "_promote", boom)
    pending = PendingCacheWrites(tmp_path)
    pending.add(tmp_path / "s", "k", b"v")
    pending.flush()
    assert read_entry(tmp_path / "s", "k") is None
    assert _spool_dirs(tmp_path) == []


def test_discard_removes_spool_and_writes_nothing(tmp_path):
    pending = PendingCacheWrites(tmp_path)
    pending.add(tmp_path / "s", "k", b"v")
    assert len(_spool_dirs(tmp_path)) == 1
    pending.discard()
    pending.discard()
    assert _spool_dirs(tmp_path) == []
    assert len(pending) == 0
    pending.flush()
    assert not (tmp_path / "s").exists()


def test_no_cache_dir_stages_nothing(tmp_path):
    pending = PendingCacheWrites(None)
    pending.add(tmp_path / "s", "k", b"v")
    assert len(pending) == 0
    pending.flush()
    assert list(tmp_path.iterdir()) == []


def test_invalid_key_is_not_staged(tmp_path):
    pending = PendingCacheWrites(tmp_path)
    pending.add(tmp_path / "s", "../x", b"v")
    assert len(pending) == 0
    assert _spool_dirs(tmp_path) == []


def test_sweep_removes_every_spool_dir(tmp_path):
    (tmp_path / ".staging-1-abc").mkdir()
    (tmp_path / ".staging-1-abc" / "p0.bin").write_bytes(b"x")
    (tmp_path / ".staging-2-def").mkdir()
    (tmp_path / "overlay").mkdir()
    assert sweep_staging_dirs(tmp_path) == [".staging-1-abc", ".staging-2-def"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["overlay"]
    assert sweep_staging_dirs(tmp_path / "missing") == []


def test_env_switch(monkeypatch, tmp_path):
    from graphify_mesh.sync.config import Settings

    kwargs = dict(
        mesh_root=tmp_path,
        scan_roots=[tmp_path],
        approved_roots=[tmp_path],
        registry_path=tmp_path / "r.json",
    )
    monkeypatch.setenv("GRAPHIFY_MESH_STAGE_CACHE", "0")
    assert Settings(**kwargs).stage_cache is False
    monkeypatch.setenv("GRAPHIFY_MESH_STAGE_CACHE", "1")
    assert Settings(**kwargs).stage_cache is True
    monkeypatch.delenv("GRAPHIFY_MESH_STAGE_CACHE")
    s = Settings(**kwargs)
    assert s.stage_cache is True
    assert s.cache_dir == s.global_dir / "cache"


@pytest.fixture
def flushes(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(PendingCacheWrites, "flush", lambda self: calls.append(1))
    return calls


def _one_repo(env):
    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.write_registry()


def test_flush_reached_on_publish(env, flushes):
    _one_repo(env)
    assert run(env.settings()).published
    assert flushes == [1]


def test_flush_not_reached_on_dry_run(env, flushes):
    _one_repo(env)
    assert not run(env.settings(dry_run=True)).published
    assert flushes == []


def test_flush_not_reached_on_blocked_publish(env, flushes):
    # Two repo_ids sharing one collection_path block the run before publish.
    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.add_repo(
        "example-org.styleguide-mirror",
        "example-org",
        "styleguide",
        "mirror.example-org.dev.lo",
        "repo_a.json",
    )
    env.write_registry()
    assert not run(env.settings()).published
    assert flushes == []


def test_flush_not_reached_when_cache_off(env, flushes):
    _one_repo(env)
    assert run(env.settings(stage_cache=False)).published
    assert flushes == []


def test_stale_spool_dirs_removed_at_run_start(env):
    _one_repo(env)
    settings = env.settings()
    stale = settings.cache_dir / ".staging-999-stale"
    stale.mkdir(parents=True)
    (stale / "p0.bin").write_bytes(b"x")
    assert run(settings).published
    assert not stale.exists()


def test_cache_off_never_creates_or_touches_cache_dir(env):
    _one_repo(env)
    settings = env.settings(stage_cache=False)
    assert run(settings).published
    assert not settings.cache_dir.exists()

    stale = settings.cache_dir / ".staging-999-stale"
    stale.mkdir(parents=True)
    assert run(settings).published
    assert stale.is_dir()
    assert sorted(p.name for p in settings.cache_dir.iterdir()) == [stale.name]


def test_sweep_oserror_does_not_fail_the_run(env, monkeypatch, caplog):
    _one_repo(env)

    def denied(cache_dir):
        raise PermissionError(13, "Permission denied", str(cache_dir))

    monkeypatch.setattr(pipeline, "sweep_staging_dirs", denied)
    with caplog.at_level("WARNING"):
        assert run(env.settings()).published
    assert "could not sweep stage-cache spool dirs" in caplog.text


def test_dry_run_creates_no_cache_dir(env):
    _one_repo(env)
    settings = env.settings(dry_run=True)
    assert not settings.cache_dir.exists()
    run(settings)
    assert not settings.cache_dir.exists()


def test_crash_after_staging_discards_spool(env, monkeypatch):
    _one_repo(env)
    settings = env.settings()
    seen: list[PendingCacheWrites] = []

    def crashing(settings, staging_root, pending_cache):
        pending_cache.add(settings.cache_dir / "overlay", "k", b"v")
        seen.append(pending_cache)
        assert len(_spool_dirs(settings.cache_dir)) == 1
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "_run_locked", crashing)
    with pytest.raises(RuntimeError):
        run(settings)
    assert seen and _spool_dirs(settings.cache_dir) == []
    assert read_entry(settings.cache_dir / "overlay", "k") is None


def test_blocked_publish_discards(env, monkeypatch):
    discards: list[int] = []
    real = PendingCacheWrites.discard

    def counting(self):
        discards.append(1)
        real(self)

    monkeypatch.setattr(PendingCacheWrites, "discard", counting)
    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.add_repo(
        "example-org.styleguide-mirror",
        "example-org",
        "styleguide",
        "mirror.example-org.dev.lo",
        "repo_a.json",
    )
    env.write_registry()
    settings = env.settings()
    assert not run(settings).published
    assert discards == [1]
    if settings.cache_dir.exists():
        assert _spool_dirs(settings.cache_dir) == []
