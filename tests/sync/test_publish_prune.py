from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from graphify_mesh.sync import publish


def _make_generation(generations_dir: Path, name: str, incomplete: bool = False) -> Path:
    """A complete generation carries graph + manifest (and any artifacts the
    manifest's `artifact_sha256` map declares) — prune_old_generations treats
    a generation with `.tmp` leftovers, a missing manifest/graph, or a
    manifest-declared artifact absent on disk as incomplete (never-published)
    and removes it regardless of keep count."""
    gen_dir = generations_dir / name
    gen_dir.mkdir(parents=True)
    (gen_dir / "global-graph.json").write_text("{}", encoding="utf-8")
    (gen_dir / "cross-project-overlay.json").write_text("{}", encoding="utf-8")
    (gen_dir / "generation-manifest.json").write_text("{}", encoding="utf-8")
    if incomplete:
        (gen_dir / "lexical-index.json.tmp").write_text("{", encoding="utf-8")
        return gen_dir
    (gen_dir / "lexical-index.json").write_text("{}", encoding="utf-8")
    return gen_dir


def test_prune_keeps_current_and_n_most_recent(tmp_path):
    generations_dir = tmp_path / "generations"
    names = ["20260101T000000Z-a", "20260102T000000Z-b", "20260103T000000Z-c", "20260104T000000Z-d"]
    for name in names:
        _make_generation(generations_dir, name)

    current = tmp_path / "current"
    current.symlink_to(generations_dir / names[-1], target_is_directory=True)

    removed = publish.prune_old_generations(generations_dir, current, keep=2)

    remaining = {p.name for p in generations_dir.iterdir()}
    assert remaining == {names[-2], names[-1]}
    assert set(removed) == {names[0], names[1]}


def test_prune_never_removes_current_even_if_older_than_keep_window(tmp_path):
    generations_dir = tmp_path / "generations"
    names = ["20260101T000000Z-a", "20260102T000000Z-b", "20260103T000000Z-c"]
    for name in names:
        _make_generation(generations_dir, name)

    # current points at the OLDEST generation (e.g. a later run failed
    # validation and never flipped forward) — it must survive pruning
    # regardless of the keep-count sort order.
    current = tmp_path / "current"
    current.symlink_to(generations_dir / names[0], target_is_directory=True)

    publish.prune_old_generations(generations_dir, current, keep=1)

    remaining = {p.name for p in generations_dir.iterdir()}
    assert names[0] in remaining


def test_prune_removes_incomplete_generation_regardless_of_keep_count(tmp_path):
    """A generation dir with a dangling `.tmp` file (process killed between
    write_lexical_index's tmp-write and its rename) never finished
    publishing and was never `current` — safe to remove outright, even if
    it would otherwise fall inside the keep-N window by recency."""
    generations_dir = tmp_path / "generations"
    _make_generation(generations_dir, "20260101T000000Z-a")
    _make_generation(generations_dir, "20260102T000000Z-b", incomplete=True)

    current = tmp_path / "current"
    current.symlink_to(generations_dir / "20260101T000000Z-a", target_is_directory=True)

    removed = publish.prune_old_generations(generations_dir, current, keep=5)

    remaining = {p.name for p in generations_dir.iterdir()}
    assert "20260102T000000Z-b" not in remaining
    assert "20260101T000000Z-a" in remaining
    assert "20260102T000000Z-b" in removed


def test_prune_keeps_legacy_generation_with_graph_and_manifest_only(tmp_path):
    """A generation written via the compat wrapper `write_generation` carries
    only global-graph.json + generation-manifest.json (no artifact_sha256
    map) — it is a valid rollback target and must NOT be treated as
    incomplete."""
    generations_dir = tmp_path / "generations"
    legacy = generations_dir / "20260101T000000Z-legacy"
    legacy.mkdir(parents=True)
    (legacy / "global-graph.json").write_text("{}", encoding="utf-8")
    (legacy / "generation-manifest.json").write_text("{}", encoding="utf-8")
    _make_generation(generations_dir, "20260102T000000Z-b")

    current = tmp_path / "current"
    current.symlink_to(generations_dir / "20260102T000000Z-b", target_is_directory=True)

    removed = publish.prune_old_generations(generations_dir, current, keep=2)

    remaining = {p.name for p in generations_dir.iterdir()}
    assert legacy.name in remaining
    assert removed == []


def test_prune_removes_generation_whose_manifest_declares_missing_artifact(tmp_path):
    """If the manifest's `artifact_sha256` map names an artifact that is
    absent on disk, the generation never finished its publish sequence and
    is incomplete regardless of keep count."""
    generations_dir = tmp_path / "generations"
    broken = generations_dir / "20260101T000000Z-broken"
    broken.mkdir(parents=True)
    (broken / "global-graph.json").write_text("{}", encoding="utf-8")
    manifest = {"artifact_sha256": {"global-graph.json": "x", "lexical-index.json": "y"}}
    (broken / "generation-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _make_generation(generations_dir, "20260102T000000Z-b")

    current = tmp_path / "current"
    current.symlink_to(generations_dir / "20260102T000000Z-b", target_is_directory=True)

    removed = publish.prune_old_generations(generations_dir, current, keep=5)

    remaining = {p.name for p in generations_dir.iterdir()}
    assert broken.name not in remaining
    assert broken.name in removed


def _publish_full_generation(global_dir: Path, generation_id: str, marker: str) -> Path:
    """Drive the module's public publish sequence end-to-end, exactly in
    pipeline order: stage every artifact into the generation dir, then flip
    `current`. Uses only public entry points so publish.py-internal
    refactors don't invalidate the test."""
    generations_dir = global_dir / "generations"
    graph_data = {"nodes": [{"id": marker}], "links": []}
    gen_dir = publish.write_generation(
        generations_dir, generation_id, graph_data, {"generation_id": generation_id}
    )
    publish.write_overlay(gen_dir, {"edges": []})
    publish.write_lexical_index(gen_dir, {"marker": marker})
    publish.flip_current(global_dir, gen_dir)
    return gen_dir


def test_crash_between_generation_write_and_flip_keeps_old_current_intact(tmp_path, monkeypatch):
    """Atomicity contract: a publish that dies AFTER the new generation dir
    is fully written but BEFORE the `current` symlink flips must leave
    `current` pointing at the previous complete generation, with all of its
    artifacts still loadable. `current` is only ever touched by the flip
    step itself — everything staged before it is invisible to readers."""
    global_dir = tmp_path / "global"
    generations_dir = global_dir / "generations"

    # First publish succeeds for real: this is the "old" good generation.
    old_gen_id = "20260101T000000Z-aaaaaaaa-000000"
    _publish_full_generation(global_dir, old_gen_id, marker="old")

    # Second publish: fault injected at the symlink-flip step, AFTER all the
    # new generation's artifacts have been written.
    real_flip = publish.flip_current

    def _flip_that_crashes(global_dir_arg, gen_dir_arg):
        raise OSError("simulated crash before the current symlink flipped")

    monkeypatch.setattr(publish, "flip_current", _flip_that_crashes)
    new_gen_id = "20260102T000000Z-bbbbbbbb-000000"
    with pytest.raises(OSError, match="simulated crash"):
        _publish_full_generation(global_dir, new_gen_id, marker="new")
    monkeypatch.setattr(publish, "flip_current", real_flip)

    # The new generation's contents really were written before the crash...
    assert (generations_dir / new_gen_id / "global-graph.json").exists()

    # ...but `current` still points at the OLD complete generation.
    current = global_dir / "current"
    assert current.is_symlink()
    assert current.resolve() == (generations_dir / old_gen_id).resolve()

    # And every artifact of the old generation still loads through the
    # module's own read-side entry points.
    manifest = publish.read_current_manifest(global_dir)
    assert manifest is not None
    assert manifest["generation_id"] == old_gen_id
    graph = publish.read_current_global_graph(global_dir)
    assert graph is not None
    assert graph["nodes"] == [{"id": "old"}]
    overlay = publish.read_current_overlay(global_dir)
    assert overlay == {"edges": []}
    lexical = publish.read_current_lexical_index(global_dir)
    assert lexical == {"marker": "old"}
    # Raw bytes behind `current` are valid JSON too (no half-written file).
    raw = (current / "global-graph.json").read_text(encoding="utf-8")
    assert json.loads(raw)["nodes"][0]["id"] == "old"


def test_crash_inside_flip_after_tmp_symlink_before_rename_keeps_old_current(tmp_path, monkeypatch):
    """The window INSIDE flip_current: the staged `.current.tmp` symlink has
    been created but the process dies before the atomic rename onto
    `current`. `current` must still resolve to the old complete generation,
    and a subsequent successful publish must recover — including cleaning up
    the stale `.current.tmp` left behind by the crash."""
    global_dir = tmp_path / "global"
    generations_dir = global_dir / "generations"

    old_gen_id = "20260101T000000Z-aaaaaaaa-000000"
    _publish_full_generation(global_dir, old_gen_id, marker="old")

    # Fault only the tmp-symlink -> `current` rename (the exact primitive
    # flip_current uses at publish.py's os.rename call), never the
    # tmp-json -> .json renames done while staging artifact files.
    real_rename = publish.os.rename

    def _rename_crashing_on_current_flip(src, dst):
        if str(src).endswith(".current.tmp"):
            raise OSError("simulated crash between tmp symlink and rename onto current")
        real_rename(src, dst)

    monkeypatch.setattr(publish.os, "rename", _rename_crashing_on_current_flip)
    crashed_gen_id = "20260102T000000Z-bbbbbbbb-000000"
    with pytest.raises(OSError, match="simulated crash"):
        _publish_full_generation(global_dir, crashed_gen_id, marker="crashed")
    monkeypatch.setattr(publish.os, "rename", real_rename)

    # The crash left the staged tmp symlink behind...
    tmp_link = global_dir / ".current.tmp"
    assert tmp_link.is_symlink()

    # ...but `current` still resolves to the OLD complete generation and its
    # artifacts load.
    current = global_dir / "current"
    assert current.resolve() == (generations_dir / old_gen_id).resolve()
    manifest = publish.read_current_manifest(global_dir)
    assert manifest is not None
    assert manifest["generation_id"] == old_gen_id
    graph = publish.read_current_global_graph(global_dir)
    assert graph is not None
    assert graph["nodes"] == [{"id": "old"}]

    # A subsequent successful publish recovers: flips `current` to the new
    # generation and removes the stale `.current.tmp`.
    new_gen_id = "20260103T000000Z-cccccccc-000000"
    _publish_full_generation(global_dir, new_gen_id, marker="new")
    assert current.resolve() == (generations_dir / new_gen_id).resolve()
    assert not tmp_link.is_symlink()
    assert not tmp_link.exists()
    graph = publish.read_current_global_graph(global_dir)
    assert graph["nodes"] == [{"id": "new"}]


def test_prune_no_generations_dir_is_a_no_op(tmp_path):
    removed = publish.prune_old_generations(
        tmp_path / "does-not-exist", tmp_path / "current", keep=2
    )
    assert removed == []


# ---------------------------------------------------------------------------
# prune_dated_snapshots: graphify's pre-overwrite `<collection>/<YYYY-MM-DD>/`
# backups (written by upstream backup_if_protected) had no retention at all.
# ---------------------------------------------------------------------------

TODAY = date(2026, 10, 14)


def _make_snapshot(collection: Path, name: str) -> Path:
    snap = collection / name
    snap.mkdir(parents=True)
    (snap / "graph.json").write_text("{}", encoding="utf-8")
    return snap


def test_prune_dated_snapshots_deletes_dirs_older_than_keep_days(tmp_path):
    collection = tmp_path / "cem" / "hub"
    for name in ["2026-10-01", "2026-10-08", "2026-10-13"]:
        _make_snapshot(collection, name)

    removed = publish.prune_dated_snapshots(collection, keep_days=5, today=TODAY)

    assert sorted(removed) == ["2026-10-01", "2026-10-08"]
    assert {p.name for p in collection.iterdir()} == {"2026-10-13"}


def test_prune_dated_snapshots_keeps_dir_exactly_keep_days_old(tmp_path):
    collection = tmp_path / "c"
    _make_snapshot(collection, "2026-10-09")  # exactly 5 days old
    _make_snapshot(collection, "2026-10-08")  # 6 days old
    _make_snapshot(collection, "2026-10-14")

    removed = publish.prune_dated_snapshots(collection, keep_days=5, today=TODAY)

    assert removed == ["2026-10-08"]
    assert {p.name for p in collection.iterdir()} == {"2026-10-09", "2026-10-14"}


def test_prune_dated_snapshots_keeps_newest_even_when_all_are_old(tmp_path):
    collection = tmp_path / "c"
    for name in ["2026-08-01", "2026-09-01", "2026-09-15"]:
        _make_snapshot(collection, name)

    removed = publish.prune_dated_snapshots(collection, keep_days=5, today=TODAY)

    assert sorted(removed) == ["2026-08-01", "2026-09-01"]
    assert {p.name for p in collection.iterdir()} == {"2026-09-15"}


def test_prune_dated_snapshots_ignores_non_matching_names(tmp_path):
    collection = tmp_path / "c"
    untouched = [
        "cache",
        "chunks",
        "2026-10-12T18-18-17Z",
        "20261014T104024Z-abcdef12-000000",
        "2026-13-45",  # matches the pattern but is not a real date
        "2026-1-01",
    ]
    for name in untouched:
        _make_snapshot(collection, name)
    (collection / "2026-01-01").write_text("a file, not a dir", encoding="utf-8")
    _make_snapshot(collection, "2026-10-14")

    removed = publish.prune_dated_snapshots(collection, keep_days=5, today=TODAY)

    assert removed == []
    remaining = {p.name for p in collection.iterdir()}
    assert remaining == set(untouched) | {"2026-01-01", "2026-10-14"}


def test_prune_dated_snapshots_skips_symlinks_named_like_a_date(tmp_path):
    collection = tmp_path / "c"
    target = _make_snapshot(tmp_path / "elsewhere", "real")
    collection.mkdir()
    (collection / "2026-01-01").symlink_to(target, target_is_directory=True)
    _make_snapshot(collection, "2026-10-14")

    removed = publish.prune_dated_snapshots(collection, keep_days=5, today=TODAY)

    assert removed == []
    assert (collection / "2026-01-01").is_symlink()
    assert (target / "graph.json").is_file()


def test_prune_dated_snapshots_symlink_is_not_counted_as_newest(tmp_path):
    """A symlink named with a later date must not take the 'newest' slot and
    thereby expose the real newest snapshot to deletion."""
    collection = tmp_path / "c"
    target = _make_snapshot(tmp_path / "elsewhere", "real")
    _make_snapshot(collection, "2026-09-01")
    (collection / "2026-10-14").symlink_to(target, target_is_directory=True)

    removed = publish.prune_dated_snapshots(collection, keep_days=5, today=TODAY)

    assert removed == []
    assert (collection / "2026-09-01").is_dir()


def test_prune_dated_snapshots_keep_days_zero_deletes_nothing(tmp_path):
    collection = tmp_path / "c"
    for name in ["2026-01-01", "2026-05-01", "2026-10-14"]:
        _make_snapshot(collection, name)

    removed = publish.prune_dated_snapshots(collection, keep_days=0, today=TODAY)

    assert removed == []
    assert len(list(collection.iterdir())) == 3


def test_prune_dated_snapshots_missing_collection_dir_returns_empty(tmp_path):
    assert publish.prune_dated_snapshots(tmp_path / "nope", keep_days=5, today=TODAY) == []


def test_prune_dated_snapshots_rmtree_failure_is_logged_and_does_not_abort(
    tmp_path, monkeypatch, caplog
):
    collection = tmp_path / "c"
    for name in ["2026-01-01", "2026-02-01", "2026-10-14"]:
        _make_snapshot(collection, name)
    real_rmtree = publish.shutil.rmtree

    def _rmtree(path, *args, **kwargs):
        if Path(path).name == "2026-01-01":
            raise PermissionError("simulated")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(publish.shutil, "rmtree", _rmtree)

    with caplog.at_level("WARNING"):
        removed = publish.prune_dated_snapshots(collection, keep_days=5, today=TODAY)

    assert removed == ["2026-02-01"]
    assert (collection / "2026-01-01").is_dir()
    assert "2026-01-01" in caplog.text


def test_prune_dated_snapshots_unlistable_collection_returns_empty(tmp_path, monkeypatch, caplog):
    collection = tmp_path / "c"
    _make_snapshot(collection, "2026-01-01")
    real_iterdir = Path.iterdir

    def _iterdir(self):
        if self == collection:
            raise PermissionError("simulated")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", _iterdir)

    with caplog.at_level("WARNING"):
        removed = publish.prune_dated_snapshots(collection, keep_days=5, today=TODAY)

    assert removed == []
    assert str(collection) in caplog.text
