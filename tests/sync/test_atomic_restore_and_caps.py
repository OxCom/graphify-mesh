from __future__ import annotations

import tempfile
from pathlib import Path

from graphify_mesh.sync import source_cache, sync_project

# ---------------------------------------------------------------------------
# snapshot restore is atomic
# ---------------------------------------------------------------------------


def test_restore_snapshot_swaps_whole_file_and_leaves_no_temp(tmp_path):
    out_dir = tmp_path / "graphify-out"
    out_dir.mkdir()
    graph_path = out_dir / "graph.json"
    graph_path.write_text('{"nodes": []}', encoding="utf-8")
    snapshot_path = out_dir / "graph.json.bak"
    snapshot_payload = '{"nodes": [' + ", ".join(f'{{"id": {i}}}' for i in range(500)) + "]}"
    snapshot_path.write_text(snapshot_payload, encoding="utf-8")

    sync_project._restore_snapshot(snapshot_path, graph_path)

    assert graph_path.read_text(encoding="utf-8") == snapshot_payload
    assert sorted(p.name for p in out_dir.iterdir()) == ["graph.json", "graph.json.bak"]


def test_restore_snapshot_without_snapshot_is_a_noop(tmp_path):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text("original", encoding="utf-8")

    sync_project._restore_snapshot(None, graph_path)

    assert graph_path.read_text(encoding="utf-8") == "original"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["graph.json"]


# ---------------------------------------------------------------------------
# source cache byte bounds
# ---------------------------------------------------------------------------


def test_single_huge_line_is_truncated_and_total_bounded(tmp_path):
    source_cache.clear_source_cache()
    src = tmp_path / "bundle.min.js"
    src.write_text("x" * 10_000_000, encoding="utf-8")

    lines = source_cache.get_source_lines(Path(src), 4000)

    assert lines is not None
    assert len(lines) == 1
    assert len(lines[0]) == source_cache.SOURCE_CACHE_MAX_LINE_CHARS
    assert sum(len(line) for line in lines) <= source_cache.SOURCE_CACHE_MAX_FILE_CHARS


def test_file_total_cap_stops_reading(tmp_path):
    source_cache.clear_source_cache()
    line = "y" * 1000
    src = tmp_path / "wide.py"
    src.write_text("\n".join(line for _ in range(4000)), encoding="utf-8")

    lines = source_cache.get_source_lines(Path(src), 4000)

    assert lines is not None
    total = sum(len(item) for item in lines)
    assert total <= source_cache.SOURCE_CACHE_MAX_FILE_CHARS
    assert len(lines) < 4000


def test_file_total_cap_holds_when_line_length_does_not_divide_it(tmp_path):
    source_cache.clear_source_cache()
    line = "z" * source_cache.SOURCE_CACHE_MAX_LINE_CHARS
    src = tmp_path / "wide_uneven.py"
    src.write_text("\n".join(line for _ in range(300)), encoding="utf-8")

    lines = source_cache.get_source_lines(Path(src), 4000)

    assert lines is not None
    assert sum(len(item) for item in lines) <= source_cache.SOURCE_CACHE_MAX_FILE_CHARS


def test_trailing_line_without_newline_is_clamped_at_the_boundary(tmp_path):
    source_cache.clear_source_cache()
    line = "z" * source_cache.SOURCE_CACHE_MAX_LINE_CHARS
    full_lines = (
        source_cache.SOURCE_CACHE_MAX_FILE_CHARS // source_cache.SOURCE_CACHE_MAX_LINE_CHARS
    )
    remainder = source_cache.SOURCE_CACHE_MAX_FILE_CHARS - full_lines * len(line)
    src = tmp_path / "boundary.py"
    src.write_text(
        "".join(f"{line}\n" for _ in range(full_lines)) + "z" * (remainder + 100),
        encoding="utf-8",
    )

    lines = source_cache.get_source_lines(Path(src), 4000)

    assert lines is not None
    assert len(lines) == full_lines + 1
    assert len(lines[-1]) == remainder
    assert sum(len(item) for item in lines) == source_cache.SOURCE_CACHE_MAX_FILE_CHARS


def test_short_lines_are_unchanged(tmp_path):
    source_cache.clear_source_cache()
    src = tmp_path / "small.py"
    src.write_text("alpha\nbeta\ngamma", encoding="utf-8")

    assert source_cache.get_source_lines(Path(src), 4000) == ("alpha", "beta", "gamma")


# ---------------------------------------------------------------------------
# a refused launch leaves no snapshot behind
# ---------------------------------------------------------------------------


def test_snapshot_is_removed_when_the_invoker_raises(tmp_path, monkeypatch):
    """`ChildSandboxUnavailable` and `SandboxContainmentError` both come out of
    the invoker before it spawns anything. The mkstemp snapshot used to stay on
    disk, one file per refused launch, in a directory nothing sweeps."""
    import pytest

    from graphify_mesh.sync import graphify_cli
    from graphify_mesh.sync.state import SourceDigest

    collection = tmp_path / "coll"
    collection.mkdir()
    (collection / "graph.json").write_text('{"nodes": [], "edges": []}', encoding="utf-8")

    def refusing_invoker(*_args, **_kwargs):
        raise graphify_cli.ChildSandboxUnavailable("bwrap missing")

    monkeypatch.setitem(sync_project._INVOKERS, sync_project.ACTION_UPDATE, refusing_invoker)
    before = set(Path(tempfile.gettempdir()).glob("graphify-mesh-sync-snapshot-*"))

    with pytest.raises(graphify_cli.ChildSandboxUnavailable):
        sync_project.apply_action(
            "example-org.aaa",
            "graphify",
            tmp_path / "repo",
            collection,
            sync_project.ACTION_UPDATE,
            SourceDigest(code_hash="c", semantic_hash="s", file_count=1),
            tmp_path / "home",
        )

    after = set(Path(tempfile.gettempdir()).glob("graphify-mesh-sync-snapshot-*"))
    assert after == before
