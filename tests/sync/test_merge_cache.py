"""Merge-stage reuse: unchanged inputs skip `graphify merge-graphs` and give
the same merged bytes; any input, producer or child-env change runs the merge
again, and a producer the key cannot pin is never reused."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from graphify_mesh.sync import graphify_cli, merge_cache, publish, repo_tags
from graphify_mesh.sync.pipeline import run

_VERSION = "0.9.89"
_BIN = "/opt/graphify"


def _inputs(tmp_path: Path) -> list[Path]:
    paths = []
    for name in ("a", "b"):
        path = tmp_path / name / "graph.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"nodes": [{"id": name}], "links": []}), encoding="utf-8")
        paths.append(path)
    return paths


@pytest.fixture
def pinned(monkeypatch):
    monkeypatch.setattr(merge_cache, "child_runs_this_interpreter", lambda _bin: True)
    monkeypatch.setattr(merge_cache, "_installed_graphify", lambda: (_VERSION, "record-sha"))
    monkeypatch.setattr(repo_tags, "probed_binary_version", lambda _bin: _VERSION)


_ENV = {"PATH": "/usr/bin", "HOME": "/tmp/staging-1/graphify-home", "PYTHONPATH": "/site"}


def _key(paths: list[Path], env: dict | None = None) -> str | None:
    return merge_cache.merge_key(
        paths, merge_cache.snapshot_inputs(paths), _BIN, dict(_ENV if env is None else env)
    )


# --- key contents ----------------------------------------------------------------


def test_key_is_stable_for_unchanged_inputs(tmp_path, pinned):
    paths = _inputs(tmp_path)
    assert _key(paths) is not None
    assert _key(paths) == _key(paths)


def test_key_changes_with_input_content(tmp_path, pinned):
    paths = _inputs(tmp_path)
    before = _key(paths)
    paths[1].write_text(json.dumps({"nodes": [], "links": []}), encoding="utf-8")
    assert _key(paths) != before


def test_key_changes_with_input_order(tmp_path, pinned):
    paths = _inputs(tmp_path)
    assert _key(paths) != _key(list(reversed(paths)))


@pytest.mark.parametrize(
    "change",
    [
        {"PATH": "/other/bin"},
        {"PYTHONPATH": "/elsewhere"},
        {"LC_ALL": "C"},
        {"OLLAMA_HOST": "x"},
        {"GRAPHIFY_MAX_GRAPH_BYTES": "10"},
    ],
)
def test_key_changes_with_any_child_env_variable(tmp_path, pinned, change):
    paths = _inputs(tmp_path)
    assert _key(paths, {**_ENV, **change}) != _key(paths)


def test_key_ignores_per_run_staging_home_path(tmp_path, pinned):
    paths = _inputs(tmp_path)
    assert _key(paths, {**_ENV, "HOME": "/tmp/staging-2/graphify-home"}) == _key(paths)


def test_merge_graphs_env_is_what_the_child_receives(monkeypatch, tmp_path):
    """The key hashes graphify_cli's own env builder, filter included."""
    monkeypatch.setenv("GRAPHIFY_MESH_SECRET_TOKEN", "never-in-child")
    monkeypatch.setenv("LC_ALL", "C")
    monkeypatch.setenv("GRAPHIFY_MESH_CHILD_ENV_EXTRA", "MY_EXTRA")
    monkeypatch.setenv("MY_EXTRA", "1")
    captured: dict = {}

    def fake_run(argv, **kwargs):
        captured.update(kwargs["env"])
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(graphify_cli.subprocess, "run", fake_run)
    graphify_cli.run_merge_graphs("graphify", [tmp_path / "g.json"], tmp_path / "o", tmp_path)
    env = graphify_cli.merge_graphs_env(tmp_path)
    assert env == captured
    assert "GRAPHIFY_MESH_SECRET_TOKEN" not in env
    assert env["LC_ALL"] == "C" and env["MY_EXTRA"] == "1"


def test_snapshot_detects_mtime_only_change(tmp_path):
    paths = _inputs(tmp_path)
    before = merge_cache.snapshot_inputs(paths)
    st = paths[0].stat()
    os.utime(paths[0], ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    after = merge_cache.snapshot_inputs(paths)
    assert before is not None and after is not None
    assert after.hashes == before.hashes
    assert after != before


def test_no_key_for_unreadable_input(tmp_path, pinned):
    paths = _inputs(tmp_path)
    paths[0].unlink()
    assert merge_cache.snapshot_inputs(paths) is None
    assert _key(paths) is None


# --- producer pinning --------------------------------------------------------------


def test_no_key_when_binary_reports_another_version(tmp_path, pinned, monkeypatch):
    paths = _inputs(tmp_path)
    monkeypatch.setattr(repo_tags, "probed_binary_version", lambda _bin: "0.9.90")
    assert _key(paths) is None
    monkeypatch.setattr(repo_tags, "probed_binary_version", lambda _bin: None)
    assert _key(paths) is None


def test_no_key_when_child_runs_another_interpreter(tmp_path, pinned, monkeypatch):
    paths = _inputs(tmp_path)
    monkeypatch.setattr(merge_cache, "child_runs_this_interpreter", lambda _bin: False)
    assert _key(paths) is None


def test_no_key_when_install_cannot_be_pinned(tmp_path, pinned, monkeypatch):
    paths = _inputs(tmp_path)
    monkeypatch.setattr(merge_cache, "_installed_graphify", lambda: None)
    assert _key(paths) is None


def _script(tmp_path: Path, shebang: str) -> str:
    script = tmp_path / "graphify-bin"
    script.write_text(f"#!{shebang}\nprint('x')\n", encoding="utf-8")
    script.chmod(0o755)
    return str(script)


def test_shebang_on_this_interpreter_is_pinned(tmp_path):
    merge_cache.child_runs_this_interpreter.cache_clear()
    assert merge_cache.child_runs_this_interpreter(_script(tmp_path, sys.executable))


@pytest.mark.parametrize("shebang", ["/usr/bin/env python3", "/nonexistent/bin/python"])
def test_env_or_foreign_shebang_is_not_pinned(tmp_path, shebang):
    merge_cache.child_runs_this_interpreter.cache_clear()
    assert not merge_cache.child_runs_this_interpreter(_script(tmp_path, shebang))


def test_other_venv_on_same_base_interpreter_is_not_pinned(tmp_path):
    """A venv's python resolves to the same realpath; its prefix differs."""
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(os.path.realpath(sys.executable))
    merge_cache.child_runs_this_interpreter.cache_clear()
    assert not merge_cache.child_runs_this_interpreter(_script(tmp_path, str(venv_python)))


def test_multi_word_bin_is_not_pinned():
    merge_cache.child_runs_this_interpreter.cache_clear()
    assert not merge_cache.child_runs_this_interpreter(f"{sys.executable} -m graphify")


class _FakeDist:
    def __init__(self, direct_url: str | None, origin: str) -> None:
        self.version = _VERSION
        self._direct_url = direct_url
        self._origin = origin

    def read_text(self, name: str) -> str | None:
        return {"RECORD": "graphify/__init__.py,sha256=x,1\n", "direct_url.json": self._direct_url}[
            name
        ]

    def locate_file(self, _path: str) -> str:
        return self._origin


def _real_origin() -> str:
    spec = importlib.util.find_spec("graphify")
    assert spec is not None and spec.origin is not None
    return spec.origin


@pytest.mark.parametrize(
    ("direct_url", "pinned_ok"),
    [
        (None, True),
        (json.dumps({"url": "file:///src/graphify", "dir_info": {}}), True),
        (json.dumps({"url": "file:///src/graphify", "dir_info": {"editable": True}}), False),
        ("not json", False),
    ],
)
def test_editable_install_is_not_pinned(monkeypatch, direct_url, pinned_ok):
    dist = _FakeDist(direct_url, _real_origin())
    monkeypatch.setattr(merge_cache.importlib.metadata, "distribution", lambda _name: dist)
    merge_cache._installed_graphify.cache_clear()
    try:
        assert (merge_cache._installed_graphify() is not None) is pinned_ok
    finally:
        merge_cache._installed_graphify.cache_clear()


def test_graphify_imported_from_elsewhere_is_not_pinned(monkeypatch, tmp_path):
    dist = _FakeDist(None, str(tmp_path / "other" / "graphify" / "__init__.py"))
    monkeypatch.setattr(merge_cache.importlib.metadata, "distribution", lambda _name: dist)
    merge_cache._installed_graphify.cache_clear()
    try:
        assert merge_cache._installed_graphify() is None
    finally:
        merge_cache._installed_graphify.cache_clear()


def test_parity_check_records_binary_version(monkeypatch):
    repo_tags.reset_version_parity_cache()
    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: _VERSION)
    monkeypatch.setattr(repo_tags, "_binary_graphify_version", lambda _bin: _VERSION)
    repo_tags.check_graphify_version_parity("/bin/graphify")
    assert repo_tags.probed_binary_version("/bin/graphify") == _VERSION
    repo_tags.reset_version_parity_cache()
    assert repo_tags.probed_binary_version("/bin/graphify") is None


# --- pipeline ----------------------------------------------------------------------


@pytest.fixture
def cached_env(env, monkeypatch):
    """Two repos, with the producer pinned so the merge key can be computed."""
    repo_tags.reset_version_parity_cache()
    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: _VERSION)
    monkeypatch.setattr(repo_tags, "_binary_graphify_version", lambda _bin: _VERSION)
    monkeypatch.setattr(merge_cache, "child_runs_this_interpreter", lambda _bin: True)
    monkeypatch.setattr(merge_cache, "_installed_graphify", lambda: (_VERSION, "record-sha"))
    # Naming exports GRAPHIFY_API_TIMEOUT into os.environ after the merge, so a
    # second run in the same process would see another child env. Production
    # runs one sync per process; the env's part of the key is tested above.
    monkeypatch.setattr(graphify_cli, "merge_graphs_env", lambda _home: {})
    env.add_repo("a", "prod", "a", "proj-a", "repo_a.json")
    env.add_repo("b", "prod", "b", "proj-b", "repo_b.json")
    env.write_registry()
    yield env
    repo_tags.reset_version_parity_cache()


@pytest.fixture
def merge_calls(monkeypatch):
    calls: list[list[Path]] = []
    real = graphify_cli.run_merge_graphs

    def recording(graphify_bin, paths, *args, **kwargs):
        calls.append(list(paths))
        return real(graphify_bin, paths, *args, **kwargs)

    monkeypatch.setattr(graphify_cli, "run_merge_graphs", recording)
    return calls


def _entries(env) -> list[Path]:
    return list((env.settings().cache_dir / merge_cache.MERGE_CACHE_SUBDIR).glob("*.bin"))


def _published(env) -> tuple[str, bytes]:
    settings = env.settings(shrink_policy="count")
    manifest = publish.read_current_manifest(settings.global_dir)
    assert manifest is not None
    graph = (settings.global_dir / "current" / "global-graph.json").read_bytes()
    return manifest["output_hash"], graph


def test_unchanged_inputs_reuse_merge_and_publish_identical_graph(cached_env, merge_calls):
    env = cached_env
    first = run(env.settings(shrink_policy="count"))
    assert first.published
    assert len(merge_calls) == 1
    assert len(_entries(env)) == 1
    hash_1, graph_1 = _published(env)

    second = run(env.settings(shrink_policy="count"))

    assert second.published
    assert second.merge_ok
    assert len(merge_calls) == 1, "unchanged inputs must not run graphify merge-graphs"
    hash_2, graph_2 = _published(env)
    assert hash_2 == hash_1
    assert graph_2 == graph_1


def test_corrupt_entry_falls_back_to_merge(cached_env, merge_calls):
    env = cached_env
    run(env.settings(shrink_policy="count"))
    hash_1, graph_1 = _published(env)
    (entry,) = _entries(env)
    raw = bytearray(entry.read_bytes())
    raw[-1] ^= 0xFF
    entry.write_bytes(bytes(raw))

    report = run(env.settings(shrink_policy="count"))

    assert report.published
    assert len(merge_calls) == 2
    assert _published(env) == (hash_1, graph_1)


def test_changed_producer_runs_merge_and_replaces_entry(cached_env, merge_calls, monkeypatch):
    env = cached_env
    run(env.settings(shrink_policy="count"))
    before = {p.name for p in _entries(env)}
    monkeypatch.setattr(merge_cache, "_installed_graphify", lambda: (_VERSION, "reinstalled"))

    run(env.settings(shrink_policy="count"))

    assert len(merge_calls) == 2
    after = {p.name for p in _entries(env)}
    assert len(after) == 1 and after != before


def test_unpinned_producer_always_merges(cached_env, merge_calls, monkeypatch):
    env = cached_env
    monkeypatch.setattr(merge_cache, "child_runs_this_interpreter", lambda _bin: False)
    run(env.settings(shrink_policy="count"))
    run(env.settings(shrink_policy="count"))
    assert len(merge_calls) == 2
    assert not _entries(env)


def test_stage_cache_off_always_merges(cached_env, merge_calls):
    env = cached_env
    run(env.settings(shrink_policy="count", stage_cache=False))
    run(env.settings(shrink_policy="count", stage_cache=False))
    assert len(merge_calls) == 2
    assert not (env.settings().cache_dir / merge_cache.MERGE_CACHE_SUBDIR).exists()


@pytest.mark.parametrize("change", ["content", "mtime"])
def test_input_changed_during_merge_is_not_cached(cached_env, monkeypatch, change):
    env = cached_env
    real = graphify_cli.run_merge_graphs
    graph_a = env.collection_path("prod", "a") / "graph.json"

    def racing(graphify_bin, paths, *args, **kwargs):
        result = real(graphify_bin, paths, *args, **kwargs)
        if change == "content":
            graph_a.write_bytes(graph_a.read_bytes() + b" ")
        else:
            st = graph_a.stat()
            os.utime(graph_a, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        return result

    monkeypatch.setattr(graphify_cli, "run_merge_graphs", racing)

    report = run(env.settings(shrink_policy="count"))

    # Published, so the cache flush ran: the entry is missing because of the guard.
    assert report.published
    assert not _entries(env)


def test_failed_publish_discards_spooled_entry(cached_env, monkeypatch):
    env = cached_env

    def broken_flip(*_args, **_kwargs):
        raise OSError("flip failed")

    monkeypatch.setattr(publish, "flip_current", broken_flip)

    with pytest.raises(OSError, match="flip failed"):
        run(env.settings(shrink_policy="count"))

    assert not _entries(env)
    assert not list(env.settings().cache_dir.glob(".staging-*"))
