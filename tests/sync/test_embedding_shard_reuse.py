"""Shard meta `inputs_digest`/`matrix_sha256`, the re-hash skip for unchanged
inputs, and hard-linked reuse of unchanged shards across generations."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from graphify_mesh.sync import embedding
from graphify_mesh.sync.config import Settings, categorize_file
from graphify_mesh.sync.pipeline import run
from graphify_mesh.sync.source_cache import clear_source_cache

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
FAKE_GRAPHIFY = FIXTURES_DIR / "fake_graphify" / "graphify"
REPO = "repo.a"
SHARD_FILES = (f"{REPO}.meta.json", f"{REPO}.npy")


def _settings(tmp_path: Path, **overrides) -> Settings:
    mesh_root = tmp_path / "mesh"
    return Settings.from_env(
        mesh_root=mesh_root,
        scan_roots=[tmp_path / "www"],
        registry_path=mesh_root / "bin" / "registry.json",
        graphify_bin=str(FAKE_GRAPHIFY),
        **overrides,
    )


def _repo(tmp_path: Path) -> tuple[Path, Path, dict]:
    root = tmp_path / "src-root"
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    (root / "src" / "b.py").write_text("def beta():\n    return 1\n", encoding="utf-8")
    graph = {
        "nodes": [
            {"id": "n1", "label": "alpha", "source_file": "src/a.py"},
            {"id": "n2", "label": "beta", "source_file": "src/b.py"},
        ]
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    return root, graph_path, graph


def _sha(path: Path) -> bytes:
    return hashlib.sha256(path.read_bytes()).digest()


def _fresh(base: Path) -> Path:
    """One path for both the on and off runs: inputs_digest folds the repo root."""
    shutil.rmtree(base, ignore_errors=True)
    return base


def _pin_mtimes(root: Path, ns: int) -> None:
    """Same mtimes in the on and off trees: inputs_digest folds them in."""
    for path in root.rglob("*"):
        if path.is_file():
            os.utime(path, ns=(ns, ns))


def _embed(tmp_path, settings, root, graph_path, graph, digest, unchanged, gen):
    """One embed stage + staging + persist, as the pipeline runs them."""
    clear_source_cache()
    result = embedding.run_embedding_stage(
        graph_paths_by_repo={REPO: graph_path},
        repo_roots_by_id={REPO: root},
        graphs_by_repo={REPO: graph},
        unchanged_repo_ids={REPO} if unchanged else set(),
        settings=settings,
        provisional_generation_id=f"provisional-{gen}",
        health_check=lambda base_url, timeout: True,
        source_digests_by_repo={REPO: digest},
        graph_sha256_by_repo={REPO: _sha(graph_path)},
    )
    staged = embedding.stage_embeddings(
        tmp_path / f"staging-{gen}",
        result.shards_by_repo,
        result.id_map,
        skip_repo_ids=result.reused_repo_ids,
    )
    embedding.persist_generation(
        settings.embeddings_dir,
        gen,
        staged,
        reused_shards={r: result.shards_by_repo[r] for r in result.reused_repo_ids},
    )
    return result, staged


def _fake_embed(monkeypatch, calls: list[str] | None = None) -> None:
    def fake(base_url, model, inputs, timeout=30.0):
        if calls is not None:
            calls.extend(inputs)
        return [[float(len(text)), 1.0] for text in inputs]

    monkeypatch.setattr(embedding, "embed_batch", fake)


def _no_rehash(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("unchanged inputs must not re-hash the shard")

    monkeypatch.setattr(embedding, "_shard_content_current", fail)


def _gen(settings, gen: str) -> Path:
    return settings.embeddings_dir / "generations" / gen


def test_meta_records_inputs_digest_and_matrix_sha256(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    root, graph_path, graph = _repo(tmp_path)
    _fake_embed(monkeypatch)

    result, _ = _embed(tmp_path, settings, root, graph_path, graph, "d1", False, "gen-0")

    meta = json.loads((_gen(settings, "gen-0") / SHARD_FILES[0]).read_text(encoding="utf-8"))
    stamp = embedding.recipe_stamp_for_model(settings.ollama_embed_model)
    assert meta["shard_format"] == 3
    assert meta["inputs_digest"] == embedding.inputs_digest(
        _sha(graph_path), "d1", stamp, graph, root
    )
    assert meta["matrix_sha256"] == embedding.shard_matrix_sha256(
        result.shards_by_repo[REPO].vectors
    )
    matrix = np.load(_gen(settings, "gen-0") / SHARD_FILES[1])
    expected = hashlib.sha256(
        "\n".join(sorted(meta["entries"])).encode("utf-8") + matrix.astype(np.float32).tobytes()
    ).hexdigest()
    assert meta["matrix_sha256"] == expected


def test_inputs_digest_changes_with_each_input(tmp_path):
    root, graph_path, graph = _repo(tmp_path)

    graph_sha = _sha(graph_path)

    def digest(sha: bytes | None = graph_sha, source: str | None = "d1", stamp: str = "s1"):
        return embedding.inputs_digest(sha, source, stamp, graph, root)

    base = digest()
    assert base is not None
    assert digest(source="d2") != base
    assert digest(stamp="s2") != base
    assert embedding.inputs_digest(_sha(graph_path), "d1", "s1", graph, None) != base
    (root / "src" / "a.py").write_text("def alpha():\n    return 10\n", encoding="utf-8")
    after_edit = digest()
    assert after_edit != base
    (root / "src" / "a.py").unlink()
    assert digest() not in (base, after_edit)
    graph_path.write_text(graph_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert digest(sha=_sha(graph_path)) not in (base, digest())
    assert digest(sha=None) is None
    assert digest(source=None) is None


def test_edit_to_a_graphed_ignore_class_file_misses_and_matches_cache_off(tmp_path, monkeypatch):
    # A graphed file whose extension categorize_file classes "ignore" is not in
    # the source digest, but its snippet feeds the embedding input.
    assert categorize_file(Path("conf/app.zzz")) == "ignore"
    files: dict[str, bytes] = {}
    for label, cache in (("on", True), ("off", False)):
        base = _fresh(tmp_path / "w")
        settings = _settings(base, stage_cache=cache)
        root, graph_path, graph = _repo(base)
        (root / "conf").mkdir()
        (root / "conf" / "app.zzz").write_text("timeout = 5\n", encoding="utf-8")
        graph["nodes"].append({"id": "n3", "label": "timeout", "source_file": "conf/app.zzz"})
        graph_path.write_text(json.dumps(graph), encoding="utf-8")
        _pin_mtimes(root, 1_000_000_000)
        _fake_embed(monkeypatch)
        _embed(base, settings, root, graph_path, graph, "d1", False, "gen-0")
        old = json.loads((_gen(settings, "gen-0") / SHARD_FILES[0]).read_text(encoding="utf-8"))
        (root / "conf" / "app.zzz").write_text("timeout = 500\n", encoding="utf-8")
        _pin_mtimes(root, 2_000_000_000)
        embedded: list[str] = []
        _fake_embed(monkeypatch, embedded)

        # Same source digest and graph.json: only the stat of app.zzz moved.
        result, _ = _embed(base, settings, root, graph_path, graph, "d1", True, "gen-1")

        new = json.loads((_gen(settings, "gen-1") / SHARD_FILES[0]).read_text(encoding="utf-8"))
        assert new["inputs_digest"] != old["inputs_digest"]
        assert result.reused_repo_ids == set()
        assert len(embedded) == 1 and "timeout = 500" in embedded[0]
        files[label] = b"".join((_gen(settings, "gen-1") / n).read_bytes() for n in SHARD_FILES)

    assert files["on"] == files["off"]


def test_reused_shard_is_a_hard_link_with_cold_write_bytes(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    root, graph_path, graph = _repo(tmp_path)
    _fake_embed(monkeypatch)
    _embed(tmp_path, settings, root, graph_path, graph, "d1", False, "gen-0")
    _no_rehash(monkeypatch)

    result, staged = _embed(tmp_path, settings, root, graph_path, graph, "d1", True, "gen-1")

    assert result.reused_repo_ids == {REPO}
    assert result.stats.reused_repos_unchanged == 1
    for name in SHARD_FILES:
        assert not (staged / name).exists()
        linked = _gen(settings, "gen-1") / name
        assert linked.stat().st_nlink == 2
        assert linked.stat().st_ino == (_gen(settings, "gen-0") / name).stat().st_ino
    # A cold write of the same reused shard produces the same bytes.
    cold = embedding.stage_embeddings(tmp_path / "cold", result.shards_by_repo, result.id_map)
    for name in SHARD_FILES:
        assert (cold / name).read_bytes() == (_gen(settings, "gen-1") / name).read_bytes()


def test_linked_shard_still_reads_after_gc_of_the_old_generation(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    root, graph_path, graph = _repo(tmp_path)
    _fake_embed(monkeypatch)
    _embed(tmp_path, settings, root, graph_path, graph, "d1", False, "gen-0")
    _embed(tmp_path, settings, root, graph_path, graph, "d1", True, "gen-1")

    removed = embedding.gc_embedding_generations(settings.embeddings_dir, keep=1)

    assert removed == ["gen-0"]
    shard = embedding.read_previous_shard(settings.embeddings_dir / "current", REPO)
    assert len(shard.vectors) == 2
    assert shard.inputs_digest is not None
    assert (_gen(settings, "gen-1") / SHARD_FILES[1]).stat().st_nlink == 1


def test_source_edit_with_unchanged_graph_changes_digest_and_rehashes(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    root, graph_path, graph = _repo(tmp_path)
    _fake_embed(monkeypatch)
    _embed(tmp_path, settings, root, graph_path, graph, "d1", False, "gen-0")
    old_meta = json.loads((_gen(settings, "gen-0") / SHARD_FILES[0]).read_text(encoding="utf-8"))
    (root / "src" / "b.py").write_text("def beta():\n    return 22\n", encoding="utf-8")
    rehashed: list[str] = []
    real_check = embedding._shard_content_current

    def spy(repo_id, *args, **kwargs):
        rehashed.append(repo_id)
        return real_check(repo_id, *args, **kwargs)

    monkeypatch.setattr(embedding, "_shard_content_current", spy)
    embedded: list[str] = []
    _fake_embed(monkeypatch, embedded)

    # A deferred repo: graph.json untouched, the live source digest moved.
    result, _ = _embed(tmp_path, settings, root, graph_path, graph, "d2", True, "gen-1")

    assert rehashed == [REPO]
    assert result.reused_repo_ids == set()
    assert len(embedded) == 1 and "return 22" in embedded[0]
    new_meta = json.loads((_gen(settings, "gen-1") / SHARD_FILES[0]).read_text(encoding="utf-8"))
    assert new_meta["inputs_digest"] != old_meta["inputs_digest"]
    assert (_gen(settings, "gen-1") / SHARD_FILES[1]).stat().st_nlink == 1


def test_cache_off_writes_identical_bytes_and_links_nothing(tmp_path, monkeypatch):
    files: dict[str, dict[str, bytes]] = {}
    for label, cache in (("on", True), ("off", False)):
        base = _fresh(tmp_path / "w")
        settings = _settings(base, stage_cache=cache)
        root, graph_path, graph = _repo(base)
        _pin_mtimes(root, 1_000_000_000)
        _fake_embed(monkeypatch)
        _embed(base, settings, root, graph_path, graph, "d1", False, "gen-0")
        result, _ = _embed(base, settings, root, graph_path, graph, "d1", True, "gen-1")
        expected_links = 2 if cache else 1
        assert result.reused_repo_ids == ({REPO} if cache else set())
        for name in SHARD_FILES:
            assert (_gen(settings, "gen-1") / name).stat().st_nlink == expected_links
        files[label] = {name: (_gen(settings, "gen-1") / name).read_bytes() for name in SHARD_FILES}

    assert files["on"] == files["off"]


def test_skip_embedding_carry_forward_keeps_old_digest_so_next_run_rehashes(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    root, graph_path, graph = _repo(tmp_path)
    _fake_embed(monkeypatch)
    _embed(tmp_path, settings, root, graph_path, graph, "d1", False, "gen-0")
    old_meta = json.loads((_gen(settings, "gen-0") / SHARD_FILES[0]).read_text(encoding="utf-8"))
    (root / "src" / "b.py").write_text("def beta():\n    return 22\n", encoding="utf-8")

    # --skip-embedding: shards carried forward unverified, as the pipeline does.
    carried = embedding.carry_forward_stage({REPO: graph}, settings, "skipped", "skip")
    staged = embedding.stage_embeddings(tmp_path / "staging-skip", carried.shards_by_repo, {})
    embedding.persist_generation(settings.embeddings_dir, "gen-1", staged)
    carried_meta = json.loads(
        (_gen(settings, "gen-1") / SHARD_FILES[0]).read_text(encoding="utf-8")
    )
    assert carried_meta["inputs_digest"] == old_meta["inputs_digest"]

    embedded: list[str] = []
    _fake_embed(monkeypatch, embedded)
    result, _ = _embed(tmp_path, settings, root, graph_path, graph, "d2", True, "gen-2")

    assert result.reused_repo_ids == set()
    assert len(embedded) == 1 and "return 22" in embedded[0]


def test_pipeline_second_run_links_the_unchanged_shard(env, monkeypatch):
    env.add_repo("acme.core", "acme", "core", "core.acme.dev.lo", "repo_a.json")
    env.write_registry()
    _fake_embed(monkeypatch)
    settings = env.settings(ollama_embed_health_check=lambda *a, **kw: True)

    first = run(settings)
    assert first.published, first.publish_blocked_reason
    _no_rehash(monkeypatch)
    second = run(settings)

    assert second.published, second.publish_blocked_reason
    assert second.embedding_stats["reused_repos_unchanged"] == 1
    current = settings.embeddings_dir / "current"
    for name in ("acme.core.meta.json", "acme.core.npy"):
        assert (current / name).stat().st_nlink == 2


def test_link_failure_falls_back_to_a_copy(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    root, graph_path, graph = _repo(tmp_path)
    _fake_embed(monkeypatch)
    _embed(tmp_path, settings, root, graph_path, graph, "d1", False, "gen-0")

    def exdev(src, dst, *args, **kwargs):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(embedding.os, "link", exdev)
    result, _ = _embed(tmp_path, settings, root, graph_path, graph, "d1", True, "gen-1")

    assert result.reused_repo_ids == {REPO}
    for name in SHARD_FILES:
        copied = _gen(settings, "gen-1") / name
        assert copied.stat().st_nlink == 1
        assert copied.read_bytes() == (_gen(settings, "gen-0") / name).read_bytes()


def test_vanished_previous_shard_is_written_from_memory(tmp_path, monkeypatch, caplog):
    settings = _settings(tmp_path)
    root, graph_path, graph = _repo(tmp_path)
    _fake_embed(monkeypatch)
    _embed(tmp_path, settings, root, graph_path, graph, "d1", False, "gen-0")
    clear_source_cache()
    result = embedding.run_embedding_stage(
        graph_paths_by_repo={REPO: graph_path},
        repo_roots_by_id={REPO: root},
        graphs_by_repo={REPO: graph},
        unchanged_repo_ids={REPO},
        settings=settings,
        provisional_generation_id="provisional-gen-1",
        health_check=lambda base_url, timeout: True,
        source_digests_by_repo={REPO: "d1"},
        graph_sha256_by_repo={REPO: _sha(graph_path)},
    )
    assert result.reused_repo_ids == {REPO}
    staged = embedding.stage_embeddings(
        tmp_path / "staging-gen-1",
        result.shards_by_repo,
        result.id_map,
        skip_repo_ids=result.reused_repo_ids,
    )
    cold = embedding.stage_embeddings(tmp_path / "cold", result.shards_by_repo, result.id_map)
    # The matrix is already mmap'd, so the shard stays readable after the unlink.
    (_gen(settings, "gen-0") / SHARD_FILES[1]).unlink()

    embedding.persist_generation(
        settings.embeddings_dir, "gen-1", staged, reused_shards={REPO: result.shards_by_repo[REPO]}
    )

    assert "writing it from memory instead" in caplog.text
    for name in SHARD_FILES:
        written = _gen(settings, "gen-1") / name
        assert written.stat().st_nlink == 1
        assert written.read_bytes() == (cold / name).read_bytes()
