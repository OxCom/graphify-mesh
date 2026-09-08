from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from graphify_mesh.sync import embedding
from graphify_mesh.sync.config import Settings
from graphify_mesh.sync.vectors import RepoVectors

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
FAKE_GRAPHIFY = FIXTURES_DIR / "fake_graphify" / "graphify"


def _settings(tmp_path: Path, **overrides) -> Settings:
    mesh_root = tmp_path / "mesh"
    return Settings.from_env(
        mesh_root=mesh_root,
        scan_roots=[tmp_path / "www"],
        registry_path=mesh_root / "bin" / "registry.json",
        graphify_bin=str(FAKE_GRAPHIFY),
        **overrides,
    )


# ---------------------------------------------------------------------------
# snippet builder
# ---------------------------------------------------------------------------


def test_build_snippet_bounded_around_line(tmp_path):
    src = tmp_path / "big.py"
    src.write_text("\n".join(f"line {i}" for i in range(200)), encoding="utf-8")

    snippet = embedding.build_snippet(tmp_path, "big.py", line=100)

    lines = snippet.splitlines()
    assert len(lines) <= embedding.SNIPPET_WINDOW_LINES
    assert "line 100" in snippet or "line 99" in snippet  # window centers near the requested line
    assert "line 0" not in snippet
    assert "line 199" not in snippet


def test_build_snippet_max_chars_enforced(tmp_path):
    src = tmp_path / "wide.py"
    src.write_text("\n".join("x" * 200 for _ in range(20)), encoding="utf-8")

    snippet = embedding.build_snippet(tmp_path, "wide.py", line=1)
    assert len(snippet) <= embedding.SNIPPET_MAX_CHARS


def test_build_snippet_missing_file_returns_empty(tmp_path):
    assert embedding.build_snippet(tmp_path, "does-not-exist.py", line=1) == ""


def test_build_snippet_no_root_returns_empty():
    assert embedding.build_snippet(None, "any.py", line=1) == ""


def test_build_embedding_input_truncated_to_max_chars():
    huge_snippet = "y" * 10000
    text = embedding.build_embedding_input("MyLabel", huge_snippet, "src/x.py", "Some Community")
    assert len(text) <= embedding.EMBED_INPUT_MAX_CHARS
    assert text.startswith("label: MyLabel")


def test_build_embedding_input_no_snippet_still_includes_required_parts():
    text = embedding.build_embedding_input("MyLabel", "", "src/x.py", "Some Community")
    assert "label: MyLabel" in text
    assert "path: src/x.py" in text
    assert "community: Some Community" in text
    assert "snippet:" not in text


# ---------------------------------------------------------------------------
# skip heuristic
# ---------------------------------------------------------------------------


def test_trivial_getter_with_one_line_body_is_skipped():
    assert embedding.is_trivial_node("getName", "return self.name") is True


def test_getter_with_meaningful_body_is_not_skipped():
    snippet = "\n".join(f"    step_{i}()" for i in range(5))
    assert embedding.is_trivial_node("getName", snippet) is False


def test_non_accessor_label_is_never_trivial():
    assert embedding.is_trivial_node("computeInvoiceTotal", "return 1") is False


# ---------------------------------------------------------------------------
# batched /api/embed call — native contract, mocked HTTP layer only
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: dict):
        self._body = json.dumps(body).encode("utf-8")
        self.status = 200

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_embed_batch_request_shape_matches_verified_native_contract(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None, context=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(
            {
                "model": "qwen3-embedding:0.6b",
                "embeddings": [[0.1, 0.2], [0.3, 0.4]],
                "total_duration": 1,
                "load_duration": 1,
                "prompt_eval_count": 2,
            }
        )

    monkeypatch.setattr(embedding.urllib.request, "urlopen", fake_urlopen)

    vectors = embedding.embed_batch(
        "https://ollama.example.com:11434", "qwen3-embedding:0.6b", ["a", "b"]
    )

    # C9: native endpoint, NOT /v1/embeddings.
    assert captured["url"] == "https://ollama.example.com:11434/api/embed"
    assert captured["method"] == "POST"
    assert captured["body"] == {"model": "qwen3-embedding:0.6b", "input": ["a", "b"]}
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_batch_raises_on_shape_mismatch(monkeypatch):
    def fake_urlopen(req, timeout=None, context=None):
        return _FakeResponse(
            {"model": "x", "embeddings": [[0.1, 0.2]]}
        )  # only 1, but 2 inputs requested

    monkeypatch.setattr(embedding.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError):
        embedding.embed_batch("https://host", "model", ["a", "b"])


def test_embed_batch_empty_input_short_circuits_without_a_call(monkeypatch):
    def fake_urlopen(req, timeout=None, context=None):
        raise AssertionError("must not be called for empty input")

    monkeypatch.setattr(embedding.urllib.request, "urlopen", fake_urlopen)
    assert embedding.embed_batch("https://host", "model", []) == []


# ---------------------------------------------------------------------------
# id-map + tombstones (C27) across two synthetic generations
# ---------------------------------------------------------------------------


def test_id_map_tombstones_removed_node_instead_of_dropping():
    gen1_keys = {"repo.a\x1fsrc/a.py\x1fAlpha", "repo.a\x1fsrc/b.py\x1fBeta"}
    id_map_gen1 = embedding.build_id_map({}, gen1_keys, "gen-1")
    assert all(v["status"] == "active" for v in id_map_gen1.values())

    # Gen 2: "Beta" node removed.
    gen2_keys = {"repo.a\x1fsrc/a.py\x1fAlpha"}
    id_map_gen2 = embedding.build_id_map(id_map_gen1, gen2_keys, "gen-2")

    beta_key = "repo.a\x1fsrc/b.py\x1fBeta"
    assert beta_key in id_map_gen2  # never silently dropped
    assert id_map_gen2[beta_key]["status"] == "tombstoned"
    assert id_map_gen2[beta_key]["tombstoned_at"] == "gen-2"

    alpha_key = "repo.a\x1fsrc/a.py\x1fAlpha"
    assert id_map_gen2[alpha_key]["status"] == "active"


def test_id_map_reappearing_key_goes_back_to_active():
    keys_gen1 = {"repo.a\x1fsrc/a.py\x1fAlpha"}
    id_map_gen1 = embedding.build_id_map({}, keys_gen1, "gen-1")
    id_map_gen2 = embedding.build_id_map(id_map_gen1, set(), "gen-2")
    assert id_map_gen2["repo.a\x1fsrc/a.py\x1fAlpha"]["status"] == "tombstoned"

    id_map_gen3 = embedding.build_id_map(id_map_gen2, keys_gen1, "gen-3")
    assert id_map_gen3["repo.a\x1fsrc/a.py\x1fAlpha"]["status"] == "active"
    assert id_map_gen3["repo.a\x1fsrc/a.py\x1fAlpha"]["tombstoned_at"] is None


# ---------------------------------------------------------------------------
# resumable / changed-only recompute
# ---------------------------------------------------------------------------


def _graph_with_one_node(label="Widget", source_file="src/w.py", extra: str = "") -> dict:
    return {
        "nodes": [
            {
                "id": "n1",
                "label": label,
                "source_file": source_file,
                "community_name": "Comm" + extra,
            },
        ]
    }


def test_compute_repo_shard_reuses_unchanged_node_without_calling_embed(tmp_path, monkeypatch):
    graph = _graph_with_one_node()
    stats = embedding.EmbeddingStats()

    call_count = {"n": 0}

    def fake_embed_batch(base_url, model, inputs, timeout=30.0):
        call_count["n"] += 1
        return [[1.0, 2.0] for _ in inputs]

    monkeypatch.setattr(embedding, "embed_batch", fake_embed_batch)

    first_shard = embedding.compute_repo_shard(
        "repo.a",
        graph,
        tmp_path,
        previous_shard=embedding.RepoShard.empty(),
        base_url="https://host",
        model="m",
        stats=stats,
    )
    assert call_count["n"] == 1
    key = next(iter(first_shard.entries))
    assert list(first_shard.vectors.get(key)) == pytest.approx([1.0, 2.0])
    assert first_shard.entries[key]["row"] == 0

    # Second run, node completely unchanged -> content_hash matches, must be
    # reused without another embed_batch call.
    second_shard = embedding.compute_repo_shard(
        "repo.a",
        graph,
        tmp_path,
        previous_shard=first_shard,
        base_url="https://host",
        model="m",
        stats=stats,
    )
    assert call_count["n"] == 1  # unchanged: no new call
    assert list(second_shard.vectors.get(key)) == pytest.approx([1.0, 2.0])
    assert stats.reused == 1


def test_compute_repo_shard_recomputes_when_content_changes(tmp_path, monkeypatch):
    stats = embedding.EmbeddingStats()
    call_count = {"n": 0}

    def fake_embed_batch(base_url, model, inputs, timeout=30.0):
        call_count["n"] += 1
        return [[float(call_count["n"])] for _ in inputs]

    monkeypatch.setattr(embedding, "embed_batch", fake_embed_batch)

    first_graph = _graph_with_one_node(extra="A")
    first_shard = embedding.compute_repo_shard(
        "repo.a",
        first_graph,
        tmp_path,
        previous_shard=embedding.RepoShard.empty(),
        base_url="https://host",
        model="m",
        stats=stats,
    )
    assert call_count["n"] == 1

    # community_name changed -> embedding input hash changes -> must recompute.
    second_graph = _graph_with_one_node(extra="B")
    second_shard = embedding.compute_repo_shard(
        "repo.a",
        second_graph,
        tmp_path,
        previous_shard=first_shard,
        base_url="https://host",
        model="m",
        stats=stats,
    )
    assert call_count["n"] == 2
    key = next(iter(second_shard.entries))
    assert list(second_shard.vectors.get(key)) == pytest.approx([2.0])


def test_compute_repo_shard_skips_trivial_node_without_calling_embed(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    (src / "w.py").write_text("def getName(self):\n    return self.name\n", encoding="utf-8")
    graph = {
        "nodes": [
            {
                "id": "n1",
                "label": "getName",
                "source_file": "src/w.py",
                "line": 1,
                "community_name": "Comm",
            },
        ]
    }
    stats = embedding.EmbeddingStats()

    def fake_embed_batch(base_url, model, inputs, timeout=30.0):
        raise AssertionError("must not be called for a trivial node")

    monkeypatch.setattr(embedding, "embed_batch", fake_embed_batch)

    shard = embedding.compute_repo_shard(
        "repo.a",
        graph,
        tmp_path,
        previous_shard=embedding.RepoShard.empty(),
        base_url="https://host",
        model="m",
        stats=stats,
    )
    key = next(iter(shard.entries))
    assert shard.entries[key]["row"] is None
    assert shard.vectors.get(key) is None
    assert stats.skipped_trivial == 1


def test_compute_repo_shard_returns_repo_vectors(tmp_path, monkeypatch):
    graph = {
        "nodes": [
            {
                "id": "n1",
                "label": "computeAlpha",
                "source_file": "src/w.py",
                "community_name": "Comm",
            },
            {
                "id": "n2",
                "label": "computeBeta",
                "source_file": "src/w.py",
                "community_name": "Comm",
            },
        ]
    }
    stats = embedding.EmbeddingStats()

    def fake_embed_batch(base_url, model, inputs, timeout=30.0):
        return [[float(i), float(i) + 1.0] for i, _ in enumerate(inputs)]

    monkeypatch.setattr(embedding, "embed_batch", fake_embed_batch)

    shard = embedding.compute_repo_shard(
        "repo.a",
        graph,
        tmp_path,
        previous_shard=embedding.RepoShard.empty(),
        base_url="https://host",
        model="m",
        stats=stats,
    )
    assert isinstance(shard.vectors, RepoVectors)
    assert shard.vectors.matrix.dtype == np.float32
    for key, entry in shard.entries.items():
        if entry["row"] is None:
            continue
        assert shard.vectors.keys[entry["row"]] == key


# ---------------------------------------------------------------------------
# GC keeps exactly N generations
# ---------------------------------------------------------------------------


def test_gc_old_generations_keeps_only_last_two(tmp_path):
    generations_dir = tmp_path / "generations"
    for name in [
        "20260101T000000Z-a",
        "20260102T000000Z-b",
        "20260103T000000Z-c",
        "20260104T000000Z-d",
    ]:
        (generations_dir / name).mkdir(parents=True)

    removed = embedding.gc_old_generations(generations_dir, keep=2)

    remaining = sorted(p.name for p in generations_dir.iterdir())
    assert remaining == ["20260103T000000Z-c", "20260104T000000Z-d"]
    assert sorted(removed) == ["20260101T000000Z-a", "20260102T000000Z-b"]


def test_gc_old_generations_noop_when_within_limit(tmp_path):
    generations_dir = tmp_path / "generations"
    (generations_dir / "gen-1").mkdir(parents=True)
    removed = embedding.gc_old_generations(generations_dir, keep=2)
    assert removed == []
    assert (generations_dir / "gen-1").exists()


def test_gc_old_generations_pins_current_target_despite_sort_order(tmp_path):
    # Clock skew: the LIVE generation sorts oldest by name. GC must pin the
    # dir `current` points at, mirroring publish.prune_old_generations.
    generations_dir = tmp_path / "generations"
    for name in [
        "20260101T000000Z-live",
        "20260102T000000Z-b",
        "20260103T000000Z-c",
        "20260104T000000Z-d",
    ]:
        (generations_dir / name).mkdir(parents=True)
    current = tmp_path / "current"
    current.symlink_to(generations_dir / "20260101T000000Z-live", target_is_directory=True)

    removed = embedding.gc_old_generations(generations_dir, keep=2, current=current)

    assert (generations_dir / "20260101T000000Z-live").exists()
    assert "20260101T000000Z-live" not in removed
    assert "20260102T000000Z-b" in removed


# ---------------------------------------------------------------------------
# run_embedding_stage: unchanged-repo short-circuit + degraded fallback
# ---------------------------------------------------------------------------


def test_run_embedding_stage_reuses_whole_shard_for_unchanged_repo(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    # Seed a "previous published" shard by writing directly under embeddings_current_symlink target.
    prev_gen_dir = settings.embeddings_dir / "generations" / "gen-0"
    prev_gen_dir.mkdir(parents=True)
    (prev_gen_dir / "repo.a.json").write_text(
        json.dumps(
            {
                "repo_id": "repo.a",
                "entries": {
                    "repo.a\x1fsrc/w.py\x1fWidget": {"content_hash": "abc", "embedding": [9.0]}
                },
            }
        ),
        encoding="utf-8",
    )
    (prev_gen_dir / "id-map.json").write_text(json.dumps({}), encoding="utf-8")
    settings.embeddings_dir.mkdir(parents=True, exist_ok=True)
    (settings.embeddings_dir / "current").symlink_to(prev_gen_dir, target_is_directory=True)

    def fail_embed_batch(base_url, model, inputs, timeout=30.0):
        raise AssertionError("unchanged repo must not trigger any embed call")

    monkeypatch.setattr(embedding, "embed_batch", fail_embed_batch)

    graphs_by_repo = {"repo.a": _graph_with_one_node()}
    result = embedding.run_embedding_stage(
        graph_paths_by_repo={},
        repo_roots_by_id={"repo.a": tmp_path},
        graphs_by_repo=graphs_by_repo,
        unchanged_repo_ids={"repo.a"},
        settings=settings,
        provisional_generation_id="gen-1",
        health_check=lambda base_url, timeout: True,
    )

    assert result.status == embedding.EMBED_HEALTHY
    assert result.stats.reused_repos_unchanged == 1
    assert list(result.vectors_by_repo["repo.a"].get("repo.a\x1fsrc/w.py\x1fWidget")) == [9.0]


def test_run_embedding_stage_degraded_carries_forward_previous_vectors(tmp_path):
    settings = _settings(tmp_path)
    prev_gen_dir = settings.embeddings_dir / "generations" / "gen-0"
    prev_gen_dir.mkdir(parents=True)
    (prev_gen_dir / "repo.a.json").write_text(
        json.dumps(
            {
                "repo_id": "repo.a",
                "entries": {
                    "repo.a\x1fsrc/w.py\x1fWidget": {"content_hash": "abc", "embedding": [7.0]}
                },
            }
        ),
        encoding="utf-8",
    )
    (prev_gen_dir / "id-map.json").write_text(json.dumps({}), encoding="utf-8")
    (settings.embeddings_dir / "current").symlink_to(prev_gen_dir, target_is_directory=True)

    graphs_by_repo = {"repo.a": _graph_with_one_node()}
    result = embedding.run_embedding_stage(
        graph_paths_by_repo={},
        repo_roots_by_id={"repo.a": tmp_path},
        graphs_by_repo=graphs_by_repo,
        unchanged_repo_ids=set(),
        settings=settings,
        provisional_generation_id="gen-1",
        health_check=lambda base_url, timeout: False,
    )

    assert result.status == embedding.EMBED_DEGRADED
    assert list(result.vectors_by_repo["repo.a"].get("repo.a\x1fsrc/w.py\x1fWidget")) == [7.0]


def test_run_embedding_stage_mid_run_failure_is_partial_not_crash(tmp_path, monkeypatch):
    """Regression test: a real network failure DURING embedding (health
    check passed, then embed_batch itself raises — e.g. a DNS blip) must
    degrade the run, not propagate and crash the whole pipeline. repo.a is
    processed successfully before the failure; repo.b's embed_batch call
    fails; repo.c (not yet reached) must fall back to its previous shard
    rather than being silently skipped or crashing the process."""
    settings = _settings(tmp_path)
    prev_gen_dir = settings.embeddings_dir / "generations" / "gen-0"
    prev_gen_dir.mkdir(parents=True)
    (prev_gen_dir / "repo.c.json").write_text(
        json.dumps(
            {
                "repo_id": "repo.c",
                "entries": {
                    "repo.c\x1fsrc/z.py\x1fZeta": {"content_hash": "prevc", "embedding": [3.0]}
                },
            }
        ),
        encoding="utf-8",
    )
    (prev_gen_dir / "id-map.json").write_text(json.dumps({}), encoding="utf-8")
    (settings.embeddings_dir / "current").symlink_to(prev_gen_dir, target_is_directory=True)

    def flaky_embed_batch(base_url, model, inputs, timeout=30.0):
        if "network-blip-marker" in inputs[0]:
            raise RuntimeError(
                "embed_batch request failed: <urlopen error [Errno -2] Name or service not known>"
            )
        return [[1.0] for _ in inputs]

    monkeypatch.setattr(embedding, "embed_batch", flaky_embed_batch)

    def _graph_with_marker(marker: str) -> dict:
        return {
            "nodes": [
                {
                    "id": "n1",
                    "label": marker,
                    "source_file": "src/w.py",
                    "loc": "L1",
                    "community_name": "Widgets",
                }
            ]
        }

    graphs_by_repo = {
        "repo.a": _graph_with_marker("Alpha"),
        "repo.b": _graph_with_marker("network-blip-marker-Bravo"),
        "repo.c": _graph_with_marker("Gamma"),
    }
    result = embedding.run_embedding_stage(
        graph_paths_by_repo={},
        repo_roots_by_id={"repo.a": tmp_path, "repo.b": tmp_path, "repo.c": tmp_path},
        graphs_by_repo=graphs_by_repo,
        unchanged_repo_ids=set(),
        settings=settings,
        provisional_generation_id="gen-1",
        health_check=lambda base_url, timeout: True,
    )

    assert result.status == embedding.EMBED_PARTIAL
    assert "repo.b" in result.reason
    # repo.a completed for real before the failure.
    assert len(result.vectors_by_repo["repo.a"])
    # repo.c (not yet reached when the failure hit) fell back to its
    # previous published vector rather than being lost or crashing.
    assert list(result.vectors_by_repo["repo.c"].get("repo.c\x1fsrc/z.py\x1fZeta")) == [3.0]


# ---------------------------------------------------------------------------
# shard format v2: writer + version-aware reader
# ---------------------------------------------------------------------------


def test_stage_embeddings_writes_v2_meta_and_npy(tmp_path):
    vectors = RepoVectors.from_mapping({"repo.a\x1fsrc/w.py\x1fWidget": [1.0, 2.0]})
    shard = embedding.RepoShard(
        entries={"repo.a\x1fsrc/w.py\x1fWidget": {"content_hash": "abc", "row": 0}},
        vectors=vectors,
    )
    out_dir = embedding.stage_embeddings(tmp_path, {"repoA": shard}, id_map={})

    meta = json.loads((out_dir / "repoA.meta.json").read_text())
    assert meta["shard_format"] == embedding.SHARD_FORMAT_VERSION
    assert meta["repo_id"] == "repoA"
    assert meta["dim"] == 2

    matrix = np.load(out_dir / "repoA.npy")
    assert matrix.dtype == np.float32
    assert matrix.shape[0] == sum(1 for e in meta["entries"].values() if e["row"] is not None)


def test_read_previous_shard_v1_json_compat(tmp_path):
    (tmp_path / "repoA.json").write_text(
        json.dumps(
            {
                "repo_id": "repoA",
                "entries": {
                    "k1": {"content_hash": "h1", "embedding": [1.0, 0.0]},
                    "k2": {"content_hash": None, "embedding": None},
                },
            }
        ),
        encoding="utf-8",
    )
    shard = embedding.read_previous_shard(tmp_path, "repoA")
    assert shard.entries["k1"]["content_hash"] == "h1"
    assert shard.vectors.get("k1") is not None
    assert list(shard.vectors.get("k1")) == pytest.approx([1.0, 0.0])
    assert shard.entries["k2"]["row"] is None
    assert shard.vectors.get("k2") is None


def test_read_previous_shard_v2_roundtrip(tmp_path):
    vectors = RepoVectors.from_mapping({"k1": [1.0, 2.0], "k2": [3.0, 4.0]})
    shard = embedding.RepoShard(
        entries={
            "k1": {"content_hash": "h1", "row": 0},
            "k2": {"content_hash": "h2", "row": 1},
        },
        vectors=vectors,
    )
    out_dir = embedding.stage_embeddings(tmp_path, {"repoA": shard}, id_map={})

    read_back = embedding.read_previous_shard(out_dir, "repoA")
    assert read_back.entries["k1"]["content_hash"] == "h1"
    assert read_back.entries["k2"]["content_hash"] == "h2"
    assert list(read_back.vectors.get("k1")) == pytest.approx([1.0, 2.0])
    assert list(read_back.vectors.get("k2")) == pytest.approx([3.0, 4.0])


def test_read_previous_shard_v2_missing_npy_is_empty(tmp_path):
    vectors = RepoVectors.from_mapping({"k1": [1.0, 2.0]})
    shard = embedding.RepoShard(entries={"k1": {"content_hash": "h1", "row": 0}}, vectors=vectors)
    out_dir = embedding.stage_embeddings(tmp_path, {"repoA": shard}, id_map={})
    (out_dir / "repoA.npy").unlink()

    read_back = embedding.read_previous_shard(out_dir, "repoA")
    assert read_back.entries == {}
    assert len(read_back.vectors) == 0


# ---------------------------------------------------------------------------
# v2 reader hardening: reject malformed/adversarial meta+matrix combinations
# BEFORE building a RepoVectors from them, degrading to empty shard instead
# of raising or (worse) silently trusting bad row assignments.
# ---------------------------------------------------------------------------


def _write_raw_v2_shard(
    out_dir: Path, repo_id: str, entries: dict, matrix: np.ndarray, dim: int | None = None
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "repo_id": repo_id,
        "shard_format": embedding.SHARD_FORMAT_VERSION,
        "dim": matrix.shape[1] if dim is None else dim,
        "entries": entries,
    }
    (out_dir / f"{repo_id}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    np.save(out_dir / f"{repo_id}.npy", matrix)


def test_read_previous_shard_v2_duplicate_row_is_empty(tmp_path):
    # ONE-row matrix with BOTH entries claiming that same row 0: every row
    # (there's only one) is "owned", so the unowned-row guard alone would
    # let this through — only explicit duplicate-claim detection catches it.
    matrix = np.array([[1.0, 2.0]], dtype=np.float32)
    entries = {
        "k1": {"content_hash": "h1", "row": 0},
        "k2": {"content_hash": "h2", "row": 0},  # duplicate claim on row 0
    }
    _write_raw_v2_shard(tmp_path, "repoA", entries, matrix)

    shard = embedding.read_previous_shard(tmp_path, "repoA")
    assert shard.entries == {}
    assert len(shard.vectors) == 0


def test_read_previous_shard_v2_unhashable_shard_format_is_empty(tmp_path):
    # meta.get("shard_format") in SUPPORTED_SHARD_FORMATS (a frozenset) would
    # raise TypeError for an unhashable JSON value like a list — must be
    # rejected before the membership test, not raise.
    matrix = np.array([[1.0, 2.0]], dtype=np.float32)
    entries = {"k1": {"content_hash": "h1", "row": 0}}
    out_dir = tmp_path
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "repo_id": "repoA",
        "shard_format": [],  # unhashable
        "dim": 2,
        "entries": entries,
    }
    (out_dir / "repoA.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    np.save(out_dir / "repoA.npy", matrix)

    shard = embedding.read_previous_shard(out_dir, "repoA")
    assert shard.entries == {}
    assert len(shard.vectors) == 0


def test_read_previous_shard_v2_dim_mismatch_is_empty(tmp_path):
    matrix = np.array([[1.0, 2.0]], dtype=np.float32)
    entries = {"k1": {"content_hash": "h1", "row": 0}}
    _write_raw_v2_shard(tmp_path, "repoA", entries, matrix, dim=99)

    shard = embedding.read_previous_shard(tmp_path, "repoA")
    assert shard.entries == {}
    assert len(shard.vectors) == 0


def test_read_previous_shard_v2_wrong_dtype_is_empty(tmp_path):
    matrix = np.array([[1.0, 2.0]], dtype=np.float64)  # wrong dtype, not float32
    entries = {"k1": {"content_hash": "h1", "row": 0}}
    _write_raw_v2_shard(tmp_path, "repoA", entries, matrix)

    shard = embedding.read_previous_shard(tmp_path, "repoA")
    assert shard.entries == {}
    assert len(shard.vectors) == 0


def test_read_previous_shard_v2_one_dimensional_matrix_is_empty(tmp_path):
    out_dir = tmp_path
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "repo_id": "repoA",
        "shard_format": embedding.SHARD_FORMAT_VERSION,
        "dim": 2,
        "entries": {"k1": {"content_hash": "h1", "row": 0}},
    }
    (out_dir / "repoA.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    np.save(out_dir / "repoA.npy", np.array([1.0, 2.0], dtype=np.float32))  # 1-D, not 2-D

    shard = embedding.read_previous_shard(out_dir, "repoA")
    assert shard.entries == {}
    assert len(shard.vectors) == 0


def test_read_previous_shard_v2_entries_not_dict_is_empty(tmp_path):
    out_dir = tmp_path
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "repo_id": "repoA",
        "shard_format": embedding.SHARD_FORMAT_VERSION,
        "dim": 2,
        "entries": ["not", "a", "dict"],  # entries must be a dict
    }
    (out_dir / "repoA.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    np.save(out_dir / "repoA.npy", np.zeros((0, 2), dtype=np.float32))

    shard = embedding.read_previous_shard(out_dir, "repoA")
    assert shard.entries == {}
    assert len(shard.vectors) == 0


def test_read_previous_shard_v2_row_out_of_range_is_empty(tmp_path):
    matrix = np.array([[1.0, 2.0]], dtype=np.float32)
    entries = {"k1": {"content_hash": "h1", "row": 5}}  # out of range, n_rows=1
    _write_raw_v2_shard(tmp_path, "repoA", entries, matrix)

    shard = embedding.read_previous_shard(tmp_path, "repoA")
    assert shard.entries == {}
    assert len(shard.vectors) == 0


def test_read_previous_shard_v2_bool_row_is_empty(tmp_path):
    matrix = np.array([[1.0, 2.0]], dtype=np.float32)
    entries = {"k1": {"content_hash": "h1", "row": True}}  # bool is not an int row index
    _write_raw_v2_shard(tmp_path, "repoA", entries, matrix)

    shard = embedding.read_previous_shard(tmp_path, "repoA")
    assert shard.entries == {}
    assert len(shard.vectors) == 0


def test_read_previous_shard_v2_permuted_rows_reordered_to_sorted_keys(tmp_path):
    """`RepoVectors` requires row i <-> keys[i] in SORTED order. A shard
    whose meta rows are NOT in sorted-key order (here: row 0 -> "z", row 1
    -> "a" — the reverse of sorted order) must still load with `keys`
    sorted and each key's vector matching its OWN row — exercises the sync
    side's `_read_v2_shard` -> `RepoVectors.from_rows` path, mirroring the
    server-side parity test in tests/server/test_store.py."""
    entries = {
        "z": {"content_hash": "hz", "row": 0},
        "a": {"content_hash": "ha", "row": 1},
    }
    matrix = np.array([[9.0, 9.0], [1.0, 2.0]], dtype=np.float32)  # row0="z", row1="a"
    _write_raw_v2_shard(tmp_path, "repoA", entries, matrix)

    shard = embedding.read_previous_shard(tmp_path, "repoA")
    assert shard.vectors.keys == ["a", "z"]  # sorted
    assert list(shard.vectors.get("a")) == pytest.approx([1.0, 2.0])
    assert list(shard.vectors.get("z")) == pytest.approx([9.0, 9.0])


def test_read_previous_shard_v1_invalid_utf8_is_empty_no_exception(tmp_path):
    (tmp_path / "repoA.json").write_bytes(b"\xff\xfe{not valid utf-8")

    shard = embedding.read_previous_shard(tmp_path, "repoA")
    assert shard.entries == {}
    assert len(shard.vectors) == 0


def test_read_previous_shard_v2_meta_invalid_utf8_is_empty_no_exception(tmp_path):
    out_dir = tmp_path
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "repoA.meta.json").write_bytes(b"\xff\xfe{not valid utf-8")
    np.save(out_dir / "repoA.npy", np.zeros((0, 2), dtype=np.float32))

    shard = embedding.read_previous_shard(out_dir, "repoA")
    assert shard.entries == {}
    assert len(shard.vectors) == 0


def test_read_previous_shard_v2_takes_priority_over_stale_v1(tmp_path):
    # If both a v1 .json and v2 .meta.json/.npy exist for the same repo_id
    # (e.g. leftover from an older generation dir), v2 must win.
    vectors = RepoVectors.from_mapping({"k1": [1.0]})
    shard = embedding.RepoShard(entries={"k1": {"content_hash": "h1", "row": 0}}, vectors=vectors)
    out_dir = embedding.stage_embeddings(tmp_path, {"repoA": shard}, id_map={})
    (out_dir / "repoA.json").write_text(
        json.dumps(
            {"repo_id": "repoA", "entries": {"stale": {"content_hash": "x", "embedding": [9.0]}}}
        ),
        encoding="utf-8",
    )

    read_back = embedding.read_previous_shard(out_dir, "repoA")
    assert "stale" not in read_back.entries
    assert "k1" in read_back.entries


# ---------------------------------------------------------------------------
# persist_generation: staging -> permanent storage -> GC, only on publish
# ---------------------------------------------------------------------------


def test_persist_generation_writes_current_and_gcs_old(tmp_path):
    embeddings_dir = tmp_path / "embeddings"
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    (staged_dir / "repo.a.json").write_text(
        json.dumps({"repo_id": "repo.a", "entries": {}}), encoding="utf-8"
    )
    (staged_dir / "id-map.json").write_text("{}", encoding="utf-8")

    embedding.persist_generation(embeddings_dir, "20260101T000000Z-aaa", staged_dir, keep=2)
    embedding.persist_generation(embeddings_dir, "20260102T000000Z-bbb", staged_dir, keep=2)
    embedding.persist_generation(embeddings_dir, "20260103T000000Z-ccc", staged_dir, keep=2)

    generations = sorted(p.name for p in (embeddings_dir / "generations").iterdir())
    assert generations == ["20260102T000000Z-bbb", "20260103T000000Z-ccc"]

    current = embeddings_dir / "current"
    assert current.is_symlink()
    assert current.resolve().name == "20260103T000000Z-ccc"
    assert (current / "repo.a.json").is_file()


# ---------------------------------------------------------------------------
# hostile-input guards: shard filenames and snippet paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_id", ["../evil", "a/b", "a\\b", ".hidden", "..", ".", ""])
def test_shard_filename_rejects_unsafe_repo_id(bad_id):
    with pytest.raises(ValueError, match="unsafe repo_id"):
        embedding._shard_filename(bad_id)


def test_read_previous_shard_refuses_traversal_repo_id(tmp_path):
    current_dir = tmp_path / "current"
    current_dir.mkdir()
    (tmp_path / "outside.json").write_text(
        json.dumps({"entries": {"leaked": {}}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unsafe repo_id"):
        embedding.read_previous_shard(current_dir, "../outside")


def test_stage_embeddings_refuses_traversal_repo_id(tmp_path):
    with pytest.raises(ValueError, match="unsafe repo_id"):
        embedding.stage_embeddings(tmp_path, {"../escape": {}}, id_map={})


def test_build_snippet_rejects_absolute_source_file(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("token=very-secret", encoding="utf-8")
    root = tmp_path / "repo"
    root.mkdir()
    assert embedding.build_snippet(root, str(secret), line=1) == ""


def test_build_snippet_rejects_dotdot_traversal(tmp_path):
    (tmp_path / "secret.txt").write_text("token=very-secret", encoding="utf-8")
    root = tmp_path / "repo"
    root.mkdir()
    assert embedding.build_snippet(root, "../secret.txt", line=1) == ""
