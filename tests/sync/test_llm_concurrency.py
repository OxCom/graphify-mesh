"""GRAPHIFY_MESH_LABEL_CONCURRENCY / GRAPHIFY_MESH_EMBED_CONCURRENCY.

Both default to 1 (the pre-existing sequential behavior) and degrade bad input
to a floor or ceiling instead of raising, same policy as
GRAPHIFY_MESH_SCAN_DEPTH.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import threading
import types
from pathlib import Path

import pytest

from graphify_mesh.sync import clustering, embedding, naming
from graphify_mesh.sync.config import (
    EMBED_MAX_CONCURRENCY,
    LABEL_MAX_CONCURRENCY,
    LLM_MIN_CONCURRENCY,
    Settings,
    _llm_concurrency_from_env,
)

# Captured at import time, before conftest's autouse `_no_live_labeling` fixture
# replaces the module attribute with a fake that never reaches graphify.
_REAL_LLM_NAMES = clustering.llm_names

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
FAKE_GRAPHIFY = FIXTURES_DIR / "fake_graphify" / "graphify"
LABEL_ENV = "GRAPHIFY_MESH_LABEL_CONCURRENCY"
EMBED_ENV = "GRAPHIFY_MESH_EMBED_CONCURRENCY"


def _settings(tmp_path: Path, **overrides) -> Settings:
    mesh_root = tmp_path / "mesh"
    return Settings.from_env(
        mesh_root=mesh_root,
        scan_roots=[tmp_path / "www"],
        registry_path=mesh_root / "bin" / "registry.json",
        graphify_bin=str(FAKE_GRAPHIFY),
        **overrides,
    )


def test_ceilings():
    assert LLM_MIN_CONCURRENCY == 1
    # Upstream fans out over the batches of one call, and a chunk holds 4.
    assert LABEL_MAX_CONCURRENCY == 4
    assert EMBED_MAX_CONCURRENCY == 8


@pytest.mark.parametrize(
    ("env_name", "ceiling"),
    [(LABEL_ENV, LABEL_MAX_CONCURRENCY), (EMBED_ENV, EMBED_MAX_CONCURRENCY)],
)
@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 1), ("3", 3), ("1", 1), ("0", 1), ("-2", 1), ("abc", 1), ("", 1), ("99", None)],
)
def test_env_parsing(monkeypatch, env_name, ceiling, raw, expected):
    if raw is None:
        monkeypatch.delenv(env_name, raising=False)
    else:
        monkeypatch.setenv(env_name, raw)
    want = ceiling if expected is None else expected
    assert _llm_concurrency_from_env(env_name, 1, ceiling) == want


def test_settings_defaults_are_sequential(tmp_path, monkeypatch):
    monkeypatch.delenv(LABEL_ENV, raising=False)
    monkeypatch.delenv(EMBED_ENV, raising=False)
    s = _settings(tmp_path)
    assert s.label_concurrency == 1
    assert s.embed_concurrency == 1


def test_settings_read_env(tmp_path, monkeypatch):
    monkeypatch.setenv(LABEL_ENV, "3")
    monkeypatch.setenv(EMBED_ENV, "6")
    s = _settings(tmp_path)
    assert s.label_concurrency == 3
    assert s.embed_concurrency == 6


def test_settings_above_ceiling_clamped(tmp_path, monkeypatch):
    monkeypatch.setenv(LABEL_ENV, "8")
    monkeypatch.setenv(EMBED_ENV, "100")
    s = _settings(tmp_path)
    assert s.label_concurrency == LABEL_MAX_CONCURRENCY
    assert s.embed_concurrency == EMBED_MAX_CONCURRENCY


def test_direct_construction_is_clamped(tmp_path):
    s = _settings(tmp_path, label_concurrency=0, embed_concurrency=50)
    assert s.label_concurrency == LLM_MIN_CONCURRENCY
    assert s.embed_concurrency == EMBED_MAX_CONCURRENCY


def _stub_upstream(monkeypatch, captured: list[dict]) -> None:
    def fake_generate_community_labels(G, communities, **kwargs):
        captured.append(dict(kwargs, ollama_parallel=os.environ.get("GRAPHIFY_OLLAMA_PARALLEL")))
        return ({cid: f"Name {cid}" for cid in communities}, "llm")

    stub = types.ModuleType("graphify.llm")
    stub.generate_community_labels = fake_generate_community_labels
    monkeypatch.setitem(sys.modules, "graphify.llm", stub)


def test_label_concurrency_reaches_upstream_and_unlocks_ollama_gate(monkeypatch):
    """Upstream forces ollama to one worker unless GRAPHIFY_OLLAMA_PARALLEL=1,
    so the mesh sets it for the labeling call only and restores it after:
    GRAPHIFY_* is forwarded to extract children, which read it too."""
    monkeypatch.delenv("GRAPHIFY_OLLAMA_PARALLEL", raising=False)
    captured: list[dict] = []
    _stub_upstream(monkeypatch, captured)

    _REAL_LLM_NAMES(
        None, {cid: [f"n{cid}"] for cid in range(3)}, backend="ollama", model="m", max_concurrency=3
    )

    assert captured[0]["max_concurrency"] == 3
    assert captured[0]["ollama_parallel"] == "1"
    assert "GRAPHIFY_OLLAMA_PARALLEL" not in os.environ


def test_label_concurrency_one_leaves_ollama_gate_alone(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_OLLAMA_PARALLEL", "0")
    captured: list[dict] = []
    _stub_upstream(monkeypatch, captured)

    _REAL_LLM_NAMES(None, {0: ["n0"]}, backend="ollama", model="m")

    assert captured[0]["max_concurrency"] == 1
    assert captured[0]["ollama_parallel"] == "0"
    assert os.environ["GRAPHIFY_OLLAMA_PARALLEL"] == "0"


def test_ollama_gate_restored_when_upstream_raises(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_OLLAMA_PARALLEL", "0")

    def boom(G, communities, **kwargs):
        raise RuntimeError("upstream broke")

    stub = types.ModuleType("graphify.llm")
    stub.generate_community_labels = boom
    monkeypatch.setitem(sys.modules, "graphify.llm", stub)

    with pytest.raises(RuntimeError):
        _REAL_LLM_NAMES(None, {0: ["n0"]}, backend="ollama", model="m", max_concurrency=2)
    assert os.environ["GRAPHIFY_OLLAMA_PARALLEL"] == "0"


def test_run_naming_passes_label_concurrency(tmp_path, monkeypatch):
    seen: list[int] = []

    def fake_llm(G, communities, *, backend, model, max_concurrency=1):
        seen.append(max_concurrency)
        return {cid: f"Community Name {cid}" for cid in communities}, "llm"

    monkeypatch.setattr(clustering, "llm_names", fake_llm)
    graph = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": [
            {"id": "a.one::p", "label": "Alpha", "source_file": "a.py", "repo": "a.one"},
            {"id": "a.one::q", "label": "Alpha helper", "source_file": "a.py", "repo": "a.one"},
        ],
        "links": [{"source": "a.one::p", "target": "a.one::q", "relation": "calls"}],
    }
    naming.run_naming(
        "graphify",
        tmp_path / "naming",
        tmp_path / "home",
        graph,
        _settings(tmp_path, ollama_model="test-model", label_concurrency=3),
        health_check=lambda *a, **k: True,
    )
    assert seen and set(seen) == {3}


def _many_nodes(n: int) -> dict:
    return {
        "nodes": [
            {"id": f"n{i}", "label": f"Widget{i:03d}", "source_file": f"src/w{i}.py"}
            for i in range(n)
        ]
    }


def _vector_for(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in digest[:4]]


def _shard(graph: dict, tmp_path: Path, concurrency: int, stats=None):
    return embedding.compute_repo_shard(
        "repo.a",
        graph,
        tmp_path,
        embedding.RepoShard(),
        "https://host",
        "m",
        stats or embedding.EmbeddingStats(),
        embedding.recipe_stamp_for_model("m"),
        concurrency=concurrency,
    )


def test_parallel_embedding_gives_identical_shard(tmp_path, monkeypatch, caplog):
    def fake_embed_batch(base_url, model, inputs, timeout=30.0):
        return [_vector_for(text) for text in inputs]

    monkeypatch.setattr(embedding, "embed_batch", fake_embed_batch)
    graph = _many_nodes(5 * embedding.EMBED_BATCH_SIZE + 3)  # 6 batches

    seq_stats = embedding.EmbeddingStats()
    sequential = _shard(graph, tmp_path, 1, seq_stats)
    par_stats = embedding.EmbeddingStats()
    with caplog.at_level(logging.INFO, logger="graphify_mesh.sync.embedding"):
        parallel = _shard(graph, tmp_path, 4, par_stats)

    assert parallel.entries == sequential.entries
    assert parallel.vectors.keys == sequential.vectors.keys
    assert (parallel.vectors.matrix == sequential.vectors.matrix).all()
    assert par_stats.to_dict() == seq_stats.to_dict()
    assert par_stats.embedded == len(graph["nodes"])

    done = [r.getMessage() for r in caplog.records if " done (" in r.getMessage()]
    assert [m.split("batch ")[1].split(" done")[0] for m in done] == [f"{i}/6" for i in range(1, 7)]
    counts = [int(m.split("(")[1].split(" ")[0]) for m in done]
    assert counts == sorted(counts)
    assert counts[-1] == len(graph["nodes"])


@pytest.mark.parametrize("concurrency", [1, 3])
def test_failing_batch_raises_runtime_error(tmp_path, monkeypatch, concurrency):
    monkeypatch.setattr(embedding, "EMBED_RETRY_BACKOFF_SECONDS", 0.0)

    def fake_embed_batch(base_url, model, inputs, timeout=30.0):
        if any("Widget020" in text for text in inputs):
            raise RuntimeError("backend down")
        return [_vector_for(text) for text in inputs]

    monkeypatch.setattr(embedding, "embed_batch", fake_embed_batch)
    with pytest.raises(RuntimeError, match="backend down"):
        _shard(_many_nodes(4 * embedding.EMBED_BATCH_SIZE), tmp_path, concurrency)


def test_parallel_failure_cancels_pending_batches(tmp_path, monkeypatch):
    """A failure must stop new batches from starting: the caller falls back to
    the previous shard, so further embed calls would be wasted backend time."""
    monkeypatch.setattr(embedding, "EMBED_RETRY_BACKOFF_SECONDS", 0.0)
    release = threading.Event()
    calls: list[int] = []
    lock = threading.Lock()

    def fake_embed_batch(base_url, model, inputs, timeout=30.0):
        with lock:
            calls.append(1)
        if any("Widget000" in text for text in inputs):
            raise RuntimeError("first batch failed")
        release.wait(0.3)
        return [_vector_for(text) for text in inputs]

    monkeypatch.setattr(embedding, "embed_batch", fake_embed_batch)
    total_batches = 10
    with pytest.raises(RuntimeError, match="first batch failed"):
        _shard(_many_nodes(total_batches * embedding.EMBED_BATCH_SIZE), tmp_path, 2)
    # 2 attempts of the failing batch plus whatever the other worker had started.
    assert len(calls) < total_batches


def test_stage_passes_embed_concurrency(tmp_path, monkeypatch):
    seen: list[int] = []
    real = embedding.compute_repo_shard

    def recording(*args, **kwargs):
        seen.append(kwargs.get("concurrency", 1))
        return real(*args, **kwargs)

    monkeypatch.setattr(embedding, "compute_repo_shard", recording)
    monkeypatch.setattr(
        embedding,
        "embed_batch",
        lambda base_url, model, inputs, timeout=30.0: [[1.0] for _ in inputs],
    )
    embedding.run_embedding_stage(
        graph_paths_by_repo={},
        repo_roots_by_id={"repo.a": tmp_path},
        graphs_by_repo={"repo.a": _many_nodes(1)},
        unchanged_repo_ids=set(),
        settings=_settings(tmp_path, embed_concurrency=5),
        provisional_generation_id="gen-1",
        health_check=lambda base_url, timeout: True,
    )
    assert seen == [5]
