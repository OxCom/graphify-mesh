from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from graphify_mesh.server.config import ServerConfig
from graphify_mesh.server.store import (
    GenerationStore,
    GenerationUnavailableError,
    validate_manifest_consistency,
)
from graphify_mesh.sync.embedding import RepoShard, stage_embeddings
from graphify_mesh.sync.vectors import RepoVectors


def _write_generation(
    global_dir: Path, generation_id: str, nodes: list[dict], links: list[dict] | None = None
) -> None:
    """Writes a minimal, hash-consistent generation under
    `global_dir/generations/<id>/` and flips `current` to point at it —
    mirrors `graphify_mesh.sync.publish` closely enough for store.py's
    consistency gate without depending on the sync package's I/O helpers."""
    from graphify_mesh.sync.lexical_index import (
        LEXICAL_SCHEMA_VERSION,
        TOKENIZER_VERSION,
        build_lexical_index,
    )
    from graphify_mesh.sync.publish import output_hash

    graph = {"nodes": nodes, "links": links or []}
    graphs_by_repo: dict[str, dict] = {}
    for node in nodes:
        graphs_by_repo.setdefault(node["repo"], {"nodes": []})["nodes"].append(node)
    lexical = build_lexical_index(graphs_by_repo, {}).data

    manifest = {
        "generation_id": generation_id,
        "created_at": "2026-07-20T00:00:00Z",
        "repo_input_hashes": {},
        "registry_hash": "test",
        "config_hash": "test",
        "output_node_count": len(nodes),
        "output_edge_count": len(links or []),
        "output_hash": output_hash(graph),
        "clustering_backend": "louvain",
        "embedding_model": "test-model",
        "labeling": "skipped",
        "stale_repos": [],
        "lexical_index_tokenizer_version": TOKENIZER_VERSION,
        "lexical_index_schema_version": LEXICAL_SCHEMA_VERSION,
    }

    gen_dir = global_dir / "generations" / generation_id
    gen_dir.mkdir(parents=True, exist_ok=True)
    (gen_dir / "global-graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (gen_dir / "generation-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (gen_dir / "cross-project-overlay.json").write_text(json.dumps({"edges": []}), encoding="utf-8")
    (gen_dir / "lexical-index.json").write_text(json.dumps(lexical), encoding="utf-8")

    current = global_dir / "current"
    if current.exists() or current.is_symlink():
        current.unlink()
    current.symlink_to(gen_dir, target_is_directory=True)


def _config(tmp_path: Path) -> ServerConfig:
    return ServerConfig.from_env(
        mesh_root=tmp_path, registry_path=tmp_path / "bin" / "registry.json"
    )


def test_validate_manifest_accepts_schema_2_and_3():
    for version in (2, 3):
        manifest = {"lexical_index_schema_version": version}
        lexical = {"schema_version": version}
        errors = validate_manifest_consistency(manifest, {}, lexical)
        assert not any("schema_version" in e for e in errors)


def test_validate_manifest_rejects_unknown_schema():
    manifest = {"lexical_index_schema_version": 99}
    lexical = {"schema_version": 99}
    errors = validate_manifest_consistency(manifest, {}, lexical)
    assert any("schema_version" in e for e in errors)


def test_no_generation_published_raises_generation_unavailable(tmp_path):
    store = GenerationStore(_config(tmp_path))
    with pytest.raises(GenerationUnavailableError):
        _ = store.generation
    assert "no_generation_published" in store.degraded


def test_loads_valid_generation_and_builds_indexes(tmp_path):
    config = _config(tmp_path)
    _write_generation(
        config.global_dir,
        "gen-1",
        [{"id": "n1", "repo": "repo.a", "label": "Alpha", "source_file": "a.py"}],
    )
    store = GenerationStore(config)
    generation = store.generation
    assert generation.generation_id == "gen-1"
    assert "n1" in generation.node_by_id
    assert store.degraded == [
        "embeddings_unavailable"
    ]  # no embeddings dir published in this fixture


def test_missing_lexical_file_is_degraded_not_an_error(tmp_path):
    """Absence of `lexical-index.json` is the documented degraded state
    (lexical/structural still serve) and must NOT produce a validation
    error — distinct from a present-but-corrupt file (see next test)."""
    config = _config(tmp_path)
    _write_generation(
        config.global_dir,
        "gen-1",
        [{"id": "n1", "repo": "repo.a", "label": "Alpha", "source_file": "a.py"}],
    )
    gen_dir = config.global_dir / "generations" / "gen-1"
    (gen_dir / "lexical-index.json").unlink()

    store = GenerationStore(config)
    generation = store.generation
    assert generation.generation_id == "gen-1"
    assert generation.lexical == {}
    assert not any("lexical-index" in reason for reason in store.degraded)


def test_present_but_corrupt_lexical_file_is_validation_error(tmp_path):
    """A `lexical-index.json` that exists but is not parseable JSON (or
    parses to something other than a JSON object) is a real artifact
    problem, not a degraded-absence state — must be surfaced as a
    validation error and the generation rejected."""
    config = _config(tmp_path)
    _write_generation(
        config.global_dir,
        "gen-1",
        [{"id": "n1", "repo": "repo.a", "label": "Alpha", "source_file": "a.py"}],
    )
    gen_dir = config.global_dir / "generations" / "gen-1"
    (gen_dir / "lexical-index.json").write_text("{not valid json", encoding="utf-8")

    store = GenerationStore(config)
    with pytest.raises(GenerationUnavailableError):
        _ = store.generation
    assert any("lexical-index" in reason for reason in store.degraded)


def test_present_but_invalid_utf8_lexical_file_is_validation_error(tmp_path):
    """A `lexical-index.json` containing invalid UTF-8 bytes must raise
    UnicodeDecodeError on read — surfaced as the same validation error as
    unparseable JSON, not an uncaught crash."""
    config = _config(tmp_path)
    _write_generation(
        config.global_dir,
        "gen-1",
        [{"id": "n1", "repo": "repo.a", "label": "Alpha", "source_file": "a.py"}],
    )
    gen_dir = config.global_dir / "generations" / "gen-1"
    (gen_dir / "lexical-index.json").write_bytes(b"\xff\xfe{")

    store = GenerationStore(config)
    with pytest.raises(GenerationUnavailableError):
        _ = store.generation
    assert any("lexical-index" in reason for reason in store.degraded)


def test_present_but_non_dict_lexical_file_is_validation_error(tmp_path):
    """Parses fine as JSON but is not a JSON object (e.g. a bare list) —
    still a real artifact problem, same as unparseable JSON."""
    config = _config(tmp_path)
    _write_generation(
        config.global_dir,
        "gen-1",
        [{"id": "n1", "repo": "repo.a", "label": "Alpha", "source_file": "a.py"}],
    )
    gen_dir = config.global_dir / "generations" / "gen-1"
    (gen_dir / "lexical-index.json").write_text(
        json.dumps(["not", "an", "object"]), encoding="utf-8"
    )

    store = GenerationStore(config)
    with pytest.raises(GenerationUnavailableError):
        _ = store.generation
    assert any("lexical-index" in reason for reason in store.degraded)


def test_inconsistent_manifest_rejected_all_or_nothing_keeps_previous(tmp_path):
    config = _config(tmp_path)
    _write_generation(
        config.global_dir,
        "gen-1",
        [{"id": "n1", "repo": "repo.a", "label": "Alpha", "source_file": "a.py"}],
    )
    store = GenerationStore(config)
    first = store.generation
    assert first.generation_id == "gen-1"

    # Publish a second generation whose manifest LIES about node count —
    # store.py must reject it and keep serving gen-1, not crash or half-load.
    gen2_dir = config.global_dir / "generations" / "gen-2"
    gen2_dir.mkdir(parents=True)
    graph = {
        "nodes": [{"id": "n1", "repo": "repo.a", "label": "Alpha", "source_file": "a.py"}],
        "links": [],
    }
    (gen2_dir / "global-graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (gen2_dir / "generation-manifest.json").write_text(
        json.dumps(
            {
                "generation_id": "gen-2",
                "created_at": "2026-07-20T00:00:00Z",
                "repo_input_hashes": {},
                "registry_hash": "test",
                "config_hash": "test",
                "output_node_count": 999,  # deliberately wrong — graph.json has 1 node
                "output_edge_count": 0,
                "clustering_backend": "louvain",
                "embedding_model": "test-model",
                "labeling": "skipped",
                "stale_repos": [],
            }
        ),
        encoding="utf-8",
    )
    (gen2_dir / "cross-project-overlay.json").write_text(
        json.dumps({"edges": []}), encoding="utf-8"
    )
    (gen2_dir / "lexical-index.json").write_text(json.dumps({}), encoding="utf-8")
    current = config.global_dir / "current"
    current.unlink()
    current.symlink_to(gen2_dir, target_is_directory=True)

    second = store.generation  # triggers ensure_fresh() -> reload attempt -> rejected
    assert second.generation_id == "gen-1"  # still serving the last-good generation
    assert "reload_rejected_previous_generation_still_serving" in store.degraded
    assert any("output_node_count" in reason for reason in store.degraded)


def test_stale_lexical_schema_version_rejected(tmp_path):
    """A `lexical-index.json` published with an old schema_version (e.g. the
    pre-fix v1 per-entry-dict shape) must be rejected rather than served —
    this server's readers only understand the supported v2/v3 shapes
    (`SUPPORTED_LEXICAL_SCHEMA_VERSIONS`) and would misindex into a v1
    dict's keys otherwise."""
    config = _config(tmp_path)
    _write_generation(
        config.global_dir,
        "gen-1",
        [{"id": "n1", "repo": "repo.a", "label": "Alpha", "source_file": "a.py"}],
    )
    store = GenerationStore(config)
    first = store.generation
    assert first.generation_id == "gen-1"

    gen2_dir = config.global_dir / "generations" / "gen-2"
    gen2_dir.mkdir(parents=True)
    graph = {
        "nodes": [{"id": "n1", "repo": "repo.a", "label": "Alpha", "source_file": "a.py"}],
        "links": [],
    }
    (gen2_dir / "global-graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (gen2_dir / "generation-manifest.json").write_text(
        json.dumps(
            {
                "generation_id": "gen-2",
                "created_at": "2026-07-20T00:00:00Z",
                "repo_input_hashes": {},
                "registry_hash": "test",
                "config_hash": "test",
                "output_node_count": 1,
                "output_edge_count": 0,
                "clustering_backend": "louvain",
                "embedding_model": "test-model",
                "labeling": "skipped",
                "stale_repos": [],
                "lexical_index_schema_version": 1,
            }
        ),
        encoding="utf-8",
    )
    (gen2_dir / "cross-project-overlay.json").write_text(
        json.dumps({"edges": []}), encoding="utf-8"
    )
    (gen2_dir / "lexical-index.json").write_text(
        json.dumps({"schema_version": 1, "postings": {}, "alias_exact": {}}), encoding="utf-8"
    )
    current = config.global_dir / "current"
    current.unlink()
    current.symlink_to(gen2_dir, target_is_directory=True)

    second = store.generation
    assert second.generation_id == "gen-1"
    assert any("schema_version" in reason for reason in store.degraded)


def test_hot_reload_picks_up_new_valid_generation(tmp_path):
    config = _config(tmp_path)
    _write_generation(
        config.global_dir,
        "gen-1",
        [{"id": "n1", "repo": "repo.a", "label": "Alpha", "source_file": "a.py"}],
    )
    store = GenerationStore(config)
    assert store.generation.generation_id == "gen-1"

    _write_generation(
        config.global_dir,
        "gen-2",
        [{"id": "n2", "repo": "repo.a", "label": "Beta", "source_file": "b.py"}],
    )
    assert store.generation.generation_id == "gen-2"
    assert "n2" in store.generation.node_by_id


# --- _load_embeddings: corrupt/oversized shard tolerance ---------------------


def test_load_embeddings_skips_oversized_shard(tmp_path, monkeypatch):
    from graphify_mesh.server import store

    (tmp_path / "big.json").write_text(
        json.dumps({"repo_id": "big.repo", "entries": {"k": {"embedding": [1.0]}}}),
        encoding="utf-8",
    )
    (tmp_path / "ok.json").write_text(
        json.dumps({"repo_id": "ok.repo", "entries": {"k": {"embedding": [2.0]}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(store, "MAX_SHARD_BYTES", (tmp_path / "ok.json").stat().st_size)

    out = store._load_embeddings(tmp_path)
    assert "ok.repo" in out
    assert "big.repo" not in out


def test_load_embeddings_tolerates_malformed_shards(tmp_path):
    from graphify_mesh.server.store import _load_embeddings

    (tmp_path / "not-dict.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    (tmp_path / "entries-not-dict.json").write_text(
        json.dumps({"repo_id": "bad.repo", "entries": "oops"}), encoding="utf-8"
    )
    (tmp_path / "entry-not-dict.json").write_text(
        json.dumps({"repo_id": "half.repo", "entries": {"a": "oops", "b": {"embedding": [1.0]}}}),
        encoding="utf-8",
    )

    out = _load_embeddings(tmp_path)
    assert "bad.repo" not in out
    assert out["half.repo"].to_mapping() == {"b": [1.0]}


# --- _load_embeddings: v1/v2 shard format parity ------------------------


def _repo_shard(raw_vectors: dict[str, list[float]]) -> RepoShard:
    """Builds a RepoShard the way the sync pipeline would: RepoVectors from
    a plain mapping, plus an `entries` dict recording each key's matrix
    row — mirrors `_read_v1_shard`'s post-build bookkeeping in
    `graphify_mesh.sync.embedding`."""
    vectors = RepoVectors.from_mapping(raw_vectors)
    entries = {key: {"content_hash": None, "row": None} for key in vectors.keys}
    for row, key in enumerate(vectors.keys):
        entries[key]["row"] = row
    return RepoShard(entries=entries, vectors=vectors)


def test_load_embeddings_v1_dir_and_v2_dir_produce_equal_mapping(tmp_path):
    from graphify_mesh.server.store import _load_embeddings

    raw = {"k1": [1.0, 0.0], "k2": [0.0, 1.0]}

    v1_dir = tmp_path / "v1"
    v1_dir.mkdir()
    (v1_dir / "repo.a.json").write_text(
        json.dumps(
            {
                "repo_id": "repo.a",
                "entries": {
                    key: {"content_hash": None, "embedding": vec} for key, vec in raw.items()
                },
            }
        ),
        encoding="utf-8",
    )

    v2_dir = stage_embeddings(tmp_path / "v2-staging", {"repo.a": _repo_shard(raw)}, id_map={})

    v1_out = _load_embeddings(v1_dir)
    v2_out = _load_embeddings(v2_dir)

    assert v1_out.keys() == v2_out.keys()
    assert v1_out["repo.a"].to_mapping() == v2_out["repo.a"].to_mapping()


def test_load_embeddings_v2_skips_unsupported_shard_format_others_still_load(tmp_path):
    from graphify_mesh.server.store import _load_embeddings

    (tmp_path / "bad.repo.meta.json").write_text(
        json.dumps({"repo_id": "bad.repo", "shard_format": 99, "dim": 2, "entries": {}}),
        encoding="utf-8",
    )
    np.save(tmp_path / "bad.repo.npy", np.zeros((0, 2), dtype=np.float32))

    (tmp_path / "ok.repo.meta.json").write_text(
        json.dumps(
            {
                "repo_id": "ok.repo",
                "shard_format": 2,
                "dim": 2,
                "entries": {"k": {"content_hash": None, "row": 0}},
            }
        ),
        encoding="utf-8",
    )
    np.save(tmp_path / "ok.repo.npy", np.array([[1.0, 0.0]], dtype=np.float32))

    out = _load_embeddings(tmp_path)
    assert "bad.repo" not in out
    assert out["ok.repo"].to_mapping() == {"k": [1.0, 0.0]}


# --- _load_embeddings: v2 sorted-keys invariant -------------------------


def test_load_embeddings_v2_permuted_rows_reordered_to_sorted_keys(tmp_path):
    """`RepoVectors` requires row i <-> keys[i] in SORTED order. A shard
    whose meta rows are NOT in sorted-key order (here: row 0 -> "z", row 1
    -> "a" — the reverse of sorted order) must still load with `keys`
    sorted and each key's vector matching its OWN row, not silently
    reinterpreted through the wrong permutation."""
    from graphify_mesh.server.store import _load_embeddings

    (tmp_path / "perm.repo.meta.json").write_text(
        json.dumps(
            {
                "repo_id": "perm.repo",
                "shard_format": 2,
                "dim": 2,
                "entries": {
                    "z": {"content_hash": None, "row": 0},
                    "a": {"content_hash": None, "row": 1},
                },
            }
        ),
        encoding="utf-8",
    )
    np.save(
        tmp_path / "perm.repo.npy",
        np.array([[9.0, 9.0], [1.0, 2.0]], dtype=np.float32),  # row0="z", row1="a"
    )

    out = _load_embeddings(tmp_path)
    rv = out["perm.repo"]
    assert rv.keys == ["a", "z"]  # sorted
    assert rv.get("a") is not None
    assert rv.get("z") is not None
    assert list(rv.get("a")) == pytest.approx([1.0, 2.0])
    assert list(rv.get("z")) == pytest.approx([9.0, 9.0])


def test_load_embeddings_v2_canonical_shard_keeps_mmap(tmp_path):
    """Canonical (already-sorted-key-order) shards must keep the mmap
    array as-is — no copy — since that is the whole point of the v2
    read side (server never eagerly loads the full matrix into RAM)."""
    from graphify_mesh.server.store import _load_embeddings

    (tmp_path / "ok.repo.meta.json").write_text(
        json.dumps(
            {
                "repo_id": "ok.repo",
                "shard_format": 2,
                "dim": 2,
                "entries": {
                    "a": {"content_hash": None, "row": 0},
                    "z": {"content_hash": None, "row": 1},
                },
            }
        ),
        encoding="utf-8",
    )
    np.save(
        tmp_path / "ok.repo.npy",
        np.array([[1.0, 2.0], [9.0, 9.0]], dtype=np.float32),
    )

    out = _load_embeddings(tmp_path)
    rv = out["ok.repo"]
    assert rv.keys == ["a", "z"]
    assert isinstance(rv.matrix, np.memmap)


# --- reload robustness: wrong-typed artifacts, incomplete generations --------


def _publish_raw_generation(global_dir: Path, generation_id: str, graph_text: str) -> Path:
    """Publishes a generation whose `global-graph.json` holds arbitrary text —
    the manifest itself is well-formed, so the graph's shape is the only thing
    under test."""
    gen_dir = global_dir / "generations" / generation_id
    gen_dir.mkdir(parents=True, exist_ok=True)
    (gen_dir / "global-graph.json").write_text(graph_text, encoding="utf-8")
    (gen_dir / "generation-manifest.json").write_text(
        json.dumps(
            {
                "generation_id": generation_id,
                "created_at": "2026-07-20T00:00:00Z",
                "repo_input_hashes": {},
                "registry_hash": "test",
                "config_hash": "test",
                "output_node_count": 0,
                "output_edge_count": 0,
                "clustering_backend": "louvain",
                "embedding_model": "test-model",
                "labeling": "skipped",
                "stale_repos": [],
            }
        ),
        encoding="utf-8",
    )
    (gen_dir / "cross-project-overlay.json").write_text(json.dumps({"edges": []}), encoding="utf-8")
    (gen_dir / "lexical-index.json").write_text(json.dumps({}), encoding="utf-8")
    current = global_dir / "current"
    if current.exists() or current.is_symlink():
        current.unlink()
    current.symlink_to(gen_dir, target_is_directory=True)
    return gen_dir


def test_graph_parsing_to_a_list_keeps_previous_generation_serving(tmp_path):
    """A `global-graph.json` holding `[]` parses fine, so the old
    `graph is None` gate let it through and the crash landed inside the
    consistency gate — under the write lock, poisoning every later tool call.
    It must be rejected like any other unreadable artifact."""
    config = _config(tmp_path)
    _write_generation(
        config.global_dir,
        "gen-1",
        [{"id": "n1", "repo": "repo.a", "label": "Alpha", "source_file": "a.py"}],
    )
    store = GenerationStore(config)
    assert store.generation.generation_id == "gen-1"

    _publish_raw_generation(config.global_dir, "gen-2", json.dumps([]))

    assert store.generation.generation_id == "gen-1"  # previous generation still serves
    assert store.degraded == ["reload_failed_unreadable_artifacts"]
    # A second call must still work — the store is not poisoned.
    assert store.generation.generation_id == "gen-1"


def test_manifest_parsing_to_a_string_keeps_previous_generation_serving(tmp_path):
    config = _config(tmp_path)
    _write_generation(
        config.global_dir,
        "gen-1",
        [{"id": "n1", "repo": "repo.a", "label": "Alpha", "source_file": "a.py"}],
    )
    store = GenerationStore(config)
    assert store.generation.generation_id == "gen-1"

    gen2_dir = _publish_raw_generation(
        config.global_dir, "gen-2", json.dumps({"nodes": [], "links": []})
    )
    (gen2_dir / "generation-manifest.json").write_text(json.dumps("not-a-manifest"), "utf-8")

    assert store.generation.generation_id == "gen-1"
    assert store.degraded == ["reload_failed_unreadable_artifacts"]


def test_unexpected_exception_during_reload_keeps_previous_generation(tmp_path, monkeypatch):
    """Backstop for artifact shapes nobody anticipated: anything raising from
    the validation step onward must cost one reload, not the whole store."""
    from graphify_mesh.server import store as store_mod

    config = _config(tmp_path)
    _write_generation(
        config.global_dir,
        "gen-1",
        [{"id": "n1", "repo": "repo.a", "label": "Alpha", "source_file": "a.py"}],
    )
    store = GenerationStore(config)
    assert store.generation.generation_id == "gen-1"

    def boom(*args, **kwargs):
        raise RuntimeError("artifact shape nobody anticipated")

    monkeypatch.setattr(store_mod, "validate_manifest_consistency", boom)
    _write_generation(
        config.global_dir,
        "gen-2",
        [{"id": "n2", "repo": "repo.a", "label": "Beta", "source_file": "b.py"}],
    )

    assert store.generation.generation_id == "gen-1"
    assert store.degraded == ["reload_rejected_previous_generation_still_serving"]


def test_manifest_listed_but_missing_lexical_index_is_rejected(tmp_path):
    """`artifact_sha256` is the manifest's own claim about what it published.
    A listed artifact that is absent on disk means the generation directory is
    incomplete — the same judgement `sync.publish._is_incomplete` makes — so it
    must be rejected, not loaded with an empty lexical index."""
    import hashlib

    config = _config(tmp_path)
    _write_generation(
        config.global_dir,
        "gen-1",
        [{"id": "n1", "repo": "repo.a", "label": "Alpha", "source_file": "a.py"}],
    )
    store = GenerationStore(config)
    assert store.generation.generation_id == "gen-1"

    _write_generation(
        config.global_dir,
        "gen-2",
        [{"id": "n2", "repo": "repo.a", "label": "Beta", "source_file": "b.py"}],
    )
    gen2_dir = config.global_dir / "generations" / "gen-2"
    manifest = json.loads((gen2_dir / "generation-manifest.json").read_text(encoding="utf-8"))
    manifest["artifact_sha256"] = {
        name: hashlib.sha256((gen2_dir / name).read_bytes()).hexdigest()
        for name in ("global-graph.json", "cross-project-overlay.json", "lexical-index.json")
    }
    (gen2_dir / "generation-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (gen2_dir / "lexical-index.json").unlink()

    assert store.generation.generation_id == "gen-1"
    assert "reload_rejected_previous_generation_still_serving" in store.degraded
    assert any("lexical-index.json" in reason for reason in store.degraded)


def test_lexical_index_absent_from_hash_map_and_disk_still_loads_degraded(tmp_path):
    """The never-published case stays as it was: no entry in the map and no
    file on disk is the documented degraded mode, not an error."""
    import hashlib

    config = _config(tmp_path)
    _write_generation(
        config.global_dir,
        "gen-1",
        [{"id": "n1", "repo": "repo.a", "label": "Alpha", "source_file": "a.py"}],
    )
    gen_dir = config.global_dir / "generations" / "gen-1"
    manifest = json.loads((gen_dir / "generation-manifest.json").read_text(encoding="utf-8"))
    manifest["artifact_sha256"] = {
        name: hashlib.sha256((gen_dir / name).read_bytes()).hexdigest()
        for name in ("global-graph.json", "cross-project-overlay.json")
    }
    (gen_dir / "generation-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (gen_dir / "lexical-index.json").unlink()

    store = GenerationStore(config)
    assert store.generation.lexical == {}
    assert not any("lexical-index" in reason for reason in store.degraded)


# --- reload trigger: embeddings symlink flip --------------------------------


def _publish_embeddings_generation(global_dir: Path, generation_id: str) -> Path:
    """Writes one v2 shard under `embeddings/generations/<id>/` and flips
    `embeddings/current` to it — the sibling publish
    `sync.embedding.persist_generation` performs."""
    emb_dir = global_dir / "embeddings" / "generations" / generation_id
    emb_dir.mkdir(parents=True, exist_ok=True)
    (emb_dir / "repo.a.meta.json").write_text(
        json.dumps(
            {
                "repo_id": "repo.a",
                "shard_format": 2,
                "dim": 2,
                "entries": {"k": {"content_hash": None, "row": 0}},
            }
        ),
        encoding="utf-8",
    )
    np.save(emb_dir / "repo.a.npy", np.array([[1.0, 0.0]], dtype=np.float32))

    link = global_dir / "embeddings" / "current"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(emb_dir, target_is_directory=True)
    return emb_dir


def test_embeddings_symlink_flip_alone_triggers_a_reload(tmp_path):
    """Embeddings are published on their own symlink flip. A stamp mismatch
    drops the vector channel, and nothing else about the graph generation
    changes when the matching embeddings land — so the freshness signature has
    to watch `embeddings/current` too, or vector search stays dead until the
    next graph publish."""
    config = _config(tmp_path)
    _write_generation(
        config.global_dir,
        "gen-1",
        [{"id": "n1", "repo": "repo.a", "label": "Alpha", "source_file": "a.py"}],
    )
    _publish_embeddings_generation(config.global_dir, "gen-0")  # stale stamp

    store = GenerationStore(config)
    assert store.generation.embeddings == {}
    assert "embeddings_generation_mismatch" in store.degraded

    # Only the embeddings symlink moves — the graph generation is untouched.
    _publish_embeddings_generation(config.global_dir, "gen-1")

    generation = store.generation
    assert generation.generation_id == "gen-1"
    assert "repo.a" in generation.embeddings
    assert "embeddings_generation_mismatch" not in store.degraded
