"""Stale-semantic flag: LLM-written nodes on a file graphify changed since the LLM pass."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphify_mesh.sync import lexical_index


def _write_manifest(path: Path, rows: dict) -> Path:
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _row(ast: str, semantic: str) -> dict:
    return {"mtime": 1.0, "seen": 1.0, "ast_hash": ast, "semantic_hash": semantic}


def test_stale_files_need_ast_hash_and_empty_semantic_hash(tmp_path):
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        {
            "changed.yaml": _row("a1", ""),
            "unchanged.yaml": _row("a2", "a2"),
            "semantic_only.yaml": _row("", ""),
            "legacy_float.yaml": 1717594375.0,
            "no_semantic_key.yaml": {"ast_hash": "a3"},
        },
    )
    assert lexical_index.stale_semantic_files(manifest) == {"changed.yaml"}


def test_stale_files_missing_or_corrupt_manifest_is_empty(tmp_path):
    assert lexical_index.stale_semantic_files(None) == set()
    assert lexical_index.stale_semantic_files(tmp_path / "absent.json") == set()
    corrupt = tmp_path / "manifest.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert lexical_index.stale_semantic_files(corrupt) == set()
    _write_manifest(corrupt, {})
    corrupt.write_text("[1, 2]", encoding="utf-8")
    assert lexical_index.stale_semantic_files(corrupt) == set()


def test_stale_files_follow_graphify_save_manifest(tmp_path):
    detect = pytest.importorskip("graphify.detect")
    root = tmp_path / "repo"
    root.mkdir()
    changed = root / "values.yaml"
    unchanged = root / "other.yaml"
    changed.write_text("a: 1\n", encoding="utf-8")
    unchanged.write_text("b: 1\n", encoding="utf-8")
    files = {"document": [str(changed), str(unchanged)]}
    manifest = tmp_path / "manifest.json"
    detect.save_manifest(files, str(manifest), kind="both", root=root)
    assert lexical_index.stale_semantic_files(manifest) == set()

    changed.write_text("a: 2\n", encoding="utf-8")
    detect.save_manifest(files, str(manifest), kind="ast", root=root)
    assert lexical_index.stale_semantic_files(manifest) == {"values.yaml"}

    # A second AST update with no further edit keeps the flag and adds none.
    detect.save_manifest(files, str(manifest), kind="ast", root=root)
    assert lexical_index.stale_semantic_files(manifest) == {"values.yaml"}


def _graph() -> dict:
    return {
        "nodes": [
            {
                "id": "s_stale",
                "label": "replicaCount",
                "source_file": "a.yaml",
                "_origin": "semantic",
            },
            {"id": "ast_stale", "label": "image", "source_file": "a.yaml", "_origin": "ast"},
            {"id": "s_fresh", "label": "service", "source_file": "b.yaml", "_origin": "semantic"},
            {"id": "s_norow", "label": "ingress", "source_file": "c.yaml", "_origin": "semantic"},
        ]
    }


def _stale_keys(data: dict) -> set[str]:
    return {data["documents"][doc_id][1] for doc_id in data["stale_semantic"]}


def test_build_flags_only_semantic_nodes_of_stale_files(tmp_path):
    graph_dir = tmp_path / "collection"
    graph_dir.mkdir()
    _write_manifest(
        graph_dir / "manifest.json",
        {"a.yaml": _row("h1", ""), "b.yaml": _row("h2", "h2")},
    )
    result = lexical_index.build_lexical_index(
        {"repo.a": _graph()}, {}, graph_paths_by_repo={"repo.a": graph_dir / "graph.json"}
    )
    keys = _stale_keys(result.data)
    assert len(keys) == 1
    assert next(iter(keys)).endswith("replicaCount")
    assert result.stats.stale_semantic_documents == 1
    assert result.stats.to_dict()["stale_semantic_documents"] == 1
    # The flag is additive: schema and document row shape are unchanged.
    assert result.data["schema_version"] == lexical_index.LEXICAL_SCHEMA_VERSION == 4
    assert all(len(entry) == 2 for entry in result.data["documents"])
    assert result.data["document_count"] == 4


def test_build_without_manifest_paths_flags_nothing():
    result = lexical_index.build_lexical_index({"repo.a": _graph()}, {})
    assert result.data["stale_semantic"] == []
    assert result.stats.stale_semantic_documents == 0
