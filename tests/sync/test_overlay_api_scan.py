"""API fact scan: graphify's ignore rules, one walk per repo, and the stage cache."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
from pathlib import Path

import pytest

from graphify_mesh.sync import overlay, overlay_api
from graphify_mesh.sync import state as state_mod
from graphify_mesh.sync.config import CURRENT_CLASSIFICATION
from graphify_mesh.sync.overlay_api_cache import ApiFactsCache, ApiSourceId, facts_key
from graphify_mesh.sync.stage_cache import PendingCacheWrites
from graphify_mesh.sync.state import (
    IgnoreRules,
    compute_source_manifests,
    read_build_ignore_rules,
)

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "examples" / "manual-relations.schema.json"


def _route(path: Path, route: str, cls: str = "WidgetController") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"<?php\nclass {cls}\n{{\n    #[Route('{route}', methods: ['GET'])]\n"
        "    public function show() {}\n}\n",
        encoding="utf-8",
    )


def _provider_repo(root: Path) -> Path:
    _route(root / "src" / "Controller" / "WidgetController.php", "/api/v1/widgets")
    _route(root / ".superpowers" / "plans" / "Draft.php", "/api/v1/drafts", "DraftController")
    (root / ".graphifyignore").write_text(".superpowers/\n", encoding="utf-8")
    return root


def _consumer_repo(root: Path) -> Path:
    (root / "src").mkdir(parents=True)
    (root / "src" / "client.ts").write_text("fetch('/api/v1/widgets');\n", encoding="utf-8")
    return root


def _count_walks(monkeypatch) -> list[Path]:
    calls: list[Path] = []
    real = overlay_api._walk_source_files

    def counting(root, *args, **kwargs):
        calls.append(root)
        return real(root, *args, **kwargs)

    monkeypatch.setattr(overlay_api, "_walk_source_files", counting)
    return calls


def _source_ids(root: Path) -> ApiSourceId:
    return ApiSourceId.from_digest(_digest(root))


def _digest(root: Path) -> state_mod.SourceDigest:
    rules = read_build_ignore_rules(root)
    return compute_source_manifests(root, (CURRENT_CLASSIFICATION,), rules)[0]


def _key(repo_id: str, root: Path) -> str:
    ids = _source_ids(root)
    return facts_key(repo_id, root, ids.attempt_id, ids.ignore_hash)


def _cache(cache_dir: Path, ids: dict) -> ApiFactsCache:
    return ApiFactsCache(cache_dir, PendingCacheWrites(cache_dir), ids)


def test_ignored_dir_is_excluded_from_api_facts(tmp_path):
    root = _provider_repo(tmp_path / "provider")
    providers, _ = overlay_api.extract_repo_api_facts("acme.provider", root)
    assert [p.path_template for p in providers] == ["/api/v1/widgets"]
    assert providers[0].source_file == "src/Controller/WidgetController.php"


def test_build_excludes_apply_to_api_facts(tmp_path):
    root = _provider_repo(tmp_path / "provider")
    (root / ".graphifyignore").unlink()
    rules = read_build_ignore_rules(tmp_path / "provider")
    providers, _ = overlay_api.extract_repo_api_facts("acme.provider", root, rules)
    assert sorted(p.path_template for p in providers) == ["/api/v1/drafts", "/api/v1/widgets"]
    excluded = IgnoreRules(extra_excludes=(".superpowers/",))
    providers, _ = overlay_api.extract_repo_api_facts("acme.provider", root, excluded)
    assert [p.path_template for p in providers] == ["/api/v1/widgets"]


def test_build_overlay_walks_each_repo_once(tmp_path, monkeypatch):
    roots = {
        "acme.consumer": _consumer_repo(tmp_path / "consumer"),
        "acme.provider": _provider_repo(tmp_path / "provider"),
    }
    calls = _count_walks(monkeypatch)
    result = overlay.build_overlay(
        {}, roots, tmp_path / "absent.json", SCHEMA_PATH, graphs_by_repo={}
    )
    assert sorted(calls) == sorted(roots.values())
    assert result.edge_counts_by_type == {"provides_api": 1, "consumes_api": 1}


def test_cache_hit_skips_walk_on_unchanged_repo(tmp_path, monkeypatch):
    root = _provider_repo(tmp_path / "provider")
    roots = {"acme.provider": root}
    ids = {"acme.provider": _source_ids(root)}
    cache_dir = tmp_path / "cache"

    cold = _cache(cache_dir, ids)
    expected = overlay_api.collect_api_facts(roots, api_cache=cold)
    assert (cold.hits, cold.misses) == (0, 1)
    cold.pending.flush()

    def no_walk(*_args, **_kwargs):
        raise AssertionError("cache hit must not walk the repo")

    monkeypatch.setattr(overlay_api, "_walk_source_files", no_walk)
    warm = _cache(cache_dir, ids)
    assert overlay_api.collect_api_facts(roots, api_cache=warm) == expected
    assert (warm.hits, warm.misses) == (1, 0)
    assert warm.collect_garbage() == 0


def test_repo_without_source_ids_is_not_cached(tmp_path, monkeypatch):
    root = _provider_repo(tmp_path / "provider")
    cache = _cache(tmp_path / "cache", {})
    calls = _count_walks(monkeypatch)
    overlay_api.collect_api_facts({"acme.provider": root}, api_cache=cache)
    assert calls == [root]
    assert len(cache.pending) == 0
    assert (cache.hits, cache.misses) == (0, 0)


def test_key_changes_on_file_edit_and_on_graphifyignore_edit(tmp_path):
    root = _provider_repo(tmp_path / "provider")
    first = _key("acme.provider", root)

    controller = root / "src" / "Controller" / "WidgetController.php"
    controller.write_text(controller.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    edited = _key("acme.provider", root)
    assert edited != first

    (root / ".graphifyignore").write_text(".superpowers/\nsrc/Legacy/\n", encoding="utf-8")
    reignored = _key("acme.provider", root)
    assert reignored not in (first, edited)

    assert _key("acme.other", root) != reignored


def test_corrupt_or_foreign_entry_reads_as_miss(tmp_path):
    root = _provider_repo(tmp_path / "provider")
    ids = {"acme.provider": _source_ids(root), "acme.twin": _source_ids(root)}
    cache_dir = tmp_path / "cache"
    cache = _cache(cache_dir, ids)
    overlay_api.collect_api_facts({"acme.provider": root}, api_cache=cache)
    cache.pending.flush()
    key = _key("acme.provider", root)
    entry = cache.dir / f"{key}.bin"
    assert entry.is_file()

    entry.write_bytes(b"garbage")
    again = _cache(cache_dir, ids)
    assert again.lookup("acme.provider", root) is None
    assert again.misses == 1


@pytest.mark.parametrize("cache_on", [False, True])
def test_collect_api_facts_matches_public_extractors(tmp_path, cache_on):
    root = _provider_repo(tmp_path / "provider")
    _consumer_repo(tmp_path / "consumer")
    roots = {"acme.provider": root, "acme.consumer": tmp_path / "consumer"}
    cache = None
    if cache_on:
        ids = {repo_id: _source_ids(path) for repo_id, path in roots.items()}
        cache = _cache(tmp_path / "cache", ids)
    providers, candidates = overlay_api.collect_api_facts(roots, api_cache=cache)
    for repo_id, path in roots.items():
        assert providers[repo_id] == overlay_api.extract_symfony_route_providers(repo_id, path)
        assert candidates[repo_id] == overlay_api.extract_consumer_literal_paths(repo_id, path)


def _staged_after_scan(tmp_path: Path, root: Path, ids=None) -> ApiFactsCache:
    ids = ids if ids is not None else {"acme.provider": _source_ids(root)}
    cache = _cache(tmp_path / "cache", ids)
    overlay_api.collect_api_facts({"acme.provider": root}, api_cache=cache)
    return cache


def test_complete_scan_is_staged(tmp_path):
    root = _provider_repo(tmp_path / "provider")
    cache = _staged_after_scan(tmp_path, root)
    assert (len(cache.pending), cache.skipped) == (1, 0)


def test_read_failure_skips_store(tmp_path, monkeypatch, caplog):
    root = _provider_repo(tmp_path / "provider")
    ids = {"acme.provider": _source_ids(root)}
    real = Path.read_bytes

    def failing(self):
        if self.name == "WidgetController.php":
            raise PermissionError("denied")
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", failing)
    with caplog.at_level(logging.INFO, logger="graphify_mesh.sync.overlay_api_cache"):
        cache = _staged_after_scan(tmp_path, root, ids=ids)
    assert (len(cache.pending), cache.skipped) == (0, 1)
    assert any(
        r.levelno == logging.INFO
        and "acme.provider" in r.getMessage()
        and "failed" in r.getMessage()
        for r in caplog.records
    )


def test_walk_failure_skips_store(tmp_path, monkeypatch):
    root = _provider_repo(tmp_path / "provider")
    ids = {"acme.provider": _source_ids(root)}
    real = os.walk

    def failing_walk(top, *args, onerror=None, **kwargs):
        if onerror is not None:
            onerror(PermissionError("denied"))
        return real(top, *args, onerror=onerror, **kwargs)

    monkeypatch.setattr(overlay_api.os, "walk", failing_walk)
    cache = _staged_after_scan(tmp_path, root, ids=ids)
    assert (len(cache.pending), cache.skipped) == (0, 1)


def test_digest_with_mtime_markers_skips_store(tmp_path):
    root = _provider_repo(tmp_path / "provider")
    ids = dataclasses.replace(_source_ids(root), mtime_markers=True)
    cache = _staged_after_scan(tmp_path, root, ids={"acme.provider": ids})
    assert (len(cache.pending), cache.skipped) == (0, 1)


def test_file_changed_between_digest_and_scan_skips_store(tmp_path):
    root = _provider_repo(tmp_path / "provider")
    ids = {"acme.provider": _source_ids(root)}
    controller = root / "src" / "Controller" / "WidgetController.php"
    # Same size, other bytes: only the content marker can tell.
    text = controller.read_text(encoding="utf-8")
    controller.write_text(text.replace("widgets", "gadgets"), encoding="utf-8")
    cache = _staged_after_scan(tmp_path, root, ids=ids)
    assert (len(cache.pending), cache.skipped) == (0, 1)


def test_file_added_between_digest_and_scan_skips_store(tmp_path):
    root = _provider_repo(tmp_path / "provider")
    ids = {"acme.provider": _source_ids(root)}
    (root / "src" / "late.ts").write_text("fetch('/api/v1/late');\n", encoding="utf-8")
    cache = _staged_after_scan(tmp_path, root, ids=ids)
    assert (len(cache.pending), cache.skipped) == (0, 1)


def test_file_the_digest_skipped_but_the_scan_read_skips_store(tmp_path):
    # The ignore state diverged (e.g. git failed in one walk): the file sets differ.
    root = _provider_repo(tmp_path / "provider")
    (root / ".graphifyignore").write_text(".superpowers/\nsrc/Controller/\n", encoding="utf-8")
    ids = {"acme.provider": _source_ids(root)}
    (root / ".graphifyignore").write_text(".superpowers/\n", encoding="utf-8")
    cache = _staged_after_scan(tmp_path, root, ids=ids)
    assert (len(cache.pending), cache.skipped) == (0, 1)


def test_digest_exposes_code_markers(tmp_path):
    root = _provider_repo(tmp_path / "provider")
    (root / "README.md").write_text("# readme\n", encoding="utf-8")
    digest = _digest(root)
    controller = root / "src" / "Controller" / "WidgetController.php"
    expected = hashlib.sha256(controller.read_bytes()).hexdigest()[:32]
    assert digest.code_markers == {"src/Controller/WidgetController.php": expected}
    assert state_mod.content_marker(controller.read_bytes()) == expected
    assert digest.mtime_markers is False
    # Not part of equality, attempt_id or the serialized form.
    bare = dataclasses.replace(digest, code_markers={}, mtime_markers=True)
    assert bare == digest
    assert bare.attempt_id == digest.attempt_id
    assert bare.to_dict() == digest.to_dict()
    assert set(digest.to_dict()) == {"code_hash", "semantic_hash", "file_count"}


def test_digest_flags_mtime_markers(tmp_path, monkeypatch):
    root = _provider_repo(tmp_path / "provider")
    monkeypatch.setenv("GRAPHIFY_MESH_CONTENT_DIGEST", "0")
    assert _digest(root).mtime_markers is True
    monkeypatch.delenv("GRAPHIFY_MESH_CONTENT_DIGEST")

    def unreadable(_path):
        raise PermissionError("denied")

    monkeypatch.setattr(state_mod, "file_content_hash", unreadable)
    assert _digest(root).mtime_markers is True


def test_decoding_matches_read_text(tmp_path):
    path = tmp_path / "mixed.php"
    path.write_bytes(b"\xef\xbb\xbf<?php\r\n$a = '/x';\r$b = '\xff/y';\n\r\n")
    assert overlay_api._decode_text(path.read_bytes()) == path.read_text(
        encoding="utf-8", errors="ignore"
    )


def test_dotfile_named_like_a_suffix_is_not_read(tmp_path):
    root = _consumer_repo(tmp_path / "consumer")
    (root / "src" / ".js").write_text("fetch('/api/v1/hidden');\n", encoding="utf-8")
    (root / "src" / ".php").write_text("<?php $c->get('/api/v1/hidden');\n", encoding="utf-8")
    _, candidates = overlay_api.extract_repo_api_facts("acme.consumer", root)
    assert [c[2] for c in candidates] == ["/api/v1/widgets"]
