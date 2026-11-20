"""Per-repo lexical partial cache (stage caches, Task 4).

Three repos sort as repo.a < repo.b < repo.c. repo.b grows between runs, so
every doc id of repo.c shifts while repo.c's cached partial stays valid. Each
repo carries a real snippet, a stale semantic node, aliases, YAML config keys,
a duplicate durable key and a term shared with the other repos, so every
offset the assembly applies is exercised.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from graphify_mesh.sync import lexical_index, publish
from graphify_mesh.sync.lexical_partial_cache import (
    LEXICAL_CACHE_SUBDIR,
    LexicalPartialCache,
    partial_key,
)
from graphify_mesh.sync.source_cache import clear_source_cache
from graphify_mesh.sync.stage_cache import PendingCacheWrites, read_entry, write_entry

REPOS = ("repo.a", "repo.b", "repo.c")
VALUES_YAML = "ingress:\n  enabled: true\n  host: sharedterm.example\nreplicaCount: 2\n"


def _code(name: str, extra: int) -> str:
    lines = [f"class {name}Service:", "    def sharedTerm(self):", "        return 1"]
    lines += [f"def helper_{name.lower()}_{i}():\n    return {i}" for i in range(extra)]
    return "\n".join(lines) + "\n"


def _graph(name: str, root: Path, extra: int) -> dict:
    yaml_config = pytest.importorskip("graphify.extractors.yaml_config")
    yaml_graph = yaml_config.extract_yaml(root / "conf" / "values.yaml")
    # The extractor names the absolute path; graphs carry root-relative ones.
    for item in yaml_graph["nodes"] + yaml_graph["edges"]:
        item["source_file"] = "conf/values.yaml"
    code = "src/service.py"
    nodes = [
        {"id": "f", "label": "service.py", "source_file": code, "source_location": "L1"},
        {"id": "c", "label": f"{name}Service", "source_file": code, "source_location": "L1"},
        {"id": "m", "label": f"{name}Service::sharedTerm", "source_file": code, "line": 2},
        # Same (source_file, label) as "m": one durable key shared by two nodes.
        {"id": "m2", "label": f"{name}Service::sharedTerm", "source_file": code, "line": 3},
        {
            "id": "sem",
            "label": "Ingress settings",
            "source_file": "conf/values.yaml",
            "_origin": "semantic",
        },
    ]
    nodes += [
        {
            "id": f"h{i}",
            "label": f"helper_{name.lower()}_{i}",
            "source_file": code,
            "source_location": f"L{4 + 2 * i}",
        }
        for i in range(extra)
    ]
    return {"nodes": nodes + yaml_graph["nodes"], "links": yaml_graph["edges"]}


class Corpus:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.extra = {repo_id: 1 for repo_id in REPOS}
        self.digests = {repo_id: f"digest-{repo_id}-0" for repo_id in REPOS}
        for repo_id in REPOS:
            self.write(repo_id)

    def root(self, repo_id: str) -> Path:
        return self.tmp / "roots" / repo_id

    def graph_path(self, repo_id: str) -> Path:
        return self.tmp / "collections" / repo_id / "graph.json"

    def write(self, repo_id: str) -> None:
        name = repo_id.split(".")[1].upper()
        root = self.root(repo_id)
        (root / "src").mkdir(parents=True, exist_ok=True)
        (root / "conf").mkdir(parents=True, exist_ok=True)
        (root / "src" / "service.py").write_text(_code(name, self.extra[repo_id]))
        (root / "conf" / "values.yaml").write_text(VALUES_YAML)
        graph_path = self.graph_path(repo_id)
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_text(json.dumps(_graph(name, root, self.extra[repo_id])))
        # values.yaml changed after the LLM pass: its semantic nodes are stale.
        manifest = {"conf/values.yaml": {"ast_hash": "h1", "semantic_hash": ""}}
        (graph_path.parent / "manifest.json").write_text(json.dumps(manifest))

    def grow(self, repo_id: str) -> None:
        self.extra[repo_id] += 3
        self.digests[repo_id] = f"digest-{repo_id}-{self.extra[repo_id]}"
        self.write(repo_id)

    def build(self, cache: LexicalPartialCache | None = None) -> lexical_index.LexicalIndexResult:
        clear_source_cache()
        raw = {r: self.graph_path(r).read_bytes() for r in REPOS}
        return lexical_index.build_lexical_index(
            {r: json.loads(raw[r]) for r in REPOS},
            {r: self.root(r) for r in REPOS},
            graph_paths_by_repo={r: self.graph_path(r) for r in REPOS},
            source_digests_by_repo=dict(self.digests),
            partial_cache=cache,
            graph_sha256_by_repo={r: hashlib.sha256(raw[r]).digest() for r in REPOS},
        )


def _published_bytes(tmp_path: Path, name: str, result: lexical_index.LexicalIndexResult) -> bytes:
    gen_dir = tmp_path / "gens" / name
    gen_dir.mkdir(parents=True)
    publish.write_lexical_index(gen_dir, result.data)
    return (gen_dir / lexical_index.LEXICAL_INDEX_FILENAME).read_bytes()


def _run(
    cache_dir: Path, corpus: Corpus
) -> tuple[lexical_index.LexicalIndexResult, LexicalPartialCache]:
    pending = PendingCacheWrites(cache_dir)
    cache = LexicalPartialCache(cache_dir, pending)
    result = corpus.build(cache)
    pending.flush()
    return result, cache


def _first_doc_id(data: dict, repo_id: str) -> int:
    return next(i for i, (repo, _key) in enumerate(data["documents"]) if repo == repo_id)


def test_cached_assembly_is_byte_identical_when_a_middle_repo_grows(tmp_path):
    corpus = Corpus(tmp_path)
    cache_dir = tmp_path / "cache"
    first, cold = _run(cache_dir, corpus)
    assert (cold.hits, cold.misses) == (0, 3)

    corpus.grow("repo.b")
    warm_result, warm = _run(cache_dir, corpus)
    assert (warm.hits, warm.misses) == (2, 1)
    fresh_result = corpus.build()

    warm_bytes = _published_bytes(tmp_path, "warm", warm_result)
    assert warm_bytes == _published_bytes(tmp_path, "fresh", fresh_result)
    assert warm_result.stats == fresh_result.stats

    data = fresh_result.data
    # The fixture must exercise every offset in the repo that moved.
    assert _first_doc_id(data, "repo.c") > _first_doc_id(first.data, "repo.c")
    c_docs = {i for i, (repo, _key) in enumerate(data["documents"]) if repo == "repo.c"}
    assert c_docs & set(data["stale_semantic"])
    assert c_docs & set(data["alias_exact"]["cservice"])
    shared = {p >> lexical_index.FIELD_PACK_BITS for p in data["postings"]["sharedterm"]}
    assert {data["documents"][d][0] for d in shared} == set(REPOS)
    snippet_index = lexical_index.FIELD_ORDER.index("snippet")
    assert any(
        p & 3 == snippet_index and (p >> lexical_index.FIELD_PACK_BITS) in c_docs
        for p in data["postings"]["return"]
    )
    # Warm and fresh share the merge, so check the ids against what they name.
    stale_keys = {data["documents"][d][1] for d in data["stale_semantic"]}
    assert stale_keys == {f"{r}\x1fconf/values.yaml\x1fIngress settings" for r in REPOS}
    for doc_id in data["alias_exact"]["cservice"]:
        assert data["documents"][doc_id][1].startswith("repo.c\x1fsrc/service.py\x1fCService")
    label_index = lexical_index.FIELD_ORDER.index("label")
    helper_docs = {
        p >> lexical_index.FIELD_PACK_BITS
        for p in data["postings"]["helper_c_0"]
        if p & 3 == label_index
    }
    assert [data["documents"][d][1] for d in helper_docs] == [
        "repo.c\x1fsrc/service.py\x1fhelper_c_0"
    ]
    assert "repo.c\x1fm2" in data["node_id_index"]
    assert any(key.endswith("\x1fingress.enabled") for _repo, key in data["documents"])
    assert fresh_result.stats.key_collisions >= 3
    assert fresh_result.stats.config_key_collisions == 0


def test_cache_on_and_off_publish_identical_bytes_unchanged_inputs(tmp_path):
    corpus = Corpus(tmp_path)
    _run(tmp_path / "cache", corpus)
    warm_result, warm = _run(tmp_path / "cache", corpus)
    assert (warm.hits, warm.misses) == (3, 0)
    assert _published_bytes(tmp_path, "warm", warm_result) == _published_bytes(
        tmp_path, "off", corpus.build()
    )


def test_source_digest_change_with_unchanged_graph_misses(tmp_path):
    corpus = Corpus(tmp_path)
    _run(tmp_path / "cache", corpus)
    corpus.digests["repo.b"] = "digest-repo.b-edited"
    _result, cache = _run(tmp_path / "cache", corpus)
    assert (cache.hits, cache.misses) == (2, 1)


def test_graphed_file_edit_with_same_digest_and_graph_misses(tmp_path):
    """A snippet can come from a file outside the source digest (ledger carry):
    the stat of every graphed file is in the key."""
    corpus = Corpus(tmp_path)
    _run(tmp_path / "cache", corpus)
    source = corpus.root("repo.b") / "src" / "service.py"
    source.write_text(source.read_text().replace("return 1", "return 'editedvalue'"))

    result, cache = _run(tmp_path / "cache", corpus)
    assert (cache.hits, cache.misses) == (2, 1)
    assert "editedvalue" in result.data["postings"]
    assert result.data == corpus.build().data


def test_manifest_change_misses(tmp_path):
    corpus = Corpus(tmp_path)
    _run(tmp_path / "cache", corpus)
    (corpus.graph_path("repo.a").parent / "manifest.json").unlink()
    _result, cache = _run(tmp_path / "cache", corpus)
    assert (cache.hits, cache.misses) == (2, 1)


def test_warm_run_reads_no_snippet(tmp_path, monkeypatch):
    calls = []
    real = lexical_index.build_snippet

    def counting(*args):
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(lexical_index, "build_snippet", counting)
    corpus = Corpus(tmp_path)
    _run(tmp_path / "cache", corpus)
    assert len(calls) > 0
    calls.clear()
    _result, cache = _run(tmp_path / "cache", corpus)
    assert cache.hits == 3
    assert calls == []


def test_repo_without_digest_or_graph_path_is_built_fresh_and_not_cached(tmp_path):
    corpus = Corpus(tmp_path)
    corpus.digests.pop("repo.a")
    result, cache = _run(tmp_path / "cache", corpus)
    assert (cache.hits, cache.misses) == (0, 2)
    assert len(list((tmp_path / "cache" / LEXICAL_CACHE_SUBDIR).glob("*.bin"))) == 2
    manifest = lexical_index.read_manifest(corpus.graph_path("repo.a"))
    assert partial_key("repo.a", None, manifest, "d", {}, None) is None
    assert partial_key("repo.a", b"sha", manifest, None, {}, None) is None
    unreadable = lexical_index.ManifestRead(None, readable=False)
    assert partial_key("repo.a", b"sha", unreadable, "d", {}, None) is None
    assert partial_key("repo.a", b"sha", manifest, "d", {}, None) is not None
    assert result.data == corpus.build().data


def test_corrupt_or_foreign_partial_is_a_miss(tmp_path):
    corpus = Corpus(tmp_path)
    _run(tmp_path / "cache", corpus)
    graphs = {r: json.loads(corpus.graph_path(r).read_text()) for r in REPOS}
    key = partial_key(
        "repo.b",
        hashlib.sha256(corpus.graph_path("repo.b").read_bytes()).digest(),
        lexical_index.read_manifest(corpus.graph_path("repo.b")),
        corpus.digests["repo.b"],
        graphs["repo.b"],
        corpus.root("repo.b"),
    )
    assert key is not None
    stored = read_entry(tmp_path / "cache" / LEXICAL_CACHE_SUBDIR, key)
    assert stored is not None
    good = lexical_index.RepoLexicalPartial.from_bytes(stored)
    assert good is not None
    bad_ids = json.loads(good.to_bytes())
    bad_ids["alias_exact"]["cservice"] = [10**6]
    for payload in (b"not json", json.dumps(bad_ids).encode()):
        write_entry(tmp_path / "cache" / LEXICAL_CACHE_SUBDIR, key, payload)
        result, cache = _run(tmp_path / "cache", corpus)
        assert (cache.hits, cache.misses) == (2, 1)
        assert result.data == corpus.build().data


def test_partial_parses_the_manifest_read_it_is_given(tmp_path):
    corpus = Corpus(tmp_path)
    graph_path = corpus.graph_path("repo.c")
    graph = json.loads(graph_path.read_text())
    root = corpus.root("repo.c")
    from_disk = lexical_index.build_repo_partial("repo.c", graph, root, graph_path)
    assert from_disk.stale_semantic
    # The key's read wins over a manifest rewritten after it.
    given = lexical_index.ManifestRead(b"{}")
    partial = lexical_index.build_repo_partial("repo.c", graph, root, graph_path, given)
    assert partial.stale_semantic == []


def test_partial_round_trips_through_bytes(tmp_path):
    corpus = Corpus(tmp_path)
    graph = json.loads(corpus.graph_path("repo.c").read_text())
    partial = lexical_index.build_repo_partial(
        "repo.c", graph, corpus.root("repo.c"), corpus.graph_path("repo.c")
    )
    assert lexical_index.RepoLexicalPartial.from_bytes(partial.to_bytes()) == partial
    other = json.loads(partial.to_bytes())
    other["recipe"] = "lexical-partial-v0"
    assert lexical_index.RepoLexicalPartial.from_bytes(json.dumps(other).encode()) is None


def test_garbage_collection_keeps_used_entries_only(tmp_path):
    corpus = Corpus(tmp_path)
    cache_dir = tmp_path / "cache"
    _run(cache_dir, corpus)
    entries = cache_dir / LEXICAL_CACHE_SUBDIR

    untouched = LexicalPartialCache(cache_dir, PendingCacheWrites(cache_dir))
    assert untouched.collect_garbage() == 0
    assert len(list(entries.glob("*.bin"))) == 3

    corpus.grow("repo.a")
    _result, cache = _run(cache_dir, corpus)
    assert len(list(entries.glob("*.bin"))) == 4
    assert cache.collect_garbage() == 1
    assert {p.stem for p in entries.glob("*.bin")} == cache.used_keys


def test_cache_dir_none_stages_nothing(tmp_path):
    corpus = Corpus(tmp_path)
    pending = PendingCacheWrites(None)
    cache = LexicalPartialCache(tmp_path / "cache", pending)
    result = corpus.build(cache)
    pending.flush()
    assert len(pending) == 0
    assert not (tmp_path / "cache").exists()
    assert result.data == corpus.build().data
