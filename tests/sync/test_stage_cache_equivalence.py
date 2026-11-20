"""End-to-end equivalence of the stage caches: with the same inputs, a run that
reads and writes the overlay, lexical and embedding caches publishes the same
bytes as a run with the cache off.

Every scenario rebuilds its tree at one absolute path and pins source mtimes:
shard meta records `inputs_digest`, which folds the repo root and the mtime of
every graphed source file, even with the cache off.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from graphify_mesh.sync import embedding, pipeline, repo_tags, validate
from graphify_mesh.sync.pipeline import run
from graphify_mesh.sync.source_cache import clear_source_cache

REPOS = ("a", "b", "c")
PINNED_MTIME_NS = 1_700_000_000_000_000_000
_PROVISIONAL_ID = re.compile(rb"provisional-\d{8}T\d{6}Z")
PUBLISHED_FILES = ("global-graph.json", "cross-project-overlay.json", "lexical-index.json")


def _source(name: str, body: str) -> str:
    return (
        f"class {name.title()}Parser:\n"
        f"    def parse_{name}(self, text):\n"
        f"        return text.split({body!r})\n"
        "\n"
        f"def {name}_build_index(items):\n"
        f"    return {{item: len(item) for item in items}}\n"
    )


def _graph(name: str) -> dict:
    core = f"src/{name}_core.py"
    util = f"src/{name}_util.py"
    return {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": [
            {
                "id": f"{name}1",
                "label": f"{name.title()}Parser",
                "source_file": core,
                "source_location": "L1",
            },
            {
                "id": f"{name}2",
                "label": f"parse_{name}",
                "source_file": core,
                "source_location": "L2",
            },
            {
                "id": f"{name}3",
                "label": f"{name}_build_index",
                "source_file": util,
                "source_location": "L1",
            },
        ],
        "links": [{"source": f"{name}1", "target": f"{name}2", "relation": "contains"}],
    }


def _pin_mtimes(root: Path) -> None:
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            os.utime(Path(dirpath) / filename, ns=(PINNED_MTIME_NS, PINNED_MTIME_NS))


def _fake_embed(base_url, model, inputs, timeout=30.0):
    # Deterministic, nearly parallel vectors: every cross-repo code pair scores
    # above the similar_approach line, so the overlay exact path emits edges.
    vectors = []
    for text in inputs:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vectors.append([1.0, digest[0] / 2550.0, digest[1] / 2550.0])
    return vectors


class Scenario:
    """One fresh tree at a fixed path: three repos, one edit, two runs."""

    def __init__(self, env_cls: type, base: Path, monkeypatch) -> None:
        shutil.rmtree(base, ignore_errors=True)
        base.mkdir(parents=True)
        self.env = env_cls(base)
        monkeypatch.setenv("FAKE_GRAPHIFY_CONTROL", str(self.env.control_path))
        monkeypatch.setenv("FAKE_GRAPHIFY_CALL_LOG", str(self.env.call_log_path))
        monkeypatch.setenv("FAKE_GRAPHIFY_ARGV_LOG", str(self.env.argv_log_path))
        self.roots: dict[str, Path] = {}
        self.last: pipeline.RunReport | None = None
        for name in REPOS:
            root = self.env.add_repo(name, "prod", name, f"proj-{name}", graph_fixture=None)
            (root / "src").mkdir()
            (root / "src" / f"{name}_core.py").write_text(_source(name, ","), encoding="utf-8")
            (root / "src" / f"{name}_util.py").write_text(_source(name, ";"), encoding="utf-8")
            graph_path = self.env.collection_path("prod", name) / "graph.json"
            graph_path.write_text(json.dumps(_graph(name), indent=2), encoding="utf-8")
            _pin_mtimes(root)
            self.roots[name] = root
        self.env.write_registry()

    def settings(self, cache: bool, **overrides):
        return self.env.settings(
            shrink_policy="count",
            stage_cache=cache,
            # Healthy naming through the conftest labeler stub.
            ollama_health_check=lambda *a, **kw: True,
            ollama_embed_health_check=lambda *a, **kw: True,
            **overrides,
        )

    def run(self, cache: bool, **overrides) -> pipeline.RunReport:
        clear_source_cache()
        self.last = run(self.settings(cache, **overrides))
        return self.last

    def edit(self, name: str = "a") -> None:
        path = self.roots[name] / "src" / f"{name}_core.py"
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\ndef edited_helper():\n    return 'edited'\n")
        _pin_mtimes(self.roots[name])

    def add_ungraphed_file(self, name: str) -> None:
        """A source change no graph node names: the repo's digest changes, its
        snippets and vectors do not."""
        (self.roots[name] / "src" / f"{name}_extra.py").write_text("X = 1\n", encoding="utf-8")
        _pin_mtimes(self.roots[name])

    def published(self) -> dict[str, bytes]:
        settings = self.settings(True)
        files = {
            name: (settings.global_dir / "current" / name).read_bytes() for name in PUBLISHED_FILES
        }
        # The overlay names its generation and creation time: the only fields
        # allowed to differ between two runs with the same inputs.
        assert self.last is not None
        overlay = files["cross-project-overlay.json"]
        for value in (self.last.generation_id, json.loads(overlay)["created_at"]):
            assert overlay.count(value.encode("utf-8")) == 1
            overlay = overlay.replace(value.encode("utf-8"), b"<normalized>")
        files["cross-project-overlay.json"] = overlay
        shard_dir = settings.embeddings_dir / "current"
        for path in sorted(shard_dir.iterdir()):
            if path.name.endswith((".meta.json", ".npy")):
                files[f"embeddings/{path.name}"] = path.read_bytes()
        # Each run stamps id-map rows with its own provisional id, a UTC second.
        id_map = (shard_dir / "id-map.json").read_bytes()
        assert _PROVISIONAL_ID.search(id_map)
        files["embeddings/id-map.json"] = _PROVISIONAL_ID.sub(b"provisional-<run>", id_map)
        return files

    def cache_files(self) -> set[str]:
        cache_dir = self.settings(True).cache_dir
        if not cache_dir.exists():
            return set()
        return {str(p.relative_to(cache_dir)) for p in cache_dir.rglob("*") if p.is_file()}

    def staging_dirs(self) -> list[Path]:
        cache_dir = self.settings(True).cache_dir
        return list(cache_dir.glob(".staging-*")) if cache_dir.exists() else []


MakeScenario = Callable[[], Scenario]


@pytest.fixture
def make_scenario(env, monkeypatch) -> MakeScenario:
    """Each call rebuilds the tree at the same absolute path."""
    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.89")
    monkeypatch.setattr(embedding, "embed_batch", _fake_embed)
    base = env.tmp_path / "tree"

    def make() -> Scenario:
        return Scenario(type(env), base, monkeypatch)

    return make


def _sequence(
    make_scenario: MakeScenario,
    first_cache: bool,
    second_cache: bool,
    between: Callable[[Scenario], None] | None = None,
    edit: Callable[[Scenario], None] = Scenario.edit,
    **second_overrides,
) -> tuple[Scenario, pipeline.RunReport, pipeline.RunReport]:
    scenario = make_scenario()
    first = scenario.run(first_cache)
    assert first.published, first.publish_blocked_reason
    if between is not None:
        between(scenario)
    edit(scenario)
    second = scenario.run(second_cache, **second_overrides)
    assert second.published, second.publish_blocked_reason
    return scenario, first, second


def _assert_same(expected: dict[str, bytes], actual: dict[str, bytes]) -> None:
    assert sorted(expected) == sorted(actual)
    differing = [name for name in expected if expected[name] != actual[name]]
    assert differing == []


def _cache_off_reference(
    make_scenario: MakeScenario,
    edit: Callable[[Scenario], None] = Scenario.edit,
    **second_overrides,
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """Published files of a cache-off run before and after the edit."""
    scenario = make_scenario()
    assert scenario.run(False).published
    before = scenario.published()
    edit(scenario)
    report = scenario.run(False, **second_overrides)
    assert report.published, report.publish_blocked_reason
    assert scenario.cache_files() == set()
    assert report.stage_cache["enabled"] is False
    return before, scenario.published()


def test_warm_cache_on_publishes_cache_off_bytes(make_scenario, monkeypatch):
    before, expected = _cache_off_reference(make_scenario)
    # The edit must reach the published output, or equality proves nothing.
    assert [n for n in expected if expected[n] != before.get(n)]
    assert any(n.startswith("embeddings/") and expected[n] != before[n] for n in expected)

    scenario, _first, warm = _sequence(make_scenario, True, True)

    _assert_same(expected, scenario.published())
    assert warm.overlay_edge_counts.get("similar_approach", 0) > 0
    assert warm.stage_cache["overlay"] == {"hits": 1, "misses": 2}
    assert warm.stage_cache["lexical"] == {"hits": 2, "misses": 1}
    assert warm.stage_cache["embedding"] == {"reused": 2}
    status = json.loads(scenario.settings(True).status_path.read_text(encoding="utf-8"))
    assert status["stage_cache"] == warm.stage_cache

    # Nothing changed: every entry the last run flushed survived its GC.
    unchanged = scenario.run(True)
    assert unchanged.published, unchanged.publish_blocked_reason
    assert unchanged.stage_cache["overlay"] == {"hits": 3, "misses": 0}
    assert unchanged.stage_cache["lexical"] == {"hits": 3, "misses": 0}
    assert unchanged.stage_cache["embedding"] == {"reused": 3}
    _assert_same(expected, scenario.published())


def test_repo_flag_run_with_cache_matches_repo_flag_run_without(make_scenario, monkeypatch):
    # b changes too but is not selected, so it is deferred on both sides. Its
    # change is an ungraphed file, so its vectors and the (b, c) pair hold.
    def edit_a_and_b(scenario: Scenario) -> None:
        scenario.edit("a")
        scenario.add_ungraphed_file("b")

    only_a = frozenset({"a"})
    _before, expected = _cache_off_reference(make_scenario, edit_a_and_b, only_repos=only_a)

    scenario, _first, selected = _sequence(
        make_scenario, True, True, edit=edit_a_and_b, only_repos=only_a
    )

    assert selected.pending_repos == ["b"]
    assert selected.stage_cache["overlay"]["hits"] > 0
    _assert_same(expected, scenario.published())


def test_corrupt_cache_entries_are_recomputed(make_scenario, monkeypatch):
    _before, expected = _cache_off_reference(make_scenario)

    def corrupt(scenario: Scenario) -> None:
        cache_dir = scenario.settings(True).cache_dir
        entries = [p for p in cache_dir.rglob("*") if p.is_file()]
        assert any(p.parent.name == "overlay" for p in entries)
        assert any(p.parent.name == "lexical" for p in entries)
        for path in entries:
            path.write_bytes(b"not a cache entry")

    scenario, _first, second = _sequence(make_scenario, True, True, between=corrupt)

    assert second.stage_cache["overlay"] == {"hits": 0, "misses": 3}
    assert second.stage_cache["lexical"] == {"hits": 0, "misses": 3}
    _assert_same(expected, scenario.published())
    # The recomputed entries replaced the corrupt ones.
    assert scenario.run(True).stage_cache["lexical"] == {"hits": 3, "misses": 0}


def test_blocked_publish_writes_no_cache_entry(make_scenario, monkeypatch):
    scenario = make_scenario()
    assert scenario.run(True).published
    entries_before = scenario.cache_files()
    assert entries_before
    scenario.edit()
    monkeypatch.setattr(
        validate, "run_all", lambda *a, **kw: validate.ValidationResult(ok=False, errors=["forced"])
    )

    blocked = scenario.run(True)

    assert not blocked.published
    # The edited repo was looked up and missed, so its entries were staged.
    assert blocked.stage_cache["lexical"]["misses"] == 1
    assert scenario.cache_files() == entries_before
    assert scenario.staging_dirs() == []


def test_blocked_cold_run_leaves_cache_empty(make_scenario, monkeypatch):
    scenario = make_scenario()
    monkeypatch.setattr(
        validate, "run_all", lambda *a, **kw: validate.ValidationResult(ok=False, errors=["forced"])
    )

    blocked = scenario.run(True)

    assert not blocked.published
    assert blocked.stage_cache["overlay"]["misses"] == 3
    assert scenario.cache_files() == set()
    assert scenario.staging_dirs() == []


def _add_api_files(scenario: Scenario) -> None:
    """A provides/consumes pair a -> b, and a second pair hidden by .graphifyignore."""
    a, b = scenario.roots["a"], scenario.roots["b"]
    for root in (a, b):
        (root / ".graphifyignore").write_text(".superpowers/\n", encoding="utf-8")
    for path, route in (
        (a / "src" / "WidgetController.php", "/api/v1/widgets"),
        (a / ".superpowers" / "DraftController.php", "/api/v1/drafts"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"<?php\nclass {path.stem}\n{{\n    #[Route('{route}', methods: ['GET'])]\n"
            "    public function show() {}\n}\n",
            encoding="utf-8",
        )
    (b / "src" / "client.ts").write_text("fetch('/api/v1/widgets');\n", encoding="utf-8")
    (b / ".superpowers").mkdir()
    (b / ".superpowers" / "draft.ts").write_text("fetch('/api/v1/drafts');\n", encoding="utf-8")
    for root in (a, b):
        _pin_mtimes(root)


def test_api_facts_cache_publishes_cache_off_bytes(make_scenario):
    reference = make_scenario()
    _add_api_files(reference)
    assert reference.run(False).published
    reference.edit()
    off = reference.run(False)
    assert off.published, off.publish_blocked_reason
    # Only the pair outside .superpowers/ is an edge.
    assert off.overlay_edge_counts.get("provides_api") == 1
    assert off.overlay_edge_counts.get("consumes_api") == 1
    assert off.stage_cache["api"] == {"hits": 0, "misses": 0, "skipped": 0}
    expected = reference.published()

    scenario = make_scenario()
    _add_api_files(scenario)
    cold = scenario.run(True)
    assert cold.published, cold.publish_blocked_reason
    assert cold.stage_cache["api"] == {"hits": 0, "misses": 3, "skipped": 0}
    scenario.edit()
    warm = scenario.run(True)
    assert warm.published, warm.publish_blocked_reason
    assert warm.stage_cache["api"] == {"hits": 2, "misses": 1, "skipped": 0}
    _assert_same(expected, scenario.published())
    assert any(name.startswith("overlay_api/") for name in scenario.cache_files())

    unchanged = scenario.run(True)
    assert unchanged.published, unchanged.publish_blocked_reason
    assert unchanged.stage_cache["api"] == {"hits": 3, "misses": 0, "skipped": 0}
    _assert_same(expected, scenario.published())


def test_api_facts_cache_is_off_without_content_digests(make_scenario, monkeypatch):
    # An mtime/size digest misses an edit that keeps both: never key on it.
    monkeypatch.setenv("GRAPHIFY_MESH_CONTENT_DIGEST", "0")
    scenario = make_scenario()
    _add_api_files(scenario)
    first = scenario.run(True)
    assert first.published, first.publish_blocked_reason
    assert first.overlay_edge_counts.get("provides_api") == 1
    assert first.stage_cache["api"] == {"hits": 0, "misses": 0, "skipped": 0}
    assert not any(name.startswith("overlay_api/") for name in scenario.cache_files())
