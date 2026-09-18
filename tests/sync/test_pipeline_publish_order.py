"""Publish-order, registry-integrity and shrink-authorization guards in the
sync pipeline (`sync/pipeline.py`).

Three separate failure modes are covered here:

  * the embedding index must be durable BEFORE `current` flips, so a reader
    can never see a graph generation whose vectors are not published yet;
  * two registry entries resolving to the same collection_path must fail the
    run instead of racing two workers over one graph.json;
  * an incomplete filesystem scan must never auto-authorize publishing a
    smaller graph.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphify_mesh.sync import embedding, graphify_cli, pipeline, publish, validate
from graphify_mesh.sync.discovery import reconcile as real_reconcile
from graphify_mesh.sync.pipeline import _duplicate_collection_path_errors, run
from graphify_mesh.sync.registry import RepoEntry


def _one_repo(env) -> None:
    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.write_registry()


def _record_publish_order(monkeypatch) -> list[str]:
    """Wrap the two durability steps so the test can assert their order
    without changing what either of them actually does."""
    calls: list[str] = []
    real_persist = embedding.persist_generation
    real_flip = publish.flip_current

    def _persist(*args, **kwargs):
        calls.append("persist_embeddings")
        return real_persist(*args, **kwargs)

    def _flip(*args, **kwargs):
        calls.append("flip_current")
        return real_flip(*args, **kwargs)

    monkeypatch.setattr("graphify_mesh.sync.pipeline.embedding.persist_generation", _persist)
    monkeypatch.setattr("graphify_mesh.sync.pipeline.publish.flip_current", _flip)
    return calls


# ---------------------------------------------------------------------------
# 1. embeddings persist before the graph goes live
# ---------------------------------------------------------------------------


def test_embeddings_persist_before_current_flips(env, monkeypatch):
    _one_repo(env)
    calls = _record_publish_order(monkeypatch)

    report = run(env.settings())

    assert report.published
    assert calls == ["persist_embeddings", "flip_current"], calls


def test_published_generation_id_matches_embeddings_current(env):
    _one_repo(env)
    settings = env.settings()

    report = run(settings)

    assert report.published
    embeddings_current = settings.embeddings_dir / "current"
    assert embeddings_current.is_symlink()
    assert Path(embeddings_current).resolve().name == report.generation_id


def test_validation_failure_persists_no_embeddings(env, monkeypatch):
    _one_repo(env)
    calls = _record_publish_order(monkeypatch)
    monkeypatch.setattr(
        "graphify_mesh.sync.pipeline.validate.run_all",
        lambda *a, **kw: validate.ValidationResult(ok=False, errors=["forced failure"]),
    )

    settings = env.settings()
    report = run(settings)

    assert not report.published
    assert calls == []
    assert not (settings.embeddings_dir / "current").exists()


def test_dry_run_persists_no_embeddings(env, monkeypatch):
    _one_repo(env)
    calls = _record_publish_order(monkeypatch)

    report = run(env.settings(dry_run=True))

    assert not report.published
    assert calls == []


# ---------------------------------------------------------------------------
# 2. duplicate collection paths block the run
# ---------------------------------------------------------------------------


def test_duplicate_collection_path_blocks_run(env):
    # Two repo_ids, two distinct source roots, one shared collection_path:
    # two workers would snapshot and restore the same graph.json.
    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.add_repo(
        "example-org.styleguide-mirror",
        "example-org",
        "styleguide",
        "mirror.example-org.dev.lo",
        "repo_a.json",
    )
    env.write_registry()
    settings = env.settings()

    report = run(settings)

    assert not report.published
    assert "duplicate collection_path" in report.publish_blocked_reason
    assert "example-org.styleguide" in report.publish_blocked_reason
    assert "example-org.styleguide-mirror" in report.publish_blocked_reason
    # Nothing was published and no per-project sync was attempted.
    assert report.project_actions == []
    assert not (settings.global_dir / "current").exists()


def test_duplicate_collection_path_compared_after_resolution(tmp_path):
    real = tmp_path / "graphify" / "example-org" / "styleguide"
    real.mkdir(parents=True)
    entries = [
        RepoEntry(repo_id="a", root=tmp_path / "a", collection_path=real, enabled=True),
        RepoEntry(
            repo_id="b",
            root=tmp_path / "b",
            # Textually different, same directory once resolved.
            collection_path=real.parent / ".." / "example-org" / "styleguide",
            enabled=True,
        ),
    ]

    errors = _duplicate_collection_path_errors(entries)

    assert len(errors) == 1
    assert "a, b" in errors[0]
    assert str(real) in errors[0]


def test_distinct_collection_paths_report_no_duplicates(tmp_path):
    entries = [
        RepoEntry(repo_id="a", root=tmp_path / "a", collection_path=tmp_path / "ca", enabled=True),
        RepoEntry(repo_id="b", root=tmp_path / "b", collection_path=tmp_path / "cb", enabled=True),
    ]
    assert _duplicate_collection_path_errors(entries) == []


# ---------------------------------------------------------------------------
# 3. an incomplete scan never auto-authorizes a shrink
# ---------------------------------------------------------------------------


@pytest.fixture()
def shrink_probe(monkeypatch):
    """Capture the `allow_shrink` the pipeline hands to validate.run_all, and
    force a reconciliation that claims BOTH a removed repo and an incomplete
    scan — the combination discovery.reconcile refuses to produce, which is
    exactly what the pipeline guard must not depend on."""
    captured: dict[str, bool] = {}

    def _reconcile(discovered, registry, mesh_root, scan_errors=None):
        report = real_reconcile(discovered, registry, mesh_root, scan_errors=scan_errors)
        report.scan_incomplete = True
        report.scan_errors = ["forced: cannot read scan root"]
        report.removed = ["example-org.ghost"]
        return report

    real_run_all = validate.run_all

    def _run_all(data, previous_counts, allow_shrink, skip_labeling, **kwargs):
        captured["allow_shrink"] = allow_shrink
        return real_run_all(data, previous_counts, allow_shrink, skip_labeling, **kwargs)

    monkeypatch.setattr("graphify_mesh.sync.pipeline.reconcile", _reconcile)
    monkeypatch.setattr("graphify_mesh.sync.pipeline.validate.run_all", _run_all)
    return captured


def test_incomplete_scan_does_not_auto_authorize_shrink(env, shrink_probe):
    _one_repo(env)

    report = run(env.settings())

    assert shrink_probe["allow_shrink"] is False
    assert report.reconciliation["scan_incomplete"] is True


def test_explicit_allow_shrink_still_wins_on_incomplete_scan(env, shrink_probe):
    _one_repo(env)

    run(env.settings(allow_shrink=True))

    assert shrink_probe["allow_shrink"] is True


def test_complete_scan_with_removed_repo_still_auto_authorizes_shrink(env, monkeypatch):
    _one_repo(env)
    captured: dict[str, bool] = {}
    real_run_all = validate.run_all

    def _reconcile(discovered, registry, mesh_root, scan_errors=None):
        report = real_reconcile(discovered, registry, mesh_root, scan_errors=scan_errors)
        report.removed = ["example-org.ghost"]
        return report

    def _run_all(data, previous_counts, allow_shrink, skip_labeling, **kwargs):
        captured["allow_shrink"] = allow_shrink
        return real_run_all(data, previous_counts, allow_shrink, skip_labeling, **kwargs)

    monkeypatch.setattr("graphify_mesh.sync.pipeline.reconcile", _reconcile)
    monkeypatch.setattr("graphify_mesh.sync.pipeline.validate.run_all", _run_all)

    run(env.settings())

    assert captured["allow_shrink"] is True


def test_status_json_surfaces_scan_incompleteness(env, shrink_probe):
    import json

    _one_repo(env)
    settings = env.settings()

    run(settings)

    status = json.loads(settings.status_path.read_text(encoding="utf-8"))
    assert status["scan_incomplete"] is True
    assert status["scan_errors"] == ["forced: cannot read scan root"]
    # Atomic write leaves no stray tmp file behind.
    assert not settings.status_path.with_name(settings.status_path.name + ".tmp").exists()


# ---------------------------------------------------------------------------
# 4. recipe change forces a re-embed of repos whose sources did not change
# ---------------------------------------------------------------------------


def _healthy_embed(monkeypatch) -> dict:
    """Deterministic stand-in for the native /api/embed call, plus a counter
    so a test can tell a real embed from a shard reuse."""
    calls = {"n": 0}

    def fake_embed_batch(base_url, model, inputs, timeout=30.0):
        calls["n"] += 1
        return [[1.0, 0.0, 0.0] for _ in inputs]

    monkeypatch.setattr("graphify_mesh.sync.embedding.embed_batch", fake_embed_batch)
    return calls


def test_unchanged_repo_reuses_its_shard_when_the_recipe_is_unchanged(env, monkeypatch):
    _one_repo(env)
    _healthy_embed(monkeypatch)
    kwargs = {
        "ollama_embed_health_check": lambda *a, **kw: True,
        "ollama_embed_model": "model-a",
    }

    first = run(env.settings(**kwargs))
    assert first.published
    second = run(env.settings(**kwargs))

    assert second.published
    assert second.embedding_stats["reused_repos_unchanged"] >= 1
    assert second.embedding_stats["embedded"] == 0


def test_unchanged_repo_is_re_embedded_after_the_model_changes(env, monkeypatch):
    _one_repo(env)
    _healthy_embed(monkeypatch)

    first = run(
        env.settings(ollama_embed_health_check=lambda *a, **kw: True, ollama_embed_model="model-a")
    )
    assert first.published

    # Sources untouched, so the repo is still ACTION_SKIP for the graphify CLI
    # stages — only the embed stage's input set may change.
    second = run(
        env.settings(ollama_embed_health_check=lambda *a, **kw: True, ollama_embed_model="model-b")
    )

    assert second.published
    assert [a["status"] for a in second.project_actions] == ["unchanged"]
    assert second.embedding_stats["reused_repos_unchanged"] == 0
    assert second.embedding_stats["embedded"] > 0


# ---------------------------------------------------------------------------
# 5. the repo-tag map is derived under a version-parity check
# ---------------------------------------------------------------------------


def test_repo_tag_computation_receives_the_graphify_binary(env, monkeypatch):
    _one_repo(env)
    captured: dict[str, object] = {}
    real = __import__(
        "graphify_mesh.sync.repo_tags", fromlist=["compute_tag_to_repo_id"]
    ).compute_tag_to_repo_id

    def _spy(paths, repo_ids, *, graphify_bin=None):
        captured["graphify_bin"] = graphify_bin
        return real(paths, repo_ids, graphify_bin=graphify_bin)

    monkeypatch.setattr("graphify_mesh.sync.pipeline.repo_tags.compute_tag_to_repo_id", _spy)

    settings = env.settings()
    report = run(settings)

    assert report.published
    assert captured["graphify_bin"] == settings.graphify_bin


# ---------------------------------------------------------------------------
# 6. upstream cross-repo edges are stripped, not left to block the publish
# ---------------------------------------------------------------------------


def _two_repos(env) -> None:
    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.add_repo(
        "example-org.services",
        "example-org",
        "services",
        "services.example-org.dev.lo",
        "repo_b.json",
    )
    env.write_registry()


def _inject_upstream_cross_repo_edges(monkeypatch) -> None:
    """Reproduce what `graphify merge-graphs` does on its own: a `same_type_as`
    link marked `context="cross_repo"` between two repos' type declarations,
    plus a plain `calls` edge spanning the same two repos
    (graphify/cli.py:2719-2727)."""
    real_merge = graphify_cli.run_merge_graphs

    def _merge(*args, **kwargs):
        result = real_merge(*args, **kwargs)
        out_path = Path(args[2])
        data = json.loads(out_path.read_text(encoding="utf-8"))
        first_node_per_repo: dict[str, str] = {}
        for node in data["nodes"]:
            first_node_per_repo.setdefault(str(node["id"]).partition("::")[0], node["id"])
        prefixes = sorted(first_node_per_repo)
        assert len(prefixes) >= 2, prefixes
        left, right = first_node_per_repo[prefixes[0]], first_node_per_repo[prefixes[1]]
        data["links"].extend(
            [
                {
                    "source": left,
                    "target": right,
                    "relation": "same_type_as",
                    "context": "cross_repo",
                    "confidence": "INFERRED",
                },
                {"source": right, "target": left, "relation": "calls"},
            ]
        )
        out_path.write_text(json.dumps(data), encoding="utf-8")
        return result

    monkeypatch.setattr("graphify_mesh.sync.pipeline.graphify_cli.run_merge_graphs", _merge)


def test_upstream_cross_repo_edges_are_stripped_and_the_run_publishes(env, monkeypatch):
    _two_repos(env)
    _inject_upstream_cross_repo_edges(monkeypatch)
    settings = env.settings()

    report = run(settings)

    assert report.published, report.validation_errors
    assert report.stripped_cross_repo_edges == 2
    assert report.validation_errors == [] or all(
        "forbidden-edge" not in e for e in report.validation_errors
    )

    published = json.loads(
        (settings.global_dir / "current" / "global-graph.json").read_text(encoding="utf-8")
    )
    for link in published["links"]:
        assert link.get("context") != "cross_repo"
        src = str(link["source"]).partition("::")[0]
        dst = str(link["target"]).partition("::")[0]
        assert src == dst, link

    status = json.loads(settings.status_path.read_text(encoding="utf-8"))
    assert status["stripped_cross_repo_edges"] == 2


def test_no_cross_repo_edges_means_nothing_is_stripped(env):
    _two_repos(env)

    report = run(env.settings())

    assert report.published
    assert report.stripped_cross_repo_edges == 0


def test_a_cross_repo_edge_added_after_the_strip_still_blocks_publish(env, monkeypatch):
    """The strip is the primary mechanism; validate_forbidden_edges is the
    backstop. An edge that appears after the strip is a genuine bug, and the
    publish must fail rather than ship it."""
    _two_repos(env)
    real_strip = pipeline.strip_cross_repo_edges

    def _strip_then_reintroduce(graph_data):
        removed = real_strip(graph_data)
        first_node_per_repo: dict[str, str] = {}
        for node in graph_data["nodes"]:
            first_node_per_repo.setdefault(str(node["id"]).partition("::")[0], node["id"])
        prefixes = sorted(first_node_per_repo)
        graph_data["links"].append(
            {
                "source": first_node_per_repo[prefixes[0]],
                "target": first_node_per_repo[prefixes[1]],
                "relation": "calls",
            }
        )
        return removed

    monkeypatch.setattr(pipeline, "strip_cross_repo_edges", _strip_then_reintroduce)

    settings = env.settings()
    report = run(settings)

    assert not report.validation_ok
    assert any("forbidden-edge" in e for e in report.validation_errors)
    assert not report.published
    assert not (settings.global_dir / "current").exists()


def test_strip_keeps_bare_and_same_repo_edges():
    graph = {
        "nodes": [],
        "links": [
            {"source": "repo.a::x", "target": "repo.a::y", "relation": "calls"},
            {"source": "external-thing", "target": "repo.a::y", "relation": "calls"},
            {"source": "repo.a::x", "target": "repo.b::y", "relation": "calls"},
            {
                "source": "repo.a::x",
                "target": "repo.a::y",
                "relation": "same_type_as",
                "context": "cross_repo",
            },
        ],
    }

    removed = pipeline.strip_cross_repo_edges(graph)

    assert removed == 2
    assert [link["source"] for link in graph["links"]] == ["repo.a::x", "external-thing"]


# ---------------------------------------------------------------------------
# 7. embedding GC runs only after the graph's own `current` has flipped
# ---------------------------------------------------------------------------


def test_embedding_gc_runs_after_flip_current(env, monkeypatch):
    _one_repo(env)
    calls: list[str] = []
    real_flip = publish.flip_current
    real_gc = embedding.gc_embedding_generations

    def _flip(*args, **kwargs):
        calls.append("flip_current")
        return real_flip(*args, **kwargs)

    def _gc(*args, **kwargs):
        calls.append("gc_embeddings")
        return real_gc(*args, **kwargs)

    monkeypatch.setattr("graphify_mesh.sync.pipeline.publish.flip_current", _flip)
    monkeypatch.setattr("graphify_mesh.sync.pipeline.embedding.gc_embedding_generations", _gc)

    report = run(env.settings())

    assert report.published
    assert calls == ["flip_current", "gc_embeddings"], calls
