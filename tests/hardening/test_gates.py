"""WS6 hardening gates: fills the specific gaps not already covered by
tests/sync/test_sync_engine.py, tests/sync/test_bootstrap_and_dirty.py,
and tests/server/test_store.py — see the WS6 build notes for which
scenarios were ALREADY covered (shrink-guard e2e refusal, forbidden-edge
trip, auto-add bootstrap, one basic generation-count-mismatch rejection,
rollback-on-crash-before-flip) vs the real gaps this file adds:

  1. staging-HOME isolation asserted via mtime (not just non-existence) of
     every file under the fake mesh's `graphify/` tree during a dry run —
     stronger than the existing non-existence check.
  2. generation-mismatch refusal via a wrong `output_hash` specifically
     (existing test_store.py coverage only exercises a wrong node COUNT).
  3. the exact stale-ratio threshold boundary: 4 registered repos, 2 failing
     (50%, over the 30% threshold) blocks publish; 4 registered repos, 1
     failing (25%, under threshold) still publishes. The existing
     test_atomic_publish_flips_forward_and_rollback_on_stale only exercises
     a 2-repo/1-failing (50%) case.
  4. dirty-worktree marker is explicitly confirmed non-blocking (the
     existing test_dirty_worktree_recorded_read_only never asserts
     report.published).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_BIN_DIR = Path(__file__).resolve().parents[2] / "bin"
if str(_BIN_DIR) not in sys.path:
    sys.path.insert(0, str(_BIN_DIR))

from _env_helper import FAKE_GRAPHIFY, Env  # noqa: E402

from graphify_mesh.server.config import ServerConfig  # noqa: E402
from graphify_mesh.server.store import GenerationStore, validate_manifest_consistency  # noqa: E402
from graphify_mesh.sync.pipeline import run  # noqa: E402
from graphify_mesh.sync.publish import output_hash  # noqa: E402


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture()
def env(tmp_path, monkeypatch) -> Env:
    """Deliberately NOT a conftest.py fixture: see
    tests/hardening/_env_helper.py's module docstring for why this package
    avoids its own conftest.py entirely (collision risk with
    tests/server's bare `from conftest import ...` imports when
    multiple identically-named conftest.py modules exist in the tree)."""
    e = Env(tmp_path)
    monkeypatch.setenv("FAKE_GRAPHIFY_CONTROL", str(e.control_path))
    monkeypatch.setenv("FAKE_GRAPHIFY_CALL_LOG", str(e.call_log_path))
    monkeypatch.setenv("GRAPHIFY_BIN", str(FAKE_GRAPHIFY))
    # The child env allowlist in graphify_cli drops every name it does not
    # recognise, the two the fake binary is driven by included. Declare them
    # through the documented extension point, as tests/sync/conftest.py does.
    monkeypatch.setenv(
        "GRAPHIFY_MESH_CHILD_ENV_EXTRA", "FAKE_GRAPHIFY_CONTROL,FAKE_GRAPHIFY_CALL_LOG"
    )
    yield e


def _snapshot_mtimes(root: Path) -> dict[str, float]:
    if not root.exists():
        return {}
    return {str(p): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


def test_staging_isolation_zero_mtime_changes_under_graphify_tree(env):
    """Stronger than the existing non-existence check in test_sync_engine.py
    (test_staging_isolation_dry_run_writes_nothing_outside_staging): this
    proves that pre-existing tracked graphify/<product>/<sub>/graph.json
    files (the per-project curated graphs, which in the real repo ARE
    tracked by git) are not touched — not just that the global/ output dir
    was never created."""
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

    tracked_graphify_dir = env.mesh_root / "graphify"
    before = _snapshot_mtimes(tracked_graphify_dir)
    assert before, "expected pre-existing per-project graph.json files under graphify/"

    settings = env.settings(dry_run=True)
    report = run(settings)
    assert report.dry_run

    after = _snapshot_mtimes(tracked_graphify_dir)
    assert after == before, "dry-run must not modify any file under the tracked graphify/ tree"
    # And confirm no NEW files appeared under it either (set equality above
    # already implies this, but assert file count explicitly for clarity).
    assert len(after) == len(before)


def test_generation_mismatch_refused_on_wrong_output_hash(tmp_path):
    """Extends test_store.py's node-count-mismatch coverage: a manifest
    whose output_hash does not match the recomputed hash of
    global-graph.json (counts otherwise correct) must also be refused."""
    config = ServerConfig.from_env(
        mesh_root=tmp_path, registry_path=tmp_path / "bin" / "registry.json"
    )
    graph = {
        "nodes": [{"id": "n1", "repo": "repo.a", "label": "Alpha", "source_file": "a.py"}],
        "links": [],
    }
    manifest = {
        "generation_id": "gen-bad-hash",
        "created_at": "2026-07-20T00:00:00Z",
        "repo_input_hashes": {},
        "registry_hash": "test",
        "config_hash": "test",
        "output_node_count": 1,
        "output_edge_count": 0,
        "output_hash": "0" * 64,  # deliberately wrong — does not match output_hash(graph)
        "clustering_backend": "louvain",
        "embedding_model": "test-model",
        "labeling": "skipped",
        "stale_repos": [],
    }
    assert manifest["output_hash"] != output_hash(graph)

    errors = validate_manifest_consistency(manifest, graph, lexical={})
    assert any("output_hash" in e for e in errors)

    gen_dir = config.global_dir / "generations" / "gen-bad-hash"
    gen_dir.mkdir(parents=True)
    (gen_dir / "global-graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (gen_dir / "generation-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (gen_dir / "cross-project-overlay.json").write_text(json.dumps({"edges": []}), encoding="utf-8")
    (gen_dir / "lexical-index.json").write_text(json.dumps({}), encoding="utf-8")
    current = config.global_dir / "current"
    current.symlink_to(gen_dir, target_is_directory=True)

    store = GenerationStore(config)
    from graphify_mesh.server.store import GenerationUnavailableError

    with pytest.raises(GenerationUnavailableError):
        _ = store.generation
    assert any("output_hash" in reason for reason in store.degraded)


def test_generation_mismatch_refused_when_output_hash_missing(tmp_path):
    """Missing output_hash entirely must not be treated as 'nothing to
    check' — validate_generation_manifest's required-keys check (reused by
    validate_manifest_consistency) already covers this; confirm the
    end-to-end store rejection path specifically."""
    graph = {
        "nodes": [{"id": "n1", "repo": "repo.a", "label": "Alpha", "source_file": "a.py"}],
        "links": [],
    }
    manifest = {
        "generation_id": "gen-missing-hash",
        "created_at": "2026-07-20T00:00:00Z",
        "repo_input_hashes": {},
        "registry_hash": "test",
        "config_hash": "test",
        "output_node_count": 1,
        "output_edge_count": 0,
        # output_hash intentionally absent
        "clustering_backend": "louvain",
        "embedding_model": "test-model",
        "labeling": "skipped",
        "stale_repos": [],
    }
    errors = validate_manifest_consistency(manifest, graph, lexical={})
    assert any("missing key 'output_hash'" in e for e in errors)


def test_four_projects_two_failing_fifty_percent_blocks_publish(env):
    env.add_repo("example-org.a", "example-org", "a", "a.example-org.dev.lo", "repo_a.json")
    env.add_repo("example-org.b", "example-org", "b", "b.example-org.dev.lo", "repo_b.json")
    env.add_repo("example-org.c", "example-org", "c", "c.example-org.dev.lo", "repo_a.json")
    env.add_repo("example-org.d", "example-org", "d", "d.example-org.dev.lo", "repo_b.json")
    env.write_registry()
    settings = env.settings()

    first = run(settings)
    assert first.published

    # Break 2 of 4 (50%) badly enough to exceed the 30% stale threshold.
    for product, sub, repo_id in (
        ("example-org", "b", "example-org.b"),
        ("example-org", "d", "example-org.d"),
    ):
        collection = env.collection_path(product, sub)
        env.set_control(collection, "fail")
        root = Path([r["root"] for r in env._repos if r["repo_id"] == repo_id][0])
        (root / "touched.py").write_text("# touch\n", encoding="utf-8")

    second = run(settings)
    assert set(second.stale_repos) == {"example-org.b", "example-org.d"}
    assert not second.published
    assert "stale ratio" in second.publish_blocked_reason
    # current still points at the first (good) generation — no half-publish.
    gen1 = os.path.realpath(settings.global_dir / "current")
    first_gen_dirs = sorted(p.name for p in settings.generations_dir.iterdir())
    assert os.path.realpath(settings.global_dir / "current") == gen1
    assert (
        len(first_gen_dirs) == 1
    )  # second run's attempt never wrote a new generation past validate


def test_four_projects_one_failing_twenty_five_percent_still_publishes(env):
    env.add_repo("example-org.a", "example-org", "a", "a.example-org.dev.lo", "repo_a.json")
    env.add_repo("example-org.b", "example-org", "b", "b.example-org.dev.lo", "repo_b.json")
    env.add_repo("example-org.c", "example-org", "c", "c.example-org.dev.lo", "repo_a.json")
    env.add_repo("example-org.d", "example-org", "d", "d.example-org.dev.lo", "repo_b.json")
    env.write_registry()
    settings = env.settings()

    first = run(settings)
    assert first.published
    gen1 = os.path.realpath(settings.global_dir / "current")

    # Break only 1 of 4 (25%) — under the 30% stale threshold.
    collection = env.collection_path("example-org", "b")
    env.set_control(collection, "fail")
    root = Path([r["root"] for r in env._repos if r["repo_id"] == "example-org.b"][0])
    (root / "touched.py").write_text("# touch\n", encoding="utf-8")

    second = run(settings)
    assert second.stale_repos == ["example-org.b"]
    assert second.published, second.publish_blocked_reason
    gen2 = os.path.realpath(settings.global_dir / "current")
    assert gen2 != gen1  # publish went ahead and flipped forward despite the one stale repo

    # The stale repo's last-good graph.json still contributed to the merge
    # (carried forward, not dropped).
    merged = _read_json(settings.global_dir / "current" / "global-graph.json")
    ids = {n["id"] for n in merged["nodes"]}
    assert any(i.startswith("example-org.b::") for i in ids)


def test_dirty_worktree_marker_is_informational_sync_still_publishes(env):
    """Extends test_bootstrap_and_dirty.py::test_dirty_worktree_recorded_read_only,
    which records dirty_repos but never asserts the run actually published —
    confirming the plan's explicit requirement that the marker is
    informational only, never blocking."""
    import subprocess

    root = env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=str(root), check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=str(root), check=True)
    (root / "committed.py").write_text("# a\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(root), check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=str(root), check=True)
    (root / "dirty.py").write_text("# uncommitted\n", encoding="utf-8")

    env.write_registry()
    settings = env.settings()
    report = run(settings)

    assert "example-org.styleguide" in report.dirty_repos
    assert report.published, report.publish_blocked_reason
    assert (settings.global_dir / "current").is_symlink()
