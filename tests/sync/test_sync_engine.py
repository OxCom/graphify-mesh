from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from graphify_mesh.sync import graphify_cli, publish
from graphify_mesh.sync.pipeline import run


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_first_run_publishes_generation(env):
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

    settings = env.settings()
    report = run(settings)

    assert report.merge_ok
    assert report.validation_ok, report.validation_errors
    assert report.published
    assert (settings.global_dir / "current").is_symlink()
    merged = _read_json(settings.global_dir / "current" / "global-graph.json")
    ids = {n["id"] for n in merged["nodes"]}
    assert any(i.startswith("example-org.styleguide::") for i in ids)
    assert any(i.startswith("example-org.services::") for i in ids)


def test_delete_project_pruned_from_global_no_dangling_edges(env):
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
    settings = env.settings()
    first = run(settings)
    assert first.published

    # Delete the second project's graphify-out entirely (registered project
    # disappears mid-run — root dir gone, symlink gone with it).
    shutil.rmtree(env.scan_root / "services.example-org.dev.lo")

    second = run(settings)
    assert "example-org.services" in second.reconciliation["removed"]
    assert second.merge_ok
    assert second.published
    merged = _read_json(settings.global_dir / "current" / "global-graph.json")
    ids = {n["id"] for n in merged["nodes"]}
    assert not any(i.startswith("example-org.services::") for i in ids)
    assert any(i.startswith("example-org.styleguide::") for i in ids)
    # No dangling edges: every edge endpoint resolves to a node in the set.
    for link in merged["links"]:
        assert link["source"] in ids
        assert link["target"] in ids


def test_remove_project_flagged_and_excluded(env):
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
    settings = env.settings()
    run(settings)

    shutil.rmtree(env.scan_root / "services.example-org.dev.lo")
    report = run(settings)

    assert "example-org.services" in report.reconciliation["removed"]
    project_repo_ids = {a["repo_id"] for a in report.project_actions}
    assert (
        "example-org.services" not in project_repo_ids
    )  # pruned before per-project sync, not silently kept


def test_broken_symlink_reported_not_crashed_pipeline(env):
    root = env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.write_registry()
    settings = env.settings()
    first = run(settings)
    assert first.published

    link = root / "graphify-out"
    ghost = env.mesh_root / "graphify" / "example-org" / "ghost"
    link.unlink()
    link.symlink_to(ghost, target_is_directory=True)

    second = run(settings)
    assert "example-org.styleguide" in second.reconciliation["broken"]
    # Last-good graph.json still present in the collection_path, so it's
    # still contributed to the merge (engine doesn't crash or drop it).
    assert second.merge_ok
    assert second.published


def test_shrink_refusal_detected_last_good_kept(env):
    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.write_registry()
    collection = env.collection_path("example-org", "styleguide")
    original = _read_json(collection / "graph.json")

    env.set_control(collection, "shrink")
    settings = env.settings()
    report = run(settings)

    statuses = {a["repo_id"]: a["status"] for a in report.project_actions}
    assert statuses["example-org.styleguide"] == "shrink_refused"
    assert "example-org.styleguide" in report.stale_repos
    # Last-good graph.json restored byte-for-byte, not the shrunk content.
    restored = _read_json(collection / "graph.json")
    assert restored == original
    # 1/1 registered repos stale -> exceeds 30% threshold -> no publish.
    assert not report.published
    assert "stale ratio" in report.publish_blocked_reason


def test_shrink_accepted_with_allow_shrink_state_advances(env):
    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.write_registry()
    collection = env.collection_path("example-org", "styleguide")
    original = _read_json(collection / "graph.json")

    env.set_control(collection, "shrink")
    settings = env.settings(allow_shrink=True)
    report = run(settings)

    actions = {a["repo_id"]: a for a in report.project_actions}
    row = actions["example-org.styleguide"]
    assert row["status"] == "updated"
    assert "operator-authorized" in row["reason"]
    assert "example-org.styleguide" not in report.stale_repos
    # The shrunken graph is kept on disk, not the last-good snapshot.
    kept = _read_json(collection / "graph.json")
    assert kept != original
    assert len(kept["nodes"]) < len(original["nodes"])

    # State advanced: a second run (no allow_shrink needed) decides skip.
    second = run(env.settings())
    second_statuses = {a["repo_id"]: a["status"] for a in second.project_actions}
    assert second_statuses["example-org.styleguide"] == "unchanged"


def test_forbidden_edge_invariant_trips_validation(env):
    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.add_repo(
        "example-org.gamma",
        "example-org",
        "gamma",
        "gamma.example-org.dev.lo",
        "repo_forbidden_edge.json",
    )
    env.write_registry()
    settings = env.settings()
    report = run(settings)

    assert report.merge_ok
    assert not report.validation_ok
    assert any("forbidden-edge" in e for e in report.validation_errors)
    assert not report.published
    assert not (settings.global_dir / "current").exists()


def test_atomic_publish_flips_forward_and_rollback_on_stale(env):
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
    settings = env.settings()

    first = run(settings)
    assert first.published
    gen1 = os.path.realpath(settings.global_dir / "current")

    # Break one repo badly enough to exceed the stale threshold (1/2 = 50%).
    collection_b = env.collection_path("example-org", "services")
    env.set_control(collection_b, "fail")
    # Force decide_action to actually invoke the CLI on repo_b by touching
    # its source tree so the code manifest changes on this run.
    root_b = Path([r["root"] for r in env._repos if r["repo_id"] == "example-org.services"][0])
    (root_b / "touched.py").write_text("# touch\n", encoding="utf-8")

    second = run(settings)
    assert "example-org.services" in second.stale_repos
    assert not second.published
    assert "stale ratio" in second.publish_blocked_reason
    # Rollback = current was never moved.
    assert os.path.realpath(settings.global_dir / "current") == gen1

    # Fix it back to normal and confirm forward progress resumes.
    env.set_control(collection_b, "normal")
    third = run(settings)
    assert third.published
    gen3 = os.path.realpath(settings.global_dir / "current")
    assert gen3 != gen1


def test_publish_failure_between_write_and_flip_leaves_current_untouched(env, monkeypatch):
    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.write_registry()
    settings = env.settings()

    first = run(settings)
    assert first.published
    gen1 = os.path.realpath(settings.global_dir / "current")
    generations_before = sorted(p.name for p in settings.generations_dir.iterdir())

    def _boom(*a, **kw):
        raise RuntimeError("simulated crash after generation dir write, before symlink flip")

    monkeypatch.setattr(publish, "flip_current", _boom)
    monkeypatch.setattr("graphify_mesh.sync.pipeline.publish.flip_current", _boom)

    root = Path([r["root"] for r in env._repos if r["repo_id"] == "example-org.styleguide"][0])
    (root / "touched2.py").write_text("# touch\n", encoding="utf-8")

    with pytest.raises(RuntimeError):
        run(settings)

    # current must still point at the previous good generation.
    assert os.path.realpath(settings.global_dir / "current") == gen1
    generations_after = sorted(p.name for p in settings.generations_dir.iterdir())
    # The new (unpublished) generation dir was written and kept, not deleted.
    assert len(generations_after) == len(generations_before) + 1


def test_staging_isolation_dry_run_writes_nothing_outside_staging(env):
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
    settings = env.settings(dry_run=True)

    assert not settings.global_dir.exists()
    report = run(settings)
    assert report.dry_run

    # No file was written anywhere under the real mesh tree.
    assert not settings.global_dir.exists()
    assert not settings.state_path.exists()
    assert not settings.status_path.exists()
    assert not (settings.mesh_root / "graphify" / "global" / "generations").exists()


def test_home_only_redirected_during_merge_graphs_call(env, monkeypatch):
    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.write_registry()
    settings = env.settings()

    captured_envs = []
    real_run = graphify_cli._run

    def spy_run(argv, cwd, env, timeout=900):
        if len(argv) > 1 and "merge-graphs" in argv:
            captured_envs.append(dict(env or {}))
        return real_run(argv, cwd, env, timeout)

    monkeypatch.setattr(graphify_cli, "_run", spy_run)

    real_home = os.environ.get("HOME")
    report = run(settings)
    assert report.published
    # Parent process HOME is untouched after the call.
    assert os.environ.get("HOME") == real_home
    assert len(captured_envs) == 1
    assert captured_envs[0]["HOME"] != real_home
    assert "graphify-mesh-sync-staging-" in captured_envs[0]["HOME"]


def test_skip_labeling_makes_zero_naming_calls(env):
    """Regression test: --skip-labeling must guarantee zero cluster-only/
    label invocations (and therefore zero possible Ollama contact),
    regardless of ollama_base_url or health-check state. A prior bug called
    naming.run_naming() unconditionally and only used skip_labeling for a
    log message + the validation bypass, so a "skip" run still silently
    executed real cluster-only/label subprocess calls."""
    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.write_registry()
    # Force the health check to True (would-be-healthy) so if the bug
    # regresses, run_naming would actually proceed into cluster-only/label
    # rather than bailing out via the unrelated degraded-mode path — this
    # test must fail for the RIGHT reason (calls happened) not by accident.
    settings = env.settings(skip_labeling=True, ollama_health_check=lambda *a, **kw: True)

    report = run(settings)

    assert report.published
    assert report.labeling == "skipped (--skip-labeling)"
    call_log = env.read_call_log()
    naming_calls = [c for c in call_log if c.get("cmd") in ("cluster-only", "label")]
    assert naming_calls == [], (
        f"skip_labeling=True must make zero naming calls, got: {naming_calls}"
    )
