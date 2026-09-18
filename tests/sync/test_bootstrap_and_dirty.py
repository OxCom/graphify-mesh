from __future__ import annotations

import json
import subprocess
from pathlib import Path

from graphify_mesh.sync.pipeline import run
from graphify_mesh.sync.state import is_worktree_dirty


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_auto_add_bootstraps_fresh_project(env):
    # Registered (collection dir + symlink exist) but no graph.json yet.
    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        graph_fixture=None,
    )
    env.write_registry()
    settings = env.settings()

    report = run(settings)

    assert "example-org.styleguide" in report.reconciliation["auto_add"]
    action = next(a for a in report.project_actions if a["repo_id"] == "example-org.styleguide")
    assert action["action"] == "bootstrap"
    assert action["status"] == "bootstrapped"
    assert (env.collection_path("example-org", "styleguide") / "graph.json").exists()
    assert report.published


def test_failed_bootstrap_not_claimed_as_auto_added(env):
    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        graph_fixture=None,
    )
    env.write_registry()
    collection = env.collection_path("example-org", "styleguide")
    env.set_control(collection, "bootstrap_fail")
    settings = env.settings()

    report = run(settings)

    action = next(a for a in report.project_actions if a["repo_id"] == "example-org.styleguide")
    assert action["status"] == "bootstrap_failed"
    assert not (collection / "graph.json").exists()
    assert "example-org.styleguide" in report.stale_repos
    # Not published as a successfully-added project: no graph.json exists,
    # so it can never be part of the merge input, and with only 1 repo
    # registered and it being 100% stale, publish is refused entirely.
    assert not report.published


def test_dirty_worktree_recorded_read_only(env):
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
    # Engine only ever ran read-only `git status --porcelain` — the repo is
    # still dirty afterwards (nothing was committed on the project's behalf).
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(root), capture_output=True, text=True, check=True
    )
    assert status.stdout.strip()


def test_status_disables_scanned_repo_config_hooks(tmp_path, monkeypatch):
    # The scanned repo's own config must not be able to run commands here:
    # core.fsmonitor and core.hooksPath both execute programs it names.
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    seen: list[list[str]] = []

    def fake_run(argv, **kwargs):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert is_worktree_dirty(root) is False
    assert seen[0][:6] == [
        "git",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
        "status",
    ]
    assert seen[0][-1] == "--porcelain"
