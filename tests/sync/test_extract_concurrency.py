"""Task 7: bounded thread-pool parallelism for the per-repo extract loop.

Two repos both needing ACTION_EXTRACT should run overlapped when
settings.extract_concurrency >= 2 (apply_action is self-contained per repo:
own collection_path writes, unique mkstemp snapshots -> thread-safe), but
report.project_actions / state mutations must land in sorted repo_id order
regardless of which future actually finishes first.
"""

from __future__ import annotations

import threading
import time

from graphify_mesh.sync import pipeline
from graphify_mesh.sync.sync_project import ProjectOutcome


def _make_two_extract_repos(env):
    """Two repos, both with an established prior state, then both touched so
    their semantic manifest changes -> decide_action returns ACTION_EXTRACT
    for both on the next run()."""
    env.add_repo(
        "example-org.aaa-first",
        "example-org",
        "aaa-first",
        "aaa-first.example-org.dev.lo",
        "repo_a.json",
    )
    env.add_repo(
        "example-org.zzz-second",
        "example-org",
        "zzz-second",
        "zzz-second.example-org.dev.lo",
        "repo_b.json",
    )
    env.write_registry()
    settings = env.settings()
    first = pipeline.run(settings)
    assert first.published

    for _repo_id, root_name in (
        ("example-org.aaa-first", "aaa-first.example-org.dev.lo"),
        ("example-org.zzz-second", "zzz-second.example-org.dev.lo"),
    ):
        root = env.scan_root / root_name
        (root / "touched.md").write_text("# semantic touch\n", encoding="utf-8")

    return settings


def _fake_apply_action_factory(active, lock, delay=0.05):
    def fake_apply_action(
        repo_id,
        graphify_bin,
        root,
        collection_path,
        action,
        current_manifest,
        staging_home=None,
        *,
        allow_shrink=False,
        # Accept any further keyword-only guard settings the real apply_action
        # grows (shrink_tolerance, ...) — this double only measures concurrency,
        # so it must not pin the full signature.
        **_guard_settings,
    ):
        with lock:
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
        time.sleep(delay)
        with lock:
            active["count"] -= 1
        return ProjectOutcome(
            repo_id=repo_id,
            action=action,
            status="updated",
            reason=None,
            dirty_worktree=False,
            new_manifest=current_manifest,
        )

    return fake_apply_action


def test_extract_actions_run_concurrently_and_report_in_order(env, monkeypatch):
    settings = _make_two_extract_repos(env)
    settings.extract_concurrency = 2

    active = {"count": 0, "max": 0}
    lock = threading.Lock()
    monkeypatch.setattr(pipeline, "apply_action", _fake_apply_action_factory(active, lock))

    report = pipeline.run(settings)

    assert active["max"] == 2, "both extract actions should have overlapped"
    extract_rows = [a for a in report.project_actions if a["action"] != "skip"]
    extract_repo_ids = [a["repo_id"] for a in extract_rows if a["status"] == "updated"]
    assert extract_repo_ids == sorted(extract_repo_ids)
    assert extract_repo_ids == ["example-org.aaa-first", "example-org.zzz-second"]


def test_extract_concurrency_one_is_sequential(env, monkeypatch):
    settings = _make_two_extract_repos(env)
    settings.extract_concurrency = 1

    active = {"count": 0, "max": 0}
    lock = threading.Lock()
    monkeypatch.setattr(pipeline, "apply_action", _fake_apply_action_factory(active, lock))

    report = pipeline.run(settings)

    assert active["max"] == 1, "concurrency=1 must stay fully sequential"
    extract_rows = [a for a in report.project_actions if a["action"] != "skip"]
    extract_repo_ids = [a["repo_id"] for a in extract_rows if a["status"] == "updated"]
    assert extract_repo_ids == sorted(extract_repo_ids)
