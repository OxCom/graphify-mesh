from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphify_mesh.sync import cli, embedding, graphify_cli, pipeline, repo_tags
from graphify_mesh.sync.locking import LockHeldError
from graphify_mesh.sync.pipeline import run
from graphify_mesh.sync.state import load_state
from graphify_mesh.sync.sync_project import (
    ACTION_BOOTSTRAP,
    ACTION_SKIP,
    ACTION_UPDATE,
    GRAPH_HASH_KEY,
    IGNORE_RULES_KEY,
)


def _argv(env, *extra: str) -> list[str]:
    return [
        "--once",
        "--mesh-root",
        str(env.mesh_root),
        "--scan-root",
        str(env.scan_root),
        "--registry",
        str(env.registry_path),
        *extra,
    ]


def _fail_if_run(monkeypatch) -> None:
    def _run(settings):
        pytest.fail("run() must not be reached when --repo validation fails")

    monkeypatch.setattr(cli, "run", _run)


def _capture_run(monkeypatch) -> list:
    seen: list = []

    def _run(settings):
        seen.append(settings)
        # Stop after capture; main maps this to exit code 3.
        raise LockHeldError("stop")

    monkeypatch.setattr(cli, "run", _run)
    return seen


def test_unknown_repo_exits_2_before_lock(env, monkeypatch, capsys):
    env.add_repo("a", "prod", "a", "proj-a")
    env.write_registry()
    _fail_if_run(monkeypatch)

    assert cli.main(_argv(env, "--repo", "nope")) == 2

    err = capsys.readouterr().err
    assert "nope" in err
    assert "unknown" in err


def test_disabled_repo_exits_2(env, monkeypatch, capsys):
    env.add_repo("a", "prod", "a", "proj-a")
    env.add_repo("off", "prod", "off", "proj-off", enabled=False)
    env.write_registry()
    _fail_if_run(monkeypatch)

    assert cli.main(_argv(env, "--repo", "off")) == 2

    err = capsys.readouterr().err
    assert "off" in err
    assert "disabled" in err


def test_repo_in_registry_disabled_list_exits_2(env, monkeypatch, capsys):
    env.add_repo("a", "prod", "a", "proj-a")
    env.write_registry(disabled=["a"])
    _fail_if_run(monkeypatch)

    assert cli.main(_argv(env, "--repo", "a")) == 2

    assert "disabled" in capsys.readouterr().err


def test_malformed_registry_exits_2(env, monkeypatch, capsys):
    env.registry_path.write_text("{bad", encoding="utf-8")
    _fail_if_run(monkeypatch)

    assert cli.main(_argv(env, "--repo", "a")) == 2

    assert "cannot read registry" in capsys.readouterr().err


def test_missing_registry_means_unknown_repo(env, monkeypatch, capsys):
    assert not env.registry_path.exists()
    _fail_if_run(monkeypatch)

    assert cli.main(_argv(env, "--repo", "a")) == 2

    err = capsys.readouterr().err
    assert "a (unknown)" in err


def test_repo_flag_sets_only_repos(env, monkeypatch):
    env.add_repo("a", "prod", "a", "proj-a")
    env.add_repo("b", "prod", "b", "proj-b")
    env.write_registry()
    seen = _capture_run(monkeypatch)

    cli.main(_argv(env, "--repo", "a", "--repo", "b"))

    assert seen[0].only_repos == frozenset({"a", "b"})


def test_no_repo_flag_means_all(env, monkeypatch):
    env.add_repo("a", "prod", "a", "proj-a")
    env.write_registry()
    seen = _capture_run(monkeypatch)

    cli.main(_argv(env))

    assert seen[0].only_repos == frozenset()


@pytest.fixture
def two_repos(env, monkeypatch):
    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.89")
    root_a = env.add_repo("a", "prod", "a", "proj-a", "repo_a.json")
    root_b = env.add_repo("b", "prod", "b", "proj-b", "repo_b.json")
    env.write_registry()
    return env, root_a, root_b


@pytest.fixture
def apply_calls(monkeypatch):
    """Every apply_action call the pipeline makes, as (repo_id, action, force)."""
    calls: list[tuple[str, str, bool]] = []
    real = pipeline.apply_action

    def recording(*args, **kwargs):
        calls.append((args[0], args[4], kwargs["force_rebuild"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "apply_action", recording)
    return calls


@pytest.fixture
def merge_inputs(monkeypatch):
    """The graph.json paths handed to each merge, one list per run."""
    seen: list[list[Path]] = []
    real = graphify_cli.run_merge_graphs

    def recording(graphify_bin, paths, *args, **kwargs):
        seen.append(list(paths))
        return real(graphify_bin, paths, *args, **kwargs)

    monkeypatch.setattr(graphify_cli, "run_merge_graphs", recording)
    return seen


def _pipeline_settings(env, **overrides):
    return env.settings(shrink_policy="count", **overrides)


def _rows(report) -> dict[str, dict]:
    return {a["repo_id"]: a for a in report.project_actions}


def _state_entry(env, repo_id: str) -> str:
    entry = load_state(_pipeline_settings(env).state_path)[repo_id]
    return json.dumps(entry, sort_keys=True)


def _edit(root: Path) -> None:
    with (root / "app.php").open("a", encoding="utf-8") as fh:
        fh.write("<?php // edit\n")


def _graph_b(env) -> Path:
    return env.collection_path("prod", "b") / "graph.json"


def test_repo_flag_refreshes_only_selected(two_repos, apply_calls, merge_inputs):
    env, root_a, root_b = two_repos
    run(_pipeline_settings(env))
    b_before = _state_entry(env, "b")
    _edit(root_a)
    _edit(root_b)
    applied = len(apply_calls)

    report = run(_pipeline_settings(env, only_repos=frozenset({"a"})))

    assert [c[0] for c in apply_calls[applied:]] == ["a"]
    assert _state_entry(env, "b") == b_before
    assert report.pending_repos == ["b"]
    assert _rows(report)["b"] == {"repo_id": "b", "action": ACTION_UPDATE, "status": "deferred"}
    assert _graph_b(env) in merge_inputs[-1]
    assert report.published
    status = json.loads(_pipeline_settings(env).status_path.read_text(encoding="utf-8"))
    assert status["pending_repos"] == ["b"]


def test_next_full_run_picks_up_deferred(two_repos, apply_calls):
    env, root_a, root_b = two_repos
    run(_pipeline_settings(env))
    _edit(root_a)
    _edit(root_b)
    run(_pipeline_settings(env, only_repos=frozenset({"a"})))
    applied = len(apply_calls)

    report = run(_pipeline_settings(env))

    assert [(c[0], c[1]) for c in apply_calls[applied:]] == [("b", ACTION_UPDATE)]
    assert _rows(report)["b"]["status"] == "updated"
    assert report.pending_repos == []


def test_selected_unchanged_repo_skips(two_repos, apply_calls):
    env, _root_a, _root_b = two_repos
    run(_pipeline_settings(env))
    applied = len(apply_calls)

    report = run(_pipeline_settings(env, only_repos=frozenset({"a"})))

    assert _rows(report)["a"]["action"] == ACTION_SKIP
    assert _rows(report)["b"]["action"] == ACTION_SKIP
    assert report.pending_repos == []
    assert len(apply_calls) == applied


def test_unselected_ignore_change_is_deferred(two_repos, apply_calls):
    env, _root_a, root_b = two_repos
    run(_pipeline_settings(env))
    b_before = _state_entry(env, "b")
    ignore_hash = json.loads(b_before)[IGNORE_RULES_KEY]
    assert ignore_hash, "precondition: accepted state records the rules"
    with (root_b / ".graphifyignore").open("a", encoding="utf-8") as fh:
        fh.write("legacy/\n")
    applied = len(apply_calls)

    report = run(_pipeline_settings(env, only_repos=frozenset({"a"})))

    assert all(c[0] != "b" for c in apply_calls[applied:])
    assert _state_entry(env, "b") == b_before
    assert report.pending_repos == ["b"]
    assert _rows(report)["b"]["status"] == "deferred"


NEEDS_FULL_SYNC = "--repo: unselected repo(s) need a full sync: "


def test_unselected_drift_blocks_publish(env, monkeypatch, apply_calls):
    """One drifted repo in four stays under the 30% stale ratio; the --repo
    block must refuse publish on its own."""
    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.89")
    env.add_repo("a", "prod", "a", "proj-a", "repo_a.json")
    env.add_repo("b", "prod", "b", "proj-b", "repo_b.json")
    env.add_repo("c", "prod", "c", "proj-c", "repo_a.json")
    env.add_repo("d", "prod", "d", "proj-d", "repo_b.json")
    env.write_registry()
    assert run(_pipeline_settings(env)).published, "precondition: first run publishes"
    b_before = _state_entry(env, "b")
    assert json.loads(b_before)[GRAPH_HASH_KEY], "precondition: accepted hash recorded"
    data = json.loads(_graph_b(env).read_text())
    data["nodes"].append({**data["nodes"][-1], "id": "b-extra", "label": "beta_extra"})
    _graph_b(env).write_text(json.dumps(data), encoding="utf-8")
    applied = len(apply_calls)

    report = run(_pipeline_settings(env, only_repos=frozenset({"a"})))

    assert all(c[0] != "b" for c in apply_calls[applied:])
    assert _state_entry(env, "b") == b_before
    assert _rows(report)["b"]["status"] == "graph_drift_held"
    assert report.publish_blocking_repos == ["b"]
    assert report.pending_repos == []
    assert not report.published
    assert report.publish_blocked_reason == NEEDS_FULL_SYNC + "b"


def _delete_accepted_graph_b(env, apply_calls, merge_inputs, **settings):
    run(_pipeline_settings(env))
    b_before = _state_entry(env, "b")
    assert json.loads(b_before)[GRAPH_HASH_KEY], "precondition: accepted hash recorded"
    _graph_b(env).unlink()
    applied = len(apply_calls)

    report = run(_pipeline_settings(env, only_repos=frozenset({"a"}), **settings))

    assert _rows(report)["b"] == {
        "repo_id": "b",
        "action": ACTION_BOOTSTRAP,
        "status": "graph_missing_held",
        "reason": "graph.json missing; bootstrap deferred",
    }
    assert report.pending_repos == []
    assert all(c[0] != "b" for c in apply_calls[applied:])
    assert _graph_b(env) not in merge_inputs[-1]
    assert _state_entry(env, "b") == b_before
    assert not report.published
    assert report.publish_blocked_reason == NEEDS_FULL_SYNC + "b"


def test_unselected_deleted_graph_blocks_publish(two_repos, apply_calls, merge_inputs):
    env, _root_a, _root_b = two_repos
    _delete_accepted_graph_b(env, apply_calls, merge_inputs)


def test_unselected_deleted_graph_blocks_publish_with_allow_shrink(
    two_repos, apply_calls, merge_inputs
):
    env, _root_a, _root_b = two_repos
    _delete_accepted_graph_b(env, apply_calls, merge_inputs, allow_shrink=True)


def test_selected_repo_not_active_blocks_publish(two_repos, apply_calls):
    """The registry can change between the CLI check and the lock."""
    env, _root_a, _root_b = two_repos

    report = run(_pipeline_settings(env, only_repos=frozenset({"gone"})))

    assert not report.published
    assert report.publish_blocked_reason == "--repo: selected repo(s) not active: gone"
    assert report.project_actions == []
    assert apply_calls == []


def test_unselected_bootstrap_is_deferred(env, monkeypatch, apply_calls, merge_inputs):
    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.89")
    env.add_repo("a", "prod", "a", "proj-a", "repo_a.json")
    env.add_repo("b", "prod", "b", "proj-b", graph_fixture=None)
    env.write_registry()

    report = run(_pipeline_settings(env, only_repos=frozenset({"a"})))

    assert _rows(report)["b"] == {
        "repo_id": "b",
        "action": ACTION_BOOTSTRAP,
        "status": "deferred",
    }
    assert report.pending_repos == ["b"]
    assert all(c[0] != "b" for c in apply_calls)
    assert not _graph_b(env).exists()
    assert merge_inputs[-1] == [env.collection_path("prod", "a") / "graph.json"]
    assert report.published


def test_selected_broken_repo_reported(two_repos, apply_calls):
    env, root_a, root_b = two_repos
    run(_pipeline_settings(env))
    link = root_a / "graphify-out"
    link.unlink()
    link.symlink_to(env.collection_path("prod", "ghost"), target_is_directory=True)
    _edit(root_b)
    applied = len(apply_calls)

    report = run(_pipeline_settings(env, only_repos=frozenset({"a"})))

    assert _rows(report)["a"]["status"] == "broken_symlink"
    assert _rows(report)["b"]["status"] == "deferred"
    assert report.pending_repos == ["b"]
    assert apply_calls[applied:] == []
    assert report.published


def test_dry_run_with_repo_flag(two_repos, apply_calls):
    env, root_a, root_b = two_repos
    run(_pipeline_settings(env))
    _edit(root_a)
    _edit(root_b)
    applied = len(apply_calls)

    report = run(_pipeline_settings(env, dry_run=True, only_repos=frozenset({"a"})))

    assert _rows(report)["a"]["status"] == "would_update"
    assert _rows(report)["b"]["status"] == "deferred"
    assert report.pending_repos == ["b"]
    assert len(apply_calls) == applied


def test_embed_reuse_candidates_include_deferred():
    rows = [
        {"repo_id": "a", "status": "updated"},
        {"repo_id": "b", "status": pipeline.STATUS_DEFERRED},
        {"repo_id": "c", "status": "unchanged"},
        {"repo_id": "d", "status": "graph_drift_held"},
    ]

    assert pipeline._embed_reuse_candidates(rows) == {"b", "c"}


def test_deferred_repo_reaches_embed_reuse(two_repos, monkeypatch):
    env, root_a, root_b = two_repos
    run(_pipeline_settings(env))
    _edit(root_a)
    _edit(root_b)
    seen: list[set[str]] = []
    real = embedding.stale_recipe_repo_ids

    def recording(current_dir, repo_ids, model):
        seen.append(set(repo_ids))
        return real(current_dir, repo_ids, model)

    monkeypatch.setattr(embedding, "stale_recipe_repo_ids", recording)

    run(_pipeline_settings(env, only_repos=frozenset({"a"})))

    assert seen == [{"b"}]


def test_report_carries_overlay_rows_and_limit(two_repos):
    env, _, _ = two_repos
    report = run(_pipeline_settings(env, exact_max_rows=1234))

    assert report.overlay_exact_max_rows == 1234
    assert isinstance(report.overlay_rows, int)
    status = json.loads(_pipeline_settings(env).status_path.read_text(encoding="utf-8"))
    assert status["overlay_exact_max_rows"] == 1234
    assert status["overlay_rows"] == report.overlay_rows
