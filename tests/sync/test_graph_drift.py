"""A graph.json rewritten outside the sync is rebuilt, never published as-is.

decide_action used to return skip on an unchanged source digest, and the SKIP
path merged whatever graph.json was on disk. A file written by another program
between runs (a Stop hook building without PHP stubs, a hand edit) reached the
published generation without passing any shrink check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphify_mesh.sync.pipeline import run
from graphify_mesh.sync.state import SourceDigest, file_content_hash
from graphify_mesh.sync.sync_project import (
    ACTION_SKIP,
    ACTION_UPDATE,
    GRAPH_HASH_KEY,
    STATUS_SHRINK_REFUSED,
    STATUS_UPDATED,
    ProjectOutcome,
    decide_action,
)

REPO_ID = "example-org.drift"


@pytest.fixture
def repo(env):
    root = env.add_repo(REPO_ID, "example-org", "drift", "drift.example-org.dev.lo", "repo_a.json")
    env.write_registry()
    return env, root


def _recording_apply(calls: list, status: str = STATUS_UPDATED):
    def fake(repo_id, graphify_bin, root, collection_path, action, current_manifest, *a, **kw):
        calls.append({"action": action, **kw})
        if status == STATUS_SHRINK_REFUSED:
            return ProjectOutcome(repo_id, action, status, refused_manifest=current_manifest)
        return ProjectOutcome(repo_id, action, status, new_manifest=current_manifest)

    return fake


def _collection_graph(env) -> Path:
    return env.collection_path("example-org", "drift") / "graph.json"


def _state(env) -> dict:
    return json.loads(env.settings().state_path.read_text())[REPO_ID]


def _rewrite_out_of_band(graph_path) -> None:
    data = json.loads(graph_path.read_text())
    data["nodes"] = data["nodes"][:-1]
    graph_path.write_text(json.dumps(data), encoding="utf-8")


def _digest() -> SourceDigest:
    return SourceDigest(code_hash="c", semantic_hash="s", file_count=1)


def test_decide_action_rebuilds_a_drifted_graph():
    prior = {"code_hash": "c", "semantic_hash": "s"}
    assert decide_action(prior, _digest(), True) == ACTION_SKIP
    assert decide_action(prior, _digest(), True, graph_drifted=True) == ACTION_UPDATE


def test_accepted_outcome_records_the_graph_hash(repo, monkeypatch):
    env, _ = repo
    monkeypatch.setattr("graphify_mesh.sync.pipeline.apply_action", _recording_apply([]))
    run(env.settings(shrink_policy="count"))
    assert _state(env)[GRAPH_HASH_KEY] == file_content_hash(_collection_graph(env))


def test_unchanged_graph_still_skips(repo, monkeypatch):
    env, _ = repo
    calls: list = []
    monkeypatch.setattr("graphify_mesh.sync.pipeline.apply_action", _recording_apply(calls))
    run(env.settings(shrink_policy="count"))
    run(env.settings(shrink_policy="count"))
    assert len(calls) == 1


def test_out_of_band_rewrite_forces_a_rebuild(repo, monkeypatch):
    env, _ = repo
    calls: list = []
    monkeypatch.setattr("graphify_mesh.sync.pipeline.apply_action", _recording_apply(calls))
    run(env.settings(shrink_policy="count"))
    _rewrite_out_of_band(_collection_graph(env))

    report = run(env.settings(shrink_policy="count"))
    assert len(calls) == 2
    assert calls[-1]["action"] == ACTION_UPDATE
    # An incremental update keeps the rewritten file's extraction of unchanged files.
    assert calls[-1]["force_rebuild"] is True
    row = {a["repo_id"]: a for a in report.project_actions}[REPO_ID]
    assert row["action"] == ACTION_UPDATE


def test_state_without_a_hash_is_seeded_on_skip(repo, monkeypatch):
    env, _ = repo
    calls: list = []
    monkeypatch.setattr("graphify_mesh.sync.pipeline.apply_action", _recording_apply(calls))
    run(env.settings(shrink_policy="count"))
    state_path = env.settings().state_path
    payload = json.loads(state_path.read_text())
    del payload[REPO_ID][GRAPH_HASH_KEY]
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    run(env.settings(shrink_policy="count"))
    assert len(calls) == 1, "a state written before this check must not rebuild every repo"
    assert _state(env)[GRAPH_HASH_KEY] == file_content_hash(_collection_graph(env))


def test_held_drift_is_reported_and_blocks_publish(repo, monkeypatch):
    env, _ = repo
    monkeypatch.setattr("graphify_mesh.sync.pipeline.apply_action", _recording_apply([]))
    run(env.settings(ast_only=True, shrink_policy="attributed"))
    accepted_hash = _state(env)[GRAPH_HASH_KEY]
    _rewrite_out_of_band(_collection_graph(env))

    calls: list = []
    monkeypatch.setattr(
        "graphify_mesh.sync.pipeline.apply_action",
        _recording_apply(calls, status=STATUS_SHRINK_REFUSED),
    )
    run(env.settings(ast_only=True, shrink_policy="attributed"))
    report = run(env.settings(ast_only=True, shrink_policy="attributed"))
    assert len(calls) == 1, "precondition: the refused rebuild is held"
    assert _state(env)[GRAPH_HASH_KEY] == accepted_hash, "a held drift must not be adopted"
    assert REPO_ID in report.publish_blocking_repos
    assert REPO_ID in report.publish_blocked_reason
