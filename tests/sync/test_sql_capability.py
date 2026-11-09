"""SQL extractor-capability stamp in producer_versions.

graphify parses .sql only when tree_sitter_sql is importable (the `sql` extra).
Installing or removing it changes no source digest, so the stamp is what makes
a repo with .sql files rebuild once, with `update --force`, as a graphify-php
upgrade does. Repos without .sql files must not notice it.
"""

from __future__ import annotations

import pytest

from graphify_mesh.sync import pipeline, repo_tags
from graphify_mesh.sync.pipeline import (
    PRODUCER_VERSIONS_KEY,
    SQL_ABSENT,
    SQL_CAPABILITY_KEY,
    SQL_PRESENT,
    _key_version,
    _producer_upgraded,
    _repo_versions,
    run,
)
from graphify_mesh.sync.registry import BUILDER_GRAPHIFY, BUILDER_GRAPHIFY_PHP
from graphify_mesh.sync.state import CONTENT_DIGEST_ENV, compute_source_manifest, load_state
from graphify_mesh.sync.sync_project import ACTION_SKIP, ACTION_UPDATE

ACCEPTED = {"semantic_hash": "s", "graphify_version": "0.9.103"}


def _caps(value: str) -> dict:
    return {SQL_CAPABILITY_KEY: value}


def _upgraded(prior: dict, capabilities: dict | None) -> bool:
    versions = _repo_versions(BUILDER_GRAPHIFY, "0.9.103", None, prior, capabilities)
    return _producer_upgraded(prior, versions.graphify, versions.producer_versions)


class TestStamp:
    def test_install_on_state_from_before_the_stamp_rebuilds(self):
        assert _upgraded(dict(ACCEPTED), _caps(SQL_PRESENT))

    def test_absent_on_state_from_before_the_stamp_is_unchanged(self):
        assert not _upgraded(dict(ACCEPTED), _caps(SQL_ABSENT))

    @pytest.mark.parametrize(
        ("recorded", "now", "expected"),
        [
            (SQL_ABSENT, SQL_PRESENT, True),
            (SQL_PRESENT, SQL_ABSENT, True),
            (SQL_PRESENT, SQL_PRESENT, False),
            (SQL_ABSENT, SQL_ABSENT, False),
        ],
    )
    def test_both_directions(self, recorded, now, expected):
        prior = {**ACCEPTED, PRODUCER_VERSIONS_KEY: _caps(recorded)}
        assert _upgraded(prior, _caps(now)) is expected

    def test_repo_without_sql_is_unaffected(self):
        prior = {**ACCEPTED, PRODUCER_VERSIONS_KEY: _caps(SQL_PRESENT)}
        assert not _upgraded(prior, None)

    def test_php_repos_carry_no_mesh_capability(self):
        """graphify-php builds in another interpreter; this one's grammar says nothing."""
        versions = _repo_versions(
            BUILDER_GRAPHIFY_PHP,
            "0.9.103",
            {"graphify_php": "1.0", "graphify": "0.9.103"},
            {},
            _caps(SQL_PRESENT),
        )
        assert SQL_CAPABILITY_KEY not in (versions.producer_versions or {})

    def test_refusal_key_changes_only_when_present(self):
        assert _key_version("0.9.103", _caps(SQL_ABSENT)) == "0.9.103"
        assert _key_version("0.9.103", None) == "0.9.103"
        assert _key_version("0.9.103", _caps(SQL_PRESENT)) == "0.9.103+sql"

    def test_digest_records_sql_files(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONTENT_DIGEST_ENV, raising=False)
        (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
        assert not compute_source_manifest(tmp_path).has_sql
        (tmp_path / "migrations").mkdir()
        (tmp_path / "migrations" / "0001.SQL").write_text("CREATE TABLE t (id int);\n")
        assert compute_source_manifest(tmp_path).has_sql


SQL_REPO = "example-org.db"
PLAIN_REPO = "example-org.svc"


@pytest.fixture
def repos(env, monkeypatch):
    monkeypatch.delenv(CONTENT_DIGEST_ENV, raising=False)
    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.103")
    sql_root = env.add_repo(SQL_REPO, "example-org", "db", "db.example-org.dev.lo")
    plain_root = env.add_repo(PLAIN_REPO, "example-org", "svc", "svc.example-org.dev.lo")
    env.write_registry()
    (sql_root / "app.py").write_text("x = 1\n", encoding="utf-8")
    (sql_root / "schema.sql").write_text("CREATE TABLE t (id int);\n", encoding="utf-8")
    (plain_root / "app.py").write_text("y = 2\n", encoding="utf-8")
    return env


def _run(env, monkeypatch, capability: str):
    monkeypatch.setattr(pipeline, "_sql_capability", lambda: capability)
    before = len(env.read_argv_log())
    report = run(env.settings(shrink_policy="count"))
    updates = [r for r in env.read_argv_log()[before:] if r["cmd"] == "update"]
    rows = {a["repo_id"]: a["action"] for a in report.project_actions}
    return rows, updates


def test_capability_change_forces_one_update_for_sql_repos_only(repos, monkeypatch):
    env = repos
    _run(env, monkeypatch, SQL_ABSENT)
    state = load_state(env.settings().state_path)
    assert state[SQL_REPO][PRODUCER_VERSIONS_KEY] == _caps(SQL_ABSENT)
    assert PRODUCER_VERSIONS_KEY not in state[PLAIN_REPO]

    rows, updates = _run(env, monkeypatch, SQL_PRESENT)
    assert rows == {SQL_REPO: ACTION_UPDATE, PLAIN_REPO: ACTION_SKIP}
    assert len(updates) == 1 and "--force" in updates[0]["argv"]
    state = load_state(env.settings().state_path)
    assert state[SQL_REPO][PRODUCER_VERSIONS_KEY] == _caps(SQL_PRESENT)

    rows, updates = _run(env, monkeypatch, SQL_PRESENT)
    assert rows == {SQL_REPO: ACTION_SKIP, PLAIN_REPO: ACTION_SKIP}
    assert not updates

    rows, updates = _run(env, monkeypatch, SQL_ABSENT)
    assert rows == {SQL_REPO: ACTION_UPDATE, PLAIN_REPO: ACTION_SKIP}
    assert len(updates) == 1 and "--force" in updates[0]["argv"]
