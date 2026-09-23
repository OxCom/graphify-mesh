"""`allow_shrink_once` must name exactly one reviewed attempt and be spent durably.

Review of e3997cc (2026-09-23) found two gaps in the per-repo shrink grant:

1. The token is the source's `semantic_hash` alone. A code-only change keeps that
   hash, so a token approved for one refused attempt also accepts a different,
   unreviewed code shrink, and two different code-only refusals print the same
   token.
2. The spent marker lives only in the in-memory state until `_finalize`, while the
   smaller per-repo graph.json is already on disk. A later stage that raises
   leaves the token unspent, so it authorizes a second shrink on the next run.

The token is read back from the refusal reason, the operator-facing contract,
not from a state key.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from graphify_mesh.sync import graphify_cli
from graphify_mesh.sync.pipeline import run

REPO_ID = "example-org.styleguide"
TOKEN_RE = re.compile(r'"allow_shrink_once": "([0-9a-f]+)"')


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(report) -> dict:
    return {a["repo_id"]: a for a in report.project_actions}[REPO_ID]


def _token(report) -> str:
    row = _row(report)
    assert row["status"] == "shrink_refused", row
    match = TOKEN_RE.search(row["reason"])
    assert match, row["reason"]
    return match.group(1)


def _grant(env, token: str) -> None:
    payload = _read_json(env.registry_path)
    for entry in payload["repos"]:
        if entry["repo_id"] == REPO_ID:
            entry["allow_shrink_once"] = token
    env.registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.fixture
def repo(env):
    root = env.add_repo(
        REPO_ID, "example-org", "styleguide", "styleguide.example-org.dev.lo", "repo_a.json"
    )
    env.write_registry()
    return env, root, env.collection_path("example-org", "styleguide")


@pytest.fixture
def accepted_baseline(repo):
    """One accepted run with a code file present, then the extractor starts
    shrinking: every later code edit is a code-only `update` attempt."""
    env, root, collection = repo
    (root / "app.py").write_text("v1\n", encoding="utf-8")
    assert _row(run(env.settings()))["status"] == "updated"
    env.set_control(collection, "shrink")
    return env, root, collection


class TestGrantIsBoundToTheWholeSourceState:
    def test_code_only_refusals_print_distinct_tokens(self, accepted_baseline):
        env, root, _ = accepted_baseline
        (root / "app.py").write_text("v2\n", encoding="utf-8")
        first = _token(run(env.settings()))
        (root / "app.py").write_text("v3 -- a different change\n", encoding="utf-8")
        second = _token(run(env.settings()))
        assert first != second

    def test_grant_does_not_cover_a_code_change_made_after_approval(self, accepted_baseline):
        env, root, collection = accepted_baseline
        (root / "app.py").write_text("v2\n", encoding="utf-8")
        token = _token(run(env.settings()))
        _grant(env, token)

        # The operator reviewed attempt v2; the source moves on before the next run.
        (root / "app.py").write_text("v3 -- not what was reviewed\n", encoding="utf-8")
        before = _read_json(collection / "graph.json")
        report = run(env.settings())

        assert _row(report)["status"] == "shrink_refused"
        assert _read_json(collection / "graph.json") == before
        state = _read_json(env.settings().state_path)
        assert "consumed_shrink_grant" not in state[REPO_ID]

    def test_grant_still_accepts_the_reviewed_code_only_attempt(self, accepted_baseline):
        env, root, collection = accepted_baseline
        (root / "app.py").write_text("v2\n", encoding="utf-8")
        before = _read_json(collection / "graph.json")
        token = _token(run(env.settings()))
        _grant(env, token)

        report = run(env.settings())

        assert _row(report)["status"] == "updated"
        assert len(_read_json(collection / "graph.json")["nodes"]) < len(before["nodes"])
        assert _read_json(env.settings().state_path)[REPO_ID]["consumed_shrink_grant"] == token


class TestSpentGrantSurvivesALaterStageFailure:
    def test_grant_is_spent_when_merge_raises_after_acceptance(self, repo, monkeypatch):
        env, _, collection = repo
        env.set_control(collection, "shrink")
        token = _token(run(env.settings()))
        _grant(env, token)
        original_nodes = len(_read_json(collection / "graph.json")["nodes"])

        real_merge = graphify_cli.run_merge_graphs

        def exploding_merge(*args, **kwargs):
            raise RuntimeError("simulated downstream failure")

        monkeypatch.setattr(graphify_cli, "run_merge_graphs", exploding_merge)
        with pytest.raises(RuntimeError, match="simulated downstream failure"):
            run(env.settings())
        # Restored by hand: monkeypatch.undo() would also drop the fake-graphify env.
        monkeypatch.setattr(graphify_cli, "run_merge_graphs", real_merge)

        # The acceptance is durable (the smaller graph is on disk), so the spend must be too.
        assert len(_read_json(collection / "graph.json")["nodes"]) < original_nodes
        state = _read_json(env.settings().state_path)
        assert state.get(REPO_ID, {}).get("consumed_shrink_grant") == token

        # State now records the accepted shrink, so the next run sees an unchanged
        # source: no second acceptance, the graph stays put, the token stays spent.
        shrunk = _read_json(collection / "graph.json")
        report = run(env.settings())
        assert _row(report)["status"] == "unchanged"
        assert _read_json(collection / "graph.json") == shrunk
        assert _read_json(env.settings().state_path)[REPO_ID]["consumed_shrink_grant"] == token
