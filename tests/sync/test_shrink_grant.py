"""Per-repo, single-use shrink authorization (`allow_shrink_once`).

`--allow-shrink` disarms the guard for every repo in the run: on 2026-09-22 four
repos were refused at once and approving the single intended deletion (cem.k8s,
200 -> 168 nodes) would have accepted the other three too. The registry key names
one repo and one refused attempt id, so approving one shrink cannot blind the
guard anywhere else, and a token is spent once.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from graphify_mesh.sync.pipeline import _shrink_grant_for, run
from graphify_mesh.sync.registry import RepoEntry, load_registry
from graphify_mesh.sync.state import SourceDigest, compute_source_manifest
from graphify_mesh.sync.sync_project import (
    ACTION_EXTRACT,
    ACTION_SKIP,
    REFUSAL_RETRY_LIMIT,
    decide_action,
)

REPO_ID = "example-org.styleguide"
TOKEN_RE = re.compile(r'"allow_shrink_once": "([0-9a-f]+)"')


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _grant_repo(env, *, token: str | None) -> None:
    """Rewrite the registry with `allow_shrink_once` set on the one repo."""
    payload = _read_json(env.registry_path)
    for entry in payload["repos"]:
        if entry["repo_id"] == REPO_ID:
            if token is None:
                entry.pop("allow_shrink_once", None)
            else:
                entry["allow_shrink_once"] = token
    env.registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _refused_token(report) -> str:
    """The attempt id the refusal reason tells the operator to copy."""
    match = TOKEN_RE.search(_row(report)["reason"])
    assert match, _row(report)["reason"]
    return match.group(1)


@pytest.fixture
def shrinking_repo(env):
    env.add_repo(
        REPO_ID,
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.write_registry()
    env.set_control(env.collection_path("example-org", "styleguide"), "shrink")
    return env


def _status(report) -> str:
    return {a["repo_id"]: a["status"] for a in report.project_actions}[REPO_ID]


def _row(report) -> dict:
    return {a["repo_id"]: a for a in report.project_actions}[REPO_ID]


class TestRegistryParsing:
    """An unknown or absent key means "guard armed"; garbage is a hard error."""

    def _write(self, tmp_path: Path, entry_extra: dict) -> Path:
        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps(
                {
                    "repos": [
                        {
                            "repo_id": "a.b",
                            "root": "/srv/a",
                            "collection_path": "/srv/out/a",
                            **entry_extra,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_absent_key_is_armed(self, tmp_path):
        reg = load_registry(self._write(tmp_path, {}))
        assert reg.repos[0].allow_shrink_once is None

    @pytest.mark.parametrize("value", [None, ""])
    def test_null_and_empty_are_armed(self, tmp_path, value):
        reg = load_registry(self._write(tmp_path, {"allow_shrink_once": value}))
        assert reg.repos[0].allow_shrink_once is None

    def test_digest_token_is_kept(self, tmp_path):
        reg = load_registry(self._write(tmp_path, {"allow_shrink_once": "0123456789abcdef"}))
        assert reg.repos[0].allow_shrink_once == "0123456789abcdef"

    @pytest.mark.parametrize(
        "value",
        [True, 1, ["0123456789abcdef"], "yes", "0123456789ABCDEF", "abc", "0123456789abcdef "],
    )
    def test_malformed_value_is_rejected_at_load(self, tmp_path, value):
        with pytest.raises(ValueError, match="allow_shrink_once"):
            load_registry(self._write(tmp_path, {"allow_shrink_once": value}))


class TestGrantResolution:
    """Single use is tracked in per-repo state, not by rewriting the registry."""

    def _entry(self, token: str | None) -> RepoEntry:
        return RepoEntry(
            repo_id=REPO_ID,
            root=Path("/srv/a"),
            collection_path=Path("/srv/out/a"),
            allow_shrink_once=token,
        )

    def test_no_token_no_grant(self):
        assert _shrink_grant_for(self._entry(None), {"semantic_hash": "x"}) is None

    def test_unspent_token_is_granted(self):
        assert _shrink_grant_for(self._entry("deadbeefdeadbeef"), None) == "deadbeefdeadbeef"

    def test_spent_token_is_inert(self):
        state = {"consumed_shrink_grant": "deadbeefdeadbeef"}
        assert _shrink_grant_for(self._entry("deadbeefdeadbeef"), state) is None

    def test_new_token_after_a_spent_one_is_granted(self):
        state = {"consumed_shrink_grant": "deadbeefdeadbeef"}
        assert _shrink_grant_for(self._entry("feedfacefeedface"), state) == "feedfacefeedface"


class TestDecideActionHonoursTheGrant:
    """A repo held by the refusal-retry limit must still re-extract once approved."""

    def _prior(self, digest: str) -> dict:
        return {
            "semantic_hash": "old",
            "code_hash": "code",
            "refused_semantic_hash": digest,
            "refusal_streak": REFUSAL_RETRY_LIMIT,
        }

    def test_refused_digest_still_skips_without_a_grant(self):
        current = SourceDigest(code_hash="code", semantic_hash="d1", file_count=1)
        assert decide_action(self._prior("d1"), current, has_graph=True) == ACTION_SKIP

    def test_matching_grant_re_extracts(self):
        current = SourceDigest(code_hash="code", semantic_hash="d1", file_count=1)
        grant = current.attempt_id
        assert decide_action(self._prior("d1"), current, True, grant) == ACTION_EXTRACT

    def test_semantic_hash_alone_is_not_a_grant(self):
        current = SourceDigest(code_hash="code", semantic_hash="d1", file_count=1)
        assert decide_action(self._prior("d1"), current, True, "d1") == ACTION_SKIP

    def test_grant_for_another_digest_still_skips(self):
        current = SourceDigest(code_hash="code", semantic_hash="d1", file_count=1)
        assert decide_action(self._prior("d1"), current, True, "d2") == ACTION_SKIP


class TestAttemptId:
    def test_differs_when_only_code_hash_differs(self):
        a = SourceDigest(code_hash="c1", semantic_hash="s", file_count=1)
        b = SourceDigest(code_hash="c2", semantic_hash="s", file_count=1)
        assert a.attempt_id != b.attempt_id

    def test_empty_sentinel_is_sixteen_hex_chars(self):
        empty = SourceDigest(code_hash="empty", semantic_hash="empty", file_count=0)
        assert re.fullmatch(r"[0-9a-f]{16}", empty.attempt_id)

    def test_not_serialized(self):
        digest = SourceDigest(code_hash="c", semantic_hash="s", file_count=1)
        assert "attempt_id" not in digest.to_dict()


class TestEndToEnd:
    def test_refusal_names_the_digest_to_authorize(self, shrinking_repo):
        env = shrinking_repo
        report = run(env.settings())
        assert _status(report) == "shrink_refused"
        root = env.scan_root / "styleguide.example-org.dev.lo"
        digest = compute_source_manifest(root).attempt_id
        assert f'"allow_shrink_once": "{digest}"' in _row(report)["reason"]

    def test_matching_grant_accepts_the_shrink_once(self, shrinking_repo):
        env = shrinking_repo
        collection = env.collection_path("example-org", "styleguide")
        original = _read_json(collection / "graph.json")

        first = run(env.settings())
        assert _status(first) == "shrink_refused"
        assert _read_json(collection / "graph.json") == original

        digest = _refused_token(first)
        _grant_repo(env, token=digest)
        second = run(env.settings())

        assert _status(second) == "updated"
        assert "allow_shrink_once" in _row(second)["reason"]
        kept = _read_json(collection / "graph.json")
        assert len(kept["nodes"]) < len(original["nodes"])

        state = _read_json(env.settings().state_path)
        assert state[REPO_ID]["consumed_shrink_grant"] == digest
        accepted = SourceDigest(
            code_hash=state[REPO_ID]["code_hash"],
            semantic_hash=state[REPO_ID]["semantic_hash"],
            file_count=state[REPO_ID]["file_count"],
        )
        assert accepted.attempt_id == digest

    def test_spent_grant_does_not_authorize_a_second_shrink(self, shrinking_repo):
        env = shrinking_repo
        digest = _refused_token(run(env.settings()))
        _grant_repo(env, token=digest)
        assert _status(run(env.settings())) == "updated"

        # Source moves on, the repo shrinks again: the spent token is inert and
        # the new attempt carries a different digest, so the guard refuses.
        root = env.scan_root / "styleguide.example-org.dev.lo"
        (root / "README.md").write_text("changed", encoding="utf-8")
        third = run(env.settings())
        assert _status(third) == "shrink_refused"
        assert _refused_token(third) != digest

    def test_grant_for_another_digest_does_not_accept(self, shrinking_repo):
        env = shrinking_repo
        assert _status(run(env.settings())) == "shrink_refused"
        _grant_repo(env, token="0123456789abcdef")
        second = run(env.settings())
        # No re-extract (the refusal-retry hold stands) and no acceptance.
        assert _status(second) in {"unchanged", "shrink_refused"}
        state = _read_json(env.settings().state_path)
        assert "consumed_shrink_grant" not in state[REPO_ID]

    def test_grant_on_one_repo_leaves_the_other_guarded(self, shrinking_repo):
        env = shrinking_repo
        env.add_repo(
            "example-org.gamma",
            "example-org",
            "gamma",
            "gamma.example-org.dev.lo",
            "repo_b.json",
        )
        env.write_registry()
        env.set_control(env.collection_path("example-org", "gamma"), "shrink")

        first = run(env.settings())
        statuses = {a["repo_id"]: a["status"] for a in first.project_actions}
        assert statuses[REPO_ID] == "shrink_refused"
        assert statuses["example-org.gamma"] == "shrink_refused"

        _grant_repo(env, token=_refused_token(first))
        second = run(env.settings())
        statuses = {a["repo_id"]: a["status"] for a in second.project_actions}
        assert statuses[REPO_ID] == "updated"
        assert statuses["example-org.gamma"] != "updated"

    def test_allow_shrink_flag_still_accepts_without_any_grant(self, shrinking_repo):
        env = shrinking_repo
        report = run(env.settings(allow_shrink=True))
        assert _status(report) == "updated"
        assert "operator-authorized" in _row(report)["reason"]
        state = _read_json(env.settings().state_path)
        assert "consumed_shrink_grant" not in state[REPO_ID]
