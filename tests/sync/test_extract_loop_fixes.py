"""Three coupled fixes for the four-week publish deadlock (2026-07-21..08-17).

Root cause chain that was observed in production:

1. The manifest digest keyed on `(size, mtime_ns)`, so a checkout or an
   editor save marked a repo dirty with byte-identical content.
2. That scheduled `graphify extract` — a non-deterministic LLM pass — whose
   output legitimately differs run to run.
3. A shrink refusal did not advance per-repo state, so the accepted digest
   stayed stale, `decide_action` saw a difference forever, and the repo
   re-extracted and was refused on every hourly run.
4. Every such repo counted toward one 30% stale gate, so a handful of
   permanently-looping repos blocked publishing for the whole fleet — which
   also froze the embedding channel, since embeddings persist only on publish.
"""

from __future__ import annotations

import os
import time

import pytest

from graphify_mesh.sync.pipeline import (
    PUBLISH_BLOCKING_STATUSES,
    UNREFRESHED_PUBLISH_THRESHOLD,
    UNREFRESHED_STATUSES,
    _unrefreshed_threshold,
)
from graphify_mesh.sync.state import CONTENT_DIGEST_ENV, compute_source_manifest
from graphify_mesh.sync.sync_project import (
    ACTION_EXTRACT,
    ACTION_SKIP,
    ACTION_UPDATE,
    REFUSAL_RETRY_LIMIT,
    STATUS_BOOTSTRAP_FAILED,
    STATUS_FAILED,
    STATUS_SHRINK_REFUSED,
    decide_action,
)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.delenv(CONTENT_DIGEST_ENV, raising=False)
    (tmp_path / "app.php").write_text("<?php echo 1;", encoding="utf-8")
    (tmp_path / "README.md").write_text("docs", encoding="utf-8")
    return tmp_path


class TestContentDigest:
    """Step 1: the digest must track content, not mtime."""

    def test_touch_without_edit_does_not_change_digest(self, repo):
        before = compute_source_manifest(repo)
        future = time.time() + 10_000
        os.utime(repo / "README.md", (future, future))
        after = compute_source_manifest(repo)
        assert after.semantic_hash == before.semantic_hash
        assert after.code_hash == before.code_hash

    def test_rewriting_identical_bytes_does_not_change_digest(self, repo):
        before = compute_source_manifest(repo)
        (repo / "README.md").write_text("docs", encoding="utf-8")  # same bytes, new mtime
        assert compute_source_manifest(repo).semantic_hash == before.semantic_hash

    def test_real_edit_still_changes_digest(self, repo):
        before = compute_source_manifest(repo)
        (repo / "README.md").write_text("docs, expanded", encoding="utf-8")
        assert compute_source_manifest(repo).semantic_hash != before.semantic_hash

    def test_code_and_semantic_channels_stay_independent(self, repo):
        before = compute_source_manifest(repo)
        (repo / "app.php").write_text("<?php echo 2;", encoding="utf-8")
        after = compute_source_manifest(repo)
        assert after.code_hash != before.code_hash
        assert after.semantic_hash == before.semantic_hash

    def test_env_zero_restores_mtime_behaviour(self, repo, monkeypatch):
        monkeypatch.setenv(CONTENT_DIGEST_ENV, "0")
        before = compute_source_manifest(repo)
        future = time.time() + 10_000
        os.utime(repo / "README.md", (future, future))
        assert compute_source_manifest(repo).semantic_hash != before.semantic_hash

    def test_unreadable_file_degrades_that_entry_not_the_manifest(self, repo):
        # A file that vanishes or cannot be read must not abort the walk.
        blocked = repo / "secret.md"
        blocked.write_text("x", encoding="utf-8")
        blocked.chmod(0o000)
        try:
            digest = compute_source_manifest(repo)
        finally:
            blocked.chmod(0o644)
        assert digest.file_count >= 3


class TestRefusalBreaksTheLoop:
    """Step 2: a refused digest must stop re-extracting identical sources."""

    def test_changed_source_extracts(self, repo):
        current = compute_source_manifest(repo)
        prior = {"semantic_hash": "old", "code_hash": current.code_hash}
        assert decide_action(prior, current, has_graph=True) == ACTION_EXTRACT

    def test_refused_digest_at_limit_skips_instead_of_re_extracting(self, repo):
        current = compute_source_manifest(repo)
        prior = {
            "semantic_hash": "old",
            "code_hash": current.code_hash,
            "refused_semantic_hash": current.semantic_hash,
            "refusal_streak": REFUSAL_RETRY_LIMIT,
        }
        assert decide_action(prior, current, has_graph=True) == ACTION_SKIP

    def test_refused_digest_below_limit_still_retries_once(self, repo):
        current = compute_source_manifest(repo)
        prior = {
            "semantic_hash": "old",
            "code_hash": current.code_hash,
            "refused_semantic_hash": current.semantic_hash,
            "refusal_streak": REFUSAL_RETRY_LIMIT - 1,
        }
        expected = ACTION_SKIP if REFUSAL_RETRY_LIMIT <= 0 else ACTION_EXTRACT
        assert decide_action(prior, current, has_graph=True) == expected

    def test_a_different_refused_digest_does_not_suppress_a_new_source(self, repo):
        current = compute_source_manifest(repo)
        prior = {
            "semantic_hash": "old",
            "code_hash": current.code_hash,
            "refused_semantic_hash": "some-other-digest",
            "refusal_streak": 99,
        }
        assert decide_action(prior, current, has_graph=True) == ACTION_EXTRACT

    def test_code_only_change_is_unaffected_by_refusal_memory(self, repo):
        current = compute_source_manifest(repo)
        prior = {
            "semantic_hash": current.semantic_hash,
            "code_hash": "old",
            "refused_semantic_hash": current.semantic_hash,
            "refusal_streak": 99,
        }
        # The deterministic AST path must stay available even while the LLM path
        # is suppressed — that is the whole point of keeping the channels split.
        assert decide_action(prior, current, has_graph=True) == ACTION_UPDATE


class TestPublishGateSplit:
    """Step 3: `failed` is unrefreshed, not suspect, and gated separately."""

    def test_status_buckets_are_disjoint_and_correctly_assigned(self):
        assert STATUS_SHRINK_REFUSED in PUBLISH_BLOCKING_STATUSES
        assert STATUS_BOOTSTRAP_FAILED in PUBLISH_BLOCKING_STATUSES
        # A timeout leaves last-good data intact, so it is not "suspect".
        assert STATUS_FAILED not in PUBLISH_BLOCKING_STATUSES
        assert STATUS_FAILED in UNREFRESHED_STATUSES
        assert not (PUBLISH_BLOCKING_STATUSES & UNREFRESHED_STATUSES)

    def test_unrefreshed_threshold_is_looser_than_the_suspect_gate(self):
        from graphify_mesh.sync.config import STALE_PUBLISH_THRESHOLD

        assert UNREFRESHED_PUBLISH_THRESHOLD > STALE_PUBLISH_THRESHOLD

    def test_threshold_env_override(self, monkeypatch):
        monkeypatch.setenv("GRAPHIFY_MESH_UNREFRESHED_THRESHOLD", "0.75")
        assert _unrefreshed_threshold() == 0.75

    @pytest.mark.parametrize("raw", ["", "  ", "abc", "0", "-0.5", "1.5"])
    def test_threshold_env_invalid_falls_back(self, monkeypatch, raw):
        monkeypatch.setenv("GRAPHIFY_MESH_UNREFRESHED_THRESHOLD", raw)
        assert _unrefreshed_threshold() == UNREFRESHED_PUBLISH_THRESHOLD

    def test_observed_production_mix_would_publish(self):
        """The 2026-08-17 run: 16 repos, 2 suspect, 3 unrefreshed.

        Under the old single gate that was 5/16 = 31.25% > 30% and publish was
        refused. Split, the suspect ratio is 12.5% and the unrefreshed ratio
        18.75%, so neither gate trips.
        """
        total, suspect, unrefreshed = 16, 2, 3
        from graphify_mesh.sync.config import STALE_PUBLISH_THRESHOLD

        assert (suspect + unrefreshed) / total > STALE_PUBLISH_THRESHOLD  # old gate blocked
        assert suspect / total <= STALE_PUBLISH_THRESHOLD
        assert unrefreshed / total <= UNREFRESHED_PUBLISH_THRESHOLD

    def test_systemic_failure_still_blocks(self):
        """The hardening invariant: 2 of 4 repos failing must still block."""
        assert 2 / 4 > UNREFRESHED_PUBLISH_THRESHOLD
