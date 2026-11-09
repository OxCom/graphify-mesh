"""Classification table v3: the digest follows graphify's own ignore rules.

Decision 2026-11-06 (post-upgrade open issues, Phase 2): the mesh hashed about
3x the files graphify reads (Symfony var/cache, dist output, gitignored
CLAUDE.md). An edit graphify never reads must not rebuild the repo, v2 state
must migrate with an update, and a graphify upgrade that moves the ignore rules
must be adopted with an update, never an extract.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from graphify_mesh.sync import repo_tags
from graphify_mesh.sync.config import (
    CLASSIFICATION_TABLES,
    CLASSIFICATION_VERSION,
    CURRENT_CLASSIFICATION,
    LEGACY_CLASSIFICATION_V2,
)
from graphify_mesh.sync.pipeline import run
from graphify_mesh.sync.state import (
    BUILD_CONFIG_FILENAME,
    CONTENT_DIGEST_ENV,
    IgnoreRules,
    compute_source_manifest,
    compute_source_manifests,
    load_state,
    read_build_ignore_rules,
    save_state,
)
from graphify_mesh.sync.sync_project import (
    ACTION_EXTRACT,
    ACTION_SKIP,
    ACTION_UPDATE,
    CLASSIFICATION_VERSION_KEY,
    DIGEST_RECIPE_KEY,
    decide_action,
    digest_recipe,
    digest_recipe_changed,
    needs_classification_migration,
)

RULES = IgnoreRules(extra_excludes=("docs/", "dist/", "generated/"))

FILES = {
    ".gitignore": "var/cache/\nCLAUDE.md\n",
    ".graphifyignore": "private_notes/\n",
    "src/app.py": "print(1)\n",
    "README.md": "# repo\n",
    "var/cache/Container.php": "<?php // compiled\n",
    "CLAUDE.md": "# local agent notes\n",
    "private_notes/plan.md": "# plan\n",
    "generated/client.py": "x = 1\n",
    "dist/bundle.js": "var a;\n",
    "docs/guide.md": "# guide\n",
}
# Each is excluded by a different rule: .gitignore dir, .gitignore file,
# .graphifyignore, persisted --exclude, and the docs/dist names both sides prune.
IGNORED = [
    "var/cache/Container.php",
    "CLAUDE.md",
    "private_notes/plan.md",
    "generated/client.py",
    "dist/bundle.js",
    "docs/guide.md",
]
TRACKED = [".gitignore", ".graphifyignore", "src/app.py", "README.md"]


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@example.org", "-c", "user.name=t", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.delenv(CONTENT_DIGEST_ENV, raising=False)
    root = tmp_path / "repo"
    root.mkdir()
    _write(root, FILES)
    _git(root, "init", "-q", ".")
    _git(root, "add", *TRACKED)
    _git(root, "commit", "-qm", "init")
    return root


def _edit(root: Path, rel: str) -> None:
    with (root / rel).open("a", encoding="utf-8") as fh:
        fh.write("// edited\n")


def _v3_state(root: Path) -> dict:
    state = compute_source_manifest(root, ignore_rules=RULES).to_dict()
    state[CLASSIFICATION_VERSION_KEY] = CLASSIFICATION_VERSION
    state[DIGEST_RECIPE_KEY] = digest_recipe("0.9.103")
    return state


def _v2_state(root: Path) -> dict:
    state = compute_source_manifest(root, LEGACY_CLASSIFICATION_V2).to_dict()
    state[CLASSIFICATION_VERSION_KEY] = 2
    return state


def _decide(prior: dict, root: Path, **kwargs) -> str:
    """decide_action wired the way pipeline.py wires it."""
    if needs_classification_migration(prior):
        legacy_table = CLASSIFICATION_TABLES[int(prior[CLASSIFICATION_VERSION_KEY])]
        current, legacy = compute_source_manifests(
            root, (CURRENT_CLASSIFICATION, legacy_table), RULES
        )
    else:
        current, legacy = compute_source_manifest(root, ignore_rules=RULES), None
    return decide_action(prior, current, True, legacy_manifest=legacy, **kwargs)


class TestTable:
    def test_v3_keeps_v2_categories_and_adds_the_rules(self):
        assert CLASSIFICATION_VERSION == 3
        assert CLASSIFICATION_TABLES[3] is CURRENT_CLASSIFICATION
        assert CURRENT_CLASSIFICATION.honours_ignore_rules
        assert not LEGACY_CLASSIFICATION_V2.honours_ignore_rules
        for name in (
            "code_extensions",
            "semantic_extensions",
            "code_filenames",
            "ignored_filenames",
            "pruned_dir_names",
            "pruned_dir_suffixes",
        ):
            assert getattr(CURRENT_CLASSIFICATION, name) == getattr(LEGACY_CLASSIFICATION_V2, name)

    def test_v3_counts_only_what_graphify_reads(self, repo):
        v3 = compute_source_manifest(repo, ignore_rules=RULES)
        v2 = compute_source_manifest(repo, LEGACY_CLASSIFICATION_V2)
        # v3: src/app.py, README.md. v2 also counts var/cache, CLAUDE.md,
        # private_notes and generated (docs/ and dist/ are pruned by name).
        assert v3.file_count == 2
        assert v2.file_count == 6


class TestDecisions:
    @pytest.mark.parametrize("rel", IGNORED)
    def test_edit_to_an_ignored_file_skips(self, repo, rel):
        prior = _v3_state(repo)
        _edit(repo, rel)
        assert _decide(prior, repo) == ACTION_SKIP

    def test_real_doc_edit_extracts(self, repo):
        prior = _v3_state(repo)
        _edit(repo, "README.md")
        assert _decide(prior, repo) == ACTION_EXTRACT

    def test_code_edit_updates(self, repo):
        prior = _v3_state(repo)
        _edit(repo, "src/app.py")
        assert _decide(prior, repo) == ACTION_UPDATE

    def test_tracked_file_matching_gitignore_still_counts(self, repo):
        """Git ignores no tracked file, and neither does graphify (#2759)."""
        _write(repo, {"var/cache/kept.md": "# tracked\n"})
        _git(repo, "add", "-f", "var/cache/kept.md")
        _git(repo, "commit", "-qm", "kept")
        prior = _v3_state(repo)
        _edit(repo, "var/cache/kept.md")
        assert _decide(prior, repo) == ACTION_EXTRACT

    def test_persisted_excludes_are_read_from_the_collection(self, tmp_path):
        collection = tmp_path / "graphify-out"
        collection.mkdir()
        assert read_build_ignore_rules(collection) == IgnoreRules()
        (collection / BUILD_CONFIG_FILENAME).write_text(
            json.dumps({"excludes": ["docs/", "", 3], "gitignore": False}), encoding="utf-8"
        )
        assert read_build_ignore_rules(collection) == IgnoreRules(("docs/",), False)
        (collection / BUILD_CONFIG_FILENAME).write_text("{not json", encoding="utf-8")
        assert read_build_ignore_rules(collection) == IgnoreRules()


class TestMigrationFromV2:
    def test_unchanged_tree_updates_never_extracts(self, repo):
        assert _decide(_v2_state(repo), repo) == ACTION_UPDATE

    def test_code_edit_in_window_updates(self, repo):
        prior = _v2_state(repo)
        _edit(repo, "src/app.py")
        assert _decide(prior, repo) == ACTION_UPDATE

    def test_ignored_doc_edit_in_window_extracts_once(self, repo):
        """v2 hashed CLAUDE.md as a document, so its edit in the window cannot be
        told from a real one. One extract is the conservative cost, as for v1."""
        prior = _v2_state(repo)
        _edit(repo, "CLAUDE.md")
        assert _decide(prior, repo) == ACTION_EXTRACT

    def test_real_doc_edit_in_window_extracts(self, repo):
        prior = _v2_state(repo)
        _edit(repo, "README.md")
        assert _decide(prior, repo) == ACTION_EXTRACT

    def test_one_walk_reproduces_the_stored_v2_hashes(self, repo):
        stored = _v2_state(repo)
        _current, legacy = compute_source_manifests(
            repo, (CURRENT_CLASSIFICATION, LEGACY_CLASSIFICATION_V2), RULES
        )
        assert legacy.code_hash == stored["code_hash"]
        assert legacy.semantic_hash == stored["semantic_hash"]
        assert legacy.file_count == stored["file_count"]


class TestRecipe:
    def test_recipe_names_table_and_graphify(self):
        assert digest_recipe("0.9.103") == f"v{CLASSIFICATION_VERSION}+graphify-ignore-0.9.103"
        assert digest_recipe(None) is None

    def test_changed_only_for_current_table_state(self, repo):
        state = _v3_state(repo)
        assert not digest_recipe_changed(state, digest_recipe("0.9.103"))
        assert digest_recipe_changed(state, digest_recipe("0.9.104"))
        assert not digest_recipe_changed(state, None), "unknown version is not a change"
        assert not digest_recipe_changed(_v2_state(repo), digest_recipe("0.9.104")), (
            "a migration stamps the recipe itself"
        )
        assert not digest_recipe_changed(None, digest_recipe("0.9.104"))

    def test_recipe_change_moving_a_doc_updates_not_extracts(self, repo):
        """A new graphify stops ignoring a document: the semantic hash moves
        with no edit. The old digest cannot be recomputed, so update."""
        prior = _v3_state(repo)
        (repo / ".graphifyignore").write_text("", encoding="utf-8")
        assert _decide(prior, repo) == ACTION_EXTRACT, (
            "without the recipe flag it looks like an edit"
        )
        assert _decide(prior, repo, recipe_changed=True) == ACTION_UPDATE

    def test_recipe_change_with_matching_hashes_skips(self, repo):
        assert _decide(_v3_state(repo), repo, recipe_changed=True) == ACTION_SKIP


class TestIgnoreRulesAreSafe:
    def test_repo_fsmonitor_command_never_runs(self, repo, tmp_path):
        """Upstream ignored_predicate runs `git ls-files` with the repo's own
        config, and ls-files runs core.fsmonitor: code execution from repo config."""
        marker = tmp_path / "fsmonitor-ran"
        hook = tmp_path / "hook.sh"
        hook.write_text(f"#!/bin/sh\ntouch {marker}\necho 0\n", encoding="utf-8")
        hook.chmod(0o755)
        _git(repo, "config", "core.fsmonitor", str(hook))

        compute_source_manifest(repo, ignore_rules=RULES)

        assert not marker.exists()

    def test_symlinked_or_fifo_ignore_file_is_not_read(self, repo, tmp_path):
        """A cloned `.gitignore -> FIFO` would block upstream's reader forever."""
        fifo = tmp_path / "fifo"
        os.mkfifo(fifo)
        nested = repo / "src" / ".gitignore"
        nested.symlink_to(fifo)
        (repo / "private_notes" / ".graphifyignore").unlink(missing_ok=True)
        os.mkfifo(repo / "private_notes" / ".gitignore")
        result: list = []
        worker = threading.Thread(
            target=lambda: result.append(compute_source_manifest(repo, ignore_rules=RULES)),
            daemon=True,
        )
        worker.start()
        worker.join(timeout=20)
        assert result, "the digest blocked on a FIFO ignore file"

    def test_git_failure_keeps_gitignored_files(self, repo, monkeypatch):
        """No tracked-file list means .gitignore could drop a tracked file;
        keep more files instead (a needless rebuild beats a missed one)."""
        # GIT_DIR in the env no longer reaches git (state._tracked_path_keys
        # scrubs GIT_*), so the failure is simulated at the subprocess call.
        monkeypatch.setattr(
            "graphify_mesh.sync.state.subprocess.run",
            lambda argv, **kw: subprocess.CompletedProcess(argv, 128, b"", b""),
        )
        digest = compute_source_manifest(repo, ignore_rules=RULES)
        assert digest.file_count == 4, (
            "CLAUDE.md and var/cache back in, .graphifyignore still applied"
        )


PIPELINE_REPO = "example-org.svc"


@pytest.fixture
def pipeline_repo(env, monkeypatch):
    monkeypatch.delenv(CONTENT_DIGEST_ENV, raising=False)
    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.103")
    root = env.add_repo(PIPELINE_REPO, "example-org", "svc", "svc.example-org.dev.lo")
    env.write_registry()
    _write(root, FILES)
    collection = env.collection_path("example-org", "svc")
    (collection / BUILD_CONFIG_FILENAME).write_text(
        json.dumps({"excludes": list(RULES.extra_excludes)}), encoding="utf-8"
    )
    return env, root


def _row(report) -> dict:
    return {a["repo_id"]: a for a in report.project_actions}[PIPELINE_REPO]


def _settings(env, **overrides):
    return env.settings(shrink_policy="count", **overrides)


class TestPipeline:
    def test_ignored_edits_skip_and_doc_edit_extracts(self, pipeline_repo):
        env, root = pipeline_repo
        run(_settings(env))
        entry = load_state(env.settings().state_path)[PIPELINE_REPO]
        assert entry[CLASSIFICATION_VERSION_KEY] == CLASSIFICATION_VERSION
        assert entry[DIGEST_RECIPE_KEY] == digest_recipe("0.9.103")

        for rel in IGNORED:
            _edit(root, rel)
        assert _row(run(_settings(env)))["action"] == ACTION_SKIP

        _edit(root, "README.md")
        assert _row(run(_settings(env, dry_run=True)))["action"] == ACTION_EXTRACT

    def test_v2_state_migrates_with_an_update(self, pipeline_repo):
        env, root = pipeline_repo
        run(_settings(env))
        path = env.settings().state_path
        state = load_state(path)
        state[PIPELINE_REPO].update(
            compute_source_manifest(root, LEGACY_CLASSIFICATION_V2).to_dict()
        )
        state[PIPELINE_REPO][CLASSIFICATION_VERSION_KEY] = 2
        state[PIPELINE_REPO].pop(DIGEST_RECIPE_KEY, None)
        save_state(path, state)

        assert _row(run(_settings(env)))["action"] == ACTION_UPDATE
        entry = load_state(path)[PIPELINE_REPO]
        assert entry[CLASSIFICATION_VERSION_KEY] == CLASSIFICATION_VERSION
        assert entry[DIGEST_RECIPE_KEY] == digest_recipe("0.9.103")
        assert _row(run(_settings(env)))["action"] == ACTION_SKIP

    def test_graphify_upgrade_moving_the_rules_updates(self, pipeline_repo, monkeypatch):
        """Stored hashes from another recipe that no longer match: update."""
        env, _root = pipeline_repo
        run(_settings(env))
        path = env.settings().state_path
        state = load_state(path)
        state[PIPELINE_REPO]["semantic_hash"] = "taken-under-old-rules"
        save_state(path, state)
        monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.104")

        assert _row(run(_settings(env)))["action"] == ACTION_UPDATE
        entry = load_state(path)[PIPELINE_REPO]
        assert entry[DIGEST_RECIPE_KEY] == digest_recipe("0.9.104")
