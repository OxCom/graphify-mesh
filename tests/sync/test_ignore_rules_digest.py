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
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from graphify_mesh.sync import graphify_cli, repo_tags
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


def _ignore_hash(root: Path, rules: IgnoreRules = RULES) -> str:
    return compute_source_manifest(root, ignore_rules=rules).ignore_hash


class TestIgnoreHash:
    """SourceDigest.ignore_hash: the pipeline forces a full rebuild when it moves."""

    def test_graphifyignore_line_changes_it_and_removal_restores_it(self, repo):
        before = _ignore_hash(repo)
        assert before

        original = (repo / ".graphifyignore").read_text(encoding="utf-8")
        (repo / ".graphifyignore").write_text(original + "src/\n", encoding="utf-8")
        assert _ignore_hash(repo) != before

        (repo / ".graphifyignore").write_text(original, encoding="utf-8")
        assert _ignore_hash(repo) == before

    def test_nested_ignore_file_counts(self, repo):
        before = _ignore_hash(repo)
        (repo / "src" / ".graphifyignore").write_text("app.py\n", encoding="utf-8")
        assert _ignore_hash(repo) != before

    def test_gitignore_counts_only_when_honoured(self, repo):
        honoured = IgnoreRules(RULES.extra_excludes, gitignore=True)
        not_honoured = IgnoreRules(RULES.extra_excludes, gitignore=False)
        before_on = _ignore_hash(repo, honoured)
        before_off = _ignore_hash(repo, not_honoured)
        assert before_on != before_off, "the gitignore switch itself is a rule"

        with (repo / ".gitignore").open("a", encoding="utf-8") as fh:
            fh.write("README.md\n")

        assert _ignore_hash(repo, honoured) != before_on
        assert _ignore_hash(repo, not_honoured) == before_off

    def test_persisted_excludes_count(self, repo):
        more = IgnoreRules((*RULES.extra_excludes, "src/"))
        assert _ignore_hash(repo, more) != _ignore_hash(repo)

    def test_unrelated_source_edit_leaves_it(self, repo):
        before = _ignore_hash(repo)
        _edit(repo, "src/app.py")
        (repo / "src" / "new.py").write_text("x = 2\n", encoding="utf-8")
        assert _ignore_hash(repo) == before

    def test_new_file_beside_an_unchanged_ignore_file_leaves_it(self, repo):
        (repo / "scratch").mkdir()
        (repo / "scratch" / ".gitignore").write_text("*.log\n", encoding="utf-8")
        before = _ignore_hash(repo)
        (repo / "scratch" / "x.py").write_text("x = 1\n", encoding="utf-8")
        assert _ignore_hash(repo) == before

    def test_symlinked_dir_outside_the_repo_is_not_read(self, repo, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / ".graphifyignore").write_text("a/\n", encoding="utf-8")
        (outside / "b.py").write_text("x = 1\n", encoding="utf-8")
        (repo / "linked").symlink_to(outside, target_is_directory=True)
        before = _ignore_hash(repo)
        (outside / ".graphifyignore").write_text("b/\n", encoding="utf-8")
        assert _ignore_hash(repo) == before

    def test_table_that_ignores_no_files_has_none(self, repo):
        assert compute_source_manifest(repo, LEGACY_CLASSIFICATION_V2).ignore_hash == ""

    def test_migration_walk_matches_the_steady_walk(self, repo):
        """A legacy table descends var/cache, which graphify prunes; an ignore
        file found there must not move the hash, or the run after the migration
        is accepted rebuilds again."""
        (repo / "var" / "cache" / ".gitignore").write_text("*.php\n", encoding="utf-8")
        # A descended dir holding only an uncategorized file next to its rules.
        (repo / "pkg" / "sub").mkdir(parents=True)
        (repo / "pkg" / ".graphifyignore").write_text("sub/\n", encoding="utf-8")
        (repo / "pkg" / "notes.ini").write_text("[a]\n", encoding="utf-8")
        (repo / "pkg" / "sub" / "settings.ini").write_text("[b]\n", encoding="utf-8")
        steady = compute_source_manifests(repo, (CURRENT_CLASSIFICATION,), RULES)[0]
        migrating = compute_source_manifests(
            repo, (CURRENT_CLASSIFICATION, LEGACY_CLASSIFICATION_V2), RULES
        )
        assert migrating[0].ignore_hash == steady.ignore_hash
        assert migrating[1].ignore_hash == ""

    def test_not_part_of_the_stored_digest(self, repo):
        digest = compute_source_manifest(repo, ignore_rules=RULES)
        assert "ignore_hash" not in digest.to_dict()


def _graph_sources(root: Path) -> set[str]:
    graph = json.loads((root / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    return {
        str(node.get("source_file") or "").replace("\\", "/") for node in graph.get("nodes", [])
    }


def _under_tests(sources: set[str]) -> bool:
    return any(s == "tests" or s.startswith("tests/") or "/tests/" in s for s in sources)


def test_real_graphify_force_update_drops_newly_ignored_files(tmp_path, monkeypatch):
    """The bug: incremental `update` keeps nodes of a newly ignored file;
    `update --force`, which the pipeline now runs on an ignore change, drops them."""
    graphify_bin = shutil.which("graphify") or ""
    if not graphify_bin:
        pytest.skip("the real `graphify` binary is not on PATH")
    monkeypatch.delenv(graphify_cli.CHILD_SANDBOX_VAR, raising=False)
    root = tmp_path / "repo"
    root.mkdir()
    _write(
        root,
        {
            "src/a.py": "def alpha():\n    return 1\n",
            "tests/test_a.py": "def test_alpha():\n    assert True\n",
        },
    )
    _git(root, "init", "-q", ".")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "init")
    collection = root / "graphify-out"
    home = tmp_path / "home"

    def update(force: bool = False) -> set[str]:
        result = graphify_cli.run_update(graphify_bin, root, collection, home, force=force)
        assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
        return _graph_sources(root)

    assert _under_tests(update()), "precondition: tests/ is extracted"

    (root / ".graphifyignore").write_text("tests/\n", encoding="utf-8")
    sources = update(force=True)
    assert not _under_tests(sources), sorted(sources)
    assert any(s.endswith("src/a.py") for s in sources)

    (root / ".graphifyignore").write_text("", encoding="utf-8")
    assert _under_tests(update()), "removing the rule brings the nodes back"
