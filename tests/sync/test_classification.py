"""Classification table v2 and the per-repo migration from v1 state.

Decision 2026-11-06 (yaml-json-classification): the mesh category of a file is
the image of upstream graphify `classify_file`. State written under the v1
table must migrate with an `update`, never an `extract`, unless a document
really changed in the migration window.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graphify_mesh.sync import pipeline, repo_tags
from graphify_mesh.sync import state as state_mod
from graphify_mesh.sync.config import (
    CLASSIFICATION_TABLES,
    CLASSIFICATION_VERSION,
    CURRENT_CLASSIFICATION,
    DELIBERATELY_EXCLUDED_DOC,
    LEGACY_CLASSIFICATION_V1,
    categorize_file,
)
from graphify_mesh.sync.pipeline import run
from graphify_mesh.sync.state import (
    CONTENT_DIGEST_ENV,
    compute_source_manifest,
    compute_source_manifests,
    load_state,
    save_state,
)
from graphify_mesh.sync.sync_project import (
    ACTION_EXTRACT,
    ACTION_SKIP,
    ACTION_UPDATE,
    CLASSIFICATION_VERSION_KEY,
    PRODUCER_UPDATE,
    REFUSAL_RETRY_LIMIT,
    decide_action,
    effective_shrink_grant,
    needs_classification_migration,
    recorded_classification_version,
    refusal_key,
)

FILES = {
    "m.py": "print(1)\n",
    "compose.yaml": "services: {}\n",
    "README.md": "# repo\n",
    "settings.toml": "[a]\nb = 1\n",
    "pyproject.toml": '[project]\nname = "m"\n',
    "package-lock.json": "{}\n",
}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.delenv(CONTENT_DIGEST_ENV, raising=False)
    for name, text in FILES.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def _v1_state(root: Path) -> dict:
    """State as written before CLASSIFICATION_VERSION existed."""
    return compute_source_manifest(root, LEGACY_CLASSIFICATION_V1).to_dict()


def _v2_state(root: Path) -> dict:
    """Accepted state as the pipeline records it now."""
    state = compute_source_manifest(root).to_dict()
    state[CLASSIFICATION_VERSION_KEY] = CLASSIFICATION_VERSION
    return state


def _decide(prior: dict, root: Path, **kwargs) -> str:
    """decide_action wired the way pipeline.py wires it."""
    legacy = (
        compute_source_manifest(root, LEGACY_CLASSIFICATION_V1)
        if needs_classification_migration(prior)
        else None
    )
    return decide_action(
        prior, compute_source_manifest(root), True, legacy_manifest=legacy, **kwargs
    )


def _edit(root: Path, name: str) -> None:
    (root / name).write_text(FILES[name] + "# edited\n", encoding="utf-8")


PIPELINE_REPO = "example-org.svc"


def _pipeline_repo(env, monkeypatch) -> Path:
    """One registered repo holding FILES, synced with a fixed graphify version."""
    monkeypatch.delenv(CONTENT_DIGEST_ENV, raising=False)
    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.89")
    root = env.add_repo(PIPELINE_REPO, "example-org", "svc", "svc.example-org.dev.lo")
    env.write_registry()
    for name, text in FILES.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


def _rewrite_as_v1(env, root: Path, **fields) -> None:
    """Turn the repo's accepted state into state written before v2, plus `fields`."""
    path = env.settings().state_path
    state = load_state(path)
    entry = state[PIPELINE_REPO]
    entry.update(compute_source_manifest(root, LEGACY_CLASSIFICATION_V1).to_dict())
    entry.pop(CLASSIFICATION_VERSION_KEY, None)
    entry.update(fields)
    save_state(path, state)


class TestMigration:
    def test_v1_state_unchanged_tree_updates_never_extracts(self, repo):
        assert _decide(_v1_state(repo), repo) == ACTION_UPDATE

    def test_v1_state_with_real_doc_edit_extracts(self, repo):
        prior = _v1_state(repo)
        _edit(repo, "README.md")
        assert _decide(prior, repo) == ACTION_EXTRACT

    def test_v1_state_yaml_edit_in_window_extracts_once(self, repo):
        """The v1 semantic hash covered YAML, so a YAML edit in the window cannot
        be told from a document edit. One extract is the conservative cost."""
        prior = _v1_state(repo)
        _edit(repo, "compose.yaml")
        assert _decide(prior, repo) == ACTION_EXTRACT

    def test_v1_state_without_moved_files_skips(self, tmp_path, monkeypatch):
        """No file changed category: hashes already match, nothing to rebuild.
        The pipeline stamps the version on this skip."""
        monkeypatch.delenv(CONTENT_DIGEST_ENV, raising=False)
        (tmp_path / "m.py").write_text("print(1)\n", encoding="utf-8")
        (tmp_path / "README.md").write_text("# r\n", encoding="utf-8")
        assert _decide(_v1_state(tmp_path), tmp_path) == ACTION_SKIP

    def test_v1_state_without_legacy_digest_keeps_old_contract(self, repo):
        """A direct caller that passes no legacy digest sees the raw hash diff."""
        current = compute_source_manifest(repo)
        assert decide_action(_v1_state(repo), current, True) == ACTION_EXTRACT

    def test_held_update_stays_held_during_migration(self, repo):
        prior = {**_v1_state(repo), "refused_key": "k", "refusal_streak": REFUSAL_RETRY_LIMIT}
        assert _decide(prior, repo, refusal_key="k") == ACTION_SKIP
        assert _decide(prior, repo, refusal_key="other") == ACTION_UPDATE

    def test_recorded_v2_state_needs_no_migration(self, repo):
        assert needs_classification_migration(_v1_state(repo))
        assert not needs_classification_migration(_v2_state(repo))
        assert not needs_classification_migration(None)

    def test_held_extract_refused_under_v1_survives_migration(self, repo):
        """The refusal hash was recorded under v1; comparing it with the v2 hash
        released the hold and paid for another LLM extract of the same source."""
        prior = _v1_state(repo)
        _edit(repo, "README.md")
        legacy = compute_source_manifest(repo, LEGACY_CLASSIFICATION_V1)
        assert legacy.semantic_hash != compute_source_manifest(repo).semantic_hash
        prior.update(refused_semantic_hash=legacy.semantic_hash, refusal_streak=1)
        assert _decide(prior, repo) == ACTION_SKIP

    def test_v1_grant_still_releases_its_attempt_while_migrating(self, repo):
        prior = _v1_state(repo)
        _edit(repo, "README.md")
        legacy = compute_source_manifest(repo, LEGACY_CLASSIFICATION_V1)
        current = compute_source_manifest(repo)
        prior.update(refused_semantic_hash=legacy.semantic_hash, refusal_streak=1)
        grant = effective_shrink_grant(legacy.attempt_id, current, legacy)
        assert grant == current.attempt_id
        assert _decide(prior, repo, shrink_grant=grant) == ACTION_EXTRACT

    def test_grant_is_translated_only_from_the_legacy_attempt(self, repo):
        legacy = compute_source_manifest(repo, LEGACY_CLASSIFICATION_V1)
        current = compute_source_manifest(repo)
        assert effective_shrink_grant(legacy.attempt_id, current, None) == legacy.attempt_id
        assert effective_shrink_grant("feedface", current, legacy) == "feedface"
        assert effective_shrink_grant(None, current, legacy) is None

    def test_pipeline_holds_an_update_refused_under_v1(self, env, monkeypatch):
        root = _pipeline_repo(env, monkeypatch)
        run(env.settings(shrink_policy="count"))
        legacy = compute_source_manifest(root, LEGACY_CLASSIFICATION_V1)
        assert legacy.attempt_id != compute_source_manifest(root).attempt_id
        _rewrite_as_v1(
            env,
            root,
            refused_key=refusal_key(legacy, PRODUCER_UPDATE, "count", "0.9.89"),
            refusal_streak=1,
        )
        calls = len(env.read_argv_log())

        report = run(env.settings(shrink_policy="count"))

        row = {a["repo_id"]: a for a in report.project_actions}[PIPELINE_REPO]
        assert row["action"] == ACTION_SKIP
        assert len(env.read_argv_log()) == calls

    def test_pipeline_recomputes_under_the_recorded_table(self, env, monkeypatch):
        """A v4 bump must recompute v3 state under v3, not under a frozen older table."""
        _pipeline_repo(env, monkeypatch)
        run(env.settings(shrink_policy="count"))
        # A v4 bump, simulated: the active table is the selected setting.
        monkeypatch.setitem(CLASSIFICATION_TABLES, 4, CURRENT_CLASSIFICATION)
        calls: list = []
        real = pipeline.compute_source_manifests

        def recording(path, tables, *rest):
            calls.append(tuple(tables))
            return real(path, tables, *rest)

        monkeypatch.setattr(pipeline, "compute_source_manifests", recording)

        report = run(env.settings(shrink_policy="count", dry_run=True, classification_version=4))

        row = {a["repo_id"]: a for a in report.project_actions}[PIPELINE_REPO]
        assert row["action"] == ACTION_SKIP, "hashes still valid under v2: nothing to rebuild"
        assert calls == [(CURRENT_CLASSIFICATION, CLASSIFICATION_TABLES[3])], (
            "one walk yields both digests, the legacy one under v3, never older"
        )

    def test_unparseable_version_compares_raw_hashes(self, env, monkeypatch, caplog):
        """A corrupt version must not abort the run: no table is known for it,
        so the raw hashes decide, as for an unknown version."""
        _pipeline_repo(env, monkeypatch)
        run(env.settings(shrink_policy="count"))
        path = env.settings().state_path
        state = load_state(path)
        state[PIPELINE_REPO][CLASSIFICATION_VERSION_KEY] = "two"
        save_state(path, state)
        calls: list = []
        real = pipeline.compute_source_manifests

        def recording(path, tables, *rest):
            calls.append(tuple(tables))
            return real(path, tables, *rest)

        monkeypatch.setattr(pipeline, "compute_source_manifests", recording)

        with caplog.at_level("WARNING", logger="graphify_mesh.sync"):
            report = run(env.settings(shrink_policy="count", dry_run=True))

        row = {a["repo_id"]: a for a in report.project_actions}[PIPELINE_REPO]
        assert row["action"] == ACTION_SKIP, "raw v2 hashes still match the tree"
        assert calls == [(CURRENT_CLASSIFICATION,)]
        assert any(
            PIPELINE_REPO in r.getMessage() and "'two'" in r.getMessage() for r in caplog.records
        )


class TestV2Decisions:
    def test_unchanged_tree_skips(self, repo):
        assert _decide(_v2_state(repo), repo) == ACTION_SKIP

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("compose.yaml", ACTION_UPDATE),
            ("README.md", ACTION_EXTRACT),
            ("settings.toml", ACTION_SKIP),
            ("package-lock.json", ACTION_SKIP),
            ("pyproject.toml", ACTION_UPDATE),
            ("m.py", ACTION_UPDATE),
        ],
    )
    def test_edit_routes_by_category(self, repo, name, expected):
        prior = _v2_state(repo)
        _edit(repo, name)
        assert _decide(prior, repo) == expected


class TestCategorize:
    @pytest.mark.parametrize(
        ("name", "v1", "v2"),
        [
            ("compose.yaml", "semantic", "code"),
            ("data.json", "semantic", "code"),
            ("settings.toml", "semantic", "ignore"),
            ("pyproject.toml", "semantic", "code"),
            ("Cargo.toml", "semantic", "code"),
            ("package-lock.json", "semantic", "ignore"),
            ("pnpm-lock.yaml", "semantic", "ignore"),
            (".env.example", "ignore", "ignore"),
            ("README.md", "semantic", "semantic"),
            ("m.py", "code", "code"),
        ],
    )
    def test_v1_table_is_frozen_and_v2_differs(self, name, v1, v2):
        assert categorize_file(Path(name), LEGACY_CLASSIFICATION_V1) == v1
        assert categorize_file(Path(name)) == v2


class TestTableV2Contents:
    def test_registry_is_keyed_by_each_tables_own_version(self):
        assert {v: t.version for v, t in CLASSIFICATION_TABLES.items()} == {1: 1, 2: 2, 3: 3}
        assert CLASSIFICATION_TABLES[CLASSIFICATION_VERSION] is CURRENT_CLASSIFICATION
        assert CLASSIFICATION_TABLES[1] is LEGACY_CLASSIFICATION_V1

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("run.sh", "code"),
            ("schema.sql", "code"),
            ("x.mjs", "code"),
            ("main.tf", "code"),
            ("README.md", "semantic"),
            ("index.html", "ignore"),
            ("tpl.j2", "ignore"),
            ("paper.pdf", "ignore"),
            (".graphifyinclude", "ignore"),
        ],
    )
    def test_v2_routing(self, name, expected):
        assert categorize_file(Path(name)) == expected

    def test_excluded_doc_never_routes_to_semantic_or_code(self):
        assert not DELIBERATELY_EXCLUDED_DOC & CURRENT_CLASSIFICATION.semantic_extensions
        assert not DELIBERATELY_EXCLUDED_DOC & CURRENT_CLASSIFICATION.code_extensions


V1_PRUNED_DIRS = {
    ".git",
    "node_modules",
    "vendor",
    "graphify-out",
    "docs",
    "doc",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".idea",
    ".vscode",
}


class TestPrunedDirs:
    def test_v1_pruned_dirs_are_frozen(self):
        assert LEGACY_CLASSIFICATION_V1.pruned_dir_names == V1_PRUNED_DIRS
        assert LEGACY_CLASSIFICATION_V1.pruned_dir_suffixes == ()

    def test_v2_prunes_everything_v1_prunes(self):
        assert V1_PRUNED_DIRS <= CURRENT_CLASSIFICATION.pruned_dir_names

    @pytest.mark.parametrize(
        "rel",
        [
            ".mypy_cache/3.12/m.meta.json",
            "target/debug/build.sh",
            ".next/cache/data.json",
            "m.egg-info/SOURCES.txt",
        ],
    )
    def test_build_output_moves_v1_but_not_v2(self, repo, rel):
        """v2 makes .json/.sh code, so a cache file would churn code_hash on
        every run; v1 still hashes it, as the fleet's stored state expects."""
        v2_before, v1_before = compute_source_manifests(
            repo, (CURRENT_CLASSIFICATION, LEGACY_CLASSIFICATION_V1)
        )
        path = repo / rel
        path.parent.mkdir(parents=True)
        path.write_text("x\n", encoding="utf-8")

        v2_after, v1_after = compute_source_manifests(
            repo, (CURRENT_CLASSIFICATION, LEGACY_CLASSIFICATION_V1)
        )

        assert v2_after == v2_before
        if categorize_file(path, LEGACY_CLASSIFICATION_V1) != "ignore":
            assert v1_after != v1_before


class TestSingleWalk:
    def test_both_digests_equal_separate_walks(self, repo):
        (repo / ".mypy_cache").mkdir()
        (repo / ".mypy_cache" / "a.json").write_text("{}\n", encoding="utf-8")
        both = compute_source_manifests(repo, (CURRENT_CLASSIFICATION, LEGACY_CLASSIFICATION_V1))
        assert both == (
            compute_source_manifest(repo),
            compute_source_manifest(repo, LEGACY_CLASSIFICATION_V1),
        )

    def test_each_file_is_read_once_for_both_tables(self, repo, monkeypatch):
        """Both digests come from one snapshot: a file read once cannot show one
        content to the current table and another to the legacy table."""
        reads: dict[str, int] = {}
        real = state_mod.file_content_hash

        def counting(path):
            reads[path.name] = reads.get(path.name, 0) + 1
            return real(path)

        monkeypatch.setattr(state_mod, "file_content_hash", counting)

        compute_source_manifests(repo, (CURRENT_CLASSIFICATION, LEGACY_CLASSIFICATION_V1))

        # package-lock.json counts too: v2 ignores it, v1 still hashes it.
        assert reads == dict.fromkeys(FILES, 1)

    def test_missing_root_gives_one_empty_digest_per_table(self, tmp_path):
        digests = compute_source_manifests(
            tmp_path / "gone", (CURRENT_CLASSIFICATION, LEGACY_CLASSIFICATION_V1)
        )
        assert [d.code_hash for d in digests] == ["empty", "empty"]


class TestRecordedVersion:
    @pytest.mark.parametrize(
        ("stored", "expected"),
        [(2, 2), ("2", 2), ("two", None), (None, None), ([2], None)],
    )
    def test_parse(self, stored, expected):
        assert recorded_classification_version({CLASSIFICATION_VERSION_KEY: stored}) == expected

    def test_absent_version_is_v1(self):
        assert recorded_classification_version({}) == 1

    def test_unparseable_version_needs_migration_but_compares_raw(self, repo):
        prior = _v2_state(repo)
        prior[CLASSIFICATION_VERSION_KEY] = "two"
        assert needs_classification_migration(prior)
        assert pipeline._legacy_table(prior, "example-org.svc") is None
        # No legacy digest, as the pipeline passes for it: raw compare.
        assert decide_action(prior, compute_source_manifest(repo), True) == ACTION_SKIP
