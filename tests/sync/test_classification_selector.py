"""GRAPHIFY_MESH_CLASSIFICATION_VERSION: roll the digest back to an older table.

Selecting 1 makes the mesh hash sources like it did before the v2/v3 change
(legacy table, no ignore rules). State records the selected version and digest
recipe, so switching back to 3 runs the normal one-walk migration: an update,
never an extract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graphify_mesh.sync import pipeline, repo_tags
from graphify_mesh.sync.config import (
    CLASSIFICATION_TABLES,
    CLASSIFICATION_VERSION,
    LEGACY_CLASSIFICATION_V1,
    Settings,
)
from graphify_mesh.sync.pipeline import run
from graphify_mesh.sync.state import CONTENT_DIGEST_ENV, SourceDigest, load_state
from graphify_mesh.sync.sync_project import (
    ACTION_EXTRACT,
    ACTION_SKIP,
    ACTION_UPDATE,
    CLASSIFICATION_VERSION_KEY,
    DIGEST_RECIPE_KEY,
    PRODUCER_UPDATE,
    decide_action,
    digest_recipe,
    digest_recipe_changed,
    needs_classification_migration,
    refusal_key,
)

ENV_VAR = "GRAPHIFY_MESH_CLASSIFICATION_VERSION"
REPO = "example-org.svc"
GRAPHIFY_VERSION = "0.9.89"
FILES = {
    "m.py": "print(1)\n",
    "compose.yaml": "services: {}\n",
    "README.md": "# repo\n",
    "settings.toml": "[a]\nb = 1\n",
    "notes.md": "ignored by graphify\n",
    ".graphifyignore": "notes.md\n",
}


def _settings(tmp_path: Path, **overrides) -> Settings:
    mesh_root = tmp_path / "mesh"
    return Settings.from_env(
        mesh_root=mesh_root,
        scan_roots=[tmp_path / "www"],
        registry_path=mesh_root / "bin" / "registry.json",
        **overrides,
    )


class TestSetting:
    def test_default_is_the_current_version(self, tmp_path, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        settings = _settings(tmp_path)
        assert settings.classification_version == CLASSIFICATION_VERSION == 3
        assert settings.classification_table is CLASSIFICATION_TABLES[3]

    @pytest.mark.parametrize("version", sorted(CLASSIFICATION_TABLES))
    def test_every_known_table_is_selectable(self, tmp_path, monkeypatch, version):
        monkeypatch.setenv(ENV_VAR, str(version))
        settings = _settings(tmp_path)
        assert settings.classification_version == version
        assert settings.classification_table is CLASSIFICATION_TABLES[version]

    @pytest.mark.parametrize("raw", ["0", "4", "-1", "two", "1.5", ""])
    def test_unknown_version_fails_at_startup(self, tmp_path, monkeypatch, raw):
        monkeypatch.setenv(ENV_VAR, raw)
        with pytest.raises(ValueError, match=ENV_VAR):
            _settings(tmp_path)

    def test_direct_construction_is_validated(self, tmp_path, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        with pytest.raises(ValueError, match="classification_version"):
            _settings(tmp_path, classification_version=9)

    def test_config_hash_follows_the_selected_version(self):
        assert pipeline.config_hash() == pipeline.config_hash(CLASSIFICATION_VERSION)
        assert pipeline.config_hash(1) != pipeline.config_hash(3)


def _repo(env, monkeypatch) -> Path:
    monkeypatch.delenv(CONTENT_DIGEST_ENV, raising=False)
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: GRAPHIFY_VERSION)
    root = env.add_repo(REPO, "example-org", "svc", "svc.example-org.dev.lo")
    env.write_registry()
    for name, text in FILES.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


def _row(report) -> dict:
    return {a["repo_id"]: a for a in report.project_actions}[REPO]


def _stored(env) -> dict:
    return load_state(env.settings().state_path)[REPO]


def _record_tables(monkeypatch) -> list:
    calls: list = []
    real = pipeline.compute_source_manifests

    def recording(path, tables, *rest):
        calls.append(tuple(tables))
        return real(path, tables, *rest)

    monkeypatch.setattr(pipeline, "compute_source_manifests", recording)
    return calls


class TestPipeline:
    def test_rollback_to_v1_and_back_never_extracts(self, env, monkeypatch):
        _repo(env, monkeypatch)
        run(env.settings(shrink_policy="count"))
        assert _stored(env)[CLASSIFICATION_VERSION_KEY] == 3
        calls = len(env.read_argv_log())

        rolled_back = run(env.settings(shrink_policy="count", classification_version=1))

        assert _row(rolled_back)["action"] == ACTION_UPDATE
        stored = _stored(env)
        assert stored[CLASSIFICATION_VERSION_KEY] == 1
        assert stored[DIGEST_RECIPE_KEY] == f"v1+graphify-ignore-{GRAPHIFY_VERSION}"
        assert all(a["cmd"] != "extract" for a in env.read_argv_log()[calls:])

        steady = run(env.settings(shrink_policy="count", classification_version=1))
        assert _row(steady)["action"] == ACTION_SKIP
        assert _stored(env)[CLASSIFICATION_VERSION_KEY] == 1

        calls = len(env.read_argv_log())
        restored = run(env.settings(shrink_policy="count"))

        assert _row(restored)["action"] == ACTION_UPDATE, "one-walk migration, not extract"
        stored = _stored(env)
        assert stored[CLASSIFICATION_VERSION_KEY] == 3
        assert stored[DIGEST_RECIPE_KEY] == f"v3+graphify-ignore-{GRAPHIFY_VERSION}"
        assert all(a["cmd"] != "extract" for a in env.read_argv_log()[calls:])
        assert _row(run(env.settings(shrink_policy="count")))["action"] == ACTION_SKIP

    def test_v1_walks_only_the_legacy_table_once_settled(self, env, monkeypatch):
        _repo(env, monkeypatch)
        run(env.settings(shrink_policy="count"))
        run(env.settings(shrink_policy="count", classification_version=1))
        calls = _record_tables(monkeypatch)

        run(env.settings(shrink_policy="count", classification_version=1, dry_run=True))

        assert calls == [(LEGACY_CLASSIFICATION_V1,)], "v1 never applies ignore rules"

    def test_rollback_walks_both_tables_in_one_pass(self, env, monkeypatch):
        _repo(env, monkeypatch)
        run(env.settings(shrink_policy="count"))
        calls = _record_tables(monkeypatch)

        run(env.settings(shrink_policy="count", classification_version=1, dry_run=True))

        assert calls == [(LEGACY_CLASSIFICATION_V1, CLASSIFICATION_TABLES[3])]

    def test_rollback_with_unknown_graphify_version_never_extracts(self, env, monkeypatch):
        """No recipe without a graphify version, so only the migration keeps it an update."""
        _repo(env, monkeypatch)
        monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: None)
        run(env.settings(shrink_policy="count"))
        calls = len(env.read_argv_log())

        rolled_back = run(env.settings(shrink_policy="count", classification_version=1))

        assert _row(rolled_back)["action"] == ACTION_UPDATE
        assert _stored(env)[CLASSIFICATION_VERSION_KEY] == 1
        assert all(a["cmd"] != "extract" for a in env.read_argv_log()[calls:])

        calls = len(env.read_argv_log())
        restored = run(env.settings(shrink_policy="count"))

        assert _row(restored)["action"] == ACTION_UPDATE
        assert _stored(env)[CLASSIFICATION_VERSION_KEY] == 3
        assert all(a["cmd"] != "extract" for a in env.read_argv_log()[calls:])


# Digests of one file snapshot under two tables: the v1 table counts more
# files as documents, so its semantic hash differs while no document changed.
V3 = SourceDigest(code_hash="c3", semantic_hash="s3", file_count=4)
V1 = SourceDigest(code_hash="c1", semantic_hash="s1", file_count=4)


def _state(digest: SourceDigest, version: int, **extra) -> dict:
    return {**digest.to_dict(), CLASSIFICATION_VERSION_KEY: version, **extra}


def _decide(prior: dict, current: SourceDigest, legacy: SourceDigest, active: int, **kw) -> str:
    """decide_action wired the way pipeline.py wires it, recipe unknown."""
    key = refusal_key(legacy, PRODUCER_UPDATE, "count")
    return decide_action(
        prior,
        current,
        True,
        refusal_key=key,
        legacy_manifest=legacy,
        recipe_changed=digest_recipe_changed(prior, None, active),
        active_version=active,
        **kw,
    )


class TestSwitchDecision:
    def test_helpers_follow_the_active_version(self):
        v3_state = _state(V3, 3)
        assert needs_classification_migration(v3_state, 1)
        assert not needs_classification_migration(v3_state, 3)
        assert not needs_classification_migration(_state(V1, 1), 1)
        assert digest_recipe("0.9.89", 1) == "v1+graphify-ignore-0.9.89"
        assert digest_recipe("0.9.89") == "v3+graphify-ignore-0.9.89"
        recorded = {**_state(V1, 1), DIGEST_RECIPE_KEY: digest_recipe("0.9.89", 1)}
        assert not digest_recipe_changed(recorded, digest_recipe("0.9.89", 1), 1)
        assert not digest_recipe_changed(recorded, digest_recipe("0.9.89"), 3), (
            "a switch is a migration, not a recipe change"
        )

    @pytest.mark.parametrize(
        ("prior", "current", "legacy", "active"),
        [(_state(V3, 3), V1, V3, 1), (_state(V1, 1), V3, V1, 3)],
        ids=["3->1", "1->3"],
    )
    def test_switch_without_document_change_updates(self, prior, current, legacy, active):
        assert _decide(prior, current, legacy, active) == ACTION_UPDATE

    @pytest.mark.parametrize(
        ("prior", "current", "legacy", "active"),
        [(_state(V3, 3), V1, V3, 1), (_state(V1, 1), V3, V1, 3)],
        ids=["3->1", "1->3"],
    )
    def test_switch_with_document_change_extracts(self, prior, current, legacy, active):
        edited_current = SourceDigest(current.code_hash, current.semantic_hash + "x", 4)
        edited_legacy = SourceDigest(legacy.code_hash, legacy.semantic_hash + "x", 4)
        assert _decide(prior, edited_current, edited_legacy, active) == ACTION_EXTRACT

    @pytest.mark.parametrize(
        ("version", "current", "legacy", "active"),
        [(3, V1, V3, 1), (1, V3, V1, 3)],
        ids=["3->1", "1->3"],
    )
    def test_held_extract_refusal_stays_held_across_a_switch(
        self, version, current, legacy, active
    ):
        """Refusal fields hold the old table's hash; the switch run must compare that."""
        edited_current = SourceDigest(current.code_hash, current.semantic_hash + "x", 4)
        edited_legacy = SourceDigest(legacy.code_hash, legacy.semantic_hash + "x", 4)
        prior = _state(
            legacy,
            version,
            refused_semantic_hash=edited_legacy.semantic_hash,
            refusal_streak=1,
        )
        assert _decide(prior, edited_current, edited_legacy, active) == ACTION_SKIP
        assert (
            _decide(prior, edited_current, edited_legacy, active, allow_shrink=True)
            == ACTION_EXTRACT
        )

    @pytest.mark.parametrize(
        ("version", "current", "legacy", "active"),
        [(3, V1, V3, 1), (1, V3, V1, 3)],
        ids=["3->1", "1->3"],
    )
    def test_held_update_refusal_stays_held_across_a_switch(self, version, current, legacy, active):
        code_edit_current = SourceDigest(current.code_hash + "x", current.semantic_hash, 4)
        code_edit_legacy = SourceDigest(legacy.code_hash + "x", legacy.semantic_hash, 4)
        prior = _state(
            legacy,
            version,
            refused_key=refusal_key(code_edit_legacy, PRODUCER_UPDATE, "count"),
            refusal_streak=1,
        )
        assert _decide(prior, code_edit_current, code_edit_legacy, active) == ACTION_SKIP
