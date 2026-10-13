from graphify_mesh.sync.cli import build_overrides, parse_args
from graphify_mesh.sync.config import Settings
from graphify_mesh.sync.state import SourceDigest
from graphify_mesh.sync.sync_project import (
    ACTION_BOOTSTRAP,
    ACTION_EXTRACT,
    ACTION_SKIP,
    ACTION_UPDATE,
    decide_action,
)


def digest(code="c1", semantic="s1"):
    return SourceDigest(code_hash=code, semantic_hash=semantic, file_count=1)


PRIOR = {"code_hash": "c1", "semantic_hash": "s1"}


def test_default_still_extracts_on_semantic_change():
    assert decide_action(PRIOR, digest(semantic="s2"), True) == ACTION_EXTRACT


def test_ast_only_turns_extract_into_update():
    assert decide_action(PRIOR, digest(semantic="s2"), True, ast_only=True) == ACTION_UPDATE


def test_ast_only_docs_only_change_still_updates():
    # Code unchanged, docs changed: UPDATE rather than SKIP so the manifest advances.
    assert (
        decide_action(PRIOR, digest(code="c1", semantic="s2"), True, ast_only=True) == ACTION_UPDATE
    )


def test_ast_only_code_change_updates():
    assert (
        decide_action(PRIOR, digest(code="c2", semantic="s1"), True, ast_only=True) == ACTION_UPDATE
    )


def test_ast_only_unchanged_skips():
    assert decide_action(PRIOR, digest(), True, ast_only=True) == ACTION_SKIP


def test_ast_only_keeps_bootstrap():
    assert decide_action(None, digest(), False, ast_only=True) == ACTION_BOOTSTRAP


def test_ast_only_ignores_refusal_hold():
    # The refusal-streak hold guards the non-deterministic extractor only; an
    # AST update on the refused digest must still run.
    refused = dict(PRIOR, refused_semantic_hash="s2", refusal_streak=1)
    assert decide_action(refused, digest(semantic="s2"), True) == ACTION_SKIP
    assert decide_action(refused, digest(semantic="s2"), True, ast_only=True) == ACTION_UPDATE


def test_settings_reads_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_AST_ONLY", "1")
    s = Settings.from_env(
        mesh_root=tmp_path, scan_roots=[tmp_path], registry_path=tmp_path / "r.json"
    )
    assert s.ast_only is True


def test_settings_default_off(monkeypatch, tmp_path):
    monkeypatch.delenv("GRAPHIFY_MESH_AST_ONLY", raising=False)
    s = Settings.from_env(
        mesh_root=tmp_path, scan_roots=[tmp_path], registry_path=tmp_path / "r.json"
    )
    assert s.ast_only is False


def test_cli_flag_sets_override():
    assert parse_args(["--once", "--ast-only"]).ast_only is True
    assert parse_args(["--once"]).ast_only is None
    assert parse_args(["--once", "--no-ast-only"]).ast_only is False


def test_cli_no_flag_leaves_env_in_charge():
    assert "ast_only" not in build_overrides(parse_args(["--once"]))
    assert build_overrides(parse_args(["--once", "--ast-only"]))["ast_only"] is True
    assert build_overrides(parse_args(["--once", "--no-ast-only"]))["ast_only"] is False


def test_cli_no_ast_only_beats_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_AST_ONLY", "1")
    overrides = build_overrides(parse_args(["--once", "--no-ast-only"]))
    s = Settings.from_env(
        mesh_root=tmp_path, scan_roots=[tmp_path], registry_path=tmp_path / "r.json", **overrides
    )
    assert s.ast_only is False
