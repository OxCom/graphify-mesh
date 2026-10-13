"""GRAPHIFY_MESH_SKIP_LABELING / GRAPHIFY_MESH_RELABEL_ALL and their CLI flags.

Only an explicit flag overrides, either way: an absent flag must leave the env
value in charge, the same rule as --ast-only (test_ast_only.py).
"""

from __future__ import annotations

import pytest

from graphify_mesh.sync.cli import build_overrides, parse_args
from graphify_mesh.sync.config import Settings


def _settings(tmp_path, argv: list[str]) -> Settings:
    return Settings.from_env(
        mesh_root=tmp_path,
        scan_roots=[tmp_path],
        registry_path=tmp_path / "r.json",
        **build_overrides(parse_args(["--once", *argv])),
    )


@pytest.mark.parametrize(("value", "expected"), [(None, False), ("0", False), ("1", True)])
def test_skip_labeling_env(monkeypatch, tmp_path, value, expected):
    if value is None:
        monkeypatch.delenv("GRAPHIFY_MESH_SKIP_LABELING", raising=False)
    else:
        monkeypatch.setenv("GRAPHIFY_MESH_SKIP_LABELING", value)
    assert _settings(tmp_path, []).skip_labeling is expected


def test_skip_labeling_flag_beats_env_off(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_SKIP_LABELING", "0")
    assert _settings(tmp_path, ["--skip-labeling"]).skip_labeling is True


def test_no_skip_labeling_flag_beats_env_on(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_SKIP_LABELING", "1")
    assert _settings(tmp_path, ["--no-skip-labeling"]).skip_labeling is False


def test_skip_labeling_absent_flag_leaves_env_in_charge():
    assert parse_args(["--once"]).skip_labeling is None
    assert "skip_labeling" not in build_overrides(parse_args(["--once"]))


@pytest.mark.parametrize(("value", "expected"), [(None, False), ("0", False), ("1", True)])
def test_relabel_all_env(monkeypatch, tmp_path, value, expected):
    if value is None:
        monkeypatch.delenv("GRAPHIFY_MESH_RELABEL_ALL", raising=False)
    else:
        monkeypatch.setenv("GRAPHIFY_MESH_RELABEL_ALL", value)
    assert _settings(tmp_path, []).relabel_all is expected


def test_relabel_all_flag(monkeypatch, tmp_path):
    monkeypatch.delenv("GRAPHIFY_MESH_RELABEL_ALL", raising=False)
    assert _settings(tmp_path, ["--relabel-all"]).relabel_all is True
    assert parse_args(["--once"]).relabel_all is None
    assert "relabel_all" not in build_overrides(parse_args(["--once"]))
