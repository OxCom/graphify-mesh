"""Startup permission audit (sync/perms.py): findings, skips, and no reads.

Every file these tests check is created in tmp_path with an explicit mode.
The test container runs as a different uid than the owner of the mounted
repo files, so repo files cannot be used to assert write permission.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from graphify_mesh.sync import perms
from graphify_mesh.sync.graphify_cli import reset_log_once


@pytest.fixture(autouse=True)
def _clear_log_ledger():
    reset_log_once()
    yield
    reset_log_once()


def _write(path: Path, mode: int, text: str = "{}") -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


def _registry(tmp_path: Path, mode: int = 0o644) -> Path:
    return _write(tmp_path / "registry.json", mode, '{"repos": []}')


# ---------------------------------------------------------------------------
# Secret-bearing env files
# ---------------------------------------------------------------------------


def test_env_file_0600_is_not_a_finding(tmp_path):
    registry = _registry(tmp_path)
    _write(tmp_path / "graphify-mesh.env", 0o600, "OLLAMA_API_KEY=x\n")
    assert perms.audit_config_permissions(registry) == []


@pytest.mark.parametrize("mode", [0o644, 0o664, 0o604, 0o640])
def test_env_file_group_or_other_readable_is_a_finding(tmp_path, mode):
    registry = _registry(tmp_path)
    env = _write(tmp_path / "graphify-mesh.env", mode, "OLLAMA_API_KEY=x\n")
    findings = perms.audit_config_permissions(registry)
    assert len(findings) == 1
    assert str(env) in findings[0]
    assert f"mode {mode:04o}" in findings[0]
    assert "chmod 600" in findings[0]


def test_every_env_file_in_the_config_dir_is_audited(tmp_path):
    registry = _registry(tmp_path)
    _write(tmp_path / "a.env", 0o664, "A=1\n")
    _write(tmp_path / "b.env", 0o644, "B=1\n")
    _write(tmp_path / "c.env", 0o600, "C=1\n")
    findings = perms.audit_config_permissions(registry)
    assert len(findings) == 2
    assert str(tmp_path / "c.env") not in "\n".join(findings)


def test_non_env_files_are_ignored(tmp_path):
    registry = _registry(tmp_path)
    _write(tmp_path / "notes.txt", 0o666, "x")
    _write(tmp_path / "env", 0o666, "x")
    assert perms.audit_config_permissions(registry) == []


def test_server_call_site_skips_env_files(tmp_path):
    registry = _registry(tmp_path)
    _write(tmp_path / "graphify-mesh.env", 0o664, "OLLAMA_API_KEY=x\n")
    assert perms.audit_config_permissions(registry, check_env_files=False) == []


# ---------------------------------------------------------------------------
# Registry and manual relations
# ---------------------------------------------------------------------------


def test_world_readable_registry_is_not_a_finding(tmp_path):
    # The registry holds paths, not secrets: only writability matters.
    assert perms.audit_config_permissions(_registry(tmp_path, 0o644)) == []


@pytest.mark.parametrize("mode", [0o664, 0o666, 0o646])
def test_group_or_other_writable_registry_is_a_finding(tmp_path, mode):
    registry = _registry(tmp_path, mode)
    findings = perms.audit_config_permissions(registry)
    assert len(findings) == 1
    assert str(registry) in findings[0]
    assert f"mode {mode:04o}" in findings[0]
    assert "chmod 644" in findings[0]


def test_writable_manual_relations_is_a_finding(tmp_path):
    registry = _registry(tmp_path)
    manual = _write(tmp_path / "manual-relations.json", 0o666, "[]")
    findings = perms.audit_config_permissions(registry, manual)
    assert len(findings) == 1
    assert str(manual) in findings[0]


def test_manual_relations_at_0644_is_not_a_finding(tmp_path):
    registry = _registry(tmp_path)
    manual = _write(tmp_path / "manual-relations.json", 0o644, "[]")
    assert perms.audit_config_permissions(registry, manual) == []


# ---------------------------------------------------------------------------
# Skips: missing files, missing dirs, unstattable paths
# ---------------------------------------------------------------------------


def test_missing_registry_is_a_skip(tmp_path):
    assert perms.audit_config_permissions(tmp_path / "registry.json") == []


def test_missing_config_dir_is_a_skip(tmp_path):
    assert perms.audit_config_permissions(tmp_path / "nope" / "registry.json") == []


def test_missing_manual_relations_is_a_skip(tmp_path):
    registry = _registry(tmp_path)
    assert perms.audit_config_permissions(registry, tmp_path / "manual-relations.json") == []


def test_stat_oserror_is_a_skip(tmp_path, monkeypatch):
    registry = _registry(tmp_path, 0o666)

    def boom(self, *a, **kw):
        raise PermissionError("no stat for you")

    monkeypatch.setattr(Path, "stat", boom)
    assert perms.audit_config_permissions(registry) == []


def test_glob_oserror_is_a_skip(tmp_path, monkeypatch):
    registry = _registry(tmp_path)

    def boom(self, *a, **kw):
        raise PermissionError("no listing for you")

    monkeypatch.setattr(Path, "glob", boom)
    assert perms.audit_config_permissions(registry) == []


def test_dangling_symlink_registry_is_a_skip(tmp_path):
    link = tmp_path / "registry.json"
    link.symlink_to(tmp_path / "gone.json")
    assert perms.audit_config_permissions(link) == []


# ---------------------------------------------------------------------------
# The audit never opens a file
# ---------------------------------------------------------------------------


def test_audit_never_reads_file_contents(tmp_path, monkeypatch):
    registry = _registry(tmp_path, 0o666)
    _write(tmp_path / "graphify-mesh.env", 0o664, "OLLAMA_API_KEY=secret\n")

    def forbidden(*a, **kw):
        raise AssertionError("the permission audit must not open a file")

    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr("builtins.open", forbidden)

    findings = perms.audit_config_permissions(registry)
    assert len(findings) == 2


# ---------------------------------------------------------------------------
# Logging: warn once per process, return findings every time
# ---------------------------------------------------------------------------


def test_logs_once_but_returns_findings_every_call(tmp_path, caplog):
    registry = _registry(tmp_path, 0o664)
    with caplog.at_level(logging.WARNING, logger="graphify_mesh.sync"):
        first = perms.audit_config_permissions(registry)
        second = perms.audit_config_permissions(registry)
    assert first == second != []
    assert len([r for r in caplog.records if "insecure permissions" in r.getMessage()]) == 1


def test_never_raises_on_a_weird_tree(tmp_path):
    # A directory named like an env file, a self-referential symlink, and no
    # registry at all. Whatever the audit decides, it must not propagate.
    (tmp_path / "weird.env").mkdir()
    loop = tmp_path / "loop.env"
    loop.symlink_to(loop)
    assert isinstance(perms.audit_config_permissions(tmp_path / "registry.json"), list)


# ---------------------------------------------------------------------------
# Call site: the sync CLI audits before running the pipeline
# ---------------------------------------------------------------------------


def test_sync_cli_audits_before_run(monkeypatch, tmp_path):
    from graphify_mesh.sync import cli as cli_module
    from graphify_mesh.sync.pipeline import RunReport

    calls: list[tuple] = []

    def fake_audit(registry_path, manual_relations_path=None, **kw):
        calls.append((registry_path, manual_relations_path, tuple(kw.items())))
        return []

    def fake_run(settings):
        assert calls, "the audit must run before the pipeline"
        return RunReport(dry_run=settings.dry_run, reconciliation={})

    monkeypatch.setattr(cli_module, "audit_config_permissions", fake_audit)
    monkeypatch.setattr(cli_module, "run", fake_run)
    assert cli_module.main(["--once", "--dry-run", "--mesh-root", str(tmp_path)]) == 0
    assert len(calls) == 1
    registry_path, manual_path, _ = calls[0]
    assert registry_path.name == "registry.json"
    assert manual_path is not None and manual_path.name == "manual-relations.json"
