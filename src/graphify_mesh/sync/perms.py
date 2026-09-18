"""Startup permission audit for this package's own configuration files.

Two kinds of file in the mesh configuration directory carry more authority
than their mode usually reflects:

* A secret-bearing ``*.env`` file next to ``registry.json``. The documented
  layout (``examples/systemd/graphify-mesh-sync.env.example``) puts the
  backend API keys there, and systemd never tells the process which file it
  read its ``EnvironmentFile`` from, so the path cannot be learned at
  runtime — the directory layout is the only thing this package can audit.
  Anyone who can read the file has the keys.
* ``registry.json`` and ``manual-relations.json``. The registry declares
  ``repos[].root``, ``repos[].collection_path``, ``disabled`` and
  ``external_roots``, which decide which directories get stat-walked and
  handed to ``graphify extract``, and which also serve as the child
  sandbox's containment roots and writable bind targets. Anyone who can
  write the registry can redirect extraction and steer a writable bind.
  World-*readable* is not a finding here: the registry holds paths, not
  secrets.

This module warns and nothing more. It never raises, never blocks a run and
never changes a mode: the files belong to the operator, and their
permissions are the operator's decision. A missing file, an unreadable
directory or an ``OSError`` from ``stat`` is a skip, not a finding.

The audit only ever looks at the mesh configuration directory. It stats
nothing inside a scanned repository, and it never opens a file — mode bits
come from ``stat`` alone, so a secret's contents are never read.
"""

from __future__ import annotations

import logging
import stat
from pathlib import Path

from graphify_mesh.sync.graphify_cli import _log_once

log = logging.getLogger("graphify_mesh.sync")

# Any permission bit held by group or other. A secret-bearing env file is
# documented as mode 0600, so anything wider than owner-only is a finding.
SECRET_LOOSE_MASK = 0o077

# Write permission held by group or other. The registry is a trust anchor for
# path containment, so only its writability matters, not its readability.
TRUST_LOOSE_MASK = 0o022


def _mode_of(path: Path) -> int | None:
    """Permission bits of `path`, or None when it cannot be stat'ed.

    Follows symlinks deliberately: the mode that decides who can rewrite the
    configuration is the target's, not the link's.
    """
    try:
        return stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return None


def _audit_secret_env_files(config_dir: Path) -> list[str]:
    try:
        candidates = sorted(p for p in config_dir.glob("*.env"))
    except OSError:
        return []

    findings: list[str] = []
    for path in candidates:
        mode = _mode_of(path)
        if mode is None or not mode & SECRET_LOOSE_MASK:
            continue
        findings.append(
            f"insecure permissions on {path} (mode {mode:04o}): group or other can read a "
            f"secret-bearing env file. Fix with: chmod 600 {path}"
        )
    return findings


def _audit_trusted_file(path: Path) -> list[str]:
    mode = _mode_of(path)
    if mode is None or not mode & TRUST_LOOSE_MASK:
        return []
    return [
        f"insecure permissions on {path} (mode {mode:04o}): group or other can write it, and "
        f"this file decides which directories are extracted and bind-mounted writable into the "
        f"child. Fix with: chmod 644 {path}"
    ]


def audit_config_permissions(
    registry_path: Path,
    manual_relations_path: Path | None = None,
    *,
    check_env_files: bool = True,
) -> list[str]:
    """Warn once per process about loose modes on this package's config files.

    Returns the findings in the order they were logged, so a caller (or a
    test) can inspect them. The return value is independent of the
    once-per-process log ledger: a second call reports the same findings
    without logging them again.

    `check_env_files` is off for the MCP server, whose own secret lives in a
    file outside the configuration directory that this package cannot locate.
    """
    findings: list[str] = []
    if check_env_files:
        findings.extend(_audit_secret_env_files(registry_path.parent))
    findings.extend(_audit_trusted_file(registry_path))
    if manual_relations_path is not None:
        findings.extend(_audit_trusted_file(manual_relations_path))

    for message in findings:
        _log_once(f"perms:{message}", logging.WARNING, "%s", message)
    return findings
