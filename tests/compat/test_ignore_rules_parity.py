"""Parity check: the mesh's ignore predicate against upstream ignored_predicate.

state.GraphifyIgnoreRules rebuilds `graphify.detect.ignored_predicate` from its
private helpers, because the upstream function runs `git ls-files` with the
scanned repo's own config (fsmonitor) and reads ignore files through symlinks.
This lane notices a renamed helper or a changed rule. On a fixture where those
hazards are absent, both must give the same answer for every path.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from graphify_mesh.sync.state import GraphifyIgnoreRules, IgnoreRules

detect = pytest.importorskip("graphify.detect")

HELPERS = (
    "_SKIP_FILES",
    "_find_vcs_root",
    "_git_info_exclude",
    "_is_installed_graphify_file",
    "_is_noise_dir",
    "_is_regular_file",
    "_is_scan_ignored",
    "_linked_worktree_main_root",
    "_parse_gitignore_line",
    "_path_identity",
    "_resolves_under_root",
    "ignored_predicate",
)

FILES = {
    ".gitignore": "var/cache/\nCLAUDE.md\n*.log\n!keep.log\n",
    ".graphifyignore": "private_notes/\n",
    "src/app.py": "",
    "src/.gitignore": "generated_*.py\n",
    "src/generated_client.py": "",
    "src/keep.log": "",
    "src/debug.log": "",
    "README.md": "",
    "CLAUDE.md": "",
    "var/cache/Container.php": "",
    "var/cache/tracked.md": "",
    "private_notes/plan.md": "",
    "excluded/x.py": "",
    "node_modules/pkg/index.js": "",
    "package-lock.json": "",
    "lib/coverage/lcov.info": "",
    "lib/coverage/report.py": "",
}
EXCLUDES = ("excluded/",)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@example.org", "-c", "user.name=t", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def test_upstream_helpers_exist():
    missing = [name for name in HELPERS if not hasattr(detect, name)]
    assert not missing, f"graphify.detect no longer has {missing}"


@pytest.mark.parametrize("gitignore", [True, False])
def test_same_answer_as_upstream(tmp_path, monkeypatch, gitignore):
    # Upstream also reads the user-wide git excludes file; the mesh does not.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home" / ".config"))
    root = tmp_path / "repo"
    for rel, text in FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(root, "init", "-q", ".")
    _git(root, "add", ".gitignore", ".graphifyignore", "src/app.py", "README.md")
    _git(root, "add", "-f", "var/cache/tracked.md")
    _git(root, "commit", "-qm", "init")

    upstream = detect.ignored_predicate(root, extra_excludes=list(EXCLUDES), gitignore=gitignore)
    mesh = GraphifyIgnoreRules(root, IgnoreRules(EXCLUDES, gitignore))

    differ = {
        rel: (upstream(root / rel), mesh.ignores_file(root / rel))
        for rel in FILES
        if upstream(root / rel) != mesh.ignores_file(root / rel)
    }
    assert not differ, f"(upstream, mesh) per path: {differ}"
    # The fixture must exercise both outcomes, or parity proves nothing.
    answers = {mesh.ignores_file(root / rel) for rel in FILES}
    assert answers == {True, False}
