"""Ignore-file size cap and git environment scrubbing in sync/state.py."""

from __future__ import annotations

import logging
import subprocess

from graphify_mesh.sync import state as state_mod
from graphify_mesh.sync.state import MAX_IGNORE_FILE_BYTES, _ignore_file_lines, _tracked_path_keys


class TestIgnoreFileCap:
    def test_cap_is_one_mib(self):
        assert MAX_IGNORE_FILE_BYTES == 1024 * 1024

    def test_file_at_the_cap_is_read(self, tmp_path):
        path = tmp_path / ".gitignore"
        body = b"*.log\n"
        path.write_bytes(body + b"#" * (MAX_IGNORE_FILE_BYTES - len(body)))
        assert _ignore_file_lines(path) == ["*.log"]

    def test_oversized_file_is_skipped_with_a_warning(self, tmp_path, caplog):
        path = tmp_path / ".gitignore"
        path.write_bytes(b"*.log\n" + b"#" * MAX_IGNORE_FILE_BYTES)
        with caplog.at_level(logging.WARNING, logger="graphify_mesh.sync"):
            assert _ignore_file_lines(path) == []
        assert any(str(path) in r.getMessage() for r in caplog.records)


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)  # noqa: S603,S607


class TestTrackedPathKeysEnv:
    def _repo(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "-q")
        (repo / "tracked.py").write_text("x = 1\n", encoding="utf-8")
        _git(repo, "add", "tracked.py")
        return repo

    def test_bogus_git_dir_in_env_is_ignored(self, tmp_path, monkeypatch):
        repo = self._repo(tmp_path)
        monkeypatch.setenv("GIT_DIR", str(tmp_path / "nowhere"))
        monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))
        monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "bogus-index"))

        result = _tracked_path_keys(repo, repo)

        assert result is not None, "GIT_* from the parent env must not reach git"
        files, _dirs = result
        assert len(files) == 1

    def test_child_env_drops_every_git_variable_and_keeps_the_rest(self, tmp_path, monkeypatch):
        repo = self._repo(tmp_path)
        monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
        monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.fsmonitor")
        monkeypatch.setenv("GIT_CONFIG_VALUE_0", "touch /tmp/pwned")
        monkeypatch.setenv("KEEP_ME", "yes")
        seen: dict = {}
        real_run = subprocess.run

        def recording(*args, **kwargs):
            seen.update(kwargs["env"])
            return real_run(*args, **kwargs)

        monkeypatch.setattr(state_mod.subprocess, "run", recording)

        _tracked_path_keys(repo, repo)

        assert seen["KEEP_ME"] == "yes"
        assert seen["GIT_OPTIONAL_LOCKS"] == "0"
        assert not [k for k in seen if k.startswith("GIT_") and k != "GIT_OPTIONAL_LOCKS"]
