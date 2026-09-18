"""Per-subprocess CLI timeout and its env override.

`extract` runs an LLM pass whose wall time scales with repo size and with load on
the shared Ollama host. A repo that needs longer than the timeout fails with
returncode 124 on every run, never advances per-repo state, and counts toward the
stale-ratio publish gate indefinitely — so the timeout has to be tunable per
deployment without editing the package.
"""

from __future__ import annotations

import logging
import os
import subprocess

import pytest

from graphify_mesh.sync.graphify_cli import (
    CHILD_ENV_EXTRA_VAR,
    DEFAULT_CLI_TIMEOUT_SECONDS,
    MAX_CLI_TIMEOUT_SECONDS,
    MIN_CLI_TIMEOUT_SECONDS,
    _cli_timeout,
    _is_known_backend_env,
    _run,
)

ENV_VAR = "GRAPHIFY_MESH_CLI_TIMEOUT"
EXTRA_VAR = CHILD_ENV_EXTRA_VAR

# Backend variables the allowlist no longer forwards: the extraction backend is
# pinned to ollama at both call sites, so no environment can reach their code paths.
NON_OLLAMA_BACKEND_VARS = [
    "AWS_SECRET_ACCESS_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "CLAUDE_CONFIG_DIR",
]


class TestCliTimeoutResolution:
    def test_absent_env_uses_default(self, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        assert _cli_timeout() == DEFAULT_CLI_TIMEOUT_SECONDS

    @pytest.mark.parametrize("raw", ["1800", "1800.0", " 1800 "])
    def test_valid_value_honoured(self, monkeypatch, raw):
        monkeypatch.setenv(ENV_VAR, raw)
        assert _cli_timeout() == 1800

    def test_bounds_are_inclusive(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, str(MIN_CLI_TIMEOUT_SECONDS))
        assert _cli_timeout() == MIN_CLI_TIMEOUT_SECONDS
        monkeypatch.setenv(ENV_VAR, str(MAX_CLI_TIMEOUT_SECONDS))
        assert _cli_timeout() == MAX_CLI_TIMEOUT_SECONDS

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   ",
            "abc",
            "0",  # would make every call time out instantly
            "59",  # below floor
            "7201",  # above ceiling — must not outlive the unit's TimeoutStartSec
            "-1800",
        ],
    )
    def test_invalid_or_out_of_range_falls_back_to_default(self, monkeypatch, raw):
        monkeypatch.setenv(ENV_VAR, raw)
        assert _cli_timeout() == DEFAULT_CLI_TIMEOUT_SECONDS

    @pytest.mark.parametrize("raw", ["abc", "7201"])
    def test_bad_value_is_logged(self, monkeypatch, caplog, raw):
        monkeypatch.setenv(ENV_VAR, raw)
        with caplog.at_level(logging.WARNING, logger="graphify_mesh.sync"):
            assert _cli_timeout() == DEFAULT_CLI_TIMEOUT_SECONDS
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(raw in m and str(DEFAULT_CLI_TIMEOUT_SECONDS) in m for m in messages)


class TestChildEnvAllowlist:
    def _child_env(self, monkeypatch) -> dict:
        seen = {}

        def fake_run(argv, **kwargs):
            seen.update(kwargs)
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        _run(["graphify", "--version"], cwd=None, env=None)
        return seen["env"]

    def test_mesh_secrets_never_reach_the_child(self, monkeypatch):
        monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "s3cret")
        monkeypatch.setenv("GRAPHIFY_MESH_ROOT", "/var/mesh")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s3cret")
        env = self._child_env(monkeypatch)
        assert "GRAPHIFY_MESH_HTTP_TOKEN" not in env
        assert "GRAPHIFY_MESH_ROOT" not in env
        assert "AWS_SECRET_ACCESS_KEY" not in env

    def test_ollama_vars_reach_the_child(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:11434")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434/v1")
        monkeypatch.setenv("GRAPHIFY_OLLAMA_MODEL", "qwen2.5-coder:7b")
        env = self._child_env(monkeypatch)
        assert env["OLLAMA_HOST"] == "http://gpu-box:11434"
        assert env["OLLAMA_BASE_URL"] == "http://gpu-box:11434/v1"
        assert env["GRAPHIFY_OLLAMA_MODEL"] == "qwen2.5-coder:7b"
        assert "GRAPHIFY_MESH_HTTP_TOKEN" not in env

    @pytest.mark.parametrize("name", NON_OLLAMA_BACKEND_VARS)
    def test_non_ollama_backend_vars_do_not_reach_the_child(self, monkeypatch, name):
        monkeypatch.setenv(name, "value")
        assert name not in self._child_env(monkeypatch)

    @pytest.mark.parametrize("name", NON_OLLAMA_BACKEND_VARS)
    def test_non_ollama_backend_vars_reach_the_child_via_the_hatch(self, monkeypatch, name):
        monkeypatch.setenv(EXTRA_VAR, name)
        monkeypatch.setenv(name, "value")
        assert self._child_env(monkeypatch)[name] == "value"

    def test_dropped_backend_var_is_logged_at_info_without_value(self, monkeypatch, caplog):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with caplog.at_level(logging.INFO, logger="graphify_mesh.sync"):
            env = self._child_env(monkeypatch)
        assert "OPENAI_API_KEY" not in env
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert any("OPENAI_API_KEY" in m and EXTRA_VAR in m for m in messages)
        assert not any("sk-test" in m for m in messages)

    def test_no_info_log_without_a_dropped_backend_var(self, monkeypatch, caplog):
        for name in [n for n in os.environ if _is_known_backend_env(n)]:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "s3cret")
        with caplog.at_level(logging.INFO, logger="graphify_mesh.sync"):
            self._child_env(monkeypatch)
        assert not [r for r in caplog.records if r.levelno == logging.INFO]

    def test_mesh_token_never_reaches_the_child_even_via_the_hatch(self, monkeypatch):
        monkeypatch.setenv(EXTRA_VAR, "GRAPHIFY_MESH_HTTP_TOKEN")
        monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "s3cret")
        assert "GRAPHIFY_MESH_HTTP_TOKEN" not in self._child_env(monkeypatch)

    def test_dropped_names_are_logged_without_values(self, monkeypatch, caplog):
        monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "s3cret")
        with caplog.at_level(logging.DEBUG, logger="graphify_mesh.sync"):
            env = self._child_env(monkeypatch)
        assert "GRAPHIFY_MESH_HTTP_TOKEN" not in env
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("GRAPHIFY_MESH_HTTP_TOKEN" in m for m in messages)
        assert not any("s3cret" in m for m in messages)

    def test_upstream_and_runtime_vars_pass_through(self, monkeypatch):
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setenv("PYTHONPATH", "/opt/site-packages")
        monkeypatch.setenv("GRAPHIFY_BIN", "/usr/bin/graphify")
        monkeypatch.setenv("LC_ALL", "C")
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy:3128")
        env = self._child_env(monkeypatch)
        assert env["PATH"] == "/usr/bin"
        assert env["PYTHONPATH"] == "/opt/site-packages"
        assert env["GRAPHIFY_BIN"] == "/usr/bin/graphify"
        assert env["LC_ALL"] == "C"
        assert env["HTTPS_PROXY"] == "http://proxy:3128"
        assert env["GRAPHIFY_NO_BACKUP"] == "1"

    def test_extra_names_pass_through(self, monkeypatch):
        monkeypatch.setenv(EXTRA_VAR, "FAKE_GRAPHIFY_CONTROL, FAKE_GRAPHIFY_CALL_LOG ,,")
        monkeypatch.setenv("FAKE_GRAPHIFY_CONTROL", "/tmp/control.json")
        monkeypatch.setenv("FAKE_GRAPHIFY_CALL_LOG", "/tmp/call-log.jsonl")
        env = self._child_env(monkeypatch)
        assert env["FAKE_GRAPHIFY_CONTROL"] == "/tmp/control.json"
        assert env["FAKE_GRAPHIFY_CALL_LOG"] == "/tmp/call-log.jsonl"

    def test_extra_cannot_re_enable_mesh_config(self, monkeypatch, caplog):
        monkeypatch.setenv(EXTRA_VAR, "GRAPHIFY_MESH_HTTP_TOKEN")
        monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "s3cret")
        with caplog.at_level(logging.WARNING, logger="graphify_mesh.sync"):
            env = self._child_env(monkeypatch)
        assert "GRAPHIFY_MESH_HTTP_TOKEN" not in env
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("GRAPHIFY_MESH_HTTP_TOKEN" in m for m in messages)
        assert not any("s3cret" in m for m in messages)

    def test_extra_var_itself_never_reaches_the_child(self, monkeypatch):
        monkeypatch.setenv(EXTRA_VAR, "FAKE_GRAPHIFY_CONTROL")
        monkeypatch.setenv("FAKE_GRAPHIFY_CONTROL", "/tmp/control.json")
        assert EXTRA_VAR not in self._child_env(monkeypatch)

    def test_per_call_overrides_still_apply(self, monkeypatch):
        seen = {}

        def fake_run(argv, **kwargs):
            seen.update(kwargs)
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        _run(["graphify", "--version"], cwd=None, env={"HOME": "/staging"})
        assert seen["env"]["HOME"] == "/staging"


class TestRunUsesResolvedTimeout:
    def test_run_passes_env_timeout_to_subprocess(self, monkeypatch):
        seen = {}

        def fake_run(argv, **kwargs):
            seen.update(kwargs)
            return subprocess.CompletedProcess(argv, 0, "out", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setenv(ENV_VAR, "1234")
        result = _run(["graphify", "--version"], cwd=None, env=None)
        assert result.returncode == 0
        assert seen["timeout"] == 1234

    def test_explicit_timeout_still_wins(self, monkeypatch):
        seen = {}

        def fake_run(argv, **kwargs):
            seen.update(kwargs)
            return subprocess.CompletedProcess(argv, 0, "out", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setenv(ENV_VAR, "1234")
        _run(["graphify", "--version"], cwd=None, env=None, timeout=42)
        assert seen["timeout"] == 42

    def test_timeout_expiry_reports_124_with_partial_stdout(self, monkeypatch):
        def fake_run(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"], output="partial work")

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.delenv(ENV_VAR, raising=False)
        result = _run(["graphify", "extract", "/repo"], cwd=None, env=None)
        assert result.returncode == 124
        assert result.stdout == "partial work"
        assert "timeout" in result.stderr
