"""Per-subprocess CLI timeout and its env override.

`extract` runs an LLM pass whose wall time scales with repo size and with load on
the shared Ollama host. A repo that needs longer than the timeout fails with
returncode 124 on every run, never advances per-repo state, and counts toward the
stale-ratio publish gate indefinitely — so the timeout has to be tunable per
deployment without editing the package.
"""

from __future__ import annotations

import subprocess

import pytest

from graphify_mesh.sync.graphify_cli import (
    DEFAULT_CLI_TIMEOUT_SECONDS,
    MAX_CLI_TIMEOUT_SECONDS,
    MIN_CLI_TIMEOUT_SECONDS,
    _cli_timeout,
    _run,
)

ENV_VAR = "GRAPHIFY_MESH_CLI_TIMEOUT"


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
