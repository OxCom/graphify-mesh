"""Subprocess tests for examples/hooks/graphify-mesh-cwd.py.

Runs the hook script as a real subprocess, feeding it JSON on stdin, the
same way Claude Code invokes a PreToolUse hook.
"""

import json
import subprocess
import sys
from pathlib import Path

HOOK_PATH = Path(__file__).resolve().parents[2] / "examples" / "hooks" / "graphify-mesh-cwd.py"


def run_hook(payload_text: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(HOOK_PATH)],
        input=payload_text,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )


def test_adds_cwd_when_missing():
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": "/repo/current-project",
        "tool_name": "mcp__graphify-mesh__search",
        "tool_input": {"query": "foo", "scope": "current"},
    }
    result = run_hook(json.dumps(payload))

    assert result.returncode == 0
    out = json.loads(result.stdout)
    updated = out["hookSpecificOutput"]["updatedInput"]
    assert updated["cwd"] == "/repo/current-project"
    assert updated["query"] == "foo"
    assert out["hookSpecificOutput"]["hookEventName"] == "PreToolUse"


def test_leaves_existing_cwd_untouched():
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": "/repo/session-dir",
        "tool_name": "mcp__graphify-mesh__context_pack",
        "tool_input": {"query": "bar", "cwd": "/repo/other-project"},
    }
    result = run_hook(json.dumps(payload))

    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_other_tool_produces_no_output():
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": "/repo/session-dir",
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
    }
    result = run_hook(json.dumps(payload))

    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_malformed_payload_produces_no_output():
    result = run_hook("not json at all {{{")

    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_missing_tool_input_produces_no_output():
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": "/repo/session-dir",
        "tool_name": "mcp__graphify-mesh__search",
    }
    result = run_hook(json.dumps(payload))

    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_relative_directory_never_emitted(monkeypatch, tmp_path):
    # No top-level `cwd`, no CLAUDE_PROJECT_DIR: falls back to the process's
    # own working directory, which must still come out absolute.
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "mcp__graphify-mesh__search",
        "tool_input": {"query": "baz"},
    }
    env = {"PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        [sys.executable, str(HOOK_PATH)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
        timeout=10,
    )

    assert result.returncode == 0
    out = json.loads(result.stdout)
    emitted_cwd = out["hookSpecificOutput"]["updatedInput"]["cwd"]
    assert Path(emitted_cwd).is_absolute()


# --- a present-but-invalid `cwd` belongs to the server's validation ---
#
# Filling one in substitutes the session directory for what the caller asked
# for: the call then succeeds against a DIFFERENT scope instead of failing,
# which is the silent wrong answer the per-call `cwd` contract exists to
# prevent. The key being ABSENT is the only case the hook may fill.


def _payload(cwd_value):
    return {
        "hook_event_name": "PreToolUse",
        "cwd": "/repo/session",
        "tool_name": "mcp__graphify-mesh__search",
        "tool_input": {"query": "foo", "cwd": cwd_value},
    }


def test_empty_string_cwd_is_left_for_the_server_to_reject():
    result = run_hook(json.dumps(_payload("")))
    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_whitespace_cwd_is_left_for_the_server_to_reject():
    result = run_hook(json.dumps(_payload("   ")))
    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_null_cwd_is_left_for_the_server_to_reject():
    result = run_hook(json.dumps(_payload(None)))
    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_numeric_cwd_is_left_for_the_server_to_reject():
    result = run_hook(json.dumps(_payload(42)))
    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_still_fails_open_on_an_unexpected_payload():
    """Anything the hook cannot make sense of must produce no output and
    exit 0 — a hook bug may never block a tool call."""
    for payload in ('{"tool_name": 5}', '{"tool_input": "not a dict"}', "[]", "null", ""):
        result = run_hook(payload)
        assert result.returncode == 0
        assert result.stdout.strip() == ""
