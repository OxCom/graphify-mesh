"""WS6 leak reaper (bin/graphify_mesh.sync/reaper.py) — pure logic tests.

Every test feeds a synthetic `ps -eo pid,ppid,args` snapshot (never spawns
or touches a real process) and, for the --kill path, injects a fake killer
so no real signal is ever sent. This mirrors the project's existing
test-injection convention for external calls (Settings.ollama_health_check
etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path

_BIN_DIR = Path(__file__).resolve().parents[2] / "bin"
if str(_BIN_DIR) not in sys.path:
    sys.path.insert(0, str(_BIN_DIR))

from graphify_mesh.sync.reaper import (  # noqa: E402 - import must follow the sys.path setup above
    ProcRow,
    find_orphan_candidates,
    parse_ps_output,
    run_reaper,
)

_LIVE_CLAUDE = "/home/user/.local/share/claude/versions/2.1.215 --session-id abc"


def test_parse_ps_output_skips_header_and_short_lines():
    text = "  PID  PPID ARGS\n  123     1 python -m graphify.serve /x/global-graph.json\ngarbage\n"
    rows = parse_ps_output(text)
    assert len(rows) == 1
    assert rows[0].pid == 123
    assert rows[0].ppid == 1


def test_parse_ps_output_without_header_line():
    text = "123 456 python -m graphify.serve /x/global-graph.json\n"
    rows = parse_ps_output(text)
    assert rows[0].pid == 123 and rows[0].ppid == 456


def test_ppid_1_graphify_serve_is_flagged_orphan():
    rows = [
        ProcRow(
            pid=100,
            ppid=1,
            args="/opt/pipx/venvs/graphifyy/bin/python -m graphify.serve /home/x/global-graph.json",
        )
    ]
    candidates = find_orphan_candidates(rows)
    assert len(candidates) == 1
    assert candidates[0].pid == 100
    assert "ppid==1" in candidates[0].reason


def test_parent_pid_no_longer_exists_is_flagged_orphan():
    rows = [ProcRow(pid=200, ppid=999, args="python -m graphify.serve /home/x/global-graph.json")]
    candidates = find_orphan_candidates(rows)
    assert len(candidates) == 1
    assert "no longer exists" in candidates[0].reason


def test_parent_is_live_claude_session_is_never_flagged():
    rows = [
        ProcRow(pid=1188889, ppid=1188858, args=_LIVE_CLAUDE),
        ProcRow(
            pid=1189714,
            ppid=1188889,
            args="/opt/pipx/venvs/graphifyy/bin/python -m graphify.serve /home/x/global-graph.json",
        ),
    ]
    candidates = find_orphan_candidates(rows)
    assert candidates == []


def test_parent_is_live_shell_is_never_flagged():
    rows = [
        ProcRow(pid=50, ppid=1, args="/bin/bash"),
        ProcRow(pid=51, ppid=50, args="python -m graphify.serve /home/x/global-graph.json"),
    ]
    candidates = find_orphan_candidates(rows)
    assert candidates == []


def test_parent_named_like_claude_but_not_claude_is_flagged_orphan():
    """Anchoring regression test: a parent whose cmdline merely CONTAINS the
    substring "claude" (e.g. "my-claude-tool --x") is NOT a live Claude Code
    session and must not shield its children from the reaper."""
    rows = [
        ProcRow(pid=20, ppid=1, args="my-claude-tool --x"),
        ProcRow(pid=21, ppid=20, args="python -m graphify.serve /home/x/global-graph.json"),
    ]
    candidates = find_orphan_candidates(rows)
    assert len(candidates) == 1
    assert candidates[0].pid == 21


def test_parent_claude_binary_path_is_live():
    rows = [
        ProcRow(pid=30, ppid=1, args="/usr/local/bin/claude --resume"),
        ProcRow(pid=31, ppid=30, args="python -m graphify.serve /home/x/global-graph.json"),
    ]
    assert find_orphan_candidates(rows) == []


def test_parent_bare_claude_command_is_live():
    rows = [
        ProcRow(pid=40, ppid=1, args="claude"),
        ProcRow(pid=41, ppid=40, args="python -m graphify.serve /home/x/global-graph.json"),
    ]
    assert find_orphan_candidates(rows) == []


def test_parent_shell_with_arguments_is_live():
    """Shell-pattern anchoring: a live shell often carries arguments
    ("bash -lc ..."); the old end-of-string anchor rejected those."""
    rows = [
        ProcRow(pid=60, ppid=1, args="/bin/bash -lc some-command"),
        ProcRow(pid=61, ppid=60, args="python -m graphify.serve /home/x/global-graph.json"),
    ]
    assert find_orphan_candidates(rows) == []


def test_parent_exists_but_is_not_shell_or_claude_is_flagged_orphan():
    rows = [
        ProcRow(pid=10, ppid=1, args="/usr/lib/systemd/systemd --user"),
        ProcRow(pid=11, ppid=10, args="python -m graphify.serve /home/x/global-graph.json"),
    ]
    candidates = find_orphan_candidates(rows)
    assert len(candidates) == 1
    assert candidates[0].pid == 11
    assert "not a shell/claude-code process" in candidates[0].reason


def test_graphify_extract_process_is_watched_too():
    rows = [
        ProcRow(
            pid=300,
            ppid=1,
            args=(
                "/opt/pipx/venvs/graphifyy/bin/python -m graphify extract "
                "/workspace/some-repo --backend ollama"
            ),
        )
    ]
    candidates = find_orphan_candidates(rows)
    assert len(candidates) == 1
    assert candidates[0].pid == 300


def test_every_pipeline_graphify_subcommand_is_watched():
    # graphify_cli.py invokes update, extract, merge-graphs, cluster-only
    # and label — a SIGKILLed pipeline can orphan any of them.
    for subcommand in ("update", "extract", "merge-graphs", "cluster-only", "label"):
        rows = [
            ProcRow(
                pid=400,
                ppid=1,
                args=f"/opt/pipx/venvs/graphifyy/bin/python -m graphify {subcommand} /x",
            )
        ]
        candidates = find_orphan_candidates(rows)
        assert len(candidates) == 1, f"graphify {subcommand} orphan not flagged"
        assert candidates[0].pid == 400


def test_parent_sync_console_script_is_live():
    """A graphify child of a running sync pipeline is active pipeline work,
    not a leak: reaping it aborts the run."""
    rows = [
        ProcRow(pid=700, ppid=1, args="/usr/bin/python3 /usr/local/bin/graphify-mesh-sync --once"),
        ProcRow(
            pid=701,
            ppid=700,
            args="/opt/pipx/venvs/graphifyy/bin/python -m graphify extract /some/repo",
        ),
    ]
    assert find_orphan_candidates(rows) == []


def test_parent_sync_module_invocation_is_live():
    rows = [
        ProcRow(pid=710, ppid=1, args="/usr/bin/python3 -m graphify_mesh.sync.cli --once"),
        ProcRow(
            pid=711,
            ppid=710,
            args="/opt/pipx/venvs/graphifyy/bin/python -m graphify extract /some/repo",
        ),
    ]
    assert find_orphan_candidates(rows) == []


def test_unrelated_processes_never_flagged():
    rows = [
        ProcRow(pid=1, ppid=0, args="/sbin/init"),
        ProcRow(pid=2, ppid=1, args="/usr/bin/python3 -m http.server"),
        ProcRow(pid=3, ppid=1, args="node /usr/local/bin/some-daemon"),
    ]
    candidates = find_orphan_candidates(rows)
    assert candidates == []


def test_live_parent_check_is_a_hard_filter_kill_never_touches_it(monkeypatch):
    """--kill must never send a signal to a live-parent process: the filter
    runs before candidates are ever built, so there is nothing for --kill to
    act on in that case."""
    fake_ps_output = (
        "PID  PPID ARGS\n"
        f"{1188889} {1188858} {_LIVE_CLAUDE}\n"
        f"{1189714} {1188889} python -m graphify.serve /home/x/global-graph.json\n"
    )

    def fake_runner(argv, capture_output, text, check):
        class _Result:
            stdout = fake_ps_output

        return _Result()

    killed_pids: list[int] = []

    def fake_killer(pid, sig):
        killed_pids.append(pid)

    candidates = run_reaper(kill=True, runner=fake_runner, killer=fake_killer)
    assert candidates == []
    assert killed_pids == []


def test_kill_sends_sigterm_only_to_confirmed_orphans():
    fake_ps_output = (
        "PID  PPID ARGS\n"
        "500 1 python -m graphify.serve /home/x/global-graph.json\n"
        "501 1 /bin/bash\n"
    )

    def fake_runner(argv, capture_output, text, check):
        class _Result:
            stdout = fake_ps_output

        return _Result()

    killed: list[tuple[int, int]] = []

    def fake_killer(pid, sig):
        killed.append((pid, sig))

    candidates = run_reaper(kill=True, runner=fake_runner, killer=fake_killer)
    assert [c.pid for c in candidates] == [500]
    assert killed == [(500, 15)]  # SIGTERM


def test_report_only_default_never_kills():
    fake_ps_output = "PID PPID ARGS\n600 1 python -m graphify.serve /x/global-graph.json\n"

    def fake_runner(argv, capture_output, text, check):
        class _Result:
            stdout = fake_ps_output

        return _Result()

    calls: list = []

    def fake_killer(pid, sig):
        calls.append(pid)

    candidates = run_reaper(kill=False, runner=fake_runner, killer=fake_killer)
    assert len(candidates) == 1
    assert calls == []  # report-only: nothing sent even though killer was provided
