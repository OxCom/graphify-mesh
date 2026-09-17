"""Leak reaper for stale graphify subprocesses (WS6).

Observed fact (2026-07-20, real machine, verified live during WS6 build):
two `graphify.serve` stdio MCP processes were running (~165-186MB RSS
each). Both were confirmed to have LIVE parents (real `claude` sessions —
see the process-tree check performed during this workstream's build), so
neither was an actual leak at the time of writing. The plan's WS5 note
about "10 stale graphify.serve processes leaked from ended sessions" was a
point-in-time observation from an earlier session, not a permanent state
of the machine — this module exists so the leak condition (a still-running
`graphify.serve`/`graphify extract` process whose parent has already
exited) can be detected and optionally cleaned up whenever it recurs,
without a human re-deriving the process-tree logic by hand each time.

Finds long-lived `graphify.serve` (per-client-session stdio MCP process,
C11) and `graphify extract` (one-shot CLI call the sync pipeline invokes
per changed project) processes whose PARENT process is dead:
  * PPID == 1 (reparented to init after the real parent exited), or
  * the PPID no longer exists in the current process table at all, or
  * the parent's own command is not a shell or a Claude Code process.

A LIVE parent (a real Claude Code session, or the shell that launched one)
means the child is a legitimately active session and must NEVER be
touched by this tool, regardless of RSS or how long it has been running.

Design notes:
  * Report-only by default. `--kill` (wired in the `reap-graphify-serve.py`
    CLI, not here) is required, explicit opt-in to send SIGTERM. Never
    SIGKILL first — cooperative shutdown matches the stdio transport's
    clean-exit-on-stdin-EOF contract for graphify.serve (now the SDK's
    `stdio_server`, wrapped by `server/stdio_guard.py`), and a leaked
    process with a dead parent has nothing left reading its stdin anyway,
    so SIGTERM is sufficient.
  * The live-parent check is a hard filter applied BEFORE any candidate is
    ever produced — `--kill` can only ever act on the already-filtered
    orphan list, never on a live-parent process.
  * Pure `ps`-based (no new dependency — matches project's "check the
    existing stack before adding a dependency" security-rule convention;
    `ps -eo pid,ppid,args` is already present on this machine).
  * `runner`/`killer` are injectable (mirrors `Settings.ollama_health_check`
    test-injection pattern elsewhere in this codebase) so tests never spawn
    or kill a real process — they feed synthetic `ps` output and record
    calls to a fake killer instead.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
from dataclasses import dataclass

# Processes this reaper watches for. Both are graphify subprocesses that
# are expected to be SHORT-LIVED relative to their owning session:
# `graphify.serve` lives exactly as long as one MCP client session (C11);
# `graphify extract`/`graphify update` live exactly as long as one sync
# pipeline invocation. A still-running instance whose parent has already
# exited is, by definition, a leak — the session or pipeline run that
# started it is gone, but the child never noticed.
WATCHED_PATTERNS = (
    re.compile(r"graphify\.serve"),
    # Every graphify subcommand the sync pipeline invokes (graphify_cli.py):
    # a SIGKILLed pipeline leaves ANY of them orphaned, not just extract.
    re.compile(r"\bgraphify\b.*\b(extract|update|merge-graphs|cluster-only|label)\b"),
)

# A parent whose full command line matches one of these is a legitimate,
# still-active owner of the child. Matched against the PARENT's `args`
# (full cmdline), not just `comm` (which `ps` truncates to 15 chars and
# would mangle e.g. the versioned `claude` binary name).
#
# Applied with `pattern.search(args)`, so every pattern must anchor its own
# token boundaries: a bare substring like r"claude" would match ANY command
# line containing "claude" (e.g. "my-claude-tool --x"), letting an attacker
# or an unlucky naming choice shield a leaked process from the reaper.
#   * (^|/)   — token starts at the beginning of the cmdline or after a path
#               separator, never mid-word ("my-claude-tool" does not match).
#   * shells: (\s|$) after the name — a live shell may carry args
#             ("/bin/bash", "bash -lc ...") but "bashful" does not match.
#   * claude: (/|\s|$) after the name — additionally allows "claude" as a
#             path COMPONENT, because the real Claude Code binary runs as
#             ".../share/claude/versions/<ver> --session-id ..." where
#             "claude" is a directory in the path, not the final token.
LIVE_PARENT_PATTERNS = (
    re.compile(r"(^|/)(bash|zsh|sh|dash|fish)(\s|$)"),
    re.compile(r"(^|/)claude(/|\s|$)"),
)


@dataclass
class ProcRow:
    pid: int
    ppid: int
    args: str


@dataclass
class ReapCandidate:
    pid: int
    ppid: int
    args: str
    reason: str


def parse_ps_output(text: str) -> list[ProcRow]:
    """Parses `ps -eo pid,ppid,args` output. Tolerates a leading header
    line (real `ps` output has one; synthetic test fixtures may omit it)."""
    rows: list[ProcRow] = []
    lines = text.splitlines()
    if lines and lines[0].strip().upper().startswith("PID"):
        lines = lines[1:]
    for line in lines:
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        rows.append(ProcRow(pid=pid, ppid=ppid, args=parts[2]))
    return rows


def is_watched(args: str) -> bool:
    return any(pattern.search(args) for pattern in WATCHED_PATTERNS)


def _parent_is_live(ppid: int, by_pid: dict[int, ProcRow]) -> bool:
    if ppid <= 1:
        return False
    parent = by_pid.get(ppid)
    if parent is None:
        return False
    return any(pattern.search(parent.args) for pattern in LIVE_PARENT_PATTERNS)


def find_orphan_candidates(rows: list[ProcRow]) -> list[ReapCandidate]:
    """Pure function over an already-captured process snapshot — the core
    logic this module tests directly, independent of any real `ps` call."""
    by_pid = {row.pid: row for row in rows}
    candidates: list[ReapCandidate] = []
    for row in rows:
        if not is_watched(row.args):
            continue
        if row.ppid <= 1:
            candidates.append(
                ReapCandidate(row.pid, row.ppid, row.args, "ppid==1 (reparented to init)")
            )
            continue
        if row.ppid not in by_pid:
            candidates.append(
                ReapCandidate(
                    row.pid, row.ppid, row.args, f"parent pid {row.ppid} no longer exists"
                )
            )
            continue
        if not _parent_is_live(row.ppid, by_pid):
            candidates.append(
                ReapCandidate(
                    row.pid,
                    row.ppid,
                    row.args,
                    f"parent pid {row.ppid} is not a shell/claude-code process",
                )
            )
    return candidates


def snapshot_ps(runner=subprocess.run) -> list[ProcRow]:
    result = runner(["ps", "-eo", "pid,ppid,args"], capture_output=True, text=True, check=True)
    return parse_ps_output(result.stdout)


def terminate(pid: int, killer=None) -> bool:
    send = killer or os.kill
    try:
        send(pid, signal.SIGTERM)
        return True
    except ProcessLookupError:
        return False


def run_reaper(kill: bool = False, runner=subprocess.run, killer=None) -> list[ReapCandidate]:
    """Full report(+optional kill) pass. `kill=True` sends SIGTERM to every
    already-filtered orphan candidate — never to anything with a live
    parent, since those never make it into the candidate list at all."""
    rows = snapshot_ps(runner)
    candidates = find_orphan_candidates(rows)
    if kill:
        for candidate in candidates:
            terminate(candidate.pid, killer)
    return candidates
