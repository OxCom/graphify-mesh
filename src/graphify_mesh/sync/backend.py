"""Runtime assertion of the ACTUAL graphify clustering backend (C25).

`graphify cluster-only`/`graphify label` have no CLI flag that selects
Leiden vs Louvain and no output that reports which one ran — cluster.py's
`_partition()` tries `from graspologic.partition import leiden` and falls
back to `nx.community.louvain_communities` on ImportError, purely as a
function of what's importable in whatever Python interpreter actually runs
the graphify process. This module answers that question directly, by
resolving the interpreter behind `graphify_bin` and probing it for
`graspologic` importability, then compares the answer against the pinned
constant in config.py and hard-fails (raises) on any disagreement.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from graphify_mesh.sync.config import PINNED_CLUSTERING_BACKEND
from graphify_mesh.sync.graphify_cli import resolve_bin_argv, run_probe

LEIDEN_BACKEND = "leiden"
LOUVAIN_BACKEND = "louvain"

# importlib.util.find_spec is a pure introspection call — it does not import
# (and therefore does not execute) graspologic, so this probe is cheap and
# side-effect-free even if graspologic happens to be installed.
_PROBE_CODE = (
    "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('graspologic') else 1)"
)


class BackendMismatchError(RuntimeError):
    """The graphify process backing `graphify_bin` would actually use a
    different clustering backend than the one pinned in config.py. This is
    a hard failure — the naming stage must never run against an
    unexpectedly different (and un-reviewed) clustering algorithm."""


@dataclass
class BackendCheckResult:
    backend: str
    matches_pinned: bool
    interpreter: str


def _resolve_interpreter(graphify_bin: str) -> str:
    """Best-effort resolution of the Python interpreter that will actually
    execute the graphify process for `graphify_bin`.

    `graphify_bin` may be a bare command (resolved via PATH with
    `shutil.which`) or an absolute/relative path to a script. If the
    resolved file has a `#!` shebang line, that interpreter is used
    directly — this covers both pipx console-script wrappers (real
    deployment: `#!/opt/pipx/venvs/graphifyy/bin/python3`) and the test
    fake_graphify stub (`#!/usr/bin/env python3`). Falls back to
    `sys.executable` if no shebang can be read.
    """
    bin_argv = resolve_bin_argv(graphify_bin)
    argv0 = bin_argv[0] if bin_argv else graphify_bin
    candidate = Path(argv0)
    if not candidate.is_absolute() or not candidate.exists():
        resolved = shutil.which(argv0)
        if resolved:
            candidate = Path(resolved)
    if candidate.exists():
        try:
            with candidate.open("r", encoding="utf-8", errors="ignore") as fh:
                first_line = fh.readline()
        except OSError:
            first_line = ""
        if first_line.startswith("#!"):
            shebang = first_line[2:].strip()
            if shebang:
                return shebang
    return sys.executable


def detect_actual_backend(graphify_bin: str, timeout: int = 15) -> str:
    """Runs the probe in the resolved interpreter and returns
    LEIDEN_BACKEND or LOUVAIN_BACKEND — never raises for a "louvain" result,
    only for a probe/exec failure.

    Routed through `graphify_cli.run_probe` so the probe gets the same
    allowlisted environment as every other child. A bare `subprocess.run` here
    handed an interpreter this package resolved from operator config the
    parent's whole environment, `GRAPHIFY_MESH_HTTP_TOKEN` included.

    `_PROBE_CODE` exits 0 or 1 and nothing else, so any other code is a
    failure to run the probe at all (124 timeout, 127 exec error) and raises
    instead of being read as "louvain".
    """
    interpreter = _resolve_interpreter(graphify_bin)
    argv = resolve_bin_argv(interpreter) + ["-c", _PROBE_CODE]
    result = run_probe(argv, timeout)
    if result.returncode == 0:
        return LEIDEN_BACKEND
    if result.returncode == 1:
        return LOUVAIN_BACKEND
    raise BackendMismatchError(
        f"could not probe interpreter {interpreter!r} for graphify_bin={graphify_bin!r}: "
        f"exit={result.returncode}: {result.stderr.strip()[:300]}"
    )


def assert_pinned_backend(graphify_bin: str) -> BackendCheckResult:
    """Raises BackendMismatchError if the actual backend disagrees with
    PINNED_CLUSTERING_BACKEND. Callers (naming.run_naming) must let this
    propagate uncaught — a mismatch blocks the naming stage and, by not
    being swallowed anywhere in pipeline.py, blocks publish end-to-end."""
    interpreter = _resolve_interpreter(graphify_bin)
    actual = detect_actual_backend(graphify_bin)
    matches = actual == PINNED_CLUSTERING_BACKEND
    result = BackendCheckResult(backend=actual, matches_pinned=matches, interpreter=interpreter)
    if not matches:
        raise BackendMismatchError(
            f"pinned clustering backend is {PINNED_CLUSTERING_BACKEND!r} but the graphify "
            f"process behind graphify_bin={graphify_bin!r} (interpreter={interpreter!r}) would "
            f"actually use {actual!r} (graspologic importable={actual == LEIDEN_BACKEND}); "
            "refusing to run the WS2 naming stage."
        )
    return result
