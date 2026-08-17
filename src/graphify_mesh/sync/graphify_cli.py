"""Subprocess wrapper around the (real or fake) graphify CLI binary.

The executable is always resolved from GRAPHIFY_BIN (default "graphify") so
tests can point it at tests/graph-sync/fixtures/fake_graphify/graphify
instead of the real CLI. GRAPHIFY_NO_BACKUP=1 (C22) is always set on the
subprocess environment for update/extract/merge-graphs calls.
"""

from __future__ import annotations

import os
import shlex
import site
import subprocess
from dataclasses import dataclass
from pathlib import Path


def _isolated_home_env(staging_home: Path) -> dict:
    """HOME override for the calls below (C17: private staging dir so nothing
    touches the real ~/.graphify), plus an explicit PYTHONPATH carrying the
    *real* interpreter's site-packages.

    Overriding HOME alone breaks Python: `site.py` derives the user-site
    directory (~/.local/lib/pythonX.Y/site-packages on this platform) from
    the HOME env var at subprocess interpreter startup, before any of our
    code runs. If `graphify`/`graphifyy` is installed to the real user's
    site-packages (e.g. `pip install --user`) rather than a venv baked into
    GRAPHIFY_BIN's shebang, redirecting HOME silently makes it
    unimportable in the child process (`ModuleNotFoundError: graphify`)
    even though `graphify_bin` itself resolves fine via PATH. Carrying the
    real site-packages paths through PYTHONPATH keeps the HOME isolation
    (C17's actual goal) without breaking package resolution.
    """
    real_site_paths = [*site.getsitepackages(), site.getusersitepackages()]
    existing = os.environ.get("PYTHONPATH", "")
    pythonpath = os.pathsep.join([*real_site_paths, existing] if existing else real_site_paths)
    return {"HOME": str(staging_home), "PYTHONPATH": pythonpath}


@dataclass
class CliResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def resolve_bin_argv(graphify_bin: str) -> list[str]:
    """The ONE canonical GRAPHIFY_BIN -> argv-head resolution, shared by every
    call site (this module and backend.py's interpreter probe) so paths
    containing spaces can't be split three different ways.

    If the raw value names an existing file (covers absolute paths containing
    spaces, e.g. `/opt/my tools/graphify`), it is used verbatim as a single
    argv element; otherwise it is shlex-split so multi-word values like
    "python -m graphify" keep working.
    """
    raw = graphify_bin.strip()
    if not raw:
        return []
    if Path(raw).is_file():
        return [raw]
    return shlex.split(raw)


def _base_argv(graphify_bin: str) -> list[str]:
    return resolve_bin_argv(graphify_bin)


DEFAULT_CLI_TIMEOUT_SECONDS = 900

# Bounds for the env override. The floor keeps a typo from making every call
# time out instantly; the ceiling stays under the unit's own TimeoutStartSec so a
# single hung subprocess can never outlive the run that spawned it.
MIN_CLI_TIMEOUT_SECONDS = 60
MAX_CLI_TIMEOUT_SECONDS = 7200


def _cli_timeout() -> int:
    """Per-subprocess timeout, overridable via GRAPHIFY_MESH_CLI_TIMEOUT.

    `extract` runs an LLM pass over a whole repo, so its wall time scales with
    repo size and with how loaded the shared Ollama host is. The 900 s default
    is comfortable for small repos and too tight for large ones: a repo that
    needs longer fails with returncode 124 on every run, never advances state,
    and counts toward the stale-ratio publish gate forever (observed on
    cryengine.mage, cryengine.styleguide and agentscopex.mcp-context-gateway).

    Clamped to [MIN, MAX] with a fallback to the default on unparsable input, so
    a bad value degrades to the old behaviour rather than disabling the timeout.
    """
    raw = os.environ.get("GRAPHIFY_MESH_CLI_TIMEOUT")
    if raw is None or not raw.strip():
        return DEFAULT_CLI_TIMEOUT_SECONDS
    try:
        value = int(float(raw))
    except ValueError:
        return DEFAULT_CLI_TIMEOUT_SECONDS
    if value < MIN_CLI_TIMEOUT_SECONDS or value > MAX_CLI_TIMEOUT_SECONDS:
        return DEFAULT_CLI_TIMEOUT_SECONDS
    return value


def _run(
    argv: list[str], cwd: Path | None, env: dict | None, timeout: int | None = None
) -> CliResult:
    if timeout is None:
        timeout = _cli_timeout()
    full_env = dict(os.environ)
    full_env["GRAPHIFY_NO_BACKUP"] = "1"
    if env:
        full_env.update(env)
    try:
        proc = subprocess.run(  # noqa: S603 - structured argv, no shell; binary from operator config
            argv,
            cwd=str(cwd) if cwd else None,
            env=full_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raw = exc.stdout or ""
        partial = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        return CliResult(returncode=124, stdout=partial, stderr=f"timeout: {exc}")
    except OSError as exc:
        return CliResult(returncode=127, stdout="", stderr=f"exec error: {exc}")
    return CliResult(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def run_update(graphify_bin: str, root: Path) -> CliResult:
    """`graphify update <root>` — AST-only, code-only change (C18)."""
    return _run(_base_argv(graphify_bin) + ["update", str(root)], cwd=None, env=None)


def run_extract(graphify_bin: str, root: Path) -> CliResult:
    """`graphify extract <root> --backend ollama --force --max-concurrency 1`
    — semantic-inclusive incremental extraction (C18), also used for
    first-time bootstrap of a brand-new project."""
    argv = _base_argv(graphify_bin) + [
        "extract",
        str(root),
        "--backend",
        "ollama",
        "--force",
        "--max-concurrency",
        "1",
    ]
    return _run(argv, cwd=None, env=None)


def run_merge_graphs(
    graphify_bin: str, graph_paths: list[Path], out_path: Path, staging_home: Path
) -> CliResult:
    """`graphify merge-graphs <sorted paths> --out <path>`, run with HOME
    pointed at a private staging directory for the duration of this call only
    (C17) so nothing touches the real ~/.graphify or the tracked mesh tree.
    """
    argv = (
        _base_argv(graphify_bin)
        + ["merge-graphs"]
        + [str(p) for p in graph_paths]
        + ["--out", str(out_path)]
    )
    staging_home.mkdir(parents=True, exist_ok=True)
    return _run(argv, cwd=None, env=_isolated_home_env(staging_home))


def run_cluster_only(graphify_bin: str, target_dir: Path, staging_home: Path) -> CliResult:
    """`graphify cluster-only <target_dir> --no-viz`.

    No `--graph` override is passed: per the real CLI's output-location
    rule, when the positional path's own `graphify-out/graph.json` is used
    (no override), outputs land in `<target_dir>/graphify-out/` — exactly
    where the naming stage staged the merged graph. HOME is redirected to a
    private staging dir for the duration of this call only (C17), matching
    `run_merge_graphs`.
    """
    argv = _base_argv(graphify_bin) + ["cluster-only", str(target_dir), "--no-viz"]
    staging_home.mkdir(parents=True, exist_ok=True)
    return _run(argv, cwd=None, env=_isolated_home_env(staging_home))


def run_label(
    graphify_bin: str,
    target_dir: Path,
    staging_home: Path,
    backend: str,
    model: str,
) -> CliResult:
    """`graphify label <target_dir> --missing-only --backend <backend>
    --model <model> --no-viz`.

    `--missing-only` means only cids absent from `.graphify_labels.json`
    (or equal to the literal placeholder `Community {cid}`) get sent to the
    LLM backend — the naming stage relies on this to scope the LLM call to
    exactly the communities it deleted from the labels file after the
    sig-diff (C23/WS2 deliverable 2).
    """
    argv = _base_argv(graphify_bin) + [
        "label",
        str(target_dir),
        "--missing-only",
        "--backend",
        backend,
        "--model",
        model,
        "--no-viz",
    ]
    staging_home.mkdir(parents=True, exist_ok=True)
    return _run(argv, cwd=None, env=_isolated_home_env(staging_home))
