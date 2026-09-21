"""Subprocess wrapper around the (real or fake) graphify CLI binary.

The executable is always resolved from GRAPHIFY_BIN (default "graphify") so
tests can point it at tests/graph-sync/fixtures/fake_graphify/graphify
instead of the real CLI. GRAPHIFY_NO_BACKUP=1 (C22) is always set on the
subprocess environment for update/extract/merge-graphs calls.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import site
import subprocess
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("graphify_mesh.sync")

# Keys already logged by `_log_once`. Several of the diagnostics below describe a
# property of the process environment, not of one child: the allowlist drops the
# same names for every `update`, `extract`, `merge-graphs`, `cluster-only`,
# `label` and `--version` call a run makes, so logging per call repeats one fact
# hundreds of times. Cleared by tests via `reset_log_once`.
_LOGGED_ONCE: set[str] = set()
_LOGGED_ONCE_LOCK = threading.Lock()


def _log_once(key: str, level: int, msg: str, *args: object) -> None:
    """Emit `msg` the first time `key` is seen in this process, then never again."""
    with _LOGGED_ONCE_LOCK:
        if key in _LOGGED_ONCE:
            return
        _LOGGED_ONCE.add(key)
    log.log(level, msg, *args)


def reset_log_once() -> None:
    """Forget every `_log_once` key. For tests, which drive one process through
    many configurations and would otherwise see only the first one's warning."""
    with _LOGGED_ONCE_LOCK:
        _LOGGED_ONCE.clear()


# Names passed through to the child verbatim. An allowlist, not a denylist: this
# process runs from a systemd unit that also carries operator secrets such as
# GRAPHIFY_MESH_HTTP_TOKEN, and a denylist would leak every secret nobody thought
# to name — including ones added to the unit after this code was written.
CHILD_ENV_PASSTHROUGH = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "TMPDIR",
        "PYTHONPATH",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "NO_COLOR",
    }
)


# The child's backend configuration: the variables of the one backend family this
# pipeline can actually reach. A name belongs in this base allowlist only when a
# code path reachable without editing this repository reads it. Least privilege is
# scoped by reachability, not by vendor catalogue, so a prefix family is admitted
# for a backend in use and never for a backend held in reserve.
#
# The extraction backend is pinned to Ollama by a literal at both call sites:
# `--backend ollama` in `run_extract` below, and `backend="ollama"` in the
# `run_label` call in sync/naming.py. No setting reaches either string, so nothing
# an operator puts in the environment can select OpenAI, Anthropic, Gemini,
# DeepSeek, Azure or Bedrock, and forwarding those credentials (the AWS_ prefix
# covers the whole AWS credential chain) hands the child secrets for code it will
# never run. Widening this tuple belongs in the same commit that makes the backend
# selectable, not before it.
#
# Upstream's Ollama resolution also reads GRAPHIFY_OLLAMA_* (num-ctx, keep-alive,
# vision, parallel). Those arrive through the GRAPHIFY_*-minus-GRAPHIFY_MESH_* rule
# at the end of `_child_env_allows`, so they need no entry here. GRAPHIFY_MESH_*
# stays excluded under every rule, and GRAPHIFY_MESH_CHILD_ENV_EXTRA is how an
# operator passes any name this allowlist drops.
CHILD_BACKEND_ENV_PREFIXES = ("OLLAMA_",)


# Diagnostic only. These names and prefixes are what the allowlist used to forward
# before it narrowed to the backend actually in use. They exist for one purpose: to
# let `_run` tell an operator that a backend variable they set was dropped, instead
# of leaving them to infer it from degraded extraction output. Membership grants
# nothing — `_child_env_allows` never consults either of these.
KNOWN_BACKEND_ENV_NAMES = frozenset(
    {
        "GOOGLE_API_KEY",
        "KIMI_BASE_URL",
        "MOONSHOT_API_KEY",
        "CLAUDE_CONFIG_DIR",
        "CLAUDE_PROJECT_DIR",
        "FALKORDB_PASSWORD",
        "LOCALAPPDATA",
    }
)

KNOWN_BACKEND_ENV_PREFIXES = (
    "OPENAI_",
    "ANTHROPIC_",
    "GEMINI_",
    "DEEPSEEK_",
    "AZURE_OPENAI_",
    "AWS_",
)


def _is_known_backend_env(name: str) -> bool:
    """Whether `name` is a backend variable the old wide allowlist would have
    passed. Diagnostic use only: see the constants above."""
    return name in KNOWN_BACKEND_ENV_NAMES or name.startswith(KNOWN_BACKEND_ENV_PREFIXES)


# Operator-controlled extension to the allowlist: a comma-separated list of
# variable NAMES. An operator (or the test harness driving the fake binary) may
# have to hand the upstream binary a variable this package cannot know about,
# and the only alternative — turning the allowlist into a denylist — leaks every
# secret nobody thought to name. Names only: the values stay unlogged.
CHILD_ENV_EXTRA_VAR = "GRAPHIFY_MESH_CHILD_ENV_EXTRA"


def _child_env_extra() -> frozenset[str]:
    """Extra passthrough names from GRAPHIFY_MESH_CHILD_ENV_EXTRA, read at call
    time so a unit override takes effect without a restart.

    GRAPHIFY_MESH_* names are rejected here, and so is the variable itself: this
    package's own config (HTTP token, mesh roots, tuning) is none of the child's
    business, and letting an entry re-enable it would turn the one rule the
    allowlist exists to enforce into an opt-out.
    """
    raw = os.environ.get(CHILD_ENV_EXTRA_VAR)
    if not raw:
        return frozenset()
    names = {part.strip() for part in raw.split(",")}
    names.discard("")
    rejected = sorted(n for n in names if n.startswith("GRAPHIFY_MESH_"))
    if rejected:
        _log_once(
            f"env-extra-rejected:{','.join(rejected)}",
            logging.WARNING,
            "%s lists this package's own config names %s — refusing to pass them to the child",
            CHILD_ENV_EXTRA_VAR,
            ", ".join(rejected),
        )
    return frozenset(names - set(rejected))


def _child_env_allows(name: str, extra: frozenset[str] = frozenset()) -> bool:
    """Whether `name` may reach the child process.

    `extra` carries the operator-declared names from CHILD_ENV_EXTRA_VAR (see
    `_child_env_extra`), which is how a variable this package has never heard of
    reaches the upstream binary without weakening the base allowlist.
    """
    if name == CHILD_ENV_EXTRA_VAR:
        return False
    # GRAPHIFY_MESH_* configures this package only (tokens, mesh roots, tuning)
    # and is none of the child's business, whichever rule below would match it.
    if name.startswith("GRAPHIFY_MESH_"):
        return False
    if name in extra:
        return True
    if name in CHILD_ENV_PASSTHROUGH or name.startswith("LC_"):
        return True
    if name.startswith(CHILD_BACKEND_ENV_PREFIXES):
        return True
    # Upstream graphify reads GRAPHIFY_* for its own settings.
    return name.startswith("GRAPHIFY_")


def _isolated_home_env(staging_home: Path) -> dict:
    """HOME override for the calls below (C17: private staging dir so nothing
    touches the real ~/.graphify), plus an explicit PYTHONPATH carrying the
    *real* interpreter's site-packages.

    The isolation is not only about leaving the operator's ~/.graphify alone.
    Upstream resolves custom LLM providers from `Path.home()/".graphify"/
    "providers.json"` (graphify/llm.py:227) and its own comment there calls a
    custom provider's `base_url` an exfiltration channel that receives the full
    corpus plus the user's API key. A staging HOME means the child never sees
    that file, so nothing on disk can redirect where a parsed repository goes.
    Upstream also writes its query log and rebuild log under the real ~/.cache
    (graphify/querylog.py:30 and graphify/hooks.py:289 in 0.9.56); those follow
    HOME too.

    What this does NOT do: the child still runs as the same UID, so it can open
    any absolute path the operator can, ~/.aws/credentials included.
    Substituting HOME only redirects code that goes through `Path.home()` or
    `expanduser`. The OS-level control is the systemd unit's ProtectHome=tmpfs
    (examples/systemd/graphify-mesh-sync.service).

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


# Opt-in bubblewrap sandbox for the graphify children. Default OFF: an existing
# deployment behaves exactly as before until an operator turns it on.
CHILD_SANDBOX_VAR = "GRAPHIFY_MESH_CHILD_SANDBOX"
CHILD_SANDBOX_BIN = "bwrap"
_TRUTHY = frozenset({"1", "on", "true", "yes", "y", "enable", "enabled"})
_FALSY = frozenset({"0", "off", "false", "no", "n", "disable", "disabled", ""})

# Operator-declared extra paths to mask from the child, on top of the ones this
# module derives. Comma-separated absolute paths. A setting rather than a
# constant because the mesh layout is per-deployment: this package knows the
# directory holding its own registry and EnvironmentFile only because the caller
# names it (see `SandboxPolicy.masked_paths`), and a deployment that keeps other
# operator state near the mesh tree has to be able to add to the set without a
# code change. GRAPHIFY_MESH_* names never reach the child, so naming the
# variable here does not hand the child the list.
CHILD_MASK_PATHS_VAR = "GRAPHIFY_MESH_CHILD_MASK_PATHS"


class ChildSandboxUnavailable(RuntimeError):
    """The child sandbox is switched on but `bwrap` is not on PATH.

    Raised instead of silently launching the child unsandboxed: an operator who
    asked for the sandbox must not get a weaker guarantee than the one they
    configured.
    """


def _child_sandbox_enabled() -> bool:
    """Whether GRAPHIFY_MESH_CHILD_SANDBOX asks for the bubblewrap wrapper.

    Read at call time, like the other child-env knobs, so a unit override takes
    effect without editing code.

    Three outcomes, never two: a recognised truthy value turns the sandbox on, a
    recognised falsy value or an unset variable leaves it off, and anything else
    warns once and turns it ON. An unrecognised value means the operator wrote
    something in this variable, so they wanted the sandbox; the old
    `value in _TRUTHY` test silently produced an unsandboxed run for `enabled`,
    `Y` or `2`, which is the one outcome nobody asks for by typing a value at
    all. Failing closed here can still stop the run — with no `bwrap` installed
    the first child raises `ChildSandboxUnavailable` — and a run that stops with
    a named cause beats a run that quietly drops the isolation.
    """
    raw = os.environ.get(CHILD_SANDBOX_VAR, "").strip().lower()
    if raw in _TRUTHY:
        return True
    if raw in _FALSY:
        return False
    _log_once(
        f"sandbox-flag:{raw}",
        logging.WARNING,
        "%s=%r is not a recognised on/off value — treating it as ON. "
        "Use 1/on/true/yes or 0/off/false/no.",
        CHILD_SANDBOX_VAR,
        os.environ.get(CHILD_SANDBOX_VAR),
    )
    return True


class SandboxContainmentError(RuntimeError):
    """A path the child would get a writable bind on resolves outside the
    approved roots.

    Raised, never dropped: a write path that escapes containment is either a
    misconfigured registry or a repository steering the bind, and both mean the
    run must stop rather than hand the child a writable mount it was never
    meant to have.
    """


@dataclass(frozen=True)
class SandboxPolicy:
    """Where the sandboxed child may write, and what is masked from it.

    `containment_roots` are the run's approved roots, the mesh root and the
    run's staging root. Every
    writable bind must resolve under one of them. Empty means the caller
    declared no roots (direct callers and tests), and no containment is
    enforced — the pipeline always declares them for the two calls that parse
    untrusted repository source.

    `masked_paths` get an empty tmpfs each, after the read-only bind of `/`.
    The pipeline passes the mesh `bin/` directory here: it holds the registry
    and the systemd EnvironmentFile, and the child reads neither.
    """

    containment_roots: tuple[Path, ...] = field(default=())
    masked_paths: tuple[Path, ...] = field(default=())


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _extra_mask_paths() -> list[Path]:
    """Absolute paths from GRAPHIFY_MESH_CHILD_MASK_PATHS, read at call time."""
    raw = os.environ.get(CHILD_MASK_PATHS_VAR)
    if not raw:
        return []
    out: list[Path] = []
    for part in raw.split(","):
        candidate = part.strip()
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            _log_once(
                f"mask-relative:{candidate}",
                logging.WARNING,
                "%s entry %r is not an absolute path — ignoring it",
                CHILD_MASK_PATHS_VAR,
                candidate,
            )
            continue
        out.append(path)
    return out


def _mask_paths(policy: SandboxPolicy | None) -> list[str]:
    """Existing directories to replace with an empty tmpfs, de-duplicated.

    The runtime directory is in here and not in the caller's policy because it
    is a property of the session, not of the mesh: `--ro-bind / /` leaves
    /run/user/<uid> intact, and read-only is no protection at all for a unix
    socket — `connect()` on the ssh-agent or D-Bus socket needs no write access
    to the socket's directory.
    """
    candidates: list[Path] = []
    runtime_dir = _runtime_dir()
    if runtime_dir is not None:
        candidates.append(runtime_dir)
    if policy is not None:
        candidates.extend(policy.masked_paths)
    candidates.extend(_extra_mask_paths())
    seen: dict[str, None] = {}
    for path in candidates:
        resolved = Path(path).resolve()
        if resolved.is_dir():
            seen.setdefault(str(resolved), None)
        else:
            log.debug("child sandbox: mask target %s is not a directory — not masked", resolved)
    return list(seen)


def _runtime_dir() -> Path | None:
    """The session runtime directory to mask, or None when there is none.

    XDG_RUNTIME_DIR first, because a session can move it; /run/user/<uid> is the
    default this platform uses when the variable is unset (it is unset for a
    systemd unit that does not set it, while the directory still exists).
    """
    raw = os.environ.get("XDG_RUNTIME_DIR", "").strip() or f"/run/user/{os.getuid()}"
    path = Path(raw)
    if not path.is_absolute():
        return None
    resolved = path.resolve()
    return resolved if resolved.is_dir() else None


def _venv_root(binary: Path) -> Path | None:
    """The virtualenv root above `binary`, identified by its `pyvenv.cfg`.

    `pipx install graphifyy` (the layout docs/setup.md recommends) puts the
    console script in `<venv>/bin` and the package in `<venv>/lib/pythonX.Y/
    site-packages`, both under the home directory the tmpfs empties. Re-binding
    only the binary's own directory leaves the interpreter with no stdlib-
    relative venv and no package: the child either dies with ModuleNotFoundError
    or imports a different `graphify` copy from the inherited PYTHONPATH.
    Binding the whole venv root read-only keeps the install intact.
    """
    for parent in binary.parents:
        if (parent / "pyvenv.cfg").is_file():
            return parent
    return None


def _read_paths(argv: list[str]) -> list[str]:
    """Read-only binds needed to keep the interpreter and the graphify install
    importable once the tmpfs has emptied the home directory.

    Paths are kept as given rather than resolved: a pipx console script is
    reached through `~/.local/bin/graphify`, a symlink into the venv, and argv
    carries that literal path. Binding only the resolved target leaves the
    symlink's own directory under the tmpfs and the child fails to exec at all.
    Both the literal directory and the resolved one are bound.
    """
    candidates: list[Path] = [
        Path(p) for p in (*site.getsitepackages(), site.getusersitepackages())
    ]
    raw_binary = shutil.which(argv[0]) if argv else None
    if raw_binary:
        literal = Path(raw_binary)
        resolved = literal.resolve()
        candidates += [literal.parent, resolved.parent]
        venv_root = _venv_root(resolved)
        if venv_root is not None:
            candidates.append(venv_root)
    seen: dict[str, None] = {}
    for path in candidates:
        if not path.is_absolute():
            continue
        if path.exists():
            seen.setdefault(str(path), None)
        else:
            log.debug("child sandbox: read path %s does not exist — not bound", path)
    return list(seen)


def _checked_write_paths(
    write_paths: Sequence[Path | str], policy: SandboxPolicy | None
) -> list[str]:
    """Resolved, de-duplicated, existing write paths — after containment.

    Containment runs before the existence test and before de-duplication, so an
    escaping path can never be silently dropped for being absent. bwrap refuses
    to start when a bind source does not exist, so a contained path that is not
    there yet (a collection directory before the first run) is dropped with a
    debug line rather than turned into a launch failure.
    """
    roots = [Path(root).resolve() for root in policy.containment_roots] if policy else []
    seen: dict[str, None] = {}
    for raw in write_paths:
        if not raw:
            continue
        resolved = Path(raw).resolve()
        if roots and not any(_is_under(resolved, root) for root in roots):
            raise SandboxContainmentError(
                f"refusing to give the graphify child a writable bind on {str(raw)!r}: it "
                f"resolves to {str(resolved)!r}, outside the approved roots "
                f"[{', '.join(str(root) for root in roots)}]"
            )
        if resolved.exists():
            seen.setdefault(str(resolved), None)
        else:
            log.debug("child sandbox: write path %s does not exist yet — not bound", resolved)
    return list(seen)


def _sandbox_argv(
    argv: list[str],
    write_paths: Sequence[Path],
    policy: SandboxPolicy | None = None,
) -> list[str]:
    """`argv`, prefixed with a bubblewrap invocation when the sandbox is on.

    The whole filesystem stays readable (`--ro-bind / /`) because the extractor
    legitimately reads compilers, headers and language toolchains from anywhere.
    What the sandbox removes is the operator's credentials: the home directory
    is replaced by an empty tmpfs, so `~/.aws/credentials`, `~/.ssh` and
    `~/.config` are gone for the child even though it runs as the same UID, and
    the session runtime directory is replaced too, so the ssh-agent and D-Bus
    sockets under it cannot be connected to. Key files alone were not enough: a
    read-only bind does not stop `connect()`, and a live agent signs with keys
    whose files the child never has to read.

    `write_paths` are re-bound writable after the tmpfs and after the masks, and
    every one of them must resolve under `policy.containment_roots` first. The
    repository does not get to choose where they land: the collection directory
    comes from the registry, which `assert_registry_containment` has already
    checked, and is never re-derived from a symlink inside the scanned tree.

    The child gets its own PID namespace (`--unshare-pid`), so it cannot signal
    the sync parent or the mesh HTTP daemon, dies with the bwrap monitor
    (`--die-with-parent`), so a `subprocess.run` timeout ends the real work and
    not just the wrapper, and gets its own session (`--new-session`), so it
    cannot push characters back onto a shared terminal with TIOCSTI. `--dev`
    replaces the real `/dev` with a minimal private one: the child needs
    null/zero/random/urandom/tty and a `/dev/shm`, all of which `--dev`
    provides, and no local device — the extract backend is an HTTP service, so
    nothing here touches a GPU node.

    Network stays shared (no `--unshare-net`): the extract backend is an HTTP
    service the child must reach.
    """
    if not _child_sandbox_enabled():
        return argv
    bwrap = shutil.which(CHILD_SANDBOX_BIN)
    if bwrap is None:
        raise ChildSandboxUnavailable(
            f"{CHILD_SANDBOX_VAR} is enabled but {CHILD_SANDBOX_BIN!r} was not found on PATH. "
            f"Install bubblewrap (an OS package) or unset {CHILD_SANDBOX_VAR}."
        )
    # Resolved before anything else can raise, so a containment failure never
    # leaves a half-built argv behind.
    writable = _checked_write_paths(write_paths, policy)
    real_home = Path.home().resolve()
    wrapper = [
        bwrap,
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--unshare-pid",
        "--new-session",
        "--die-with-parent",
        "--tmpfs",
        str(real_home),
    ]
    for masked in _mask_paths(policy):
        wrapper += ["--tmpfs", masked]
    for path in _read_paths(argv):
        wrapper += ["--ro-bind", path, path]
    for path in writable:
        wrapper += ["--bind", path, path]
    return wrapper + argv


def _project_write_paths(root: Path, collection_path: Path, staging_home: Path) -> list[Path]:
    """Paths an `update`/`extract` child must be able to write.

    `collection_path` comes from the registry, not from `root/graphify-out`.
    That symlink lives inside the scanned repository, so resolving it here let
    the repository pick the bind target: discovery rejects an out-of-tree link,
    but only at scan time, and on a multi-repo run the gap between the scan and
    this launch is minutes wide. Swapping a symlink in that window needs no code
    execution, and the resulting `--bind` landed after the `--tmpfs` over the
    home directory, so a link pointing at the home directory undid the mask
    entirely.

    The repo root stays writable. `graphify extract <root>` creates
    `<root>/graphify-out` itself on a first-time bootstrap and writes its own
    artifacts beside it, so a read-only root breaks the bootstrap case outright.
    Unlike the symlink target, `root` is registry-declared and was already
    checked by `assert_registry_containment`, and it is checked again here
    against the same roots before it is bound.
    """
    return [root, collection_path, staging_home]


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
        log.warning(
            "GRAPHIFY_MESH_CLI_TIMEOUT=%r is not a number — using the default of %d s instead",
            raw,
            DEFAULT_CLI_TIMEOUT_SECONDS,
        )
        return DEFAULT_CLI_TIMEOUT_SECONDS
    if value < MIN_CLI_TIMEOUT_SECONDS or value > MAX_CLI_TIMEOUT_SECONDS:
        log.warning(
            "GRAPHIFY_MESH_CLI_TIMEOUT=%r is outside [%d, %d] — using the default of %d s instead",
            raw,
            MIN_CLI_TIMEOUT_SECONDS,
            MAX_CLI_TIMEOUT_SECONDS,
            DEFAULT_CLI_TIMEOUT_SECONDS,
        )
        return DEFAULT_CLI_TIMEOUT_SECONDS
    return value


def _run(
    argv: list[str], cwd: Path | None, env: dict | None, timeout: int | None = None
) -> CliResult:
    if timeout is None:
        timeout = _cli_timeout()
    extra = _child_env_extra()
    full_env = {k: v for k, v in os.environ.items() if _child_env_allows(k, extra)}
    # Names only, never values: an operator whose backend variable never reaches
    # the child sees it here instead of guessing from degraded extraction output.
    dropped = sorted(k for k in os.environ if k not in full_env)
    backend_dropped = [k for k in dropped if _is_known_backend_env(k)]
    if backend_dropped:
        # Louder than the DEBUG line below, and only for the case that actually
        # misleads: a backend variable is set here, so somebody meant it to take
        # effect, but upstream would fall back to its own defaults without an
        # error. Names only, and the message carries the fix. Once per process:
        # the set is a property of this environment, identical for every child a
        # run spawns, `--version` probes included.
        _log_once(
            f"backend-dropped:{','.join(backend_dropped)}",
            logging.INFO,
            "child env: backend variables set here are not passed to the graphify child "
            "(the extraction backend is pinned to ollama): %s. Declare the name in %s to "
            "pass it anyway.",
            ", ".join(backend_dropped),
            CHILD_ENV_EXTRA_VAR,
        )
    if dropped:
        _log_once(
            f"env-dropped:{','.join(dropped)}",
            logging.DEBUG,
            "child env: dropped parent variables by name: %s",
            ", ".join(dropped),
        )
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


def run_update(
    graphify_bin: str,
    root: Path,
    collection_path: Path,
    staging_home: Path,
    *,
    policy: SandboxPolicy | None = None,
) -> CliResult:
    """`graphify update <root>` — AST-only, code-only change (C18).

    Runs under a private staging HOME like the merge/cluster/label calls: this
    call parses untrusted repository source, so denying it the provider config
    described in `_isolated_home_env` matters more here than there.

    `collection_path` is the registry-declared output directory. It is passed in
    rather than derived from `root/graphify-out` so the scanned repository
    cannot choose where the writable bind lands: see `_project_write_paths`.
    """
    staging_home.mkdir(parents=True, exist_ok=True)
    argv = _sandbox_argv(
        _base_argv(graphify_bin) + ["update", str(root)],
        _project_write_paths(root, collection_path, staging_home),
        policy,
    )
    return _run(argv, cwd=None, env=_isolated_home_env(staging_home))


def run_extract(
    graphify_bin: str,
    root: Path,
    collection_path: Path,
    staging_home: Path,
    *,
    policy: SandboxPolicy | None = None,
) -> CliResult:
    """`graphify extract <root> --backend ollama --max-concurrency 1`
    — semantic-inclusive incremental extraction (C18), also used for
    first-time bootstrap of a brand-new project.

    Same private staging HOME as `run_update`, for the same reason: this is the
    call that ships repository content to an LLM backend, so the provider file
    that could redirect that backend must not be visible to it. `collection_path`
    is the registry-declared output directory, for the reason in `run_update`.
    """
    # No `--force`. It skips the semantic cache READ (graphify/cli.py:3932-3937),
    # so every run re-dispatched every semantic file to the model although the
    # cache is keyed by content hash and prompt fingerprint and was being written
    # correctly all along. This pipeline already decides update/extract/noop from
    # its own source digest (`sync_project.py`), so `--force` only duplicated that
    # decision at the cost of a full re-extraction. Measured 2026-09-21 on
    # cryengine.hub, unchanged tree: with `--force` 527 s and 56 files re-sent to
    # the model; without it 6 s, "semantic cache: 6 hit / 0 miss", 0 re-extracted,
    # same graph. Replaying unchanged files from cache also stops the model
    # randomly omitting one of them, which used to arm graphify's shrink guard and
    # fail the whole repo.
    argv = _base_argv(graphify_bin) + [
        "extract",
        str(root),
        "--backend",
        "ollama",
        "--max-concurrency",
        "1",
    ]
    staging_home.mkdir(parents=True, exist_ok=True)
    argv = _sandbox_argv(argv, _project_write_paths(root, collection_path, staging_home), policy)
    return _run(argv, cwd=None, env=_isolated_home_env(staging_home))


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
    argv = _sandbox_argv(argv, [out_path.parent, staging_home])
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
    argv = _sandbox_argv(argv, [target_dir, staging_home])
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
    argv = _sandbox_argv(argv, [target_dir, staging_home])
    return _run(argv, cwd=None, env=_isolated_home_env(staging_home))
