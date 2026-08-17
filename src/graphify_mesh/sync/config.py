"""Configuration/settings for the graphify-mesh sync pipeline.

All paths are configurable (CLI flags / GRAPHIFY_MESH_* env vars) so tests never
touch a real filesystem tree. The path defaults below are placeholders for a
typical single-host deployment; override them for your environment.
"""

from __future__ import annotations

import math
import os
import urllib.parse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Upper bound for the *_HEALTH_TIMEOUT env overrides: a health check is a
# cheap liveness probe, anything above this is certainly a typo (and would
# just turn a down service into a multi-minute pipeline hang).
HEALTH_TIMEOUT_MAX_SECONDS = 300.0

# Only plain HTTP(S) endpoints are ever legitimate LLM/embed base URLs; the
# package is published publicly so file://, gopher:// etc. must never reach
# urllib (SSRF/local-file-read surface).
ALLOWED_BASE_URL_SCHEMES = frozenset({"http", "https"})


def is_valid_http_base_url(url: str) -> bool:
    """True iff `url` parses with an http/https scheme and a non-empty host.

    Used by the WS2 naming stage and WS3 embed stage to gate their health
    checks: an invalid base URL fails the health-check path (documented
    degrade-gracefully behavior) without any request ever being attempted.
    """
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        return False
    if parsed.scheme not in ALLOWED_BASE_URL_SCHEMES:
        return False
    return bool(parsed.hostname)


def _health_timeout_from_env(var_name: str, default: float) -> float:
    """Parse a *_HEALTH_TIMEOUT env override. Must be a number, > 0 and
    <= HEALTH_TIMEOUT_MAX_SECONDS; anything else raises ValueError naming the
    env var — failing fast at startup with a clear message beats a hang (or a
    bare float() traceback) mid-pipeline."""
    raw = os.environ.get(var_name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{var_name} must be a number of seconds, got {raw!r}") from exc
    if not value > 0:
        raise ValueError(f"{var_name} must be > 0 seconds, got {raw!r}")
    if value > HEALTH_TIMEOUT_MAX_SECONDS:
        raise ValueError(f"{var_name} must be <= {HEALTH_TIMEOUT_MAX_SECONDS} seconds, got {raw!r}")
    return value


# C19: never re-litigate — see graphify_mesh.sync/__init__.py docstring for the
# full evidence citation of why `merge-graphs` (stateless) is used instead of
# `global add` (stateful, corrupts on out-of-order re-add).
GRAPHIFY_MERGE_SUBCOMMAND = "merge-graphs"

# C25: pinned WS2 clustering backend.
#
# graphify's `cluster.py:_partition()` tries `from graspologic.partition
# import leiden` and silently falls back to `nx.community.louvain_communities`
# on ImportError — there is no CLI flag that selects Leiden vs Louvain, it is
# purely a function of what's importable in the graphify process's Python
# environment. As of this writing, graspologic/leidenalg/igraph are absent
# from both the pipx venv (/opt/pipx/venvs/graphifyy) and the active
# ~/.local graphify install, so the real clustering backend today is
# Louvain. Installing graspologic/leidenalg/igraph to get Leiden is
# explicitly out of scope for this change (avoid touching global/system
# state when uncertain, per operator policy) — Louvain is accepted here
# deliberately, not silently. `graphify_mesh.sync.backend.assert_pinned_backend`
# verifies at runtime that the actual backend still matches this constant
# and hard-fails the pipeline if it ever drifts (e.g. someone installs
# graspologic later and Leiden silently takes over).
PINNED_CLUSTERING_BACKEND = "louvain"

# Naming-stage Ollama defaults. Point these at your own Ollama host via the
# GRAPHIFY_MESH_OLLAMA_* env vars. A small coder model (e.g. qwen2.5-coder:14b)
# works well for community labeling; some larger models return hollow/empty
# JSON from the OpenAI-compat endpoint, so validate any model you swap in.
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"
OLLAMA_DEFAULT_MODEL = "qwen2.5-coder:14b"
OLLAMA_DEFAULT_API_KEY = "dummy"
OLLAMA_DEFAULT_HEALTH_TIMEOUT = 3.0

# >30% of registered repos stale => refuse to publish (WS1 item 7 / plan
# Verification #5). Kept as a named constant per dict-dispatch/no-magic-number
# style rule.
STALE_PUBLISH_THRESHOLD = 0.30

# Per-repo shrink tolerance. The extract action re-derives entities with an LLM
# (`graphify extract --backend ollama`), which is not deterministic: repeated runs
# over unchanged sources differ by a few percent in node/edge counts. An absolute
# "must not shrink" guard against a jittery extractor is unsatisfiable in steady
# state — any negative wobble refuses the result, the refusal does not advance
# per-repo state, so the repo re-extracts and wobbles again on every run, and the
# repo stays permanently stale (observed 2026-07-21..08-17: publishes blocked for
# four weeks by refusals of -1.4%, -4.0% and -5.7%).
#
# A material loss still has to be caught, so the guard refuses only when counts
# drop by more than this fraction. Set GRAPHIFY_MESH_SHRINK_TOLERANCE=0 to restore
# the old absolute behaviour (appropriate if every repo uses AST-only `update`,
# which IS deterministic).
SHRINK_TOLERANCE = 0.10


def _read_shrink_tolerance() -> float:
    """Resolve the shrink tolerance from the environment, clamped to [0, 1).

    Fails safe: an unparseable or out-of-range value falls back to the default
    rather than accidentally disabling the guard (a tolerance of 1.0 would accept
    a graph collapsing to zero nodes).
    """
    raw = os.environ.get("GRAPHIFY_MESH_SHRINK_TOLERANCE")
    if raw is None or not raw.strip():
        return SHRINK_TOLERANCE
    try:
        value = float(raw)
    except ValueError:
        return SHRINK_TOLERANCE
    if value < 0.0 or value >= 1.0:
        return SHRINK_TOLERANCE
    return value


# WS3 embedding-stage defaults (C9): the NATIVE Ollama `/api/embed` endpoint,
# NOT the OpenAI-compat `/v1` surface used by OLLAMA_DEFAULT_BASE_URL above
# for the WS2 naming/labeling LLM calls — different contract, different base
# URL (no `/v1` suffix). Contract of the native /api/embed endpoint:
#   POST {base}/api/embed  {"model": "...", "input": "str-or-list-of-str"}
#   -> {"model", "embeddings": [[float, ...], ...], "total_duration",
#       "load_duration", "prompt_eval_count"}
# `qwen3-embedding:0.6b` was confirmed present in `/api/tags` and returns
# 1024-dim vectors for both single-string and batched (list) input.
# `nomic-embed-text:latest` is also present as a fallback candidate but was
# not made the default since qwen3-embedding responded correctly first and
# switching models later just needs GRAPHIFY_MESH_OLLAMA_EMBED_MODEL.
EMBED_DEFAULT_BASE_URL = "http://localhost:11434"
EMBED_DEFAULT_MODEL = "qwen3-embedding:0.6b"
EMBED_DEFAULT_DIM = 1024
EMBED_DEFAULT_HEALTH_TIMEOUT = 3.0

# Extract-backend health-probe defaults. The probe guards the per-repo
# `graphify extract --backend ollama` children (NOT the AST-only `update`
# path) so a remote-Ollama blip is classified as an infra outage instead of
# marking half the fleet `failed` and vetoing an otherwise complete publish.
# The URL deliberately defaults from OLLAMA_BASE_URL — the env var the extract
# CHILD itself resolves its endpoint from — not from
# GRAPHIFY_MESH_OLLAMA_BASE_URL, which configures this package's own naming
# stage and can legitimately point at a different host. With neither env set
# the URL is empty and the whole feature is inert (every failure stays
# `failed`, no probe is ever attempted).
EXTRACT_HEALTH_DEFAULT_TIMEOUT = 5.0

# Values of GRAPHIFY_MESH_EXTRACT_HEALTH that disable the probe entirely.
EXTRACT_HEALTH_DISABLED_VALUES = frozenset({"off", "0", "false", "no"})

# How long (hours) infra_* repos are exempt from the publish gate before they
# start counting toward the unrefreshed ratio. 0 means: no exemption, infra
# statuses count immediately.
INFRA_GRACE_DEFAULT_HOURS = 24.0


def _extract_health_url_from_env() -> str:
    """Resolve the extract-backend probe URL.

    GRAPHIFY_MESH_EXTRACT_HEALTH_URL overrides whenever it is SET — including
    set-but-empty, which yields "" and pins the feature inert (an operator's
    explicit "no probe URL" must never silently fall back to probing
    OLLAMA_BASE_URL). Only when the override var is entirely unset does the
    default apply: OLLAMA_BASE_URL (the extract child's own endpoint env —
    see the block comment above). Empty when neither is set, which makes the
    probe feature inert.
    """
    override = os.environ.get("GRAPHIFY_MESH_EXTRACT_HEALTH_URL")
    if override is not None:
        return override.strip()
    return (os.environ.get("OLLAMA_BASE_URL") or "").strip()


def _extract_health_enabled_from_env(name: str) -> bool:
    """Parse the GRAPHIFY_MESH_EXTRACT_HEALTH disable knob. Unset/empty (and
    any unrecognized value) means enabled; only the explicit disable values
    turn the probe off."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() not in EXTRACT_HEALTH_DISABLED_VALUES


def _read_infra_grace_hours(name: str, default: float) -> float:
    """Parse the GRAPHIFY_MESH_INFRA_GRACE_HOURS override. Must be a FINITE
    number >= 0 (0 is allowed and means infra statuses count toward the
    publish gate immediately; float() accepts "nan"/"inf", which would make
    the grace-window arithmetic silently nonsensical); anything else raises
    ValueError naming the env var — same fail-fast-at-startup policy as
    _health_timeout_from_env."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number of hours, got {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number of hours, got {raw!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0 hours, got {raw!r}")
    return value


# WS3 C27: keep only the last N *published* generations' embedding shards on
# disk; GC prunes older ones at publish time (see embedding.persist_generation).
KEEP_EMBEDDING_GENERATIONS = 2

# Structural generation dirs (global-graph.json + overlay + lexical-index,
# tens to 100+ MB each) had NO GC at all until this constant was added — a
# scheduled sync (e.g. hourly) accumulated one full generation per run
# forever. Keep the same count as embeddings for consistency; see
# publish.prune_old_generations, called from pipeline.py right after
# flip_current succeeds.
KEEP_STRUCTURAL_GENERATIONS = 2

# Overlay-only relation types (WS4). These must NEVER appear in the
# structural (per-repo or merged global) graph output (C5, WS1 item 7).
FORBIDDEN_OVERLAY_RELATION_TYPES = frozenset(
    {
        "semantically_similar_to",
        "similar_approach",
        "depends_on",
        "provides_api",
        "consumes_api",
    }
)

# Bounded parallelism for per-repo `graphify extract`/`update` children
# (Task 7 perf plan). Every child's RSS lands in the SAME cgroup MemoryMax as
# the parent sync process (steady-state parent peak observed ~1.9G inside a
# 4G cgroup), so this is a tuned, hard-capped setting — default 2, NEVER
# derived from len(repos). subprocess.run releases the GIL while the child
# runs, so a bounded thread pool (not a process pool) is sufficient.
EXTRACT_DEFAULT_CONCURRENCY = 2
# Hard floor: 1 = fully sequential (pre-parallelism behavior). Bad env input
# degrades to this floor rather than crashing the pipeline at startup.
EXTRACT_MIN_CONCURRENCY = 1


def _extract_concurrency_from_env(name: str, default: int) -> int:
    """Parse GRAPHIFY_MESH_EXTRACT_CONCURRENCY. Unset -> default; unparsable,
    zero, or negative -> the hard floor (EXTRACT_MIN_CONCURRENCY) rather than
    raising — a bad value here should degrade to safe sequential behavior,
    not abort the run."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return EXTRACT_MIN_CONCURRENCY
    if value < EXTRACT_MIN_CONCURRENCY:
        return EXTRACT_MIN_CONCURRENCY
    return value


# Multi-root, configurable-depth discovery (WS discovery generalization).
# depth = max nesting of the *project dir* below a scan root; depth 2 is
# today's pre-existing behavior (root/a/graphify-out and root/a/b/graphify-out).
SCAN_DEFAULT_DEPTH = 4
# Hard floor: 1 = only immediate children of a scan root are candidate
# project dirs. Bad env/CLI input degrades to this floor rather than
# raising, same policy as EXTRACT_MIN_CONCURRENCY.
SCAN_MIN_DEPTH = 1
# Hard ceiling: an unbounded depth turns one typo'd env var / CLI flag into
# an effectively-unbounded filesystem walk. 8 levels below a scan root is
# far beyond any real project layout observed so far; a request for more is
# almost certainly a mistake, not a real deployment need. Bad env/CLI input
# above this degrades to the ceiling rather than raising, same policy as the
# floor above.
SCAN_MAX_DEPTH = 8


def _scan_depth_from_env(name: str, default: int) -> int:
    """Parse GRAPHIFY_MESH_SCAN_DEPTH. Unset -> default; unparsable or below
    SCAN_MIN_DEPTH -> the hard floor; above SCAN_MAX_DEPTH -> the hard
    ceiling — degrade, never raise, same policy as
    _extract_concurrency_from_env."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return SCAN_MIN_DEPTH
    if value < SCAN_MIN_DEPTH:
        return SCAN_MIN_DEPTH
    if value > SCAN_MAX_DEPTH:
        return SCAN_MAX_DEPTH
    return value


def _split_colon_list(raw: str) -> list[str]:
    """Split a colon-separated root list, strip each segment, and drop
    empty/whitespace-only segments. Shared normalization for
    GRAPHIFY_MESH_SCAN_ROOTS and GRAPHIFY_MESH_APPROVED_ROOTS."""
    return [segment.strip() for segment in raw.split(":") if segment.strip()]


def _clean_root_args(items: Sequence[str | Path]) -> list[str]:
    """Strip and drop empty/whitespace-only elements from an explicit root
    list (e.g. repeated --scan-root values). Shares the same
    empty-value-is-unset policy as the env-var parsers below, so a stray
    `--scan-root ""` is dropped rather than silently resolving to cwd."""
    return [str(item).strip() for item in items if str(item).strip()]


def _scan_roots_from_env(scan_roots: Sequence[str | Path] | None) -> list[Path]:
    """Resolve the ordered list of scan roots.

    Precedence:
      1. explicit `scan_roots` arg (from CLI) if non-None and, after
         dropping empty/whitespace-only elements, still non-empty.
      2. GRAPHIFY_MESH_SCAN_ROOTS env: colon-split, strip each segment, drop
         empty segments; if set but splits to zero usable segments (e.g.
         ":::"), fall through to the next source.
      3. GRAPHIFY_MESH_SCAN_ROOT env (single path, back-compat); stripped,
         with whitespace-only treated as unset (falls through).
      4. Path.cwd() — the package's original, portable default.

    Every element is returned as `Path(p).resolve()`.
    """
    if scan_roots:
        cleaned_args = _clean_root_args(scan_roots)
        if cleaned_args:
            return [Path(item).resolve() for item in cleaned_args]

    raw_roots = os.environ.get("GRAPHIFY_MESH_SCAN_ROOTS")
    if raw_roots is not None:
        usable = _split_colon_list(raw_roots)
        if usable:
            return [Path(item).resolve() for item in usable]

    raw_root = os.environ.get("GRAPHIFY_MESH_SCAN_ROOT")
    if raw_root is not None:
        stripped_root = raw_root.strip()
        if stripped_root:
            return [Path(stripped_root).resolve()]

    return [Path.cwd().resolve()]


def _approved_roots_from_env(scan_roots: Sequence[Path]) -> list[Path]:
    """Resolve the ordered list of approved roots, independently of
    scan_roots.

    GRAPHIFY_MESH_APPROVED_ROOTS: same colon-split/strip/drop-empty
    normalization as GRAPHIFY_MESH_SCAN_ROOTS (see `_split_colon_list`).
    When unset, or set but splitting to zero usable segments: approved_roots
    defaults to `list(scan_roots)` — the coupling that was the only
    behavior before this env var existed.
    """
    raw = os.environ.get("GRAPHIFY_MESH_APPROVED_ROOTS")
    if raw is not None:
        usable = _split_colon_list(raw)
        if usable:
            return [Path(item).resolve() for item in usable]
    return list(scan_roots)


# File extension -> source category, for manifest-diff decisions (WS1 item 2).
# dict-dispatch instead of if/elif chains per project code style.
CODE_EXTENSIONS = frozenset(
    {
        ".py",
        ".php",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".go",
        ".rs",
        ".java",
        ".rb",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".cs",
        ".kt",
        ".swift",
        ".vue",
    }
)
SEMANTIC_EXTENSIONS = frozenset(
    {
        ".md",
        ".mdx",
        ".rst",
        ".txt",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".env.example",
    }
)
IGNORED_DIR_NAMES = frozenset(
    # Second consumer beyond file-categorization: sync/discovery.py's
    # `_iter_candidate_dirs` also prunes the filesystem walk on this same
    # set (skips descending into any dir whose name is listed here). Adding
    # a name for file-categorization reasons alone will silently change
    # which project dirs discovery can reach — check both call sites.
    {
        ".git",
        "node_modules",
        "vendor",
        "graphify-out",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
        ".idea",
        ".vscode",
    }
)


def categorize_file(path: Path) -> str:
    """Classify a source file as 'code', 'semantic', or 'ignore'.

    dict-dispatch (via frozenset membership) rather than an if/elif chain,
    per project code style rules.
    """
    suffix = path.suffix.lower()
    if suffix in CODE_EXTENSIONS:
        return "code"
    if suffix in SEMANTIC_EXTENSIONS:
        return "semantic"
    return "ignore"


@dataclass
class Settings:
    """Resolved runtime configuration for one pipeline run."""

    mesh_root: Path
    scan_roots: list[Path]
    approved_roots: list[Path]
    registry_path: Path
    graphify_bin: str = field(default_factory=lambda: os.environ.get("GRAPHIFY_BIN", "graphify"))
    stale_threshold: float = STALE_PUBLISH_THRESHOLD
    shrink_tolerance: float = field(
        default_factory=lambda: _read_shrink_tolerance(),
    )
    dry_run: bool = False
    skip_labeling: bool = False
    skip_embedding: bool = False
    allow_shrink: bool = False
    ollama_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "GRAPHIFY_MESH_OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL
        )
    )
    ollama_api_key: str = field(
        default_factory=lambda: os.environ.get(
            "GRAPHIFY_MESH_OLLAMA_API_KEY", OLLAMA_DEFAULT_API_KEY
        )
    )
    ollama_model: str = field(
        default_factory=lambda: os.environ.get("GRAPHIFY_MESH_OLLAMA_MODEL", OLLAMA_DEFAULT_MODEL)
    )
    ollama_health_timeout: float = field(
        default_factory=lambda: _health_timeout_from_env(
            "GRAPHIFY_MESH_OLLAMA_HEALTH_TIMEOUT", OLLAMA_DEFAULT_HEALTH_TIMEOUT
        )
    )
    # Test-only dependency injection: a `(base_url, api_key, timeout) -> bool`
    # callable that replaces the real network health check. Never set in
    # production; tests use this to force both the healthy and unhealthy
    # naming-stage paths deterministically without touching the network.
    ollama_health_check: Callable[[str, str, float], bool] | None = None

    # WS3 embedding-stage settings (C9: SEPARATE base URL from the /v1 LLM
    # config above — the native /api/embed contract, not OpenAI-compat).
    ollama_embed_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "GRAPHIFY_MESH_OLLAMA_EMBED_BASE_URL", EMBED_DEFAULT_BASE_URL
        )
    )
    ollama_embed_model: str = field(
        default_factory=lambda: os.environ.get(
            "GRAPHIFY_MESH_OLLAMA_EMBED_MODEL", EMBED_DEFAULT_MODEL
        )
    )
    ollama_embed_health_timeout: float = field(
        default_factory=lambda: _health_timeout_from_env(
            "GRAPHIFY_MESH_OLLAMA_EMBED_HEALTH_TIMEOUT", EMBED_DEFAULT_HEALTH_TIMEOUT
        )
    )
    # Test-only dependency injection, mirrors ollama_health_check above but
    # for the embedding stage's own (native-endpoint) health check.
    ollama_embed_health_check: Callable[[str, float], bool] | None = None
    keep_embedding_generations: int = KEEP_EMBEDDING_GENERATIONS
    keep_structural_generations: int = KEEP_STRUCTURAL_GENERATIONS

    # Extract-backend health probe (infra-outage classification). Disabled OR
    # an empty URL makes the whole feature inert: no probes, every child
    # failure stays plain `failed`, exactly the pre-probe behavior. See the
    # EXTRACT_HEALTH_DEFAULT_TIMEOUT block comment for why the URL defaults
    # from OLLAMA_BASE_URL rather than GRAPHIFY_MESH_OLLAMA_BASE_URL.
    extract_health_url: str = field(default_factory=_extract_health_url_from_env)
    extract_health_api_key: str = field(
        default_factory=lambda: os.environ.get(
            "GRAPHIFY_MESH_EXTRACT_HEALTH_API_KEY", os.environ.get("OLLAMA_API_KEY", "")
        )
    )
    extract_health_timeout: float = field(
        default_factory=lambda: _health_timeout_from_env(
            "GRAPHIFY_MESH_EXTRACT_HEALTH_TIMEOUT", EXTRACT_HEALTH_DEFAULT_TIMEOUT
        )
    )
    extract_health_enabled: bool = field(
        default_factory=lambda: _extract_health_enabled_from_env("GRAPHIFY_MESH_EXTRACT_HEALTH")
    )
    # Test-only dependency injection, mirrors ollama_health_check above but
    # for the extract-backend probe. Boolean contract: True means healthy,
    # False means outage — a bool cannot express the misconfigured state.
    extract_health_check: Callable[[str, str, float], bool] | None = None
    # Test-only dependency injection for the full tri-state probe: returns
    # one of pipeline.PROBE_HEALTHY / PROBE_OUTAGE / PROBE_MISCONFIGURED.
    # Takes precedence over extract_health_check when both are set.
    extract_health_probe: Callable[[str, str, float], str] | None = None
    # Grace window (hours) during which infra_* repos are exempt from the
    # publish gate; past it they count toward the unrefreshed ratio. 0 means
    # they count immediately.
    infra_grace_hours: float = field(
        default_factory=lambda: _read_infra_grace_hours(
            "GRAPHIFY_MESH_INFRA_GRACE_HOURS", INFRA_GRACE_DEFAULT_HOURS
        )
    )

    # Bounded parallelism for per-repo `graphify extract/update` children.
    # Each child's RSS lands in the same MemoryMax cgroup as this process,
    # so this is a tuned cap (default 2), never len(repos).
    extract_concurrency: int = field(
        default_factory=lambda: _extract_concurrency_from_env(
            "GRAPHIFY_MESH_EXTRACT_CONCURRENCY", EXTRACT_DEFAULT_CONCURRENCY
        )
    )

    # Multi-root discovery depth: max nesting of the project dir below a
    # scan root (see SCAN_DEFAULT_DEPTH docstring above).
    scan_depth: int = field(
        default_factory=lambda: _scan_depth_from_env("GRAPHIFY_MESH_SCAN_DEPTH", SCAN_DEFAULT_DEPTH)
    )

    def __post_init__(self) -> None:
        # _scan_depth_from_env already clamps the env/CLI path; direct
        # construction (tests, or any other caller building Settings(...) by
        # hand) bypasses that function entirely, so re-enforce the same
        # [SCAN_MIN_DEPTH, SCAN_MAX_DEPTH] invariant here.
        if self.scan_depth < SCAN_MIN_DEPTH:
            self.scan_depth = SCAN_MIN_DEPTH
        if self.scan_depth > SCAN_MAX_DEPTH:
            self.scan_depth = SCAN_MAX_DEPTH

    @property
    def global_dir(self) -> Path:
        return self.mesh_root / "graphify" / "global"

    @property
    def generations_dir(self) -> Path:
        return self.global_dir / "generations"

    @property
    def current_symlink(self) -> Path:
        return self.global_dir / "current"

    @property
    def status_path(self) -> Path:
        return self.global_dir / "status.json"

    @property
    def state_path(self) -> Path:
        return self.global_dir / "state" / "source-manifests.json"

    @property
    def lock_path(self) -> Path:
        return self.global_dir / ".graphify-mesh-sync.lock"

    @property
    def embeddings_dir(self) -> Path:
        """WS3: untracked embedding-index storage (id-map + per-repo shards).
        Sibling of `naming_dir`/`generations_dir` under the global dir, never
        git-tracked (see .gitignore `graphify/global/embeddings/`)."""
        return self.global_dir / "embeddings"

    @property
    def embeddings_generations_dir(self) -> Path:
        return self.embeddings_dir / "generations"

    @property
    def embeddings_current_symlink(self) -> Path:
        """Mirrors `global_dir/current` (publish.flip_current): flipped
        atomically, in the same publish step, to point at the embedding
        shards for the generation that was just published."""
        return self.embeddings_dir / "current"

    @property
    def manual_relations_path(self) -> Path:
        """WS4: human-declared cross-project overlay edges the sync engine
        cannot infer. Derived from mesh_root like every other Settings path, so
        tests get an isolated file for free via their fake mesh_root."""
        return self.mesh_root / "bin" / "manual-relations.json"

    @property
    def manual_relations_schema_path(self) -> Path:
        return self.mesh_root / "bin" / "manual-relations.schema.json"

    @property
    def naming_dir(self) -> Path:
        """Persistent WS2 naming-stage workspace (graphify-out/graph.json,
        .graphify_labels.json, .graphify_labels.json.sig live here). Must
        survive across pipeline runs — unlike the per-run staging tempdir —
        so sig-gated label reuse (C23) actually has something to compare
        against on the next run. In dry-run mode the caller redirects this
        into the ephemeral staging root instead (mirrors lock_path's
        dry-run redirection) so nothing touches the real mesh tree."""
        return self.global_dir / "naming"

    @classmethod
    def from_env(
        cls,
        mesh_root: Path | None = None,
        scan_roots: Sequence[str | Path] | None = None,
        registry_path: Path | None = None,
        **overrides,
    ) -> Settings:
        # No machine-specific defaults: mesh_root and scan_roots both default
        # to the current working directory so the package is portable. Set
        # them explicitly (CLI flags or GRAPHIFY_MESH_ROOT /
        # GRAPHIFY_MESH_SCAN_ROOTS / GRAPHIFY_MESH_SCAN_ROOT) for a real
        # deployment — /var/www/ or any other host-specific path belongs in
        # deployment config (systemd unit, wrapper script, etc.), not here.
        #
        # approved_roots defaults to scan_roots (the coupling this package
        # always had) but can be set independently via
        # GRAPHIFY_MESH_APPROVED_ROOTS when a deployment needs to scan a
        # narrower tree than it trusts discovered symlink targets to resolve
        # into.
        resolved_mesh_root = Path(
            mesh_root or os.environ.get("GRAPHIFY_MESH_ROOT") or Path.cwd()
        ).resolve()
        resolved_scan_roots = _scan_roots_from_env(scan_roots)
        resolved_approved_roots = _approved_roots_from_env(resolved_scan_roots)
        resolved_registry = Path(
            registry_path
            or os.environ.get(
                "GRAPHIFY_MESH_REGISTRY", str(resolved_mesh_root / "bin" / "registry.json")
            )
        ).resolve()
        return cls(
            mesh_root=resolved_mesh_root,
            scan_roots=resolved_scan_roots,
            approved_roots=resolved_approved_roots,
            registry_path=resolved_registry,
            **overrides,
        )
