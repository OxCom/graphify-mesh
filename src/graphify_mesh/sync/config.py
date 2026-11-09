"""Configuration/settings for the graphify-mesh sync pipeline.

All paths are configurable (CLI flags / GRAPHIFY_MESH_* env vars) so tests never
touch a real filesystem tree. The path defaults below are placeholders for a
typical single-host deployment; override them for your environment.
"""

from __future__ import annotations

import logging
import math
import os
import re
import urllib.parse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Upper bound for the *_HEALTH_TIMEOUT env overrides: a health check is a
# cheap liveness probe, anything above this is certainly a typo (and would
# just turn a down service into a multi-minute pipeline hang).
HEALTH_TIMEOUT_MAX_SECONDS = 300.0

# Ceiling for one in-process labeling call (GRAPHIFY_API_TIMEOUT, which the
# naming stage exports for graphify's own HTTP client). One hour: the whole
# pipeline's systemd budget is shorter than upstream's 600 s default times the
# ~11 sequential batches a 1000-community graph produces.
API_TIMEOUT_MAX_SECONDS = 3600.0

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


def _api_timeout_from_env(var_name: str, default: float) -> float:
    """Parse the labeling-call timeout override. Same fail-fast-at-startup
    policy as `_health_timeout_from_env`, but a far larger ceiling: this bounds
    one LLM completion batch, not a `/models` probe."""
    raw = os.environ.get(var_name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{var_name} must be a number of seconds, got {raw!r}") from exc
    if not value > 0:
        raise ValueError(f"{var_name} must be > 0 seconds, got {raw!r}")
    if value > API_TIMEOUT_MAX_SECONDS:
        raise ValueError(f"{var_name} must be <= {API_TIMEOUT_MAX_SECONDS} seconds, got {raw!r}")
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
# Per-call ceiling for in-process community labeling. Upstream's own default is
# 600 s with zero retries for ollama, and the naming stage issues one call per
# batch of communities, so a wedged backend ("healthy /models, hung
# completions") could outlast the pipeline's whole systemd budget.
OLLAMA_DEFAULT_API_TIMEOUT = 180.0

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


# How the per-repo shrink guard judges a smaller graph (2026-10-13):
#   attributed - ask `graphify-shrink-check` which losses follow from deleted or
#                changed source files and refuse only unexplained loss;
#   connected  - same tool, its connectivity-based policy;
#   count      - this package's own node/edge count floor (_classify_shrink).
# Count-only refused legitimate deletions and accepted resolver-edge loss inside
# the tolerance. The two tool policies fall back to `count` when the binary is
# absent or errors, so the mesh keeps working without graphify-php installed.
SHRINK_POLICIES = frozenset({"attributed", "connected", "count"})
SHRINK_POLICY_DEFAULT = "attributed"


def _read_shrink_policy() -> str:
    """GRAPHIFY_MESH_SHRINK_POLICY, or the default for an unset or unknown value.

    An unknown value warns instead of raising: every policy still refuses a
    material loss, so a typo costs precision, never the guard itself.
    """
    raw = (os.environ.get("GRAPHIFY_MESH_SHRINK_POLICY") or "").strip().lower()
    if not raw:
        return SHRINK_POLICY_DEFAULT
    if raw not in SHRINK_POLICIES:
        logging.getLogger("graphify_mesh.sync").warning(
            "GRAPHIFY_MESH_SHRINK_POLICY=%r is not one of %s — using %r",
            raw,
            sorted(SHRINK_POLICIES),
            SHRINK_POLICY_DEFAULT,
        )
        return SHRINK_POLICY_DEFAULT
    return raw


# "repo" clusters each repo's subgraph on its own, so an edit in one repo cannot
# reshape another repo's communities; "global" is the old whole-graph clustering,
# kept as the rollback switch. Recorded in the naming recipe, so a flip relabels once.
CLUSTER_SCOPES = frozenset({"repo", "global"})
CLUSTER_SCOPE_DEFAULT = "repo"


def _read_cluster_scope() -> str:
    """GRAPHIFY_MESH_CLUSTER_SCOPE, or the default for an unset or unknown value.

    An unknown value warns instead of raising: both scopes publish a valid graph.
    """
    raw = (os.environ.get("GRAPHIFY_MESH_CLUSTER_SCOPE") or "").strip().lower()
    if not raw:
        return CLUSTER_SCOPE_DEFAULT
    if raw not in CLUSTER_SCOPES:
        logging.getLogger("graphify_mesh.sync").warning(
            "GRAPHIFY_MESH_CLUSTER_SCOPE=%r is not one of %s — using %r",
            raw,
            sorted(CLUSTER_SCOPES),
            CLUSTER_SCOPE_DEFAULT,
        )
        return CLUSTER_SCOPE_DEFAULT
    return raw


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
# Bounds one /api/embed batch request. Measured 2026-11-06 on the real backend:
# 34.7 s for 1 input, 78 s for 128, so the old 30 s failed the stage every run.
EMBED_DEFAULT_REQUEST_TIMEOUT = 120.0

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

# Upstream graphify's backup_if_protected copies graph.json & co. into
# `<collection>/<YYYY-MM-DD>/` before overwriting a labeled graph, and nothing
# ever pruned them (cem/hub reached 61 dirs, ~538M). The publish step deletes
# dated dirs older than this many days, always keeping each repo's newest;
# 0 disables rotation. See publish.prune_dated_snapshots.
KEEP_SNAPSHOT_DAYS = 5


def _keep_snapshot_days_from_env(name: str, default: int) -> int:
    """Parse GRAPHIFY_MESH_KEEP_SNAPSHOT_DAYS. Unset -> default; unparsable
    or negative -> default; 0 -> rotation disabled. Degrade, never raise —
    a bad retention value must not abort the run."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value < 0:
        return default
    return value


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
#
# Table v2 follows upstream graphify routing (detect.classify_file), unevenly:
# the code side equals upstream CODE_EXTENSIONS as of graphify 0.9.103
# (lowercased, hand-copied, never imported at runtime so a pip upgrade cannot
# change the digest); the semantic side is a deliberate subset of the upstream
# document/paper/image routing, with the rest listed in DELIBERATELY_EXCLUDED_DOC.
# tests/compat/test_classification_drift.py compares both sides against the
# installed graphify and fails on any upstream change.
# Table v3 keeps the v2 categories and also drops every file graphify's own
# ignore rules exclude (.gitignore, .graphifyignore, persisted --exclude, noise
# dirs; see state.graphify_ignore_rules). Those rules come from the installed
# graphify at runtime, so the per-repo digest recipe records its version.
# v3 also drops files graphify skips as likely secrets (is_sensitive_path, a
# hand copy of detect._is_sensitive checked by tests/compat).
# Bump CLASSIFICATION_VERSION on any change; decide_action migrates per repo.
CLASSIFICATION_VERSION = 3
CODE_EXTENSIONS = frozenset(
    {
        ".asd",
        ".astro",
        ".bash",
        ".c",
        ".cbl",
        ".cc",
        ".cjs",
        ".cl",
        ".cls",
        ".cob",
        ".cobol",
        ".cpp",
        ".cpy",
        ".cs",
        ".cshtml",
        ".csproj",
        ".cts",
        ".cu",
        ".cuh",
        ".cxx",
        ".dart",
        ".dfm",
        ".dm",
        ".dme",
        ".dmf",
        ".dmi",
        ".dmm",
        ".dpk",
        ".dpr",
        ".ejs",
        ".erl",
        ".escript",
        ".ets",
        ".ex",
        ".exs",
        ".f",
        ".f03",
        ".f08",
        ".f90",
        ".f95",
        ".fsproj",
        ".go",
        ".gradle",
        ".groovy",
        ".h",
        ".hcl",
        ".hpp",
        ".hrl",
        ".inc",
        ".java",
        ".jl",
        ".js",
        ".json",
        ".jsx",
        ".kt",
        ".kts",
        ".lfm",
        ".lisp",
        ".lpk",
        ".lpr",
        ".lsp",
        ".lua",
        ".luau",
        ".m",
        ".metal",
        ".mjs",
        ".ml",
        ".mli",
        ".mm",
        ".mts",
        ".pas",
        ".php",
        ".pp",
        ".ps1",
        ".psd1",
        ".psm1",
        ".py",
        ".r",
        ".rake",
        ".razor",
        ".rb",
        ".resource",
        ".robot",
        ".rs",
        ".scala",
        ".sh",
        ".sln",
        ".slnx",
        ".sol",
        ".sql",
        ".sv",
        ".svelte",
        ".svh",
        ".swift",
        ".tf",
        ".tfvars",
        ".toc",
        ".trigger",
        ".ts",
        ".tsx",
        ".v",
        ".vb",
        ".vbproj",
        ".vh",
        ".vue",
        ".xaml",
        ".yaml",
        ".yml",
        ".zig",
    }
)
SEMANTIC_EXTENSIONS = frozenset(
    {
        ".md",
        ".mdx",
        ".rst",
        ".txt",
    }
)
# Upstream document, paper, image, office, video and Google Workspace suffixes
# the mesh does not treat as semantic. Each one would send a repo to an LLM
# extract (GPU cost) for little graph value. Table v3 honours .graphifyignore
# (state.GraphifyIgnoreRules), which keeps them out only in a repo that lists
# them there. Measured
# 2026-11-06: 944 .html files across the 25 registered roots yield 51 nodes.
# Adding a suffix here to semantic_extensions is a table change: bump
# CLASSIFICATION_VERSION. tests/compat checks this set against upstream.
DELIBERATELY_EXCLUDED_DOC = frozenset(
    {
        ".avi",
        ".docx",
        ".gdoc",
        ".gif",
        ".gsheet",
        ".gslides",
        ".html",
        ".j2",
        ".jinja",
        ".jinja2",
        ".jpeg",
        ".jpg",
        ".m4a",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp3",
        ".mp4",
        ".ogg",
        ".pdf",
        ".png",
        ".qmd",
        ".skill",
        ".svg",
        ".wav",
        ".webm",
        ".webp",
        ".xlsx",
    }
)
# Upstream manifest_ingest.PACKAGE_MANIFEST_NAMES: parsed as code whatever the
# suffix. Matched on the lowercased file name, as upstream does.
CODE_FILENAMES = frozenset(
    {
        "apm.yml",
        "apm.yaml",
        "pyproject.toml",
        "cargo.toml",
        "go.mod",
        "pom.xml",
    }
)
# Upstream detect._SKIP_FILES: never read by graphify. Matched case-sensitively,
# as upstream does. Wins over the extension tables.
IGNORED_FILENAMES = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "Cargo.lock",
        "poetry.lock",
        "Gemfile.lock",
        "composer.lock",
        "go.sum",
        "go.work.sum",
        ".graphifyinclude",
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
        # Documentation trees are never indexed (operator decision 2026-08-20):
        # a docs/ page is prose about the code, and in the KB benchmark the
        # graph served docs/tasks/ ground-truth and the judge script back to a
        # run that had those files removed from its working copy. The per-repo
        # `--exclude docs/ doc/` in <graphify-out>/.graphify_build.json keeps
        # the upstream extractor out of them; this entry keeps the mesh's own
        # staleness hashing from marking a repo dirty on a docs-only commit.
        "docs",
        "doc",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
        ".idea",
        ".vscode",
    }
)
# Fixed-name entries of upstream detect._SKIP_DIRS: graphify never reads these,
# so their files must not move the source digest. Hand-copied like the suffix
# tables; tests/compat/test_classification_drift.py notices an upstream change.
# Upstream's evidence-gated names (env, out, coverage) are left out: a bare name
# match would prune real source dirs that upstream keeps.
UPSTREAM_SKIP_DIR_NAMES = frozenset(
    {
        "venv",
        ".venv",
        "node_modules",
        "__pycache__",
        ".git",
        "dist",
        "build",
        "target",
        "site-packages",
        "lib64",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".eggs",
        "graphify-out",
        "lcov-report",
        "visual-tests",
        "visual-test",
        "__snapshots__",
        "storybook-static",
        "dist-protected",
        ".next",
        ".nuxt",
        ".turbo",
        ".angular",
        ".idea",
        ".cache",
        ".parcel-cache",
        ".svelte-kit",
        ".terraform",
        ".serverless",
        ".graphify",
        ".obsidian",
        ".smart-env",
        ".worktrees",
    }
)
# Upstream's one glob entry, "*.egg-info", as a name suffix.
UPSTREAM_SKIP_DIR_SUFFIXES = (".egg-info",)


@dataclass(frozen=True)
class ClassificationTable:
    """One versioned file -> category mapping (see categorize_file).

    The pruned dir names belong to the table because they decide which files
    reach the digest at all: changing them changes stored hashes, exactly like
    an extension edit, and needs the same version bump.
    """

    version: int
    code_extensions: frozenset[str]
    semantic_extensions: frozenset[str]
    code_filenames: frozenset[str] = frozenset()
    ignored_filenames: frozenset[str] = frozenset()
    pruned_dir_names: frozenset[str] = IGNORED_DIR_NAMES
    pruned_dir_suffixes: tuple[str, ...] = ()
    # Whether files graphify's ignore rules exclude stay out of the digest.
    honours_ignore_rules: bool = False
    # Whether files graphify drops as likely secrets stay out (is_sensitive_path).
    drops_sensitive_paths: bool = False

    def prunes(self, name: str) -> bool:
        """Whether a path component `name` keeps its subtree out of the digest."""
        return name in self.pruned_dir_names or name.endswith(self.pruned_dir_suffixes)


CURRENT_CLASSIFICATION = ClassificationTable(
    version=CLASSIFICATION_VERSION,
    code_extensions=CODE_EXTENSIONS,
    semantic_extensions=SEMANTIC_EXTENSIONS,
    code_filenames=CODE_FILENAMES,
    ignored_filenames=IGNORED_FILENAMES,
    pruned_dir_names=IGNORED_DIR_NAMES | UPSTREAM_SKIP_DIR_NAMES,
    pruned_dir_suffixes=UPSTREAM_SKIP_DIR_SUFFIXES,
    honours_ignore_rules=True,
    drops_sensitive_paths=True,
)
# Frozen v2 table: the v3 categories without graphify's ignore rules. Never edit it.
LEGACY_CLASSIFICATION_V2 = ClassificationTable(
    version=2,
    code_extensions=CODE_EXTENSIONS,
    semantic_extensions=SEMANTIC_EXTENSIONS,
    code_filenames=CODE_FILENAMES,
    ignored_filenames=IGNORED_FILENAMES,
    pruned_dir_names=IGNORED_DIR_NAMES | UPSTREAM_SKIP_DIR_NAMES,
    pruned_dir_suffixes=UPSTREAM_SKIP_DIR_SUFFIXES,
)
# Frozen v1 table. State written before CLASSIFICATION_VERSION existed carries
# hashes under it; decide_action recomputes a repo's digest with it once to tell
# a real document edit from files that only changed category. Never edit it.
LEGACY_CLASSIFICATION_V1 = ClassificationTable(
    version=1,
    code_extensions=frozenset(
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
    ),
    semantic_extensions=frozenset(
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
    ),
)
# Version-keyed registry: migration looks up the table a repo's state was written
# under instead of hardcoding v1. Add the new entry when CLASSIFICATION_VERSION
# bumps; never drop an old one while state under it can still exist.
CLASSIFICATION_TABLES: dict[int, ClassificationTable] = {
    1: LEGACY_CLASSIFICATION_V1,
    2: LEGACY_CLASSIFICATION_V2,
    3: CURRENT_CLASSIFICATION,
}


def _classification_version_from_env(var_name: str) -> int:
    """The classification table the digest uses, CLASSIFICATION_VERSION when unset.

    An older version is a rollback switch: state records it, so selecting
    CLASSIFICATION_VERSION again migrates each repo once. Fails at startup on a
    version with no table, like the other settings parsers."""
    raw = os.environ.get(var_name)
    if raw is None:
        return CLASSIFICATION_VERSION
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(
            f"{var_name} must be one of {sorted(CLASSIFICATION_TABLES)}, got {raw!r}"
        ) from exc
    if value not in CLASSIFICATION_TABLES:
        raise ValueError(f"{var_name} must be one of {sorted(CLASSIFICATION_TABLES)}, got {raw!r}")
    return value


def categorize_file(path: Path, table: ClassificationTable = CURRENT_CLASSIFICATION) -> str:
    """Classify a source file as 'code', 'semantic', or 'ignore' under `table`.

    dict-dispatch (via frozenset membership) rather than an if/elif chain,
    per project code style rules. File-name rules win over the suffix.
    """
    if path.name in table.ignored_filenames:
        return "ignore"
    if path.name.lower() in table.code_filenames:
        return "code"
    suffix = path.suffix.lower()
    if suffix in table.code_extensions:
        return "code"
    if suffix in table.semantic_extensions:
        return "semantic"
    return "ignore"


# Hand copy of graphify 0.9.103 detect._is_sensitive and its helpers
# (_CREDENTIAL_STORE_DIRS, _AMBIGUOUS_SENSITIVE_DIRS, _SENSITIVE_PATTERNS,
# _ENV_TEMPLATE_SUFFIXES, _GENERIC_KEYWORD_PATTERNS, _PROSE_EXTS,
# _BARE_PLURAL_TOKENS, _SECRET_PRONE_DATA_EXTS, _WORD_SPLIT). detect() drops
# these files before classification, so the v3 digest must not count them.
# Never imported at runtime; tests/compat/test_sensitive_paths_drift.py
# compares it with the installed graphify. Editing it is a table change.
SENSITIVE_CREDENTIAL_STORE_DIRS = frozenset({".ssh", ".gnupg", ".aws", ".gcloud"})
SENSITIVE_AMBIGUOUS_DIRS = frozenset({"secrets", ".secrets", "credentials"})
SENSITIVE_NAME_PATTERNS = (
    re.compile(r"(^|[\\/])\.(env|envrc)(\.|$)", re.IGNORECASE),
    re.compile(r"\.(pem|key|p12|pfx|cert|crt|der|p8)$", re.IGNORECASE),
    re.compile(r"(^|[^A-Za-z0-9])(id_rsa|id_dsa|id_ecdsa|id_ed25519)(\.pub)?$", re.IGNORECASE),
    re.compile(r"^secring(\.(gpg|pgp))?$", re.IGNORECASE),
    re.compile(
        r"(\.netrc|\.pgpass|\.htpasswd|\.npmrc|\.pypirc|\.git-credentials|\.boto)$",
        re.IGNORECASE,
    ),
)
SENSITIVE_ENV_TEMPLATE_SUFFIXES = (".example", ".sample", ".template", ".dist")
SENSITIVE_KEYWORD_PATTERNS = (
    re.compile(
        r"(?<![a-zA-Z0-9])(credential|secret|passwd|password|private_key)s?(?![a-zA-Z])",
        re.IGNORECASE,
    ),
    re.compile(r"(?<![a-zA-Z0-9])tokens?(?![a-zA-Z])", re.IGNORECASE),
    re.compile(r"(?<![a-zA-Z0-9])service[._-]?account(?![a-zA-Z])", re.IGNORECASE),
)
SENSITIVE_PROSE_EXTS = frozenset({".md", ".markdown", ".rst", ".org", ".adoc", ".tex"})
SENSITIVE_BARE_PLURAL_TOKENS = re.compile(r"(?<![a-zA-Z0-9])tokens(?![a-zA-Z])", re.IGNORECASE)
SECRET_PRONE_DATA_EXTS = frozenset(
    {
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".config",
        ".xml",
        ".properties",
        ".env",
        ".txt",
        ".tfvars",
    }
)
_SENSITIVE_WORD_SPLIT = re.compile(r"[-_\s.]+")


def _is_env_template(name: str) -> bool:
    lower = name.lower()
    if not lower.endswith(SENSITIVE_ENV_TEMPLATE_SUFFIXES):
        return False
    return bool(re.match(r"\.(env|envrc)\.", lower))


def _is_graphable_source(path: Path) -> bool:
    """Upstream: classify_file is CODE and the suffix is not a data format.

    Uses the mesh code tables, which equal upstream CODE_EXTENSIONS and the
    manifest names (test_classification_drift.py). Upstream's shebang and
    .graphifyrc language routing for extensionless or redeclared files is not
    copied: those files are not code in the mesh digest either."""
    suffix = path.suffix.lower()
    is_code = suffix in CODE_EXTENSIONS or path.name.lower() in CODE_FILENAMES
    return is_code and suffix not in SECRET_PRONE_DATA_EXTS


def _is_prose_note(path: Path) -> bool:
    if path.suffix.lower() not in SENSITIVE_PROSE_EXTS:
        return False
    stem = Path(path.name).stem.lstrip(".") or Path(path.name).stem
    if SENSITIVE_BARE_PLURAL_TOKENS.fullmatch(stem):
        return True
    return not any(p.fullmatch(stem) for p in SENSITIVE_KEYWORD_PATTERNS)


def _generic_keyword_hit(name: str) -> bool:
    stem = Path(name).stem.lstrip(".") or Path(name).stem
    for pattern in SENSITIVE_KEYWORD_PATTERNS:
        hit = False
        for match in pattern.finditer(stem):
            hit = True
            if match.end() == len(stem):
                return True
        if hit and len([w for w in _SENSITIVE_WORD_SPLIT.split(stem) if w]) <= 2:
            return True
    return False


def is_sensitive_path(path: Path) -> bool:
    """Whether graphify skips `path` as a likely secret (detect._is_sensitive).

    Pass the absolute, resolved path: upstream checks every parent directory
    of the scanned path, the ones above the scan root included."""
    parents = path.parts[:-1]
    if any(part.lower() in SENSITIVE_CREDENTIAL_STORE_DIRS for part in parents):
        return True
    if any(
        part.lower() in SENSITIVE_AMBIGUOUS_DIRS for part in parents
    ) and not _is_graphable_source(path):
        return True
    name = path.name
    if any(p.search(name) for p in SENSITIVE_NAME_PATTERNS) and not _is_env_template(name):
        return True
    if _generic_keyword_hit(name):
        return not (_is_graphable_source(path) or _is_prose_note(path))
    return False


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
    # Env GRAPHIFY_MESH_SKIP_LABELING=1 or CLI --skip-labeling; an explicit CLI flag
    # either way beats the env value (cli.build_overrides).
    skip_labeling: bool = field(
        default_factory=lambda: os.environ.get("GRAPHIFY_MESH_SKIP_LABELING", "0") == "1"
    )
    # One-shot: the naming stage asks the LLM for EVERY community, ignoring carried
    # names and the unchanged-fingerprint reuse. Community ids stay stable. Env
    # GRAPHIFY_MESH_RELABEL_ALL=1 or CLI --relabel-all; skip_labeling wins over it.
    relabel_all: bool = field(
        default_factory=lambda: os.environ.get("GRAPHIFY_MESH_RELABEL_ALL", "0") == "1"
    )
    # "repo" (default) or "global". Env GRAPHIFY_MESH_CLUSTER_SCOPE.
    cluster_scope: str = field(default_factory=lambda: _read_cluster_scope())
    skip_embedding: bool = False
    # Key of CLASSIFICATION_TABLES the source digest is taken with. Env
    # GRAPHIFY_MESH_CLASSIFICATION_VERSION; an older value rolls the digest back.
    classification_version: int = field(
        default_factory=lambda: _classification_version_from_env(
            "GRAPHIFY_MESH_CLASSIFICATION_VERSION"
        )
    )
    # Changed repos get local `graphify update` instead of LLM `extract`: measured
    # 2026-10-12, 0 of 100 search hits came from LLM-extracted nodes (1-2% of a graph).
    # Bootstrap is unaffected. Env GRAPHIFY_MESH_AST_ONLY=1 or CLI --ast-only.
    ast_only: bool = field(
        default_factory=lambda: os.environ.get("GRAPHIFY_MESH_AST_ONLY", "0") == "1"
    )
    allow_shrink: bool = False
    # Env GRAPHIFY_MESH_SHRINK_POLICY or CLI --shrink-policy (explicit flag wins).
    shrink_policy: str = field(default_factory=lambda: _read_shrink_policy())
    shrink_check_bin: str = field(
        default_factory=lambda: os.environ.get(
            "GRAPHIFY_MESH_SHRINK_CHECK_BIN", "graphify-shrink-check"
        )
    )
    # Runs `update` for repos whose registry entry says builder=graphify-php.
    php_build_bin: str = field(
        default_factory=lambda: os.environ.get("GRAPHIFY_MESH_PHP_BUILD_BIN", "graphify-php-build")
    )
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
    # Bounds ONE labeling call, exported to graphify as GRAPHIFY_API_TIMEOUT by
    # naming.bind_upstream_backend. Separate from ollama_health_timeout: that
    # one bounds a `/models` probe measured in seconds.
    ollama_api_timeout: float = field(
        default_factory=lambda: _api_timeout_from_env(
            "GRAPHIFY_MESH_OLLAMA_API_TIMEOUT", OLLAMA_DEFAULT_API_TIMEOUT
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
    embed_request_timeout: float = field(
        default_factory=lambda: _api_timeout_from_env(
            "GRAPHIFY_MESH_EMBED_TIMEOUT", EMBED_DEFAULT_REQUEST_TIMEOUT
        )
    )
    # Test-only dependency injection, mirrors ollama_health_check above but
    # for the embedding stage's own (native-endpoint) health check.
    ollama_embed_health_check: Callable[[str, float], bool] | None = None
    keep_embedding_generations: int = KEEP_EMBEDDING_GENERATIONS
    keep_structural_generations: int = KEEP_STRUCTURAL_GENERATIONS
    keep_snapshot_days: int = field(
        default_factory=lambda: _keep_snapshot_days_from_env(
            "GRAPHIFY_MESH_KEEP_SNAPSHOT_DAYS", KEEP_SNAPSHOT_DAYS
        )
    )

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
        if self.classification_version not in CLASSIFICATION_TABLES:
            raise ValueError(
                f"classification_version must be one of {sorted(CLASSIFICATION_TABLES)}, "
                f"got {self.classification_version!r}"
            )

    @property
    def classification_table(self) -> ClassificationTable:
        return CLASSIFICATION_TABLES[self.classification_version]

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
