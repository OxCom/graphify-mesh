"""Top-level orchestration: discovery -> per-project sync -> merge ->
naming -> embed-changed -> overlay-resolve -> lexical-index -> validate ->
atomic publish.

Pipeline stage order (per plan WS1 item 6), now fully wired:
    update -> merge -> recluster+remap -> label -> embed changed (WS3) ->
    overlay-resolve (WS4) -> lexical-index (WS5) -> validate -> atomic publish

The overlay artifact (`cross-project-overlay.json`) and the lexical-index
bundle (`lexical-index.json`, WS5) are both written into the same generation
dir as `global-graph.json` (see publish.write_overlay /
publish.write_lexical_index) so all three flip atomically on publish, but
each is a wholly separate file that is NEVER merged into the structural
graph (C5). The WS3 embedding index (id-map + per-repo shards under
`settings.embeddings_dir`) is likewise only durably persisted (and GC'd to
the last N generations) once publish actually happens — see
`embedding.persist_generation`, called from `_finalize` below.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from graphify_mesh.sync import (
    embedding,
    graphify_cli,
    lexical_index,
    naming,
    overlay,
    progress,
    publish,
    repo_tags,
    rss,
    validate,
)
from graphify_mesh.sync.backend import assert_pinned_backend
from graphify_mesh.sync.config import (
    CODE_EXTENSIONS,
    FORBIDDEN_OVERLAY_RELATION_TYPES,
    SEMANTIC_EXTENSIONS,
    Settings,
    is_valid_http_base_url,
)
from graphify_mesh.sync.discovery import assert_registry_containment, discover_filesystem, reconcile
from graphify_mesh.sync.locking import transaction_lock
from graphify_mesh.sync.registry import Registry, RepoEntry, load_registry, registry_hash
from graphify_mesh.sync.state import (
    SourceDigest,
    compute_source_manifest,
    file_content_hash,
    load_state,
    save_state,
)
from graphify_mesh.sync.sync_project import (
    ACTION_BOOTSTRAP,
    ACTION_EXTRACT,
    ACTION_SKIP,
    STATUS_BOOTSTRAP_FAILED,
    STATUS_FAILED,
    STATUS_INFRA_FAILED,
    STATUS_INFRA_SKIPPED,
    STATUS_SHRINK_REFUSED,
    ProjectOutcome,
    apply_action,
    decide_action,
)
from graphify_mesh.sync.vectors import RepoVectors

log = logging.getLogger("graphify_mesh.sync")

STALE_STATUSES = frozenset(
    {
        STATUS_FAILED,
        STATUS_BOOTSTRAP_FAILED,
        STATUS_SHRINK_REFUSED,
        STATUS_INFRA_FAILED,
        STATUS_INFRA_SKIPPED,
    }
)

# Infra-outage classifications (see sync_project.py). Reported as stale like
# `failed`, but exempt from BOTH publish-gate ratios while the repo's outage
# is younger than settings.infra_grace_hours — an in-grace infra repo is
# removed from the gate CALCULATION entirely (numerator AND denominator),
# never left in the denominator where it would dilute the ratios for the
# repos that actually count. Past the grace they count toward the UNREFRESHED
# ratio, in both numerator and denominator (the data is merely old, never
# suspect — an extract's last-good graph still flows to the merge exactly
# like the failed path; a first-time bootstrap has no graph and is omitted
# from the generation until the backend recovers).
INFRA_STATUSES = frozenset({STATUS_INFRA_FAILED, STATUS_INFRA_SKIPPED})

# Which statuses may VETO a publish, as opposed to merely being reported as stale.
# The three stale statuses do not carry the same accuracy cost:
#
#   bootstrap_failed  - no graph for this repo at all. Publishing omits it
#                       entirely, so the merged graph is genuinely incomplete.
#   shrink_refused    - a candidate was rejected and the last-good graph was
#                       restored. Publishing ships data that is suspect in age
#                       but structurally intact.
#   failed            - the refresh did not complete (e.g. extract timeout). The
#                       last-good graph is untouched and exactly as valid as the
#                       one already published; the only cost is age.
#
# `failed` is therefore held to a SEPARATE, higher threshold rather than the same
# one: a couple of slow repos must not veto a generation that is strictly newer
# for every other repo (an unpublished generation also freezes the embedding
# channel, degrading vector search fleet-wide while the gate "protects" nothing),
# but a large fraction failing is a systemic signal — the LLM host down, the disk
# full — and must still block.
PUBLISH_BLOCKING_STATUSES = frozenset({STATUS_BOOTSTRAP_FAILED, STATUS_SHRINK_REFUSED})
UNREFRESHED_STATUSES = frozenset({STATUS_FAILED})

# Fraction of repos that may fail to refresh before publish is refused. Higher
# than STALE_PUBLISH_THRESHOLD because unrefreshed data is merely older, not
# suspect; low enough that a systemic outage still trips it.
UNREFRESHED_PUBLISH_THRESHOLD = 0.40
UNREFRESHED_THRESHOLD_ENV = "GRAPHIFY_MESH_UNREFRESHED_THRESHOLD"


def _unrefreshed_threshold() -> float:
    """Resolve UNREFRESHED_PUBLISH_THRESHOLD from the env, clamped to (0, 1]."""
    raw = os.environ.get(UNREFRESHED_THRESHOLD_ENV)
    if raw is None or not raw.strip():
        return UNREFRESHED_PUBLISH_THRESHOLD
    try:
        value = float(raw)
    except ValueError:
        return UNREFRESHED_PUBLISH_THRESHOLD
    if value <= 0.0 or value > 1.0:
        return UNREFRESHED_PUBLISH_THRESHOLD
    return value


# Actions whose child shells out to the remote Ollama extract backend
# (`graphify extract --backend ollama`). ACTION_UPDATE is local AST-only and
# must run even during a backend outage.
OLLAMA_BACKED_ACTIONS = frozenset({ACTION_EXTRACT, ACTION_BOOTSTRAP})

INFRA_SKIPPED_PREFLIGHT_REASON = "extract backend unhealthy (preflight)"

# Tri-state extract-backend probe classification. A boolean probe made ANY
# failure an outage, so a misconfigured probe (bad key, wrong path, TLS/
# gateway misconfig — anything the server answers with a 4xx) silently
# suppressed every extract forever. Misconfiguration fails OPEN instead:
# children spawn normally and real failures surface as plain `failed`.
PROBE_HEALTHY = "healthy"
PROBE_OUTAGE = "outage"
PROBE_MISCONFIGURED = "misconfigured"

# publish_blocked_reason for the run where EVERY actionable repo ended
# infra_* — nothing was refreshed, so merging/naming/embedding the same
# inputs again would only burn GPU to republish identical data.
NOOP_INFRA_OUTAGE_REASON = "noop: infra outage — no repo refreshed, nothing to integrate"

# Stages short-circuited by the zero-refresh noop above, in pipeline order.
INFRA_NOOP_SKIPPED_STAGES = (
    "merge",
    "naming",
    "embedding",
    "overlay",
    "lexical_index",
    "validate",
    "publish",
)


def _infra_guard_applies(settings: Settings, action: str) -> bool:
    """Whether the extract-backend probe guards this launch. Disabled via
    GRAPHIFY_MESH_EXTRACT_HEALTH, or an empty probe URL, makes the feature
    inert: no probes, no infra_* statuses — behavior identical to before the
    guard existed."""
    if action not in OLLAMA_BACKED_ACTIONS:
        return False
    if not settings.extract_health_enabled:
        return False
    return bool(settings.extract_health_url)


def default_extract_backend_probe(base_url: str, api_key: str, timeout: float) -> tuple[str, str]:
    """GET `{base_url}/models` — the same lowest-cost endpoint
    naming.default_ollama_health_check uses (whose own boolean contract for
    the naming/embedding stages is deliberately left untouched), classified
    tri-state for the extract guard:

      healthy       - 2xx.
      outage        - connect error, DNS failure, timeout, or a 5xx: the
                      backend itself is unreachable/down.
      misconfigured - any 4xx (bad key, wrong path, TLS/gateway misconfig),
                      or a non-http(s) probe URL: the PROBE is wrong, not the
                      backend, so the guard must fail open.

    Returns `(state, detail)` where detail carries the status code/error for
    the once-per-run log line. Never raises out of the pipeline."""
    url = base_url.rstrip("/") + "/models"
    if not is_valid_http_base_url(url):
        return PROBE_MISCONFIGURED, f"non-http(s) probe URL {url!r}"
    req = urllib.request.Request(  # noqa: S310 - scheme validated above
        url, headers={"Authorization": f"Bearer {api_key}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed internal endpoint
            status = getattr(resp, "status", resp.getcode())
    except urllib.error.HTTPError as exc:
        if 400 <= exc.code < 500:
            return PROBE_MISCONFIGURED, f"HTTP {exc.code}"
        return PROBE_OUTAGE, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 - any transport failure => outage, never crash the pipeline
        return PROBE_OUTAGE, str(exc)
    if 200 <= status < 300:
        return PROBE_HEALTHY, ""
    return PROBE_OUTAGE, f"HTTP {status}"


def _probe_extract_backend(settings: Settings) -> tuple[str, str]:
    """Probe the extract backend once, tri-state. PROBE_HEALTHY when the
    feature is inert (no probe attempted). DI order: the tri-state
    `extract_health_probe` wins; the boolean `extract_health_check` (kept for
    the existing tests' contract) maps True -> healthy, False -> outage."""
    if not settings.extract_health_enabled or not settings.extract_health_url:
        return PROBE_HEALTHY, ""
    probe_args = (
        settings.extract_health_url,
        settings.extract_health_api_key,
        settings.extract_health_timeout,
    )
    if settings.extract_health_probe is not None:
        return settings.extract_health_probe(*probe_args), ""
    if settings.extract_health_check is not None:
        return (PROBE_HEALTHY if settings.extract_health_check(*probe_args) else PROBE_OUTAGE), ""
    return default_extract_backend_probe(*probe_args)


class _MisconfigLogOnce:
    """Thread-safe once-per-run gate for the misconfigured-probe log line.
    The probe runs inside pool worker threads, once per launch, but a
    misconfigured probe endpoint is one fact per run, not one per repo — so
    the run logs it exactly once, at ERROR, naming the URL and status."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._logged = False

    def error(self, url: str, detail: str) -> None:
        with self._lock:
            if self._logged:
                return
            self._logged = True
        log.error(
            "extract-backend probe misconfigured (%s) for %s — failing open: "
            "children spawn normally and real failures stay `failed`",
            detail,
            url,
        )


def _guarded_apply_action(
    repo_id: str,
    graphify_bin: str,
    root: Path,
    collection_path: Path,
    action: str,
    current_manifest,
    *,
    settings: Settings,
    allow_shrink: bool = False,
    shrink_tolerance: float = 0.0,
    misconfig_once: _MisconfigLogOnce | None = None,
) -> ProjectOutcome:
    """apply_action wrapped in the extract-backend infra guard.

    Probes BEFORE spawning each ollama-backed child (per-launch re-probe:
    catches both a start-of-run outage and mid-run recovery): an OUTAGE means
    the child is never spawned and the repo is reported `infra_skipped` with
    state untouched — an existing last-good graph.json still flows to the
    merge, same as the failed path (a first-time bootstrap has none and is
    omitted from the generation). A MISCONFIGURED probe (4xx) fails open: the
    guard disarms for this launch, the child spawns normally, real failures
    stay plain `failed`, and the misconfiguration is logged once per run.
    When a spawned child comes back plain `failed` under an armed guard, one
    more probe decides attribution: an outage reclassifies the outcome to
    `infra_failed`; anything else leaves `failed` untouched. bootstrap_failed
    and shrink_refused are never reclassified — those verdicts are about the
    repo's own output, not about reachability.
    """
    guarded = _infra_guard_applies(settings, action)
    if guarded:
        state, detail = _probe_extract_backend(settings)
        if state == PROBE_MISCONFIGURED:
            (misconfig_once or _MisconfigLogOnce()).error(settings.extract_health_url, detail)
            guarded = False
        elif state == PROBE_OUTAGE:
            return ProjectOutcome(
                repo_id, action, STATUS_INFRA_SKIPPED, reason=INFRA_SKIPPED_PREFLIGHT_REASON
            )
    outcome = apply_action(
        repo_id,
        graphify_bin,
        root,
        collection_path,
        action,
        current_manifest,
        allow_shrink=allow_shrink,
        shrink_tolerance=shrink_tolerance,
    )
    if not guarded:
        return outcome
    if outcome.status != STATUS_FAILED:
        return outcome
    state, _detail = _probe_extract_backend(settings)
    if state != PROBE_OUTAGE:
        return outcome
    outcome.status = STATUS_INFRA_FAILED
    outcome.reason = f"{outcome.reason}; backend probe failed"
    return outcome


def _within_infra_grace(now: float, infra_since: float, grace_hours: float) -> bool:
    """Whether an infra_* repo is still exempt from the publish gate. A grace
    of 0 means no exemption at all — infra statuses count toward the
    unrefreshed ratio immediately."""
    if grace_hours <= 0.0:
        return False
    return (now - infra_since) <= grace_hours * 3600.0


STAGING_PREFIX = "graphify-mesh-sync-staging-"
# A SIGKILLed/OOM-killed run never reaches the `finally` that removes its
# staging dir, so stale siblings accumulate in the temp dir forever. Swept
# at startup (under the transaction lock) once they're old enough that no
# live run can plausibly still own them — a full sync is minutes, so 6h is
# very conservative (and also clears concurrently-running dry-runs' dirs
# only long after those dry-runs are dead).
STALE_STAGING_MAX_AGE_SECONDS = 6 * 3600.0


def _sweep_stale_staging(
    own_staging: Path, max_age_seconds: float = STALE_STAGING_MAX_AGE_SECONDS
) -> list[str]:
    """Remove leftover staging dirs from previous killed runs. Only dirs
    matching our own mkdtemp prefix, never our own live dir, and only past
    the age threshold. Called while holding the transaction lock, so no
    other non-dry-run sync can be mid-run."""
    removed: list[str] = []
    now = time.time()
    for candidate in own_staging.parent.glob(STAGING_PREFIX + "*"):
        if candidate == own_staging or not candidate.is_dir():
            continue
        try:
            age = now - candidate.stat().st_mtime
        except OSError:
            continue
        if age < max_age_seconds:
            continue
        shutil.rmtree(candidate, ignore_errors=True)
        removed.append(candidate.name)
    return removed


def config_hash() -> str:
    payload = {
        "forbidden_relations": sorted(FORBIDDEN_OVERLAY_RELATION_TYPES),
        "code_extensions": sorted(CODE_EXTENSIONS),
        "semantic_extensions": sorted(SEMANTIC_EXTENSIONS),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


@dataclass
class RunReport:
    dry_run: bool
    reconciliation: dict
    project_actions: list[dict] = field(default_factory=list)
    stale_repos: list[str] = field(default_factory=list)
    # Reported separately from stale_repos so an operator can see WHY a publish
    # was refused: only these statuses veto it.
    publish_blocking_repos: list[str] = field(default_factory=list)
    unrefreshed_repos: list[str] = field(default_factory=list)
    # infra_* repos split by the grace window: in-grace ones are removed from
    # the gate calculation entirely; past-grace ones also appear in
    # unrefreshed_repos. Reported so an operator can see which repos an
    # outage is currently hiding from the gate — and for how much longer.
    infra_repos_in_grace: list[str] = field(default_factory=list)
    infra_repos_past_grace: list[str] = field(default_factory=list)
    dirty_repos: list[str] = field(default_factory=list)
    merge_ok: bool = False
    merge_error: str = ""
    validation_ok: bool = False
    validation_errors: list[str] = field(default_factory=list)
    published: bool = False
    publish_blocked_reason: str = ""
    generation_id: str = ""
    skipped_stages: list[str] = field(default_factory=list)
    auto_add_failed: list[str] = field(default_factory=list)
    labeling: str = ""
    clustering_backend: str = ""
    overlay_edge_counts: dict = field(default_factory=dict)
    overlay_manual_relation_count: int = 0
    embedding_status: str = ""
    embedding_stats: dict = field(default_factory=dict)
    lexical_index_stats: dict = field(default_factory=dict)
    stage_rss: dict = field(default_factory=dict)


def _embedding_skip_reason(settings: Settings) -> str:
    """Reason the embed stage must not run live Ollama /api/embed calls this
    run, or "" to run it. Dry-run must be network-free here: with dry-run
    project statuses being `would_*`, `unchanged_repo_ids` is always empty,
    so shard reuse is fully defeated and every node would be re-embedded for
    real — expensive live calls whose staged output is then discarded with
    the staging dir anyway (dry-run never publishes/persists)."""
    if settings.skip_embedding:
        return "skipped (--skip-embedding)"
    if settings.dry_run:
        return "skipped (dry-run: live embedding calls suppressed; nothing would persist)"
    return ""


def _repos_for_run(registry: Registry, reconciliation: dict) -> list:
    removed = set(reconciliation["removed"])
    missing = set(reconciliation["missing"])
    return [
        entry
        for entry in registry.repos
        if entry.enabled
        and entry.repo_id not in registry.disabled
        and entry.repo_id not in removed
        and entry.repo_id not in missing
    ]


def _root_for(entry, reconciliation: dict) -> Path:
    for renamed in reconciliation["renamed"]:
        if renamed["repo_id"] == entry.repo_id:
            return Path(renamed["new_root"])
    return entry.root


def run(settings: Settings) -> RunReport:
    staging_root = Path(tempfile.mkdtemp(prefix="graphify-mesh-sync-staging-"))
    if settings.dry_run:
        # Dry-run must not create or touch anything under the real mesh tree —
        # route the transaction lock into the private staging dir too.
        lock_path = staging_root / "dry-run.lock"
    else:
        settings.global_dir.mkdir(parents=True, exist_ok=True)
        lock_path = settings.lock_path

    try:
        with transaction_lock(lock_path):
            if not settings.dry_run:
                swept = _sweep_stale_staging(staging_root)
                if swept:
                    log.info(
                        "startup: swept %d stale staging dir(s) from killed runs: %s",
                        len(swept),
                        ", ".join(swept),
                    )
            return _run_locked(settings, staging_root)
    finally:
        # Staging is per-run scratch (mkdtemp above). Without this, every run
        # — success, blocked publish, or crash — leaks a tempdir; anything
        # durable was already copied out by publish/persist_generation.
        shutil.rmtree(staging_root, ignore_errors=True)


def _record_stage_rss(tracker: rss.StageRssTracker, report: RunReport) -> None:
    """Snapshot per-stage RSS marks recorded so far onto the report and log a
    one-line summary. Called at every return point in `_run_locked`, so a run
    that exits early (merge failure, validation failure, dry-run, stale
    threshold) still reports whatever stages it actually reached."""
    report.stage_rss = tracker.to_dict()
    log.info(
        "stage RSS (MB, hwm growth): %s",
        {name: round(entry["hwm_growth_kb"] / 1024) for name, entry in report.stage_rss.items()},
    )


def _run_locked(settings: Settings, staging_root: Path) -> RunReport:
    # Staleness baseline for the eventual manifest: captured BEFORE any
    # pipeline stage runs, so files edited DURING a long sync (mtime after
    # this instant but before publish) still compare newer-than-baseline and
    # get flagged stale by readers. `created_at` stays the publish stamp.
    sync_started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tracker = rss.StageRssTracker()
    log.info(
        "discovery: scanning %s (depth %d) ...",
        ", ".join(map(str, settings.scan_roots)),
        settings.scan_depth,
    )
    discovered = discover_filesystem(
        settings.scan_roots, settings.approved_roots, settings.scan_depth
    )
    registry = load_registry(settings.registry_path)
    # Hard error (never a degrade) if any enabled registry entry's
    # collection_path escapes the approved roots — same containment rule the
    # discovery symlink guard enforces, applied at the point where
    # approved_roots is known.
    assert_registry_containment(registry, settings.approved_roots)
    reconciliation = reconcile(discovered, registry, settings.mesh_root)
    report = RunReport(dry_run=settings.dry_run, reconciliation=reconciliation.to_dict())
    log.info(
        "discovery: %d registered, %d discovered, %d broken, %d removed, %d renamed",
        len(registry.repos),
        len(discovered),
        len(reconciliation.broken),
        len(reconciliation.removed),
        len(reconciliation.renamed),
    )

    state = load_state(settings.state_path)
    broken_ids = set(reconciliation.broken)
    active_repos = [
        e for e in _repos_for_run(registry, reconciliation.to_dict()) if e.repo_id not in broken_ids
    ]

    # Broken-symlink projects (WS1 item 1: "reported as broken, not crashed")
    # are handled separately: their source root is unreachable this cycle so
    # no update/extract is attempted, but their last-good collection_path
    # graph.json (which lives inside the mesh tree, independent of the broken
    # symlink at the source project) still contributes to the merge. They do
    # NOT count toward the stale-repo publish threshold — that threshold is
    # reserved for failed update/extract *attempts* (WS1 item 3), and a
    # project we never attempted to touch is not "stale data", just
    # unrefreshed this cycle.
    stale_repos: list[str] = []
    # Subset of stale_repos whose status may veto publish (see
    # PUBLISH_BLOCKING_STATUSES): a timeout leaves valid last-good data behind
    # and must not block a generation that is newer for every other repo.
    publish_blocking_repos: list[str] = []
    # Repos that merely failed to refresh (their last-good graph is intact and as
    # valid as what is already published). Gated separately, at a higher ratio.
    unrefreshed_repos: list[str] = []
    # infra_* repos split by grace: in-grace ones leave the gate calculation
    # entirely (numerator AND denominator); past-grace ones also land in
    # unrefreshed_repos above.
    infra_in_grace: list[str] = []
    infra_past_grace: list[str] = []
    dirty_repos: list[str] = []
    graph_paths_by_repo: dict[str, Path] = {}
    # WS4: source roots per repo_id, for depends_on/API extraction. Broken
    # repos keep their last-known root even though it's unreachable this
    # cycle — the extractors treat a missing root as "nothing found" rather
    # than an error, so this is harmless and avoids yet another repo-id ->
    # root lookup living in overlay.py.
    repo_roots_by_id: dict[str, Path] = {}
    bootstrap_failed_repo_ids: set[str] = set()

    for repo_id in reconciliation.broken:
        entry = registry.by_repo_id().get(repo_id)
        if entry is not None and (entry.collection_path / "graph.json").exists():
            graph_paths_by_repo[repo_id] = entry.collection_path / "graph.json"
        if entry is not None:
            repo_roots_by_id[repo_id] = entry.root
        report.project_actions.append(
            {"repo_id": repo_id, "action": "none", "status": "broken_symlink"}
        )

    total_active = len(active_repos)
    bar = progress.ProgressBar(total_active, label="indexing")
    reconciliation_dict = reconciliation.to_dict()
    # Actionable repos (update/extract/bootstrap) are collected here and
    # executed on a bounded thread pool below (Task 7 perf plan). dry-run and
    # skip rows are cheap and stay fully inline. `apply_action` is
    # self-contained per repo (own collection_path writes, unique mkstemp
    # snapshots), so running several concurrently is safe as long as nothing
    # shared is mutated until after every future has been joined.
    actionable: list[tuple[int, RepoEntry, Path, str, SourceDigest]] = []
    outcomes: dict[str, ProjectOutcome] = {}
    misconfig_once = _MisconfigLogOnce()
    with ThreadPoolExecutor(max_workers=settings.extract_concurrency) as pool:
        # Per-repo source-manifest computation (a full stat-walk of every
        # repo tree) is pure read-only work, independent per repo — submit it
        # all up front on the same bounded pool instead of walking each tree
        # sequentially before any decision can be made.
        manifest_futures = {
            entry.repo_id: pool.submit(
                compute_source_manifest, _root_for(entry, reconciliation_dict)
            )
            for entry in active_repos
        }
        for i, entry in enumerate(active_repos, start=1):
            root = _root_for(entry, reconciliation_dict)
            repo_roots_by_id[entry.repo_id] = root
            graph_path = entry.collection_path / "graph.json"
            has_graph = graph_path.exists()
            current_manifest = manifest_futures[entry.repo_id].result()
            prior_state = state.get(entry.repo_id)
            action = decide_action(prior_state, current_manifest, has_graph)
            log.info("[%d/%d] %s: %s ...", i, total_active, entry.repo_id, action)
            bar.tick(i - 1, f"{entry.repo_id}: {action} ...")

            if settings.dry_run:
                planned_status = (
                    "would_bootstrap" if action == ACTION_BOOTSTRAP else f"would_{action}"
                )
                report.project_actions.append(
                    {"repo_id": entry.repo_id, "action": action, "status": planned_status}
                )
                if has_graph:
                    graph_paths_by_repo[entry.repo_id] = graph_path
                bar.tick(i, f"{entry.repo_id}: {planned_status}")
                continue

            if action == ACTION_SKIP:
                report.project_actions.append(
                    {"repo_id": entry.repo_id, "action": action, "status": "unchanged"}
                )
                graph_paths_by_repo[entry.repo_id] = graph_path
                log.info("[%d/%d] %s: unchanged, skipped", i, total_active, entry.repo_id)
                bar.tick(i, f"{entry.repo_id}: unchanged")
                continue

            actionable.append((i, entry, root, action, current_manifest))

        futures = {
            entry.repo_id: pool.submit(
                _guarded_apply_action,
                entry.repo_id,
                settings.graphify_bin,
                root,
                entry.collection_path,
                action,
                current_manifest,
                # The infra guard probes inside the worker thread, right
                # before (and, on failure, right after) each child — a
                # per-launch re-probe, not one snapshot for the whole run.
                settings=settings,
                # Per-repo shrink acceptance follows the explicit operator
                # flag only — reconciliation.removed (removed repos) must not
                # loosen per-repo guards, so effective_allow_shrink does not
                # apply here.
                allow_shrink=settings.allow_shrink,
                shrink_tolerance=settings.shrink_tolerance,
                # Shared across workers: a misconfigured probe endpoint is
                # logged once per run, not once per launch.
                misconfig_once=misconfig_once,
            )
            for (i, entry, root, action, current_manifest) in actionable
        }
        # Consume futures in submission (sorted-registry) order, NOT
        # completion order, so logs/progress/report rows/state land in
        # exactly the same order the sequential loop used to produce.
        for i, entry, _root, action, _current_manifest in actionable:
            outcomes[entry.repo_id] = futures[entry.repo_id].result()
            log.info(
                "[%d/%d] %s: %s (%s)",
                i,
                total_active,
                entry.repo_id,
                outcomes[entry.repo_id].status,
                action,
            )
            bar.tick(i, f"{entry.repo_id}: {outcomes[entry.repo_id].status}")

    # All shared-structure mutation happens here, single-threaded, strictly
    # after every future in `actionable` has been joined above — no locking
    # needed because nothing below runs concurrently with anything else.
    now = time.time()
    for _i, entry, _root, action, _current_manifest in actionable:
        outcome = outcomes[entry.repo_id]
        graph_path = entry.collection_path / "graph.json"
        report.project_actions.append(
            {
                "repo_id": entry.repo_id,
                "action": action,
                "status": outcome.status,
                "reason": outcome.reason,
            }
        )
        if outcome.dirty_worktree:
            dirty_repos.append(entry.repo_id)
        if outcome.status in STALE_STATUSES:
            stale_repos.append(entry.repo_id)
            if outcome.status in PUBLISH_BLOCKING_STATUSES:
                publish_blocking_repos.append(entry.repo_id)
            elif outcome.status in UNREFRESHED_STATUSES:
                unrefreshed_repos.append(entry.repo_id)
            elif outcome.status in INFRA_STATUSES:
                # Grace window: `infra_since` marks the START of the outage
                # and is set only if absent, so it survives across runs until
                # a successful refresh replaces the state entry wholesale
                # (which clears it). Within the grace the repo is removed
                # from the gate calculation entirely (neither ratio's
                # numerator nor denominator); past it, it counts toward the
                # unrefreshed ratio — its last-good graph is old, never
                # suspect.
                prior = state.get(entry.repo_id) or {}
                infra_since = prior.get("infra_since")
                if infra_since is None:
                    infra_since = now
                    state[entry.repo_id] = {**prior, "infra_since": infra_since}
                if _within_infra_grace(now, float(infra_since), settings.infra_grace_hours):
                    infra_in_grace.append(entry.repo_id)
                else:
                    infra_past_grace.append(entry.repo_id)
                    unrefreshed_repos.append(entry.repo_id)
            if outcome.status == STATUS_BOOTSTRAP_FAILED:
                # Only genuine bootstrap failures — an infra_skipped
                # bootstrap never even spawned, so reporting it as a failed
                # auto-add would blame the repo for the backend's outage.
                bootstrap_failed_repo_ids.add(entry.repo_id)
            if outcome.refused_manifest is not None:
                # Advance *attempted* state only. The accepted digest and the
                # last-good graph.json stay as they were, but recording which
                # source digest was refused (and how many times) lets
                # decide_action stop re-extracting an unchanged source forever.
                prior = state.get(entry.repo_id) or {}
                refused_hash = outcome.refused_manifest.semantic_hash
                streak = int(prior.get("refusal_streak") or 0)
                streak = streak + 1 if prior.get("refused_semantic_hash") == refused_hash else 1
                state[entry.repo_id] = {
                    **prior,
                    "refused_semantic_hash": refused_hash,
                    "refusal_streak": streak,
                }
        else:
            if outcome.new_manifest is not None:
                # A successful outcome clears any refusal memory: the source is
                # accepted now, so a future refusal starts its own streak.
                state[entry.repo_id] = outcome.new_manifest.to_dict()
        if graph_path.exists():
            graph_paths_by_repo[entry.repo_id] = graph_path
    bar.finish()

    report.stale_repos = sorted(set(stale_repos))
    report.publish_blocking_repos = sorted(set(publish_blocking_repos))
    report.unrefreshed_repos = sorted(set(unrefreshed_repos))
    report.infra_repos_in_grace = sorted(set(infra_in_grace))
    report.infra_repos_past_grace = sorted(set(infra_past_grace))
    report.dirty_repos = sorted(set(dirty_repos))
    report.auto_add_failed = sorted(bootstrap_failed_repo_ids)
    tracker.mark("extract")

    # Zero-refresh noop: every actionable repo this run ended infra_*, so not
    # a single graph changed — merging/naming/embedding the same inputs again
    # would only burn GPU/LLM time to republish identical data. Repos that
    # were skip/unchanged neither count as refreshed nor prevent the noop.
    # State is still saved via _finalize below so `infra_since` persists.
    #
    # Two constraints keep the noop honest — it exists for the true-blip case
    # only:
    #   - a removed repo must still be merged OUT and published, so a run
    #     with reconciliation.removed falls through to the normal tail;
    #   - once ANY infra repo is past its grace window the run must end with
    #     the unrefreshed-threshold publish_blocked_reason instead — a >grace
    #     total outage must stop looking like routine idling in status.json.
    infra_outage_noop = (
        bool(actionable)
        and not reconciliation.removed
        and not infra_past_grace
        and all(
            outcomes[entry.repo_id].status in INFRA_STATUSES for _i, entry, *_rest in actionable
        )
    )
    if infra_outage_noop:
        report.skipped_stages = list(INFRA_NOOP_SKIPPED_STAGES)
        report.publish_blocked_reason = NOOP_INFRA_OUTAGE_REASON
        log.info(
            "noop: %s — skipping %s",
            NOOP_INFRA_OUTAGE_REASON,
            ", ".join(INFRA_NOOP_SKIPPED_STAGES),
        )
        _record_stage_rss(tracker, report)
        _finalize(settings, staging_root, report, state, published_data=None, generation_id="")
        return report

    sorted_repo_ids = sorted(graph_paths_by_repo.keys())
    sorted_graph_paths = [graph_paths_by_repo[rid] for rid in sorted_repo_ids]

    staging_home = staging_root / "graphify-home"
    merged_out_path = staging_root / "merged-graph.json"

    if not sorted_graph_paths:
        report.merge_ok = False
        report.merge_error = "no per-repo graphs available to merge"
    else:
        log.info(
            "merge: merging %d per-repo graphs (from empty, sorted order) ...",
            len(sorted_graph_paths),
        )
        merge_result = graphify_cli.run_merge_graphs(
            settings.graphify_bin, sorted_graph_paths, merged_out_path, staging_home
        )
        report.merge_ok = merge_result.ok
        if not merge_result.ok:
            report.merge_error = merge_result.stderr.strip()[:500]
        log.info("merge: %s", "ok" if report.merge_ok else "FAILED: " + report.merge_error)
    tracker.mark("merge")

    if not report.merge_ok:
        _record_stage_rss(tracker, report)
        _finalize(settings, staging_root, report, state, published_data=None, generation_id="")
        return report

    graph_data = json.loads(merged_out_path.read_text(encoding="utf-8"))
    # WS5 prerequisite: normalize graphify's auto-derived per-repo node-id
    # tags (which collide/diverge from the registry repo_id for this workspace's
    # collection-path layout, see repo_tags.py module docstring) back to the
    # true repo_id, BEFORE naming/embedding/overlay/lexical-index so every
    # downstream stage and the published artifact carry real repo
    # attribution (baseline systemic failure #1).
    tag_to_repo_id = repo_tags.compute_tag_to_repo_id(sorted_graph_paths, sorted_repo_ids)
    graph_data = repo_tags.rewrite_repo_tags(graph_data, tag_to_repo_id)
    previous_manifest = publish.read_current_manifest(settings.global_dir)
    previous_counts: tuple[int, int] | None = None
    if previous_manifest is not None:
        prev_nodes = previous_manifest.get("output_node_count")
        prev_edges = previous_manifest.get("output_edge_count")
        if isinstance(prev_nodes, int) and isinstance(prev_edges, int):
            previous_counts = (prev_nodes, prev_edges)

    # WS2: unconditionally strip any per-project community/community_name
    # carried over via merge (C23), then run the naming stage on the
    # stripped graph. This runs even in dry-run so validation reflects what
    # would actually publish; in dry-run mode the naming workspace is routed
    # into the ephemeral staging root instead of the real mesh tree, matching
    # the existing dry-run isolation guarantee for lock_path.
    stripped_graph_data = naming.strip_project_community_attrs(graph_data)
    naming_dir = (staging_root / "naming") if settings.dry_run else settings.naming_dir
    naming_dir.mkdir(parents=True, exist_ok=True)
    naming_staging_home = staging_root / "naming-home"

    if settings.skip_labeling:
        # --skip-labeling must guarantee ZERO network calls, no exceptions.
        # Do NOT call naming.run_naming() here even indirectly — that
        # function unconditionally runs `graphify cluster-only` + (usually)
        # `graphify label --backend ollama`, contacting whatever
        # ollama_base_url resolves to regardless of this flag. Only the
        # network-free backend assertion (C25, local import check, no
        # Ollama involved) runs; graph_data and community/community_name
        # stay stripped/placeholder, exactly as the flag promises.
        log.info("naming: skipped (--skip-labeling) — no naming/Ollama calls made")
        backend_check = assert_pinned_backend(settings.graphify_bin)
        graph_data = stripped_graph_data
        report.labeling = "skipped (--skip-labeling)"
        report.clustering_backend = backend_check.backend
    else:
        log.info("naming: reclustering + labeling communities ...")
        naming_result = naming.run_naming(
            settings.graphify_bin,
            naming_dir,
            naming_staging_home,
            stripped_graph_data,
            settings,
            health_check=settings.ollama_health_check,
        )
        report.labeling = naming_result.labeling
        report.clustering_backend = naming_result.backend
        log.info("naming: %s (backend=%s)", naming_result.labeling, naming_result.backend)
        if naming_result.labeling == naming.LABELING_DEGRADED:
            previous_global_graph = publish.read_current_global_graph(settings.global_dir)
            graph_data = naming.restore_last_global_community_names(
                naming_result.graph_data, previous_global_graph
            )
        else:
            graph_data = naming_result.graph_data
    # The pre-naming stripped graph (same object as the pre-strip merged
    # graph — strip_project_community_attrs mutates in place) is dead from
    # here on: when naming returned a freshly parsed copy, keeping this
    # alias pinned a second full graph in RAM through embed/overlay/lexical.
    del stripped_graph_data
    tracker.mark("naming")

    # WS3: embed-changed stage. Runs after naming/label, before overlay-resolve
    # (plan WS1 item 6 order). Per-repo raw graphs are loaded once here and
    # reused by the overlay stage below, rather than re-reading every
    # per-repo graph.json twice.
    graphs_by_repo = overlay.load_graphs_by_repo(graph_paths_by_repo)
    embedding_vectors_by_repo: dict[str, RepoVectors] = {}
    embedding_model_for_overlay = "unknown"
    embedding_recipe: dict = {}
    embeddings_staged_dir: Path | None = None
    embed_skip_reason = _embedding_skip_reason(settings)
    if embed_skip_reason:
        report.embedding_status = embed_skip_reason
        log.info("embedding: %s", embed_skip_reason)
    if not embed_skip_reason:
        unchanged_repo_ids = {
            a["repo_id"] for a in report.project_actions if a.get("status") == "unchanged"
        }
        # Provisional id used only for this run's staged id-map bookkeeping
        # (tombstoned_at/generation_id) — the real, durable generation_id is
        # not known until after validate/publish below (see
        # embedding.persist_generation, which re-stages shards under the
        # real generation_id once publish actually happens).
        provisional_generation_id = time.strftime("provisional-%Y%m%dT%H%M%SZ", time.gmtime())
        embed_result = embedding.run_embedding_stage(
            graph_paths_by_repo,
            repo_roots_by_id,
            graphs_by_repo,
            unchanged_repo_ids,
            settings,
            provisional_generation_id,
            health_check=settings.ollama_embed_health_check,
        )
        report.embedding_status = embed_result.status
        report.embedding_stats = embed_result.stats.to_dict()
        embedding_vectors_by_repo = embed_result.vectors_by_repo
        embedding_model_for_overlay = settings.ollama_embed_model
        if embed_result.recipe is not None:
            embedding_recipe = embed_result.recipe.to_dict()
        embeddings_staged_dir = embedding.stage_embeddings(
            staging_root, embed_result.shards_by_repo, embed_result.id_map
        )
    tracker.mark("embedding")

    # WS4: overlay-resolve stage. Runs after the WS3 embed stage, before the
    # WS5 lexical-index stage — per plan WS1 item 6 pipeline order. Every
    # logical ref is resolved fresh
    # against graph_paths_by_repo/repo_roots_by_id for THIS generation (C27)
    # — nothing overlay-related is cached across runs. Dangling
    # manual-relation refs raise uncaught (same hard-fail convention as
    # naming's BackendMismatchError) and intentionally crash the run.
    overlay_result = overlay.build_overlay(
        graph_paths_by_repo,
        repo_roots_by_id,
        settings.manual_relations_path,
        settings.manual_relations_schema_path,
        embedding_vectors_by_repo=embedding_vectors_by_repo,
        embedding_model=embedding_model_for_overlay,
        graphs_by_repo=graphs_by_repo,
    )
    report.overlay_edge_counts = overlay_result.edge_counts_by_type
    report.overlay_manual_relation_count = overlay_result.manual_relation_count
    tracker.mark("overlay")

    # WS5: lexical-index stage. Runs after overlay-resolve, before validate
    # (plan WS1 item 6 order, now fully wired — see module docstring).
    # Rebuilt fresh every generation from the same per-repo raw graphs/roots
    # overlay.py already loaded above; nothing here is incremental/cached
    # across runs, matching the "bundle artifact built once per generation"
    # contract (WS5 deliverable 2) rather than a per-MCP-session rebuild.
    lexical_result = lexical_index.build_lexical_index(graphs_by_repo, repo_roots_by_id)
    report.lexical_index_stats = lexical_result.stats.to_dict()
    # Last consumer of the per-repo raw graphs (embed -> overlay -> lexical
    # all share this one load). Drop the full per-repo graph contents now
    # rather than keeping them resident alongside the merged graph through
    # validate/publish. (Dropped whole rather than slimmed to a field
    # projection up front: the three consumer modules are evolving
    # concurrently, so pinning their field usage here would be fragile.)
    del graphs_by_repo
    tracker.mark("lexical_index")

    # Repo removal is intentional pruning (WS1 item 1: "removed repos get
    # pruned from the global merge and flagged"), not the silent data loss
    # the shrink-guard exists to catch (C21) — a smaller merged graph this
    # run is expected when repos were removed, so auto-authorize the shrink
    # in that case instead of requiring the operator to pass --allow-shrink.
    effective_allow_shrink = settings.allow_shrink or bool(reconciliation.removed)
    validation = validate.run_all(
        graph_data, previous_counts, effective_allow_shrink, settings.skip_labeling
    )
    report.validation_ok = validation.ok
    report.validation_errors = validation.errors

    # In-grace infra repos are excluded from the denominator of BOTH ratios,
    # not just their numerators: left in the denominator they would dilute
    # the ratios for the repos that actually count (e.g. 7 in-grace infra +
    # 6 failed + 3 ok would publish at 6/16 = 37.5% < 40% although 13/16
    # repos went unrefreshed). A denominator of 0 — every considered repo
    # in-grace infra — means there is nothing left to gate on: no block.
    total_considered = len(active_repos) - len(report.infra_repos_in_grace)
    # Gate on the blocking subset, not on every stale repo: `failed` (e.g. an
    # extract timeout) leaves the previously published data intact for that repo,
    # so it costs age, not correctness, and must not veto the whole generation.
    blocking_ratio = (
        (len(report.publish_blocking_repos) / total_considered) if total_considered else 0.0
    )
    unrefreshed_ratio = (
        (len(report.unrefreshed_repos) / total_considered) if total_considered else 0.0
    )
    unrefreshed_limit = _unrefreshed_threshold()
    suspect_blocks = blocking_ratio > settings.stale_threshold
    unrefreshed_blocks = unrefreshed_ratio > unrefreshed_limit
    stale_blocks_publish = suspect_blocks or unrefreshed_blocks

    if settings.dry_run:
        report.publish_blocked_reason = "dry-run: no publish performed"
        tracker.mark("validate_publish")
        _record_stage_rss(tracker, report)
        _finalize(settings, staging_root, report, state, published_data=None, generation_id="")
        return report

    if not report.validation_ok:
        report.publish_blocked_reason = "validation failed: " + "; ".join(
            report.validation_errors[:3]
        )
        tracker.mark("validate_publish")
        _record_stage_rss(tracker, report)
        _finalize(settings, staging_root, report, state, published_data=None, generation_id="")
        return report

    if stale_blocks_publish:
        # Both gates can trip in the same run; the reason names every gate
        # that did, not just the first.
        reasons: list[str] = []
        if suspect_blocks:
            reasons.append(
                f"stale ratio {blocking_ratio:.2%} of suspect repos exceeds threshold "
                f"{settings.stale_threshold:.0%} "
                f"({len(report.publish_blocking_repos)}/{total_considered}: "
                f"{', '.join(report.publish_blocking_repos)})"
            )
        if unrefreshed_blocks:
            reasons.append(
                f"stale ratio {unrefreshed_ratio:.2%} of unrefreshed repos exceeds threshold "
                f"{unrefreshed_limit:.0%} "
                f"({len(report.unrefreshed_repos)}/{total_considered}: "
                f"{', '.join(report.unrefreshed_repos)})"
            )
        report.publish_blocked_reason = "; ".join(reasons)
        tracker.mark("validate_publish")
        _record_stage_rss(tracker, report)
        _finalize(settings, staging_root, report, state, published_data=None, generation_id="")
        return report

    graph_output_hash = publish.output_hash(graph_data)
    generation_id = publish.make_generation_id(graph_output_hash)
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # Per-repo input hashes: apply_action already hashed each graph.json it
    # touched this run (carried on ProjectOutcome.graph_content_hash) —
    # re-read only the repos it didn't (skip/broken/failed paths).
    repo_input_hashes: dict[str, str | None] = {}
    for rid in sorted_repo_ids:
        repo_outcome = outcomes.get(rid)
        carried = repo_outcome.graph_content_hash if repo_outcome is not None else None
        if carried is not None:
            repo_input_hashes[rid] = carried
            continue
        repo_input_hashes[rid] = file_content_hash(graph_paths_by_repo[rid])
    manifest = {
        "generation_id": generation_id,
        "created_at": created_at,
        "sync_started_at": sync_started_at,
        "repo_input_hashes": repo_input_hashes,
        "registry_hash": registry_hash(settings.registry_path),
        "config_hash": config_hash(),
        "output_node_count": len(graph_data.get("nodes", [])),
        "output_edge_count": len(graph_data.get("links", graph_data.get("edges", []))),
        "output_hash": graph_output_hash,
        "clustering_backend": report.clustering_backend,
        # C28: real embedding recipe (model/dim/snippet window/skip
        # heuristic), not the pre-WS3 "none" placeholder. `embedding_status`
        # is "ok"/"degraded"/"skipped (--skip-embedding)" so a degraded or
        # intentionally-skipped generation is distinguishable from one that
        # actually ran the embed stage.
        "embedding_model": "none" if settings.skip_embedding else embedding_model_for_overlay,
        "embedding_status": report.embedding_status,
        "embedding_recipe": embedding_recipe,
        "embedding_stats": report.embedding_stats,
        "labeling": report.labeling,
        "stale_repos": report.stale_repos,
        "dirty_repos": report.dirty_repos,
        "overlay_edge_counts": report.overlay_edge_counts,
        "overlay_manual_relation_count": report.overlay_manual_relation_count,
        # C28: WS5 lexical-index recipe/stats, so a generation's manifest is
        # self-describing about what the companion server will load — same
        # "record what produced the artifact" pattern as embedding_recipe.
        "lexical_index_tokenizer_version": lexical_index.TOKENIZER_VERSION,
        "lexical_index_schema_version": lexical_index.LEXICAL_SCHEMA_VERSION,
        "lexical_index_stats": report.lexical_index_stats,
    }
    gen_dir = publish.create_generation_dir(settings.generations_dir, generation_id)
    artifact_sha256: dict[str, str] = {}
    artifact_sha256["global-graph.json"] = publish.write_global_graph(gen_dir, graph_data)
    # WS4: overlay artifact is staged inside the SAME generation dir so it
    # flips atomically with global-graph.json/generation-manifest.json on
    # publish, but stays a wholly separate file — never merged into the
    # structural graph (C5).
    overlay_data = overlay.overlay_artifact(overlay_result, generation_id, created_at)
    artifact_sha256["cross-project-overlay.json"] = publish.write_overlay(gen_dir, overlay_data)
    # WS5: lexical-index bundle artifact, same atomic-flip treatment.
    artifact_sha256["lexical-index.json"] = publish.write_lexical_index(
        gen_dir, lexical_result.data
    )
    # Raw-byte sha256 of every artifact as written (hashed from the tmp-file
    # bytes before each atomic rename) — the companion MCP server verifies
    # artifact integrity against this map. Legacy logical hashes
    # (output_hash, repo_input_hashes) are untouched. Manifest is written
    # LAST so it can cover the other artifacts' bytes.
    manifest["artifact_sha256"] = artifact_sha256
    publish.write_manifest(settings.generations_dir, gen_dir, manifest)
    publish.flip_current(settings.global_dir, gen_dir)

    # Structural generations (global-graph.json + overlay + lexical-index,
    # tens to 100+ MB each) had no GC at all before this — runs only AFTER
    # flip_current succeeds, mirroring embedding.persist_generation's timing
    # rule, and never removes whatever `current` now points at.
    pruned = publish.prune_old_generations(
        settings.generations_dir, settings.current_symlink, settings.keep_structural_generations
    )
    if pruned:
        log.info(
            "publish: pruned %d old/incomplete generation(s): %s", len(pruned), ", ".join(pruned)
        )

    # WS3: only now (a successful publish) does the embedding index become
    # durable — mirrors the rest of the pipeline's "nothing persists unless
    # publish happens" rule. Nothing to persist if the embed stage was
    # skipped/degraded-with-nothing-new this run.
    if embeddings_staged_dir is not None:
        embedding.persist_generation(
            settings.embeddings_dir,
            generation_id,
            embeddings_staged_dir,
            settings.keep_embedding_generations,
        )

    report.published = True
    report.generation_id = generation_id
    tracker.mark("validate_publish")
    _record_stage_rss(tracker, report)

    _finalize(
        settings, staging_root, report, state, published_data=manifest, generation_id=generation_id
    )
    return report


def _finalize(
    settings: Settings,
    staging_root: Path,
    report: RunReport,
    state: dict,
    published_data,
    generation_id: str,
) -> None:
    if not settings.dry_run:
        save_state(settings.state_path, state)
        status = {
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dry_run": False,
            "reconciliation": report.reconciliation,
            "stale_repos": report.stale_repos,
            # Operator visibility: WHICH repos held a publish back (or would
            # have), and which infra repos the grace window is currently
            # hiding from the gate — split so a lingering outage is visible
            # in status.json before it starts blocking.
            "publish_blocking_repos": report.publish_blocking_repos,
            "unrefreshed_repos": report.unrefreshed_repos,
            "infra_repos_in_grace": report.infra_repos_in_grace,
            "infra_repos_past_grace": report.infra_repos_past_grace,
            "dirty_repos": report.dirty_repos,
            "merge_ok": report.merge_ok,
            "merge_error": report.merge_error,
            "validation_ok": report.validation_ok,
            "validation_errors": report.validation_errors,
            "published": report.published,
            "publish_blocked_reason": report.publish_blocked_reason,
            "generation_id": generation_id,
        }
        settings.status_path.parent.mkdir(parents=True, exist_ok=True)
        settings.status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    else:
        status_path = staging_root / "status.json"
        status_path.write_text(json.dumps(asdict(report), indent=2, default=str), encoding="utf-8")
