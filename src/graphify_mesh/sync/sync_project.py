"""Per-project decision + invocation + shrink-guard defense (WS1 items 2-4).

Decision (`decide_action`) and outcome classification
(`_classify_post_invoke`) both use dict-dispatch instead of if/elif chains,
per project code style rules.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

from graphify_mesh.sync import graphify_cli
from graphify_mesh.sync.config import CLASSIFICATION_VERSION, CURRENT_CLASSIFICATION
from graphify_mesh.sync.registry import BUILDER_GRAPHIFY, BUILDER_GRAPHIFY_PHP
from graphify_mesh.sync.state import (
    SourceDigest,
    file_content_hash,
    graph_hash_and_counts,
    is_worktree_dirty,
)

ACTION_SKIP = "skip"
ACTION_UPDATE = "update"
ACTION_EXTRACT = "extract"
ACTION_BOOTSTRAP = "bootstrap"

STATUS_UNCHANGED = "unchanged"
STATUS_UPDATED = "updated"
STATUS_BOOTSTRAPPED = "bootstrapped"
STATUS_BOOTSTRAP_FAILED = "bootstrap_failed"
STATUS_FAILED = "failed"
STATUS_SHRINK_REFUSED = "shrink_refused"
STATUS_NOOP = "noop"
# Infra-outage classifications (assigned by pipeline._guarded_apply_action,
# never by apply_action itself): the remote extract backend — not the repo —
# is the reason no refresh happened, so the publish gate treats these
# differently from a plain `failed` (see pipeline.py's grace-window handling).
#
#   infra_skipped - the pre-launch health probe said the backend is down, so
#                   the child was never spawned. Whatever graph.json already
#                   exists is untouched: for an extract that last-good graph
#                   still flows to the merge, but a first-time bootstrap has
#                   no graph yet, so the repo is simply omitted from this
#                   generation until the backend recovers.
#   infra_failed  - the child failed AND a follow-up probe confirmed the
#                   backend is down, so the failure is attributed to the
#                   outage rather than to the repo.
STATUS_INFRA_FAILED = "infra_failed"
STATUS_INFRA_SKIPPED = "infra_skipped"

log = logging.getLogger("graphify_mesh.sync")

# Which program produced a repo's accepted graph.json, recorded in per-repo
# state. A change of producer (extract -> update, update -> php-update) moves
# node and edge counts for reasons unrelated to the source, so the shrink check
# is told about it (`--producer-changed`) and refusal holds are keyed on it.
PRODUCER_EXTRACT = "extract"
PRODUCER_UPDATE = "update"
PRODUCER_PHP_UPDATE = "php-update"
_UPDATE_PRODUCERS = {
    BUILDER_GRAPHIFY: PRODUCER_UPDATE,
    BUILDER_GRAPHIFY_PHP: PRODUCER_PHP_UPDATE,
}

SHRINK_POLICY_COUNT = "count"

# Per-repo state key: the classification table version the accepted hashes were
# taken under. Written only with accepted state, so a refused or failed
# migration keeps the old version and retries next run.
CLASSIFICATION_VERSION_KEY = "classification_version"
# Per-repo state key: what produced the accepted hashes beyond the table. The
# current table reads graphify's ignore rules from the installed graphify, so a
# graphify upgrade can move files in or out of the digest with no source edit.
DIGEST_RECIPE_KEY = "digest_recipe"
# Per-repo state key: documents an extract left unstamped that already had
# their one retry. A partial new document and a document graphify never reads
# look the same in its manifest (no row), so the bound lives here.
UNSTAMPED_DOCUMENTS_KEY = "unstamped_documents"
# Per-repo state key: sha256 of the graph.json the sync last accepted. An
# unchanged source digest skips the build, so without it a graph.json rewritten
# by another program between runs was merged unchecked.
GRAPH_HASH_KEY = "graph_content_hash"


def digest_recipe(
    graphify_version: str | None, version: int = CLASSIFICATION_VERSION
) -> str | None:
    """The digest recipe for table `version`; None when the graphify version is unknown."""
    if not graphify_version:
        return None
    return f"v{version}+graphify-ignore-{graphify_version}"


def digest_recipe_changed(
    prior_state: dict | None,
    recipe: str | None,
    active_version: int = CLASSIFICATION_VERSION,
) -> bool:
    """Whether accepted hashes under the active table came from another recipe.

    Only for state already on the active table: a classification migration
    has its own exact check (legacy digest) and stamps the recipe on accept. An
    unknown current recipe is not a change.
    """
    if (
        prior_state is None
        or recipe is None
        or needs_classification_migration(prior_state, active_version)
    ):
        return False
    return prior_state.get(DIGEST_RECIPE_KEY) != recipe


def producer_for(action: str, builder: str = BUILDER_GRAPHIFY) -> str:
    """The producer `action` runs for a repo built by `builder`."""
    if action == ACTION_UPDATE:
        return _UPDATE_PRODUCERS[builder]
    return PRODUCER_EXTRACT


def refusal_key(
    digest: SourceDigest, producer: str, policy: str, graphify_version: str | None = None
) -> str:
    """Identity of one refused update: the source, the program, the guard.

    The attempt id rather than the semantic hash: an AST update reads code, so a
    code-only fix after a refusal must retry instead of staying held. Producer,
    graphify version and policy are in the key so changing any of them retries
    the same source once. An unknown version keeps the versionless key.
    """
    if graphify_version:
        producer = f"{producer}@{graphify_version}"
    return f"{digest.attempt_id}:{producer}:{policy}"


def recorded_classification_version(prior_state: dict) -> int | None:
    """The classification version `prior_state` hashes were taken under.

    State written before the version was recorded counts as version 1. None when
    the stored value is not an integer (hand-edited or corrupt state): no table
    is known for it, so the caller compares raw hashes as for an unknown version.
    """
    raw = prior_state.get(CLASSIFICATION_VERSION_KEY, 1)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def needs_classification_migration(
    prior_state: dict | None, active_version: int = CLASSIFICATION_VERSION
) -> bool:
    """Whether `prior_state` hashes were taken under another table than `active_version`.

    `active_version` is Settings.classification_version, so a rollback to an
    older table and the later switch back each migrate once. State written
    before the version was recorded counts as version 1. An unparseable version
    also counts: the accepted run stamps the active one.
    """
    if prior_state is None:
        return False
    return recorded_classification_version(prior_state) != active_version


def stored_hashes_match(prior_state: dict, digest: SourceDigest) -> bool:
    """Whether the accepted hashes in `prior_state` equal `digest`."""
    return (
        prior_state.get("code_hash") == digest.code_hash
        and prior_state.get("semantic_hash") == digest.semantic_hash
    )


def effective_shrink_grant(
    grant: str | None, current: SourceDigest, legacy: SourceDigest | None
) -> str | None:
    """The `allow_shrink_once` token as this run's attempt id sees it.

    A token issued before a classification migration names the attempt id under
    the old table. While migrating (`legacy` given), that token authorizes the
    same source state, so it is translated to the current attempt id that
    decide_action and apply_action compare against. Any other token passes as is.
    """
    if grant is not None and legacy is not None and grant == legacy.attempt_id:
        return current.attempt_id
    return grant


@dataclass
class ProjectOutcome:
    repo_id: str
    action: str
    status: str
    reason: str = ""
    dirty_worktree: bool = False
    new_manifest: SourceDigest | None = None
    # sha256 of the repo's graph.json AS LEFT ON DISK by this outcome, when
    # known. Publish reuses it for the manifest's repo_input_hashes instead
    # of re-reading every per-repo graph.json; None means "not computed here,
    # hash at publish time" (skip/broken/failed paths).
    graph_content_hash: str | None = None
    # Source digest of an attempt that was REFUSED (shrink guard). Recorded so a
    # refusal advances *attempted* state while leaving *accepted* state (and the
    # last-good graph.json) untouched. Without this, a refusal left the accepted
    # digest stale, decide_action saw a difference forever, and the repo
    # re-extracted and was refused on every run — a permanent retry loop that
    # also kept it in the stale bucket and blocked publish fleet-wide.
    refused_manifest: SourceDigest | None = None
    # The per-repo `allow_shrink_once` token this outcome spent, when a shrink was
    # accepted because it matched. The caller records it in per-repo state so the
    # same token cannot authorize a second shrink: the registry key is operator-
    # owned and this package never writes the registry (mesh-register.py does), so
    # "single use" is enforced on the state side.
    consumed_shrink_grant: str | None = None
    # (nodes, edges) this accepted outcome lost that graphify-shrink-check
    # attributed to source changes. The global guard subtracts the run's sum
    # from the previous generation's counts, so a deletion one repo's guard
    # accepted on evidence is not refused again by the count-only global floor.
    explained_shrink: tuple[int, int] = (0, 0)
    # Documents this extract left unstamped, for UNSTAMPED_DOCUMENTS_KEY. On a
    # held outcome (no new_manifest) every one of them has now had its retry;
    # on an accepted one, the ones accepted unstamped. None: not checked.
    unstamped_documents: tuple[str, ...] | None = None


# How many consecutive refusals of the SAME source digest before the repo stops
# re-extracting. 1 means: refuse once, then skip identical sources until either
# the source changes or an operator intervenes. Kept low because a retry costs a
# full LLM extract on a shared GPU and a non-deterministic extractor is not
# meaningfully more likely to succeed on attempt N+1 with identical input.
REFUSAL_RETRY_LIMIT = 1


def decide_action(
    prior_state: dict | None,
    current_manifest: SourceDigest,
    has_graph: bool,
    shrink_grant: str | None = None,
    ast_only: bool = False,
    refusal_key: str | None = None,
    allow_shrink: bool = False,
    producer_changed: bool = False,
    legacy_manifest: SourceDigest | None = None,
    recipe_changed: bool = False,
    active_version: int = CLASSIFICATION_VERSION,
    graph_drifted: bool = False,
) -> str:
    """Pick the action for one repo.

    `refusal_key` holds a refused `update` the way `refused_semantic_hash` holds
    a refused extract; None disables that hold (direct callers keep the old
    contract). It must be built from the digest the refusal fields in
    `prior_state` were recorded under: `refusal_key(current_manifest, ...)`
    normally, `refusal_key(legacy_manifest, ...)` while migrating.

    `allow_shrink` is the run-level --allow-shrink. It releases both holds: the
    guard would accept the held attempt this run, and holding it instead left
    the flag with nothing to act on for exactly the repos it was passed for.

    `producer_changed` is the pipeline's `_producer_upgraded`: a known version
    change of the program that builds this repo, or accepted state with no
    graphify version recorded. It only makes an unchanged source update instead
    of skip, under the same refusal hold. It does not pick `--force`; the
    pipeline passes that to `apply_action` as `force_rebuild`.

    `legacy_manifest` is the repo's digest under the classification table that
    `prior_state` was written with, passed when `needs_classification_migration`.
    Its semantic hash matching the stored one means no document changed, only
    file categories moved (e.g. YAML to code), so the repo is updated, never
    extracted. A mismatch is a real document edit and takes the normal path.
    Refusal fields were recorded under the same old table, so the extract hold
    compares `refused_semantic_hash` against the legacy digest while migrating.

    `recipe_changed` is `digest_recipe_changed`: the hashes were taken with
    another graphify's ignore rules, whose digest cannot be recomputed. Any
    difference is adopted with an update, never an extract, so a document
    edited in that same run is extracted only on its next change.

    `active_version` is the table `current_manifest` was taken under
    (Settings.classification_version). A switch in either direction migrates.

    `graph_drifted`: graph.json on disk differs from the one last accepted
    (GRAPH_HASH_KEY). Like `producer_changed`, it turns a skip into an update
    under the same refusal hold; the pipeline also forces that update.
    """
    if not has_graph:
        return ACTION_BOOTSTRAP
    if prior_state is None:
        # First time this repo's manifest is observed but a graph already
        # exists on disk (e.g. pre-existing curated graph). Seed state with a
        # cheap AST refresh rather than forcing an LLM extract every run.
        return ACTION_UPDATE
    migrating = legacy_manifest is not None and needs_classification_migration(
        prior_state, active_version
    )
    hold_digest = legacy_manifest if migrating and legacy_manifest is not None else current_manifest
    if (
        migrating
        and hold_digest.semantic_hash == prior_state.get("semantic_hash")
        and not stored_hashes_match(prior_state, current_manifest)
    ):
        return _update_unless_held(
            prior_state, current_manifest, shrink_grant, refusal_key, allow_shrink
        )
    if recipe_changed and not stored_hashes_match(prior_state, current_manifest):
        return _update_unless_held(
            prior_state, current_manifest, shrink_grant, refusal_key, allow_shrink
        )
    if current_manifest.semantic_hash != prior_state.get("semantic_hash"):
        if ast_only:
            # An AST update is deterministic, so an identical source refused once
            # is refused again every run (2026-10-13). Held by refusal_key, not by
            # the extract hold below, which keys on semantic content only.
            return _update_unless_held(
                prior_state, current_manifest, shrink_grant, refusal_key, allow_shrink
            )
        # Break the refusal loop: if this exact source digest has already been
        # refused REFUSAL_RETRY_LIMIT times, re-running the extractor on
        # unchanged input just burns GPU for another refusal. Hold the last-good
        # graph instead and wait for the source to actually change.
        #
        # An unspent `allow_shrink_once` naming this attempt id overrides that
        # hold: the operator authorized this exact refused attempt, and the only
        # way to honour it is to re-run the extract and let the guard accept the
        # result this time.
        if (
            hold_digest.semantic_hash == prior_state.get("refused_semantic_hash")
            and int(prior_state.get("refusal_streak") or 0) >= REFUSAL_RETRY_LIMIT
            and shrink_grant != current_manifest.attempt_id
            and not allow_shrink
        ):
            return ACTION_SKIP
        return ACTION_EXTRACT
    if (
        current_manifest.code_hash != prior_state.get("code_hash")
        or producer_changed
        or graph_drifted
    ):
        return _update_unless_held(
            prior_state, current_manifest, shrink_grant, refusal_key, allow_shrink
        )
    return ACTION_SKIP


def refusal_held(prior_state: dict, key: str | None) -> bool:
    """True when this exact update was refused often enough that it is held.

    Shared with the pipeline so --allow-shrink forces exactly what it releases.
    """
    return (
        key is not None
        and prior_state.get("refused_key") == key
        and int(prior_state.get("refusal_streak") or 0) >= REFUSAL_RETRY_LIMIT
    )


def _update_unless_held(
    prior_state: dict,
    current_manifest: SourceDigest,
    shrink_grant: str | None,
    key: str | None,
    allow_shrink: bool = False,
) -> str:
    """ACTION_UPDATE, or ACTION_SKIP while the same refused update would repeat.

    Same overrides as the extract hold: an unspent `allow_shrink_once` naming
    this attempt id, or a run-level --allow-shrink, re-runs the update so the
    guard can accept it.
    """
    if (
        refusal_held(prior_state, key)
        and shrink_grant != current_manifest.attempt_id
        and not allow_shrink
    ):
        return ACTION_SKIP
    return ACTION_UPDATE


def _shrink_floor(old_count: int, tolerance: float) -> int:
    """Smallest new count accepted for `old_count` at `tolerance`.

    Rounds up, so a tolerance can never be widened by truncation, and a repo with
    very few nodes still gets exact-match semantics: at old_count=3 and a 10%
    tolerance the floor is 3, i.e. no shrink is tolerated where a single node is a
    third of the graph.
    """
    if old_count <= 0 or tolerance <= 0.0:
        return old_count
    return math.ceil(old_count * (1.0 - tolerance))


def _classify_shrink(
    old_hash: str,
    new_hash: str,
    old_counts: tuple[int, int],
    new_counts: tuple[int, int],
    tolerance: float = 0.0,
) -> str:
    """C21 shrink-guard defense: never trust CLI exit code alone.

    Mirrors the real CLI's own shrink-guard semantics (graphify/export.py
    `to_json`, "new graph has N nodes but existing graph.json has M") — compare
    structural node/edge counts, not raw file byte size (byte size is
    sensitive to incidental re-serialization/formatting differences and is
    not a reliable growth signal on its own).

    `tolerance` (fraction, 0.0 = strict) absorbs LLM extraction jitter: the
    `extract` action re-derives entities non-deterministically, so an unchanged
    repo legitimately varies by a few percent between runs and a strict guard
    deadlocks the pipeline (see SHRINK_TOLERANCE in sync/config.py). It defaults
    to strict here so direct callers keep the old contract; production passes
    `settings.shrink_tolerance`.
    """
    if new_hash == old_hash:
        return STATUS_NOOP
    new_nodes, new_edges = new_counts
    old_nodes, old_edges = old_counts
    if new_nodes < _shrink_floor(old_nodes, tolerance) or new_edges < _shrink_floor(
        old_edges, tolerance
    ):
        return STATUS_SHRINK_REFUSED
    return STATUS_UPDATED


# Every invoker takes (graphify_bin, root, collection_path, staging_home) plus a
# keyword `policy`. The staging home is per-run and per-repo: both calls parse
# untrusted repository source under a substituted HOME so upstream cannot read
# the operator's ~/.graphify provider config (see
# graphify_cli._isolated_home_env). `collection_path` is passed explicitly so
# the sandbox binds the registry-declared output directory rather than resolving
# the `graphify-out` symlink that lives inside the scanned repository.
_INVOKERS: dict[str, Callable[..., graphify_cli.CliResult]] = {
    ACTION_UPDATE: graphify_cli.run_update,
    ACTION_EXTRACT: graphify_cli.run_extract,
    ACTION_BOOTSTRAP: graphify_cli.run_extract,
}
# Overrides for `builder: graphify-php` entries. Bootstrap and extract stay on
# plain graphify: the wrapper exists to keep resolver edges across updates.
_PHP_INVOKERS: dict[str, Callable[..., graphify_cli.CliResult]] = {
    ACTION_UPDATE: graphify_cli.run_php_update,
}

# graphify-php-build exit codes that fail a PHP update; the wrapper has already
# restored graph.json. 2 and 3: an expected resolver failed. 4: its own shrink
# guard refused although GRAPHIFY_MESH_MANAGED=1 disables it, so the env did not
# reach the child. That stays failed, never held: a hold would hide the fault.
_PHP_EXIT_NOTES = {
    2: "graphify-php-build: an expected resolver failed, graph.json left as before",
    3: "graphify-php-build: an expected resolver failed, graph.json left as before",
    4: (
        "graphify-php-build: its own shrink guard refused although GRAPHIFY_MESH_MANAGED=1 "
        "should disable it; wiring fault (env not reaching the child), graph.json left as before"
    ),
}

# graphify-shrink-check exit codes. Anything else, and any report that does not
# parse, is a tool error and falls back to the count floor.
_SHRINK_CHECK_VERDICTS = {0: "accept", 1: "refuse"}
SHRINK_CHECK_TIMEOUT_SECONDS = 300
SHRINK_CHECK_TOP_FILES = 5
SHRINK_CHECK_PROBE_TIMEOUT_SECONDS = 30

# graphify-php's shrink attribution sidecar is written by the repo build, so it is
# untrusted input. Above this size it is not handed to graphify-shrink-check at all.
ATTRIBUTION_MAX_BYTES = 32 * 1024 * 1024
# graphify-php shrink.py ATTRIBUTION_SCHEMA / ATTRIBUTION_PRODUCER.
ATTRIBUTION_SCHEMA = 1
ATTRIBUTION_PRODUCER = "graphify-php"
_ATTRIBUTION_FLAG = "--attribution"
_ATTRIBUTION_FLAG_RE = re.compile(r"(?<![\w-])--attribution(?![\w-])")

# All four refusals of graphify's shrink guard (watch.py `_check_shrink`, 0.9.103)
# carry this; two lack "Pass --force". The unreadable-graph refusal uses ";" and
# --force does not bypass it. `update` exits 1; graphify-php-build passes it on.
GRAPHIFY_SHRINK_REFUSAL_MARKER = "Refusing to overwrite —"

# How much of a failing child's stderr reaches the log line, and where the rest
# goes. The log line used to carry `stderr[:300]`, which is the wrong end of the
# stream: the CLI prints its warnings first and its traceback last, so a repo
# that failed on 2026-09-22 reported only a RuntimeWarning about semantic-cache
# vintages and nothing about why it exited 1. The tail is what names the cause.
FAILURE_REASON_TAIL = 2000
FAILURE_LOG_NAME = ".graphify_sync_error.log"


def _failure_reason(returncode: int, stderr: str, collection_path: Path) -> str:
    """Build the outcome's reason: the tail of stderr, with the whole of it on disk.

    The full stream is written to `<collection>/.graphify_sync_error.log`, next to
    the other `.graphify_*` state files, and overwritten on each failure — it is a
    "why did the last run fail" file, not a history. A log that cannot be written
    (read-only collection, full disk) must not turn a reported failure into an
    unreported crash, so that error is swallowed and only the tail is returned.
    """
    text = (stderr or "").strip()
    if not text:
        return f"exit={returncode}: no stderr"

    log_note = ""
    try:
        log_path = collection_path / FAILURE_LOG_NAME
        log_path.write_text(text, encoding="utf-8")
        log_note = f" [full stderr: {log_path}]"
    except OSError:
        pass

    if len(text) <= FAILURE_REASON_TAIL:
        return f"exit={returncode}: {text}{log_note}"
    return (
        f"exit={returncode}: ...(first {len(text) - FAILURE_REASON_TAIL} chars in the log)... "
        f"{text[-FAILURE_REASON_TAIL:]}{log_note}"
    )


def apply_action(
    repo_id: str,
    graphify_bin: str,
    root: Path,
    collection_path: Path,
    action: str,
    current_manifest: SourceDigest,
    staging_home: Path,
    *,
    allow_shrink: bool = False,
    shrink_tolerance: float = 0.0,
    shrink_grant: str | None = None,
    sandbox_policy: graphify_cli.SandboxPolicy | None = None,
    builder: str = BUILDER_GRAPHIFY,
    php_build_bin: str = "graphify-php-build",
    shrink_policy: str = SHRINK_POLICY_COUNT,
    shrink_check_bin: str = "graphify-shrink-check",
    producer_changed: bool = False,
    force_rebuild: bool = False,
    retried_documents: frozenset[str] = frozenset(),
) -> ProjectOutcome:
    """Run `action` for one repo and judge the result with the shrink guard.

    `retried_documents` is UNSTAMPED_DOCUMENTS_KEY from prior state: documents
    that already had their one retry, so an extract leaving them unstamped is
    accepted with a warning instead of held again.

    `producer_changed` (the pipeline's `_producer_changed`, label switch
    included) only tells graphify-shrink-check `--producer-changed`.
    `force_rebuild` (a producer upgrade, or a --allow-shrink release of this
    repo's held refusal) and a `shrink_grant` naming this attempt are the only
    causes of `update --force`.

    That decision matters only for builder graphify. graphify-php-build appends
    `--force` to every `update` itself, because an incremental update cannot
    re-resolve PHP calls from unchanged files. A graphify-php update is always a
    full rebuild with graphify's own shrink guard waived; this function's guard
    and graphify-shrink-check still judge it.
    """
    graph_path = collection_path / "graph.json"
    manifest_path = collection_path / "manifest.json"
    dirty = is_worktree_dirty(root)

    if action == ACTION_SKIP:
        return ProjectOutcome(
            repo_id, action, STATUS_UNCHANGED, dirty_worktree=dirty, new_manifest=current_manifest
        )

    # Single read of graph.json: hash + counts from the same buffer.
    old_hash, old_counts_opt = graph_hash_and_counts(graph_path)
    old_counts = old_counts_opt or (0, 0)
    snapshot_path = _take_snapshot(graph_path)
    # manifest.json is graphify's per-file record of what graph.json was built
    # from. Restoring graph.json alone left a manifest describing the refused
    # graph, which the shrink check and the next incremental update both read.
    manifest_snapshot = _take_snapshot(manifest_path) if snapshot_path is not None else None

    def restore_last_good() -> None:
        _restore_snapshot(snapshot_path, graph_path)
        _restore_manifest(manifest_snapshot, manifest_path, had_graph=snapshot_path is not None)

    def cleanup() -> None:
        _cleanup_snapshot(snapshot_path)
        _cleanup_snapshot(manifest_snapshot)

    # A grant for exactly this attempt is spent by any accepted outcome, not only
    # by an accepted shrink: under `--force` graphify's guard never refuses, so
    # the mesh guard may accept without a shrink and leave the token reusable.
    grant_matches = shrink_grant is not None and shrink_grant == current_manifest.attempt_id
    spent_grant = shrink_grant if grant_matches else None
    # Incremental `update` keeps unchanged files' old extraction, so a new producer
    # needs graphify's full rebuild. An operator release needs it too: only
    # `--force` gets past graphify's own shrink guard. The guard below still judges.
    update_kwargs = {"force": force_rebuild or grant_matches} if action == ACTION_UPDATE else {}
    php_invoker = _PHP_INVOKERS.get(action) if builder == BUILDER_GRAPHIFY_PHP else None
    try:
        if php_invoker is not None:
            result = php_invoker(
                graphify_bin,
                root,
                collection_path,
                staging_home,
                policy=sandbox_policy,
                php_build_bin=php_build_bin,
                **update_kwargs,
            )
        else:
            result = _INVOKERS[action](
                graphify_bin,
                root,
                collection_path,
                staging_home,
                policy=sandbox_policy,
                **update_kwargs,
            )
    except BaseException:
        # The invoker raises on a sandbox that is configured but unavailable and
        # on a write path that escapes containment. Both used to leave the
        # mkstemp snapshot behind, one file per refused launch, in a directory
        # nothing sweeps.
        cleanup()
        raise

    if not result.ok:
        restore_last_good()
        status = STATUS_BOOTSTRAP_FAILED if action == ACTION_BOOTSTRAP else STATUS_FAILED
        cleanup()
        reason = _failure_reason(result.returncode, result.stderr, collection_path)
        if (
            action == ACTION_UPDATE
            and result.returncode == 1
            and GRAPHIFY_SHRINK_REFUSAL_MARKER in result.stderr
        ):
            # graphify's own guard refused. Held and released like the guard
            # below, since the release re-runs this update with `--force`.
            return ProjectOutcome(
                repo_id,
                action,
                STATUS_SHRINK_REFUSED,
                reason=(
                    f"graphify refused to shrink the graph ({reason}); last-good graph.json "
                    f"restored. If this shrink is intended, authorize this one attempt with "
                    f'"allow_shrink_once": "{current_manifest.attempt_id}" '
                    f"on this repo's registry entry"
                ),
                dirty_worktree=dirty,
                refused_manifest=current_manifest,
            )
        note = _PHP_EXIT_NOTES.get(result.returncode) if php_invoker is not None else None
        return ProjectOutcome(
            repo_id,
            action,
            status,
            reason=f"{note}; {reason}" if note else reason,
            dirty_worktree=dirty,
        )

    if action == ACTION_BOOTSTRAP:
        if not graph_path.exists():
            cleanup()
            return ProjectOutcome(
                repo_id,
                action,
                STATUS_BOOTSTRAP_FAILED,
                reason="cli exited 0 but no graph.json was produced",
                dirty_worktree=dirty,
            )
        cleanup()
        return ProjectOutcome(
            repo_id,
            action,
            STATUS_BOOTSTRAPPED,
            dirty_worktree=dirty,
            new_manifest=current_manifest,
            graph_content_hash=file_content_hash(graph_path),
        )

    if not graph_path.exists():
        restore_last_good()
        cleanup()
        return ProjectOutcome(
            repo_id,
            action,
            STATUS_FAILED,
            reason="cli exited 0 but graph.json disappeared",
            dirty_worktree=dirty,
        )

    # Single read of the freshly written graph.json: hash + counts from the
    # same buffer.
    new_hash, new_counts_opt = graph_hash_and_counts(graph_path)
    new_counts = new_counts_opt or (0, 0)

    if old_hash is None:
        # No prior file existed even though has_graph was assumed true
        # upstream (race) — accept whatever was produced.
        cleanup()
        return ProjectOutcome(
            repo_id,
            action,
            STATUS_UPDATED,
            dirty_worktree=dirty,
            new_manifest=current_manifest,
            graph_content_hash=new_hash,
            consumed_shrink_grant=spent_grant,
        )

    report: dict | None = None
    if shrink_policy != SHRINK_POLICY_COUNT and new_hash != old_hash:
        report = _shrink_check_report(
            shrink_check_bin,
            shrink_policy,
            shrink_tolerance,
            producer_changed,
            root,
            snapshot_path,
            graph_path,
            manifest_snapshot,
            manifest_path,
            # Only graphify-php-build writes the sidecar, and it deletes a stale one first.
            # After a plain-graphify run (php extract included) any sidecar is a leftover.
            attribution=(
                graphify_cli.php_attribution_path(collection_path)
                if php_invoker is not None
                else None
            ),
        )
    if report is None:
        outcome_status = _classify_shrink(
            old_hash, new_hash or "", old_counts, new_counts, shrink_tolerance
        )
    else:
        outcome_status = STATUS_SHRINK_REFUSED if report["verdict"] == "refuse" else STATUS_UPDATED

    if outcome_status == STATUS_SHRINK_REFUSED and allow_shrink:
        # Operator-authorized (--allow-shrink): accept the smaller graph and
        # advance state, otherwise the repo would re-extract and be refused
        # again on every subsequent run.
        cleanup()
        return ProjectOutcome(
            repo_id,
            action,
            STATUS_UPDATED,
            reason=(
                f"shrink accepted (operator-authorized via --allow-shrink): "
                f"old={old_counts} new={new_counts}"
            ),
            dirty_worktree=dirty,
            new_manifest=current_manifest,
            graph_content_hash=new_hash,
            consumed_shrink_grant=spent_grant,
            explained_shrink=_accepted_drop(old_counts, new_counts),
        )

    if outcome_status == STATUS_SHRINK_REFUSED and grant_matches:
        # Per-repo, single-use authorization: the registry entry carries the
        # attempt id (code + semantic digests) of the attempt the operator
        # inspected and approved. Matching it exactly is the whole guarantee — a
        # stale token (the source moved on, code-only changes included),
        # a token for another repo, or no token at all leaves the guard armed,
        # which is the difference from --allow-shrink waving through every repo
        # in the run. The grant is reported back so the caller can spend it.
        cleanup()
        return ProjectOutcome(
            repo_id,
            action,
            STATUS_UPDATED,
            reason=(
                f"shrink accepted (registry allow_shrink_once={shrink_grant}): "
                f"old={old_counts} new={new_counts}"
            ),
            dirty_worktree=dirty,
            new_manifest=current_manifest,
            graph_content_hash=new_hash,
            consumed_shrink_grant=shrink_grant,
            # The operator accepted all of it. Left at (0, 0), the global guard
            # refused the same loss and blocked the publish while the grant was
            # already spent (2026-10-13).
            explained_shrink=_accepted_drop(old_counts, new_counts),
        )

    if outcome_status == STATUS_SHRINK_REFUSED:
        restore_last_good()
        cleanup()
        verdict = (
            _refusal_detail(report, shrink_policy)
            if report is not None
            else "cli reported success but node/edge counts did not grow"
        )
        return ProjectOutcome(
            repo_id,
            action,
            STATUS_SHRINK_REFUSED,
            reason=(
                f"{verdict} "
                f"(old={old_counts} new={new_counts}); last-good graph.json restored. "
                f"If this shrink is intended, authorize this one attempt with "
                f'"allow_shrink_once": "{current_manifest.attempt_id}" '
                f"on this repo's registry entry"
            ),
            dirty_worktree=dirty,
            # Accepted state deliberately stays put (no new_manifest) so the
            # last-good graph remains authoritative, but the refused digest is
            # reported so the caller can record it and stop re-extracting the
            # same unchanged source every run.
            refused_manifest=current_manifest,
        )

    checked = action == ACTION_EXTRACT and spent_grant is None
    unstamped = (
        _unstamped_documents(
            manifest_snapshot,
            manifest_path,
            root,
            current_manifest.semantic_paths,
            retried_documents,
        )
        if checked
        else ([], [])
    )
    cleanup()
    newly_unstamped, still_unstamped = unstamped
    if newly_unstamped:
        # graphify exits 0 on a truncated or hollow LLM answer but leaves the
        # document's semantic_hash "" (or writes no row for a new document) so
        # its own next extract re-dispatches it. Keep the new graph and hold
        # accepted state so the next run retries. The retry is bounded: the
        # caller records these paths, so a second miss is accepted below.
        reason = (
            f"extract left {len(newly_unstamped)} new or edited document(s) unstamped "
            f"(truncated or empty LLM result): {_path_list(newly_unstamped)}; "
            f"graph kept, state held so the next run retries once"
        )
        log.warning("%s: %s", repo_id, reason)
        return ProjectOutcome(
            repo_id,
            action,
            outcome_status,
            reason=reason,
            dirty_worktree=dirty,
            graph_content_hash=new_hash,
            explained_shrink=_explained_shrink(report, old_counts, new_counts),
            unstamped_documents=tuple(sorted({*newly_unstamped, *still_unstamped})),
        )
    retry_note = ""
    if still_unstamped:
        retry_note = (
            f"{len(still_unstamped)} document(s) still unstamped after retry, accepted: "
            f"{_path_list(still_unstamped)}; graphify re-dispatches any it reads "
            f"on the next extract"
        )
        log.warning("%s: %s", repo_id, retry_note)
    # State advances to current_manifest for BOTH remaining statuses here
    # (STATUS_UPDATED and STATUS_NOOP): a NOOP means the source changed but
    # the CLI produced byte-identical output — the source digest must still
    # advance, or every subsequent run would re-invoke the CLI for the same
    # no-op change. The two branches were always identical; the old
    # conditional was dead.
    return ProjectOutcome(
        repo_id,
        action,
        outcome_status,
        reason=retry_note,
        dirty_worktree=dirty,
        new_manifest=current_manifest,
        graph_content_hash=new_hash,
        consumed_shrink_grant=spent_grant,
        explained_shrink=_explained_shrink(report, old_counts, new_counts),
        unstamped_documents=tuple(still_unstamped) if checked else None,
    )


def _shrink_check_report(
    check_bin: str,
    policy: str,
    tolerance: float,
    producer_changed: bool,
    root: Path,
    old_graph: Path | None,
    new_graph: Path,
    old_manifest: Path | None,
    new_manifest: Path,
    attribution: Path | None = None,
) -> dict | None:
    """Run graphify-shrink-check; its report, or None to use the count floor.

    `attribution` is graphify-php's sidecar for this build. It reaches the tool as
    `--attribution` only when the file exists, the installed tool knows the flag,
    and the file passes `_attribution_copy`; otherwise the check runs without it.

    None covers every way the tool cannot answer: not installed, no manifest on
    either side to attribute against, exit 2, a crash, a timeout, or an exit 0
    whose report does not parse. The mesh must keep syncing on hosts without
    graphify-php, so each of those warns once per process and falls back.
    Exit 1, or a "refuse" verdict on exit 0, is a refusal whatever the other
    channel says: a refuse report comes back, synthesized when there is none.

    Not sandboxed: the tool reads JSON files and the root's file list and writes
    nothing. Its environment still goes through the child allowlist, so the
    mesh's own tokens do not reach it.
    """
    resolved = shutil.which(check_bin)
    if resolved is None:
        graphify_cli._log_once(
            f"shrink-check-missing:{check_bin}",
            logging.WARNING,
            "shrink check: %r not found on PATH — using the count floor (policy %s ignored)",
            check_bin,
            policy,
        )
        return None
    if old_graph is None or old_manifest is None or not new_manifest.exists():
        log.info("shrink check: manifest.json missing on one side — using the count floor")
        return None
    argv = [
        resolved,
        "--old-graph",
        str(old_graph),
        "--new-graph",
        str(new_graph),
        "--old-manifest",
        str(old_manifest),
        "--new-manifest",
        str(new_manifest),
        "--root",
        str(root),
        "--policy",
        policy,
        "--tolerance",
        repr(tolerance),
    ]
    if producer_changed:
        argv.append("--producer-changed")
    attribution_copy = None
    if (
        attribution is not None
        and os.path.lexists(attribution)
        and _attribution_supported(resolved)
    ):
        attribution_copy = _attribution_copy(attribution)
    if attribution_copy is not None:
        argv += [_ATTRIBUTION_FLAG, str(attribution_copy)]
    env = {k: v for k, v in os.environ.items() if graphify_cli._child_env_allows(k)}
    try:
        proc = subprocess.run(  # noqa: S603 - structured argv, binary from operator config
            argv,
            capture_output=True,
            text=True,
            timeout=SHRINK_CHECK_TIMEOUT_SECONDS,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _warn_shrink_check_error(check_bin, f"did not run: {exc}")
        return None
    finally:
        if attribution_copy is not None:
            attribution_copy.unlink(missing_ok=True)
    expected = _SHRINK_CHECK_VERDICTS.get(proc.returncode)
    if expected is None:
        tail = (proc.stderr or "").strip()[-500:]
        _warn_shrink_check_error(check_bin, f"exit={proc.returncode}: {tail}")
        return None
    report = _parse_shrink_report(proc.stdout)
    verdict = report.get("verdict") if report is not None else None
    # Any refusal signal wins over the count floor (2026-10-13). An exit code
    # and a verdict that disagree are a broken tool, and falling back there
    # let the count floor accept a shrink the tool had refused on one channel.
    # Only a tool that said nothing usable on exit 0 falls back.
    if expected == "refuse":
        if verdict == "refuse":
            return report
        detail = "no parseable report" if report is None else f"report verdict {verdict!r}"
        _warn_shrink_check_mismatch(check_bin, f"exit=1 with {detail}")
        return {
            "verdict": "refuse",
            "reasons": [f"exit 1 with {detail}; refused on the exit code"],
        }
    if verdict == "refuse":
        _warn_shrink_check_mismatch(check_bin, "exit=0 with report verdict 'refuse'")
        return report
    if verdict != "accept":
        _warn_shrink_check_error(check_bin, "exit=0 without a JSON report carrying a verdict")
        return None
    return report


def _parse_shrink_report(stdout: str) -> dict | None:
    try:
        report = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    return report if isinstance(report, dict) else None


def _attribution_supported(resolved: str) -> bool:
    """Whether this graphify-shrink-check accepts `--attribution`; cached per binary version.

    The tool exposes no version or capability flag, so its argparse `--help` is probed
    for the option string itself, which argparse prints verbatim. Keyed on mtime too,
    so an in-place upgrade during a long-running process is seen.
    """
    try:
        mtime_ns = os.stat(resolved).st_mtime_ns
    except OSError:
        return False
    return _shrink_check_supports_attribution(resolved, mtime_ns)


@functools.lru_cache(maxsize=16)
def _shrink_check_supports_attribution(resolved: str, mtime_ns: int) -> bool:
    del mtime_ns  # cache key only
    env = {k: v for k, v in os.environ.items() if graphify_cli._child_env_allows(k)}
    try:
        proc = subprocess.run(  # noqa: S603 - structured argv, binary from operator config
            [resolved, "--help"],
            capture_output=True,
            text=True,
            timeout=SHRINK_CHECK_PROBE_TIMEOUT_SECONDS,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.info("shrink check: --help probe of %s failed (%s); no attribution", resolved, exc)
        return False
    supported = proc.returncode == 0 and bool(_ATTRIBUTION_FLAG_RE.search(proc.stdout or ""))
    if not supported:
        log.info("shrink check: %s has no %s flag; no attribution", resolved, _ATTRIBUTION_FLAG)
    return supported


def _attribution_copy(sidecar: Path) -> Path | None:
    """A private copy of a well-formed sidecar for the tool to read, or None.

    The sidecar sits in a path the sandboxed build can write, so it is untrusted:
    no symlinks, regular files only, at most ATTRIBUTION_MAX_BYTES, UTF-8 JSON
    object with graphify-php's schema and producer. The tool validates the rest
    (shrink.py `_parse_attribution`). Any defect means no attribution, never a
    failed repo. The copy also keeps the tool off the build-writable path.
    """
    try:
        raw = _read_attribution(sidecar)
    except _SidecarRejectedError as exc:
        log.warning("shrink check: attribution sidecar %s ignored: %s", sidecar, exc)
        return None
    fd, name = tempfile.mkstemp(prefix="graphify-mesh-attribution-", suffix=".json")
    with os.fdopen(fd, "wb") as out:
        out.write(raw)
    return Path(name)


class _SidecarRejectedError(Exception):
    """Why an attribution sidecar is not handed to graphify-shrink-check."""


def _read_attribution(sidecar: Path) -> bytes:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    too_big = f"larger than {ATTRIBUTION_MAX_BYTES} bytes"
    try:
        fd = os.open(sidecar, flags)
    except OSError as exc:
        raise _SidecarRejectedError(f"cannot open ({exc.strerror or exc})") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise _SidecarRejectedError("not a regular file")
        if info.st_size > ATTRIBUTION_MAX_BYTES:
            raise _SidecarRejectedError(too_big)
        raw = b""
        while len(raw) <= ATTRIBUTION_MAX_BYTES:
            chunk = os.read(fd, ATTRIBUTION_MAX_BYTES + 1 - len(raw))
            if not chunk:
                break
            raw += chunk
    except OSError as exc:
        raise _SidecarRejectedError(f"unreadable ({exc.strerror or exc})") from None
    finally:
        os.close(fd)
    if len(raw) > ATTRIBUTION_MAX_BYTES:
        raise _SidecarRejectedError(too_big)
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise _SidecarRejectedError("not UTF-8 JSON") from None
    if not isinstance(data, dict):
        raise _SidecarRejectedError("not a JSON object")
    schema = data.get("schema")
    if isinstance(schema, bool) or schema != ATTRIBUTION_SCHEMA:
        raise _SidecarRejectedError(f"schema {schema!r}, expected {ATTRIBUTION_SCHEMA}")
    if data.get("producer") != ATTRIBUTION_PRODUCER:
        raise _SidecarRejectedError("producer is not graphify-php")
    return raw


def _warn_shrink_check_mismatch(check_bin: str, detail: str) -> None:
    graphify_cli._log_once(
        f"shrink-check-mismatch:{check_bin}",
        logging.WARNING,
        "shrink check: %s answered inconsistently (%s) — refusing, not using the count floor",
        check_bin,
        detail,
    )


def _warn_shrink_check_error(check_bin: str, detail: str) -> None:
    graphify_cli._log_once(
        f"shrink-check-error:{check_bin}",
        logging.WARNING,
        "shrink check: %s failed (%s) — using the count floor",
        check_bin,
        detail,
    )


def _report_int(section: object, key: str) -> int:
    value = section.get(key) if isinstance(section, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _is_count(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _attributed_total(section: object) -> int:
    """`attributed_total` when the report's own figures vouch for it, else 0.

    A count no larger than the section's `lost` that equals the sum of its
    per-reason `attributed` breakdown; any other shape is ignored whole.
    """
    if not isinstance(section, dict):
        return 0
    total, lost = section.get("attributed_total"), section.get("lost")
    breakdown = section.get("attributed")
    if not (_is_count(total) and _is_count(lost) and isinstance(breakdown, dict)):
        return 0
    values = list(breakdown.values())
    if not all(_is_count(v) for v in values) or sum(values) != total or total > lost:
        return 0
    return total


def _refusal_detail(report: dict, policy: str) -> str:
    """The refusal line: the tool's reasons, then the files that lost the most."""
    reasons = [r for r in report.get("reasons") or [] if isinstance(r, str)]
    files = [f for f in report.get("files") or [] if isinstance(f, dict)]
    top = sorted(files, key=lambda f: _report_int(f, "lost"), reverse=True)
    top_text = ", ".join(
        f"{f.get('file')}:{f.get('class')}:{_report_int(f, 'lost')}"
        for f in top[:SHRINK_CHECK_TOP_FILES]
    )
    detail = f"graphify-shrink-check refused (policy={policy}): " + (
        "; ".join(reasons) or "no reason given"
    )
    return f"{detail}; top files: {top_text}" if top_text else detail


def _explained_shrink(
    report: dict | None, old_counts: tuple[int, int], new_counts: tuple[int, int]
) -> tuple[int, int]:
    """Loss the tool attributed to source changes or to graphify-php's own
    recorded removals (`attributed_total`), capped at the actual loss."""
    if report is None or report.get("verdict") != "accept":
        return (0, 0)
    nodes, edges = report.get("nodes"), report.get("edges")
    explained_nodes = (
        _report_int(nodes, "explained") + _report_int(nodes, "orphan") + _attributed_total(nodes)
    )
    explained_edges = _report_int(edges, "explained") + _attributed_total(edges)
    return (
        min(max(0, old_counts[0] - new_counts[0]), explained_nodes),
        min(max(0, old_counts[1] - new_counts[1]), explained_edges),
    )


def _accepted_drop(old_counts: tuple[int, int], new_counts: tuple[int, int]) -> tuple[int, int]:
    """The whole (nodes, edges) loss, for a shrink an operator accepted outright."""
    return (max(0, old_counts[0] - new_counts[0]), max(0, old_counts[1] - new_counts[1]))


def _take_snapshot(path: Path) -> Path | None:
    if not path.exists():
        return None
    fd, tmp_name = tempfile.mkstemp(prefix="graphify-mesh-sync-snapshot-", suffix=".json")
    os.close(fd)
    snapshot = Path(tmp_name)
    shutil.copy2(path, snapshot)
    return snapshot


def _restore_manifest(snapshot: Path | None, manifest_path: Path, *, had_graph: bool) -> None:
    """Put manifest.json back in step with the restored graph.json.

    No snapshot but a last-good graph means the failed run created the manifest,
    so it describes a graph that is not on disk any more and is removed.
    """
    if snapshot is not None:
        _restore_snapshot(snapshot, manifest_path)
    elif had_graph:
        manifest_path.unlink(missing_ok=True)


def _restore_snapshot(snapshot_path: Path | None, graph_path: Path) -> None:
    if snapshot_path is None:
        return
    # Copying straight onto the live graph.json leaves a truncated last-good
    # graph if the process dies mid-copy, and the next run would take that
    # truncation as its baseline. Stage next to the target (same filesystem,
    # so os.replace is atomic) and swap.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{graph_path.name}.restore.", dir=str(graph_path.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copy2(snapshot_path, tmp_path)
        os.replace(tmp_path, graph_path)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise


def _cleanup_snapshot(snapshot_path: Path | None) -> None:
    if snapshot_path is not None and snapshot_path.exists():
        snapshot_path.unlink()


_UNSTAMPED_LIST_MAX = 5


def _path_list(paths: list[str]) -> str:
    shown = ", ".join(paths[:_UNSTAMPED_LIST_MAX])
    extra = len(paths) - _UNSTAMPED_LIST_MAX
    return f"{shown} (+{extra} more)" if extra > 0 else shown


def _manifest_rows(path: Path | None) -> dict[str, dict] | None:
    """graphify's manifest.json as {path: row}; None when absent or unreadable."""
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, dict)}


def _unstamped_documents(
    before_path: Path | None,
    after_path: Path,
    root: Path,
    documents: tuple[str, ...] = (),
    retried: frozenset[str] = frozenset(),
) -> tuple[list[str], list[str]]:
    """Documents an extract left unstamped in graphify's manifest.

    Unstamped is a row with `semantic_hash` "" or, for a document in this
    run's digest (`documents`), no row at all: graphify writes none for a new
    document whose LLM result was partial. Returns (newly, still): `still`
    already had its retry (in `retried`, or "" before this run) and is
    accepted; `newly` holds state once. A document graphify never reads also
    has no row; `retried` bounds it to one retry until it is stamped. Only the
    mesh's semantic suffixes count, so code rows never qualify. Unreadable
    manifests give ([], []): accept as before.
    """
    before = _manifest_rows(before_path)
    after = _manifest_rows(after_path)
    if before is None or after is None:
        return [], []
    before = _relative_rows(before, root)
    after = _relative_rows(after, root)
    suffixes = CURRENT_CLASSIFICATION.semantic_extensions
    in_digest = {doc for doc in documents if Path(doc).suffix.lower() in suffixes}
    candidates = {doc for doc in in_digest if doc not in after}
    for key, row in after.items():
        if Path(key).suffix.lower() not in suffixes or row.get("semantic_hash") != "":
            continue
        # A "" row with no earlier row and outside the digest is not a
        # document this run asked graphify to read.
        if key in before or key in in_digest:
            candidates.add(key)
    newly: list[str] = []
    still: list[str] = []
    for key in sorted(candidates):
        prior = before.get(key)
        if key in retried or (prior is not None and not prior.get("semantic_hash")):
            still.append(key)
        else:
            newly.append(key)
    return newly, still


def _relative_rows(rows: dict[str, dict], root: Path) -> dict[str, dict]:
    """`rows` keyed root-relative POSIX, the way graphify stores in-root files.

    graphify writes absolute keys for out-of-root files and in legacy
    manifests; an absolute key under `root` is relativized to match the digest.
    """
    anchors = {root}
    try:
        anchors.add(root.resolve())
    except (OSError, RuntimeError):
        pass
    out: dict[str, dict] = {}
    for key, row in rows.items():
        path = Path(key)
        if path.is_absolute():
            for anchor in anchors:
                try:
                    key = path.relative_to(anchor).as_posix()
                    break
                except ValueError:
                    continue
        out[key] = row
    return out
