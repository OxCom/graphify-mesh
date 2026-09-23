"""Per-project decision + invocation + shrink-guard defense (WS1 items 2-4).

Decision (`decide_action`) and outcome classification
(`_classify_post_invoke`) both use dict-dispatch instead of if/elif chains,
per project code style rules.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from graphify_mesh.sync import graphify_cli
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
) -> str:
    if not has_graph:
        return ACTION_BOOTSTRAP
    if prior_state is None:
        # First time this repo's manifest is observed but a graph already
        # exists on disk (e.g. pre-existing curated graph). Seed state with a
        # cheap AST refresh rather than forcing an LLM extract every run.
        return ACTION_UPDATE
    if current_manifest.semantic_hash != prior_state.get("semantic_hash"):
        # Break the refusal loop: if this exact source digest has already been
        # refused REFUSAL_RETRY_LIMIT times, re-running the extractor on
        # unchanged input just burns GPU for another refusal. Hold the last-good
        # graph instead and wait for the source to actually change.
        #
        # An unspent `allow_shrink_once` for this digest overrides that hold:
        # the operator authorized this exact refused attempt, and the only way
        # to honour it is to re-run the extract and let the guard accept the
        # result this time.
        if (
            current_manifest.semantic_hash == prior_state.get("refused_semantic_hash")
            and int(prior_state.get("refusal_streak") or 0) >= REFUSAL_RETRY_LIMIT
            and shrink_grant != current_manifest.semantic_hash
        ):
            return ACTION_SKIP
        return ACTION_EXTRACT
    if current_manifest.code_hash != prior_state.get("code_hash"):
        return ACTION_UPDATE
    return ACTION_SKIP


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

    Mirrors the real CLI's own shrink-guard semantics (export.py:270-286,
    "new graph has N nodes but existing graph.json has M") — compare
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
) -> ProjectOutcome:
    graph_path = collection_path / "graph.json"
    dirty = is_worktree_dirty(root)

    if action == ACTION_SKIP:
        return ProjectOutcome(
            repo_id, action, STATUS_UNCHANGED, dirty_worktree=dirty, new_manifest=current_manifest
        )

    snapshot_path: Path | None = None
    # Single read of graph.json: hash + counts from the same buffer.
    old_hash, old_counts_opt = graph_hash_and_counts(graph_path)
    old_counts = old_counts_opt or (0, 0)
    if graph_path.exists():
        fd, tmp_name = tempfile.mkstemp(prefix="graphify-mesh-sync-snapshot-", suffix=".json")
        os.close(fd)
        snapshot_path = Path(tmp_name)
        shutil.copy2(graph_path, snapshot_path)

    invoker = _INVOKERS[action]
    try:
        result = invoker(graphify_bin, root, collection_path, staging_home, policy=sandbox_policy)
    except BaseException:
        # The invoker raises on a sandbox that is configured but unavailable and
        # on a write path that escapes containment. Both used to leave the
        # mkstemp snapshot behind, one file per refused launch, in a directory
        # nothing sweeps.
        _cleanup_snapshot(snapshot_path)
        raise

    if not result.ok:
        _restore_snapshot(snapshot_path, graph_path)
        status = STATUS_BOOTSTRAP_FAILED if action == ACTION_BOOTSTRAP else STATUS_FAILED
        _cleanup_snapshot(snapshot_path)
        return ProjectOutcome(
            repo_id,
            action,
            status,
            reason=_failure_reason(result.returncode, result.stderr, collection_path),
            dirty_worktree=dirty,
        )

    if action == ACTION_BOOTSTRAP:
        if not graph_path.exists():
            _cleanup_snapshot(snapshot_path)
            return ProjectOutcome(
                repo_id,
                action,
                STATUS_BOOTSTRAP_FAILED,
                reason="cli exited 0 but no graph.json was produced",
                dirty_worktree=dirty,
            )
        _cleanup_snapshot(snapshot_path)
        return ProjectOutcome(
            repo_id,
            action,
            STATUS_BOOTSTRAPPED,
            dirty_worktree=dirty,
            new_manifest=current_manifest,
            graph_content_hash=file_content_hash(graph_path),
        )

    if not graph_path.exists():
        _restore_snapshot(snapshot_path, graph_path)
        _cleanup_snapshot(snapshot_path)
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
        _cleanup_snapshot(snapshot_path)
        return ProjectOutcome(
            repo_id,
            action,
            STATUS_UPDATED,
            dirty_worktree=dirty,
            new_manifest=current_manifest,
            graph_content_hash=new_hash,
        )

    outcome_status = _classify_shrink(
        old_hash, new_hash or "", old_counts, new_counts, shrink_tolerance
    )

    if outcome_status == STATUS_SHRINK_REFUSED and allow_shrink:
        # Operator-authorized (--allow-shrink): accept the smaller graph and
        # advance state, otherwise the repo would re-extract and be refused
        # again on every subsequent run.
        _cleanup_snapshot(snapshot_path)
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
        )

    if outcome_status == STATUS_SHRINK_REFUSED and shrink_grant == current_manifest.semantic_hash:
        # Per-repo, single-use authorization: the registry entry carries the
        # digest of the attempt the operator inspected and approved. Matching it
        # exactly is the whole guarantee — a stale token (the source moved on),
        # a token for another repo, or no token at all leaves the guard armed,
        # which is the difference from --allow-shrink waving through every repo
        # in the run. The grant is reported back so the caller can spend it.
        _cleanup_snapshot(snapshot_path)
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
        )

    if outcome_status == STATUS_SHRINK_REFUSED:
        _restore_snapshot(snapshot_path, graph_path)
        _cleanup_snapshot(snapshot_path)
        return ProjectOutcome(
            repo_id,
            action,
            STATUS_SHRINK_REFUSED,
            reason=(
                f"cli reported success but node/edge counts did not grow "
                f"(old={old_counts} new={new_counts}); last-good graph.json restored. "
                f"If this shrink is intended, authorize this one attempt with "
                f'"allow_shrink_once": "{current_manifest.semantic_hash}" '
                f"on this repo's registry entry"
            ),
            dirty_worktree=dirty,
            # Accepted state deliberately stays put (no new_manifest) so the
            # last-good graph remains authoritative, but the refused digest is
            # reported so the caller can record it and stop re-extracting the
            # same unchanged source every run.
            refused_manifest=current_manifest,
        )

    _cleanup_snapshot(snapshot_path)
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
        dirty_worktree=dirty,
        new_manifest=current_manifest,
        graph_content_hash=new_hash,
    )


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
