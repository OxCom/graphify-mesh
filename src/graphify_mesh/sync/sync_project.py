"""Per-project decision + invocation + shrink-guard defense (WS1 items 2-4).

Decision (`decide_action`) and outcome classification
(`_classify_post_invoke`) both use dict-dispatch instead of if/elif chains,
per project code style rules.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from graphify_mesh.sync import graphify_cli
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
) -> str:
    """Pick the action for one repo.

    `refusal_key` is `refusal_key(current_manifest, <update producer>, policy)`.
    It holds a refused `update` the way `refused_semantic_hash` holds a refused
    extract; None disables that hold (direct callers keep the old contract).

    `allow_shrink` is the run-level --allow-shrink. It releases both holds: the
    guard would accept the held attempt this run, and holding it instead left
    the flag with nothing to act on for exactly the repos it was passed for.
    """
    if not has_graph:
        return ACTION_BOOTSTRAP
    if prior_state is None:
        # First time this repo's manifest is observed but a graph already
        # exists on disk (e.g. pre-existing curated graph). Seed state with a
        # cheap AST refresh rather than forcing an LLM extract every run.
        return ACTION_UPDATE
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
            current_manifest.semantic_hash == prior_state.get("refused_semantic_hash")
            and int(prior_state.get("refusal_streak") or 0) >= REFUSAL_RETRY_LIMIT
            and shrink_grant != current_manifest.attempt_id
            and not allow_shrink
        ):
            return ACTION_SKIP
        return ACTION_EXTRACT
    if current_manifest.code_hash != prior_state.get("code_hash"):
        return _update_unless_held(
            prior_state, current_manifest, shrink_grant, refusal_key, allow_shrink
        )
    return ACTION_SKIP


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
        key is not None
        and prior_state.get("refused_key") == key
        and int(prior_state.get("refusal_streak") or 0) >= REFUSAL_RETRY_LIMIT
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
# Overrides for `builder: graphify-php` entries. Bootstrap and extract stay on
# plain graphify: the wrapper exists to keep resolver edges across updates.
_PHP_INVOKERS: dict[str, Callable[..., graphify_cli.CliResult]] = {
    ACTION_UPDATE: graphify_cli.run_php_update,
}

# graphify-php-build exit codes that mean "an expected resolver failed". The
# wrapper has already restored graph.json; the note says so in the reason.
_PHP_EXIT_NOTES = {
    2: "graphify-php-build: an expected resolver failed, graph.json left as before",
    3: "graphify-php-build: an expected resolver failed, graph.json left as before",
}

# graphify-shrink-check exit codes. Anything else, and any report that does not
# parse, is a tool error and falls back to the count floor.
_SHRINK_CHECK_VERDICTS = {0: "accept", 1: "refuse"}
SHRINK_CHECK_TIMEOUT_SECONDS = 300
SHRINK_CHECK_TOP_FILES = 5

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
) -> ProjectOutcome:
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
            )
        else:
            result = _INVOKERS[action](
                graphify_bin, root, collection_path, staging_home, policy=sandbox_policy
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
            explained_shrink=_accepted_drop(old_counts, new_counts),
        )

    if outcome_status == STATUS_SHRINK_REFUSED and shrink_grant == current_manifest.attempt_id:
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

    cleanup()
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
        explained_shrink=_explained_shrink(report, old_counts, new_counts),
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
) -> dict | None:
    """Run graphify-shrink-check; its report, or None to use the count floor.

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
    """Loss the tool attributed to source changes, capped at the actual loss."""
    if report is None or report.get("verdict") != "accept":
        return (0, 0)
    nodes, edges = report.get("nodes"), report.get("edges")
    explained_nodes = _report_int(nodes, "explained") + _report_int(nodes, "orphan")
    explained_edges = _report_int(edges, "explained")
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
