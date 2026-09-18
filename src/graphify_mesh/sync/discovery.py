"""Filesystem discovery + reconciliation against the registry (WS1 item 1).

Scans one or more configured scan roots, each up to a configurable depth
(default `SCAN_DEFAULT_DEPTH`, see graphify_mesh.sync.config) below the root
for `graphify-out` symlinks/dirs — depth = max nesting of the project dir
below the root (depth 2 == `<scan_root>/*/graphify-out` and
`<scan_root>/*/*/graphify-out`, e.g. AgentSpaceX's layout). The walk itself
never follows directory symlinks (a symlinked project dir is not
discovered); only the `graphify-out` link at the bottom of each candidate
dir is resolved. Both the candidate dir and its resolved `graphify-out`
target are rejected if they resolve outside the approved roots (C16
path-traversal guard). Reconciles the result against `registry.json` (the
source of truth for repo identity) to produce a report of registered /
renamed / missing / broken / removed / duplicate projects.
"""

from __future__ import annotations

import logging
import os
import stat
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from graphify_mesh.sync.config import IGNORED_DIR_NAMES, SCAN_DEFAULT_DEPTH
from graphify_mesh.sync.registry import Registry

log = logging.getLogger(__name__)


@dataclass
class DiscoveredLink:
    source_root: Path
    link_path: Path
    target: Path | None
    broken: bool = False
    rejected_traversal: bool = False


class ScanError(str):
    """One swallowed OS error, carrying the path that failed as an attribute.

    The entries of the `scan_errors` sink have to stay plain strings: callers
    log them, copy them into the reconciliation report and serialize them into
    status JSON. They also have to be usable as evidence about *which* part of
    the tree went unread, and recovering that by parsing the formatted message
    back apart is how such a guard rots. A str subclass satisfies both: the
    message is the string, `path` is structured data beside it.
    """

    path: Path

    def __new__(cls, message: str, path: Path) -> ScanError:
        error = super().__new__(cls, message)
        error.path = path
        return error


@dataclass
class ReconciliationReport:
    registered: list[str] = field(default_factory=list)
    renamed: list[dict] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    broken: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    duplicates: list[dict] = field(default_factory=list)
    unregistered_discovered: list[str] = field(default_factory=list)
    auto_add: list[str] = field(default_factory=list)
    rejected_traversal: list[str] = field(default_factory=list)
    # True when the filesystem scan that produced `discovered` hit at least
    # one swallowed OS error, so absence of a discovered link proves nothing.
    scan_incomplete: bool = False
    scan_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "registered": self.registered,
            "renamed": self.renamed,
            "missing": self.missing,
            "broken": self.broken,
            "removed": self.removed,
            "duplicates": self.duplicates,
            "unregistered_discovered": self.unregistered_discovered,
            "auto_add": self.auto_add,
            "rejected_traversal": self.rejected_traversal,
            "scan_incomplete": self.scan_incomplete,
            "scan_errors": self.scan_errors,
        }


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def assert_registry_containment(registry: Registry, approved_roots: Sequence[Path]) -> None:
    """C16 companion guard for registry-declared paths.

    Discovered graphify-out symlink targets are already resolved and
    prefix-checked against the approved roots (see `discover_filesystem`), but
    paths coming straight from the registry were used by the pipeline
    unchecked — a registry entry could route the pipeline outside the approved
    trees without ever passing the discovery guard. This applies the exact
    same resolve-then-`_is_under` comparison so the two guards can never
    disagree: for every *enabled* entry, BOTH the resolved `root` and the
    resolved `collection_path` must land under one of approved_roots or one of
    the registry's external_roots, else hard error naming the repo_id.

    `root` is contained as well as `collection_path` because the pipeline
    stat-walks it (`compute_source_manifest`) and hands it to
    `graphify update|extract <root>`: an entry with `root: "/"` would
    otherwise walk and extract the whole filesystem.
    """
    resolved_approved = [root.resolve() for root in approved_roots]
    allowed_roots = resolved_approved + [Path(root).resolve() for root in registry.external_roots]
    for entry in registry.repos:
        if not entry.enabled:
            continue
        if entry.repo_id in registry.disabled:
            continue
        for field_name, declared in (
            ("root", entry.root),
            ("collection_path", entry.collection_path),
        ):
            resolved = Path(declared).resolve()
            if any(_is_under(resolved, root) for root in allowed_roots):
                continue
            raise ValueError(
                f"registry repo {entry.repo_id!r}: {field_name} {str(declared)!r} "
                f"resolves to {str(resolved)!r}, outside approved roots "
                f"[{', '.join(str(root) for root in resolved_approved)}] and registry "
                f"external_roots"
            )


def _iter_candidate_dirs(
    root: Path, depth: int, visited: dict[Path, int], errors: list[str] | None = None
) -> Iterator[Path]:
    """DFS preorder over candidate project dirs below `root`, up to `depth`
    levels of nesting: same ordering as the old fixed depth-2 scan (child,
    then that child's own matching grandchildren, then the next child), but
    membership now differs — hidden dirs and IGNORED_DIR_NAMES are pruned,
    and directory symlinks are no longer followed.

    Never follows directory symlinks during descent: candidacy and the
    recursion decision are both driven by `entry.is_dir(follow_symlinks=False)`
    via `os.scandir`, so a symlinked project dir is not discovered at all
    (this is intended — see Fix 1). Skips dotfiles/dirs and anything in
    IGNORED_DIR_NAMES (this is what keeps the walk from ever descending INTO
    a `graphify-out` dir itself — a dir that itself *contains* a
    graphify-out child is still descended into, since nested-project layouts
    are legitimate).

    `visited` maps each resolved dir already walked to the greatest
    remaining-depth budget it has been explored with, shared across all scan
    roots and recursive calls. A dir is always yielded as a candidate when
    encountered (dedup never suppresses yielding — see Fix 5), but its
    subtree is only walked if the budget available now exceeds whatever
    budget it was already explored with; this covers both symlink cycles
    (bounded by depth, but still deduped) and overlapping/nested scan roots
    (skip a redundant shallower re-walk, but allow a deeper one).

    OSError around scandir or any per-child is_dir/resolve call is caught,
    that entry (or the whole listing) is skipped, and a single aggregated
    warning is logged per directory rather than one line per bad child.
    Every such swallowed error is also appended to `errors` when a sink list
    is passed, so callers can tell "this subtree holds no projects" apart
    from "this subtree could not be read" — see `discover_filesystem`.
    """
    if depth <= 0:
        return
    try:
        entries = sorted(os.scandir(root), key=lambda entry: entry.name)
    except OSError as exc:
        log.warning("cannot list %s (%s); skipping", root, exc)
        if errors is not None:
            errors.append(ScanError(f"cannot list {root}: {exc}", root))
        return

    skipped = 0
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.name in IGNORED_DIR_NAMES:
            continue
        try:
            is_real_dir = entry.is_dir(follow_symlinks=False)
        except OSError:
            skipped += 1
            continue
        if not is_real_dir:
            continue
        child = Path(entry.path)
        try:
            resolved_child = child.resolve()
        except OSError:
            skipped += 1
            continue

        yield child

        remaining = depth - 1
        if remaining <= 0:
            continue
        prior_budget = visited.get(resolved_child)
        if prior_budget is not None and prior_budget >= remaining:
            continue
        visited[resolved_child] = remaining
        yield from _iter_candidate_dirs(child, remaining, visited, errors)

    if skipped:
        log.warning(
            "skipped %d entr%s under %s due to OS errors (is_dir/resolve)",
            skipped,
            "y" if skipped == 1 else "ies",
            root,
        )
        if errors is not None:
            errors.append(
                ScanError(f"skipped {skipped} entries under {root}: is_dir/resolve OS error", root)
            )


def discover_filesystem(
    scan_roots: Sequence[Path],
    approved_roots: Sequence[Path],
    depth: int = SCAN_DEFAULT_DEPTH,
    scan_errors: list[str] | None = None,
) -> list[DiscoveredLink]:
    """Scan for `graphify-out` symlinks/dirs under each of `scan_roots`, up to
    `depth` levels of nesting below the root (depth = max nesting of the
    project dir below a scan root; depth 2 is the original fixed-depth
    behavior). Roots are scanned in the given order, sharing one
    resolved-dir -> remaining-depth-budget `visited` map (see
    `_iter_candidate_dirs`) so overlapping/nested roots and symlink cycles
    are only walked as deep as necessary once; a root that is not a
    directory, or one already covered by a prior root at equal or greater
    depth, is logged at warning level and skipped — other roots still run.
    Results are returned in roots-order, preorder-within-root — the combined
    list is never re-sorted.

    Pass a list as `scan_errors` to collect every OS error this scan swallows
    (unreadable directory, unreadable child, unusable scan root, unreadable
    candidate). Entries are `ScanError` instances: ordinary strings that also
    carry the failing path, which is what lets `reconcile` decide per repo
    whether a missing link proves removal or only proves an unread subtree. A
    non-empty list means the returned results are an incomplete view of the
    filesystem. The return value and its ordering are unchanged; the sink is
    append-only and never read here.
    """
    resolved_approved_roots = [Path(r).resolve() for r in approved_roots]
    results: list[DiscoveredLink] = []
    visited: dict[Path, int] = {}

    # Fix round 4, Finding #1: overlapping scan roots processed outer-first
    # can hand the exact same candidate dir to `_iter_candidate_dirs` twice
    # (once via the outer root's own descent, once again as its own scan
    # root) -> byte-identical duplicate DiscoveredLink entries for the same
    # (source dir, resolved target) pair, which then show up as a spurious
    # `multiple_symlinks_same_target` report row with the same source path
    # repeated, and a doubled `unregistered_discovered` entry.
    #
    # Keyed on the RAW (unresolved) source path plus the resolved target —
    # NOT the resolved source path. An identical-path repeat from overlapping
    # roots yields the exact same raw path both times, so it still collides
    # and gets dropped. Two DISTINCT alias dirs (different raw paths that
    # happen to resolve to the same real dir and the same target) differ on
    # raw path and both survive — that distinction is load-bearing for
    # `multiple_symlinks_same_target` detection in `reconcile`, so it must
    # not be collapsed by resolving the source before keying.
    seen_link_keys: set[tuple[str, str | None]] = set()

    def _record(link: DiscoveredLink, raw_source: Path) -> bool:
        """Append `link` unless an identical (raw source path, resolved
        target) pair was already recorded. Returns True if newly appended,
        False if suppressed as a duplicate (caller uses this to keep
        per-root link counters honest — see the zero-links warning below)."""
        key = (str(raw_source), str(link.target) if link.target is not None else None)
        if key in seen_link_keys:
            return False
        seen_link_keys.add(key)
        results.append(link)
        return True

    for raw_root in scan_roots:
        try:
            root = Path(raw_root).resolve()
            root_is_dir = root.is_dir()
        except OSError as exc:
            log.warning("cannot stat scan root %s (%s); skipping", raw_root, exc)
            if scan_errors is not None:
                scan_errors.append(
                    ScanError(f"cannot stat scan root {raw_root}: {exc}", Path(raw_root))
                )
            continue
        if not root_is_dir:
            log.warning("scan root %s is not a directory; skipping", root)
            if scan_errors is not None:
                scan_errors.append(ScanError(f"scan root {root} is not a directory", root))
            continue
        prior_budget = visited.get(root)
        if prior_budget is not None and prior_budget >= depth:
            log.warning(
                "scan root %s already covered by a prior scan root at equal or "
                "greater depth; skipping",
                root,
            )
            continue
        visited[root] = depth

        results_before = len(results)
        candidate_count = 0
        # Fix round 4, Finding #1 side effect: a dedup-suppressed repeat is a
        # real link this root DID find, just one already recorded via an
        # earlier overlapping root — it must still count toward "did this
        # root find anything" for the zero-links warning below, or a root
        # whose links are all identical-path repeats of a prior root would
        # misreport link_count == 0 and fire a spurious warning.
        dedup_suppressed = 0
        for project_dir in _iter_candidate_dirs(root, depth, visited, scan_errors):
            candidate_count += 1
            link_path = project_dir / "graphify-out"

            # Fix round 2, Finding #4: every per-candidate OS call below can
            # race a live tree (deleted mid-walk, ESTALE on a network mount,
            # ELOOP on a pathological symlink, etc.) — none of that may abort
            # the whole run. Each op is individually guarded; a failure here
            # skips only this one candidate, with a warning naming it.
            try:
                resolved_project_dir = project_dir.resolve()
            except OSError as exc:
                log.warning("cannot resolve candidate dir %s (%s); skipping", project_dir, exc)
                if scan_errors is not None:
                    scan_errors.append(
                        ScanError(f"cannot resolve candidate dir {project_dir}: {exc}", project_dir)
                    )
                continue

            if not any(_is_under(resolved_project_dir, r) for r in resolved_approved_roots):
                # Fix 1's symmetric guard: the candidate dir itself (not just
                # a graphify-out target inside it) escapes every approved
                # root. Same reporting path as the target guard below.
                if not _record(
                    DiscoveredLink(
                        source_root=project_dir,
                        link_path=link_path,
                        target=None,
                        rejected_traversal=True,
                    ),
                    project_dir,
                ):
                    dedup_suppressed += 1
                continue

            # Fix round 3, Finding #4 (narrow follow-up): pathlib's
            # is_symlink()/is_dir()/exists() booleans internally swallow
            # ENOENT/ENOTDIR/EBADF/ELOOP and just return False — so a
            # genuine OS-level failure (as opposed to "graphify-out simply
            # doesn't exist here") would silently misclassify this candidate
            # as "no link" instead of warn-and-skip. Use os.lstat/os.stat
            # directly so a real OSError can be told apart from the
            # ordinary "nothing here" case.
            try:
                link_lstat = os.lstat(link_path)
            except FileNotFoundError:
                # The common case for every non-project dir: no graphify-out
                # entry at all. Must stay silent — this is not an error.
                continue
            except OSError as exc:
                log.warning(
                    "cannot lstat graphify-out under %s (errno %s: %s); skipping candidate",
                    project_dir,
                    exc.errno,
                    exc,
                )
                if scan_errors is not None:
                    scan_errors.append(
                        ScanError(
                            f"cannot lstat graphify-out under {project_dir}: {exc}", project_dir
                        )
                    )
                continue

            is_symlink_entry = stat.S_ISLNK(link_lstat.st_mode)
            is_dir_entry = stat.S_ISDIR(link_lstat.st_mode)
            if not is_symlink_entry and not is_dir_entry:
                # graphify-out exists but is neither a symlink nor a
                # directory (e.g. a stray regular file) — not a candidate,
                # same treatment as "doesn't exist".
                continue

            if is_symlink_entry:
                try:
                    os.stat(link_path)  # follow the symlink to detect dangling targets
                except FileNotFoundError:
                    # Dangling symlink (target missing) — broken, do not crash.
                    if not _record(
                        DiscoveredLink(
                            source_root=project_dir, link_path=link_path, target=None, broken=True
                        ),
                        project_dir,
                    ):
                        dedup_suppressed += 1
                    continue
                except OSError as exc:
                    log.warning(
                        "cannot stat graphify-out target under %s (errno %s: %s); "
                        "skipping candidate",
                        project_dir,
                        exc.errno,
                        exc,
                    )
                    if scan_errors is not None:
                        scan_errors.append(
                            ScanError(
                                f"cannot stat graphify-out target under {project_dir}: {exc}",
                                project_dir,
                            )
                        )
                    continue

            try:
                target = link_path.resolve()
            except OSError as exc:
                log.warning(
                    "cannot resolve graphify-out target under %s (errno %s: %s); "
                    "skipping candidate",
                    project_dir,
                    exc.errno,
                    exc,
                )
                if scan_errors is not None:
                    scan_errors.append(
                        ScanError(
                            f"cannot resolve graphify-out target under {project_dir}: {exc}",
                            project_dir,
                        )
                    )
                continue

            if not any(_is_under(target, r) for r in resolved_approved_roots):
                # Path traversal guard: resolved symlink target escapes every
                # approved root (the configured scan roots). A legitimate
                # graphify-out target always resolves inside one of
                # approved_roots; anything else is rejected.
                if not _record(
                    DiscoveredLink(
                        source_root=project_dir,
                        link_path=link_path,
                        target=target,
                        rejected_traversal=True,
                    ),
                    project_dir,
                ):
                    dedup_suppressed += 1
                continue
            if not _record(
                DiscoveredLink(source_root=project_dir, link_path=link_path, target=target),
                project_dir,
            ):
                dedup_suppressed += 1

        link_count = len(results) - results_before
        effective_link_count = link_count + dedup_suppressed
        log.info(
            "discovery: root %s -> %d candidate dir(s), %d link(s)",
            root,
            candidate_count,
            link_count,
        )
        if candidate_count == 0:
            log.warning("discovery: root %s yielded zero candidate dirs", root)
        if candidate_count > 0 and effective_link_count == 0:
            # Fix round 2, Finding #10: candidates existed but none produced
            # a discovered link — the real "misconfigured/empty root" signal
            # (candidate_count == 0 alone only catches an empty/pruned tree).
            # Uses effective_link_count (link_count + dedup_suppressed), not
            # link_count alone, so a root whose links are all identical-path
            # repeats of an earlier overlapping root (Fix round 4, Finding
            # #1) isn't misreported as having found zero links.
            log.warning(
                "discovery: root %s had %d candidate dir(s) but discovered zero links",
                root,
                candidate_count,
            )
    return results


def _failed_scan_paths(scan_errors: Sequence[str]) -> list[Path] | None:
    """Resolved paths of the subtrees this scan could not read.

    Returns None when the damage cannot be localized: an error entry that is a
    plain string carries no path, and a path that will not resolve tells us
    nothing either. In both cases the caller must fall back to suppressing
    `removed` for every repo, since it has no way to know which ones the
    failure could have hidden.
    """
    paths: list[Path] = []
    for error in scan_errors:
        raw = getattr(error, "path", None)
        if raw is None:
            return None
        try:
            paths.append(Path(raw).resolve())
        except OSError:
            return None
    return paths


def _scan_could_hide(root: Path, failed_paths: list[Path] | None) -> bool:
    """True when a failed part of the scan could have hidden `root`'s link.

    `failed_paths` of None means the failures could not be localized, so every
    root is treated as possibly hidden. Otherwise the link at
    `<root>/graphify-out` is only invisible if the failure was at `root`
    itself or at one of its ancestors: an error deeper in the tree cannot hide
    an entry that sits directly in `root`. Resolve-then-`_is_under`, the same
    comparison every other containment check in this module uses.
    """
    if failed_paths is None:
        return True
    try:
        resolved_root = root.resolve()
    except OSError:
        return True
    return any(_is_under(resolved_root, failed) for failed in failed_paths)


def reconcile(
    discovered: list[DiscoveredLink],
    registry: Registry,
    mesh_root: Path,
    scan_errors: Sequence[str] | None = None,
) -> ReconciliationReport:
    """Reconcile the discovered links against the registry.

    Pass the error sink `discover_filesystem` filled as `scan_errors`. A
    non-empty sink sets `report.scan_incomplete` and copies the reasons into
    `report.scan_errors`, which is what operators and the pipeline read.

    Removal must be a confirmed fact: either the collection_path is gone, or a
    part of the filesystem that would have shown the link was read successfully
    and showed nothing. It is never an inference from a scan that failed to
    read the relevant subtree. Downstream, `removed` auto-authorizes shrinking
    the published graph.

    The suppression is therefore per repo, not global. A repo keeps its normal
    `removed` classification unless one of the recorded failing paths (see
    `ScanError.path`) is the repo's own `root` or an ancestor of it — only such
    a failure can have hidden `<root>/graphify-out`. An unreadable directory in
    an unrelated part of the tree must not protect a repo from being
    classified, or the guard turns into a permanent veto that every shrink has
    to be forced past. A repo the scan could have hidden stays in
    `report.registered` when its collection_path still exists; a genuinely
    absent collection_path is still reported as `missing`.

    Errors that carry no path at all cannot be localized, so they fall back to
    suppressing `removed` for every repo.
    """
    report = ReconciliationReport()
    failed_scan_paths: list[Path] | None = None
    if scan_errors:
        report.scan_incomplete = True
        report.scan_errors = list(scan_errors)
        failed_scan_paths = _failed_scan_paths(scan_errors)
    mesh_root = mesh_root.resolve()

    # Guard: a discovered target must resolve under the mesh tree (that's the
    # only legitimate destination for a graphify-out symlink in this design).
    valid_discovered = []
    for d in discovered:
        if d.rejected_traversal:
            report.rejected_traversal.append(str(d.link_path))
            continue
        if d.target is not None and not _is_under(d.target, mesh_root):
            report.rejected_traversal.append(str(d.link_path))
            continue
        valid_discovered.append(d)

    # Registry-internal duplicate collection_path detection.
    seen_collection_paths: dict[str, list[str]] = {}
    for entry in registry.repos:
        seen_collection_paths.setdefault(str(entry.collection_path), []).append(entry.repo_id)
    for collection_path, repo_ids in seen_collection_paths.items():
        if len(repo_ids) > 1:
            report.duplicates.append(
                {
                    "reason": "registry_duplicate_collection_path",
                    "collection_path": collection_path,
                    "repo_ids": repo_ids,
                }
            )

    # Map resolved target -> list of discovered links pointing at it (dedup
    # detection for two discovered symlinks resolving to the same real path).
    by_target: dict[str, list[DiscoveredLink]] = {}
    for d in valid_discovered:
        if d.target is None:
            continue
        by_target.setdefault(str(d.target), []).append(d)
    for target, links in by_target.items():
        if len(links) > 1:
            report.duplicates.append(
                {
                    "reason": "multiple_symlinks_same_target",
                    "collection_path": target,
                    "source_roots": sorted(str(link.source_root) for link in links),
                }
            )

    broken_by_root = {str(d.source_root): d for d in valid_discovered if d.broken}

    registered_collection_paths = set(seen_collection_paths.keys())

    for entry in registry.repos:
        if entry.repo_id in registry.disabled or not entry.enabled:
            continue
        cp = str(entry.collection_path)
        matches = by_target.get(cp, [])

        if not matches:
            broken_link = broken_by_root.get(str(entry.root))
            if broken_link is not None:
                report.broken.append(entry.repo_id)
            elif not entry.collection_path.exists():
                report.missing.append(entry.repo_id)
            elif report.scan_incomplete and _scan_could_hide(entry.root, failed_scan_paths):
                # The scan could not read the part of the filesystem this
                # repo's link lives in, so "no link discovered" is not
                # evidence the project is gone. Keep the repo registered;
                # report.scan_errors carries the reason.
                report.registered.append(entry.repo_id)
            else:
                report.removed.append(entry.repo_id)
        else:
            # Prefer the match that is actually the registry-known root, if
            # discovered, over an arbitrary sorted tie-break — otherwise two
            # discovered symlinks resolving to the same target could report
            # a false "renamed" even though the registered root was among
            # the matches all along (Fix 3).
            match_for_known_root = next((d for d in matches if d.source_root == entry.root), None)
            canonical = match_for_known_root or sorted(matches, key=lambda d: str(d.source_root))[0]
            if canonical.source_root == entry.root:
                report.registered.append(entry.repo_id)
            else:
                report.renamed.append(
                    {
                        "repo_id": entry.repo_id,
                        "old_root": str(entry.root),
                        "new_root": str(canonical.source_root),
                    }
                )

        graph_file = entry.collection_path / "graph.json"
        if entry.collection_path.exists() and not graph_file.exists():
            report.auto_add.append(entry.repo_id)

    for d in valid_discovered:
        if d.target is None:
            continue
        if str(d.target) not in registered_collection_paths:
            report.unregistered_discovered.append(str(d.source_root))

    return report
