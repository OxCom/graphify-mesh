"""Repo-tag normalization after `graphify merge-graphs` (WS5 prerequisite).

Verified against the installed graphify 0.9.56 package
(`graphify/cli.py:1902`, `graphify/build.py:distinct_repo_tags`): the real
`merge-graphs` CLI has no flag to pin an explicit per-input repo tag. It
derives each merged node's id-prefix/`repo` attribute purely from
`graph_paths[i].parent.parent.name` (widening on collision by walking up
another directory level and adding an index suffix — see
`distinct_repo_tags`'s own docstring).

For the collection-path layout this engine uses
(`graphify/<product>/<sub>/graph.json`, e.g.
`graphify/example-org/backend-a/graph.json` and
`graphify/example-org/frontend-b/graph.json`), `parent.parent` is
`graphify/<product>` for every repo under the same product — verified
directly against `graphify.build.distinct_repo_tags` with paths shaped
exactly like the collection paths:

    >>> distinct_repo_tags([
    ...     Path("/path/to/graph-mesh/graphify/example-org/backend-a/graph.json"),
    ...     Path("/path/to/graph-mesh/graphify/example-org/frontend-b/graph.json"),
    ... ])
    ['graphify_example-org', 'graphify_example-org-2']

That is NOT the registry's stable `repo_id` (`example-org.backend-a`,
`example-org.frontend-b`) — it is exactly baseline systemic failure #1, "no
repo attribution", reproducible for this directory layout. WS5's scope contract
and every tool's repo-attribution requirement depend on merged-graph nodes
carrying the true registry `repo_id`, not graphify's auto-derived tag.

Fix: `distinct_repo_tags` is a pure function of the input path list (no
randomness, no filesystem reads beyond `.name`/`.parent` string ops) — the
exact same call graphify's own merge-graphs CLI made internally can be
replicated here, in the same order, to recover the auto-tag -> repo_id
mapping, then every node id / edge endpoint / `repo` attribute is rewritten
from `<auto_tag>::...` to `<true_repo_id>::...` in place. Runs once,
immediately after merge, before naming/embedding/overlay/lexical-index, so
every downstream stage and the published artifact carry the real repo_id.
"""

from __future__ import annotations

import logging
import re
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

from packaging.version import InvalidVersion, Version

from graphify_mesh.sync.graphify_cli import _run, resolve_bin_argv

log = logging.getLogger("graphify_mesh.sync")

# A `--version` probe starts an interpreter and prints one line. Anything
# slower than this is a broken binary, and the merge-time default of 900 s
# would stall the whole pipeline waiting for it.
VERSION_PROBE_TIMEOUT_SECONDS = 30

# Matches the numeric part of a version banner such as "graphify 0.9.56".
_VERSION_RE = re.compile(r"\d+(?:\.\d+)+")

# A banner may print an interpreter or framework version before the package
# version ("Python 3.11.2, graphify 0.9.56"), and the first numeric match would
# then be the wrong one. Prefer a number that follows the word "graphify".
_NAMED_VERSION_RE = re.compile(r"graphify[^0-9\n]{0,24}(\d+(?:\.\d+)+)", re.IGNORECASE)

# The PyPI project name; the import name is `graphify`, and the package
# exposes no `__version__` attribute, so installed metadata is the only way
# to learn the in-process version.
GRAPHIFY_DISTRIBUTION = "graphifyy"


def _in_process_graphify_version() -> str | None:
    """Version of the `graphify` package this interpreter imports, or None
    when it carries no installed metadata (a source checkout on PYTHONPATH)."""
    try:
        return package_version(GRAPHIFY_DISTRIBUTION)
    except PackageNotFoundError:
        return None


def _binary_graphify_version(graphify_bin: str) -> str | None:
    """Version reported by `<graphify_bin> --version`, or None when the
    binary rejects the flag, fails, or prints nothing version-shaped."""
    argv = resolve_bin_argv(graphify_bin)
    if not argv:
        return None
    result = _run(argv + ["--version"], cwd=None, env=None, timeout=VERSION_PROBE_TIMEOUT_SECONDS)
    if not result.ok:
        return None
    banner = f"{result.stdout}\n{result.stderr}"
    named = _NAMED_VERSION_RE.search(banner)
    if named:
        return named.group(1)
    match = _VERSION_RE.search(banner)
    return match.group(0) if match else None


def _release(raw: str) -> str | None:
    """Release part of a version string ("0.9.64.dev0" -> "0.9.64"), or None
    when it is not a version this parser understands."""
    try:
        return Version(raw).base_version
    except InvalidVersion:
        return None


# Binaries already compared in this process. The check is meant to run before
# `graphify merge-graphs` is paid for, while the tag map is built after it, so
# without this memo the same binary would be probed twice per sync run.
_parity_checked: set[str] = set()


def reset_version_parity_cache() -> None:
    """Forget which binaries were already probed. Test hook — a long-lived
    process never changes which binary a given path points at mid-run."""
    _parity_checked.clear()


def check_graphify_version_parity(graphify_bin: str | None) -> None:
    """Raises ValueError when the in-process `graphify` and the configured
    binary report different release versions. Every other outcome logs and
    returns.

    Call this BEFORE `run_merge_graphs`: a mismatch invalidates the merge that
    the check guards, so paying the merge timeout first wastes up to 900 s.
    `compute_tag_to_repo_id` calls it again, and the second call is a no-op for
    a binary already compared in this process.

    Only the release parts are compared (`packaging.version.Version.base_version`),
    so a dev install, a local build tag or a post-release of the same release
    ("0.9.64.dev0", "0.9.64+g1a2b3c", "0.9.64.post1") is not a mismatch. Those
    suffixes never change how `distinct_repo_tags` derives auto tags; different
    releases can.
    """
    if graphify_bin is not None and graphify_bin in _parity_checked:
        return
    if graphify_bin is None:
        log.warning(
            "graphify version parity check skipped: no graphify binary path was passed, so "
            "the repo-tag algorithm used here cannot be compared against the one the merge ran"
        )
        return
    in_process = _in_process_graphify_version()
    if in_process is None:
        log.warning(
            "graphify version parity check skipped: the imported graphify package carries no "
            "installed metadata for distribution %r",
            GRAPHIFY_DISTRIBUTION,
        )
        return
    from_binary = _binary_graphify_version(graphify_bin)
    if from_binary is None:
        log.warning(
            "graphify version parity check skipped: %r did not report a usable version "
            "(no --version support, or unparseable output); in-process graphify is %s",
            graphify_bin,
            in_process,
        )
        _parity_checked.add(graphify_bin)
        return
    in_process_release = _release(in_process)
    binary_release = _release(from_binary)
    if in_process_release is None or binary_release is None:
        log.warning(
            "graphify version parity check skipped: version strings are not comparable "
            "(in-process %r, binary %r reported by %r)",
            in_process,
            from_binary,
            graphify_bin,
        )
        _parity_checked.add(graphify_bin)
        return
    if binary_release != in_process_release:
        raise ValueError(
            f"graphify version mismatch: this interpreter imports graphify {in_process}, "
            f"but the merge binary {graphify_bin!r} reports graphify {from_binary}. "
            "The repo-tag map is computed from the imported package while the merge is run "
            "by the binary, so two versions can derive auto tags differently and the map "
            "would silently fail to match the merged graph. Point GRAPHIFY_BIN at the same "
            "environment this process imports graphify from, or align the two versions."
        )
    _parity_checked.add(graphify_bin)


def compute_tag_to_repo_id(
    sorted_graph_paths: list[Path],
    sorted_repo_ids: list[str],
    *,
    graphify_bin: str | None = None,
) -> dict[str, str]:
    """`sorted_graph_paths`/`sorted_repo_ids` must be the SAME order-aligned
    lists passed to `graphify_cli.run_merge_graphs` (pipeline.py already
    builds them this way: `sorted_repo_ids = sorted(graph_paths_by_repo)`,
    `sorted_graph_paths = [graph_paths_by_repo[rid] for rid in
    sorted_repo_ids]`).

    The `graphify` import is deferred to call time (not module top) so that
    importing this package — and running `--help` on the console scripts —
    never requires the upstream `graphify` package to be installed; it is only
    needed when a merge actually runs.

    `graphify_bin` is the same GRAPHIFY_BIN value the merge ran with. The tag
    map is derived from the graphify this interpreter imports, while the merge
    itself runs that binary, which may live in a different environment. Two
    versions can derive auto tags differently, and the count check below does
    not catch a same-count divergence, so the versions are compared first and a
    mismatch raises ValueError naming both. `graphify_bin=None` skips the
    check and logs that it was skipped, and a binary with no usable `--version`
    output degrades to a logged warning rather than blocking the pipeline.

    The check belongs before the merge, not here: call
    `check_graphify_version_parity` ahead of `run_merge_graphs` so a mismatch
    is not discovered after the merge timeout has been spent. Calling it here
    as well costs nothing, because the same binary is probed once per
    process."""
    from graphify.build import distinct_repo_tags

    check_graphify_version_parity(graphify_bin)

    tags = distinct_repo_tags(sorted_graph_paths)
    if len(tags) != len(sorted_repo_ids):
        raise ValueError(
            f"repo-tag count mismatch: distinct_repo_tags returned {len(tags)} tags "
            f"for {len(sorted_repo_ids)} repo ids — merge input paths and repo id list "
            "must be the same, order-aligned list"
        )
    return dict(zip(tags, sorted_repo_ids, strict=True))


def rewrite_repo_tags(graph_data: dict, tag_to_repo_id: dict[str, str]) -> dict:
    """Rewrites `graph_data` in place (and returns it) so every node id /
    edge endpoint prefixed `<auto_tag>::` becomes `<true_repo_id>::`, and
    every node's `repo` attribute becomes the true repo_id, and every
    hyperedge id and member id is remapped the same way. A node/edge
    whose prefix does not match any known auto tag is left untouched
    (defensive: never crash on an id shape this module doesn't recognize —
    e.g. a bare external node with no `::` at all)."""

    def _remap_id(node_id):
        if not isinstance(node_id, str) or "::" not in node_id:
            return node_id
        tag, sep, rest = node_id.partition("::")
        repo_id = tag_to_repo_id.get(tag)
        if repo_id is None:
            return node_id
        return f"{repo_id}{sep}{rest}"

    id_remap: dict = {}
    for node in graph_data.get("nodes", []):
        if not isinstance(node, dict):
            continue
        old_id = node.get("id")
        new_id = _remap_id(old_id)
        if new_id != old_id:
            id_remap[old_id] = new_id
            node["id"] = new_id
        repo_tag = node.get("repo")
        if repo_tag in tag_to_repo_id:
            node["repo"] = tag_to_repo_id[repo_tag]

    for link in graph_data.get("links", graph_data.get("edges", [])):
        if not isinstance(link, dict):
            continue
        if link.get("source") in id_remap:
            link["source"] = id_remap[link["source"]]
        if link.get("target") in id_remap:
            link["target"] = id_remap[link["target"]]

    # Upstream writes hyperedges into BOTH the top-level key and the nested
    # graph.hyperedges slot (graphify/build.py:885-894 folds the nested one onto
    # the top-level key when the latter is missing). Remapping only the
    # top-level slot leaves the nested copy dangling: measured on the live
    # merged graph, 39/39 members resolved in the top-level slot and 0/39 in the
    # nested one.
    _MEMBER_KEYS = ("nodes", "members", "node_ids")
    hyperedge_slots = [graph_data.get("hyperedges")]
    nested = graph_data.get("graph")
    if isinstance(nested, dict):
        hyperedge_slots.append(nested.get("hyperedges"))
    seen_hyperedges: set[int] = set()
    for hyperedges in hyperedge_slots:
        if not isinstance(hyperedges, list):
            continue
        for hyperedge in hyperedges:
            # The two slots can hold the same dict object; remapping it twice
            # would re-prefix an id that is already correct.
            if not isinstance(hyperedge, dict) or id(hyperedge) in seen_hyperedges:
                continue
            seen_hyperedges.add(id(hyperedge))
            if "id" in hyperedge:
                hyperedge["id"] = _remap_id(hyperedge["id"])
            for key in _MEMBER_KEYS:
                members = hyperedge.get(key)
                if isinstance(members, list):
                    hyperedge[key] = [_remap_id(m) for m in members]

    return graph_data
