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

from pathlib import Path


def compute_tag_to_repo_id(
    sorted_graph_paths: list[Path], sorted_repo_ids: list[str]
) -> dict[str, str]:
    """`sorted_graph_paths`/`sorted_repo_ids` must be the SAME order-aligned
    lists passed to `graphify_cli.run_merge_graphs` (pipeline.py already
    builds them this way: `sorted_repo_ids = sorted(graph_paths_by_repo)`,
    `sorted_graph_paths = [graph_paths_by_repo[rid] for rid in
    sorted_repo_ids]`).

    The `graphify` import is deferred to call time (not module top) so that
    importing this package — and running `--help` on the console scripts —
    never requires the upstream `graphify` package to be installed; it is only
    needed when a merge actually runs."""
    from graphify.build import distinct_repo_tags

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
    every node's `repo` attribute becomes the true repo_id. A node/edge
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

    return graph_data
