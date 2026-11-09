"""Dotted key paths for structural YAML/JSON config-key nodes.

graphify's YAML and JSON extractors emit one node per key, labelled with the
bare key (`enabled`, `tls`), linked to its parent by a `contains` edge, with
the file node at the root. Bare labels collide inside one file, so the mesh's
durable key `(repo, source_file, label)` cannot tell `image.registry` from
`global.registry`. This module derives the dotted path from the structure so
every consumer of that key (embedding shards, lexical index, overlay refs,
the server's key maps) can address each key node on its own.

Derived only: the merged graph is never rewritten. Sync (per-repo graphs) and
the server (merged graph, ids prefixed `repo::`) both call `config_key_paths`
and must agree, so nothing here depends on node ids — only on labels, lines,
`source_file`, `repo` and the `contains` structure, which both graphs share.

Extractor facts this relies on (graphify 0.9.103, `extractors/yaml_config.py`
and `json_config.py`):

* A key node's id is `stem + parent key + key`, so the same key under two
  same-named parents is ONE node with two `contains` parents. Its
  `source_location` is its first occurrence. The parent of that occurrence
  is the parent with the greatest line not after the node's own line.
* Sequence items are walked without index nodes: keys of list items hang off
  the list's key. Paths therefore carry no array index, matching the
  extractor's own convention (`containers.name`, not `containers[0].name`).
* JSON arrays are not walked at all (only `extends` emits edges).
* The graph is undirected (`directed: false`) but `contains` links keep
  parent -> child orientation; checked on the scratch generation, 0 of 1741
  links reversed.
"""

from __future__ import annotations

import re

CONFIG_SUFFIXES = (".yaml", ".yml", ".json")

# Bound on the parent chain. The extractors stop at depth 6 (+1 for a
# multi-document node); anything deeper comes from a malformed graph and is
# left unqualified rather than recursed into.
MAX_PATH_DEPTH = 32

PATH_SEPARATOR = "."

_LINE = re.compile(r"^L(\d+)$")


def is_config_file(source_file: object) -> bool:
    return isinstance(source_file, str) and source_file.lower().endswith(CONFIG_SUFFIXES)


def is_file_node_label(label: object, source_file: object) -> bool:
    """True when `label` names the file itself. graphify labels a file node
    with its basename, or with enough parent directories to tell same-named
    files apart (`styleguide/values.yaml`), so a path-suffix test is needed."""
    if not isinstance(label, str) or not label or not isinstance(source_file, str):
        return False
    return source_file == label or source_file.endswith("/" + label)


def _line(node: dict) -> int | None:
    match = _LINE.match(str(node.get("source_location") or "").strip())
    return int(match.group(1)) if match else None


def config_key_paths(graph_data: dict) -> dict:
    """Map graph node id -> dotted key path for every structural config-key
    node reachable from its file node through `contains` links inside the same
    file (and repo). The file node itself and unreachable nodes are absent."""
    config_nodes: dict = {}
    for node in graph_data.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        label = node.get("label")
        if node_id is None or not isinstance(label, str) or not label:
            continue
        if not is_config_file(node.get("source_file")) or node.get("_origin") == "semantic":
            continue
        config_nodes[node_id] = node

    parents: dict = {}
    for link in graph_data.get("links", graph_data.get("edges", [])):
        if not isinstance(link, dict) or link.get("relation") != "contains":
            continue
        if link.get("_origin") == "semantic":
            continue
        src, dst = link.get("source"), link.get("target")
        if src == dst:
            continue
        parent, child = config_nodes.get(src), config_nodes.get(dst)
        if parent is None or child is None:
            continue
        if parent["source_file"] != child["source_file"] or parent.get("repo") != child.get("repo"):
            continue
        parents.setdefault(dst, []).append(src)

    # One parent per node, chosen from line rank alone. Choosing by resolved
    # path instead made the result depend on node order: extractor ids are
    # `stem + parent key + key`, so `y: {x: {y: {x: 1}}}` yields a real
    # contains-cycle, and a path memoised mid-cycle leaked into its neighbours.
    chosen_parent: dict = {}
    for node_id, candidates in parents.items():
        own_line = _line(config_nodes[node_id])
        chosen_parent[node_id] = min(
            candidates, key=lambda pid: _parent_rank(config_nodes[pid], own_line)
        )

    def resolve(node_id) -> str | None:
        labels: list[str] = []
        seen: set = set()
        current = node_id
        while current in chosen_parent:
            if current in seen or len(labels) > MAX_PATH_DEPTH:
                return None
            seen.add(current)
            labels.append(config_nodes[current]["label"])
            current = chosen_parent[current]
        root = config_nodes[current]
        if not is_file_node_label(root["label"], root["source_file"]):
            return None
        return PATH_SEPARATOR.join(reversed(labels))

    paths: dict = {}
    for node_id in config_nodes:
        path = resolve(node_id)
        if path:
            paths[node_id] = path
    return paths


def _parent_rank(parent: dict, own_line: int | None) -> tuple:
    """Sort key for a node's candidate parents: the greatest line not after the
    node's own line first, then any other parent by line. Label and the id
    without its merged-graph `repo::` prefix break ties, so sync and server pick
    the same parent."""
    parent_line = _line(parent)
    if own_line is not None and parent_line is not None and parent_line <= own_line:
        line_rank: tuple = (0, -parent_line)
    else:
        line_rank = (1, parent_line if parent_line is not None else float("inf"))
    parent_id = str(parent.get("id"))
    prefix = f"{parent.get('repo')}::"
    if parent.get("repo") and parent_id.startswith(prefix):
        parent_id = parent_id[len(prefix) :]
    return (*line_rank, parent["label"], parent_id)
