"""The `[repo:source_file:line]` citation every node-returning tool emits.

Kept in its own module so `server.py` (search/cross_project/find_similar
hits) and `traverse.py` (the `neighbors` tool) render citations through one
function and can never drift apart. The line comes from
`graphify_mesh.sync.embedding.node_line`, which also reads graphify's
`source_location: "L34"` form; an unknown line renders as `?`.
"""

from __future__ import annotations

from graphify_mesh.sync.embedding import node_line


def citation(repo: str, source_file: str, node: dict) -> str:
    line = node_line(node)
    return f"[{repo}:{source_file}:{line if line is not None else '?'}]"
