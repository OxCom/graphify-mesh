"""Every call into graphify's Python API lives here.

The naming stage used to shell out to `graphify cluster-only` / `graphify label`,
which rebuild the graph through `graphify.build.build_from_json`. That builder
merges a non-AST node into an AST node sharing `(source_file, label)` and, in a
second pass, by label alone — neither considers a node's `repo`, so in the merged
multi-repo graph it collapses nodes across repositories and rewrites labels into
full paths. Loading the graph with `node_link_graph` is what graphify's own read
side does (`graphify/serve.py`), and it applies none of those heuristics.
"""

from __future__ import annotations

import networkx as nx
from networkx.readwrite import json_graph


def graph_from_node_link(graph_data: dict) -> nx.Graph:
    """The merged graph as an `nx.Graph`, with no merge heuristics applied."""
    # `graph={}` on purpose: networkx does NOT copy that mapping —
    # `G.graph is data["graph"]` is True — so anything upstream writes into
    # `G.graph` (hyperedges, for instance) would land in the mesh's own merged
    # dict, past the naming stage's integrity guard, which covers nodes and
    # links. Clustering and labeling never read graph-level metadata.
    data = dict(graph_data, graph={})
    if "links" not in data and "edges" in data:
        data["links"] = data["edges"]
    try:
        return json_graph.node_link_graph(data, edges="links")
    except TypeError:  # networkx < 3.4 has no `edges` keyword
        return json_graph.node_link_graph(data)


def cluster_graph(
    G: nx.Graph,
    resolution: float = 1.0,
    previous_assignments: dict[str, int] | None = None,
) -> dict[int, list[str]]:
    """`{community_id: [node_id]}` for every node in `G`.

    `previous_assignments` is last run's `{node_id: community_id}`. graphify's
    own `cluster-only` remaps ids onto the previous assignment
    (`graphify/cli.py:2117`); without it community ids renumber on every run and
    anything that recorded an id points at a different community.
    """
    from graphify.cluster import cluster, remap_communities_to_previous

    communities = cluster(G, resolution=resolution)
    if previous_assignments:
        communities = remap_communities_to_previous(communities, previous_assignments)
    return communities


def membership_sigs(communities: dict[int, list[str]]) -> dict[int, str]:
    """`{community_id: sha256-of-sorted-member-ids}` — the reuse key for names."""
    from graphify.cluster import community_member_sigs

    return community_member_sigs(communities)


def hub_names(G: nx.Graph, communities: dict[int, list[str]]) -> dict[int, str]:
    """LLM-free names: each community takes the label of its highest-degree member."""
    from graphify.cluster import label_communities_by_hub

    return label_communities_by_hub(G, communities)


def llm_names(
    G: nx.Graph,
    communities: dict[int, list[str]],
    *,
    backend: str,
    model: str,
) -> tuple[dict[int, str], str]:
    """`({community_id: name}, source)` where source is "llm" or "placeholder".

    Never raises: upstream catches its own backend failures and degrades to
    `Community N` placeholders, which the caller treats as "no name yet".
    """
    from graphify.llm import generate_community_labels

    # max_concurrency=1, against upstream's default of 4 (graphify/llm.py:3500).
    # Communities are labeled in batches of 100 (_LABEL_BATCH_SIZE, llm.py:3233),
    # so the batch count grows with the graph but a single call does not. What
    # upstream's 4-way fan-out does instead is inflate the latency of each call,
    # and the backend behind this endpoint serves one request at a time.
    # Measured 2026-09-21, one 100-community batch: 12.2 s alone; four in
    # parallel took 41.5 s wall with the slowest single call also 41.5 s, against
    # 48.7 s to run the same four serially. That is 1.17x throughput for 3.4x
    # per-call latency — and per-call latency is exactly what
    # `Settings.ollama_api_timeout` bounds, so the fan-out was spending the
    # timeout budget it had no way to earn back. The 2026-09-21 production run
    # degraded 14 communities to provisional names this way.
    return generate_community_labels(
        G, communities, backend=backend, model=model, quiet=True, max_concurrency=1
    )
