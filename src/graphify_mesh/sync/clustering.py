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

import logging
import os
import time

import networkx as nx
from networkx.readwrite import json_graph

from graphify_mesh.sync.config import LABEL_MAX_CONCURRENCY, LLM_DEFAULT_CONCURRENCY
from graphify_mesh.sync.validate import PLACEHOLDER_RE

log = logging.getLogger("graphify_mesh.sync.clustering")


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
    (`graphify.cluster.remap_communities_to_previous`, called from
    `graphify/cli.py`); without it community ids renumber on every run and
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


UNATTRIBUTED_GROUP = ""


def repo_groups(G: nx.Graph) -> dict[str, list[str]]:
    """`{repo_id: sorted node ids}` covering every node of `G` exactly once.

    A node without a `repo` (a stub or sourceless reference) joins the smallest
    repo_id among its attributed neighbours. The pass repeats so a stub reached
    only through other stubs is placed too; each pass reads only the previous
    pass's assignments, so the result does not depend on node order. A node no
    attributed node reaches lands in `UNATTRIBUTED_GROUP`.
    """
    owner: dict[str, str] = {}
    pending: list[str] = []
    for node, attrs in G.nodes(data=True):
        repo = attrs.get("repo")
        if isinstance(repo, str) and repo:
            owner[node] = repo
        else:
            pending.append(node)
    while pending:
        placed: dict[str, str] = {}
        for node in pending:
            candidates = [owner[nb] for nb in G.neighbors(node) if nb in owner]
            if candidates:
                placed[node] = min(candidates)
        if not placed:
            break
        owner.update(placed)
        pending = [node for node in pending if node not in placed]
    for node in pending:
        owner[node] = UNATTRIBUTED_GROUP
    groups: dict[str, list[str]] = {}
    for node, repo in owner.items():
        groups.setdefault(repo, []).append(node)
    return {repo: sorted(nodes, key=str) for repo, nodes in sorted(groups.items())}


def label_still_applies(
    cid: int,
    members: list[str],
    previous_assignments: dict[str, int],
    saved_sig: str | None,
    current_sig: str | None,
) -> bool:
    """graphify's carry rule: same signature, or a strict majority overlap with
    the previous community of the same id in both directions."""
    from graphify.cluster import label_still_applies as upstream

    return upstream(cid, members, previous_assignments, saved_sig, current_sig)


_LABEL_BATCH_SIZE_DEFAULT = 50


def _label_batch_size() -> int:
    """Communities per labeling call, overridable with GRAPHIFY_LABEL_BATCH_SIZE.

    Read per call rather than at import so a test, or an operator editing the env file
    the sync unit reads, does not need this module reloaded to take effect.
    """
    raw = os.environ.get("GRAPHIFY_LABEL_BATCH_SIZE", "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return _LABEL_BATCH_SIZE_DEFAULT


def llm_names(
    G: nx.Graph,
    communities: dict[int, list[str]],
    *,
    backend: str,
    model: str,
    max_concurrency: int = LLM_DEFAULT_CONCURRENCY,
) -> tuple[dict[int, str], str]:
    """`({community_id: name}, source)` where source is "llm" or "placeholder".

    Never raises: upstream catches its own backend failures and degrades to
    `Community N` placeholders, which the caller treats as "no name yet".
    """
    from graphify.llm import generate_community_labels

    # max_concurrency defaults to 1, against upstream's default of 4
    # (`generate_community_labels` in graphify/llm.py), and is raised by
    # GRAPHIFY_MESH_LABEL_CONCURRENCY (Settings.label_concurrency).
    # Upstream's fan-out inflates the latency of each call when the backend
    # serves one request at a time, as the backend here did on 2026-09-21.
    # Measured that day, one 100-community batch: 12.2 s alone; four in
    # parallel took 41.5 s wall with the slowest single call also 41.5 s, against
    # 48.7 s to run the same four serially. That is 1.17x throughput for 3.4x
    # per-call latency — and per-call latency is exactly what
    # `Settings.ollama_api_timeout` bounds, so the fan-out was spending the
    # timeout budget it had no way to earn back. The 2026-09-21 production run
    # degraded 14 communities to provisional names this way. Raise the setting
    # once the backend handles parallel requests, and re-measure per-call latency.
    #
    # Upstream runs min(max_concurrency, batches in this call) workers, and each
    # call gets one chunk of LABEL_MAX_CONCURRENCY batches (below), so that
    # constant is the ceiling: a higher value would never get a worker. For the
    # ollama backend upstream also forces one worker unless
    # GRAPHIFY_OLLAMA_PARALLEL=1, so that is set for the call and restored after:
    # the child env allowlist forwards GRAPHIFY_*, and upstream's extraction
    # reads the same switch.
    # batch_size against upstream's default of 100 (_LABEL_BATCH_SIZE, graphify's
    # llm.py). That default is sized by context window, and the window was never the
    # binding constraint: measured 2026-10-01 through the shim at num_ctx=16384, a
    # 100-community prompt is 6.3k tokens and no reply was ever truncated. What a long
    # list costs is attention, not context — the model answers the first stretch of it
    # and stops. Same model, same communities, same prompt:
    #
    #   batch=100  53/100 names  1 call   15.1 s
    #   batch= 50  95/100 names  2 calls  21.2 s
    #   batch= 25  94/100 names  4 calls  15.4 s
    #
    # 25 buys nothing over 50, so 50 is the knee. The extra call is far cheaper than an
    # unnamed community: upstream re-asks the missing ids in split batches, and whatever
    # is still missing ships into the graph as a `Community N` placeholder.
    #
    # This is passed from here rather than patched into graphify because graphify is a
    # third-party package (github.com/Graphify-Labs/graphify) — a local edit there is
    # lost on the next upgrade, silently. `batch_size` is a documented keyword argument,
    # so nothing has to be forked to set it.
    #
    # Re-measure when the labeling model changes: the knee is a property of the model's
    # attention over a long list, not of graphify or of the window.
    #
    # Chunked here only for progress: one upstream call over every community printed
    # nothing for many minutes (quiet=True). Each chunk is LABEL_MAX_CONCURRENCY (4)
    # batches and upstream still batches inside it with the same batch_size, so the
    # 2026-10-01 measurement holds. The progress line reports whole chunks either way.
    batch_size = _label_batch_size()
    chunk_size = LABEL_MAX_CONCURRENCY * batch_size
    parallel_gate = (
        "GRAPHIFY_OLLAMA_PARALLEL" if backend == "ollama" and max_concurrency > 1 else None
    )
    saved_gate = os.environ.get(parallel_gate) if parallel_gate else None
    items = list(communities.items())
    total = len(items)
    chunk_count = max(1, -(-total // chunk_size))
    labels: dict[int, str] = {}
    source = "placeholder"
    any_llm = False
    started = time.monotonic()
    for index in range(chunk_count):
        chunk = dict(items[index * chunk_size : (index + 1) * chunk_size])
        if parallel_gate:
            os.environ[parallel_gate] = "1"
        try:
            chunk_labels, source = generate_community_labels(
                G,
                chunk,
                backend=backend,
                model=model,
                quiet=True,
                max_concurrency=max_concurrency,
                batch_size=batch_size,
            )
        finally:
            if parallel_gate:
                if saved_gate is None:
                    os.environ.pop(parallel_gate, None)
                else:
                    os.environ[parallel_gate] = saved_gate
        any_llm = any_llm or source == "llm"
        labels.update(chunk_labels)
        named = sum(
            1 for label in labels.values() if label and not PLACEHOLDER_RE.match(label.strip())
        )
        log.info(
            "naming: labeled %d/%d communities (%d named, chunk %d/%d, %.1fs)",
            min((index + 1) * chunk_size, total),
            total,
            named,
            index + 1,
            chunk_count,
            time.monotonic() - started,
        )
    return labels, ("llm" if any_llm else source)
