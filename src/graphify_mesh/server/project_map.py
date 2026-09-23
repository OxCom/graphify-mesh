"""`project_map(repo)` tool (WS5 tool 4): a structural overview of one
registered repo in the CURRENT generation — node count, a trimmed
community_name breakdown, and the top hub nodes by structural degree. Purely
read-only over the already-loaded `Generation`; `Generation.nodes_by_repo` / `.adjacency`
already cover everything this needs, so no separate index is built here.

Fails closed like everything else in this package: an unresolvable repo
(unknown to the registry, or simply absent from the current generation —
e.g. registered but never synced yet) returns `resolved=False` with a
`degraded` reason, never a silently-empty-looking success.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from graphify_mesh.server import ranking
from graphify_mesh.server.store import Generation

# How many top-degree nodes to surface per repo. Kept small and named
# (project style: no magic numbers) — this is an orientation summary, not a
# full node dump.
TOP_HUBS_LIMIT = 15

# `community_breakdown` keeps only communities of at least this size, and at
# most this many of them (largest first). Community labels are LLM-named
# clusters: an unreliable orientation hint, not a module map. Singletons were
# most of the list — about 4 KB of noise per call on cem.hub's 128
# communities. `communities_total` / `communities_omitted` report the cut.
COMMUNITY_MIN_SIZE = 2
COMMUNITY_BREAKDOWN_LIMIT = 25


@dataclass
class ProjectMapResult:
    resolved: bool
    repo: str = ""
    node_count: int = 0
    community_breakdown: dict = field(default_factory=dict)
    communities_total: int = 0  # every community in the repo, "unassigned" included
    communities_omitted: int = 0  # communities_total minus len(community_breakdown)
    top_hubs: list = field(default_factory=list)  # [{key, label, degree, source_file, is_hub}]
    degraded: list = field(default_factory=list)


def project_map(repo_id: str, generation: Generation) -> ProjectMapResult:
    node_ids = generation.nodes_by_repo.get(repo_id)
    if not node_ids:
        return ProjectMapResult(
            resolved=False, repo=repo_id, degraded=["repo_not_in_current_generation"]
        )

    community_breakdown: dict[str, int] = {}
    scored: list[tuple[int, str, str]] = []  # (degree, key, node_id) — key used as tie-break
    for node_id in node_ids:
        node = generation.node_by_id.get(node_id, {})
        community = node.get("community_name") or "unassigned"
        community_breakdown[community] = community_breakdown.get(community, 0) + 1
        key = generation.key_by_node_id.get(node_id)
        if key is None:
            continue
        scored.append((generation.degree(node_id), key, node_id))

    # Deterministic: degree desc, then key asc (never dict/set iteration order).
    scored.sort(key=lambda t: (-t[0], t[1]))
    top_hubs = []
    for degree, key, node_id in scored[:TOP_HUBS_LIMIT]:
        node = generation.node_by_id.get(node_id, {})
        top_hubs.append(
            {
                "key": key,
                "label": node.get("label", ""),
                "degree": degree,
                "source_file": node.get("source_file", ""),
                "is_hub": degree > ranking.HUB_DEGREE_THRESHOLD,
            }
        )

    # Deterministic: size desc, then name asc; then the size floor and limit.
    ranked_communities = sorted(community_breakdown.items(), key=lambda kv: (-kv[1], kv[0]))
    kept = [kv for kv in ranked_communities if kv[1] >= COMMUNITY_MIN_SIZE]
    kept = kept[:COMMUNITY_BREAKDOWN_LIMIT]

    return ProjectMapResult(
        resolved=True,
        repo=repo_id,
        node_count=len(node_ids),
        community_breakdown=dict(kept),
        communities_total=len(ranked_communities),
        communities_omitted=len(ranked_communities) - len(kept),
        top_hubs=top_hubs,
    )
