"""WS4 orchestrator: build the cross-project overlay artifact.

Slots into the pipeline between the WS3 embed-changed stage and the WS5
lexical-index stage (plan WS1 item 6 order: `... -> label -> embed changed ->
overlay resolve -> lexical index -> validate -> atomic publish`). The
overlay is a SEPARATE
artifact (`cross-project-overlay.json`, staged alongside `global-graph.json`
and `generation-manifest.json` inside the same generation dir) and is NEVER
merged into the structural graph.json (C5) — `validate.validate_forbidden_edges`
is the invariant that catches any accidental leak into structural output.

Every logical ref this module produces is resolved fresh against the current
generation's inputs (C27) — nothing here persists a raw graph node id across
runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from graphify_mesh.sync.embed_similarity import PairCacheSource
from graphify_mesh.sync.overlay_api import (
    ApiFactsSource,
    collect_api_facts,
    match_provides_consumes_edges,
)
from graphify_mesh.sync.overlay_depends import (
    build_manual_relation_edges,
    build_package_identity_map,
    extract_depends_on_edges,
    load_manual_relations,
)
from graphify_mesh.sync.overlay_refs import OverlayEdge
from graphify_mesh.sync.overlay_similar import compute_similar_approach_edges
from graphify_mesh.sync.state import IgnoreRules
from graphify_mesh.sync.vectors import RepoVectors

log = logging.getLogger("graphify_mesh.sync.overlay")

OVERLAY_ARTIFACT_FILENAME = "cross-project-overlay.json"
OVERLAY_SCHEMA_VERSION = 1


@dataclass
class OverlayResult:
    edges: list[OverlayEdge] = field(default_factory=list)
    edge_counts_by_type: dict[str, int] = field(default_factory=dict)
    manual_relation_count: int = 0
    errors: list[str] = field(default_factory=list)


def load_graphs_by_repo(graph_paths_by_repo: dict[str, Path]) -> dict[str, dict]:
    return load_graphs_and_sha256(graph_paths_by_repo)[0]


def load_graphs_and_sha256(
    graph_paths_by_repo: dict[str, Path],
) -> tuple[dict[str, dict], dict[str, bytes]]:
    """Per-repo graphs plus the sha256 of the exact bytes each was parsed from.
    Cache keys take this digest, never a second read: graphify can rewrite
    graph.json mid-run, which would file an old-graph result under a new key."""
    graphs: dict[str, dict] = {}
    sha256_by_repo: dict[str, bytes] = {}
    for repo_id, path in graph_paths_by_repo.items():
        if not path.is_file():
            continue
        try:
            raw = path.read_bytes()
        except OSError as exc:
            log.warning(
                "overlay: could not load per-repo graph for %s (%s): %s", repo_id, path, exc
            )
            continue
        sha256 = hashlib.sha256(raw).digest()
        # Drop the bytes before parsing: only the text and the parsed graph coexist.
        text = raw.decode("utf-8")
        del raw
        try:
            graphs[repo_id] = json.loads(text)
        except json.JSONDecodeError as exc:
            log.warning(
                "overlay: could not load per-repo graph for %s (%s): %s", repo_id, path, exc
            )
            continue
        finally:
            del text
        sha256_by_repo[repo_id] = sha256
    return graphs, sha256_by_repo


def build_overlay(
    graph_paths_by_repo: dict[str, Path],
    repo_roots_by_id: dict[str, Path],
    manual_relations_path: Path,
    manual_relations_schema_path: Path,
    similar_top_k: int = 5,
    embedding_vectors_by_repo: dict[str, RepoVectors] | None = None,
    embedding_model: str = "unknown",
    graphs_by_repo: dict[str, dict] | None = None,
    pair_cache: PairCacheSource | None = None,
    exact_max_rows: int | None = None,
    ignore_rules_by_repo: dict[str, IgnoreRules] | None = None,
    api_cache: ApiFactsSource | None = None,
) -> OverlayResult:
    """Runs all three WS4 ship-order stages (depends_on+manual,
    similar_approach, provides/consumes API) and returns every overlay edge
    for this generation. Dangling manual-relation refs raise
    `overlay_refs.DanglingReferenceError` uncaught — same hard-fail
    convention as the WS2 naming stage's `BackendMismatchError` (crashes the
    run rather than silently degrading).

    `graphs_by_repo` may be passed in (pipeline.py already loads it once for
    the WS3 embed stage and reuses it here rather than reading every
    per-repo graph.json twice); if omitted it is loaded fresh from
    `graph_paths_by_repo`, same as before WS3.

    `ignore_rules_by_repo` holds each repo's persisted build options for the
    API scan (state.read_build_ignore_rules); a missing repo gets graphify's
    defaults. `api_cache` may serve an unchanged repo's API facts."""
    if graphs_by_repo is None:
        graphs_by_repo = load_graphs_by_repo(graph_paths_by_repo)
    edges: list[OverlayEdge] = []

    # 1. depends_on (manifest + lockfile) + manual relations.
    package_identity_map = build_package_identity_map(repo_roots_by_id)
    for repo_id, root in sorted(repo_roots_by_id.items()):
        edges.extend(extract_depends_on_edges(repo_id, root, package_identity_map))

    raw_relations: list[dict] = []
    if manual_relations_path.is_file():
        schema = json.loads(manual_relations_schema_path.read_text(encoding="utf-8"))
        raw_relations = load_manual_relations(manual_relations_path, schema)
    manual_edges = build_manual_relation_edges(raw_relations, graphs_by_repo)
    edges.extend(manual_edges)

    # 2. similar_approach: real ANN mutual top-k over the WS3 embedding
    # index when available, falling back to the exact-match placeholder for
    # any node without a vector (see overlay_similar.py module docstring).
    edges.extend(
        compute_similar_approach_edges(
            graphs_by_repo,
            embedding_vectors_by_repo=embedding_vectors_by_repo,
            top_k=similar_top_k,
            embedding_model=embedding_model,
            pair_cache=pair_cache,
            exact_max_rows=exact_max_rows,
        )
    )

    # 3. provides/consumes API (Symfony route attributes only — no
    # swagger/openapi spec exists in any registered repo, see overlay_api.py
    # module docstring). One walk per repo feeds both sides.
    providers_by_repo, consumer_candidates_by_repo = collect_api_facts(
        repo_roots_by_id, ignore_rules_by_repo, api_cache
    )
    edges.extend(match_provides_consumes_edges(providers_by_repo, consumer_candidates_by_repo))

    counts: dict[str, int] = {}
    for edge in edges:
        counts[edge.type] = counts.get(edge.type, 0) + 1

    return OverlayResult(
        edges=edges, edge_counts_by_type=counts, manual_relation_count=len(manual_edges)
    )


def overlay_artifact(result: OverlayResult, generation_id: str, created_at: str) -> dict:
    return {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "generation_id": generation_id,
        "created_at": created_at,
        "edge_counts_by_type": result.edge_counts_by_type,
        "manual_relation_count": result.manual_relation_count,
        "edges": [e.to_dict() for e in result.edges],
    }
