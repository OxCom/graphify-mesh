"""WS2 community-naming stage.

Given the merged global graph (post-strip, see `strip_project_community_attrs`)
and `Settings`, `run_naming`:

  1. Asserts the pinned clustering backend (C25, see `graphify_mesh.sync.backend`)
     before doing anything else — a mismatch is a hard failure that
     propagates uncaught, blocking the naming stage and (by not being
     swallowed anywhere in pipeline.py) publish end-to-end.
  2. Clusters and labels IN PROCESS through `graphify_mesh.sync.clustering`,
     never through `graphify cluster-only` / `graphify label`. Those
     subcommands rebuild the graph with `graphify.build.build_from_json`,
     which collapses nodes sharing `(source_file, label)` across repository
     boundaries and rewrites labels into full paths. The published graph is
     now the mesh's own merged dict plus `community`/`community_name`, and
     `assert_only_community_attrs_added` fails the run if it is anything else.
  3. Carries names across runs through `<naming_dir>/naming-state.json`
     (`graphify_mesh.sync.naming_state`), keyed by community membership
     signature rather than by community id — ids renumber whenever membership
     shifts. The state also records the clustering and labeling recipe, so a
     changed model, backend or resolution invalidates reuse.
  4. Degrades rather than crashes on a recoverable failure. Clustering is
     local computation and always runs; only labeling is gated on the health
     check. A community with no usable name takes a deterministic hub name and
     is recorded `provisional`, which the next healthy run relabels. The two
     failures that still propagate are the backend pin mismatch and
     `NamingIntegrityError` — both mean the run must not publish.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import sys
import urllib.request
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path

from graphify_mesh.sync import clustering, naming_state, publish
from graphify_mesh.sync.backend import BackendCheckResult, assert_pinned_backend
from graphify_mesh.sync.config import Settings, is_valid_http_base_url
from graphify_mesh.sync.tls import resolve_tls_mode, ssl_context
from graphify_mesh.sync.validate import PLACEHOLDER_RE

log = logging.getLogger("graphify_mesh.sync.naming")

LABELING_OK = "ok"
LABELING_DEGRADED = "degraded"
LABELING_REUSED = "ok (reused: merged-graph fingerprint unchanged)"

CLUSTER_RESOLUTION = 1.0

HealthCheckFn = Callable[[str, str, float], bool]


@dataclass
class NamingResult:
    labeling: str
    graph_data: dict
    backend: str
    reason: str = ""
    changed_cids: list = field(default_factory=list)
    backend_check: BackendCheckResult | None = None


def default_ollama_health_check(base_url: str, api_key: str, timeout: float) -> bool:
    """GET `{base_url}/models` — the lowest-cost OpenAI-compatible endpoint
    available to confirm the backend is reachable and authenticating,
    without triggering any actual generation/embedding work. Any failure
    (DNS, connect, timeout, non-2xx) means "unhealthy"; this must never
    raise out of the pipeline — a naming-stage health check failure is
    exactly the degraded path it exists to detect, not a crash.
    """
    url = base_url.rstrip("/") + "/models"
    if not is_valid_http_base_url(url):
        log.warning("ollama health check refused non-http(s) URL %s", url)
        return False
    req = urllib.request.Request(  # noqa: S310 - scheme validated above
        url, headers={"Authorization": f"Bearer {api_key}"}
    )
    try:
        with urllib.request.urlopen(  # noqa: S310 - fixed internal endpoint
            req, timeout=timeout, context=ssl_context()
        ) as resp:
            status = getattr(resp, "status", resp.getcode())
            return 200 <= status < 300
    except Exception as exc:  # noqa: BLE001 - any failure => unhealthy, never crash the pipeline
        log.warning("ollama health check failed for %s: %s", url, exc)
        return False


def strip_project_community_attrs(graph_data: dict) -> dict:
    """C23: unconditionally strip `community`/`community_name` from every
    node in the merged graph, before it is handed to the naming stage.

    These attributes were carried over from per-project graphs during merge
    and are meaningless (or actively misleading) at global scope: per-project
    community ids collide across repos, and per-project community names
    (e.g. "Alpha Domain") must never leak into the global graph's naming.
    Global `community`/`community_name` must always come from this stage's
    own fresh clustering, or — in degraded mode — from the last published
    GLOBAL generation (see pipeline.py's restore step), never from a
    per-project graph.

    Mutates `graph_data` in place and returns the same object — the merged
    graph is ~38K nodes and a full copy here was a second graph's worth of
    RAM for nothing; no caller uses the pre-strip dict after this point
    (see pipeline.py).
    """
    for node in graph_data.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node.pop("community", None)
        node.pop("community_name", None)
    return graph_data


_COMMUNITY_ATTRS = ("community", "community_name")


class NamingIntegrityError(RuntimeError):
    """The naming stage changed something other than community annotations.

    Publishing that graph would ship a silently altered structure — the exact
    failure this stage stopped delegating to graphify's builder to avoid — so
    the run fails instead.
    """


def structure_snapshot(graph_data: dict) -> dict:
    """Everything the guard compares, without holding a second copy of the graph."""
    nodes: dict[str, str] = {}
    for node in graph_data.get("nodes", []):
        if not isinstance(node, dict) or "id" not in node:
            continue
        if node["id"] in nodes:
            # Two nodes under one id would collapse into a single snapshot entry
            # and hide a later divergence. The merged graph must not contain one.
            raise NamingIntegrityError(f"duplicate node id in the merged graph: {node['id']!r}")
        payload = {k: v for k, v in node.items() if k not in _COMMUNITY_ATTRS}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        nodes[node["id"]] = digest
    return {"nodes": nodes, "edges": _edge_endpoints(graph_data)}


def assert_only_community_attrs_added(before_snapshot: dict, after: dict) -> None:
    """Fail unless `after` is the graph `before_snapshot` was taken of, plus
    `community`/`community_name`. Take the snapshot with `structure_snapshot`
    before the stage mutates anything."""
    after_snapshot = structure_snapshot(after)

    before_nodes = before_snapshot["nodes"]
    after_nodes = after_snapshot["nodes"]
    if set(before_nodes) != set(after_nodes):
        lost = sorted(set(before_nodes) - set(after_nodes))[:5]
        gained = sorted(set(after_nodes) - set(before_nodes))[:5]
        raise NamingIntegrityError(
            f"naming changed the node id set: {len(before_nodes)} -> {len(after_nodes)} "
            f"(lost e.g. {lost}, gained e.g. {gained})"
        )
    changed_nodes = sorted(
        node_id for node_id, digest in before_nodes.items() if after_nodes[node_id] != digest
    )
    if changed_nodes:
        raise NamingIntegrityError(
            f"naming changed attribute(s) on {len(changed_nodes)} node(s), e.g. {changed_nodes[:5]}"
        )
    if before_snapshot["edges"] != after_snapshot["edges"]:
        raise NamingIntegrityError(
            f"naming changed the edge set: {len(before_snapshot['edges'])} -> "
            f"{len(after_snapshot['edges'])} endpoint triples"
        )


def _edge_endpoints(graph_data: dict) -> list[str]:
    """One digest per edge, over the WHOLE edge record.

    Endpoints plus relation would miss weights, confidence and every other edge
    attribute. Sorting the digests (not the records) keeps multiplicity, so a
    multigraph's duplicate edges cannot collapse into one entry.
    """
    links = graph_data.get("links", graph_data.get("edges", []))
    return sorted(
        hashlib.sha256(json.dumps(link, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        for link in links
        if isinstance(link, dict)
    )


def _clustering_recipe(backend: str) -> dict:
    return {"backend": backend, "resolution": CLUSTER_RESOLUTION}


def _labeling_recipe(settings: Settings) -> dict:
    """The endpoint is part of the recipe: the same model on a different server
    is a different labeler, and two units on this host already point the same
    variable at different hosts."""
    return {
        "backend": "ollama",
        "model": settings.ollama_model,
        "base_url": settings.ollama_base_url,
    }


def bind_upstream_backend(settings: Settings, env: MutableMapping[str, str] | None = None) -> str:
    """Point graphify's ollama adapter at the endpoint this stage health-probes,
    bound its per-call timeout, and return the URL it will actually use.

    `BACKENDS["ollama"]["base_url"]` is resolved when `graphify.llm` is first
    imported (graphify/llm.py:128), and the client is built from that cached
    value — a later environment change does not move it. So the environment is
    set BEFORE the first import, and the cached value is corrected and asserted
    afterwards. The mesh probes `GRAPHIFY_MESH_OLLAMA_BASE_URL` while graphify
    labels against `OLLAMA_BASE_URL`; nothing else enforces that they agree, and
    the sibling MCP daemon unit already sets a different (HTTPS) value.
    """
    # `env` is injectable so a test does not leak these two keys into the rest
    # of the pytest process; production passes os.environ, which is what
    # graphify reads.
    env = os.environ if env is None else env
    # Only the API key has to travel through the environment: upstream reads it
    # at call time (_get_backend_api_key), while the base URL is patched on
    # BACKENDS below and the model is passed explicitly. Writing OLLAMA_BASE_URL
    # and OLLAMA_MODEL here would also leak into every child spawned afterwards,
    # because the child env allowlist forwards the OLLAMA_ prefix
    # (graphify_cli.py:97).
    if settings.ollama_api_key:
        env["OLLAMA_API_KEY"] = settings.ollama_api_key
    # The CLI child used to be bounded by GRAPHIFY_MESH_CLI_TIMEOUT; in process
    # that bound is gone and upstream's own default is 600 s PER CALL with zero
    # retries for ollama (llm.py:407, :1387). 1097 communities batch into ~11
    # sequential calls, so a wedged backend — "healthy /models, hung
    # completions", recorded on this host 2026-08-17 — could sit for 6600 s on
    # top of a 75-90 minute pipeline and be SIGKILLed by TimeoutStartSec,
    # skipping every cleanup. A per-call ceiling keeps the worst case bounded.
    env["GRAPHIFY_API_TIMEOUT"] = str(settings.ollama_api_timeout)

    from graphify.llm import BACKENDS

    BACKENDS["ollama"]["base_url"] = settings.ollama_base_url
    effective = BACKENDS["ollama"]["base_url"]
    if effective != settings.ollama_base_url:
        raise RuntimeError(
            f"graphify would label against {effective!r} while this stage probes "
            f"{settings.ollama_base_url!r}"
        )
    return effective


def _tls_allows_labeling(settings: Settings, effective_url: str) -> bool:
    """Refuse to label over an intercepted HTTPS chain instead of failing silently.

    graphify builds its own `openai` client and exposes no transport seam
    (`generate_community_labels` takes no client/http_client argument), so
    `sync/tls.py`'s relaxed mode does not reach it. On this host an intercepted
    HTTPS endpoint fails certificate validation, and
    `generate_community_labels` turns that into placeholder names rather than an
    error. Loopback HTTP — the configured naming endpoint — is unaffected.
    """
    if not effective_url.lower().startswith("https://"):
        return True
    try:
        mode = resolve_tls_mode()
    except ValueError as exc:
        # An unparsable GRAPHIFY_MESH_TLS_MODE used to be swallowed by the
        # health-check try block; it must not start crashing the stage now.
        log.warning("labeling skipped: %s", exc)
        return False
    if mode == "strict":
        return True
    log.warning(
        "labeling skipped: %s is https and GRAPHIFY_MESH_TLS_MODE=%s — graphify builds its own "
        "client and this package's relaxed context cannot reach it, so a failure here would "
        "surface as placeholder names rather than an error",
        effective_url,
        mode,
    )
    return False


def _openai_importable() -> bool:
    """graphify's ollama backend builds its client with `from openai import
    OpenAI`. Checked explicitly so a missing client degrades with a stated
    reason instead of silently becoming placeholder names."""
    return importlib.util.find_spec("openai") is not None


def _reuse_is_safe(
    state: naming_state.NamingState | None,
    fingerprint: str,
    graph_data: dict,
    recipe_ok: bool,
) -> bool:
    """Reuse only against state that fully covers this graph and is final.

    Fingerprint equality alone is not enough: it authenticates the INPUT, not
    the stored assignments. A state missing one node would publish that node
    with no community at all, and `validate_community_names` skips nodes whose
    `community` is None, so nothing downstream would catch it.
    """
    if state is None or not recipe_ok or state.merged_fingerprint != fingerprint:
        return False
    node_ids = {n["id"] for n in graph_data.get("nodes", []) if isinstance(n, dict) and "id" in n}
    if set(state.assignments) != node_ids:
        return False
    for cid in set(state.assignments.values()):
        entry = state.communities.get(str(cid))
        if entry is None or not entry.name:
            return False
    # A provisional name means the LLM has not spoken for that community yet.
    # Returning REUSED here is what froze hub names permanently: an unchanged
    # graph never gets a second chance once the backend recovers.
    return not any(e.provisional for e in state.communities.values())


def run_naming(
    graphify_bin: str,
    naming_dir: Path,
    staging_home: Path,  # noqa: ARG001 - kept for call-site compatibility; no child runs
    merged_graph_data: dict,
    settings: Settings,
    health_check: HealthCheckFn | None = None,
) -> NamingResult:
    backend_check = assert_pinned_backend()
    fingerprint = publish.output_hash(merged_graph_data)
    state = naming_state.load(naming_dir)
    # Two separate eligibilities, deliberately not one flag:
    #   - clustering recipe decides whether the stored ASSIGNMENTS may seed id
    #     stabilization. A labeling-model change must not renumber communities.
    #   - labeling recipe decides whether stored NAMES may be carried.
    clustering_ok = state is not None and state.clustering == _clustering_recipe(
        backend_check.backend
    )
    labeling_ok = state is not None and state.labeling == _labeling_recipe(settings)
    recipe_ok = clustering_ok and labeling_ok

    if state is not None and _reuse_is_safe(state, fingerprint, merged_graph_data, recipe_ok):
        reuse_before = structure_snapshot(merged_graph_data)
        names = {cid: e.name for cid, e in state.communities.items()}
        _apply(merged_graph_data, state.assignments, names)
        # The reuse path writes onto the graph too, so it is guarded like any
        # other return path — a stale or malformed state must not reshape it.
        assert_only_community_attrs_added(reuse_before, merged_graph_data)
        log.info("naming: merged-graph fingerprint unchanged — reusing stored names")
        return NamingResult(
            labeling=LABELING_REUSED,
            graph_data=merged_graph_data,
            backend=backend_check.backend,
            backend_check=backend_check,
        )

    before = structure_snapshot(merged_graph_data)
    G = clustering.graph_from_node_link(merged_graph_data)
    previous = state.assignments if (state is not None and clustering_ok) else {}
    communities = clustering.cluster_graph(
        G, resolution=CLUSTER_RESOLUTION, previous_assignments=previous
    )
    sigs = clustering.membership_sigs(communities)

    if state is None:
        # Migration runs once, only when no state file exists. Seeding on an
        # empty `carried` instead would resurrect sidecars a later run had
        # legitimately emptied.
        carried = {
            e.sig: (e.name, e.provisional)
            for e in naming_state.seed_from_graphify_sidecars(naming_dir).values()
        }
    else:
        carried = naming_state.names_by_sig(state) if labeling_ok else {}
        if not labeling_ok:
            log.info(
                "naming: labeling recipe changed since the last run (model or backend) — "
                "every community will be relabeled; community ids stay stable"
            )

    names_by_cid: dict[int, str] = {}
    provisional: dict[int, bool] = {}
    needs_llm: dict[int, list[str]] = {}
    for cid, members in communities.items():
        carried_name, was_provisional = carried.get(sigs[cid], ("", True))
        if carried_name and not was_provisional:
            names_by_cid[cid] = carried_name
            provisional[cid] = False
        else:
            needs_llm[cid] = members

    effective_url = bind_upstream_backend(settings)
    healthy = _backend_healthy(settings, health_check, effective_url)
    llm_source = ""
    if needs_llm and healthy:
        llm_labels, llm_source = clustering.llm_names(
            G, needs_llm, backend="ollama", model=settings.ollama_model
        )
        for cid in list(needs_llm):
            label = (llm_labels.get(cid) or "").strip()
            if label and not PLACEHOLDER_RE.match(label):
                names_by_cid[cid] = label
                provisional[cid] = False
                needs_llm.pop(cid)

    if needs_llm:
        fallback = clustering.hub_names(G, needs_llm)
        for cid in needs_llm:
            # A carried name that is only *provisional* (a migrated sidecar
            # entry, or a hub name from an earlier outage) is still better than
            # a fresh hub name: it may be a real LLM name from before the
            # migration. Keep the text, keep the provisional flag, relabel on
            # the next healthy run.
            carried_text = carried.get(sigs[cid], ("", True))[0]
            names_by_cid[cid] = carried_text or fallback[cid]
            provisional[cid] = True

    assignments = {node_id: cid for cid, members in communities.items() for node_id in members}
    _apply(merged_graph_data, assignments, {str(cid): n for cid, n in names_by_cid.items()})
    assert_only_community_attrs_added(before, merged_graph_data)

    naming_state.save(
        naming_dir,
        naming_state.NamingState(
            merged_fingerprint=fingerprint,
            clustering=_clustering_recipe(backend_check.backend),
            labeling=_labeling_recipe(settings),
            communities={
                str(cid): naming_state.CommunityEntry(
                    sig=sigs[cid], name=names_by_cid[cid], provisional=provisional[cid]
                )
                for cid in communities
            },
            assignments=assignments,
        ),
    )

    any_provisional = any(provisional.values())
    changed = sorted(str(cid) for cid, was in provisional.items() if was)
    if any_provisional:
        reason = (
            "ollama unreachable — provisional hub names kept"
            if not healthy
            else f"labeling incomplete (source={llm_source or 'none'})"
        )
        log.warning("naming: %d community(ies) provisional — %s", len(changed), reason)
        return NamingResult(
            labeling=LABELING_DEGRADED,
            graph_data=merged_graph_data,
            backend=backend_check.backend,
            changed_cids=changed,
            reason=reason,
            backend_check=backend_check,
        )
    _retire_legacy_sidecars(naming_dir)
    return NamingResult(
        labeling=LABELING_OK,
        graph_data=merged_graph_data,
        backend=backend_check.backend,
        backend_check=backend_check,
    )


_LEGACY_SIDECARS = (
    ".graphify_labels.json",
    ".graphify_labels.json.sig",
    "graph.json",
    "GRAPH_REPORT.md",
)


def _retire_legacy_sidecars(naming_dir: Path) -> None:
    """Delete the files the CLI-based stage used to own.

    Only on a fully successful run: a degraded run has not demonstrated that
    `naming-state.json` carries everything these held, and they are the only
    fallback if the migration turns out wrong.
    """
    out_dir = naming_dir / "graphify-out"
    for name in _LEGACY_SIDECARS:
        path = out_dir / name
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:  # a stale file is untidy, never fatal
            log.warning("could not remove legacy sidecar %s: %s", path, exc)


def _apply(graph_data: dict, assignments: dict, names: dict) -> None:
    """Write `community`/`community_name` onto nodes. Mutates in place: the
    merged graph is ~33K nodes and a copy here costs a second graph's RAM."""
    for node in graph_data.get("nodes", []):
        if not isinstance(node, dict):
            continue
        cid = assignments.get(node.get("id"))
        if cid is None:
            continue
        node["community"] = int(cid)
        name = names.get(str(cid)) or names.get(cid)
        if name:
            node["community_name"] = name


def _backend_healthy(
    settings: Settings, health_check: HealthCheckFn | None, effective_url: str
) -> bool:
    if not _tls_allows_labeling(settings, effective_url):
        return False
    if not _openai_importable():
        log.warning(
            "openai is not importable in %s — labeling skipped, clustering still runs. "
            "Install the package's openai dependency.",
            sys.executable,
        )
        return False
    if not is_valid_http_base_url(settings.ollama_base_url):
        log.warning(
            "invalid ollama base URL %r — labeling skipped, clustering still runs",
            settings.ollama_base_url,
        )
        return False
    check = health_check if health_check is not None else default_ollama_health_check
    healthy = check(
        settings.ollama_base_url, settings.ollama_api_key, settings.ollama_health_timeout
    )
    if not healthy:
        log.warning("ollama unhealthy at %s — labeling skipped", settings.ollama_base_url)
    return healthy
