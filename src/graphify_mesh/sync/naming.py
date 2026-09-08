"""WS2 community-naming stage.

Given the merged global graph (post-strip, see `strip_project_community_attrs`)
and `Settings`, `run_naming`:

  1. Asserts the pinned clustering backend (C25, see `graphify_mesh.sync.backend`)
     before doing anything else — a mismatch is a hard failure that
     propagates uncaught, blocking the naming stage and (by not being
     swallowed anywhere in pipeline.py) publish end-to-end.
  2. Health-checks the configured Ollama endpoint with a short timeout. On
     failure, `cluster-only`/`label` are not invoked AT ALL — this is
     "degraded" mode: whatever `community`/`community_name` the input graph
     already carries (which, by the time pipeline.py calls this, is nothing —
     see `strip_project_community_attrs`) is returned untouched by this
     stage. The pipeline-level restore-from-last-published-global fallback
     (C23) lives in pipeline.py, not here, so this stage's "untouched"
     contract stays simple and testable in isolation.
  3. On success: stages the graph at `<naming_dir>/graphify-out/graph.json`
     (a persistent, pipeline-owned workspace — NOT the ephemeral per-run
     staging tempdir, and NOT any real project directory) so `cluster-only`/
     `label` write outputs beside it. Runs `cluster-only --no-viz` (cheap,
     deterministic hub-name fallback + sig-gated label reuse). Snapshots
     `.graphify_labels.json.sig` before that call, diffs it against the
     freshly written one afterward to find new/changed community ids,
     deletes exactly those ids from `.graphify_labels.json`, then runs
     `label --missing-only` so the LLM backend is invoked ONLY for
     communities that are new or whose membership actually changed —
     everything else keeps its previous name untouched.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from graphify_mesh.sync import graphify_cli, publish
from graphify_mesh.sync.backend import BackendCheckResult, assert_pinned_backend
from graphify_mesh.sync.config import Settings, is_valid_http_base_url
from graphify_mesh.sync.tls import ssl_context

log = logging.getLogger("graphify_mesh.sync.naming")

LABELING_OK = "ok"
LABELING_DEGRADED = "degraded"
LABELING_REUSED = "ok (reused: merged-graph fingerprint unchanged)"
# Canonical-hash sidecar of the STRIPPED merged graph that produced the
# named graph.json sitting next to it. Written only after a fully
# successful (LABELING_OK) naming run — reuse is only ever offered against
# a known-good named graph, never a degraded/partial one.
FINGERPRINT_FILENAME = ".merged-graph.fingerprint"

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


def _read_json_object(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _diff_changed_cids(old_sigs: dict, new_sigs: dict) -> list[str]:
    """cids that are new this run or whose membership fingerprint changed
    relative to the previous run's sig sidecar."""
    return sorted(cid for cid, sig in new_sigs.items() if old_sigs.get(cid) != sig)


def _try_reuse_named_graph(
    graph_path: Path, fingerprint_path: Path, fingerprint: str
) -> dict | None:
    """Last run's fully-named graph, iff its fingerprint sidecar matches the
    current stripped merged graph AND the named file still carries at least
    one community_name (paranoia against a hand-truncated file). None means
    'do a full naming run'."""
    if not fingerprint_path.is_file() or not graph_path.is_file():
        return None
    try:
        stored = fingerprint_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if stored != fingerprint:
        return None
    named = _read_json_object(graph_path)
    if not isinstance(named, dict):
        return None
    nodes = named.get("nodes", [])
    named_count = sum(1 for n in nodes if isinstance(n, dict) and n.get("community_name"))
    if not nodes or named_count == 0:
        return None
    return named


def run_naming(
    graphify_bin: str,
    naming_dir: Path,
    staging_home: Path,
    merged_graph_data: dict,
    settings: Settings,
    health_check: HealthCheckFn | None = None,
) -> NamingResult:
    backend_check = assert_pinned_backend(graphify_bin)

    out_dir = naming_dir / "graphify-out"
    graph_path = out_dir / "graph.json"
    fingerprint_path = out_dir / FINGERPRINT_FILENAME
    fingerprint = publish.output_hash(merged_graph_data)

    reused = _try_reuse_named_graph(graph_path, fingerprint_path, fingerprint)
    if reused is not None:
        log.info(
            "naming: merged-graph fingerprint unchanged — reusing on-disk names, "
            "skipping cluster-only + label (no Ollama calls)"
        )
        return NamingResult(
            labeling=LABELING_REUSED,
            graph_data=reused,
            backend=backend_check.backend,
            backend_check=backend_check,
        )

    # URL scheme gate BEFORE any health check runs: a base URL that is not
    # plain http(s)-with-host (file://, gopher://, ...) fails this stage's
    # health-check path outright — degraded mode, zero requests attempted.
    if not is_valid_http_base_url(settings.ollama_base_url):
        log.warning(
            "invalid ollama base URL %r (scheme must be http or https with a non-empty host) — "
            "skipping cluster-only/label entirely (degraded mode)",
            settings.ollama_base_url,
        )
        return NamingResult(
            labeling=LABELING_DEGRADED,
            graph_data=merged_graph_data,
            backend=backend_check.backend,
            reason="invalid ollama base URL",
            backend_check=backend_check,
        )

    check = health_check if health_check is not None else default_ollama_health_check
    healthy = check(
        settings.ollama_base_url, settings.ollama_api_key, settings.ollama_health_timeout
    )

    if not healthy:
        log.warning(
            "ollama unhealthy at %s — skipping cluster-only/label entirely (degraded mode)",
            settings.ollama_base_url,
        )
        return NamingResult(
            labeling=LABELING_DEGRADED,
            graph_data=merged_graph_data,
            backend=backend_check.backend,
            reason="ollama health check failed",
            backend_check=backend_check,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    # Any crash between here and the OK-path sidecar write must leave no
    # sidecar matching a half-labeled (or fully unlabeled) graph.json on
    # disk — otherwise the next run's reuse gate would serve a stale,
    # partially-named graph as if it were complete.
    fingerprint_path.unlink(missing_ok=True)
    # Overwrite this run's graph, but leave any pre-existing
    # .graphify_labels.json / .sig from a prior run in place — that is
    # exactly what makes sig-gated label reuse possible across runs.
    # Streamed json.dump — never materializes the multi-MB serialized graph
    # as one string. Direct (non-atomic) overwrite preserves the previous
    # semantics here: the fingerprint sidecar was already unlinked above, so
    # a crash mid-write can never be mistaken for a reusable named graph.
    with graph_path.open("w", encoding="utf-8") as fh:
        json.dump(merged_graph_data, fh)

    labels_path = out_dir / ".graphify_labels.json"
    sig_path = out_dir / ".graphify_labels.json.sig"
    old_sigs = _read_json_object(sig_path)

    cluster_result = graphify_cli.run_cluster_only(graphify_bin, naming_dir, staging_home)
    if not cluster_result.ok:
        raise RuntimeError(
            f"graphify cluster-only failed: exit={cluster_result.returncode}: "
            f"{cluster_result.stderr.strip()[:500]}"
        )

    new_sigs = _read_json_object(sig_path)
    changed_cids = _diff_changed_cids(old_sigs, new_sigs)

    if changed_cids:
        labels = _read_json_object(labels_path)
        for cid in changed_cids:
            labels.pop(cid, None)
        labels_path.write_text(json.dumps(labels), encoding="utf-8")

        label_result = graphify_cli.run_label(
            graphify_bin,
            naming_dir,
            staging_home,
            backend="ollama",
            model=settings.ollama_model,
        )
        if not label_result.ok:
            # The pre-flight health check passed (server was up), but the
            # actual `graphify label` call failed — a mid-run network blip,
            # the Ollama host going down partway through a long labeling
            # job, etc. This must degrade, not crash: clustering (local,
            # already succeeded) is still valid, so fall back to whatever
            # names are already on disk (a mix of prior names plus any
            # communities the label call finished before failing) rather
            # than losing the whole naming stage — and critically, rather
            # than crashing the entire pipeline run (which would also throw
            # away the embedding/overlay/publish stages that haven't even
            # run yet).
            log.warning(
                "graphify label failed mid-run (exit=%s: %s) — "
                "degrading naming for this generation, "
                "keeping whatever names are already on disk",
                label_result.returncode,
                label_result.stderr.strip()[:500],
            )
            final_graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
            return NamingResult(
                labeling=LABELING_DEGRADED,
                graph_data=final_graph_data,
                backend=backend_check.backend,
                changed_cids=changed_cids,
                reason=f"graphify label failed: {label_result.stderr.strip()[:300]}",
                backend_check=backend_check,
            )

    final_graph_data = json.loads(graph_path.read_text(encoding="utf-8"))

    # Verify the write actually happened before trusting exit-code success.
    # A real, observed failure mode: `graphify cluster-only`/`label` can hit
    # their OWN internal shrink-guard (e.g. a malformed node produces a
    # node-count mismatch against the input), print "Done - N communities"
    # and exit 0, yet silently refuse to write community/community_name onto
    # any node. Trusting cluster_result.ok/label_result.ok alone would report
    # LABELING_OK for a generation that carries zero real names — this check
    # catches that specific upstream failure mode rather than propagating a
    # false success.
    nodes = final_graph_data.get("nodes", [])
    named_count = sum(1 for n in nodes if isinstance(n, dict) and n.get("community_name"))
    if nodes and named_count == 0:
        log.warning(
            "graphify cluster-only/label exited 0 but wrote zero community_name values onto "
            "%d node(s) — a known upstream failure mode (internal shrink-guard silently refusing "
            "the write, e.g. after a malformed node caused a node-count mismatch); degrading "
            "rather than reporting a false LABELING_OK",
            len(nodes),
        )
        return NamingResult(
            labeling=LABELING_DEGRADED,
            graph_data=merged_graph_data,
            backend=backend_check.backend,
            changed_cids=changed_cids,
            reason="graphify cluster-only/label exited 0 but wrote no community_name onto any node",
            backend_check=backend_check,
        )

    fingerprint_path.write_text(fingerprint + "\n", encoding="utf-8")

    return NamingResult(
        labeling=LABELING_OK,
        graph_data=final_graph_data,
        backend=backend_check.backend,
        changed_cids=changed_cids,
        backend_check=backend_check,
    )


def restore_last_global_community_names(
    graph_data: dict, previous_global_graph: dict | None
) -> dict:
    """Degraded-mode fallback (C23).

    Interpretation chosen (see WS2 design deliverable 3): when Ollama is
    down, the naming stage skips clustering/labeling entirely and the graph
    handed to `run_naming` already had `community`/`community_name` stripped
    unconditionally (see `strip_project_community_attrs`). Left as-is, every
    node would publish with no community name at all, even though nothing
    about most communities actually changed.

    Rather than gratuitously losing every name on a brief outage, or
    reaching into the current merge's per-project inputs (which would leak
    per-project names into the global graph — exactly what C23 forbids),
    restore each node's community_name ONLY from the last PUBLISHED GLOBAL
    generation (`global-graph.json` behind `global/current`). A node with no
    entry there (new node, or first generation ever) stays unnamed rather
    than inventing a name from nothing.

    Mutates `graph_data` in place and returns the same object — same
    rationale as `strip_project_community_attrs`: the merged graph is ~38K
    nodes, and copying every node dict here while `previous_global_graph` is
    also resident cost a whole extra graph's worth of peak RAM for nothing
    (no caller uses the pre-restore dict afterwards, see pipeline.py).
    """
    if not previous_global_graph:
        return graph_data
    prev_names = {
        node["id"]: node.get("community_name")
        for node in previous_global_graph.get("nodes", [])
        if isinstance(node, dict) and "id" in node and node.get("community_name")
    }
    if not prev_names:
        return graph_data
    for node in graph_data.get("nodes", []):
        if not isinstance(node, dict):
            continue
        prior_name = prev_names.get(node.get("id"))
        if prior_name:
            node["community_name"] = prior_name
    return graph_data
