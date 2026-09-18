"""Pre-publish validation of the merged global graph (WS1 item 7).

All checks return (ok: bool, errors: list[str]) so the caller can aggregate
everything before deciding to publish or not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from graphify_mesh.sync.config import FORBIDDEN_OVERLAY_RELATION_TYPES
from graphify_mesh.sync.sync_project import _shrink_floor

# A placeholder is exactly the literal the CLI stamps when no label exists —
# "Community <cid>". A prefix match is wrong: real LLM labels can start with
# the word ("Community Management", observed 2026-08-18) and must pass.
PLACEHOLDER_RE = re.compile(r"^Community \d+$")

# Per-check cap on collected error strings: a badly broken merge can yield
# millions of dangling/forbidden edges, and status.json must not balloon.
# The ok/fail decision always uses the true count, never the capped list.
MAX_ERRORS_PER_CHECK = 50


def _record_error(errors: list[str], total: int, message: str) -> int:
    """Count an error, appending its message only while under the cap.
    Returns the updated true total."""
    total += 1
    if total <= MAX_ERRORS_PER_CHECK:
        errors.append(message)
    return total


def _capped_result(errors: list[str], total: int) -> ValidationResult:
    overflow = total - MAX_ERRORS_PER_CHECK
    if overflow > 0:
        errors.append(f"... +{overflow} more error(s) omitted (total {total})")
    return ValidationResult(ok=total == 0, errors=errors)


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


def validate_schema(data: dict) -> ValidationResult:
    errors = []
    if not isinstance(data.get("nodes"), list):
        errors.append("schema: 'nodes' missing or not a list")
    if not isinstance(data.get("links", data.get("edges")), list):
        errors.append("schema: 'links'/'edges' missing or not a list")
    for node in data.get("nodes", []):
        if not isinstance(node, dict) or "id" not in node:
            errors.append(f"schema: node missing 'id': {node!r}")
            break
    for link in data.get("links", data.get("edges", [])):
        if not isinstance(link, dict) or "source" not in link or "target" not in link:
            errors.append(f"schema: link missing 'source'/'target': {link!r}")
            break
    return ValidationResult(ok=not errors, errors=errors)


def validate_shrink_guard(
    new_counts: tuple[int, int],
    previous_counts: tuple[int, int] | None,
    allow_shrink: bool,
    tolerance: float = 0.0,
) -> ValidationResult:
    """Global-graph counterpart of the per-repo shrink guard.

    `tolerance` mirrors sync_project._classify_shrink: the naming stage's
    canonicalization merges (doc twins, ghosts) and LLM extraction jitter
    legitimately shrink the merged graph by a fraction of a percent per run,
    and a strict guard blocks every subsequent publish (observed 2026-07-21..
    08-18). A material collapse is still refused.
    """
    if previous_counts is None or allow_shrink:
        return ValidationResult(ok=True)
    new_nodes, new_edges = new_counts
    prev_nodes, prev_edges = previous_counts
    errors = []
    if new_nodes < _shrink_floor(prev_nodes, tolerance):
        errors.append(
            f"shrink-guard: new global graph has {new_nodes} nodes, "
            f"previous published had {prev_nodes}"
        )
    if new_edges < _shrink_floor(prev_edges, tolerance):
        errors.append(
            f"shrink-guard: new global graph has {new_edges} edges, "
            f"previous published had {prev_edges}"
        )
    return ValidationResult(ok=not errors, errors=errors)


def validate_dangling_ids(data: dict) -> ValidationResult:
    """C20: every edge endpoint must resolve to a node that exists in the merged graph."""
    node_ids = {n["id"] for n in data.get("nodes", []) if isinstance(n, dict) and "id" in n}
    errors: list[str] = []
    total = 0
    for link in data.get("links", data.get("edges", [])):
        if not isinstance(link, dict):
            continue
        src, dst = link.get("source"), link.get("target")
        if src not in node_ids:
            total = _record_error(
                errors, total, f"dangling-id: edge source {src!r} not in merged node set"
            )
        if dst not in node_ids:
            total = _record_error(
                errors, total, f"dangling-id: edge target {dst!r} not in merged node set"
            )
    return _capped_result(errors, total)


def repo_prefix(node_id: object) -> str | None:
    """Extract the `<repo_id>` half of a merged `<repo_id>::<local_id>` node
    id, or None if the id doesn't follow that convention (external/bare node)."""
    if not isinstance(node_id, str) or "::" not in node_id:
        return None
    repo_id, _, _ = node_id.partition("::")
    return repo_id


def validate_forbidden_edges(data: dict) -> ValidationResult:
    """The structural merged output must never contain cross-repo edges (C5);
    they belong exclusively in the WS4 overlay artifact.

    Three rules, in order:

    1. `cross_repo:true` is an error.
    2. Two endpoints whose `<repo_id>::` prefixes differ is an error for EVERY
       relation type, not only the overlay relation types. A structural
       relation such as `calls` spanning two repos is still a cross-repo edge.
    3. An overlay relation type (semantically_similar_to and friends) is an
       error when either endpoint carries no repo prefix, since a bare endpoint
       can only originate from the overlay's external-node handling.

    Same-repo edges sharing one of the overlay's relation-type strings are NOT
    forbidden: upstream `graphify`'s own semantic extraction can legitimately
    emit a same-repo `depends_on` edge (e.g. a Helm `Chart.yaml` subchart
    dependency, a package.json same-repo reference) as normal EXTRACTED data.

    This check is defense in depth, not the primary mechanism. `graphify
    merge-graphs` emits cross-repo `same_type_as` and `calls` edges itself, and
    `pipeline.strip_cross_repo_edges` removes them from the merged graph right
    after the repo-tag remap, long before validation runs. A cross-repo edge
    that still reaches this function is therefore a genuine bug — a stage that
    added one after the strip, or a strip that missed a shape — and blocking
    the publish is the right answer.
    """
    errors: list[str] = []
    total = 0
    for link in data.get("links", data.get("edges", [])):
        if not isinstance(link, dict):
            continue
        if link.get("cross_repo") is True:
            total = _record_error(
                errors,
                total,
                f"forbidden-edge: cross_repo:true edge {link.get('source')}->{link.get('target')}",
            )
            continue
        rel = link.get("relation") or link.get("type")
        src_repo = repo_prefix(link.get("source"))
        dst_repo = repo_prefix(link.get("target"))
        if src_repo is not None and dst_repo is not None and src_repo != dst_repo:
            total = _record_error(
                errors,
                total,
                f"forbidden-edge: cross-repo edge {src_repo!r}->{dst_repo!r} "
                f"with relation {rel!r} on {link.get('source')}->{link.get('target')}",
            )
            continue
        if rel not in FORBIDDEN_OVERLAY_RELATION_TYPES:
            continue
        if src_repo is not None and src_repo == dst_repo:
            continue  # same-repo edge, legitimate upstream-extracted data
        total = _record_error(
            errors,
            total,
            f"forbidden-edge: cross-repo overlay relation {rel!r} "
            f"on {link.get('source')}->{link.get('target')}",
        )
    return _capped_result(errors, total)


def validate_community_names(data: dict, skip_labeling: bool) -> ValidationResult:
    """Every clustered node must have a non-placeholder community_name.

    Labeling is WS2/WS3 work; WS1 implements the check but allows bypassing
    it via --skip-labeling (logged reason), since no labeling stage runs yet.
    """
    if skip_labeling:
        return ValidationResult(
            ok=True,
            errors=["community-name check skipped: --skip-labeling (labeling not wired until WS2)"],
        )
    errors = []
    for node in data.get("nodes", []):
        if not isinstance(node, dict):
            continue
        if node.get("community") is None:
            continue
        name = node.get("community_name")
        if not name or PLACEHOLDER_RE.match(name):
            errors.append(f"placeholder community_name on node {node.get('id')!r}: {name!r}")
    return ValidationResult(ok=not errors, errors=errors)


def validate_generation_manifest(manifest: dict) -> ValidationResult:
    """C28: generation manifest internal-consistency check."""
    required_keys = {
        "generation_id",
        "created_at",
        "repo_input_hashes",
        "registry_hash",
        "config_hash",
        "output_node_count",
        "output_edge_count",
        "output_hash",
        "clustering_backend",
        "embedding_model",
        "labeling",
        "stale_repos",
    }
    missing = required_keys - set(manifest.keys())
    errors = [f"generation-manifest: missing key {k!r}" for k in sorted(missing)]
    if not missing and not isinstance(manifest["repo_input_hashes"], dict):
        errors.append("generation-manifest: repo_input_hashes must be an ordered mapping")
    return ValidationResult(ok=not errors, errors=errors)


def run_all(
    data: dict,
    previous_counts: tuple[int, int] | None,
    allow_shrink: bool,
    skip_labeling: bool,
    shrink_tolerance: float = 0.0,
) -> ValidationResult:
    schema = validate_schema(data)
    if not schema.ok:
        # Downstream checks assume well-formed data; stop early.
        return schema

    new_counts = (len(data.get("nodes", [])), len(data.get("links", data.get("edges", []))))
    checks = [
        schema,
        validate_shrink_guard(new_counts, previous_counts, allow_shrink, shrink_tolerance),
        validate_dangling_ids(data),
        validate_forbidden_edges(data),
        validate_community_names(data, skip_labeling),
    ]
    all_errors: list[str] = []
    hard_fail = False
    # No skip_labeling special case needed here: validate_community_names
    # already returns ok=True (with an informational message) in skip mode.
    for check in checks:
        if not check.ok:
            hard_fail = True
        all_errors.extend(check.errors)
    return ValidationResult(ok=not hard_fail, errors=all_errors)
