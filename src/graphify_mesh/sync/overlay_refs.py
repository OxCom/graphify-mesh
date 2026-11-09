"""WS4 shared contract: durable logical references + overlay edge shape (C27).

Every overlay edge (depends_on / similar_approach / provides_api /
consumes_api / manual) references its endpoints as logical refs
`{repo, source_file, qualified_label[, signature]}` — never a raw graph node
id, and never anything that assumes id stability across generations (C20).
Raw per-repo `graph.json` node ids are only ever used internally, within a
single generation's resolution pass, to confirm a logical ref actually
resolves to something real right now; they are never stored in the overlay
artifact itself.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from graphify_mesh.sync.config_keys import config_key_paths


@dataclass(frozen=True)
class LogicalRef:
    repo: str
    source_file: str
    qualified_label: str
    signature: str | None = None

    def to_dict(self) -> dict:
        payload = {
            "repo": self.repo,
            "source_file": self.source_file,
            "qualified_label": self.qualified_label,
        }
        if self.signature:
            payload["signature"] = self.signature
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> LogicalRef:
        return cls(
            repo=data["repo"],
            source_file=data["source_file"],
            qualified_label=data["qualified_label"],
            signature=data.get("signature"),
        )

    def to_key(self) -> str:
        """Opaque, stable string form of this logical ref for use as a dict
        key (e.g. WS3 embedding shards/id-map, C27) where a hashable string
        rather than the dataclass itself is more convenient. Uses a
        unit-separator so repo/source_file/label values containing `:` or
        other common punctuation never collide."""
        return "\x1f".join((self.repo, self.source_file, self.qualified_label))


@dataclass(frozen=True)
class OverlayEdge:
    type: str
    source: LogicalRef
    target: LogicalRef
    provenance: str
    confidence: float
    evidence: str

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "provenance": self.provenance,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> OverlayEdge:
        return cls(
            type=data["type"],
            source=LogicalRef.from_dict(data["source"]),
            target=LogicalRef.from_dict(data["target"]),
            provenance=data["provenance"],
            confidence=data["confidence"],
            evidence=data["evidence"],
        )


class DanglingReferenceError(ValueError):
    """A manual-relation (or otherwise user-declared) logical ref does not
    resolve against the current generation's per-repo graphs. Per plan WS4:
    dangling references are a hard error, not a warning — this must propagate
    uncaught out of the overlay stage, blocking publish, exactly like
    `BackendMismatchError` blocks the WS2 naming stage."""


class AmbiguousReferenceError(DanglingReferenceError):
    """A manual-relation ref names a config key by its bare label, and more
    than one nested key of that file carries the label. Fails publish like a
    dangling ref: picking one would bind the relation to an arbitrary key."""


@dataclass(frozen=True)
class RefResolution:
    """What a `(source_file, label)` lookup found: the node and the label its
    durable key carries, or `node=None` with the dotted paths that all match."""

    node: dict | None
    qualified_label: str
    candidates: tuple[str, ...] = ()


def build_repo_node_index(graph_data: dict) -> dict[tuple[str, str], RefResolution]:
    """Index a single repo's raw graph.json nodes by (source_file, label) so
    a logical ref can be resolved against it. Nodes missing either field are
    not indexable and are skipped (they cannot be a dangling-ref target).

    A YAML/JSON key node is indexed under its dotted key path, the label its
    durable key carries. Its bare label stays resolvable as a fallback when no
    other node of the file claims it, so a manual relation written before
    dotted paths existed does not turn into a dangling reference. A bare label
    shared by several nested keys resolves to none of them."""
    paths = config_key_paths(graph_data)
    index: dict[tuple[str, str], RefResolution] = {}
    bare_matches: dict[tuple[str, str], list[tuple[str, dict]]] = {}
    for node in graph_data.get("nodes", []):
        if not isinstance(node, dict):
            continue
        source_file = node.get("source_file")
        label = node.get("label")
        if source_file is None or label is None:
            continue
        qualified = paths.get(node.get("id"))
        if qualified is not None and qualified != label:
            index[(source_file, qualified)] = RefResolution(node, qualified)
            bare_matches.setdefault((source_file, label), []).append((qualified, node))
        else:
            index[(source_file, label)] = RefResolution(node, label)
    for entry, matches in bare_matches.items():
        if entry in index:
            continue
        if len(matches) == 1:
            qualified, node = matches[0]
            index[entry] = RefResolution(node, qualified)
        else:
            index[entry] = RefResolution(None, entry[1], tuple(sorted(q for q, _ in matches)))
    return index


def _lookup(
    ref: LogicalRef,
    graphs_by_repo: dict[str, dict],
    index_cache: dict[str, dict] | None,
) -> RefResolution | None:
    graph_data = graphs_by_repo.get(ref.repo)
    if graph_data is None:
        return None
    if index_cache is None:
        return build_repo_node_index(graph_data).get((ref.source_file, ref.qualified_label))
    if ref.repo not in index_cache:
        index_cache[ref.repo] = build_repo_node_index(graph_data)
    return index_cache[ref.repo].get((ref.source_file, ref.qualified_label))


def resolve_ref(
    ref: LogicalRef,
    graphs_by_repo: dict[str, dict],
    index_cache: dict[str, dict] | None = None,
) -> dict | None:
    """Resolve a logical ref against this generation's per-repo graphs.
    Returns the matching node dict, or None if the repo is unknown this
    generation, no node matches (source_file, qualified_label), or a bare
    config-key label matches several nested keys.

    When `index_cache` is supplied, the per-repo node index is built once
    and reused across calls sharing the same cache dict (keyed by repo id)
    instead of being rebuilt from scratch on every call."""
    resolution = _lookup(ref, graphs_by_repo, index_cache)
    return resolution.node if resolution is not None else None


def canonical_ref(
    ref: LogicalRef,
    graphs_by_repo: dict[str, dict],
    context: str,
    index_cache: dict[str, dict] | None = None,
) -> LogicalRef:
    """`ref` with the label its node's durable key carries: a bare config-key
    label becomes its dotted path, so the server finds the edge under the
    node's key. Raises DanglingReferenceError when nothing matches and
    AmbiguousReferenceError when a bare label matches several nested keys."""
    resolution = _lookup(ref, graphs_by_repo, index_cache)
    if resolution is None:
        raise DanglingReferenceError(
            f"dangling reference in {context}: repo={ref.repo!r} source_file={ref.source_file!r} "
            f"qualified_label={ref.qualified_label!r} "
            "does not resolve against the current generation"
        )
    if resolution.node is None:
        raise AmbiguousReferenceError(
            f"ambiguous reference in {context}: repo={ref.repo!r} "
            f"source_file={ref.source_file!r} qualified_label={ref.qualified_label!r} "
            f"matches {len(resolution.candidates)} config keys "
            f"({', '.join(resolution.candidates)}); use the dotted path"
        )
    return replace(ref, qualified_label=resolution.qualified_label)
