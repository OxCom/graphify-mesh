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
import re
import sys
import urllib.request
from collections.abc import Callable, Iterable, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

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
# Bump on any change to the grounding rules: a refused label is remembered only
# under the rules that refused it, so the next run asks those communities again.
# Grounded names keep carrying; this is not part of the labeler's identity.
GROUNDING_VERSION = 1

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


def _clustering_recipe(backend: str, scope: str = "global") -> dict:
    recipe: dict = {"backend": backend, "resolution": CLUSTER_RESOLUTION}
    if scope != "global":
        # `global` keeps the recipe it had before cluster_scope existed, so a
        # rollback to it reuses a state that version wrote. Any other scope is
        # a different clustering: the flip relabels once, by design.
        recipe["scope"] = scope
    return recipe


def _labeling_recipe(settings: Settings) -> dict:
    """The endpoint is part of the recipe: the same model on a different server
    is a different labeler, and two units on this host already point the same
    variable at different hosts."""
    return {
        "backend": "ollama",
        "model": settings.ollama_model,
        "base_url": settings.ollama_base_url,
        "grounding": GROUNDING_VERSION,
    }


def _labeler(recipe: dict) -> dict:
    """The recipe minus the grounding version, which governs only ungrounded
    entries: a rule change must not relabel names that passed the check."""
    return {k: v for k, v in recipe.items() if k != "grounding"}


def bind_upstream_backend(settings: Settings, env: MutableMapping[str, str] | None = None) -> str:
    """Point graphify's ollama adapter at the endpoint this stage health-probes,
    bound its per-call timeout, and return the URL it will actually use.

    `BACKENDS["ollama"]["base_url"]` is resolved when `graphify.llm` is first
    imported (the `BACKENDS` dict in graphify/llm.py), and the client is built from that cached
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
    # that bound is gone and upstream's own default is 600 s PER CALL
    # (`_resolve_api_timeout` in graphify/llm.py). Zero SDK retries for ollama
    # applies only to the extraction path (`_call_openai_compat`); the labeling
    # path (`_call_llm`) uses `_resolve_max_retries()`, default 6, unless
    # GRAPHIFY_MAX_RETRIES is set, so a hung call can repeat: one labeling call
    # can wait (1 + retries) x timeout. 1097 communities batch into ~11
    # sequential calls, so a wedged backend — "healthy /models, hung
    # completions", recorded on this host 2026-08-17 — could sit for roughly
    # 11 x (1 + retries) x timeout on top of a 75-90 minute pipeline and be
    # SIGKILLed by TimeoutStartSec, skipping every cleanup. A lower per-call
    # timeout shrinks that product; it does not bound it by itself.
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
    # An ungrounded hub name is provisional but already answered for this
    # membership and recipe, which `recipe_ok` and the fingerprint pin.
    return not any(e.provisional and not e.ungrounded for e in state.communities.values())


def _orphan_name(label: str) -> str:
    """Readable final name for an edgeless single-node community.

    The LLM sees only the mangled id of such a node and no neighbours, so it
    returns nothing usable; the label itself is all the context there is.
    """
    spaced = label.removeprefix("ref_").translate(str.maketrans("_/\\.-", "     "))
    name = " ".join(spaced.split())
    if not name or PLACEHOLDER_RE.match(name):
        return label
    return name


# Words that occur in labels or paths of any community, so they prove nothing.
_GROUNDING_STOPWORDS = frozenset(
    {
        "an", "as", "at", "be", "by", "do", "if", "in", "is", "it", "of", "on", "or", "to",
        "and", "the", "for", "with", "from", "into", "via", "its", "per", "over", "under",
        "using", "based", "other", "misc", "various", "related", "general", "common",
        "util", "utils", "utility", "utilities", "helper", "helpers", "handling",
        "management", "logic", "module", "modules", "code", "file", "files", "src", "lib",
        "var", "www", "home",
    }
)  # fmt: skip
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_WORD_RE = re.compile(r"[\W_]+")
# A label word this long may sit inside a member word ("config" in "tsconfig");
# shorter ones would match by accident ("api" in "rapid").
_SUBSTRING_MIN = 4
# A member word this long may be the stem of a label word ("conf" of
# "configuration"). Three letters grounded "Settings" by `setFoo`.
_MEMBER_PREFIX_MIN = 4
# A compound member word may end in a label word's first letters ("env" of
# "environment" in "dotenv").
_SUFFIX_STEM_LEN = 3
# A label word longer than this is also matched by its first letters alone, the
# shared stem of its derived forms.
_STEM_LEN = 6


def _words(text: str) -> list[str]:
    return [w for w in _NON_WORD_RE.split(_CAMEL_RE.sub(" ", text).casefold()) if w]


def _grounding_tokens(text: str) -> set[str]:
    """Casefolded word tokens, split on non-alphanumerics and camelCase, with a
    light plural strip so "migrations" meets "Migration". Two-letter tokens stay
    ("ui", "cc", "r3" from "R3.0.0") but only ever match exactly."""
    tokens: set[str] = set()
    for tok in _words(text):
        if len(tok) < 2 or tok in _GROUNDING_STOPWORDS:
            continue
        if tok.endswith("ies") and len(tok) > 4:
            tok = tok[:-3] + "y"
        elif tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3:
            tok = tok[:-1]
        tokens.add(tok)
    return tokens


def _token_matches(tok: str, member: str) -> bool:
    if tok == member:
        return True
    if len(tok) >= _SUBSTRING_MIN and tok in member:
        return True
    if len(member) >= _MEMBER_PREFIX_MIN and len(tok) > len(member) and tok.startswith(member):
        return True
    if len(tok) > _STEM_LEN and tok[:_STEM_LEN] in member:
        return True  # "configuration" in "tsconfig", "sanitization" in "sanitize"
    # Compound member words keep the stem at their end: "env" of "environment"
    # in "dotenv".
    stem = tok[:_SUFFIX_STEM_LEN]
    return len(tok) >= 2 * _SUFFIX_STEM_LEN and len(member) > len(stem) and member.endswith(stem)


def _is_grounded(label: str, member_texts: Iterable[str]) -> bool:
    """True when a word of `label` occurs in the community's own member labels or
    paths. A label of another stack ("Doctrine ORM" over Go types) shares none.
    A community with no usable member words cannot refute any label."""
    members: set[str] = set()
    for text in member_texts:
        members |= _grounding_tokens(text)
    if not members:
        return True
    candidates = _grounding_tokens(label)
    # "Data Transfer Object" is grounded by a `Dto` member.
    initials = "".join(w[0] for w in _words(label) if not w.isdigit())
    if len(initials) >= 3 and initials in members:
        return True
    return any(_token_matches(tok, m) for tok in candidates for m in members)


def _member_texts(G: nx.Graph, members: list[str]) -> list[str]:
    texts: list[str] = []
    for node in members:
        if not G.has_node(node):
            continue
        attrs = G.nodes[node]
        texts.extend(str(attrs[k]) for k in ("label", "source_file") if attrs.get(k))
    return texts


@dataclass
class _Partition:
    """Communities in global ids, plus what is already known about their names."""

    communities: dict[int, list[str]] = field(default_factory=dict)
    sigs: dict[int, str] = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)
    provisional: dict[int, bool] = field(default_factory=dict)
    needs_llm: dict[int, list[str]] = field(default_factory=dict)
    # cid -> the LLM label the grounding check refused: provisional hub names
    # that are not asked again for the same membership and labeling recipe.
    ungrounded: dict[int, str] = field(default_factory=dict)
    # sig -> (text, provisional): fallback text for a community left unnamed.
    carried: dict[str, tuple[str, bool]] = field(default_factory=dict)
    # cluster_scope=repo only.
    repo_of: dict[int, str] = field(default_factory=dict)
    offsets: dict[str, int] = field(default_factory=dict)
    fingerprints: dict[str, str] = field(default_factory=dict)


def _carried_names(
    state: naming_state.NamingState | None,
    labeling_ok: bool,
    settings: Settings,
    naming_dir: Path,
) -> dict[str, tuple[str, bool]]:
    if settings.relabel_all:
        # Names only: stored assignments still seed id stabilization, so a
        # relabel renames communities without renumbering them.
        return {}
    if state is None:
        # Migration runs once, only when no state file exists. Seeding on an
        # empty `carried` instead would resurrect sidecars a later run had
        # legitimately emptied.
        return {
            e.sig: (e.name, e.provisional)
            for e in naming_state.seed_from_graphify_sidecars(naming_dir).values()
        }
    if not labeling_ok:
        log.info(
            "naming: labeling recipe changed since the last run (model or backend) — "
            "every community will be relabeled; community ids stay stable"
        )
        return {}
    return naming_state.names_by_sig(state)


def _partition_global(
    G: nx.Graph,
    state: naming_state.NamingState | None,
    clustering_ok: bool,
    carried: dict[str, tuple[str, bool]],
    ungrounded_by_sig: dict[str, str],
) -> _Partition:
    """cluster_scope=global: one clustering over the whole merged graph."""
    previous = state.assignments if (state is not None and clustering_ok) else {}
    communities = clustering.cluster_graph(
        G, resolution=CLUSTER_RESOLUTION, previous_assignments=previous
    )
    part = _Partition(
        communities=communities, sigs=clustering.membership_sigs(communities), carried=carried
    )
    for cid, members in communities.items():
        carried_name, was_provisional = carried.get(part.sigs[cid], ("", True))
        if carried_name and not was_provisional:
            part.names[cid] = carried_name
            part.provisional[cid] = False
        else:
            part.needs_llm[cid] = members
            if part.sigs[cid] in ungrounded_by_sig:
                part.ungrounded[cid] = ungrounded_by_sig[part.sigs[cid]]
    return part


def _group_fingerprint(G: nx.Graph, nodes: list[str], node_dicts: dict) -> str:
    """Digest of one repo's nodes (all attributes but the community ones) and
    its internal edges — every input its clustering and labeling read."""
    digest = hashlib.sha256()
    for node_id in nodes:
        payload = {
            k: v for k, v in node_dicts.get(node_id, {}).items() if k not in _COMMUNITY_ATTRS
        }
        digest.update(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
        digest.update(b"\x00")
    directed = G.is_directed()
    edges = []
    for u, v, data in G.subgraph(nodes).edges(data=True):
        ends = [str(u), str(v)] if directed else sorted((str(u), str(v)))
        edges.append(json.dumps([ends, data], sort_keys=True, default=str))
    for edge in sorted(edges):
        digest.update(edge.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _partition_by_repo(
    G: nx.Graph,
    graph_data: dict,
    state: naming_state.NamingState | None,
    clustering_ok: bool,
    labeling_ok: bool,
    grounding_ok: bool,
    carried: dict[str, tuple[str, bool]],
    settings: Settings,
) -> _Partition:
    """cluster_scope=repo: each repo's subgraph is clustered on its own.

    Valid because the structural graph has no cross-repo edges (invariant 1).
    Global ids are repo-local ids plus an offset taken in sorted repo order. A
    repo whose fingerprint matches the stored one reuses its partition and its
    names: no clustering, no LLM call. In a changed repo a name carries by
    exact signature, else by graphify's majority-overlap rule.
    """
    node_dicts = {
        n["id"]: n for n in graph_data.get("nodes", []) if isinstance(n, dict) and "id" in n
    }
    # Stored partitions are trusted only under the same clustering recipe.
    prev_repos = state.repos if (state is not None and clustering_ok) else {}
    carry = labeling_ok and not settings.relabel_all
    part = _Partition(carried=dict(carried))
    offset = 0
    reused = reclustered = by_sig = by_overlap = 0
    for repo, nodes in clustering.repo_groups(G).items():
        fingerprint = _group_fingerprint(G, nodes, node_dicts)
        prev = prev_repos.get(repo)
        prev_assign = prev.assignments if prev is not None else {}
        if prev is not None and prev.fingerprint == fingerprint and set(prev_assign) == set(nodes):
            local: dict[int, list[str]] = {}
            for node_id, lcid in prev_assign.items():
                local.setdefault(lcid, []).append(node_id)
            local = {lcid: sorted(members) for lcid, members in sorted(local.items())}
            status = "unchanged, partition reused"
            reused += 1
        else:
            local = clustering.cluster_graph(
                G.subgraph(nodes).copy(),
                resolution=CLUSTER_RESOLUTION,
                previous_assignments=prev_assign,
            )
            status = "changed, reclustered"
            reclustered += 1
        local_sigs = clustering.membership_sigs(local)
        prev_entries = prev.communities if (prev is not None and carry) else {}
        prev_by_sig = {e.sig: e for e in prev_entries.values()}
        to_label = 0
        for lcid in sorted(local):
            members = local[lcid]
            gcid = offset + lcid
            sig = local_sigs[lcid]
            part.communities[gcid] = members
            part.sigs[gcid] = sig
            part.repo_of[gcid] = repo
            entry = prev_by_sig.get(sig)
            if entry is not None:
                by_sig += 1
            else:
                old = prev_entries.get(str(lcid))
                if (
                    old is not None
                    and not old.ungrounded  # a membership change re-asks the LLM
                    and clustering.label_still_applies(lcid, members, prev_assign, old.sig, sig)
                ):
                    entry = old
                    by_overlap += 1
            if entry is not None and entry.name:
                part.carried[sig] = (entry.name, entry.provisional)
            text, was_provisional = part.carried.get(sig, ("", True))
            if text and not was_provisional:
                part.names[gcid] = text
                part.provisional[gcid] = False
            else:
                part.needs_llm[gcid] = members
                if entry is not None and entry.ungrounded and grounding_ok:
                    part.ungrounded[gcid] = entry.ungrounded
                else:
                    to_label += 1
        part.offsets[repo] = offset
        part.fingerprints[repo] = fingerprint
        offset += (max(local) + 1) if local else 0
        log.info(
            "naming: repo %s: %s — %d communities, %d to label",
            repo or "<unattributed>",
            status,
            len(local),
            to_label,
        )
    log.info(
        "naming: %d repo(s) unchanged, %d reclustered; names carried by signature %d, "
        "by overlap %d; %d community(ies) to label",
        reused,
        reclustered,
        by_sig,
        by_overlap,
        len(part.needs_llm),
    )
    return part


def _repo_states(part: _Partition) -> dict[str, naming_state.RepoEntry]:
    repos = {
        repo: naming_state.RepoEntry(fingerprint=part.fingerprints[repo]) for repo in part.offsets
    }
    for gcid, repo in part.repo_of.items():
        lcid = gcid - part.offsets[repo]
        entry = repos[repo]
        entry.communities[str(lcid)] = naming_state.CommunityEntry(
            sig=part.sigs[gcid],
            name=part.names[gcid],
            provisional=part.provisional[gcid],
            ungrounded=part.ungrounded.get(gcid, ""),
        )
        for node_id in part.communities[gcid]:
            entry.assignments[node_id] = lcid
    return repos


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
    scope = settings.cluster_scope
    clustering_recipe = _clustering_recipe(backend_check.backend, scope)
    clustering_ok = state is not None and state.clustering == clustering_recipe
    labeling_recipe = _labeling_recipe(settings)
    labeling_ok = state is not None and _labeler(state.labeling) == _labeler(labeling_recipe)
    # Stored ungrounded labels are trusted only under the rules that refused them.
    grounding_ok = labeling_ok and state is not None and state.labeling == labeling_recipe
    recipe_ok = clustering_ok and grounding_ok

    if settings.relabel_all:
        log.info(
            "naming: relabel_all is on — every community goes to the LLM; carried names "
            "and fingerprint reuse are ignored, community ids stay stable"
        )
    if (
        not settings.relabel_all
        and state is not None
        and _reuse_is_safe(state, fingerprint, merged_graph_data, recipe_ok)
    ):
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
    if scope == "repo":
        # Names carried from a different clustering recipe (a scope flip) are
        # dropped: the flip relabels once, and the grounding check sees them.
        carried = (
            _carried_names(state, labeling_ok, settings, naming_dir)
            if state is None or clustering_ok
            else {}
        )
        part = _partition_by_repo(
            G, merged_graph_data, state, clustering_ok, labeling_ok, grounding_ok, carried, settings
        )
    else:
        part = _partition_global(
            G,
            state,
            clustering_ok,
            _carried_names(state, labeling_ok, settings, naming_dir),
            {e.sig: e.ungrounded for e in state.communities.values() if e.ungrounded}
            if (state is not None and grounding_ok and not settings.relabel_all)
            else {},
        )
    communities = part.communities
    sigs = part.sigs
    names_by_cid = part.names
    provisional = part.provisional
    needs_llm = part.needs_llm
    carried = part.carried

    effective_url = bind_upstream_backend(settings)
    healthy = _backend_healthy(settings, health_check, effective_url)
    llm_source = ""
    rejected = 0
    ask = {cid: m for cid, m in needs_llm.items() if cid not in part.ungrounded}
    if ask and healthy:
        llm_labels, llm_source = clustering.llm_names(
            G, ask, backend="ollama", model=settings.ollama_model
        )
        # Grounding check: a label must share a word with its own members' labels
        # or paths. Names shared across repos are fine ("doctrine migrations" in
        # every Symfony repo) and find_similar relies on that equality.
        for cid in sorted(ask):
            label = (llm_labels.get(cid) or "").strip()
            if not label or PLACEHOLDER_RE.match(label):
                continue
            if not _is_grounded(label, _member_texts(G, ask[cid])):
                rejected += 1
                part.ungrounded[cid] = label
                log.info(
                    "naming: label %r for community %d (%s) rejected — not grounded in "
                    "member labels",
                    label,
                    cid,
                    part.repo_of.get(cid) or "<unattributed>",
                )
                continue
            names_by_cid[cid] = label
            provisional[cid] = False
            needs_llm.pop(cid)

    if needs_llm:
        fallback = clustering.hub_names(G, needs_llm)
        for cid in needs_llm:
            members = needs_llm[cid]
            if len(members) == 1 and G.has_node(members[0]) and G.degree(members[0]) == 0:
                # An orphan stub (observed 2026-10-13: an `external` ref node with
                # 0 edges) is never named by the LLM, so a provisional flag here
                # would keep every run degraded and re-ask it forever.
                node = members[0]
                names_by_cid[cid] = _orphan_name(str(G.nodes[node].get("label") or node))
                provisional[cid] = False
                continue
            if cid in part.ungrounded:
                # Provisional, so a membership or recipe change asks again; the
                # recorded label keeps the same membership from being re-asked.
                names_by_cid[cid] = fallback[cid]
                provisional[cid] = True
                continue
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
            clustering=clustering_recipe,
            labeling=labeling_recipe,
            communities={
                str(cid): naming_state.CommunityEntry(
                    sig=sigs[cid],
                    name=names_by_cid[cid],
                    provisional=provisional[cid],
                    ungrounded=part.ungrounded.get(cid, ""),
                )
                for cid in communities
            },
            assignments=assignments,
            repos=_repo_states(part) if scope == "repo" else {},
        ),
    )

    rejected_note = f"{rejected} label(s) rejected as not grounded in member labels, hub names kept"
    # An ungrounded community has had its answer for this membership; counting
    # it as degraded would report an outage on every run until membership moves.
    changed = sorted(
        str(cid) for cid, was in provisional.items() if was and cid not in part.ungrounded
    )
    any_provisional = bool(changed)
    if any_provisional:
        reason = (
            "ollama unreachable — provisional hub names kept"
            if not healthy
            else f"labeling incomplete (source={llm_source or 'none'})"
        )
        if rejected:
            reason += f"; {rejected_note}"
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
    if rejected:
        log.info("naming: %s", rejected_note)
    return NamingResult(
        labeling=LABELING_OK,
        graph_data=merged_graph_data,
        backend=backend_check.backend,
        reason=rejected_note if rejected else "",
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
