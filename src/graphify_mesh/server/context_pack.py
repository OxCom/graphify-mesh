"""`context_pack(goal, scope, token_budget)` tool (WS5 tool 5): evidence
cards built from `retrieval.rank`'s ranked hits, each carrying an explicit
`[repo:path:line]` citation (same shape as everywhere else in this
package), a snippet, and a confidence flag. Truncates strictly by whole-card
boundary — a card is either fully included or fully excluded, it is never
sliced mid-way (plan WS5 tool contract: "budgets").

Confidence flag is always `ranking.CONFIDENCE_EXTRACTED` here:
`context_pack` does not expose an `include_inferred` toggle the way
`rank()` itself does, so every hit this function can ever surface is, by
construction, drawn from the EXTRACTED-only default structural traversal
plus lexical/vector matches — there is no code path by which an
INFERRED-sourced hit reaches this function today. The field is still
carried explicitly (not hardcoded as a bare string) so a future
`include_inferred` passthrough only needs to plumb the real value through,
not invent the field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from graphify_mesh.server import ranking
from graphify_mesh.server.retrieval import EmbedQueryFn, Hit, rank
from graphify_mesh.server.scope import RegistryEntry
from graphify_mesh.server.store import Generation
from graphify_mesh.sync.embedding import build_snippet, node_line

# Rough, dependency-free token estimate — no tokenizer dependency added just
# for a budget heuristic. ~4 chars/token is the commonly-cited English-text
# average; deliberately errs on the side of UNDER-counting cards (budget
# runs out slightly early rather than ever overshooting the caller's stated
# token_budget).
CHARS_PER_TOKEN_ESTIMATE = 4

# How many ranked candidates to consider building cards from before token
# budget truncation. Independent of the caller's k (context_pack has no k
# parameter) — kept generous since truncation is budget-driven, not count-driven.
CONTEXT_PACK_CANDIDATE_K = 30

# Snippets are read from the live working tree at query time (see
# `_card_from_hit`), not from a snapshot taken at sync time — the citation
# line number comes from the generation's graph data, so edits made after
# the last sync can silently shift the cited line. Every card therefore
# declares its snippet provenance explicitly, and is flagged stale when the
# source file's mtime is newer than the generation manifest's `created_at`.
SNIPPET_SOURCE_LIVE_TREE = "live_tree"

# The generation manifest's `created_at` format (written by
# `sync/pipeline.py` via `time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())`).
_MANIFEST_CREATED_AT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass
class EvidenceCard:
    citation: str  # "[repo:path:line]"
    repo: str
    label: str
    source_file: str
    line: int | None
    confidence: str
    snippet: str
    score: float
    # Snippet provenance: snippets always come from the live working tree,
    # never from generation data — declared explicitly so clients can tell
    # the snippet's freshness class apart from the citation's (which IS
    # generation data). `snippet_stale=True` marks the degraded case where
    # the source file changed after the generation was built, so the cited
    # line may no longer match the snippet content.
    snippet_source: str = SNIPPET_SOURCE_LIVE_TREE
    snippet_stale: bool = False

    def estimated_tokens(self) -> int:
        # citation + label are always present and cheap; snippet dominates
        # the estimate. +1 keeps a zero-length snippet from costing 0 tokens.
        return max(
            1,
            (len(self.snippet) + len(self.label) + len(self.citation)) // CHARS_PER_TOKEN_ESTIMATE,
        )


@dataclass
class ContextPackResult:
    goal: str
    cards: list = field(default_factory=list)
    truncated: bool = False
    degraded: list = field(default_factory=list)


def _root_for_repo(repo_id: str, registry_entries: list[RegistryEntry]) -> Path | None:
    for entry in registry_entries:
        if entry.repo_id == repo_id:
            return entry.root
    return None


def _parse_manifest_timestamp(raw: object) -> float | None:
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.strptime(raw, _MANIFEST_CREATED_AT_FORMAT)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC).timestamp()


def _generation_created_at_epoch(generation: Generation) -> float | None:
    """Staleness baseline (UTC epoch) for mtime comparison. Prefers the
    manifest's `sync_started_at` (files edited DURING a long sync are newer
    than sync start and correctly flagged), falling back to `created_at`
    (publish stamp) for older generations. Returns `None` when absent or
    malformed — staleness then simply cannot be determined and no card is
    flagged."""
    started = _parse_manifest_timestamp(generation.manifest.get("sync_started_at"))
    if started is not None:
        return started
    return _parse_manifest_timestamp(generation.manifest.get("created_at"))


def _snippet_file_mtime(root: Path | None, source_file: str | None) -> float | None:
    """mtime of the live-tree file a snippet was read from, mirroring
    `build_snippet`'s own path-safety gates (relative only, must resolve
    inside `root`) so a hostile `source_file` can't be stat-probed outside
    the repo root. Returns `None` whenever no snippet could have been read."""
    if root is None or not source_file:
        return None
    if Path(source_file).is_absolute():
        return None
    path = (root / source_file).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    if not path.is_file():
        return None
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _card_from_hit(
    hit: Hit,
    generation: Generation,
    registry_entries: list[RegistryEntry],
    generation_created_at: float | None,
) -> EvidenceCard:
    node = generation.node_by_id.get(hit.node_id, {})
    line = node_line(node)
    root = _root_for_repo(hit.repo, registry_entries)
    snippet = build_snippet(root, hit.source_file, line) if hit.source_file else ""

    snippet_stale = False
    # Staleness is checked whenever a source_file is cited, NOT gated on the
    # snippet being non-empty: an empty snippet (cited line window beyond
    # EOF, e.g. file truncated after the generation was built) is exactly
    # the case where the citation is most likely invalid.
    mtime = _snippet_file_mtime(root, hit.source_file) if hit.source_file else None
    if generation_created_at is not None and mtime is not None and mtime > generation_created_at:
        snippet_stale = True

    line_part = line if line is not None else "?"
    citation = f"[{hit.repo}:{hit.source_file}:{line_part}]"
    return EvidenceCard(
        citation=citation,
        repo=hit.repo,
        label=hit.label,
        source_file=hit.source_file,
        line=line,
        confidence=ranking.CONFIDENCE_EXTRACTED,
        snippet=snippet,
        score=hit.score,
        snippet_source=SNIPPET_SOURCE_LIVE_TREE,
        snippet_stale=snippet_stale,
    )


def build_context_pack(
    goal: str,
    generation: Generation,
    repo_filter: frozenset[str] | None,
    registry_entries: list[RegistryEntry],
    token_budget: int,
    embed_query_fn: EmbedQueryFn,
) -> ContextPackResult:
    ranked = rank(goal, generation, repo_filter, CONTEXT_PACK_CANDIDATE_K, embed_query_fn)
    generation_created_at = _generation_created_at_epoch(generation)

    # Cards are built LAZILY inside the budget loop: each build costs a
    # file open + read (`build_snippet`), so building all candidates
    # eagerly wastes I/O whenever the budget keeps only the first few.
    # Truncation stays whole-card and order-preserving: iterate hits in
    # ranked order, stop at the first card that doesn't fit — a card is
    # never split mid-way, and no card after the first non-fitting one is
    # ever built.
    selected: list[EvidenceCard] = []
    remaining = token_budget
    truncated = False
    for hit in ranked.hits:
        card = _card_from_hit(hit, generation, registry_entries, generation_created_at)
        cost = card.estimated_tokens()
        if cost > remaining:
            truncated = True
            break
        selected.append(card)
        remaining -= cost

    return ContextPackResult(
        goal=goal, cards=selected, truncated=truncated, degraded=ranked.degraded
    )
