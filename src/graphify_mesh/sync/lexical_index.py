"""WS1.6 / WS5 lexical index: a bundle artifact built ONCE per generation by
the sync pipeline (not per MCP-session), consumed read-only by the `graphify-mesh`
companion MCP server (WS5).

Runs as a new pipeline stage AFTER overlay-resolve and BEFORE validate/publish
(plan WS1 item 6 order: `... -> overlay resolve -> lexical index -> validate
-> atomic publish`). Slots into the previously-unwired gap noted in
pipeline.py's `RunReport.skipped_stages`.

Why this exists instead of reusing graphify's own query-time tokenizer
(serve.py): verified against the installed graphify 0.9.103 package
(`graphify.serve._search_tokens`) — its tokenizer is a single
`re.findall(r"[^\\W_]+", _strip_diacritics(text).lower())`. That is
alphanumeric-only and splits on `_` as well as on every other non-word
character (so `Namespace\\Class::method` and `get_user_name` do split), but it
lowercases BEFORE splitting and never splits camelCase/PascalCase identifiers
into subtokens: `getUserName` is a single token. It also keeps no whole-segment
token alongside the parts, and `\\Foo\\Bar` and `Foo\\Bar` both tokenize to
`["foo","bar"]` — which works for the *backslash* case but not for the
camelCase/`Class::method` pairing case the plan calls out. This module fixes
the camelCase subtoken gap, keeps each whole segment as a token too, and
makes the alias/exact-id path an explicit O(1) table instead of substring
scanning.

Normalization reuse (per plan instruction: "don't reinvent label
normalization from scratch if it already exists"): the base
lowercase+diacritic-stripping step reuses graphify's own exported
`norm_label`-equivalent, `graphify.export._strip_diacritics`, rather than
rewriting Unicode normalization here.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from graphify_mesh.sync.config_keys import (
    PATH_SEPARATOR,
    config_key_paths,
    is_file_node_label,
)
from graphify_mesh.sync.embedding import build_snippet, node_key, node_line

log = logging.getLogger("graphify_mesh.sync.lexical_index")

try:
    from graphify.export import _strip_diacritics as _graphify_strip_diacritics
except ImportError:  # pragma: no cover - defensive: graphify package shape changed upstream

    def _graphify_strip_diacritics(text: str | None) -> str:
        if not text:
            return ""
        return "".join(
            c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
        )


LEXICAL_INDEX_FILENAME = "lexical-index.json"

# Bumped whenever tokenization/alias-extraction rules change in a way that
# would make an old index's postings/aliases inconsistent with a fresh build.
# The graphify-mesh MCP server (WS5) reads this and can refuse a stale-shaped
# index rather than silently misinterpreting it.
TOKENIZER_VERSION = "gm-lex-v1"

# Bumped whenever the ON-DISK SHAPE of `postings`/`alias_exact` entries
# changes (independent of tokenization rules — TOKENIZER_VERSION covers
# term-splitting, this covers container shape).
# v2 replaced per-entry dicts with compact `[repo, key, field]` arrays.
# v3 replaces string arrays with a `documents` table (list index = doc id)
# and int-packed postings `(doc_id << FIELD_PACK_BITS) | field_index`;
# `doc_freq` is no longer stored at all (derived by the reader from
# distinct doc ids per term). Peak-RSS motivation: at ~38K nodes the v2
# string-triple postings still peaked 3.6G against the 4G cgroup cap;
# small-int packing removes per-entry string duplication entirely.
# v4 keeps the v3 container and changes what a document key names: a YAML/JSON
# key node is keyed by its dotted path (`config_keys.config_key_paths`), and the
# file-basename alias is emitted for the file node only. The server derives
# node keys the same way for v4 and with bare labels for v2/v3
# (`uses_qualified_config_keys`), so an older published generation stays
# self-consistent after a deploy.
LEXICAL_SCHEMA_VERSION = 4
# Read-path compatibility set: the server keeps serving a published v2
# generation after this package is deployed (publish can lag, e.g. behind
# shrink-guard). Writer always emits LEXICAL_SCHEMA_VERSION.
SUPPORTED_LEXICAL_SCHEMA_VERSIONS = frozenset({2, 3, 4})
# First schema whose document keys carry config-key dotted paths.
QUALIFIED_CONFIG_KEYS_SCHEMA = 4


def uses_qualified_config_keys(manifest: object, lexical: object) -> bool:
    """Whether a published generation's keys qualify config-key nodes with their
    dotted path. The manifest's `lexical_index_schema_version` decides, then the
    index's own `schema_version`; with neither, keys are bare, as the server's
    lexical reader also treats an unversioned index as v2."""
    for container, field_name in (
        (manifest, "lexical_index_schema_version"),
        (lexical, "schema_version"),
    ):
        version = container.get(field_name) if isinstance(container, dict) else None
        if isinstance(version, int) and not isinstance(version, bool):
            return version >= QUALIFIED_CONFIG_KEYS_SCHEMA
    return False


# v3 field packing: field order is part of the on-disk contract (published
# as `fields`); index into it is stored in the low FIELD_PACK_BITS of each
# packed posting entry.
FIELD_ORDER = ("label", "path", "snippet")
FIELD_PACK_BITS = 2
_FIELD_INDEX = {name: index for index, name in enumerate(FIELD_ORDER)}

# Field boosts (label > path > snippet), per plan WS5 lexical-index bullet.
# Named constants, not magic numbers, per project style rule.
FIELD_BOOST_LABEL = 3.0
FIELD_BOOST_PATH = 1.5
FIELD_BOOST_SNIPPET = 1.0
FIELD_BOOSTS = {
    "label": FIELD_BOOST_LABEL,
    "path": FIELD_BOOST_PATH,
    "snippet": FIELD_BOOST_SNIPPET,
}

# camelCase/PascalCase subtoken boundary: before an uppercase letter that
# follows a lowercase letter/digit, or before the last uppercase letter of a
# run that is followed by a lowercase letter (handles acronym runs like
# "XMLHttpRequest" -> XML, Http, Request).
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
# Splits on any run of non-alphanumeric characters (namespace separators
# `\` and `::`, path separators `/`, dots, dashes, whitespace, etc.) — this is
# the PHP/JS-aware substitute for graphify's plain `\w+` match: segments
# produced here are then further split on camelCase/snake_case boundaries.
# Segment split keeps `_` attached (it is itself a subtoken boundary handled
# separately below) so namespace/path/punctuation separators (`\`, `::`,
# `/`, `.`, `-`, whitespace) delimit segments without prematurely destroying
# snake_case boundaries.
_SEGMENT_SPLIT = re.compile(r"[^A-Za-z0-9_]+")


def _base_normalize(text: str) -> str:
    """Diacritic-stripped, case-PRESERVING. Callers that need the final
    lowercase search/alias form should use `_norm_lower` instead — case must
    survive until after camelCase-boundary detection (see `tokenize_text`)."""
    return _graphify_strip_diacritics(text or "")


def _norm_lower(text: str) -> str:
    return _base_normalize(text).lower()


def normalize_alias_query(text: str) -> str:
    """Public wrapper so callers outside this module (the graphify-mesh MCP
    server's exact-alias bypass, WS5) can normalize a raw query string using
    the IDENTICAL normalization `extract_alias_forms` used to build the
    `alias_exact` table, without duplicating the diacritic-stripping/case
    rules here a second time."""
    return _norm_lower(text)


def _camel_split(segment: str) -> list[str]:
    if not segment:
        return []
    return [p for p in _CAMEL_BOUNDARY.split(segment) if p]


def tokenize_text(text: str | None) -> list[str]:
    """PHP/JS-aware tokenizer (verified insufficiency of graphify's plain
    `[^\\W_]+` regex above). Produces, for each raw non-alnum-delimited segment:
    the segment itself (lowercased) AND its camelCase/snake_case subtokens.
    Namespace segments (`App`, `Service`, `Foo` from `App\\Service\\Foo`) and
    `Class::method` pairs (`Class`, `method` from `Class::method`) fall out
    naturally from splitting on non-alphanumeric runs; leading-backslash FQCN
    variants are handled by `extract_alias_forms` below (this function only
    produces search tokens, not exact-alias keys).

    Case/underscore splitting must happen BEFORE lowercasing — lowercasing
    first would destroy the camelCase boundary signal entirely."""
    if not text:
        return []
    diacritics_stripped = _base_normalize(text)
    tokens: list[str] = []
    for raw_segment in _SEGMENT_SPLIT.split(diacritics_stripped):
        if not raw_segment:
            continue
        tokens.append(raw_segment.lower())
        underscore_parts = [p for p in raw_segment.split("_") if p]
        for part in underscore_parts:
            tokens.append(part.lower())
            camel_parts = _camel_split(part)
            if len(camel_parts) > 1:
                tokens.extend(p.lower() for p in camel_parts)
    # De-dup while preserving nothing order-sensitive (postings sort later);
    # a set is fine and keeps doc-frequency accounting simple (one increment
    # per distinct term per document, not per raw occurrence).
    return sorted(set(tokens))


def extract_alias_forms(
    label: str, source_file: str | None, include_basename: bool = True
) -> set[str]:
    """Exact-alias forms for O(1) lookup: the label as-is (normalized), the
    label with a leading namespace-separator stripped (leading-backslash
    variant per plan bullet), the bare method name of a `Class::method` pair,
    the bare class name of the same pair, and — when `include_basename` — the
    file basename."""
    aliases: set[str] = set()
    if label:
        norm = _norm_lower(label)
        aliases.add(norm)
        aliases.add(norm.lstrip("\\").lstrip("/"))
        if "::" in label:
            cls, _, method = label.partition("::")
            if cls:
                aliases.add(_norm_lower(cls))
            if method:
                aliases.add(_norm_lower(method))
        if "\\" in label:
            aliases.add(_norm_lower(label.split("\\")[-1]))
    if source_file and include_basename:
        aliases.add(_norm_lower(Path(source_file).name))
    aliases.discard("")
    return aliases


MANIFEST_FILENAME = "manifest.json"
ORIGIN_SEMANTIC = "semantic"


def stale_semantic_files(manifest_path: Path | None) -> set[str]:
    """Source files whose LLM-written (`_origin: semantic`) nodes graphify no
    longer refreshes. graphify's `save_manifest` (kind="ast", i.e. `update`)
    stamps the new `ast_hash` and keeps `semantic_hash` only while the content
    hash equals the previous `ast_hash`; a content change therefore leaves
    `semantic_hash == ""` next to a non-empty `ast_hash`. Unchanged rows keep
    their hashes (seeding branch), and files never seen before carry no
    semantic nodes, so neither is flagged. `extract` never sends YAML to the LLM,
    so that state does not heal on its own.

    Rows without an `ast_hash` (semantic-only files such as Go-template Helm
    charts) and legacy non-dict rows are not flagged. A missing or unreadable
    manifest yields the empty set: staleness is a ranking hint, never an error."""
    if manifest_path is None:
        return set()
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    if not isinstance(raw, dict):
        return set()
    return {
        path
        for path, row in raw.items()
        if isinstance(row, dict) and row.get("ast_hash") and row.get("semantic_hash") == ""
    }


@dataclass
class LexicalIndexStats:
    documents: int = 0
    terms: int = 0
    aliases: int = 0
    # Durable keys shared by more than one node (the server keeps one node per
    # key, so the others cannot be addressed) and the nodes in those groups.
    # `config_*` counts only groups holding a qualified YAML/JSON key node.
    key_collisions: int = 0
    key_collision_nodes: int = 0
    config_key_collisions: int = 0
    # Documents flagged `stale_semantic` (see `stale_semantic_files`).
    stale_semantic_documents: int = 0

    def to_dict(self) -> dict:
        return {
            "stale_semantic_documents": self.stale_semantic_documents,
            "documents": self.documents,
            "terms": self.terms,
            "aliases": self.aliases,
            "key_collisions": self.key_collisions,
            "key_collision_nodes": self.key_collision_nodes,
            "config_key_collisions": self.config_key_collisions,
        }


@dataclass
class LexicalIndexResult:
    data: dict = field(default_factory=dict)
    stats: LexicalIndexStats = field(default_factory=LexicalIndexStats)


def build_lexical_index(
    graphs_by_repo: dict[str, dict],
    repo_roots_by_id: dict[str, Path],
    graph_paths_by_repo: dict[str, Path] | None = None,
) -> LexicalIndexResult:
    """Builds the full lexical-index bundle artifact for one generation.

    `graph_paths_by_repo` locates each repo's `graph.json`; graphify's
    `manifest.json` sits next to it and marks semantic nodes whose file changed
    since the LLM pass. Those documents are listed under `stale_semantic`
    (doc ids), an additive top-level key, so the schema version stays 4.

    Deterministic by construction: repos are iterated in sorted order, node
    order within a repo follows the graph file (stable input), doc ids are
    assigned in that iteration order, and every postings/alias entry list is
    sorted ints — nothing depends on dict iteration order or wall-clock time.

    v3 memory layout: `(repo, key)` strings are stored ONCE in the
    `documents` table; postings/alias sets accumulate small ints only, so
    peak RSS no longer scales with (posting count x key string length).
    """
    postings: dict[str, set[int]] = {}
    alias_exact: dict[str, set[int]] = {}
    node_id_index: dict[str, str] = {}
    documents: list[list[str]] = []

    nodes_per_key: dict[str, int] = {}
    config_keys_seen: set[str] = set()
    stale_doc_ids: list[int] = []

    for repo_id in sorted(graphs_by_repo.keys()):
        graph_data = graphs_by_repo[repo_id]
        source_root = repo_roots_by_id.get(repo_id)
        graph_path = (graph_paths_by_repo or {}).get(repo_id)
        stale_files = stale_semantic_files(
            graph_path.parent / MANIFEST_FILENAME if graph_path is not None else None
        )
        paths = config_key_paths(graph_data)
        # Files that have a node naming the file itself: only that node gets
        # the basename alias. A file without one (graphify emitted only its
        # symbols) keeps the alias on every node, so it stays findable by name.
        files_with_file_node = {
            node.get("source_file")
            for node in graph_data.get("nodes", [])
            if isinstance(node, dict)
            and is_file_node_label(node.get("label"), node.get("source_file"))
        }

        for node in graph_data.get("nodes", []):
            if not isinstance(node, dict):
                continue
            qualified = paths.get(node.get("id"))
            key = node_key(repo_id, node, qualified)
            if key is None:
                continue
            nodes_per_key[key] = nodes_per_key.get(key, 0) + 1
            if qualified is not None:
                config_keys_seen.add(key)
            label = qualified or node.get("label") or ""
            source_file = node.get("source_file") or ""
            snippet = build_snippet(source_root, source_file, node_line(node))

            doc_id = len(documents)
            documents.append([repo_id, key])
            doc_base = doc_id << FIELD_PACK_BITS
            if node.get("_origin") == ORIGIN_SEMANTIC and source_file in stale_files:
                stale_doc_ids.append(doc_id)

            # A config key's own name is its label; its ancestors are where it
            # lives, like the file path. Indexing the whole dotted path at label
            # boost gave every nested key of a JSON schema or package.json the
            # label weight of words such as `integration` or `policy`, which
            # reordered code-query results (measured 2026-11-06).
            bare_label = node.get("label") or ""
            ancestors = (
                qualified[: -len(bare_label) - 1]
                if qualified and qualified.endswith(PATH_SEPARATOR + bare_label)
                else ""
            )
            field_tokens = {
                "label": set(tokenize_text(bare_label if qualified else label)),
                "path": set(tokenize_text(source_file)) | set(tokenize_text(ancestors)),
                "snippet": set(tokenize_text(snippet)) if snippet else set(),
            }
            for field_name, terms in field_tokens.items():
                packed = doc_base | _FIELD_INDEX[field_name]
                for term in terms:
                    postings.setdefault(term, set()).add(packed)

            include_basename = source_file not in files_with_file_node or is_file_node_label(
                node.get("label"), source_file
            )
            for alias in extract_alias_forms(label, source_file, include_basename):
                alias_exact.setdefault(alias, set()).add(doc_id)

            raw_id = node.get("id")
            if raw_id:
                node_id_index[f"{repo_id}\x1f{str(raw_id).lower()}"] = key

    # Pop-as-converted (kept from the v2 OOM fix): each term's set is freed
    # as soon as its sorted list is built, so accumulator and output never
    # both peak. Entries are ints, so sorted() needs no key function.
    sorted_postings: dict[str, list[int]] = {}
    for term in sorted(postings.keys()):
        sorted_postings[term] = sorted(postings.pop(term))

    sorted_aliases: dict[str, list[int]] = {}
    for alias in sorted(alias_exact.keys()):
        sorted_aliases[alias] = sorted(alias_exact.pop(alias))

    data = {
        "schema_version": LEXICAL_SCHEMA_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "field_boosts": FIELD_BOOSTS,
        "fields": list(FIELD_ORDER),
        "documents": documents,
        "postings": sorted_postings,
        "alias_exact": sorted_aliases,
        "node_id_index": dict(sorted(node_id_index.items())),
        "document_count": len(documents),
        "stale_semantic": stale_doc_ids,
    }
    colliding = [key for key, count in nodes_per_key.items() if count > 1]
    stats = LexicalIndexStats(
        documents=len(documents),
        terms=len(sorted_postings),
        aliases=len(sorted_aliases),
        key_collisions=len(colliding),
        key_collision_nodes=sum(nodes_per_key[key] for key in colliding),
        config_key_collisions=sum(1 for key in colliding if key in config_keys_seen),
        stale_semantic_documents=len(stale_doc_ids),
    )
    # Measured, never fatal: code nodes can legitimately share
    # (source_file, label), e.g. two overloads graphify labels alike.
    log.info(
        "lexical index: %d durable key(s) shared by %d node(s); %d of them config keys",
        stats.key_collisions,
        stats.key_collision_nodes,
        stats.config_key_collisions,
    )
    return LexicalIndexResult(data=data, stats=stats)
