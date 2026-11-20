"""Pins the source of the code that turns one repo's graph into a cached lexical
partial, its payload round trip, the assembly merge and the cache key. With the
stage cache on, an unchanged repo's partial is served from disk, so a change
here is invisible until LEXICAL_PARTIAL_RECIPE is bumped."""

from __future__ import annotations

import hashlib
import inspect

from graphify_mesh.sync import (
    config_keys,
    embedding,
    lexical_index,
    lexical_partial_cache,
    overlay,
    source_cache,
)
from graphify_mesh.sync.overlay_refs import LogicalRef

GUARDED = (
    # Per-repo partial: documents, postings, aliases, node ids, stale flags.
    lexical_index.build_repo_partial,
    lexical_index.RepoLexicalPartial,
    lexical_index._is_count,
    lexical_index._is_sorted_ids,
    lexical_index._sorted_id_table,
    lexical_index._merge_partial,
    lexical_index.stale_semantic_files,
    # Tokenization and alias forms.
    lexical_index.tokenize_text,
    lexical_index._camel_split,
    lexical_index._base_normalize,
    lexical_index._norm_lower,
    lexical_index.extract_alias_forms,
    # Durable keys, config-key dotted paths and snippets.
    embedding.node_key,
    LogicalRef.to_key,
    embedding.node_line,
    embedding.build_snippet,
    source_cache.get_source_lines,
    source_cache._read_capped_lines,
    config_keys,
    # The key and the stat lines it hashes.
    lexical_partial_cache.partial_key,
    lexical_index.read_manifest,
    lexical_index.ManifestRead,
    lexical_index._stale_semantic_files_from_bytes,
    lexical_partial_cache._graphify_version,
    embedding._source_file_stat_lines,
    # The single graph.json read whose sha256 the key takes.
    overlay.load_graphs_and_sha256,
)
GUARDED_VALUES = (
    lexical_index.LEXICAL_PARTIAL_RECIPE,
    lexical_index.FIELD_ORDER,
    lexical_index.FIELD_PACK_BITS,
    lexical_index._CAMEL_BOUNDARY.pattern,
    lexical_index._SEGMENT_SPLIT.pattern,
    lexical_index.ORIGIN_SEMANTIC,
    lexical_index.MANIFEST_FILENAME,
    embedding.SNIPPET_WINDOW_LINES,
    embedding.SNIPPET_MAX_CHARS,
    embedding.SNIPPET_READ_LINE_CAP,
    embedding._SOURCE_LOCATION_LINE.pattern,
    source_cache.SOURCE_CACHE_MAX_LINE_CHARS,
    source_cache.SOURCE_CACHE_MAX_FILE_CHARS,
    source_cache.SOURCE_CACHE_READ_CHUNK_CHARS,
)
PINNED_SOURCE_SHA256 = "a4c472571c0a619ab14b3ea7f38dbf0e04ee2c9f6347d91733e4d9f300e228b4"


def _source_sha256() -> str:
    h = hashlib.sha256()
    for obj in GUARDED:
        h.update(inspect.getsource(obj).encode("utf-8"))
        h.update(b"\0")
    for value in GUARDED_VALUES:
        h.update(repr(value).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def test_lexical_partial_source_is_pinned():
    actual = _source_sha256()
    assert actual == PINNED_SOURCE_SHA256, (
        "code that shapes cached lexical partials or their key changed. "
        "A comment- or docstring-only change: update PINNED_SOURCE_SHA256 only. "
        "A semantic change: bump LEXICAL_PARTIAL_RECIPE in lexical_index.py "
        "(invalidates every cached partial) and update PINNED_SOURCE_SHA256. "
        "Shared with tests/sync/test_embedding_recipe_guard.py: node_key, "
        "LogicalRef.to_key, node_line, build_snippet, source_cache, config_keys, "
        "_source_file_stat_lines and load_graphs_and_sha256 also feed the embedding "
        "inputs_digest and shards; bump SHARD_FORMAT_VERSION or RECIPE_STAMP_VERSION "
        "too and re-pin that guard. "
        f"New value: {actual!r}"
    )
