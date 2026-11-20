"""Pins the source of the code that derives embedding inputs and their hashes,
writes shard files and reads them back. With the stage cache on, an unchanged
repo's shard is reused without re-hashing its inputs and hard-linked byte for
byte, so a change here is invisible until a version bump."""

from __future__ import annotations

import hashlib
import inspect

from graphify_mesh.sync import config_keys, embedding, overlay, source_cache
from graphify_mesh.sync.overlay_refs import LogicalRef
from graphify_mesh.sync.vectors import RepoVectors

GUARDED = (
    # Embedding input text and its content hash.
    embedding._iter_embeddable_nodes,
    embedding.node_key,
    LogicalRef.to_key,
    embedding.node_line,
    embedding.build_snippet,
    source_cache.get_source_lines,
    source_cache._read_capped_lines,
    embedding.is_trivial_node,
    embedding.build_embedding_input,
    embedding._content_hash,
    # Config-key dotted paths (qualified labels) — the whole module derives them.
    config_keys,
    # Shard writer and the reader round trip the fast path's byte equality needs.
    embedding._write_shard,
    embedding.shard_matrix_sha256,
    embedding._read_v2_shard,
    embedding._validate_v2_shard,
    embedding._meta_recipe_stamp,
    embedding._meta_str,
    RepoVectors.from_rows,
    RepoVectors.from_mapping,
    # The fast-path decision: the digest, what it hashes, and the slow check it skips.
    embedding.inputs_digest,
    embedding._source_file_stat_lines,
    overlay.load_graphs_and_sha256,
    embedding._shard_content_current,
)
GUARDED_VALUES = (
    source_cache.SOURCE_CACHE_MAX_LINE_CHARS,
    source_cache.SOURCE_CACHE_MAX_FILE_CHARS,
    source_cache.SOURCE_CACHE_READ_CHUNK_CHARS,
    embedding.SNIPPET_READ_LINE_CAP,
    embedding._SOURCE_LOCATION_LINE.pattern,
)
PINNED_SOURCE_SHA256 = "f0222537392ae64946c547be2dea43178456db9d744b164e216bfc4d4d628e1b"


def _source_sha256() -> str:
    h = hashlib.sha256()
    for obj in GUARDED:
        h.update(inspect.getsource(obj).encode("utf-8"))
        h.update(b"\0")
    for value in GUARDED_VALUES:
        h.update(repr(value).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def test_embedding_input_and_shard_io_source_is_pinned():
    actual = _source_sha256()
    assert actual == PINNED_SOURCE_SHA256, (
        "code that shapes embedding inputs or shard bytes changed. "
        "A comment- or docstring-only change: update PINNED_SOURCE_SHA256 only. "
        "A semantic change: bump RECIPE_STAMP_VERSION (re-embeds every repo) or "
        "SHARD_FORMAT_VERSION (then also update SUPPORTED_SHARD_FORMATS and the "
        "server reader), and update PINNED_SOURCE_SHA256. "
        "A change to inputs_digest or what it hashes invalidates stored digests: "
        "bump SHARD_FORMAT_VERSION, which inputs_digest folds in. "
        "Shared with tests/sync/test_lexical_recipe_guard.py: node_key, "
        "LogicalRef.to_key, node_line, build_snippet, source_cache, config_keys, "
        "_source_file_stat_lines and load_graphs_and_sha256 also feed "
        "LEXICAL_PARTIAL_RECIPE; bump it too and re-pin that guard. "
        f"New value: {actual!r}"
    )
