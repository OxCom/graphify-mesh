"""Version-aware read-path for the `lexical-index.json` bundle.

The sync pipeline writes exactly one schema version (the current
`LEXICAL_SCHEMA_VERSION` in sync/lexical_index.py), but the server must keep
serving whatever the CURRENT published generation contains — publish can lag
behind a deploy (e.g. shrink-guard blocking), so the reader understands the
previous on-disk shape too. All schema branching lives HERE; retrieval.py and
similar.py consume shape-agnostic tuples.

v2 shape: postings `term -> [[repo, key, field], ...]`,
          alias_exact `alias -> [[repo, key], ...]`,
          doc_freq stored under `doc_freq.global`.
v3 shape: `documents` table (list index = doc id), `fields` table
          (list index = field id), postings `term -> [packed_int, ...]`
          with `packed = (doc_id << FIELD_PACK_BITS) | field_index`,
          alias_exact `alias -> [doc_id, ...]`, doc_freq NOT stored
          (derived: distinct doc ids in the term's postings).

Malformed entries are skipped, never raised: a partially-corrupt index
degrades retrieval quality, it must not take the server down.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable

SCHEMA_V2 = 2
SCHEMA_V3 = 3
# Low bits reserved for the field index inside a packed v3 posting entry.
FIELD_PACK_BITS = 2
_FIELD_MASK = (1 << FIELD_PACK_BITS) - 1

_V2_POSTING_LEN = 3
_V2_ALIAS_LEN = 2


def _get_dict(container: object, key: str) -> dict:
    """Fetch `container[key]` and guarantee a dict comes back, no matter how
    malformed `container` or the value at `key` is. Container-level
    hardening: `container` itself may be a non-dict scalar (e.g. the whole
    `lexical` artifact got corrupted into a string), or the value stored at
    `key` may be the wrong shape (a list, an int, ...) — either way this
    degrades to an empty dict rather than raising."""
    if not isinstance(container, dict):
        return {}
    value = container.get(key, {})
    if not isinstance(value, dict):
        return {}
    return value


def _get_list(container: object, key: str) -> list:
    """Same contract as `_get_dict`, but for list-shaped members
    (`documents`, `fields`)."""
    if not isinstance(container, dict):
        return []
    value = container.get(key, [])
    if not isinstance(value, list):
        return []
    return value


def _get_entries(container: dict, key: str) -> list:
    """Fetch `container[key]` and guarantee a list comes back. Used for a
    single term's/alias's posting list, which must be iterable — a
    corrupted artifact may store a scalar (e.g. an int) there instead."""
    value = container.get(key, [])
    if not isinstance(value, list):
        return []
    return value


def _schema(lexical: object) -> int:
    if not isinstance(lexical, dict):
        return SCHEMA_V2
    version = lexical.get("schema_version")
    if isinstance(version, int) and not isinstance(version, bool):
        return version
    return SCHEMA_V2


def _doc_ref(lexical: object, doc_id: int) -> tuple[str, str] | None:
    documents = _get_list(lexical, "documents")
    if not isinstance(doc_id, int) or isinstance(doc_id, bool):
        return None
    if doc_id < 0 or doc_id >= len(documents):
        return None
    entry = documents[doc_id]
    if not isinstance(entry, list) or len(entry) != 2:
        return None
    return (entry[0], entry[1])


def alias_refs(lexical: object, norm_alias: str) -> list[tuple[str, str]]:
    alias_exact = _get_dict(lexical, "alias_exact")
    entries = _get_entries(alias_exact, norm_alias)
    if _schema(lexical) == SCHEMA_V2:
        return [(e[0], e[1]) for e in entries if isinstance(e, list) and len(e) == _V2_ALIAS_LEN]
    refs: list[tuple[str, str]] = []
    for doc_id in entries:
        ref = _doc_ref(lexical, doc_id)
        if ref is None:
            continue
        refs.append(ref)
    return refs


def term_postings(lexical: object, term: str) -> list[tuple[str, str, str]]:
    postings = _get_dict(lexical, "postings")
    entries = _get_entries(postings, term)
    if _schema(lexical) == SCHEMA_V2:
        return [
            (e[0], e[1], e[2]) for e in entries if isinstance(e, list) and len(e) == _V2_POSTING_LEN
        ]
    fields = _get_list(lexical, "fields")
    triples: list[tuple[str, str, str]] = []
    for packed in entries:
        if not isinstance(packed, int) or isinstance(packed, bool):
            continue
        ref = _doc_ref(lexical, packed >> FIELD_PACK_BITS)
        if ref is None:
            continue
        field_index = packed & _FIELD_MASK
        if field_index >= len(fields):
            continue
        triples.append((ref[0], ref[1], fields[field_index]))
    return triples


def term_doc_freq(lexical: object, term: str) -> int:
    postings = _get_dict(lexical, "postings")
    if _schema(lexical) == SCHEMA_V2:
        doc_freq = _get_dict(lexical, "doc_freq")
        global_freq = _get_dict(doc_freq, "global")
        stored = global_freq.get(term)
        if isinstance(stored, int) and not isinstance(stored, bool):
            return stored
        return len(_get_entries(postings, term))
    entries = _get_entries(postings, term)
    distinct = {
        packed >> FIELD_PACK_BITS
        for packed in entries
        if isinstance(packed, int) and not isinstance(packed, bool)
    }
    return len(distinct)


def doc_ids_by_key(lexical: object) -> dict[str, list[int]] | None:
    """v3 only (`None` for v2): key -> doc ids from the `documents` table.
    A list, not one id: two nodes sharing repo, source file and label share a
    key but are separate documents. Malformed entries are skipped exactly as
    `_doc_ref` skips them."""
    if _schema(lexical) == SCHEMA_V2:
        return None
    by_key: dict[str, list[int]] = {}
    for doc_id, entry in enumerate(_get_list(lexical, "documents")):
        if not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[1], str):
            continue
        by_key.setdefault(entry[1], []).append(doc_id)
    return by_key


def _is_packed(entry: object) -> bool:
    return isinstance(entry, int) and not isinstance(entry, bool)


def term_doc_hits(lexical: object, term: str, doc_ids: Iterable[int]) -> set[int]:
    """v3 only: the subset of `doc_ids` that `term` has a posting for, with
    the same entry validity `term_postings` applies (int entry, known field).
    Binary search per doc id over the sorted posting list; a list whose
    entries do not compare (corrupt, mixed types) falls back to a scan."""
    entries = _get_entries(_get_dict(lexical, "postings"), term)
    field_count = len(_get_list(lexical, "fields"))
    wanted = set(doc_ids)
    try:
        return {d for d in wanted if _bisect_hit(entries, d, field_count)}
    except TypeError:
        hits: set[int] = set()
        for packed in entries:
            if _is_packed(packed) and (packed & _FIELD_MASK) < field_count:
                doc_id = packed >> FIELD_PACK_BITS
                if doc_id in wanted:
                    hits.add(doc_id)
        return hits


def _bisect_hit(entries: list, doc_id: int, field_count: int) -> bool:
    index = bisect.bisect_left(entries, doc_id << FIELD_PACK_BITS)
    while index < len(entries):
        packed = entries[index]
        index += 1
        if not _is_packed(packed):
            continue
        if packed >> FIELD_PACK_BITS != doc_id:
            return False
        if (packed & _FIELD_MASK) < field_count:
            return True
    return False


def document_count(lexical: object) -> int:
    if not isinstance(lexical, dict):
        return 0
    count = lexical.get("document_count")
    if isinstance(count, int) and not isinstance(count, bool):
        return count
    return len(_get_list(lexical, "documents"))
