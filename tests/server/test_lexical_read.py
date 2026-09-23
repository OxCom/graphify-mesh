"""Reader-side contract for lexical-index bundles: one module understands
both on-disk schema versions, so retrieval/similar never branch on shape."""

from graphify_mesh.server import lexical_read
from graphify_mesh.sync.lexical_index import build_lexical_index


def _v2_lexical() -> dict:
    return {
        "schema_version": 2,
        "field_boosts": {"label": 3.0, "path": 1.5, "snippet": 1.0},
        "postings": {
            "alphaclass": [
                ["repoA", "keyA1", "label"],
                ["repoA", "keyA1", "path"],
                ["repoB", "keyB1", "label"],
            ],
        },
        "doc_freq": {"global": {"alphaclass": 2}, "per_repo": {}},
        "alias_exact": {"alphaclass": [["repoA", "keyA1"], ["repoB", "keyB1"]]},
        "document_count": 2,
    }


def _v3_lexical() -> dict:
    # documents: index IS the doc id. fields: index IS the field id.
    # packed entry = (doc_id << 2) | field_index
    return {
        "schema_version": 3,
        "field_boosts": {"label": 3.0, "path": 1.5, "snippet": 1.0},
        "fields": ["label", "path", "snippet"],
        "documents": [["repoA", "keyA1"], ["repoB", "keyB1"]],
        "postings": {
            # doc 0 label -> 0, doc 0 path -> 1, doc 1 label -> 4
            "alphaclass": [0, 1, 4],
        },
        "alias_exact": {"alphaclass": [0, 1]},
        "document_count": 2,
    }


def test_alias_refs_v2():
    refs = lexical_read.alias_refs(_v2_lexical(), "alphaclass")
    assert refs == [("repoA", "keyA1"), ("repoB", "keyB1")]


def test_alias_refs_v3():
    refs = lexical_read.alias_refs(_v3_lexical(), "alphaclass")
    assert refs == [("repoA", "keyA1"), ("repoB", "keyB1")]


def test_term_postings_v2():
    triples = lexical_read.term_postings(_v2_lexical(), "alphaclass")
    assert ("repoA", "keyA1", "label") in triples
    assert ("repoA", "keyA1", "path") in triples
    assert ("repoB", "keyB1", "label") in triples
    assert len(triples) == 3


def test_term_postings_v3():
    triples = lexical_read.term_postings(_v3_lexical(), "alphaclass")
    assert ("repoA", "keyA1", "label") in triples
    assert ("repoA", "keyA1", "path") in triples
    assert ("repoB", "keyB1", "label") in triples
    assert len(triples) == 3


def test_term_doc_freq_v2_uses_stored_table():
    assert lexical_read.term_doc_freq(_v2_lexical(), "alphaclass") == 2


def test_term_doc_freq_v3_derived_from_distinct_docs():
    # 3 postings entries but only 2 distinct documents.
    assert lexical_read.term_doc_freq(_v3_lexical(), "alphaclass") == 2


def test_missing_term_and_alias_return_empty():
    for lex in (_v2_lexical(), _v3_lexical()):
        assert lexical_read.alias_refs(lex, "nope") == []
        assert lexical_read.term_postings(lex, "nope") == []
        assert lexical_read.term_doc_freq(lex, "nope") == 0


def test_malformed_entries_skipped_never_crash():
    lex = _v3_lexical()
    lex["postings"]["alphaclass"] = [0, "junk", 999999, 4]  # 999999 -> doc id out of range
    triples = lexical_read.term_postings(lex, "alphaclass")
    assert ("repoA", "keyA1", "label") in triples
    assert ("repoB", "keyB1", "label") in triples
    assert len(triples) == 2


# --- container-level hardening: `lexical` and its members may themselves be
# the wrong shape (non-dict scalar, list-not-dict, etc), not just individual
# postings entries. Every public function must degrade to empty/zero rather
# than raise (module docstring: "a partially-corrupt index degrades
# retrieval quality, it must not take the server down"). --------------------


def test_lexical_itself_not_a_dict():
    for lexical in ("not-a-dict", 42, None, ["nope"]):
        assert lexical_read.alias_refs(lexical, "alphaclass") == []
        assert lexical_read.term_postings(lexical, "alphaclass") == []
        assert lexical_read.term_doc_freq(lexical, "alphaclass") == 0
        assert lexical_read.document_count(lexical) == 0


def test_postings_container_is_list_not_dict():
    lex = _v3_lexical()
    lex["postings"] = ["not", "a", "dict"]
    assert lexical_read.term_postings(lex, "alphaclass") == []
    assert lexical_read.term_doc_freq(lex, "alphaclass") == 0


def test_alias_exact_container_is_wrong_shape():
    lex = _v3_lexical()
    lex["alias_exact"] = "not-a-dict"
    assert lexical_read.alias_refs(lex, "alphaclass") == []


def test_term_value_is_scalar_not_iterable():
    lex = _v3_lexical()
    lex["postings"]["alphaclass"] = 12345  # scalar instead of a list
    assert lexical_read.term_postings(lex, "alphaclass") == []
    assert lexical_read.term_doc_freq(lex, "alphaclass") == 0

    v2 = _v2_lexical()
    v2["postings"]["alphaclass"] = 12345
    assert lexical_read.term_postings(v2, "alphaclass") == []


def test_documents_container_is_dict_not_list():
    lex = _v3_lexical()
    lex["documents"] = {"0": ["repoA", "keyA1"]}
    assert lexical_read.term_postings(lex, "alphaclass") == []
    # `document_count` is not stored, so this exercises the len(documents)
    # fallback directly rather than the trusted stored-field path.
    del lex["document_count"]
    assert lexical_read.document_count(lex) == 0


def test_fields_container_is_string_not_list():
    lex = _v3_lexical()
    lex["fields"] = "label"
    assert lexical_read.term_postings(lex, "alphaclass") == []


def test_v2_doc_freq_container_malformed():
    v2 = _v2_lexical()
    v2["doc_freq"] = "not-a-dict"
    # Falls back to len(postings) rather than raising on the malformed table.
    assert lexical_read.term_doc_freq(v2, "alphaclass") == 3
    v2b = _v2_lexical()
    v2b["doc_freq"]["global"] = ["not", "a", "dict"]
    assert lexical_read.term_doc_freq(v2b, "alphaclass") == 3


# --- writer/reader contract: build via the REAL writer, decode via the
# reader. Pins the packing scheme so writer/reader can never coordinate-drift
# silently — assertions are only on decoded (repo, key, field) tuples, never
# on either side's internal packing constants. -------------------------------


def test_writer_reader_packing_round_trip():
    graphs_by_repo = {
        "repo.a": {
            "nodes": [
                {"id": "n1", "label": "AlphaClass", "source_file": "src/alpha.py"},
            ]
        },
        "repo.b": {
            "nodes": [
                {"id": "n2", "label": "BetaClass", "source_file": "src/beta.py"},
            ]
        },
    }
    lexical = build_lexical_index(graphs_by_repo, {}).data

    alias_hits = lexical_read.alias_refs(lexical, "alphaclass")
    assert len(alias_hits) == 1
    assert alias_hits[0][0] == "repo.a"

    triples = lexical_read.term_postings(lexical, "alphaclass")
    assert (alias_hits[0][0], alias_hits[0][1], "label") in triples

    path_triples = lexical_read.term_postings(lexical, "alpha")
    assert any(field == "path" for (_repo, _key, field) in path_triples)
    assert any(repo == "repo.a" for (repo, _key, _field) in path_triples)

    assert lexical_read.document_count(lexical) == 2
    assert lexical_read.term_doc_freq(lexical, "alphaclass") == 1

    beta_hits = lexical_read.alias_refs(lexical, "betaclass")
    assert len(beta_hits) == 1
    beta_triples = lexical_read.term_postings(lexical, "betaclass")
    assert (beta_hits[0][0], beta_hits[0][1], "label") in beta_triples
    assert beta_hits[0][0] == "repo.b"

    v2 = _v2_lexical()
    v2["postings"]["alphaclass"] = [["repoA", "keyA1", "label"], "junk", ["short"]]
    assert lexical_read.term_postings(v2, "alphaclass") == [("repoA", "keyA1", "label")]


def test_document_count():
    assert lexical_read.document_count(_v2_lexical()) == 2
    assert lexical_read.document_count(_v3_lexical()) == 2


def _hits_lexical(postings: list) -> dict:
    return {
        "schema_version": 3,
        "fields": ["label", "path", "snippet"],
        "documents": [["r", f"k{i}"] for i in range(6)],
        "postings": {"t": postings},
        "document_count": 6,
    }


def test_doc_ids_by_key_v3_and_none_for_v2():
    lexical = _v3_lexical()
    lexical["documents"].append(["repoA", "keyA1"])  # same key, second document
    lexical["documents"].append("corrupt")
    assert lexical_read.doc_ids_by_key(lexical) == {"keyA1": [0, 2], "keyB1": [1]}
    assert lexical_read.doc_ids_by_key(_v2_lexical()) is None


def test_term_doc_hits_hit_miss_first_and_last_doc():
    # docs 0, 2, 5 carry the term; doc 5 is the last entry of the list
    lexical = _hits_lexical([0, 1, 9, 21])
    assert lexical_read.term_doc_hits(lexical, "t", [0, 1, 2, 3, 4, 5]) == {0, 2, 5}
    assert lexical_read.term_doc_hits(lexical, "t", [0]) == {0}
    assert lexical_read.term_doc_hits(lexical, "t", [5]) == {5}
    assert lexical_read.term_doc_hits(lexical, "t", [1, 3, 4, 99]) == set()
    assert lexical_read.term_doc_hits(lexical, "missing", [0, 5]) == set()
    assert lexical_read.term_doc_hits(lexical, "t", []) == set()


def test_term_doc_hits_matches_term_postings_on_real_index():
    lexical = build_lexical_index(
        {
            "repoA": {
                "nodes": [
                    {"id": f"n{i}", "label": f"Alpha{i}", "source_file": "a.py"} for i in range(20)
                ]
            }
        },
        {},
    ).data
    by_key = lexical_read.doc_ids_by_key(lexical)
    assert by_key is not None
    for term in ("alpha3", "a", "py", "nope"):
        expected = {
            by_key[key][0] for _repo, key, _field in lexical_read.term_postings(lexical, term)
        }
        assert lexical_read.term_doc_hits(lexical, term, range(20)) == expected


def test_term_doc_hits_corrupt_entries_treated_as_absent():
    # non-int / bool entries are skipped; unknown field index is not a hit
    lexical = _hits_lexical([0, True, 4.5, 8, 15, 20])
    assert lexical_read.term_doc_hits(lexical, "t", [0, 1, 2, 3, 5]) == {0, 2, 5}
    # entries that do not compare with ints force the linear-scan fallback
    lexical = _hits_lexical(["x", 21, None, 1, [4]])
    assert lexical_read.term_doc_hits(lexical, "t", [0, 1, 5]) == {0, 5}
    # a scalar where a posting list belongs
    lexical = _hits_lexical([])
    lexical["postings"]["t"] = 7
    assert lexical_read.term_doc_hits(lexical, "t", [0, 1]) == set()
    assert lexical_read.term_doc_hits("garbage", "t", [0]) == set()
