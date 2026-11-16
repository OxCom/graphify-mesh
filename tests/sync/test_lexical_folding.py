"""The lexical index stores tokens as written: no stemming or folding."""

from __future__ import annotations

from graphify_mesh.sync import lexical_index


def test_index_stores_tokens_as_written():
    nodes = [
        {"id": "a", "label": "TeamsListing", "source_file": "src/Teams.php"},
        {"id": "b", "label": "Owner", "source_file": "src/Owner.php"},
    ]
    data = lexical_index.build_lexical_index({"repo.a": {"nodes": nodes}}, {}).data

    assert data["tokenizer_version"] == lexical_index.TOKENIZER_VERSION == "gm-lex-v1"
    assert "team" not in data["postings"]
    assert data["fields"] == ["label", "path", "snippet"]
    assert "field_lengths" not in data
    assert "avg_field_lengths" not in data
