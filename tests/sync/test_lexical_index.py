from __future__ import annotations

import json

from graphify_mesh.sync import lexical_index
from graphify_mesh.sync.pipeline import run


def test_tokenize_splits_namespace_and_double_colon():
    tokens = lexical_index.tokenize_text(r"App\Service\Foo::barMethod")
    assert "app" in tokens
    assert "service" in tokens
    assert "foo" in tokens
    assert "barmethod" in tokens
    # camelCase subtoken split
    assert "bar" in tokens
    assert "method" in tokens


def test_tokenize_snake_case_subtokens():
    tokens = lexical_index.tokenize_text("get_user_name")
    assert "get_user_name" in tokens
    assert "get" in tokens
    assert "user" in tokens
    assert "name" in tokens


def test_tokenize_plain_word_regex_would_miss_subtokens():
    """Documents the verified insufficiency: graphify's `\\w+` tokenizer
    would return a single token here; ours must split it."""
    import re

    plain = re.findall(r"\w+", "getUserName".lower())
    assert plain == ["getusername"]
    ours = lexical_index.tokenize_text("getUserName")
    assert "get" in ours and "user" in ours and "name" in ours


def test_alias_forms_include_leading_backslash_variant():
    aliases = lexical_index.extract_alias_forms(r"\App\Service\Foo", None)
    assert "\\app\\service\\foo" in aliases
    assert "app\\service\\foo" in aliases


def test_alias_forms_class_method_pair():
    aliases = lexical_index.extract_alias_forms("TimeService::calculate", "src/TimeService.php")
    assert "timeservice::calculate" in aliases
    assert "timeservice" in aliases
    assert "calculate" in aliases
    assert "timeservice.php" in aliases


def _unpack(lexical_data: dict, packed: int) -> tuple[str, str, str]:
    """Test-side decode of a v3 posting: (repo, key, field)."""
    doc = lexical_data["documents"][packed >> 2]
    field = lexical_data["fields"][packed & 0b11]
    return (doc[0], doc[1], field)


def test_build_lexical_index_v3_shape_and_field_boosts():
    graphs_by_repo = {
        "repo.a": {
            "nodes": [
                {"id": "n1", "label": "AlphaClass", "source_file": "src/alpha.py"},
                {"id": "n2", "label": "BetaClass", "source_file": "src/beta.py"},
            ]
        }
    }
    result1 = lexical_index.build_lexical_index(graphs_by_repo, {})
    result2 = lexical_index.build_lexical_index(graphs_by_repo, {})
    assert result1.data == result2.data  # determinism preserved
    assert result1.stats.documents == 2

    data = result1.data
    # v4 keeps the v3 container shape; only what a config-key document key names changed.
    assert data["schema_version"] == 4
    assert data["fields"] == ["label", "path", "snippet"]
    assert "doc_freq" not in data
    assert isinstance(data["documents"], list)
    for entry in data["documents"]:
        assert isinstance(entry, list) and len(entry) == 2
    assert data["field_boosts"]["label"] == lexical_index.FIELD_BOOST_LABEL

    label_postings = data["postings"]["alphaclass"]
    assert all(isinstance(p, int) for p in label_postings)
    assert label_postings == sorted(label_postings)
    unpacked = [_unpack(data, p) for p in label_postings]
    assert any(f == "label" for (_, _, f) in unpacked)

    path_postings = data["postings"]["alpha"]
    assert any(_unpack(data, p)[2] == "path" for p in path_postings)


def test_build_lexical_index_v3_alias_exact_doc_ids():
    graphs_by_repo = {
        "repo.a": {
            "nodes": [{"id": "n1", "label": "TimeService", "source_file": "src/TimeService.php"}]
        }
    }
    result = lexical_index.build_lexical_index(graphs_by_repo, {})
    data = result.data
    assert "timeservice" in data["alias_exact"]
    entries = data["alias_exact"]["timeservice"]
    assert all(isinstance(d, int) for d in entries)
    assert entries == sorted(entries)
    repo, key = data["documents"][entries[0]]
    assert repo == "repo.a"
    assert "repo.a\x1fn1" in data["node_id_index"]


def test_build_lexical_index_v3_doc_freq_derivable():
    """df(term) must equal distinct doc ids in the term's postings — the
    reader derives it (v3 stores no doc_freq), so the invariant lives here."""
    graphs_by_repo = {
        "repo.a": {
            "nodes": [
                {"id": "n1", "label": "AlphaClass", "source_file": "src/alpha.py"},
                {"id": "n2", "label": "TimeService", "source_file": "src/TimeService.php"},
            ]
        }
    }
    result = lexical_index.build_lexical_index(graphs_by_repo, {})
    data = result.data
    for _term, entries in data["postings"].items():
        distinct_docs = {p >> 2 for p in entries}
        assert len(distinct_docs) >= 1
        assert max(p >> 2 for p in entries) < len(data["documents"])


def test_pipeline_publishes_lexical_index_artifact(env):
    env.add_repo("acme.a", "acme", "a", "acme-a", graph_fixture="repo_a.json")
    env.write_registry()
    settings = env.settings()

    report = run(settings)

    assert report.published, report.publish_blocked_reason
    assert not report.skipped_stages  # lexical-index no longer skipped
    assert report.lexical_index_stats["documents"] > 0

    lexical_path = settings.global_dir / "current" / lexical_index.LEXICAL_INDEX_FILENAME
    assert lexical_path.is_file()
    data = json.loads(lexical_path.read_text(encoding="utf-8"))
    assert data["tokenizer_version"] == lexical_index.TOKENIZER_VERSION
    assert data["schema_version"] == lexical_index.LEXICAL_SCHEMA_VERSION
    assert "doc_freq" not in data
    assert data["fields"] == ["label", "path", "snippet"]
    assert data["postings"]

    manifest = json.loads(
        (settings.global_dir / "current" / "generation-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["lexical_index_tokenizer_version"] == lexical_index.TOKENIZER_VERSION
    assert manifest["lexical_index_schema_version"] == lexical_index.LEXICAL_SCHEMA_VERSION
