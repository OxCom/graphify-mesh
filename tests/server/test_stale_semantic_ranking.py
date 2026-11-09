from __future__ import annotations

from conftest import build_generation, fake_embed_query_fn, key_for, make_node

from graphify_mesh.server import lexical_read, ranking
from graphify_mesh.server.retrieval import rank
from graphify_mesh.server.server import _hit_to_dict


def _stale_generation():
    fresh = make_node("repo.a", "PaymentHandler", "src/fresh.yaml", node_id="fresh")
    stale = make_node("repo.a", "LegacyHandler", "src/stale.yaml", node_id="stale")
    gen = build_generation([fresh, stale])
    gen.stale_keys = frozenset({key_for("repo.a", stale)})
    return gen, key_for("repo.a", fresh), key_for("repo.a", stale)


def test_apply_penalties_stale_is_multiplicative_and_can_be_disabled(monkeypatch):
    degree = {"s": 51}
    paths = {"s": "src/DEPRECATED/x.yaml"}
    stale = frozenset({"s"})
    assert ranking.apply_penalties("s", 1.0, {}, {}, stale_keys=stale) == (
        ranking.STALE_SEMANTIC_PENALTY
    )
    combined = ranking.apply_penalties("s", 1.0, degree, paths, stale_keys=stale)
    assert combined == (
        ranking.STALE_SEMANTIC_PENALTY
        * ranking.HUB_PENALTY_FACTOR
        * ranking.DEPRECATED_PENALTY_FACTOR
    )
    assert ranking.apply_penalties("other", 1.0, {}, {}, stale_keys=stale) == 1.0
    assert ranking.apply_penalties("s", 1.0, {}, {}) == 1.0
    monkeypatch.setattr(ranking, "STALE_SEMANTIC_PENALTY", 1.0)
    assert ranking.apply_penalties("s", 1.0, {}, {}, stale_keys=stale) == 1.0


def test_stale_hit_is_flagged_demoted_and_never_dropped():
    gen, fresh_key, stale_key = _stale_generation()
    result = rank("handler", gen, None, k=10, embed_query_fn=fake_embed_query_fn())
    keys = [h.key for h in result.hits]
    assert fresh_key in keys and stale_key in keys
    by_key = {h.key: h for h in result.hits}
    assert by_key[stale_key].stale is True
    assert by_key[fresh_key].stale is False
    assert by_key[stale_key].score < by_key[fresh_key].score
    payload = _hit_to_dict(by_key[stale_key], gen)
    assert payload["stale"] is True
    assert _hit_to_dict(by_key[fresh_key], gen)["stale"] is False


def test_exact_hit_carries_stale_flag_without_penalty():
    gen, _, stale_key = _stale_generation()
    result = rank("LegacyHandler", gen, None, k=5, embed_query_fn=fake_embed_query_fn())
    top = result.hits[0]
    assert top.key == stale_key
    assert top.match_type == "exact"
    assert top.stale is True


def test_reader_tolerates_absent_and_malformed_stale_list():
    assert lexical_read.stale_semantic_keys({}) == frozenset()
    assert lexical_read.stale_semantic_keys("garbage") == frozenset()
    lexical = {
        "schema_version": 4,
        "documents": [["r", "k0"], ["r", "k1"]],
        "stale_semantic": [1, 99, "x", True],
    }
    assert lexical_read.stale_semantic_keys(lexical) == frozenset({"k1"})
    assert lexical_read.stale_semantic_keys({"schema_version": 4, "stale_semantic": 3}) == (
        frozenset()
    )
