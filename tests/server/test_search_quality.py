"""Search-quality regressions on tiny synthetic generations (no real project
data): test demotion before the candidate cut, the structural seed, older
bundles with a field-length table, MMR prefix stability, the cross-repo
anchor file rule and the query instruction."""

from __future__ import annotations

import json

import pytest
from conftest import build_generation, fake_embed_query_fn, key_for, make_link, make_node
from test_anchors import _fillers
from test_store import _write_generation

from graphify_mesh.server import anchors, embed_query, ranking, retrieval
from graphify_mesh.server.config import ServerConfig
from graphify_mesh.server.retrieval import rank
from graphify_mesh.server.store import GenerationStore
from graphify_mesh.sync.vectors import RepoVectors


def _keys(result) -> list[str]:
    return [hit.key for hit in result.hits]


# --- test demotion -------------------------------------------------------


def test_demoted_test_hit_stays_reachable_when_source_hits_fill_k():
    source = [
        make_node("repo.a", f".overtimeCalculation{i}()", f"src/Overtime{i}.php", node_id=f"s{i}")
        for i in range(60)
    ]
    test = make_node(
        "repo.a", ".testOvertimeCalculationRounding()", "tests/php/OvertimeTest.php", node_id="t"
    )
    gen = build_generation(source + [test])

    result = rank(
        "overtime calculation rounding", gen, None, k=50, embed_query_fn=fake_embed_query_fn()
    )

    keys = _keys(result)
    assert len(keys) == 50
    assert key_for("repo.a", test) in keys
    # Still demoted: it takes the first reserved slot, behind the source hits above it.
    assert keys.index(key_for("repo.a", test)) == retrieval.TEST_RESULT_SLOT_EVERY - 1


def test_source_node_behind_many_matching_tests_survives_the_candidate_cut():
    tests = [
        make_node(
            "repo.a", f".testPaymentInvoice{i}()", f"tests/Payment{i}Test.php", node_id=f"t{i}"
        )
        for i in range(200)
    ]
    target = make_node("repo.a", ".payment()", "src/Billing.php", node_id="target")
    gen = build_generation(tests + [target])
    is_test = retrieval._test_key_fn(gen)

    lexical = retrieval.lexical_candidates(
        "payment invoice", gen.lexical, None, is_test=is_test, test_depth=5
    )
    result = rank("payment invoice", gen, None, k=10, embed_query_fn=fake_embed_query_fn())

    assert lexical[0] == key_for("repo.a", target)
    assert len(lexical) == 6
    assert _keys(result)[0] == key_for("repo.a", target)


def test_vector_source_node_behind_many_closer_tests_survives_the_cut():
    tests = [
        make_node("repo.a", f".testThing{i}()", f"tests/Thing{i}Test.php", node_id=f"t{i}")
        for i in range(80)
    ]
    target = make_node("repo.a", "Thing", "src/Thing.php", node_id="target")
    vectors = {key_for("repo.a", node): [1.0, 0.0] for node in tests}
    vectors[key_for("repo.a", target)] = [0.9, 0.1]
    embeddings = {"repo.a": RepoVectors.from_mapping(vectors)}

    keys, degraded = retrieval.vector_candidates(
        "q",
        embeddings,
        None,
        fake_embed_query_fn({"q": [1.0, 0.0]}),
        depth=50,
        demote_tests=True,
        test_depth=3,
    )

    assert not degraded
    assert keys[0] == key_for("repo.a", target)
    assert len(keys) == 4


@pytest.mark.parametrize("word", ["phpunit", "jest", "cypress", "e2e", "mock"])
def test_query_naming_a_test_tool_is_not_demoted(word):
    assert ranking.query_mentions_tests(f"order {word} setup")


@pytest.mark.parametrize(
    "query",
    [
        "majestic theme loader",
        "latest invoice export",
        "contest entry form",
        "attestation upload",
    ],
)
def test_a_word_that_only_contains_a_test_term_keeps_the_demotion(query):
    assert not ranking.query_mentions_tests(query)


@pytest.mark.parametrize(
    "query",
    [
        "UserServiceTest",
        "OvertimeTests rounding",
        "test_overtime_rounding",
        "tests/Handler/Planner",
        "Order.spec.ts",
        "mockery expectations",
        "createMock repository",
        "jest.config",
        "pytest fixtures",
        "unit testing helpers",
        "TestCase setup",
    ],
)
def test_camel_case_snake_case_and_path_forms_ask_for_tests(query):
    assert ranking.query_mentions_tests(query)


def _overtime_generation(extra: list[dict] | None = None):
    source = [
        make_node("repo.a", f".overtimeCalculation{i}()", f"src/Overtime{i}.php", node_id=f"s{i}")
        for i in range(20)
    ]
    tests = [
        make_node(
            "repo.a",
            f".testOvertimeCalculation{i}()",
            f"tests/php/Overtime{i}Test.php",
            node_id=f"t{i}",
        )
        for i in range(2)
    ]
    return {key_for("repo.a", t) for t in tests}, build_generation(source + tests + (extra or []))


def test_reserved_test_slots_sit_at_fixed_positions_for_every_k():
    test_keys, gen = _overtime_generation()
    every = retrieval.TEST_RESULT_SLOT_EVERY

    top10 = _keys(
        rank("overtime calculation", gen, None, k=10, embed_query_fn=fake_embed_query_fn())
    )
    top20 = _keys(
        rank("overtime calculation", gen, None, k=20, embed_query_fn=fake_embed_query_fn())
    )

    assert top20[:10] == top10
    assert [i for i, key in enumerate(top20) if key in test_keys] == [every - 1, 2 * every - 1]


def test_an_exact_alias_hit_does_not_move_the_reserved_test_slot():
    alias = make_node("repo.a", "OvertimeCalculation", "src/OvertimeCalculation.php", node_id="x")
    test_keys, gen = _overtime_generation([alias])

    keys = _keys(rank("OvertimeCalculation", gen, None, k=10, embed_query_fn=fake_embed_query_fn()))

    assert keys[0] == key_for("repo.a", alias)
    assert keys[retrieval.TEST_RESULT_SLOT_EVERY - 1] in test_keys


def test_a_reserved_slot_needs_a_test_close_enough_to_the_source_hit_it_displaces():
    selected = [f"s{i}" for i in range(10)]
    scores = dict.fromkeys(selected, 0.03)
    floor = 0.03 * retrieval.TEST_SLOT_MIN_SCORE_RATIO

    def is_test(key: str) -> bool:
        return key.startswith("t")

    weak = list((scores | {"t": floor * 0.99}).items())
    close = list((scores | {"t": floor}).items())

    assert retrieval._reserve_test_slots(selected, weak, is_test, [9], set()) == selected
    assert retrieval._reserve_test_slots(selected, close, is_test, [9], set()) == selected[:9] + [
        "t"
    ]


# --- structural seed -----------------------------------------------------


def test_structural_retriever_is_seeded_by_a_vector_only_match():
    vector_only = make_node("repo.a", "Zebra", "src/Zebra.php", node_id="v")
    neighbour = make_node("repo.a", "Quokka", "src/Quokka.php", node_id="n")
    others = [make_node("repo.a", f"Other{i}", f"src/o{i}.php", node_id=f"o{i}") for i in range(5)]
    vectors = {key_for("repo.a", vector_only): [1.0, 0.0]}
    vectors.update({key_for("repo.a", node): [0.0, 1.0] for node in others})
    gen = build_generation(
        [vector_only, neighbour] + others,
        links=[make_link("v", "n")],
        embeddings={"repo.a": RepoVectors.from_mapping(vectors)},
    )

    result = rank(
        "striped horse",
        gen,
        None,
        k=10,
        embed_query_fn=fake_embed_query_fn({"striped horse": [1.0, 0.0]}),
    )

    assert key_for("repo.a", neighbour) in _keys(result)


def test_structural_retriever_is_seeded_by_a_lexical_only_match():
    # The seed fuses both retrievers: vector hits must not crowd out a node
    # only the lexical side found.
    lexical_only = make_node("repo.a", "Zebra", "src/Zebra.php", node_id="z")
    neighbour = make_node("repo.a", "Quokka", "src/Quokka.php", node_id="n")
    others = [make_node("repo.a", f"Other{i}", f"src/o{i}.php", node_id=f"o{i}") for i in range(12)]
    vectors = {key_for("repo.a", node): [1.0, 0.0] for node in others}
    gen = build_generation(
        [lexical_only, neighbour] + others + _fillers(),
        links=[make_link("z", "n")],
        embeddings={"repo.a": RepoVectors.from_mapping(vectors)},
    )

    # Two words, so the label is no exact alias hit (those seed on their own).
    query = "zebra crossing"
    result = rank(query, gen, None, k=20, embed_query_fn=fake_embed_query_fn({query: [1.0, 0.0]}))

    assert key_for("repo.a", neighbour) in _keys(result)


# --- older bundles ------------------------------------------------------


def test_bundle_with_a_field_length_table_still_loads_and_ranks(tmp_path):
    # Older dev builds wrote `field_lengths` / `avg_field_lengths`. The server
    # ignores unknown keys, so such a bundle loads and scores as without them.
    config = ServerConfig.from_env(
        mesh_root=tmp_path, registry_path=tmp_path / "bin" / "registry.json"
    )
    nodes = [
        make_node("repo.a", "LedgerEntry", "src/Ledger.php", node_id="ledger"),
        make_node("repo.a", "Owner", "src/Owner.php", node_id="owner"),
    ]
    _write_generation(config.global_dir, "gen-1", nodes)
    lexical_path = config.global_dir / "current" / "lexical-index.json"
    lexical = json.loads(lexical_path.read_text(encoding="utf-8"))
    expected = retrieval.lexical_candidates("ledger", lexical, None)
    lexical["field_lengths"] = [[2, 3, 0] for _ in lexical["documents"]]
    lexical["avg_field_lengths"] = {"label": 2.0, "path": 3.0, "snippet": 0.0}
    lexical_path.write_text(json.dumps(lexical), encoding="utf-8")

    store = GenerationStore(config)
    generation = store.generation

    assert generation.generation_id == "gen-1"
    assert "field_lengths" in generation.lexical
    assert expected == [key_for("repo.a", nodes[0])]
    assert retrieval.lexical_candidates("ledger", generation.lexical, None) == expected


# --- MMR -----------------------------------------------------------------


def test_mmr_top3_is_a_prefix_of_top5():
    scored = [("a", 0.050), ("b", 0.049), ("c", 0.045), ("d", 0.030), ("e", 0.010)]
    paths = {"a": "f1", "b": "f1", "c": "f2", "d": "f3", "e": "f4"}

    top3 = ranking.mmr_select(scored, paths, 3)
    top5 = ranking.mmr_select(scored, paths, 5)

    assert top5[:3] == top3


def test_mmr_penalty_scale_for_tied_scores_is_explicit():
    assert ranking.mmr_penalty_scale([0.02, 0.02, 0.02]) == 1.0
    assert ranking.mmr_penalty_scale([]) == 1.0
    assert ranking.mmr_penalty_scale([0.05, 0.03]) == pytest.approx(0.02)


# --- anchors -------------------------------------------------------------


def test_same_path_in_another_repo_is_not_the_candidate_file():
    # Defensive: no repo-to-repo structural edge exists (invariant 1); anchors skip the overlay.
    path = "src/Controller.php"
    method = make_node("repo.a", ".export()", path, node_id="m")
    role = make_node("repo.b", "RoleGuard", path, node_id="r")
    lead = make_node("repo.b", "LeadAccess", path, node_id="l")
    gen = build_generation(
        _fillers() + [method, role, lead], links=[make_link("m", "r"), make_link("m", "l")]
    )

    keys = anchors.anchor_keys("export role lead access", gen, None, exclude=set())

    assert keys == [key_for("repo.a", method)]


# --- query instruction ---------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "qwen3-embedding:0.6b",
        "dengcao/Qwen3-Embedding-0.6B:Q8_0",
        "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF",
    ],
)
def test_qwen3_detection_matches_namespaced_model_names(model):
    assert embed_query.instruct_query(model, "q").startswith("Instruct: ")


@pytest.mark.parametrize("model", ["nomic-embed-text", "qwen3-embedding-like/other-model"])
def test_other_models_get_the_bare_query(model):
    assert embed_query.instruct_query(model, "q") == "q"
