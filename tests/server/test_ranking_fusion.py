from __future__ import annotations

import dataclasses
import json

import pytest
from conftest import build_generation, fake_embed_query_fn, key_for, make_link, make_node

from graphify_mesh.server import ranking, retrieval
from graphify_mesh.server.retrieval import rank


def test_rrf_pinned_constants_are_named_not_magic():
    assert ranking.RRF_K == 60
    assert ranking.rrf_contribution(0) == 1.0 / 61
    assert ranking.rrf_contribution(1) == 1.0 / 62


def test_fuse_rankings_sums_only_over_retrievers_that_returned_something():
    fused = ranking.fuse_rankings({"lexical": ["a", "b"], "vector": [], "structural": ["b", "c"]})
    # "b" appears in both lexical (rank 1) and structural (rank 0): summed.
    assert fused["b"] > fused["a"]
    assert fused["b"] > fused["c"]
    assert set(fused) == {"a", "b", "c"}


def test_exact_alias_bypass_ranked_first_and_skips_penalties():
    hub_node = make_node("repo.a", "TimeService", "src/TimeService.php", node_id="hub")
    other_nodes = [
        make_node("repo.a", f"Neighbor{i}", f"src/n{i}.py", node_id=f"nb{i}") for i in range(60)
    ]
    links = [
        make_link("hub", n["id"]) for n in other_nodes
    ]  # push hub degree above HUB_DEGREE_THRESHOLD
    gen = build_generation([hub_node] + other_nodes, links=links)

    result = rank("TimeService", gen, None, k=5, embed_query_fn=fake_embed_query_fn())

    assert result.hits, "exact alias hit must be found even though the node is a high-degree hub"
    top = result.hits[0]
    assert top.match_type == "exact"
    assert top.key == key_for("repo.a", hub_node)
    # Exact bypass hits carry the sentinel score and are never penalized.
    assert top.score == retrieval.EXACT_MATCH_SCORE


def test_apply_penalties_hub_degree_and_deprecated_are_multiplicative():
    """Direct unit test of the penalty formula itself (isolated from
    fusion/MMR pool effects): a node whose degree exceeds
    HUB_DEGREE_THRESHOLD is multiplicatively down-weighted by
    HUB_PENALTY_FACTOR, and a DEPRECATED-path node by
    DEPRECATED_PENALTY_FACTOR — independently and multiplicatively when both
    apply."""
    degree_by_key = {"hub": 51, "low": 3, "both": 51}
    path_by_key = {"hub": "src/x.py", "low": "src/x.py", "both": "src/DEPRECATED/x.py"}

    assert (
        ranking.apply_penalties("hub", 1.0, degree_by_key, path_by_key)
        == ranking.HUB_PENALTY_FACTOR
    )
    assert ranking.apply_penalties("low", 1.0, degree_by_key, path_by_key) == 1.0
    assert ranking.apply_penalties("both", 1.0, degree_by_key, path_by_key) == pytest.approx(
        ranking.HUB_PENALTY_FACTOR * ranking.DEPRECATED_PENALTY_FACTOR
    )


def test_hub_degree_penalty_demotes_high_degree_node_vs_relevant_low_degree_one_end_to_end():
    """End-to-end through `rank()`: the fillers that inflate the hub's
    degree live in a DIFFERENT repo excluded by `repo_filter`, so they never
    enter the fused candidate pool themselves (scope-before-ranking, see
    test_scope_before_ranking.py) — this isolates the hub-degree penalty's
    effect on ranking ORDER from the separate question of candidate-pool
    pollution."""
    hub = make_node("repo.a", "CacheKeyThing", "src/hub.py", node_id="hub")
    low_degree = make_node("repo.a", "CacheKeyHandler", "src/low.py", node_id="low")
    fillers = [
        make_node("repo.b", f"Filler{i}", f"src/f{i}.py", node_id=f"f{i}") for i in range(60)
    ]
    links = [
        make_link("hub", f["id"]) for f in fillers
    ]  # hub's total degree = 60 > HUB_DEGREE_THRESHOLD(50)
    gen = build_generation([hub, low_degree] + fillers, links=links)

    assert gen.degree("hub") > ranking.HUB_DEGREE_THRESHOLD
    assert gen.degree("low") <= ranking.HUB_DEGREE_THRESHOLD

    result = rank("cache", gen, frozenset({"repo.a"}), k=10, embed_query_fn=fake_embed_query_fn())
    keys = [h.key for h in result.hits]
    hub_key = key_for("repo.a", hub)
    low_key = key_for("repo.a", low_degree)
    assert hub_key in keys and low_key in keys
    # Low-degree relevant hit must outrank the penalized hub, despite both
    # matching the same lexical query term with an identical raw score.
    assert keys.index(low_key) < keys.index(hub_key)


def test_deprecated_path_down_weighted():
    normal = make_node("repo.a", "PaymentHandler", "src/PaymentHandler.py", node_id="normal")
    deprecated = make_node(
        "repo.a", "PaymentHandlerOld", "src/DEPRECATED/PaymentHandlerOld.py", node_id="dep"
    )
    gen = build_generation([normal, deprecated])

    result = rank("handler", gen, None, k=10, embed_query_fn=fake_embed_query_fn())
    keys = [h.key for h in result.hits]
    normal_key = key_for("repo.a", normal)
    dep_key = key_for("repo.a", deprecated)
    assert keys.index(normal_key) < keys.index(dep_key)
    dep_hit = next(h for h in result.hits if h.key == dep_key)
    assert dep_hit.deprecated is True


def test_inferred_edges_excluded_by_default_from_structural_candidates():
    seed = make_node("repo.a", "OrderService", "src/order.py", node_id="seed")
    extracted_neighbor = make_node("repo.a", "ExtractedNeighbor", "src/extracted.py", node_id="ext")
    inferred_neighbor = make_node("repo.a", "InferredNeighbor", "src/inferred.py", node_id="inf")
    links = [
        make_link("seed", "ext", confidence="EXTRACTED"),
        make_link("seed", "inf", confidence="INFERRED"),
    ]
    gen = build_generation([seed, extracted_neighbor, inferred_neighbor], links=links)

    result = rank("orderservice", gen, None, k=10, embed_query_fn=fake_embed_query_fn())
    keys = {h.key for h in result.hits}
    assert key_for("repo.a", extracted_neighbor) in keys
    assert key_for("repo.a", inferred_neighbor) not in keys


def test_ambiguous_edges_excluded_by_default_from_structural_candidates():
    seed = make_node("repo.a", "OrderService", "src/order.py", node_id="seed")
    ambiguous_neighbor = make_node("repo.a", "AmbiguousNeighbor", "src/ambiguous.py", node_id="amb")
    links = [make_link("seed", "amb", confidence="AMBIGUOUS")]
    gen = build_generation([seed, ambiguous_neighbor], links=links)
    ambiguous_key = key_for("repo.a", ambiguous_neighbor)

    default = rank("orderservice", gen, None, k=10, embed_query_fn=fake_embed_query_fn())
    assert ambiguous_key not in {h.key for h in default.hits}

    opted_in = rank(
        "orderservice",
        gen,
        None,
        k=10,
        embed_query_fn=fake_embed_query_fn(),
        include_inferred=True,
    )
    assert ambiguous_key in {h.key for h in opted_in.hits}


def test_degraded_mode_renormalizes_over_available_retrievers_when_vectors_missing():
    n1 = make_node("repo.a", "AlphaThing", "src/alpha.py", node_id="n1")
    n2 = make_node("repo.a", "BetaThing", "src/beta.py", node_id="n2")
    gen = build_generation([n1, n2], embeddings={})  # no embeddings published this generation

    result = rank("thing", gen, None, k=10, embed_query_fn=fake_embed_query_fn())
    assert "embeddings_unavailable" in result.degraded
    # Lexical-only fusion must still surface real hits, not an empty result.
    assert result.hits


def test_deterministic_tie_break_repeat_query_stability():
    nodes = [
        make_node("repo.a", f"Widget{i}", f"src/widget{i}.py", node_id=f"w{i}") for i in range(5)
    ]
    gen = build_generation(nodes)

    first = rank("widget", gen, None, k=5, embed_query_fn=fake_embed_query_fn())
    second = rank("widget", gen, None, k=5, embed_query_fn=fake_embed_query_fn())
    assert [h.key for h in first.hits] == [h.key for h in second.hits]


def test_mmr_reduces_near_duplicate_flooding_from_same_source_file():
    # Several distinct labels crammed into the SAME source_file: MMR's
    # same-file similarity proxy should push the pool toward other files
    # rather than exhausting k on one file's duplicates when a comparably
    # scored candidate from a different file exists.
    same_file_nodes = [
        make_node("repo.a", f"DupThing{i}", "src/dup.py", node_id=f"dup{i}") for i in range(5)
    ]
    distinct_node = make_node("repo.a", "ThingElsewhere", "src/elsewhere.py", node_id="distinct")
    gen = build_generation(same_file_nodes + [distinct_node])

    result = rank("thing", gen, None, k=3, embed_query_fn=fake_embed_query_fn())
    files = [h.source_file for h in result.hits]
    assert "src/elsewhere.py" in files, (
        "MMR should diversify in at least one distinct-file candidate"
    )


def test_exact_hit_score_survives_a_strict_json_round_trip():
    """`float("inf")` serializes as the bare token `Infinity`, which strict
    parsers reject — `allow_nan=False` is what makes this test see that."""
    node = make_node("repo.a", "TimeService", "src/TimeService.php", node_id="n1")
    gen = build_generation([node])

    result = rank("TimeService", gen, None, k=5, embed_query_fn=fake_embed_query_fn())
    assert result.hits[0].match_type == "exact"

    payload = [dataclasses.asdict(h) for h in result.hits]
    decoded = json.loads(json.dumps(payload, allow_nan=False))
    assert decoded[0]["score"] == retrieval.EXACT_MATCH_SCORE


def test_exact_alias_hits_are_capped_at_k():
    """One alias shared by many nodes must still respect `k`."""
    nodes = [
        make_node("repo.a", "Widget", f"src/pkg{i}/widget.py", node_id=f"n{i}") for i in range(5)
    ]
    gen = build_generation(nodes)

    unbounded = rank("Widget", gen, None, k=10, embed_query_fn=fake_embed_query_fn())
    assert len([h for h in unbounded.hits if h.match_type == "exact"]) == 5

    result = rank("Widget", gen, None, k=1, embed_query_fn=fake_embed_query_fn())
    assert len(result.hits) == 1
    assert result.hits[0].match_type == "exact"


def test_k_cap_never_exceeds_max_k():
    nodes = [make_node("repo.a", f"Item{i}", f"src/item{i}.py", node_id=f"i{i}") for i in range(5)]
    gen = build_generation(nodes)
    result = rank("item", gen, None, k=10_000, embed_query_fn=fake_embed_query_fn())
    assert len(result.hits) <= ranking.MAX_K
