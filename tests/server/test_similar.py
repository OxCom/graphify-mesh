from __future__ import annotations

from conftest import build_generation, key_for, make_link, make_node

from graphify_mesh.server import similar


def test_find_similar_node_not_found_returns_unresolved():
    gen = build_generation([make_node("repo.a", "Alpha", "src/alpha.py", node_id="a1")])
    result = similar.find_similar("no-such-node", gen, k=5)
    assert result.resolved is False
    assert "node_not_found" in result.degraded
    assert result.hits == []


def test_find_similar_uses_overlay_similar_approach_edges_across_repos():
    src_node = make_node("repo.a", "OrderCalculator", "src/order_calc.py", node_id="src")
    tgt_node = make_node("repo.b", "OrderCalculatorClone", "src/clone.py", node_id="tgt")
    gen = build_generation(
        [src_node, tgt_node],
        overlay_edges=[
            {
                "type": "similar_approach",
                "source": {
                    "repo": "repo.a",
                    "source_file": "src/order_calc.py",
                    "qualified_label": "OrderCalculator",
                },
                "target": {
                    "repo": "repo.b",
                    "source_file": "src/clone.py",
                    "qualified_label": "OrderCalculatorClone",
                },
                "confidence": 0.87,
                "provenance": "ANN_COSINE",
            }
        ],
    )

    result = similar.find_similar("OrderCalculator", gen, k=5)
    assert result.resolved is True
    assert len(result.hits) == 1
    assert result.hits[0].key == key_for("repo.b", tgt_node)
    assert result.hits[0].score == 0.87


def test_find_similar_cross_repo_only_excludes_same_repo_structural_neighbors():
    seed = make_node("repo.a", "PaymentGateway", "src/gateway.py", node_id="seed")
    same_repo_neighbor = make_node("repo.a", "PaymentAdapter", "src/adapter.py", node_id="nb")
    gen = build_generation([seed, same_repo_neighbor], links=[make_link("seed", "nb")])

    result_all = similar.find_similar("PaymentGateway", gen, k=5, cross_repo_only=False)
    assert key_for("repo.a", same_repo_neighbor) in {h.key for h in result_all.hits}

    result_cross_only = similar.find_similar("PaymentGateway", gen, k=5, cross_repo_only=True)
    assert key_for("repo.a", same_repo_neighbor) not in {h.key for h in result_cross_only.hits}


def test_find_similar_does_not_follow_ambiguous_structural_edges():
    seed = make_node("repo.a", "PaymentGateway", "src/gateway.py", node_id="seed")
    extracted_neighbor = make_node("repo.a", "PaymentAdapter", "src/adapter.py", node_id="ext")
    ambiguous_neighbor = make_node("repo.a", "RefundGuess", "src/refund.py", node_id="amb")
    gen = build_generation(
        [seed, extracted_neighbor, ambiguous_neighbor],
        links=[make_link("seed", "ext"), make_link("seed", "amb", confidence="AMBIGUOUS")],
    )

    result = similar.find_similar("PaymentGateway", gen, k=5)
    keys = {h.key for h in result.hits}
    assert key_for("repo.a", extracted_neighbor) in keys
    assert key_for("repo.a", ambiguous_neighbor) not in keys


def test_find_similar_falls_back_to_exact_label_and_community_match_when_no_edges():
    seed = make_node("repo.a", "Widget", "src/widget.py", node_id="seed", community_name="commerce")
    unrelated = make_node(
        "repo.a", "Widget", "src/widget_alt.py", node_id="alt", community_name="commerce"
    )
    different_community = make_node(
        "repo.b", "Widget", "src/other.py", node_id="oth", community_name="billing"
    )
    gen = build_generation(
        [seed, unrelated, different_community]
    )  # no overlay edges, no structural links

    result = similar.find_similar("Widget", gen, k=5)
    assert result.resolved is True
    assert "similarity_fallback_exact_match" in result.degraded
    hit_keys = {h.key for h in result.hits}
    assert key_for("repo.a", unrelated) in hit_keys
    assert (
        key_for("repo.b", different_community) not in hit_keys
    )  # different community_name excluded


def test_find_similar_k_cap_respected():
    seed = make_node("repo.a", "Common", "src/common.py", node_id="seed", community_name="shared")
    matches = [
        make_node("repo.a", "Common", f"src/dup{i}.py", node_id=f"m{i}", community_name="shared")
        for i in range(10)
    ]
    gen = build_generation([seed] + matches)
    result = similar.find_similar("Common", gen, k=3)
    assert len(result.hits) <= 3
