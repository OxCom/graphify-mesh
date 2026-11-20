"""Regression shapes from real-project search probes, rebuilt on tiny synthetic
generations (no real project data).

1. Duplicate-posting stacking: several graph nodes that share one logical key
   (same file, same label, e.g. a TS property declared on several interfaces)
   each contributed their own posting, so a node matching one query term
   outranked a node matching three ("weekly maximum working hours decided").
2. MMR sinking: a second relevant hit from the same file as the first was
   pushed below every weaker candidate, because the same-file similarity
   penalty was on a different scale than the RRF scores it was subtracted from.
3. Anchor pinned on its own file: a rare-symbol anchor scored context words
   found only on sibling nodes of the same file.
"""

from __future__ import annotations

from conftest import build_generation, fake_embed_query_fn, key_for, make_link, make_node
from test_anchors import _fillers

from graphify_mesh.server import anchors, embed_query
from graphify_mesh.server.retrieval import rank


def _keys(result) -> list[str]:
    return [hit.key for hit in result.hits]


def test_one_term_node_with_duplicate_postings_does_not_outrank_three_term_match():
    one_term = [
        make_node("repo.a", ".hours()", "src/Models.ts", node_id=f"hours{i}") for i in range(8)
    ]
    three_terms = make_node(
        "repo.a", ".getMaximumWorkingHours()", "src/Overtime.php", node_id="target"
    )
    fillers = [
        make_node("repo.a", f".maximumWorking{i}()", f"src/filler{i}.php", node_id=f"f{i}")
        for i in range(6)
    ]
    gen = build_generation(one_term + [three_terms] + fillers)

    result = rank("maximum working hours", gen, None, k=5, embed_query_fn=fake_embed_query_fn())

    keys = _keys(result)
    assert keys[0] == key_for("repo.a", three_terms), keys


def test_same_file_second_hit_is_not_sunk_below_weaker_candidates():
    first = make_node("repo.a", "LockWeeks", "src/LockWeeks.php", node_id="first")
    second = make_node("repo.a", ".lockWeeks()", "src/LockWeeks.php", node_id="second")
    suffixes = ["Alpha", "Beta", "Gamma", "Delta", "Omega", "Sigma"] * 2
    weaker = [
        make_node("repo.a", f"Weeks{suffix}", f"src/other{i}.php", node_id=f"w{i}")
        for i, suffix in enumerate(suffixes)
    ]
    gen = build_generation([first, second] + weaker)

    # MMR's same-file penalty acts on per-node results only.
    result = rank(
        "lock weeks", gen, None, k=10, embed_query_fn=fake_embed_query_fn(), collapse_files=False
    )

    keys = _keys(result)
    assert keys[0] in {key_for("repo.a", first), key_for("repo.a", second)}, keys
    other = (
        key_for("repo.a", second)
        if keys[0] == key_for("repo.a", first)
        else key_for("repo.a", first)
    )
    # Before the fix the same-file hit fell behind all 12 nodes that match one
    # query term. Scaled, the same-file penalty costs it a few places only:
    # RRF scores here are rank-only, so the gap per place is small.
    assert other in keys[:5], keys


def _same_file_anchor_generation(path: str):
    """`.export()` whose context words live only on neighbours in its own file."""
    method = make_node("repo.a", ".export()", path, node_id="m")
    role = make_node("repo.a", "RoleGuard", path, node_id="r")
    lead = make_node("repo.a", "LeadAccess", path, node_id="l")
    links = [make_link("m", "r"), make_link("m", "l")]
    return method, build_generation(_fillers() + [method, role, lead], links=links)


def test_same_file_neighbours_alone_do_not_pin_an_anchor():
    # A `.hours()` method was pinned on the words of the sibling getters it
    # sums in its own file.
    _method, gen = _same_file_anchor_generation("src/Controller/Teams.php")

    assert anchors.anchor_keys("export role lead access", gen, None, exclude=set()) == []


def test_same_file_neighbours_corroborate_tokens_the_candidate_matches_itself():
    # The path carries `role` and `lead`, so the siblings only confirm them.
    method, gen = _same_file_anchor_generation("src/Role/Lead/Teams.php")

    keys = anchors.anchor_keys("export role lead access", gen, None, exclude=set())

    assert keys == [key_for("repo.a", method)]


def test_qwen3_query_gets_the_model_card_instruction_and_others_stay_bare(monkeypatch):
    sent: list[str] = []

    def fake_embed_batch(base_url, model, texts, timeout=None):
        sent.extend(texts)
        return [[1.0, 0.0]]

    monkeypatch.setattr(embed_query, "embed_batch", fake_embed_batch)
    embed_query.QueryEmbedder("http://x", "qwen3-embedding:0.6b", 1.0, 1.0)("lock weeks")
    embed_query.QueryEmbedder("http://x", "nomic-embed-text", 1.0, 1.0)("lock weeks")

    assert sent == [
        f"Instruct: {embed_query.QUERY_INSTRUCTION_TASK}\nQuery:lock weeks",
        "lock weeks",
    ]
