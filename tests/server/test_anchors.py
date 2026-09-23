from __future__ import annotations

from types import SimpleNamespace

import pytest
from conftest import build_generation, fake_embed_query_fn, key_for, make_link, make_node

from graphify_mesh.server import anchors, lexical_read, retrieval
from graphify_mesh.server.retrieval import rank

QUERY = "export role lead access"
FILLER_COUNT = 300


def _fillers(repo: str = "repo.a") -> list[dict]:
    # Unrelated documents: they keep every query token rare, so one context
    # token is worth ln(N) ~ 5.7 idf and two clear ANCHOR_MIN_SCORE.
    return [
        make_node(repo, f"Filler{i}", f"src/fill/f{i}.py", node_id=f"{repo}:f{i}")
        for i in range(FILLER_COUNT)
    ]


def _controller(
    repo: str, name: str, confidence: str = "EXTRACTED", context: bool = True
) -> tuple[list[dict], list[dict]]:
    """A `.export()` method whose EXTRACTED neighbours carry the query's
    context tokens `role`, `lead`, `access`."""
    path = f"src/Controller/{name}.php"
    method = make_node(repo, ".export()", path, node_id=f"{repo}:{name}:export")
    nodes = [method]
    links: list[dict] = []
    if context:
        guard = make_node(repo, "RoleGuard", f"src/Guard/{name}Role.php", node_id=f"{name}:g")
        lead = make_node(repo, "LeadAccess", f"src/Guard/{name}Lead.php", node_id=f"{name}:l")
        nodes += [guard, lead]
        links += [
            make_link(method["id"], guard["id"], confidence=confidence),
            make_link(method["id"], lead["id"], confidence=confidence),
        ]
    return nodes, links


def _rank_disabled(monkeypatch, *args, **kwargs):
    with monkeypatch.context() as patch:
        patch.setattr(anchors, "anchor_keys", lambda *a, **kw: [])
        return rank(*args, **kwargs)


def test_dominant_anchor_is_pinned_first_with_anchor_match_type():
    nodes, links = _controller("repo.a", "TeamsController")
    gen = build_generation(_fillers() + nodes, links=links)

    result = rank(QUERY, gen, None, k=5, embed_query_fn=fake_embed_query_fn())

    top = result.hits[0]
    assert top.match_type == "anchor"
    assert top.key == key_for("repo.a", nodes[0])
    assert top.score == anchors.ANCHOR_MATCH_SCORE
    assert [h.key for h in result.hits].count(top.key) == 1
    assert all(h.match_type == "fused" for h in result.hits[1:])


def test_anchor_follows_exact_hits():
    exact = make_node("repo.a", QUERY, "src/misc/e.py", node_id="exact")
    nodes, links = _controller("repo.a", "TeamsController")
    gen = build_generation(_fillers() + [exact] + nodes, links=links)

    result = rank(QUERY, gen, None, k=5, embed_query_fn=fake_embed_query_fn())

    assert [h.match_type for h in result.hits[:2]] == ["exact", "anchor"]
    assert result.hits[0].key == key_for("repo.a", exact)
    assert result.hits[0].score == retrieval.EXACT_MATCH_SCORE
    assert result.hits[1].key == key_for("repo.a", nodes[0])


def test_pinned_anchor_leaves_fused_hits_as_in_disabled_path(monkeypatch):
    nodes, links = _controller("repo.a", "TeamsController")
    sibling = make_node("repo.a", "RoleLeadHelper", nodes[0]["source_file"], node_id="sibling")
    gen = build_generation(_fillers() + nodes + [sibling], links=links)
    anchor_key = key_for("repo.a", nodes[0])
    k = 5

    assert anchors.anchor_keys(QUERY, gen, None, exclude=set()) == [anchor_key]
    embed = fake_embed_query_fn()
    enabled = rank(QUERY, gen, None, k=k, embed_query_fn=embed)
    disabled = _rank_disabled(monkeypatch, QUERY, gen, None, k=k, embed_query_fn=embed)

    disabled_keys = [h.key for h in disabled.hits]
    assert anchor_key in disabled_keys
    assert key_for("repo.a", sibling) in disabled_keys
    assert enabled.hits[0].key == anchor_key
    fused = [h for h in enabled.hits if h.match_type == "fused"]
    assert fused == [h for h in disabled.hits if h.key != anchor_key][: k - 1]


def test_exact_hit_is_never_duplicated_as_anchor():
    # The basename alias `lead` makes the exact hit a dominant anchor
    # candidate on its own; `rank()` must list it once, as "exact".
    exact = make_node("repo.a", "Role lead access", "src/lead", node_id="exact")
    guard = make_node("repo.a", "AccessGuard", "src/Guard/Access.php", node_id="g")
    gen = build_generation(_fillers() + [exact, guard], links=[make_link("exact", "g")])
    exact_key = key_for("repo.a", exact)
    query = "Role lead access"

    assert anchors.anchor_keys(query, gen, None, exclude=set()) == [exact_key]
    assert anchors.anchor_keys(query, gen, None, exclude={exact_key}) == []

    result = rank(query, gen, None, k=5, embed_query_fn=fake_embed_query_fn())

    assert result.hits[0].key == exact_key
    assert result.hits[0].match_type == "exact"
    assert [h.key for h in result.hits].count(exact_key) == 1
    assert all(h.match_type != "anchor" for h in result.hits)


def test_close_scores_pin_nothing_and_output_matches_disabled_path(monkeypatch):
    teams, teams_links = _controller("repo.a", "TeamsController")
    users, users_links = _controller("repo.a", "UsersController")
    gen = build_generation(_fillers() + teams + users, links=teams_links + users_links)

    assert anchors.anchor_keys(QUERY, gen, None, exclude=set()) == []
    embed = fake_embed_query_fn()
    enabled = rank(QUERY, gen, None, k=10, embed_query_fn=embed)
    disabled = _rank_disabled(monkeypatch, QUERY, gen, None, k=10, embed_query_fn=embed)
    assert enabled == disabled
    assert all(h.match_type != "anchor" for h in enabled.hits)


def test_query_without_alias_resolvable_token_matches_disabled_path(monkeypatch):
    nodes, links = _controller("repo.a", "TeamsController")
    gen = build_generation(_fillers() + nodes, links=links)
    query = "role lead access guard"

    assert anchors.anchor_keys(query, gen, None, exclude=set()) == []
    embed = fake_embed_query_fn()
    enabled = rank(query, gen, None, k=10, embed_query_fn=embed)
    disabled = _rank_disabled(monkeypatch, query, gen, None, k=10, embed_query_fn=embed)
    assert enabled == disabled


@pytest.mark.parametrize(
    ("extra_members", "pinned"),
    [
        (anchors.ANCHOR_MAX_ALIAS_MEMBERS - 1, True),
        (anchors.ANCHOR_MAX_ALIAS_MEMBERS, False),
    ],
)
def test_alias_with_too_many_members_contributes_no_candidate(extra_members, pinned):
    nodes, links = _controller("repo.a", "TeamsController")
    for i in range(extra_members):
        nodes += _controller("repo.a", f"Other{i}Controller", context=False)[0]
    gen = build_generation(_fillers() + nodes, links=links)

    keys = anchors.anchor_keys(QUERY, gen, None, exclude=set())

    assert keys == ([key_for("repo.a", nodes[0])] if pinned else [])


def test_anchor_below_min_score_is_not_pinned():
    # One context token only: ln(N) < ANCHOR_MIN_SCORE.
    method = make_node("repo.a", ".export()", "src/Controller/Teams.php", node_id="m")
    guard = make_node("repo.a", "RoleGuard", "src/Guard/Role.php", node_id="g")
    gen = build_generation(_fillers() + [method, guard], links=[make_link("m", "g")])

    assert anchors.anchor_keys("export role", gen, None, exclude=set()) == []


@pytest.mark.parametrize("confidence", ["INFERRED", "AMBIGUOUS"])
def test_non_extracted_neighbour_edges_contribute_no_score(confidence):
    nodes, links = _controller("repo.a", "TeamsController", confidence=confidence)
    gen = build_generation(_fillers() + nodes, links=links)

    assert anchors.anchor_keys(QUERY, gen, None, exclude=set()) == []


def test_repo_filter_excludes_other_repo_anchors():
    nodes, links = _controller("repo.b", "TeamsController")
    gen = build_generation(_fillers() + nodes, links=links)

    assert anchors.anchor_keys(QUERY, gen, frozenset({"repo.a"}), exclude=set()) == []
    assert anchors.anchor_keys(QUERY, gen, frozenset({"repo.b"}), exclude=set()) == [
        key_for("repo.b", nodes[0])
    ]
    result = rank(QUERY, gen, frozenset({"repo.a"}), k=5, embed_query_fn=fake_embed_query_fn())
    assert all(h.match_type != "anchor" for h in result.hits)


def _tokens_and_via(query, gen, repo_filter):
    tokens = list(dict.fromkeys(anchors.tokenize_text(query)))
    via: dict[str, set[str]] = {}
    for token in tokens:
        for form in (token, f".{token}()"):
            members = anchors._alias_members(form, gen, repo_filter)
            if 1 <= len(members) <= anchors.ANCHOR_MAX_ALIAS_MEMBERS:
                for key in members:
                    via.setdefault(key, set()).add(token)
    return tokens, via


def _scores_both_paths(monkeypatch, query, gen, repo_filter):
    """Per-candidate scores from the v3 probe path and the full-set path
    (forced by making `doc_ids_by_key` report a v2 index)."""
    tokens, via = _tokens_and_via(query, gen, repo_filter)
    monkeypatch.setattr(anchors, "_cache", None)
    probe = anchors._candidate_scores(tokens, via, gen, repo_filter)
    with monkeypatch.context() as patch:
        patch.setattr(anchors, "_cache", None)
        patch.setattr(anchors.lexical_read, "doc_ids_by_key", lambda _lexical: None)
        full = anchors._candidate_scores(tokens, via, gen, repo_filter)
        assert anchors._cache is not None and anchors._cache.doc_ids_by_key is None
    monkeypatch.setattr(anchors, "_cache", None)
    return probe, full


@pytest.mark.parametrize("repo_filter", [None, frozenset({"repo.a"}), frozenset({"repo.b"})])
def test_probe_path_scores_match_full_set_path(monkeypatch, repo_filter):
    nodes, links = _controller("repo.a", "TeamsController")
    sibling = make_node("repo.a", "RoleLeadHelper", nodes[0]["source_file"], node_id="sibling")
    other, other_links = _controller("repo.b", "UsersController")
    gen = build_generation(_fillers() + nodes + [sibling] + other, links=links + other_links)

    probe, full = _scores_both_paths(monkeypatch, QUERY, gen, repo_filter)

    assert probe
    assert probe == full
    anchors._cache = None
    pinned = anchors.anchor_keys(QUERY, gen, repo_filter, exclude=set())
    with monkeypatch.context() as patch:
        patch.setattr(anchors, "_cache", None)
        patch.setattr(anchors.lexical_read, "doc_ids_by_key", lambda _lexical: None)
        assert anchors.anchor_keys(QUERY, gen, repo_filter, exclude=set()) == pinned


def test_probe_path_counts_every_document_of_a_shared_key():
    # Nodes sharing repo, file and label share one key but are separate
    # documents; a posting on the second document alone must count.
    lexical = {
        "schema_version": 3,
        "fields": ["label", "path", "snippet"],
        "documents": [["repo.a", "k"], ["repo.a", "k"], ["repo.a", "other"]],
        "postings": {"access": [(1 << 2) | 2]},
        "document_count": 3,
    }
    key_docs = lexical_read.doc_ids_by_key(lexical)
    assert key_docs == {"k": [0, 1], "other": [2]}
    gen = SimpleNamespace(lexical=lexical)

    assert anchors._probe_token_keys("access", gen, key_docs, {0, 1, 2}) == {"k"}
    assert anchors._full_token_keys("access", gen, None) == {"k"}


def test_cache_is_rebuilt_when_generation_changes():
    nodes, links = _controller("repo.a", "TeamsController")
    first = build_generation(_fillers() + nodes, links=links, generation_id="gen-a")
    anchor_key = key_for("repo.a", nodes[0])

    assert anchors.anchor_keys(QUERY, first, None, exclude=set()) == [anchor_key]
    cached = anchors._cache
    assert cached is not None and cached.generation_id == "gen-a"
    assert anchors.anchor_keys(QUERY, first, None, exclude=set()) == [anchor_key]
    assert anchors._cache is cached

    # New generation id: the anchor lost its context, the stale map must not leak.
    bare = _controller("repo.a", "TeamsController", context=False)[0]
    second = build_generation(_fillers() + bare, generation_id="gen-b")
    assert anchors.anchor_keys(QUERY, second, None, exclude=set()) == []
    assert anchors._cache is not cached
    assert anchors._cache.generation_id == "gen-b"

    # Same id, different generation object: rebuilt as well.
    third = build_generation(_fillers() + nodes, links=links, generation_id="gen-b")
    assert anchors.anchor_keys(QUERY, third, None, exclude=set()) == [anchor_key]
    assert anchors._cache.generation() is third


def test_df_above_document_count_contributes_zero_not_negative(monkeypatch):
    # A v2 index without `doc_freq.global` counts field postings, so df can
    # exceed the document count; that token must add 0, never subtract.
    nodes, links = _controller("repo.a", "TeamsController")
    gen = build_generation(_fillers() + nodes, links=links)
    tokens, via = _tokens_and_via(QUERY, gen, None)
    total_docs = lexical_read.document_count(gen.lexical)
    real_doc_freq = anchors._doc_freq

    def scores_with_access_df(df):
        with monkeypatch.context() as patch:
            patch.setattr(anchors, "_cache", None)
            patch.setattr(
                anchors,
                "_doc_freq",
                lambda token, *a: df if token == "access" else real_doc_freq(token, *a),
            )
            return anchors._candidate_scores(tokens, via, gen, None)

    zero_idf = scores_with_access_df(total_docs)
    assert zero_idf[0][1] > 0
    assert scores_with_access_df(total_docs * 3) == zero_idf


@pytest.mark.parametrize(
    ("scores", "pinned"),
    [
        ([("A", 30.0), ("B", 14.0), ("C", 6.0)], ["A", "B"]),
        ([("A", 30.0), ("B", 14.0), ("C", 8.0)], ["A"]),
        ([("A", 30.0), ("B", 9.0)], ["A"]),
        # C also clears the floor and dominates D, but the cap stops at two.
        ([("A", 100.0), ("B", 40.0), ("C", 16.0), ("D", 6.0)], ["A", "B"]),
    ],
)
def test_second_pin_needs_its_own_dominance_and_cap_is_two(monkeypatch, scores, pinned):
    nodes, links = _controller("repo.a", "TeamsController")
    gen = build_generation(_fillers() + nodes, links=links)
    monkeypatch.setattr(anchors, "_candidate_scores", lambda *a: scores)

    assert anchors.anchor_keys(QUERY, gen, None, exclude=set()) == pinned


def test_anchor_hits_truncated_to_k(monkeypatch):
    teams, teams_links = _controller("repo.a", "TeamsController")
    users, users_links = _controller("repo.a", "UsersController")
    gen = build_generation(_fillers() + teams + users, links=teams_links + users_links)
    first, second = key_for("repo.a", teams[0]), key_for("repo.a", users[0])
    monkeypatch.setattr(anchors, "anchor_keys", lambda *a, **kw: [first, second])

    result = rank(QUERY, gen, None, k=1, embed_query_fn=fake_embed_query_fn())

    assert [(h.key, h.match_type) for h in result.hits] == [(first, "anchor")]


def test_exact_hit_filling_k_skips_anchor_lookup(monkeypatch):
    exact = make_node("repo.a", QUERY, "src/misc/e.py", node_id="exact")
    nodes, links = _controller("repo.a", "TeamsController")
    gen = build_generation(_fillers() + [exact] + nodes, links=links)

    def fail(*_a, **_kw):
        raise AssertionError("anchor_keys consulted with no slot left")

    monkeypatch.setattr(anchors, "anchor_keys", fail)

    result = rank(QUERY, gen, None, k=1, embed_query_fn=fake_embed_query_fn())

    assert [(h.key, h.match_type) for h in result.hits] == [(key_for("repo.a", exact), "exact")]
