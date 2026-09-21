from __future__ import annotations

from graphify_mesh.sync import validate


def test_dangling_id_detected():
    data = {
        "nodes": [{"id": "a"}, {"id": "b"}],
        "links": [{"source": "a", "target": "ghost"}],
    }
    result = validate.validate_dangling_ids(data)
    assert not result.ok
    assert any("ghost" in e for e in result.errors)


def test_dangling_id_clean():
    data = {"nodes": [{"id": "a"}, {"id": "b"}], "links": [{"source": "a", "target": "b"}]}
    result = validate.validate_dangling_ids(data)
    assert result.ok


def test_forbidden_edge_cross_repo_flag():
    data = {
        "nodes": [{"id": "a"}, {"id": "b"}],
        "links": [{"source": "a", "target": "b", "cross_repo": True}],
    }
    result = validate.validate_forbidden_edges(data)
    assert not result.ok


def test_forbidden_edge_relation_type():
    data = {
        "nodes": [{"id": "a"}, {"id": "b"}],
        "links": [{"source": "a", "target": "b", "relation": "semantically_similar_to"}],
    }
    result = validate.validate_forbidden_edges(data)
    assert not result.ok


def test_forbidden_edge_clean():
    data = {
        "nodes": [{"id": "a"}, {"id": "b"}],
        "links": [{"source": "a", "target": "b", "relation": "calls"}],
    }
    result = validate.validate_forbidden_edges(data)
    assert result.ok


def test_forbidden_edge_same_repo_depends_on_is_allowed():
    """Upstream graphify's own extraction can legitimately emit a same-repo
    `depends_on` edge (e.g. a Helm Chart.yaml subchart dependency) — this is
    normal EXTRACTED data, not a cross-repo overlay leak, and must not trip
    the invariant."""
    data = {
        "nodes": [{"id": "acme.infra::chart"}, {"id": "acme.infra::subchart"}],
        "links": [
            {
                "source": "acme.infra::chart",
                "target": "acme.infra::subchart",
                "relation": "depends_on",
                "confidence": "EXTRACTED",
            }
        ],
    }
    result = validate.validate_forbidden_edges(data)
    assert result.ok


def test_forbidden_edge_cross_repo_depends_on_still_caught():
    data = {
        "nodes": [{"id": "acme.a::x"}, {"id": "acme.b::y"}],
        "links": [{"source": "acme.a::x", "target": "acme.b::y", "relation": "depends_on"}],
    }
    result = validate.validate_forbidden_edges(data)
    assert not result.ok


def test_forbidden_edge_cross_repo_structural_relation_caught():
    data = {
        "nodes": [{"id": "repo-a::x"}, {"id": "repo-b::y"}],
        "links": [{"source": "repo-a::x", "target": "repo-b::y", "relation": "calls"}],
    }
    result = validate.validate_forbidden_edges(data)
    assert not result.ok


def test_forbidden_edge_same_repo_structural_relation_allowed():
    data = {
        "nodes": [{"id": "repo-a::x"}, {"id": "repo-a::y"}],
        "links": [{"source": "repo-a::x", "target": "repo-a::y", "relation": "calls"}],
    }
    result = validate.validate_forbidden_edges(data)
    assert result.ok


def test_shrink_guard_refuses_smaller_graph():
    result = validate.validate_shrink_guard((5, 5), (10, 10), allow_shrink=False)
    assert not result.ok


def test_shrink_guard_allows_with_override():
    result = validate.validate_shrink_guard((5, 5), (10, 10), allow_shrink=True)
    assert result.ok


def test_community_name_placeholder_fails():
    data = {"nodes": [{"id": "a", "community": 0, "community_name": "Community 0"}]}
    result = validate.validate_community_names(data, skip_labeling=False)
    assert not result.ok


def test_community_name_skip_labeling_bypasses():
    data = {"nodes": [{"id": "a", "community": 0, "community_name": "Community 0"}]}
    result = validate.validate_community_names(data, skip_labeling=True)
    assert result.ok
    assert result.errors  # logged reason still present


def test_generation_manifest_missing_keys():
    result = validate.validate_generation_manifest({"generation_id": "x"})
    assert not result.ok
    assert len(result.errors) > 1


def test_forbidden_edges_catches_the_naming_collapse_shape():
    """The live incident: cem.hub's README was collapsed into
    oxcom.atlassian-mcp's, moving 29 of cem.hub's edges onto a node in another
    repo. Had the collapse published, these edges are what validate must reject."""
    data = {
        "nodes": [
            {"id": "cem.hub::svc", "repo": "cem.hub"},
            {"id": "oxcom.atlassian-mcp::readme", "repo": "oxcom.atlassian-mcp"},
        ],
        "links": [
            {
                "source": "cem.hub::svc",
                "target": "oxcom.atlassian-mcp::readme",
                "relation": "references",
            }
        ],
    }
    result = validate.validate_forbidden_edges(data)
    assert result.ok is False
    assert any("cross-repo edge" in err for err in result.errors)
