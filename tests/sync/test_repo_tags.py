from __future__ import annotations

from pathlib import Path

from graphify.build import distinct_repo_tags

from graphify_mesh.sync import repo_tags


def test_real_distinct_repo_tags_collides_for_collection_layout():
    """Documents the verified gap (repo_tags.py module docstring): graphify's
    own merge-graphs tag derivation collides/diverges from registry repo_id
    for the `graphify/<product>/<sub>/graph.json` collection layout."""
    paths = [
        Path("/path/to/graph-mesh/graphify/example-org/backend-a/graph.json"),
        Path("/path/to/graph-mesh/graphify/example-org/frontend-b/graph.json"),
    ]
    tags = distinct_repo_tags(paths)
    assert tags != [
        "example-org.backend-a",
        "example-org.frontend-b",
    ]  # confirmed NOT the registry repo_id


def test_compute_tag_to_repo_id_maps_auto_tags_to_true_repo_ids():
    paths = [
        Path("/path/to/graph-mesh/graphify/example-org/backend-a/graph.json"),
        Path("/path/to/graph-mesh/graphify/example-org/frontend-b/graph.json"),
    ]
    repo_ids = sorted(["example-org.backend-a", "example-org.frontend-b"])
    mapping = repo_tags.compute_tag_to_repo_id(paths, repo_ids)
    assert set(mapping.values()) == set(repo_ids)
    assert len(mapping) == 2


def test_rewrite_repo_tags_fixes_node_ids_repo_attr_and_edges():
    tag_to_repo_id = {
        "graphify_example-org": "example-org.backend-a",
        "graphify_example-org-2": "example-org.frontend-b",
    }
    graph_data = {
        "nodes": [
            {"id": "graphify_example-org::n1", "label": "Foo", "repo": "graphify_example-org"},
            {"id": "graphify_example-org-2::n2", "label": "Bar", "repo": "graphify_example-org-2"},
            {"id": "external_node_no_prefix", "label": "Ext"},
        ],
        "links": [
            {
                "source": "graphify_example-org::n1",
                "target": "graphify_example-org-2::n2",
                "relation": "calls",
            },
        ],
    }
    result = repo_tags.rewrite_repo_tags(graph_data, tag_to_repo_id)

    ids = {n["id"] for n in result["nodes"]}
    assert "example-org.backend-a::n1" in ids
    assert "example-org.frontend-b::n2" in ids
    assert "external_node_no_prefix" in ids  # untouched, no matching tag

    repos = {n["id"]: n.get("repo") for n in result["nodes"]}
    assert repos["example-org.backend-a::n1"] == "example-org.backend-a"
    assert repos["example-org.frontend-b::n2"] == "example-org.frontend-b"

    link = result["links"][0]
    assert link["source"] == "example-org.backend-a::n1"
    assert link["target"] == "example-org.frontend-b::n2"


def test_rewrite_repo_tags_leaves_unknown_prefix_untouched():
    graph_data = {"nodes": [{"id": "unknown_tag::n1", "repo": "unknown_tag"}], "links": []}
    result = repo_tags.rewrite_repo_tags(graph_data, {})
    assert result["nodes"][0]["id"] == "unknown_tag::n1"
    assert result["nodes"][0]["repo"] == "unknown_tag"


def test_rewrite_repo_tags_remaps_hyperedge_ids_and_members():
    graph_data = {
        "nodes": [{"id": "graphify_x::a", "repo": "graphify_x"}],
        "links": [],
        "hyperedges": [{"id": "graphify_x::h", "nodes": ["graphify_x::a"]}],
    }
    result = repo_tags.rewrite_repo_tags(graph_data, {"graphify_x": "org.repo"})

    hyperedge = result["hyperedges"][0]
    assert hyperedge["id"] == "org.repo::h"
    assert hyperedge["nodes"] == ["org.repo::a"]

    no_hyperedges = {"nodes": [{"id": "graphify_x::a", "repo": "graphify_x"}], "links": []}
    plain = repo_tags.rewrite_repo_tags(no_hyperedges, {"graphify_x": "org.repo"})
    assert "hyperedges" not in plain


def test_nested_graph_hyperedges_slot_is_remapped_too():
    """Upstream writes hyperedges into BOTH the top-level key and the nested
    graph.hyperedges slot. Measured on the live merged graph: the top-level
    slot resolved 39/39 members and the nested slot 0/39."""
    shared_top = {"id": "graphify_x::h", "nodes": ["graphify_x::a", "graphify_x::b"]}
    graph_data = {
        "nodes": [
            {"id": "graphify_x::a", "repo": "graphify_x"},
            {"id": "graphify_x::b", "repo": "graphify_x"},
        ],
        "links": [],
        "hyperedges": [shared_top],
        "graph": {
            "hyperedges": [{"id": "graphify_x::h", "nodes": ["graphify_x::a", "graphify_x::b"]}]
        },
    }
    result = repo_tags.rewrite_repo_tags(graph_data, {"graphify_x": "org.repo"})

    node_ids = {n["id"] for n in result["nodes"]}
    for slot in (result["hyperedges"], result["graph"]["hyperedges"]):
        assert slot[0]["id"] == "org.repo::h"
        assert slot[0]["nodes"] == ["org.repo::a", "org.repo::b"]
        assert set(slot[0]["nodes"]) <= node_ids


def test_a_hyperedge_object_shared_between_slots_is_remapped_once():
    """Both slots can hold the SAME dict. Remapping it twice would re-prefix an
    id that is already correct."""
    shared = {"id": "graphify_x::h", "nodes": ["graphify_x::a"]}
    graph_data = {
        "nodes": [{"id": "graphify_x::a", "repo": "graphify_x"}],
        "links": [],
        "hyperedges": [shared],
        "graph": {"hyperedges": [shared]},
    }
    result = repo_tags.rewrite_repo_tags(graph_data, {"graphify_x": "org.repo"})
    assert result["hyperedges"][0]["nodes"] == ["org.repo::a"]
    assert result["graph"]["hyperedges"][0]["nodes"] == ["org.repo::a"]


def test_hyperedge_member_alias_keys_are_remapped():
    graph_data = {
        "nodes": [{"id": "graphify_x::a", "repo": "graphify_x"}],
        "links": [],
        "hyperedges": [{"id": "graphify_x::h", "members": ["graphify_x::a"]}],
    }
    result = repo_tags.rewrite_repo_tags(graph_data, {"graphify_x": "org.repo"})
    assert result["hyperedges"][0]["members"] == ["org.repo::a"]


def test_compute_tag_to_repo_id_count_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        repo_tags.compute_tag_to_repo_id([Path("/a/b/graph.json")], ["repo.a", "repo.b"])
