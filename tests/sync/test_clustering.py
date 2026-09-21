import sys
import types

from graphify_mesh.sync import clustering

# Captured at import time, before conftest's autouse `_no_live_labeling` fixture
# replaces the module attribute with a fake that never reaches graphify.
_REAL_LLM_NAMES = clustering.llm_names

TWO_REPO_GRAPH = {
    "directed": False,
    "multigraph": False,
    "graph": {},
    "nodes": [
        {
            "id": "a.one::README.md",
            "label": "README.md",
            "source_file": "README.md",
            "repo": "a.one",
            "_origin": "semantic",
            "file_type": "document",
        },
        {
            "id": "b.two::readme",
            "label": "README.md",
            "source_file": "README.md",
            "repo": "b.two",
            "_origin": "ast",
            "file_type": "document",
            "source_location": "L1",
        },
        {
            "id": "a.one::svc",
            "label": "Service",
            "source_file": "src/Service.php",
            "repo": "a.one",
            "_origin": "ast",
            "source_location": "L10",
        },
        {
            "id": "b.two::svc",
            "label": "Service",
            "source_file": "src/Service.php",
            "repo": "b.two",
            "_origin": "ast",
            "source_location": "L10",
        },
    ],
    "links": [
        {"source": "a.one::README.md", "target": "a.one::svc", "relation": "references"},
        {"source": "b.two::readme", "target": "b.two::svc", "relation": "references"},
    ],
}


def test_graph_from_node_link_keeps_every_node_across_repos():
    G = clustering.graph_from_node_link(TWO_REPO_GRAPH)
    assert set(G.nodes()) == {
        "a.one::README.md",
        "b.two::readme",
        "a.one::svc",
        "b.two::svc",
    }
    assert G.nodes["a.one::README.md"]["label"] == "README.md"
    assert G.degree("b.two::readme") == 1


def test_cluster_graph_assigns_every_node_and_sigs_are_stable():
    G = clustering.graph_from_node_link(TWO_REPO_GRAPH)
    communities = clustering.cluster_graph(G)
    assigned = {n for members in communities.values() for n in members}
    assert assigned == set(G.nodes())
    sigs = clustering.membership_sigs(communities)
    assert set(sigs) == set(communities)
    assert clustering.membership_sigs(communities) == sigs


def test_cluster_graph_keeps_ids_stable_against_previous_assignments():
    """Clustering is deterministic, so feeding back its own ids would pass with
    or without the remap. Offset the previous ids: only a real remap reproduces
    them."""
    G = clustering.graph_from_node_link(TWO_REPO_GRAPH)
    first = clustering.cluster_graph(G)
    previous = {n: cid + 5 for cid, members in first.items() for n in members}
    second = clustering.cluster_graph(G, previous_assignments=previous)
    assert set(second) == {cid + 5 for cid in first}
    for cid, members in first.items():
        assert sorted(second[cid + 5]) == sorted(members)


def test_hub_names_are_labels_not_placeholders():
    G = clustering.graph_from_node_link(TWO_REPO_GRAPH)
    communities = clustering.cluster_graph(G)
    names = clustering.hub_names(G, communities)
    assert set(names) == set(communities)
    assert all(name for name in names.values())


def test_llm_names_pins_labeling_concurrency_to_one(monkeypatch):
    """Upstream defaults to 4 concurrent label calls; the backend serves one
    request at a time, so the fan-out only inflates per-call latency (measured
    12.2 s alone vs 41.5 s at 4-way) against the api-timeout budget."""
    captured = {}

    def fake_generate_community_labels(G, communities, **kwargs):
        captured.update(kwargs)
        return ({cid: f"Community {cid}" for cid in communities}, "llm")

    stub = types.ModuleType("graphify.llm")
    stub.generate_community_labels = fake_generate_community_labels
    monkeypatch.setitem(sys.modules, "graphify.llm", stub)

    G = clustering.graph_from_node_link(TWO_REPO_GRAPH)
    communities = clustering.cluster_graph(G)
    # The real one: conftest's autouse fixture has replaced the module attribute.
    _REAL_LLM_NAMES(G, communities, backend="ollama", model="m")

    assert captured["max_concurrency"] == 1
