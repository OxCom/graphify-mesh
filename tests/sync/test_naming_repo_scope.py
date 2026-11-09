"""cluster_scope=repo: each repo is clustered and named on its own."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from graphify_mesh.sync import clustering, naming, naming_state
from graphify_mesh.sync.config import Settings

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
FAKE_GRAPHIFY = FIXTURES_DIR / "fake_graphify" / "graphify"


def _settings(tmp_path: Path, **overrides) -> Settings:
    mesh_root = tmp_path / "mesh"
    overrides.setdefault("ollama_model", "test-model")
    return Settings.from_env(
        mesh_root=mesh_root,
        scan_roots=[tmp_path / "www"],
        registry_path=mesh_root / "bin" / "registry.json",
        graphify_bin=str(FAKE_GRAPHIFY),
        **overrides,
    )


def _node(repo: str | None, name: str) -> dict:
    # "widgets" in the path grounds the default `_Recorder` label.
    node = {
        "id": f"{repo or 'stub'}::{name}",
        "label": name,
        "source_file": f"src/widgets/{name}.py",
    }
    if repo is not None:
        node["repo"] = repo
    return node


def _clique(repo: str, names: list[str]) -> tuple[list[dict], list[dict]]:
    nodes = [_node(repo, n) for n in names]
    links = [
        {"source": f"{repo}::{a}", "target": f"{repo}::{b}", "relation": "calls"}
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    ]
    return nodes, links


def _graph(*parts: tuple[list[dict], list[dict]]) -> dict:
    nodes: list[dict] = []
    links: list[dict] = []
    for part_nodes, part_links in parts:
        nodes += part_nodes
        links += part_links
    return {"directed": False, "multigraph": False, "graph": {}, "nodes": nodes, "links": links}


def _base_graph() -> dict:
    return _graph(_clique("a.one", ["x", "y", "w"]), _clique("b.two", ["m", "n", "o"]))


class _Recorder:
    def __init__(self, monkeypatch, labeler=None):
        self.cluster_calls: list[set[str]] = []
        self.llm_calls: list[dict[int, list[str]]] = []
        original = clustering.cluster_graph

        def cluster_graph(G, resolution=1.0, previous_assignments=None):
            self.cluster_calls.append(set(G.nodes()))
            return original(G, resolution, previous_assignments)

        def llm_names(G, communities, *, backend, model, max_concurrency=1):
            self.llm_calls.append(dict(communities))
            if labeler is not None:
                return labeler(communities), "llm"
            return {cid: f"Named {sorted(m)[0]} widgets" for cid, m in communities.items()}, "llm"

        monkeypatch.setattr(clustering, "cluster_graph", cluster_graph)
        monkeypatch.setattr(clustering, "llm_names", llm_names)

    @property
    def labeled_nodes(self) -> set[str]:
        return {n for call in self.llm_calls for members in call.values() for n in members}


def _run(tmp_path: Path, graph: dict, **settings):
    return naming.run_naming(
        "graphify",
        tmp_path / "naming",
        tmp_path / "home",
        json.loads(json.dumps(graph)),
        _settings(tmp_path, **settings),
        health_check=lambda *a, **k: True,
    )


def _names_by_node(result) -> dict[str, tuple[int, str]]:
    return {n["id"]: (n["community"], n["community_name"]) for n in result.graph_data["nodes"]}


def test_scope_defaults_to_repo_and_is_recorded_in_the_recipe(tmp_path, monkeypatch):
    _Recorder(monkeypatch)
    assert _settings(tmp_path).cluster_scope == "repo"
    _run(tmp_path, _base_graph())
    state = naming_state.load(tmp_path / "naming")
    assert state is not None
    assert state.clustering["scope"] == "repo"
    assert set(state.repos) == {"a.one", "b.two"}


def test_unknown_scope_env_falls_back_to_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHIFY_MESH_CLUSTER_SCOPE", "bogus")
    assert _settings(tmp_path).cluster_scope == "repo"
    monkeypatch.setenv("GRAPHIFY_MESH_CLUSTER_SCOPE", "GLOBAL")
    assert _settings(tmp_path).cluster_scope == "global"


def test_each_repo_is_clustered_alone_with_ids_offset_in_repo_order(tmp_path, monkeypatch):
    rec = _Recorder(monkeypatch)
    result = _run(tmp_path, _base_graph())
    assert result.labeling == naming.LABELING_OK
    assert [sorted(c) for c in rec.cluster_calls] == [
        ["a.one::w", "a.one::x", "a.one::y"],
        ["b.two::m", "b.two::n", "b.two::o"],
    ]
    by_node = _names_by_node(result)
    a_ids = {by_node[f"a.one::{n}"][0] for n in "xyw"}
    b_ids = {by_node[f"b.two::{n}"][0] for n in "mno"}
    assert max(a_ids) < min(b_ids)


def test_unchanged_repo_reuses_partition_and_names_without_clustering_or_llm(tmp_path, monkeypatch):
    rec = _Recorder(monkeypatch)
    first = _run(tmp_path, _base_graph())
    rec.cluster_calls.clear()
    rec.llm_calls.clear()

    # a.one gains a new, separate component; b.two is untouched.
    changed = _graph(
        _clique("a.one", ["x", "y", "w"]),
        _clique("a.one", ["p", "q"]),
        _clique("b.two", ["m", "n", "o"]),
    )
    second = _run(tmp_path, changed)

    assert second.labeling == naming.LABELING_OK
    assert len(rec.cluster_calls) == 1
    assert all(node.startswith("a.one::") for node in rec.cluster_calls[0])
    # Only the new a.one community goes to the LLM.
    assert rec.labeled_nodes == {"a.one::p", "a.one::q"}
    before, after = _names_by_node(first), _names_by_node(second)
    for node in ("b.two::m", "b.two::n", "b.two::o", "a.one::x"):
        assert after[node][1] == before[node][1]


def test_changed_community_keeps_its_name_by_majority_overlap(tmp_path, monkeypatch):
    rec = _Recorder(monkeypatch)
    first = _run(tmp_path, _graph(_clique("a.one", ["x", "y", "w"])))
    rec.llm_calls.clear()

    grown = _graph(_clique("a.one", ["x", "y", "w", "z"]))
    second = _run(tmp_path, grown)

    assert rec.llm_calls == [], "3 of 4 members shared: the name carries without an LLM call"
    name = _names_by_node(first)["a.one::x"][1]
    assert {v[1] for v in _names_by_node(second).values()} == {name}


def _migrations_graph() -> dict:
    """Two Symfony-like repos whose communities both really are migrations."""

    def repo(name: str) -> tuple[list[dict], list[dict]]:
        nodes = [
            {
                "id": f"{name}::{v}",
                "label": v,
                "source_file": f"migrations/{v}.php",
                "repo": name,
            }
            for v in ("Version20240101", "Version20240202", "Version20240303")
        ]
        links = [
            {"source": nodes[i]["id"], "target": nodes[j]["id"], "relation": "calls"}
            for i in range(3)
            for j in range(i + 1, 3)
        ]
        return nodes, links

    return _graph(repo("a.one"), repo("b.two"))


def test_label_shared_across_repos_is_accepted(tmp_path, monkeypatch):
    _Recorder(monkeypatch, labeler=lambda cs: dict.fromkeys(cs, "Doctrine migrations"))
    result = _run(tmp_path, _migrations_graph())

    assert {name for _, name in _names_by_node(result).values()} == {"Doctrine migrations"}
    assert result.labeling == naming.LABELING_OK
    assert result.reason == ""
    state = naming_state.load(tmp_path / "naming")
    assert state is not None
    entries = [e for r in state.repos.values() for e in r.communities.values()]
    assert [(e.provisional, e.ungrounded) for e in entries] == [(False, ""), (False, "")]


def test_ungrounded_label_gets_a_hub_name_and_is_not_asked_again(tmp_path, monkeypatch):
    rec = _Recorder(monkeypatch, labeler=lambda cs: dict.fromkeys(cs, "Doctrine ORM"))
    first = _run(tmp_path, _base_graph())

    assert "Doctrine ORM" not in {name for _, name in _names_by_node(first).values()}
    # Answered by the model, so not an outage: healthy, with the count reported.
    assert first.labeling == naming.LABELING_OK
    assert "2 label(s) rejected as not grounded in member labels, hub names kept" in first.reason
    state = naming_state.load(tmp_path / "naming")
    assert state is not None
    entries = [e for r in state.repos.values() for e in r.communities.values()]
    assert [(e.provisional, e.ungrounded) for e in entries] == [(True, "Doctrine ORM")] * 2

    # Same membership, same recipe: no LLM call, still healthy, same hub names.
    rec.llm_calls.clear()
    second = _run(tmp_path, _base_graph())
    assert rec.llm_calls == []
    assert second.labeling in (naming.LABELING_OK, naming.LABELING_REUSED)
    assert _names_by_node(second) == _names_by_node(first)


def test_ungrounded_community_is_asked_again_when_membership_changes(tmp_path, monkeypatch):
    labels = {"text": "Doctrine ORM"}
    rec = _Recorder(monkeypatch, labeler=lambda cs: dict.fromkeys(cs, labels["text"]))
    _run(tmp_path, _base_graph())

    # b.two grows by one member: signature changes, so only b.two is asked again.
    grown = _graph(_clique("a.one", ["x", "y", "w"]), _clique("b.two", ["m", "n", "o", "q"]))
    labels["text"] = "Widget registry"
    rec.llm_calls.clear()
    result = _run(tmp_path, grown)
    assert rec.labeled_nodes == {"b.two::m", "b.two::n", "b.two::o", "b.two::q"}
    assert _names_by_node(result)["b.two::q"][1] == "Widget registry"
    assert result.labeling == naming.LABELING_OK
    state = naming_state.load(tmp_path / "naming")
    assert state is not None
    assert [e.ungrounded for e in state.repos["b.two"].communities.values()] == [""]


def test_ungrounded_community_is_asked_again_when_labeling_recipe_changes(tmp_path, monkeypatch):
    rec = _Recorder(monkeypatch, labeler=lambda cs: dict.fromkeys(cs, "Doctrine ORM"))
    _run(tmp_path, _base_graph())

    rec.llm_calls.clear()
    _run(tmp_path, _base_graph(), ollama_model="other-model")
    assert {"b.two::m", "b.two::n", "b.two::o"} <= rec.labeled_nodes


def test_ungrounded_label_is_remembered_under_global_scope(tmp_path, monkeypatch):
    rec = _Recorder(monkeypatch, labeler=lambda cs: dict.fromkeys(cs, "Doctrine ORM"))
    first = _run(tmp_path, _base_graph(), cluster_scope="global")
    assert "Doctrine ORM" not in {name for _, name in _names_by_node(first).values()}
    assert first.labeling == naming.LABELING_OK

    rec.llm_calls.clear()
    second = _run(tmp_path, _base_graph(), cluster_scope="global")
    assert rec.llm_calls == []
    assert second.labeling in (naming.LABELING_OK, naming.LABELING_REUSED)


def _mixed_labeler(cs):
    # a.one gets a grounded label, b.two one the grounding check refuses.
    return {
        cid: "Doctrine ORM" if any(n.startswith("b.two") for n in m) else "Widget registry"
        for cid, m in cs.items()
    }


@pytest.mark.parametrize("scope", ["repo", "global"])
def test_grounding_rule_change_re_asks_only_ungrounded_entries(tmp_path, monkeypatch, scope):
    rec = _Recorder(monkeypatch, labeler=_mixed_labeler)
    first = _run(tmp_path, _base_graph(), cluster_scope=scope)
    assert _names_by_node(first)["a.one::x"][1] == "Widget registry"

    rec.llm_calls.clear()
    monkeypatch.setattr(naming, "GROUNDING_VERSION", naming.GROUNDING_VERSION + 1)
    second = _run(tmp_path, _base_graph(), cluster_scope=scope)
    assert rec.labeled_nodes == {"b.two::m", "b.two::n", "b.two::o"}
    assert _names_by_node(second)["a.one::x"][1] == "Widget registry"

    # Answered under the new rules: not asked a third time.
    rec.llm_calls.clear()
    _run(tmp_path, _base_graph(), cluster_scope=scope)
    assert rec.llm_calls == []


def test_state_without_grounding_version_keeps_grounded_names(tmp_path, monkeypatch):
    rec = _Recorder(monkeypatch, labeler=_mixed_labeler)
    _run(tmp_path, _base_graph())
    path = tmp_path / "naming" / naming_state.STATE_FILENAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["labeling"].pop("grounding") == naming.GROUNDING_VERSION
    path.write_text(json.dumps(raw), encoding="utf-8")

    # A state written before the version existed: no full relabel, only the
    # ungrounded entries are asked again.
    rec.llm_calls.clear()
    _run(tmp_path, _base_graph())
    assert rec.labeled_nodes == {"b.two::m", "b.two::n", "b.two::o"}


def test_legacy_rejected_entry_is_asked_again(tmp_path, monkeypatch):
    rec = _Recorder(monkeypatch)
    _run(tmp_path, _base_graph())
    path = tmp_path / "naming" / naming_state.STATE_FILENAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    for entry in raw["repos"]["b.two"]["communities"].values():
        entry.update(provisional=False, rejected=True)
    for entry in raw["communities"].values():
        entry.update(provisional=False, rejected=True)
    path.write_text(json.dumps(raw), encoding="utf-8")

    rec.llm_calls.clear()
    _run(tmp_path, _base_graph())
    assert {"b.two::m", "b.two::n", "b.two::o"} <= rec.labeled_nodes


@pytest.mark.parametrize(
    ("label", "members", "grounded"),
    [
        ("Helm ingress templates", ["ingress.yaml", "templates/ingress.yaml", "tls"], True),
        ("Messenger routing", ["config/packages/messenger.yaml", "routing"], True),
        ("Doctrine migrations", ["Version20240101", "migrations/Version20240101.php"], True),
        ("Authentication", ["AuthController", "src/Controller/AuthController.php"], True),
        ("HTTP client retries", ["HttpClient", "retry.go"], True),
        ("TS Config", ["tsconfig.json", "compilerOptions"], True),
        ("TypeScript Configuration", ["tsconfig.json", "compilerOptions"], True),
        ("UI", ["UI", "src/ui/index.ts"], True),
        ("CC Module", ["Cc", "src/Cc.php"], True),
        ("R3.0.0 Version", ["R3.0.0", "CHANGELOG.md"], True),
        ("R3.4.x", ["R3.0.0", "CHANGELOG.md"], True),
        ("Data Transfer Object", ["Dto", "src/Dto/UserDto.php"], True),
        ("Environment Setup", ["bootstrap.php", "Dotenv"], True),
        ("System Configuration", ["ExternalLinks", "src/Handler/System/ExternalLinks.php"], True),
        ("Monolog Logger", ["messenger.yaml", "config/packages/messenger.yaml"], False),
        ("News Comments", ["CalendarEvent", "src/Model/CalendarEvent.php"], False),
        ("Doctrine ORM", ["UserRepo", "pkg/types/user.go", "OrderType"], False),
        ("Misc utilities", ["UserRepo", "pkg/types/user.go"], False),
        # A three-letter member word is a verb or abbreviation, not a stem.
        ("Settings Management", ["setFoo", "src/Foo.php"], False),
        ("Integration Tests", ["int_parse", "src/Parse.php"], False),
    ],
)
def test_grounding_check_on_realistic_labels(label, members, grounded):
    assert naming._is_grounded(label, members) is grounded


def test_ollama_failure_still_gives_provisional_names_and_degrades(tmp_path, monkeypatch):
    _Recorder(monkeypatch)
    result = naming.run_naming(
        "graphify",
        tmp_path / "naming",
        tmp_path / "home",
        json.loads(json.dumps(_base_graph())),
        _settings(tmp_path),
        health_check=lambda *a, **k: False,
    )
    assert result.labeling == naming.LABELING_DEGRADED
    state = naming_state.load(tmp_path / "naming")
    assert state is not None
    flags = [e.provisional for r in state.repos.values() for e in r.communities.values()]
    assert flags and all(flags)


def test_stub_nodes_join_the_smallest_neighbouring_repo():
    G = nx.Graph()
    G.add_node("b.two::m", repo="b.two")
    G.add_node("a.one::x", repo="a.one")
    G.add_node("s1")  # touches both repos
    G.add_node("s2")  # reaches a repo only through s1
    G.add_node("lonely")
    G.add_edges_from([("s1", "b.two::m"), ("s1", "a.one::x"), ("s2", "s1")])

    groups = clustering.repo_groups(G)

    assert groups == {
        clustering.UNATTRIBUTED_GROUP: ["lonely"],
        "a.one": ["a.one::x", "s1", "s2"],
        "b.two": ["b.two::m"],
    }


def test_state_keeps_only_current_communities_and_repos(tmp_path, monkeypatch):
    _Recorder(monkeypatch)
    _run(
        tmp_path,
        _graph(
            _clique("a.one", ["x", "y", "w"]),
            _clique("a.one", ["p", "q"]),
            _clique("b.two", ["m", "n", "o"]),
        ),
    )
    state = naming_state.load(tmp_path / "naming")
    assert state is not None
    old_sigs = {e.sig for e in state.communities.values()}

    _run(tmp_path, _graph(_clique("a.one", ["x", "y", "w"])))
    state = naming_state.load(tmp_path / "naming")
    assert state is not None

    assert set(state.repos) == {"a.one"}
    assert len(state.communities) == 1
    assert set(state.assignments) == {"a.one::x", "a.one::y", "a.one::w"}
    current = {e.sig for e in state.communities.values()}
    assert {e.sig for e in state.repos["a.one"].communities.values()} == current
    assert len(old_sigs - current) == 2, "the dropped communities leave the state"


def test_flipping_scope_relabels_once(tmp_path, monkeypatch):
    rec = _Recorder(monkeypatch)
    _run(tmp_path, _base_graph(), cluster_scope="global")
    state = naming_state.load(tmp_path / "naming")
    assert state is not None
    assert "scope" not in state.clustering, "global keeps the pre-scope recipe"
    assert state.repos == {}

    rec.llm_calls.clear()
    _run(tmp_path, _base_graph())
    assert len(rec.labeled_nodes) == 6, "a scope flip relabels every community once"

    rec.llm_calls.clear()
    rec.cluster_calls.clear()
    third = _run(tmp_path, _base_graph())
    assert third.labeling == naming.LABELING_REUSED
    assert rec.llm_calls == [] and rec.cluster_calls == []
