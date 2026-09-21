from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphify_mesh.sync import backend, naming, naming_state, publish
from graphify_mesh.sync.config import Settings
from graphify_mesh.sync.pipeline import run
from graphify_mesh.sync.validate import PLACEHOLDER_RE

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
FAKE_GRAPHIFY = FIXTURES_DIR / "fake_graphify" / "graphify"


def _merged_graph() -> dict:
    return {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": [
            {
                "id": "example-org.styleguide::a1",
                "label": "AlphaClass",
                "repo_tag": "example-org.styleguide",
                "community": 0,
                "community_name": "Alpha Domain",
            },
            {
                "id": "example-org.styleguide::a2",
                "label": "alpha_helper",
                "repo_tag": "example-org.styleguide",
                "community": 0,
                "community_name": "Alpha Domain",
            },
            {
                "id": "example-org.services::b1",
                "label": "BetaService",
                "repo_tag": "example-org.services",
                "community": 0,
                "community_name": "Beta Domain",
            },
            {
                "id": "example-org.services::b2",
                "label": "beta_util",
                "repo_tag": "example-org.services",
                "community": 0,
                "community_name": "Beta Domain",
            },
        ],
        "links": [
            {
                "source": "example-org.styleguide::a1",
                "target": "example-org.styleguide::a2",
                "relation": "calls",
            },
            {
                "source": "example-org.services::b1",
                "target": "example-org.services::b2",
                "relation": "calls",
            },
        ],
    }


def _settings(tmp_path: Path, **overrides) -> Settings:
    mesh_root = tmp_path / "mesh"
    return Settings.from_env(
        mesh_root=mesh_root,
        scan_roots=[tmp_path / "www"],
        registry_path=mesh_root / "bin" / "registry.json",
        graphify_bin=str(FAKE_GRAPHIFY),
        **overrides,
    )


# ---------------------------------------------------------------------------
# strip_project_community_attrs
# ---------------------------------------------------------------------------


def test_strip_project_community_attrs_removes_both_keys():
    graph = _merged_graph()
    stripped = naming.strip_project_community_attrs(graph)
    for node in stripped["nodes"]:
        assert "community" not in node
        assert "community_name" not in node
    # in-place contract: same object returned, original graph is mutated too
    assert stripped is graph
    assert "community_name" not in graph["nodes"][0]


def test_strip_mutates_in_place_and_returns_same_object():
    graph = {"nodes": [{"id": "n1", "community": 3, "community_name": "X", "label": "L"}]}
    result = naming.strip_project_community_attrs(graph)
    assert result is graph
    assert "community" not in graph["nodes"][0]
    assert "community_name" not in graph["nodes"][0]
    assert graph["nodes"][0]["label"] == "L"


def test_strip_skips_non_dict_nodes():
    graph = {"nodes": ["junk", {"id": "n1", "community": 1}]}
    naming.strip_project_community_attrs(graph)
    assert graph["nodes"][0] == "junk"
    assert "community" not in graph["nodes"][1]


# ---------------------------------------------------------------------------
# run_naming unit tests (bypassing pipeline.run for direct control)
# ---------------------------------------------------------------------------

TWO_NODE_GRAPH = {
    "directed": False,
    "multigraph": False,
    "graph": {},
    "nodes": [
        {"id": "a.one::x", "label": "X", "source_file": "src/x.py", "repo": "a.one"},
        {"id": "a.one::y", "label": "Y", "source_file": "src/y.py", "repo": "a.one"},
    ],
    "links": [{"source": "a.one::x", "target": "a.one::y", "relation": "calls"}],
}

# The live incident's shape, repeated here rather than imported from
# tests/sync/test_clustering.py: two repos sharing (source_file, label), one
# AST-origin and one semantic-origin.
TWO_REPO_COLLAPSE_GRAPH = {
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


def test_degraded_when_backend_unhealthy_still_clusters_and_names_by_hub(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "graphify_mesh.sync.clustering.llm_names",
        lambda *a, **k: pytest.fail("LLM must not be called when the backend is unhealthy"),
    )
    result = naming.run_naming(
        "graphify",
        tmp_path / "naming",
        tmp_path / "home",
        json.loads(json.dumps(TWO_NODE_GRAPH)),
        _settings(tmp_path, ollama_model="test-model"),
        health_check=lambda *a, **k: False,
    )
    assert result.labeling == naming.LABELING_DEGRADED
    nodes = {n["id"]: n for n in result.graph_data["nodes"]}
    assert all(n.get("community") is not None for n in nodes.values())
    assert all(n.get("community_name") for n in nodes.values())
    state = naming_state.load(tmp_path / "naming")
    assert state is not None
    assert all(e.provisional for e in state.communities.values())


def test_second_run_with_unchanged_graph_reuses_without_clustering(tmp_path, monkeypatch):
    """Reuse requires a FULLY named state, so the first run here must succeed."""
    monkeypatch.setattr(
        "graphify_mesh.sync.clustering.llm_names",
        lambda G, communities, **k: ({cid: f"Named {cid}" for cid in communities}, "llm"),
    )
    settings = _settings(tmp_path, ollama_model="test-model")
    first = naming.run_naming(
        "graphify",
        tmp_path / "naming",
        tmp_path / "home",
        json.loads(json.dumps(TWO_NODE_GRAPH)),
        settings,
        health_check=lambda *a, **k: True,
    )
    assert first.labeling == naming.LABELING_OK

    monkeypatch.setattr(
        "graphify_mesh.sync.clustering.cluster_graph",
        lambda *a, **k: pytest.fail("clustering must not run when reuse is safe"),
    )
    again = naming.run_naming(
        "graphify",
        tmp_path / "naming",
        tmp_path / "home",
        json.loads(json.dumps(TWO_NODE_GRAPH)),
        settings,
        health_check=lambda *a, **k: True,
    )
    assert again.labeling == naming.LABELING_REUSED
    assert all(n.get("community_name") for n in again.graph_data["nodes"])
    # The reused NamingResult still carries backend/backend_check — pipeline.py's
    # manifest writer reads these regardless of which labeling outcome came back.
    assert again.backend == first.backend
    assert again.backend_check == first.backend_check


def test_provisional_names_are_retried_once_the_backend_recovers(tmp_path, monkeypatch):
    """The outage run must not freeze hub names: the next healthy run relabels
    exactly the provisional communities, on an UNCHANGED graph."""
    settings = _settings(tmp_path, ollama_model="test-model")
    naming.run_naming(
        "graphify",
        tmp_path / "naming",
        tmp_path / "home",
        json.loads(json.dumps(TWO_NODE_GRAPH)),
        settings,
        health_check=lambda *a, **k: False,
    )
    state = naming_state.load(tmp_path / "naming")
    assert state is not None and all(e.provisional for e in state.communities.values())

    asked = []

    def fake_llm(G, communities, **kwargs):
        asked.append(sorted(communities))
        return {cid: f"Named {cid}" for cid in communities}, "llm"

    monkeypatch.setattr("graphify_mesh.sync.clustering.llm_names", fake_llm)
    recovered = naming.run_naming(
        "graphify",
        tmp_path / "naming",
        tmp_path / "home",
        json.loads(json.dumps(TWO_NODE_GRAPH)),
        settings,
        health_check=lambda *a, **k: True,
    )
    assert recovered.labeling == naming.LABELING_OK
    assert asked, "the recovered run must ask the LLM for the provisional communities"
    after = naming_state.load(tmp_path / "naming")
    assert after is not None
    assert not any(e.provisional for e in after.communities.values())


def test_model_change_invalidates_carried_names(tmp_path, monkeypatch):
    calls = []

    def fake_llm(G, communities, **kwargs):
        calls.append(kwargs.get("model"))
        return {cid: f"Named {cid}" for cid in communities}, "llm"

    monkeypatch.setattr("graphify_mesh.sync.clustering.llm_names", fake_llm)

    def graph():
        return json.loads(json.dumps(TWO_NODE_GRAPH))

    naming.run_naming(
        "graphify",
        tmp_path / "naming",
        tmp_path / "home",
        graph(),
        _settings(tmp_path, ollama_model="model-a"),
        health_check=lambda *a, **k: True,
    )
    calls.clear()
    naming.run_naming(
        "graphify",
        tmp_path / "naming",
        tmp_path / "home",
        graph(),
        _settings(tmp_path, ollama_model="model-b"),
        health_check=lambda *a, **k: True,
    )
    assert calls == ["model-b"], "a model change must relabel, not reuse the old model's names"


def test_reuse_refused_when_state_misses_a_node(tmp_path, monkeypatch):
    """An assignment map that does not cover every node must not be reused:
    validate_community_names skips nodes with no community, so the gap would
    publish silently."""
    monkeypatch.setattr(
        "graphify_mesh.sync.clustering.llm_names",
        lambda G, communities, **k: ({cid: f"Named {cid}" for cid in communities}, "llm"),
    )
    settings = _settings(tmp_path, ollama_model="test-model")
    naming.run_naming(
        "graphify",
        tmp_path / "naming",
        tmp_path / "home",
        json.loads(json.dumps(TWO_NODE_GRAPH)),
        settings,
        health_check=lambda *a, **k: True,
    )
    state = naming_state.load(tmp_path / "naming")
    assert state is not None
    state.assignments.pop(next(iter(state.assignments)))
    naming_state.save(tmp_path / "naming", state)

    result = naming.run_naming(
        "graphify",
        tmp_path / "naming",
        tmp_path / "home",
        json.loads(json.dumps(TWO_NODE_GRAPH)),
        settings,
        health_check=lambda *a, **k: True,
    )
    assert result.labeling != naming.LABELING_REUSED
    assert all(n.get("community") is not None for n in result.graph_data["nodes"])


def test_llm_names_are_requested_only_for_communities_without_a_name(tmp_path, monkeypatch):
    calls = []

    def fake_llm(G, communities, *, backend, model):
        calls.append(sorted(communities))
        return {cid: f"Named {cid}" for cid in communities}, "llm"

    monkeypatch.setattr("graphify_mesh.sync.clustering.llm_names", fake_llm)
    graph = json.loads(json.dumps(TWO_NODE_GRAPH))
    graph["nodes"] += [
        {"id": "a.one::p", "label": "P", "source_file": "src/p.py", "repo": "a.one"},
        {"id": "a.one::q", "label": "Q", "source_file": "src/q.py", "repo": "a.one"},
    ]
    graph["links"].append({"source": "a.one::p", "target": "a.one::q", "relation": "calls"})
    first = naming.run_naming(
        "graphify",
        tmp_path / "naming",
        tmp_path / "home",
        graph,
        _settings(tmp_path, ollama_model="test-model"),
        health_check=lambda *a, **k: True,
    )
    assert first.labeling == naming.LABELING_OK
    assert calls and calls[0]
    state_after_first = naming_state.load(tmp_path / "naming")
    assert state_after_first is not None
    stable_sigs_before = {
        entry.sig
        for cid, entry in state_after_first.communities.items()
        if {"a.one::p", "a.one::q"}
        == {
            node
            for node, assigned in state_after_first.assignments.items()
            if str(assigned) == str(cid)
        }
    }
    assert stable_sigs_before, "the disconnected component must be its own community"

    # Second component, deliberately disconnected: its membership cannot change
    # when the first component grows, so its name must be carried, not re-asked.
    # Same graph as the first run — including the disconnected p/q component —
    # plus one node on the first component. Dropping p/q here would make the
    # "unchanged community keeps its name" assertion vacuous.
    changed = json.loads(json.dumps(graph))
    changed["nodes"].append(
        {"id": "a.one::z", "label": "Z", "source_file": "src/z.py", "repo": "a.one"}
    )
    changed["links"].append({"source": "a.one::y", "target": "a.one::z", "relation": "calls"})
    calls.clear()
    second = naming.run_naming(
        "graphify",
        tmp_path / "naming",
        tmp_path / "home",
        changed,
        _settings(tmp_path, ollama_model="test-model"),
        health_check=lambda *a, **k: True,
    )
    assert calls, "a changed community must be relabeled"
    asked_cids = set(calls[0])
    state = naming_state.load(tmp_path / "naming")
    assert state is not None
    untouched = {cid for cid, entry in state.communities.items() if entry.sig in stable_sigs_before}
    assert untouched, "the unchanged component must still be a community in run 2"
    assert not (asked_cids & {int(cid) for cid in untouched}), (
        "an unchanged community must keep its name without an LLM call"
    )
    assert {"a.one::p", "a.one::q"} <= {n["id"] for n in second.graph_data["nodes"]}
    assert second.labeling == naming.LABELING_OK


def test_cross_repo_collapse_cannot_happen_in_the_naming_stage(tmp_path):
    """The live incident's shape: two repos, same (source_file, label), one AST
    one semantic. graphify's builder collapses them; this stage must not.

    health_check=True on purpose: the degraded path returns the input untouched,
    so it would pass even against the old CLI-based stage and prove nothing.
    """
    graph = json.loads(json.dumps(TWO_REPO_COLLAPSE_GRAPH))
    result = naming.run_naming(
        "graphify",
        tmp_path / "naming",
        tmp_path / "home",
        graph,
        _settings(tmp_path, ollama_model="test-model"),
        health_check=lambda *a, **k: True,
    )
    ids = {n["id"] for n in result.graph_data["nodes"]}
    assert ids == {"a.one::README.md", "b.two::readme", "a.one::svc", "b.two::svc"}
    assert all(n.get("community_name") for n in result.graph_data["nodes"])


def test_changed_graph_takes_the_full_path_instead_of_reuse(tmp_path):
    """Replaces the fingerprint-sidecar version of this test: a graph that
    changed since the state file was written must cluster again, not reuse."""
    settings = _settings(tmp_path, ollama_model="test-model")
    first = naming.run_naming(
        "graphify",
        tmp_path / "naming",
        tmp_path / "home",
        json.loads(json.dumps(TWO_NODE_GRAPH)),
        settings,
        health_check=lambda *a, **k: True,
    )
    assert first.labeling == naming.LABELING_OK

    changed = json.loads(json.dumps(TWO_NODE_GRAPH))
    changed["nodes"].append(
        {"id": "a.one::z", "label": "Z", "source_file": "src/z.py", "repo": "a.one"}
    )
    changed["links"].append({"source": "a.one::y", "target": "a.one::z", "relation": "calls"})

    second = naming.run_naming(
        "graphify",
        tmp_path / "naming",
        tmp_path / "home",
        changed,
        settings,
        health_check=lambda *a, **k: True,
    )
    assert second.labeling == naming.LABELING_OK
    assert second.labeling != naming.LABELING_REUSED
    assert {n["id"] for n in second.graph_data["nodes"]} == {
        "a.one::x",
        "a.one::y",
        "a.one::z",
    }


def test_backend_mismatch_raises_before_anything_clusters(tmp_path, monkeypatch):
    """The backend is resolved in process now, so the mismatch is forced by
    making Leiden importable here rather than by building a fake interpreter
    behind a fake graphify_bin."""
    monkeypatch.setattr(backend, "_graspologic_importable", lambda: True)
    monkeypatch.setattr(
        "graphify_mesh.sync.clustering.cluster_graph",
        lambda *a, **k: pytest.fail("the pin check must fail before clustering"),
    )

    with pytest.raises(backend.BackendMismatchError):
        naming.run_naming(
            "graphify",
            tmp_path / "naming",
            tmp_path / "home",
            json.loads(json.dumps(TWO_NODE_GRAPH)),
            _settings(tmp_path, ollama_model="test-model"),
            health_check=lambda *a, **k: True,
        )
    assert not (tmp_path / "naming" / naming_state.STATE_FILENAME).exists()


def test_bind_upstream_backend_points_graphify_at_the_probed_endpoint(tmp_path):
    """The mesh probes GRAPHIFY_MESH_OLLAMA_BASE_URL while graphify labels
    against its own cached BACKENDS entry. Nothing else makes them agree."""
    settings = _settings(
        tmp_path, ollama_base_url="http://127.0.0.1:65500/v1", ollama_api_timeout=42.0
    )
    env: dict[str, str] = {}
    effective = naming.bind_upstream_backend(settings, env=env)

    from graphify.llm import BACKENDS

    assert effective == "http://127.0.0.1:65500/v1"
    assert BACKENDS["ollama"]["base_url"] == "http://127.0.0.1:65500/v1"
    assert env["GRAPHIFY_API_TIMEOUT"] == "42.0"
    # Neither the base URL nor the model travels through the environment: the
    # child env allowlist forwards the OLLAMA_ prefix to every later child.
    assert "OLLAMA_BASE_URL" not in env
    assert "OLLAMA_MODEL" not in env


def test_labeling_skipped_over_https_when_tls_mode_is_relaxed(tmp_path, monkeypatch):
    """graphify builds its own client, so this package's relaxed TLS context
    cannot reach it — a certificate failure there would surface as placeholder
    names rather than an error."""
    monkeypatch.setenv("GRAPHIFY_MESH_TLS_MODE", "relaxed")
    monkeypatch.setattr(
        "graphify_mesh.sync.clustering.llm_names",
        lambda *a, **k: pytest.fail("labeling must not run over an unverifiable https chain"),
    )
    settings = _settings(tmp_path, ollama_base_url="https://ollama.invalid/v1")
    result = naming.run_naming(
        "graphify",
        tmp_path / "naming",
        tmp_path / "home",
        json.loads(json.dumps(TWO_NODE_GRAPH)),
        settings,
        health_check=lambda *a, **k: True,
    )
    assert result.labeling == naming.LABELING_DEGRADED
    assert all(n.get("community_name") for n in result.graph_data["nodes"])


def test_degraded_run_publishes_its_own_names_not_the_previous_generation(tmp_path, monkeypatch):
    """A degraded naming run keeps the names IT produced. The old
    restore-from-last-published step would mix two clusterings in one graph."""
    settings = _settings(tmp_path, ollama_model="test-model")
    result = naming.run_naming(
        "graphify",
        tmp_path / "naming",
        tmp_path / "home",
        json.loads(json.dumps(TWO_NODE_GRAPH)),
        settings,
        health_check=lambda *a, **k: False,
    )
    assert result.labeling == naming.LABELING_DEGRADED
    names = {n["id"]: n["community_name"] for n in result.graph_data["nodes"]}
    by_community: dict[int, set[str]] = {}
    for node in result.graph_data["nodes"]:
        by_community.setdefault(node["community"], set()).add(node["community_name"])
    assert all(len(v) == 1 for v in by_community.values()), (
        f"one community must carry exactly one name, got {by_community} ({names})"
    )
    assert not hasattr(naming, "restore_last_global_community_names")


# ---------------------------------------------------------------------------
# End-to-end pipeline tests
# ---------------------------------------------------------------------------


def test_pipeline_strip_then_relabel_no_per_project_leakage(env):
    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.add_repo(
        "example-org.services",
        "example-org",
        "services",
        "services.example-org.dev.lo",
        "repo_b.json",
    )
    env.write_registry()
    settings = env.settings(ollama_health_check=lambda *a, **kw: True)

    report = run(settings)

    assert report.published
    assert report.labeling == "ok"
    assert report.clustering_backend == "louvain"

    graph = json.loads(
        (settings.global_dir / "current" / "global-graph.json").read_text(encoding="utf-8")
    )
    names = {n.get("community_name") for n in graph["nodes"]}
    # None of the original per-project names survived into the published output.
    assert "Alpha Domain" not in names
    assert "Beta Domain" not in names
    for node in graph["nodes"]:
        assert node.get("community_name")  # every clustered node got a real name
        # PLACEHOLDER_RE, not startswith("Community "): the latter would also
        # reject a legitimate name such as "Community Management".
        assert not PLACEHOLDER_RE.match(node["community_name"])

    manifest = json.loads(
        (settings.global_dir / "current" / "generation-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["clustering_backend"] == "louvain"
    assert manifest["labeling"] == "ok"


def test_previous_global_graph_not_loaded_on_healthy_naming(env, monkeypatch):
    """The previously published global graph (a full 38K-node JSON parse in
    production) must only be loaded when naming degrades — the healthy path
    never needs it."""
    calls = []
    real = publish.read_current_global_graph

    def counting(global_dir):
        calls.append(global_dir)
        return real(global_dir)

    monkeypatch.setattr(publish, "read_current_global_graph", counting)

    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.write_registry()
    settings = env.settings(ollama_health_check=lambda *a, **kw: True)

    report = run(settings)

    assert report.published
    assert report.labeling == "ok"
    assert calls == []


def test_pipeline_degraded_mode_no_leakage_and_publishes_stage_names(env):
    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.add_repo(
        "example-org.services",
        "example-org",
        "services",
        "services.example-org.dev.lo",
        "repo_b.json",
    )
    env.write_registry()

    # First (healthy) run establishes a published generation and a naming
    # state file; the state file is what carries names across the outage.
    settings_healthy = env.settings(ollama_health_check=lambda *a, **kw: True)
    first = run(settings_healthy)
    assert first.published
    assert first.labeling == "ok"

    # Touch a repo so a new run is actually triggered, then run degraded.
    root_a = Path([r["root"] for r in env._repos if r["repo_id"] == "example-org.styleguide"][0])
    (root_a / "touched.py").write_text("# touch\n", encoding="utf-8")

    settings_degraded = env.settings(ollama_health_check=lambda *a, **kw: False)
    second = run(settings_degraded)

    assert second.published
    assert second.labeling == "degraded"

    second_graph = json.loads(
        (env.mesh_root / "graphify" / "global" / "current" / "global-graph.json").read_text(
            encoding="utf-8"
        )
    )
    names_by_id = {n["id"]: n.get("community_name") for n in second_graph["nodes"]}

    # The degraded run publishes the names its own naming stage produced:
    # every node named, and one community carrying exactly one name. Restoring
    # from the previous generation would break the second property as soon as
    # two old communities merged under the new clustering.
    assert all(names_by_id.values())
    by_community: dict[int, set[str]] = {}
    for node in second_graph["nodes"]:
        by_community.setdefault(node["community"], set()).add(node["community_name"])
    assert all(len(v) == 1 for v in by_community.values()), by_community
    # No per-project name ever leaked in, degraded or not.
    all_names = set(names_by_id.values())
    assert "Alpha Domain" not in all_names
    assert "Beta Domain" not in all_names


def test_pipeline_backend_mismatch_blocks_publish_end_to_end(env, monkeypatch):
    """Forced backend mismatch (deliverable 1/2) must hard-fail the naming
    stage and propagate all the way out of pipeline.run() uncaught — never
    silently degrade, never publish. Mirrors the existing
    test_publish_failure_between_write_and_flip_leaves_current_untouched
    pattern (pytest.raises around run()); current must stay untouched.

    The backend now resolves in process, so Leiden is made importable here
    instead of behind a fake graphify_bin; every subcommand still runs against
    the ordinary fake stub."""
    monkeypatch.setattr(backend, "_graspologic_importable", lambda: True)

    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.write_registry()

    settings = Settings.from_env(
        mesh_root=env.mesh_root,
        scan_roots=env.scan_roots,
        registry_path=env.registry_path,
        graphify_bin=str(FAKE_GRAPHIFY),
        ollama_health_check=lambda *a, **kw: True,
    )

    with pytest.raises(backend.BackendMismatchError):
        run(settings)

    assert not (settings.global_dir / "current").exists()


def test_pipeline_degraded_mode_never_invokes_cluster_only_or_label(env):
    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.write_registry()
    settings = env.settings(ollama_health_check=lambda *a, **kw: False)

    report = run(settings)

    assert report.published
    assert report.labeling == "degraded"
    call_log = env.read_call_log()
    assert not any(e["cmd"] in ("cluster-only", "label") for e in call_log)


BEFORE = {
    "nodes": [
        {"id": "a::x", "label": "X", "repo": "a"},
        {"id": "b::y", "label": "Y", "repo": "b"},
    ],
    "links": [{"source": "a::x", "target": "b::y", "relation": "references"}],
}


def _annotated():
    return {
        "nodes": [
            {"id": "a::x", "label": "X", "repo": "a", "community": 0, "community_name": "Core"},
            {"id": "b::y", "label": "Y", "repo": "b", "community": 0, "community_name": "Core"},
        ],
        "links": [{"source": "a::x", "target": "b::y", "relation": "references"}],
    }


def test_guard_accepts_community_annotations():
    naming.assert_only_community_attrs_added(naming.structure_snapshot(BEFORE), _annotated())


def test_guard_rejects_a_dropped_node():
    after = _annotated()
    after["nodes"] = after["nodes"][:1]
    with pytest.raises(naming.NamingIntegrityError, match="node id set"):
        naming.assert_only_community_attrs_added(naming.structure_snapshot(BEFORE), after)


def test_guard_rejects_a_rewritten_label():
    after = _annotated()
    after["nodes"][0]["label"] = "a/X"
    with pytest.raises(naming.NamingIntegrityError, match="attribute"):
        naming.assert_only_community_attrs_added(naming.structure_snapshot(BEFORE), after)


def test_guard_rejects_a_rewired_edge():
    after = _annotated()
    after["links"] = [{"source": "a::x", "target": "a::x", "relation": "references"}]
    with pytest.raises(naming.NamingIntegrityError, match="edge"):
        naming.assert_only_community_attrs_added(naming.structure_snapshot(BEFORE), after)


# ---------------------------------------------------------------------------
# legacy sidecar retirement
# ---------------------------------------------------------------------------


def test_legacy_sidecars_are_removed_after_a_successful_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "graphify_mesh.sync.clustering.llm_names",
        lambda G, communities, **k: ({cid: f"Named {cid}" for cid in communities}, "llm"),
    )
    naming_dir = tmp_path / "naming"
    out_dir = naming_dir / "graphify-out"
    out_dir.mkdir(parents=True)
    (out_dir / ".graphify_labels.json").write_text(json.dumps({"0": "Auth"}))
    (out_dir / ".graphify_labels.json.sig").write_text(json.dumps({"0": "s0"}))
    (out_dir / "graph.json").write_text(json.dumps(TWO_NODE_GRAPH))
    (out_dir / "GRAPH_REPORT.md").write_text("stale")

    result = naming.run_naming(
        "graphify",
        naming_dir,
        tmp_path / "home",
        json.loads(json.dumps(TWO_NODE_GRAPH)),
        _settings(tmp_path, ollama_model="test-model"),
        health_check=lambda *a, **k: True,
    )
    assert result.labeling == naming.LABELING_OK
    for name in (
        ".graphify_labels.json",
        ".graphify_labels.json.sig",
        "graph.json",
        "GRAPH_REPORT.md",
    ):
        assert not (out_dir / name).exists(), f"{name} should be gone after a successful run"


def test_sidecars_survive_a_degraded_run(tmp_path):
    """A degraded run has not proven the new state file is good, so the fallback stays."""
    naming_dir = tmp_path / "naming"
    out_dir = naming_dir / "graphify-out"
    out_dir.mkdir(parents=True)
    (out_dir / ".graphify_labels.json").write_text(json.dumps({"0": "Auth"}))
    (out_dir / ".graphify_labels.json.sig").write_text(json.dumps({"0": "s0"}))

    naming.run_naming(
        "graphify",
        naming_dir,
        tmp_path / "home",
        json.loads(json.dumps(TWO_NODE_GRAPH)),
        _settings(tmp_path, ollama_model="test-model"),
        health_check=lambda *a, **k: False,
    )
    assert (out_dir / ".graphify_labels.json").exists()
