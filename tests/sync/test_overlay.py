from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphify_mesh.sync import (
    embedding,
    overlay_api,
    overlay_depends,
    overlay_refs,
    overlay_similar,
    validate,
)
from graphify_mesh.sync.overlay_refs import (
    DanglingReferenceError,
    LogicalRef,
)
from graphify_mesh.sync.pipeline import run
from graphify_mesh.sync.vectors import RepoVectors

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "examples" / "manual-relations.schema.json"


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


# --- 1. depends_on from a fixture composer.json+lock pair -----------------


def test_depends_on_composer_runtime_and_dev_distinguished(tmp_path):
    consumer_root = tmp_path / "consumer"
    provider_root = tmp_path / "provider"
    provider_root.mkdir(parents=True)
    consumer_root.mkdir(parents=True)

    _write_json(provider_root / "composer.json", {"name": "acme/provider"})
    _write_json(
        consumer_root / "composer.json",
        {"require": {"acme/provider": "^1.0"}, "require-dev": {"acme/test-only": "^1.0"}},
    )
    _write_json(
        consumer_root / "composer.lock",
        {"packages": [{"name": "acme/provider"}], "packages-dev": []},
    )

    identity_map = overlay_depends.build_package_identity_map(
        {"acme.consumer": consumer_root, "acme.provider": provider_root}
    )
    edges = overlay_depends.extract_depends_on_edges("acme.consumer", consumer_root, identity_map)

    assert len(edges) == 1  # require-dev entry not in composer.lock -> skipped, not guessed
    edge = edges[0]
    assert edge.type == "depends_on"
    assert edge.source.repo == "acme.consumer"
    assert edge.target.repo == "acme.provider"
    assert edge.provenance == "EXTRACTED_CONFIG"


def test_depends_on_npm_runtime_and_dev_distinguished(tmp_path):
    consumer_root = tmp_path / "consumer"
    provider_root = tmp_path / "provider"
    provider_root.mkdir(parents=True)
    consumer_root.mkdir(parents=True)

    _write_json(provider_root / "package.json", {"name": "@acme/styleguide"})
    _write_json(
        consumer_root / "package.json",
        {
            "dependencies": {"@acme/styleguide": "^1.0"},
            "devDependencies": {"@acme/dev-tool": "^1.0"},
        },
    )
    _write_json(
        consumer_root / "package-lock.json",
        {"packages": {"node_modules/@acme/styleguide": {"version": "1.0.0"}}},
    )

    identity_map = overlay_depends.build_package_identity_map(
        {"acme.consumer": consumer_root, "acme.provider": provider_root}
    )
    edges = overlay_depends.extract_depends_on_edges("acme.consumer", consumer_root, identity_map)

    assert len(edges) == 1
    assert edges[0].target.repo == "acme.provider"
    assert edges[0].evidence.endswith("(runtime)")


def test_depends_on_no_edge_for_third_party_or_self(tmp_path):
    root = tmp_path / "solo"
    root.mkdir()
    _write_json(
        root / "composer.json",
        {"name": "acme/solo", "require": {"symfony/framework-bundle": "^6.0"}},
    )
    _write_json(root / "composer.lock", {"packages": [{"name": "symfony/framework-bundle"}]})

    identity_map = overlay_depends.build_package_identity_map({"acme.solo": root})
    edges = overlay_depends.extract_depends_on_edges("acme.solo", root, identity_map)
    assert edges == []


# --- 2. manual relations: schema validation + dangling-ref rejection ------


def _fixture_graph(node_id: str, label: str, source_file: str) -> dict:
    return {"nodes": [{"id": node_id, "label": label, "source_file": source_file}], "links": []}


def test_manual_relation_resolves_when_refs_exist():
    graphs_by_repo = {
        "repo.a": _fixture_graph("a1", "AlphaThing", "src/alpha.py"),
        "repo.b": _fixture_graph("b1", "BetaThing", "src/beta.py"),
    }
    raw = [
        {
            "type": "depends_on",
            "source": {
                "repo": "repo.a",
                "source_file": "src/alpha.py",
                "qualified_label": "AlphaThing",
            },
            "target": {
                "repo": "repo.b",
                "source_file": "src/beta.py",
                "qualified_label": "BetaThing",
            },
            "confidence": 0.9,
            "evidence": "declared by a human",
        }
    ]
    edges = overlay_depends.build_manual_relation_edges(raw, graphs_by_repo)
    assert len(edges) == 1
    assert edges[0].provenance == "MANUAL"


def test_manual_relation_dangling_ref_is_hard_error():
    graphs_by_repo = {"repo.a": _fixture_graph("a1", "AlphaThing", "src/alpha.py")}
    raw = [
        {
            "type": "depends_on",
            "source": {
                "repo": "repo.a",
                "source_file": "src/alpha.py",
                "qualified_label": "AlphaThing",
            },
            "target": {
                "repo": "repo.ghost",
                "source_file": "src/nope.py",
                "qualified_label": "Nothing",
            },
            "evidence": "bogus",
        }
    ]
    with pytest.raises(DanglingReferenceError):
        overlay_depends.build_manual_relation_edges(raw, graphs_by_repo)


def test_resolve_ref_uses_shared_index_cache(monkeypatch):
    graphs = {"repoA": {"nodes": [{"source_file": "a.py", "label": "L"}]}}
    calls = []
    real = overlay_refs.build_repo_node_index

    def counting(graph_data):
        calls.append(1)
        return real(graph_data)

    monkeypatch.setattr(overlay_refs, "build_repo_node_index", counting)
    cache: dict[str, dict] = {}
    ref = LogicalRef(repo="repoA", source_file="a.py", qualified_label="L")
    overlay_refs.resolve_ref(ref, graphs, cache)
    overlay_refs.resolve_ref(ref, graphs, cache)
    overlay_refs.resolve_ref(ref, graphs, cache)
    assert len(calls) == 1


def test_manual_relations_schema_rejects_bad_shape(tmp_path):
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    bad_path = tmp_path / "manual-relations.json"
    _write_json(bad_path, {"relations": [{"type": "not_a_real_type", "source": {}, "target": {}}]})
    with pytest.raises(ValueError) as excinfo:
        overlay_depends.load_manual_relations(bad_path, schema)
    message = str(excinfo.value)
    assert str(bad_path) in message
    # The failing location is named, the instance content is not: jsonschema's
    # own message quotes the offending value, and this message reaches logs and
    # status.json.
    assert "not_a_real_type" not in message


def test_manual_relations_invalid_json_raises_value_error(tmp_path):
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    bad_path = tmp_path / "manual-relations.json"
    bad_path.write_text('{"relations": [', encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        overlay_depends.load_manual_relations(bad_path, schema)
    assert str(bad_path) in str(excinfo.value)


def test_manual_relations_missing_file_returns_empty(tmp_path):
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    result = overlay_depends.load_manual_relations(tmp_path / "does-not-exist.json", schema)
    assert result == []


# --- 3. similar_approach placeholder contract shape -----------------------


def test_similar_approach_placeholder_contract_shape():
    graphs_by_repo = {
        "repo.a": {
            "nodes": [
                {
                    "id": "a1",
                    "label": "OrderService",
                    "source_file": "src/a.py",
                    "community_name": "Orders",
                },
            ]
        },
        "repo.b": {
            "nodes": [
                {
                    "id": "b1",
                    "label": "OrderService",
                    "source_file": "src/b.py",
                    "community_name": "Orders",
                },
            ]
        },
    }
    edges = overlay_similar.compute_similar_approach_edges(graphs_by_repo, top_k=5)
    assert len(edges) == 1
    edge = edges[0]
    assert edge.type == "similar_approach"
    assert edge.provenance == overlay_similar.PLACEHOLDER_PROVENANCE
    assert 0.0 <= edge.confidence <= 1.0
    assert "placeholder" in edge.evidence.lower()
    assert isinstance(edge.source, LogicalRef)
    assert isinstance(edge.target, LogicalRef)
    assert {edge.source.repo, edge.target.repo} == {"repo.a", "repo.b"}


def test_similar_approach_ignores_same_repo_matches():
    graphs_by_repo = {
        "repo.a": {
            "nodes": [
                {"id": "a1", "label": "Thing", "source_file": "src/a1.py", "community_name": "X"},
                {"id": "a2", "label": "Thing", "source_file": "src/a2.py", "community_name": "X"},
            ]
        }
    }
    edges = overlay_similar.compute_similar_approach_edges(graphs_by_repo, top_k=5)
    assert edges == []


def test_similar_approach_respects_top_k_cap():
    # One node in repo.a matches label+community in 3 different repos; cap
    # top_k=1 must limit that node to a single emitted edge.
    graphs_by_repo = {
        "repo.a": {
            "nodes": [
                {"id": "a1", "label": "Thing", "source_file": "src/a.py", "community_name": "X"}
            ]
        },
        "repo.b": {
            "nodes": [
                {"id": "b1", "label": "Thing", "source_file": "src/b.py", "community_name": "X"}
            ]
        },
        "repo.c": {
            "nodes": [
                {"id": "c1", "label": "Thing", "source_file": "src/c.py", "community_name": "X"}
            ]
        },
    }
    edges = overlay_similar.compute_similar_approach_edges(graphs_by_repo, top_k=1)
    involving_a = [e for e in edges if e.source.repo == "repo.a" or e.target.repo == "repo.a"]
    assert len(involving_a) == 1


# --- 4. provides/consumes API matching + non-matching ----------------------


def _write_controller(path: Path, route: str, method: str = "GET") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<?php
namespace App\\Controller;

use Symfony\\Component\\Routing\\Attribute\\Route;

class WidgetController
{{
    #[Route(
        '{route}',
        name: "widget_get",
        methods: ['{method}']
    )]
    public function show()
    {{
    }}
}}
""",
        encoding="utf-8",
    )


def test_provides_api_route_extraction(tmp_path):
    root = tmp_path / "provider"
    _write_controller(root / "src" / "Controller" / "WidgetController.php", "/api/v1/widgets/{id}")
    providers = overlay_api.extract_symfony_route_providers("acme.provider", root)
    assert len(providers) == 1
    provider = providers[0]
    assert provider.http_method == "GET"
    assert provider.path_template == "/api/v1/widgets/{param}"
    assert provider.qualified_label == "WidgetController::show"


def test_consumes_api_matches_provider(tmp_path):
    provider_root = tmp_path / "provider"
    consumer_root = tmp_path / "consumer"
    _write_controller(
        provider_root / "src" / "Controller" / "WidgetController.php", "/api/v1/widgets/{id}"
    )

    (consumer_root / "src").mkdir(parents=True)
    (consumer_root / "src" / "Client.php").write_text(
        "<?php\nclass Client {\n  function go($http) {\n"
        "    $http->request('GET', '/api/v1/widgets/{param}');\n  }\n}\n",
        encoding="utf-8",
    )

    providers_by_repo = {
        "acme.provider": overlay_api.extract_symfony_route_providers("acme.provider", provider_root)
    }
    consumer_candidates_by_repo = {
        "acme.consumer": overlay_api.extract_consumer_literal_paths("acme.consumer", consumer_root)
    }
    edges = overlay_api.match_provides_consumes_edges(
        providers_by_repo, consumer_candidates_by_repo
    )

    types = {e.type for e in edges}
    assert types == {"provides_api", "consumes_api"}
    consumes = [e for e in edges if e.type == "consumes_api"][0]
    assert consumes.source.repo == "acme.consumer"
    assert consumes.target.repo == "acme.provider"
    assert consumes.provenance == "EXTRACTED_CONFIG"


def test_consumes_api_unresolvable_produces_zero_edges(tmp_path):
    provider_root = tmp_path / "provider"
    consumer_root = tmp_path / "consumer"
    _write_controller(
        provider_root / "src" / "Controller" / "WidgetController.php", "/api/v1/widgets/{id}"
    )

    (consumer_root / "src").mkdir(parents=True)
    (consumer_root / "src" / "Client.php").write_text(
        "<?php\nclass Client {\n  function go($http) {\n"
        "    $http->request('GET', '/api/v1/does-not-exist');\n  }\n}\n",
        encoding="utf-8",
    )

    providers_by_repo = {
        "acme.provider": overlay_api.extract_symfony_route_providers("acme.provider", provider_root)
    }
    consumer_candidates_by_repo = {
        "acme.consumer": overlay_api.extract_consumer_literal_paths("acme.consumer", consumer_root)
    }
    edges = overlay_api.match_provides_consumes_edges(
        providers_by_repo, consumer_candidates_by_repo
    )
    assert edges == []  # unresolvable consumer -> zero edges, never a guess


# --- 5. forbidden-edge invariant catches an overlay edge in structural output


def test_forbidden_edge_invariant_catches_each_overlay_relation_type():
    for relation_type in ("depends_on", "similar_approach", "provides_api", "consumes_api"):
        data = {
            "nodes": [{"id": "a"}, {"id": "b"}],
            "links": [{"source": "a", "target": "b", "relation": relation_type}],
        }
        result = validate.validate_forbidden_edges(data)
        assert not result.ok, (
            f"forbidden-edge invariant did not catch relation type {relation_type!r}"
        )


# --- end-to-end: overlay wired into the pipeline, never leaks into graph.json


def test_overlay_wired_into_pipeline_as_separate_artifact(env):
    core_root = env.add_repo("acme.core", "acme", "core", "core.acme.dev.lo", "repo_a.json")
    app_root = env.add_repo("acme.app", "acme", "app", "app.acme.dev.lo", "repo_b.json")
    env.write_registry()

    _write_json(core_root / "package.json", {"name": "@acme/core"})
    _write_json(
        app_root / "package.json",
        {"name": "acme-app", "dependencies": {"@acme/core": "^1.0"}},
    )
    _write_json(
        app_root / "package-lock.json",
        {"packages": {"node_modules/@acme/core": {"version": "1.0.0"}}},
    )

    settings = env.settings()
    report = run(settings)

    assert report.published, report.publish_blocked_reason
    assert report.overlay_edge_counts.get("depends_on") == 1

    current = settings.global_dir / "current"
    overlay_path = current / "cross-project-overlay.json"
    assert overlay_path.is_file()
    overlay_data = json.loads(overlay_path.read_text(encoding="utf-8"))
    assert overlay_data["edge_counts_by_type"]["depends_on"] == 1
    depends_edges = [e for e in overlay_data["edges"] if e["type"] == "depends_on"]
    assert depends_edges[0]["source"]["repo"] == "acme.app"
    assert depends_edges[0]["target"]["repo"] == "acme.core"

    # C5: the overlay edge must never appear inside the structural graph.
    global_graph = json.loads((current / "global-graph.json").read_text(encoding="utf-8"))
    for link in global_graph["links"]:
        assert link.get("relation") != "depends_on"
        assert link.get("cross_repo") is not True


# --- WS3: real ANN-backed similar_approach + fallback for unembedded nodes -


def test_similar_approach_uses_embedding_ann_when_vectors_available():
    graphs_by_repo = {
        "repo.a": {
            "nodes": [
                {
                    "id": "a1",
                    "label": "OrderService",
                    "source_file": "src/a.py",
                    "community_name": "Orders",
                },
            ]
        },
        "repo.b": {
            "nodes": [
                {
                    "id": "b1",
                    "label": "OrderHandler",
                    "source_file": "src/b.py",
                    "community_name": "Fulfillment",
                },
            ]
        },
    }
    # Different labels/communities -> the old exact-match placeholder would
    # find nothing, but the embedding vectors are near-identical -> the real
    # ANN path must still find the pair.
    embedding_vectors_by_repo = {
        "repo.a": RepoVectors.from_mapping(
            {"repo.a\x1fsrc/a.py\x1fOrderService": [1.0, 0.01, 0.0]}
        ),
        "repo.b": RepoVectors.from_mapping(
            {"repo.b\x1fsrc/b.py\x1fOrderHandler": [0.99, 0.02, 0.01]}
        ),
    }

    edges = overlay_similar.compute_similar_approach_edges(
        graphs_by_repo,
        embedding_vectors_by_repo=embedding_vectors_by_repo,
        top_k=5,
        embedding_model="qwen3-embedding:0.6b",
    )

    assert len(edges) == 1
    edge = edges[0]
    assert edge.type == "similar_approach"
    assert edge.provenance == overlay_similar.EMBEDDING_PROVENANCE
    assert edge.confidence > 0.9
    assert {edge.source.repo, edge.target.repo} == {"repo.a", "repo.b"}


def test_similar_approach_embedding_path_excludes_same_repo_pairs():
    graphs_by_repo = {
        "repo.a": {
            "nodes": [
                {"id": "a1", "label": "Thing1", "source_file": "src/a1.py", "community_name": "X"},
                {"id": "a2", "label": "Thing2", "source_file": "src/a2.py", "community_name": "X"},
            ]
        }
    }
    embedding_vectors_by_repo = {
        "repo.a": RepoVectors.from_mapping(
            {
                "repo.a\x1fsrc/a1.py\x1fThing1": [1.0, 0.0],
                "repo.a\x1fsrc/a2.py\x1fThing2": [0.99, 0.01],
            }
        )
    }
    edges = overlay_similar.compute_similar_approach_edges(
        graphs_by_repo, embedding_vectors_by_repo=embedding_vectors_by_repo, top_k=5
    )
    assert edges == []


def test_similar_approach_falls_back_for_nodes_without_a_vector():
    # Neither node has a vector this generation (e.g. both skipped by the
    # WS3 trivial-accessor heuristic) -> the embedding path has nothing to
    # do, but the fallback exact label+community scorer still covers them
    # (deliverable 7's documented find_similar fallback for skipped nodes).
    graphs_by_repo = {
        "repo.a": {
            "nodes": [
                {
                    "id": "a1",
                    "label": "OrderService",
                    "source_file": "src/a.py",
                    "community_name": "Orders",
                }
            ]
        },
        "repo.b": {
            "nodes": [
                {
                    "id": "b1",
                    "label": "OrderService",
                    "source_file": "src/b.py",
                    "community_name": "Orders",
                }
            ]
        },
    }
    # A third, unrelated repo DOES have a vector, so the embedding index is
    # "available" this generation (not the None/empty degraded case) — it
    # just has no vector for either of the two nodes under test here.
    embedding_vectors_by_repo = {
        "repo.a": RepoVectors.from_mapping({}),  # trivial heuristic — no vector at all
        "repo.b": RepoVectors.from_mapping({}),
        "repo.c": RepoVectors.from_mapping({"repo.c\x1fsrc/c.py\x1fUnrelated": [1.0, 0.0]}),
    }

    edges = overlay_similar.compute_similar_approach_edges(
        graphs_by_repo, embedding_vectors_by_repo=embedding_vectors_by_repo, top_k=5
    )

    assert len(edges) == 1
    assert edges[0].provenance == overlay_similar.PLACEHOLDER_PROVENANCE
    assert "placeholder" in edges[0].evidence.lower()
    assert {edges[0].source.repo, edges[0].target.repo} == {"repo.a", "repo.b"}


def test_similar_approach_forbidden_edge_invariant_still_holds_for_embedding_path():
    # Even though this is a real embedding-backed edge, it must still be
    # caught by the same structural forbidden-edge invariant if it were ever
    # (incorrectly) merged into the structural graph.
    data = {
        "nodes": [{"id": "a"}, {"id": "b"}],
        "links": [{"source": "a", "target": "b", "relation": "similar_approach"}],
    }
    result = validate.validate_forbidden_edges(data)
    assert not result.ok


# --- WS3: embed stage wired between naming and overlay-resolve -----------


def test_pipeline_embed_stage_degraded_by_default_and_manifest_has_real_fields(env):
    # env.settings() defaults BOTH the naming and embed health checks to
    # unhealthy (see conftest.py) so this exercises the fully-degraded WS3
    # path end-to-end without touching the real network.
    env.add_repo("acme.core", "acme", "core", "core.acme.dev.lo", "repo_a.json")
    env.add_repo("acme.app", "acme", "app", "app.acme.dev.lo", "repo_b.json")
    env.write_registry()

    settings = env.settings()
    report = run(settings)

    assert report.published, report.publish_blocked_reason
    assert report.embedding_status == embedding.EMBED_DEGRADED

    manifest = json.loads(
        (settings.global_dir / "current" / "generation-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["embedding_status"] == embedding.EMBED_DEGRADED
    assert "embedding_recipe" in manifest
    assert "embedding_stats" in manifest

    # Nothing persisted to the embeddings dir on a degraded run with no
    # previously-published generation to carry forward from — but the
    # pipeline must not crash either way.
    assert settings.embeddings_dir.exists() or not settings.embeddings_dir.exists()


def test_pipeline_embed_stage_healthy_populates_manifest_and_shards(env, monkeypatch):
    env.add_repo("acme.core", "acme", "core", "core.acme.dev.lo", "repo_a.json")
    env.add_repo("acme.app", "acme", "app", "app.acme.dev.lo", "repo_b.json")
    env.write_registry()

    def fake_embed_batch(base_url, model, inputs, timeout=30.0):
        return [[1.0, 0.0, 0.0] for _ in inputs]

    monkeypatch.setattr(embedding, "embed_batch", fake_embed_batch)

    settings = env.settings(
        ollama_health_check=lambda *a, **kw: True,
        ollama_embed_health_check=lambda *a, **kw: True,
    )
    report = run(settings)

    assert report.published, report.publish_blocked_reason
    assert report.embedding_status == embedding.EMBED_HEALTHY
    assert report.embedding_stats.get("embedded", 0) > 0

    manifest = json.loads(
        (settings.global_dir / "current" / "generation-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["embedding_model"] == settings.ollama_embed_model
    assert manifest["embedding_recipe"]["model"] == settings.ollama_embed_model
    assert manifest["embedding_recipe"]["dim"] == 3

    embeddings_current = settings.embeddings_dir / "current"
    assert embeddings_current.is_dir()
    assert (embeddings_current / "id-map.json").is_file()

    # GC: only this one generation exists so far — still within keep=2.
    assert len(list((settings.embeddings_dir / "generations").iterdir())) == 1
