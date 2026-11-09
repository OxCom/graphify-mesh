"""The server's key maps (`Generation.build_indexes`, merged graph) must equal
the keys sync wrote (per-repo graphs), including config-key dotted paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from graphify_mesh.server import similar
from graphify_mesh.server.store import Generation
from graphify_mesh.sync.embedding import graph_node_keys

VALUES_YAML = """\
ingress:
  enabled: true
  tls:
    - hosts: [a.example]
podSecurityContext:
  enabled: true
metadata:
  name: top
spec:
  template:
    metadata:
      name: inner
"""


def _per_repo_graph(tmp_path: Path) -> dict:
    yaml_config = pytest.importorskip("graphify.extractors.yaml_config")
    path = tmp_path / "values.yaml"
    path.write_text(VALUES_YAML, encoding="utf-8")
    graph = yaml_config.extract_yaml(path)
    # Per-repo graph.json carries relative source paths.
    for item in graph["nodes"] + graph["edges"]:
        item["source_file"] = "chart/values.yaml"
    for node in graph["nodes"]:
        if node["label"] == "values.yaml":
            node["label"] = "chart/values.yaml"  # graphify's disambiguated file label
    return graph


def _merged(repo_id: str, graph: dict) -> dict:
    """The merged-graph shape: `repo::`-prefixed ids, a `repo` field, other order."""
    return {
        "nodes": [
            dict(n, id=f"{repo_id}::{n['id']}", repo=repo_id) for n in reversed(graph["nodes"])
        ],
        "links": [
            dict(e, source=f"{repo_id}::{e['source']}", target=f"{repo_id}::{e['target']}")
            for e in reversed(graph["edges"])
        ],
    }


def _generation(graph: dict, schema_version: int | None, where: str = "manifest") -> Generation:
    manifest: dict = {}
    lexical: dict = {}
    if schema_version is not None and where == "manifest":
        manifest["lexical_index_schema_version"] = schema_version
    if schema_version is not None and where == "lexical":
        lexical["schema_version"] = schema_version
    gen = Generation(
        generation_id="g",
        manifest=manifest,
        graph=graph,
        overlay={},
        lexical=lexical,
        embeddings={},
    )
    gen.build_indexes()
    return gen


@pytest.mark.parametrize("where", ["manifest", "lexical"])
def test_store_keys_match_sync_keys(tmp_path, where):
    per_repo = _per_repo_graph(tmp_path)
    sync_keys = set(graph_node_keys("cem.k8s", per_repo).values())

    gen = _generation(_merged("cem.k8s", per_repo), 4, where)

    assert set(gen.key_by_node_id.values()) == sync_keys
    assert set(gen.node_id_by_key) == sync_keys
    assert len(gen.node_id_by_key) == len(gen.key_by_node_id)  # no last-writer-wins loss
    assert "cem.k8s\x1fchart/values.yaml\x1fpodSecurityContext.enabled" in gen.node_id_by_key


@pytest.mark.parametrize("schema_version", [3, None])
def test_store_keeps_bare_keys_for_a_pre_v4_generation(tmp_path, schema_version):
    """A generation published before dotted paths keys its lexical index and
    shards by bare label; the server must look them up the same way."""
    # No schema version at all is read as v2, as `lexical_read._schema` does.
    gen = _generation(_merged("cem.k8s", _per_repo_graph(tmp_path)), schema_version)
    assert "cem.k8s\x1fchart/values.yaml\x1fenabled" in gen.node_id_by_key
    assert not any(k.endswith("podSecurityContext.enabled") for k in gen.node_id_by_key)


def _two_repo_generation(tmp_path: Path, schema_version: int) -> Generation:
    per_repo = _per_repo_graph(tmp_path)
    merged_a, merged_b = _merged("repo.a", per_repo), _merged("repo.b", per_repo)
    graph = {
        "nodes": [
            dict(n, community_name="Helm values") for n in merged_a["nodes"] + merged_b["nodes"]
        ],
        "links": merged_a["links"] + merged_b["links"],
    }
    return _generation(graph, schema_version)


def test_fallback_exact_match_buckets_config_keys_by_dotted_path(tmp_path):
    """Sync's fallback scorer buckets by dotted path; the server's must too, or
    `ingress.enabled` matches every `enabled` key of the other repo."""
    gen = _two_repo_generation(tmp_path, 4)
    seed = "repo.a\x1fchart/values.yaml\x1fingress.enabled"
    matches = similar.fallback_exact_match(seed, gen, cross_repo_only=True, top_k=10)
    assert matches == ["repo.b\x1fchart/values.yaml\x1fingress.enabled"]


def test_fallback_exact_match_keeps_bare_labels_for_a_pre_v4_generation(tmp_path):
    gen = _two_repo_generation(tmp_path, 3)
    seed = "repo.a\x1fchart/values.yaml\x1fenabled"
    matches = similar.fallback_exact_match(seed, gen, cross_repo_only=True, top_k=10)
    # Both bare `enabled` nodes of repo.b share one key before v4.
    assert set(matches) == {"repo.b\x1fchart/values.yaml\x1fenabled"}
