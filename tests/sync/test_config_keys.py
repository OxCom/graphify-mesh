"""Dotted key paths for YAML/JSON config-key nodes (sync/config_keys.py) and
their use as durable keys by the embedding, lexical and overlay stages."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from graphify_mesh.sync import embedding, lexical_index
from graphify_mesh.sync.config import Settings
from graphify_mesh.sync.config_keys import config_key_paths, is_file_node_label
from graphify_mesh.sync.embedding import graph_node_keys
from graphify_mesh.sync.overlay_depends import build_manual_relation_edges
from graphify_mesh.sync.overlay_refs import AmbiguousReferenceError, LogicalRef, resolve_ref

FAKE_GRAPHIFY = Path(__file__).resolve().parents[1] / "fixtures" / "fake_graphify" / "graphify"

VALUES_YAML = """\
ingress:
  enabled: true
  tls:
    - hosts: [a.example]
      secretName: tls-a
podSecurityContext:
  enabled: true
containers:
  - name: app
    image: app:1
  - name: side
    image: side:1
metadata:
  name: top
spec:
  template:
    metadata:
      name: inner
"""


def _yaml_graph(tmp_path: Path) -> dict:
    yaml_config = pytest.importorskip("graphify.extractors.yaml_config")
    path = tmp_path / "values.yaml"
    path.write_text(VALUES_YAML, encoding="utf-8")
    return yaml_config.extract_yaml(path)


def _paths_by_line(graph: dict) -> dict[str, str]:
    paths = config_key_paths(graph)
    nodes = {node["id"]: node for node in graph["nodes"]}
    return {
        nodes[nid]["source_location"] + " " + nodes[nid]["label"]: p for nid, p in paths.items()
    }


def test_real_yaml_extractor_output_gets_dotted_paths(tmp_path):
    graph = _yaml_graph(tmp_path)
    paths = set(config_key_paths(graph).values())

    assert {"ingress", "ingress.enabled", "ingress.tls", "podSecurityContext.enabled"} <= paths
    # Sequence items carry no index: the extractor folds them into the list's key.
    assert {"ingress.tls.hosts", "ingress.tls.secretName"} <= paths
    assert {"containers.name", "containers.image"} <= paths
    assert not any("[" in p for p in paths)


def test_file_node_is_the_root_and_carries_no_path(tmp_path):
    graph = _yaml_graph(tmp_path)
    paths = config_key_paths(graph)
    file_nodes = [n for n in graph["nodes"] if n["label"] == "values.yaml"]
    assert len(file_nodes) == 1
    assert file_nodes[0]["id"] not in paths


def test_shared_node_takes_the_parent_of_its_first_occurrence(tmp_path):
    """`metadata.name` and `spec.template.metadata.name` are one extractor node
    (id = stem + parent key + key); its line is the first occurrence's, so the
    path must name that occurrence."""
    graph = _yaml_graph(tmp_path)
    by_line = _paths_by_line(graph)
    assert by_line["L14 name"] == "metadata.name"
    assert by_line["L17 metadata"] == "spec.template.metadata"


def test_sibling_keys_no_longer_share_a_durable_key(tmp_path):
    graph = _yaml_graph(tmp_path)
    keys = graph_node_keys("repo.a", graph)
    enabled_keys = sorted(k.rsplit("\x1f", 1)[1] for k in keys.values() if k.endswith("enabled"))
    assert enabled_keys == ["ingress.enabled", "podSecurityContext.enabled"]
    assert len(set(keys.values())) == len(keys)


def test_real_json_extractor_separates_nested_and_top_level_type(tmp_path):
    json_config = pytest.importorskip("graphify.extractors.json_config")
    path = tmp_path / "package.json"
    path.write_text(
        json.dumps(
            {
                "name": "pkg",
                "type": "module",
                "repository": {"type": "git", "url": "x"},
                "scripts": {"build": "tsc"},
            }
        ),
        encoding="utf-8",
    )
    graph = json_config.extract_json(path)
    paths = set(config_key_paths(graph).values())
    assert {"type", "repository.type", "scripts.build"} <= paths
    keys = graph_node_keys("repo.a", graph)
    assert len(set(keys.values())) == len(keys)


def _graph(nodes: list[dict], links: list[tuple[str, str]]) -> dict:
    return {
        "nodes": nodes,
        "links": [{"source": s, "target": t, "relation": "contains"} for s, t in links],
    }


def _n(nid: str, label: str, line: int, source_file: str = "conf/values.yaml", **extra) -> dict:
    return {
        "id": nid,
        "label": label,
        "source_file": source_file,
        "source_location": f"L{line}",
        **extra,
    }


def test_file_node_labelled_with_parent_dir_is_still_the_root():
    """graphify labels a file node `styleguide/values.yaml` when the basename alone
    is ambiguous."""
    sf = "src/helm/styleguide/values.yaml"
    graph = _graph(
        [
            _n("f", "styleguide/values.yaml", 1, sf),
            _n("a", "image", 2, sf),
            _n("b", "registry", 3, sf),
        ],
        [("f", "a"), ("a", "b")],
    )
    assert config_key_paths(graph) == {"a": "image", "b": "image.registry"}
    assert is_file_node_label("styleguide/values.yaml", sf)
    assert not is_file_node_label("values", sf)


def test_unreachable_semantic_cyclic_and_non_config_nodes_are_left_bare():
    graph = _graph(
        [
            _n("f", "values.yaml", 1),
            _n("a", "a", 2),
            _n("orphan", "orphan", 5),
            _n("sem", "Redis Sentinel", 3, _origin="semantic"),
            _n("x", "x", 6),
            _n("y", "y", 6),
            _n("code", "Widget", 1, "src/w.py"),
            _n("codefile", "w.py", 1, "src/w.py"),
        ],
        [("f", "a"), ("f", "sem"), ("x", "y"), ("y", "x"), ("codefile", "code")],
    )
    assert config_key_paths(graph) == {"a": "a"}


def test_node_ids_never_decide_the_path():
    """Sync reads per-repo ids, the server reads `repo::`-prefixed ids in another
    order; both must derive the same paths."""
    nodes = [
        _n("f", "values.yaml", 1),
        _n("p1", "metadata", 2),
        _n("p2", "metadata", 9),
        _n("c", "name", 3),
    ]
    links = [("f", "p1"), ("f", "p2"), ("p1", "c"), ("p2", "c")]
    per_repo = config_key_paths(_graph(nodes, links))
    merged_nodes = [dict(n, id=f"r::{n['id']}", repo="r") for n in reversed(nodes)]
    merged_links = [(f"r::{s}", f"r::{t}") for s, t in reversed(links)]
    merged = config_key_paths(_graph(merged_nodes, merged_links))
    assert {f"r::{k}": v for k, v in per_repo.items()} == merged
    assert per_repo["c"] == "metadata.name"


CYCLIC_YAML = """\
y:
  x:
    y:
      x: 1
      z: 2
"""


def test_contains_cycle_gives_the_same_paths_in_any_node_order(tmp_path):
    """Extractor ids are `stem + parent key + key`, so this YAML makes `y.x` and
    `y.x.y` contain each other. Paths must not depend on which node is resolved
    first, and the server's merged-graph keys must equal sync's."""
    yaml_config = pytest.importorskip("graphify.extractors.yaml_config")
    path = tmp_path / "values.yaml"
    path.write_text(CYCLIC_YAML, encoding="utf-8")
    graph = yaml_config.extract_yaml(path)
    graph["links"] = graph.pop("edges")

    expected = config_key_paths(graph)
    assert set(expected.values()) == {"y", "y.x", "y.x.y", "y.x.y.z"}
    orders = [list(reversed(graph["nodes"]))]
    rng = random.Random(7)
    for _ in range(5):
        shuffled = list(graph["nodes"])
        rng.shuffle(shuffled)
        orders.append(shuffled)
    for nodes in orders:
        assert config_key_paths(dict(graph, nodes=nodes)) == expected

    from graphify_mesh.server.store import Generation

    merged = {
        "nodes": [dict(n, id=f"r::{n['id']}", repo="r") for n in reversed(graph["nodes"])],
        "links": [
            dict(e, source=f"r::{e['source']}", target=f"r::{e['target']}")
            for e in reversed(graph["links"])
        ],
    }
    gen = Generation(
        generation_id="g",
        manifest={"lexical_index_schema_version": lexical_index.LEXICAL_SCHEMA_VERSION},
        graph=merged,
        overlay={},
        lexical={},
        embeddings={},
    )
    gen.build_indexes()
    assert set(gen.node_id_by_key) == set(graph_node_keys("r", graph).values())


# ---------------------------------------------------------------------------
# lexical index
# ---------------------------------------------------------------------------


def _docs_for_alias(data: dict, alias: str) -> list[str]:
    return [data["documents"][doc_id][1] for doc_id in data["alias_exact"].get(alias, [])]


def test_lexical_basename_alias_only_on_the_file_node(tmp_path):
    graph = _yaml_graph(tmp_path)
    data = lexical_index.build_lexical_index({"repo.a": graph}, {}).data
    holders = _docs_for_alias(data, "values.yaml")
    assert len(holders) == 1
    assert holders[0].endswith("\x1fvalues.yaml")
    # The dotted path is an alias of its own node; the bare key is not.
    assert len(_docs_for_alias(data, "ingress.enabled")) == 1
    assert "enabled" not in data["alias_exact"]


def test_lexical_file_without_a_file_node_keeps_the_basename_alias():
    graph = {"nodes": [_n("a", "Widget", 3, "src/w.py"), _n("b", "Gadget", 9, "src/w.py")]}
    data = lexical_index.build_lexical_index({"repo.a": graph}, {}).data
    assert len(_docs_for_alias(data, "w.py")) == 2


def test_lexical_label_terms_include_path_segments_and_collisions_are_counted(tmp_path):
    graph = _yaml_graph(tmp_path)
    graph["nodes"] += [_n("w1", "Widget", 3, "src/w.py"), _n("w2", "Widget", 9, "src/w.py")]
    result = lexical_index.build_lexical_index({"repo.a": graph}, {})
    data = result.data

    def docs_with(term: str, field_name: str) -> set[str]:
        field_index = data["fields"].index(field_name)
        mask = (1 << lexical_index.FIELD_PACK_BITS) - 1
        return {
            data["documents"][packed >> lexical_index.FIELD_PACK_BITS][1].rsplit("\x1f", 1)[1]
            for packed in data["postings"].get(term, [])
            if packed & mask == field_index
        }

    # Ancestors are indexed like a location (path field), the leaf key as the label.
    assert "podSecurityContext.enabled" in docs_with("security", "path")
    assert "podSecurityContext.enabled" not in docs_with("security", "label")
    assert "podSecurityContext.enabled" in docs_with("enabled", "label")
    assert "podSecurityContext" in docs_with("security", "label")
    assert result.stats.key_collisions == 1
    assert result.stats.key_collision_nodes == 2
    assert result.stats.config_key_collisions == 0
    assert result.stats.to_dict()["config_key_collisions"] == 0


# ---------------------------------------------------------------------------
# overlay refs
# ---------------------------------------------------------------------------


def test_overlay_ref_resolves_dotted_path_and_bare_fallback(tmp_path):
    graph = _yaml_graph(tmp_path)
    sf = str(tmp_path / "values.yaml")
    graphs = {"repo.a": graph}
    node = resolve_ref(LogicalRef("repo.a", sf, "podSecurityContext.enabled"), graphs)
    assert node is not None and node["label"] == "enabled"
    # A bare label still resolves (a manual relation written before dotted
    # paths) when one nested key carries it, but never shadows a node that
    # owns that label as its path, and never picks among several.
    secret = resolve_ref(LogicalRef("repo.a", sf, "secretName"), graphs)
    assert secret is not None and secret["label"] == "secretName"
    assert resolve_ref(LogicalRef("repo.a", sf, "enabled"), graphs) is None
    ingress = resolve_ref(LogicalRef("repo.a", sf, "ingress"), graphs)
    assert ingress is not None and ingress["source_location"] == "L1"


def test_manual_relation_bare_label_is_rewritten_or_rejected_as_ambiguous(tmp_path):
    graph = _yaml_graph(tmp_path)
    sf = str(tmp_path / "values.yaml")
    graphs = {"repo.a": graph}

    def relation(source_label: str) -> dict:
        return {
            "type": "manual",
            "source": {"repo": "repo.a", "source_file": sf, "qualified_label": source_label},
            "target": {"repo": "repo.a", "source_file": sf, "qualified_label": "ingress"},
        }

    [edge] = build_manual_relation_edges([relation("secretName")], graphs)
    assert edge.source.qualified_label == "ingress.tls.secretName"
    assert edge.target.qualified_label == "ingress"

    with pytest.raises(
        AmbiguousReferenceError, match="ingress.enabled, podSecurityContext.enabled"
    ):
        build_manual_relation_edges([relation("enabled")], graphs)


# ---------------------------------------------------------------------------
# embedding
# ---------------------------------------------------------------------------


def _settings(tmp_path: Path) -> Settings:
    mesh_root = tmp_path / "mesh"
    return Settings.from_env(
        mesh_root=mesh_root,
        scan_roots=[tmp_path / "www"],
        registry_path=mesh_root / "bin" / "registry.json",
        graphify_bin=str(FAKE_GRAPHIFY),
    )


def test_unchanged_repo_with_old_bare_keys_reembeds_only_changed_keys(tmp_path, monkeypatch):
    """The first run after the key change: an unchanged repo's shard holds a bare
    `enabled` key. The repo must not reuse that shard whole, and only the node
    whose key changed is embedded again."""
    settings = _settings(tmp_path)
    sf = "values.yaml"
    graph = _graph(
        [_n("f", "values.yaml", 1, sf), _n("i", "ingress", 1, sf), _n("e", "enabled", 2, sf)],
        [("f", "i"), ("i", "e")],
    )

    def h(label: str) -> str:  # no file under the root => empty snippet
        return embedding._content_hash(embedding.build_embedding_input(label, "", sf))

    prev = settings.embeddings_dir / "generations" / "gen-0"
    prev.mkdir(parents=True)
    entries = {
        f"repo.a\x1f{sf}\x1fvalues.yaml": {"content_hash": h("values.yaml"), "row": 0},
        f"repo.a\x1f{sf}\x1fingress": {"content_hash": h("ingress"), "row": 1},
        f"repo.a\x1f{sf}\x1fenabled": {"content_hash": h("enabled"), "row": 2},
    }
    (prev / "repo.a.meta.json").write_text(
        json.dumps(
            {
                "repo_id": "repo.a",
                "shard_format": embedding.SHARD_FORMAT_VERSION,
                "dim": 1,
                "recipe_stamp": embedding.recipe_stamp_for_model(settings.ollama_embed_model),
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    np.save(prev / "repo.a.npy", np.array([[1.0], [2.0], [3.0]], dtype=np.float32))
    (prev / "id-map.json").write_text("{}", encoding="utf-8")
    (settings.embeddings_dir / "current").symlink_to(prev, target_is_directory=True)

    calls: list[list[str]] = []

    def fake_embed_batch(base_url, model, inputs, timeout=30.0):
        calls.append(list(inputs))
        return [[5.0] for _ in inputs]

    monkeypatch.setattr(embedding, "embed_batch", fake_embed_batch)
    result = embedding.run_embedding_stage(
        graph_paths_by_repo={},
        repo_roots_by_id={"repo.a": tmp_path / "absent"},
        graphs_by_repo={"repo.a": graph},
        unchanged_repo_ids={"repo.a"},
        settings=settings,
        provisional_generation_id="gen-1",
        health_check=lambda base_url, timeout: True,
    )

    assert result.stats.reused_repos_unchanged == 0
    assert result.stats.reused == 2
    assert [len(batch) for batch in calls] == [1]
    assert calls[0][0].startswith("label: ingress.enabled")
    vectors = result.vectors_by_repo["repo.a"]
    vector = vectors.get(f"repo.a\x1f{sf}\x1fingress.enabled")
    assert vector is not None and list(vector) == [5.0]
    assert f"repo.a\x1f{sf}\x1fenabled" not in result.shards_by_repo["repo.a"].entries


def test_degraded_run_drops_carried_entries_whose_key_no_longer_exists(tmp_path):
    """An embed outage on the first run after the key change carries the old
    shard forward. Its bare `enabled` vector belongs to `ingress.enabled`; it
    must not be published under a key v4 would bind to a top-level `enabled`."""
    settings = _settings(tmp_path)
    sf = "values.yaml"
    graph = _graph(
        [_n("f", "values.yaml", 1, sf), _n("i", "ingress", 1, sf), _n("e", "enabled", 2, sf)],
        [("f", "i"), ("i", "e")],
    )
    prev = settings.embeddings_dir / "generations" / "gen-0"
    prev.mkdir(parents=True)
    entries = {
        f"repo.a\x1f{sf}\x1fenabled": {"content_hash": "h0", "row": 0},
        f"repo.a\x1f{sf}\x1fingress": {"content_hash": "h1", "row": 1},
        f"repo.a\x1f{sf}\x1fvalues.yaml": {"content_hash": "h2", "row": 2},
    }
    (prev / "repo.a.meta.json").write_text(
        json.dumps(
            {
                "repo_id": "repo.a",
                "shard_format": embedding.SHARD_FORMAT_VERSION,
                "dim": 1,
                "recipe_stamp": embedding.recipe_stamp_for_model(settings.ollama_embed_model),
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    np.save(prev / "repo.a.npy", np.array([[1.0], [2.0], [3.0]], dtype=np.float32))
    (prev / "id-map.json").write_text("{}", encoding="utf-8")
    (settings.embeddings_dir / "current").symlink_to(prev, target_is_directory=True)

    result = embedding.run_embedding_stage(
        graph_paths_by_repo={},
        repo_roots_by_id={"repo.a": tmp_path / "absent"},
        graphs_by_repo={"repo.a": graph},
        unchanged_repo_ids={"repo.a"},
        settings=settings,
        provisional_generation_id="gen-1",
        health_check=lambda base_url, timeout: False,
    )

    assert result.status == embedding.EMBED_DEGRADED
    assert result.vectors_dropped_repos == []
    shard = result.shards_by_repo["repo.a"]
    assert set(shard.entries) == {f"repo.a\x1f{sf}\x1fingress", f"repo.a\x1f{sf}\x1fvalues.yaml"}
    ingress = shard.vectors.get(f"repo.a\x1f{sf}\x1fingress")
    file_vector = shard.vectors.get(f"repo.a\x1f{sf}\x1fvalues.yaml")
    assert ingress is not None and list(ingress) == [2.0]
    assert file_vector is not None and list(file_vector) == [3.0]
    assert [shard.entries[k]["row"] for k in shard.vectors.keys] == [0, 1]
