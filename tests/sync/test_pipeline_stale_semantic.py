"""The pipeline hands each repo's graph path to the lexical index stage.

build_lexical_index finds graphify's manifest.json next to graph.json; without
`graph_paths_by_repo` it flags nothing, so this checks the wiring end to end:
a manifest row with an AST hash and an empty semantic hash flags the semantic
nodes of that file in the published lexical index.
"""

from __future__ import annotations

import json

from graphify_mesh.sync import repo_tags
from graphify_mesh.sync.pipeline import run
from graphify_mesh.sync.state import CONTENT_DIGEST_ENV

REPO_ID = "example-org.svc"


def test_published_lexical_index_flags_stale_semantic_nodes(env, monkeypatch):
    monkeypatch.delenv(CONTENT_DIGEST_ENV, raising=False)
    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.103")
    root = env.add_repo(REPO_ID, "example-org", "svc", "svc.example-org.dev.lo")
    env.write_registry()
    (root / "values.yaml").write_text("replicaCount: 1\n", encoding="utf-8")
    collection = env.collection_path("example-org", "svc")
    graph = json.loads((collection / "graph.json").read_text(encoding="utf-8"))
    graph["nodes"].append(
        {
            "id": "s_values",
            "label": "replicaCountSetting",
            "source_file": "values.yaml",
            "_origin": "semantic",
        }
    )
    (collection / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    row = {"mtime": 1.0, "seen": 1.0, "ast_hash": "h1", "semantic_hash": ""}
    (collection / "manifest.json").write_text(json.dumps({"values.yaml": row}), encoding="utf-8")

    settings = env.settings(shrink_policy="count")
    report = run(settings)

    assert report.published
    data = json.loads(
        (settings.global_dir / "current" / "lexical-index.json").read_text(encoding="utf-8")
    )
    flagged = [data["documents"][doc_id] for doc_id in data["stale_semantic"]]
    assert len(flagged) == 1
    repo, key = flagged[0]
    assert repo == REPO_ID
    assert key.endswith("replicaCountSetting")
    assert report.lexical_index_stats["stale_semantic_documents"] == 1
