"""Version-compatibility contract with the upstream `graphifyy` package.

Every other test in this suite fakes the graphify CLI
(`tests/fixtures/fake_graphify/graphify`), so nothing validates what this
package actually needs from the real upstream. These tests pin the three real
coupling points — the in-process `graphify.build.distinct_repo_tags` import,
the `merge-graphs --out` argv this package emits, and the resolved upstream
version — and are the suite the `graphify-compat` CI job runs against each
pinned `graphifyy` version, including the declared floor.

They skip (never fail) when the upstream package or the `graphify` binary is
absent, so `pytest tests/` stays green on a machine without graphifyy.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import networkx as nx
import pytest

from graphify_mesh.sync import config, graphify_cli

pytest.importorskip("graphify")

FIXTURE_GRAPHS = Path(__file__).resolve().parents[1] / "fixtures" / "graphs"


def _graphify_bin() -> str:
    found = shutil.which("graphify")
    if found is None:
        pytest.skip("the real `graphify` binary is not on PATH")
    return found


def _expected_version() -> str:
    return (os.environ.get("GRAPHIFY_COMPAT_EXPECTED_VERSION") or "").strip()


def test_distinct_repo_tags_is_importable_and_callable():
    """repo_tags.compute_tag_to_repo_id imports this at call time; its guard
    compares len(tags) to len(repo_ids), so a same-length list[str] return is
    the whole contract."""
    from graphify.build import distinct_repo_tags

    assert callable(distinct_repo_tags)

    paths = [
        FIXTURE_GRAPHS / "repo_a.json",
        FIXTURE_GRAPHS / "repo_b.json",
    ]
    tags = distinct_repo_tags(paths)

    assert isinstance(tags, list)
    assert len(tags) == len(paths)
    for tag in tags:
        assert isinstance(tag, str)
        assert tag != ""


def test_real_merge_graphs_argv_produces_a_merged_graph(tmp_path):
    """End-to-end against the real binary, through run_merge_graphs so the
    argv under test is the one the package actually emits (including the
    centralized `merge-graphs` subcommand name and `--out`)."""
    graphify_bin = _graphify_bin()

    graph_paths = [FIXTURE_GRAPHS / "repo_a.json", FIXTURE_GRAPHS / "repo_b.json"]
    out_path = tmp_path / "merged.json"
    staging_home = tmp_path / "staging-home"

    result = graphify_cli.run_merge_graphs(graphify_bin, graph_paths, out_path, staging_home)

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert out_path.exists()

    merged = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(merged.get("nodes"), list)
    assert merged["nodes"]

    input_node_counts = [
        len(json.loads(p.read_text(encoding="utf-8"))["nodes"]) for p in graph_paths
    ]
    # A merge must not drop a whole side of the input.
    assert len(merged["nodes"]) >= max(input_node_counts)
    assert isinstance(merged.get("links"), list)


def test_merge_subcommand_name_is_centralized():
    assert config.GRAPHIFY_MERGE_SUBCOMMAND == "merge-graphs"


def test_installed_graphifyy_matches_the_pinned_version():
    """Guards the CI matrix: a pin that pip failed to honour must fail loudly
    instead of silently re-testing the resolved version. Unset/empty env var
    is the floating "latest" lane, which has no version to assert."""
    expected = _expected_version()
    if not expected:
        pytest.skip("GRAPHIFY_COMPAT_EXPECTED_VERSION is unset — floating-latest lane")

    assert importlib.metadata.version("graphifyy") == expected

    graphify_bin = _graphify_bin()
    proc = subprocess.run(
        [graphify_bin, "--version"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr!r}"
    # `graphify --version` also prints skill/package mismatch warnings to
    # stdout, so match the version as a substring, not the whole output.
    assert re.search(rf"(?<![\w.]){re.escape(expected)}(?![\w.])", proc.stdout), (
        f"expected {expected!r} in `graphify --version` stdout: {proc.stdout!r}"
    )


def test_openai_client_is_importable_for_in_process_labeling():
    """graphify's ollama backend builds its client with `from openai import OpenAI`.
    Without this dependency every label call fails silently into placeholders."""
    import importlib.util

    assert importlib.util.find_spec("openai") is not None


def test_openai_client_constructs_against_the_installed_httpx():
    """This is what the 1.55.3 floor actually buys.

    httpx 0.28 removed `proxies=`, which openai passed until 1.55.3, so an older
    client raises TypeError at CONSTRUCTION — inside generate_community_labels,
    which swallows it into placeholder names. No request is made here.
    """
    from openai import OpenAI

    client = OpenAI(api_key="not-a-real-key", base_url="http://127.0.0.1:9/v1")
    assert client is not None


def test_graphify_exposes_the_four_functions_the_naming_stage_calls():
    from graphify.cluster import (
        cluster,
        community_member_sigs,
        label_communities_by_hub,
        remap_communities_to_previous,
    )
    from graphify.llm import generate_community_labels

    assert callable(cluster)
    assert callable(community_member_sigs)
    assert callable(label_communities_by_hub)
    assert callable(remap_communities_to_previous)
    assert callable(generate_community_labels)


def test_cluster_assigns_every_node_including_isolates():
    from graphify.cluster import cluster

    G = nx.Graph()
    G.add_edge("a::1", "a::2")
    G.add_node("a::lonely")
    communities = cluster(G)
    assigned = {n for members in communities.values() for n in members}
    assert assigned == {"a::1", "a::2", "a::lonely"}


def test_member_sigs_depend_on_membership_only():
    from graphify.cluster import community_member_sigs

    assert community_member_sigs({0: ["b", "a"]}) == community_member_sigs({0: ["a", "b"]})
    assert community_member_sigs({0: ["a"]}) != community_member_sigs({0: ["a", "b"]})


def test_remap_keeps_ids_of_an_unchanged_partition():
    from graphify.cluster import remap_communities_to_previous

    previous = {"a": 3, "b": 3, "c": 5}
    remapped = remap_communities_to_previous({0: ["a", "b"], 1: ["c"]}, previous)
    assert remapped[3] == ["a", "b"]
    assert remapped[5] == ["c"]


def test_build_from_json_still_collapses_nodes_across_repos():
    """WATCHDOG, not a wish.

    `graphify.build.build_from_json` merges a non-AST node into an AST node
    sharing (source_file, label) without considering `repo`, which in a merged
    multi-repo graph collapses across repositories. That is why this package
    loads graphs with node_link_graph instead. When this test FAILS, upstream
    has fixed the defect and the avoidance can be revisited.
    """
    from graphify.build import build_from_json

    data = {
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
        ],
        "links": [],
    }
    G = build_from_json(data, directed=False)
    assert G.number_of_nodes() == 1, (
        "upstream no longer collapses across repos — revisit the node_link_graph workaround"
    )
