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
