"""Per-repo shrink guard tolerance.

The `extract` action re-derives entities with an LLM and is not deterministic, so
an unchanged repo varies by a few percent between runs. A strict guard turns that
jitter into a permanent refusal loop (the refusal does not advance per-repo state,
so the repo re-extracts and wobbles again every run). These tests pin both halves
of the contract: small jitter is absorbed, material loss is still refused.
"""

from __future__ import annotations

import os

import pytest

from graphify_mesh.sync.config import SHRINK_TOLERANCE, _read_shrink_tolerance
from graphify_mesh.sync.sync_project import (
    STATUS_NOOP,
    STATUS_SHRINK_REFUSED,
    STATUS_UPDATED,
    _classify_shrink,
    _shrink_floor,
)

OLD = ("hash_old", (1000, 2000))


def classify(new_nodes: int, new_edges: int, tolerance: float) -> str:
    return _classify_shrink("hash_old", "hash_new", OLD[1], (new_nodes, new_edges), tolerance)


def test_identical_hash_is_noop_regardless_of_tolerance():
    assert _classify_shrink("h", "h", (10, 10), (1, 1), 0.5) == STATUS_NOOP


def test_default_tolerance_is_strict_for_direct_callers():
    # Callers that do not opt in keep the original absolute contract.
    assert _classify_shrink("a", "b", (1000, 2000), (999, 2000)) == STATUS_SHRINK_REFUSED


def test_growth_always_accepted():
    assert classify(1001, 2001, 0.0) == STATUS_UPDATED
    assert classify(1001, 2001, 0.10) == STATUS_UPDATED


@pytest.mark.parametrize("new_nodes", [999, 950, 900])
def test_jitter_within_tolerance_accepted(new_nodes):
    # -0.1%, -5% and exactly -10% are extraction noise, not data loss.
    assert classify(new_nodes, 2000, 0.10) == STATUS_UPDATED


@pytest.mark.parametrize(
    ("new_nodes", "new_edges"),
    [
        (899, 2000),  # nodes past the floor
        (1000, 1799),  # edges past the floor
        (500, 1000),  # both collapse
    ],
)
def test_material_loss_still_refused(new_nodes, new_edges):
    assert classify(new_nodes, new_edges, 0.10) == STATUS_SHRINK_REFUSED


def test_zero_tolerance_restores_absolute_guard():
    assert classify(999, 2000, 0.0) == STATUS_SHRINK_REFUSED


def test_tiny_graphs_get_exact_match_semantics():
    # Rounding up means a 3-node graph tolerates no loss at all: one node is a
    # third of the graph, which is a real deletion rather than jitter. This keeps
    # the fake-graphify fixture (3 nodes, drops 1) refused.
    assert _shrink_floor(3, 0.10) == 3
    assert _classify_shrink("a", "b", (3, 2), (2, 2), 0.10) == STATUS_SHRINK_REFUSED


def test_shrink_floor_never_widened_by_truncation():
    # ceil, not int(): 95.0 -> 95, but 94.5 must round to 95, never down to 94.
    assert _shrink_floor(1000, 0.10) == 900
    assert _shrink_floor(105, 0.10) == 95
    assert _shrink_floor(0, 0.10) == 0


class TestEnvOverride:
    def test_absent_env_uses_default(self, monkeypatch):
        monkeypatch.delenv("GRAPHIFY_MESH_SHRINK_TOLERANCE", raising=False)
        assert _read_shrink_tolerance() == SHRINK_TOLERANCE

    def test_explicit_zero_is_honoured(self, monkeypatch):
        monkeypatch.setenv("GRAPHIFY_MESH_SHRINK_TOLERANCE", "0")
        assert _read_shrink_tolerance() == 0.0

    def test_valid_value_is_honoured(self, monkeypatch):
        monkeypatch.setenv("GRAPHIFY_MESH_SHRINK_TOLERANCE", "0.25")
        assert _read_shrink_tolerance() == 0.25

    @pytest.mark.parametrize("raw", ["", "   ", "abc", "-0.1", "1.0", "2", "1e9"])
    def test_invalid_or_out_of_range_falls_back_to_default(self, monkeypatch, raw):
        # Fails safe: a tolerance of 1.0 would accept a graph collapsing to zero.
        monkeypatch.setenv("GRAPHIFY_MESH_SHRINK_TOLERANCE", raw)
        assert _read_shrink_tolerance() == SHRINK_TOLERANCE


def test_settings_default_picks_up_env(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_MESH_SHRINK_TOLERANCE", "0.2")
    # Imported lazily so the field default_factory runs under the patched env.
    from graphify_mesh.sync.config import Settings

    settings = Settings(
        mesh_root=os.curdir,
        scan_roots=[],
        approved_roots=[],
        registry_path=os.curdir,
    )
    assert settings.shrink_tolerance == 0.2
