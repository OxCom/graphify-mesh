from __future__ import annotations

import pytest

from graphify_mesh.sync import embed_similarity
from graphify_mesh.sync.config import _exact_max_rows_from_env
from graphify_mesh.sync.vectors import RepoVectors

DEFAULT = embed_similarity.EXACT_MAX_ROWS


def _vectors(rows_per_repo: int, repos: tuple[str, ...] = ("a", "b")) -> dict[str, RepoVectors]:
    vec = [1.0, 0.5, -0.25, 0.1]
    return {
        repo: RepoVectors.from_mapping({f"{repo}{i:03d}": list(vec) for i in range(rows_per_repo)})
        for repo in repos
    }


def test_env_overrides_default(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_MESH_EXACT_MAX_ROWS", "120000")
    assert _exact_max_rows_from_env() == 120000


def test_unset_uses_default(monkeypatch):
    monkeypatch.delenv("GRAPHIFY_MESH_EXACT_MAX_ROWS", raising=False)
    assert _exact_max_rows_from_env() == DEFAULT


@pytest.mark.parametrize("raw", ["abc", "", "1", "0", "-5"])
def test_bad_value_falls_back_with_warning(monkeypatch, caplog, raw):
    monkeypatch.setenv("GRAPHIFY_MESH_EXACT_MAX_ROWS", raw)
    with caplog.at_level("WARNING"):
        assert _exact_max_rows_from_env() == DEFAULT
    assert "GRAPHIFY_MESH_EXACT_MAX_ROWS" in caplog.text


def test_warning_fires_above_90_percent_only(caplog):
    vectors = _vectors(5)  # 10 rows
    with caplog.at_level("WARNING"):
        embed_similarity.mutual_top_k_pairs(vectors, top_k=3, threshold=0.5, exact_max_rows=12)
    assert "is 83% of EXACT_MAX_ROWS=12" not in caplog.text
    assert "of EXACT_MAX_ROWS" not in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        embed_similarity.mutual_top_k_pairs(vectors, top_k=3, threshold=0.5, exact_max_rows=11)
    assert (
        "similar_approach: 10 rows is 91% of EXACT_MAX_ROWS=11; above it overlay falls "
        "back to LSH and the pair cache is not used"
    ) in caplog.text


def test_no_near_limit_warning_at_exactly_90_percent(caplog):
    vectors = _vectors(3, ("a", "b", "c"))  # 9 rows, limit 10: 9 is not above 90%
    with caplog.at_level("WARNING"):
        embed_similarity.mutual_top_k_pairs(vectors, top_k=3, threshold=0.5, exact_max_rows=10)
    assert "% of EXACT_MAX_ROWS=" not in caplog.text


def test_no_near_limit_warning_above_the_limit(caplog):
    vectors = _vectors(5)  # 10 rows, limit 9: LSH path
    with caplog.at_level("WARNING"):
        embed_similarity.mutual_top_k_pairs(vectors, top_k=3, threshold=0.5, exact_max_rows=9)
    assert "% of EXACT_MAX_ROWS=" not in caplog.text


def test_exact_at_limit_lsh_above(monkeypatch):
    vectors = _vectors(5)  # 10 rows
    calls = []
    real = embed_similarity._exact_pairs

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(embed_similarity, "_exact_pairs", spy)
    embed_similarity.mutual_top_k_pairs(vectors, top_k=3, threshold=0.5, exact_max_rows=10)
    assert calls == [1]
    embed_similarity.mutual_top_k_pairs(vectors, top_k=3, threshold=0.5, exact_max_rows=9)
    assert calls == [1]


def test_comparable_row_count_matches_decision_with_mixed_dims(caplog):
    vectors = _vectors(5)  # a, b: dim 4, 10 rows
    vectors["c"] = RepoVectors.from_mapping({f"c{i}": [1.0, 0.0, 0.0] for i in range(7)})
    count = embed_similarity.comparable_row_count(vectors)
    assert count == 10
    with caplog.at_level("WARNING"):
        embed_similarity.mutual_top_k_pairs(vectors, top_k=3, threshold=0.5, exact_max_rows=11)
    assert f"similar_approach: {count} rows is 91% of EXACT_MAX_ROWS=11" in caplog.text
