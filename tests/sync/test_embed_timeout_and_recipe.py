"""Embed request timeout setting, recipe stamp v2, and the stale-shard log.

The real backend answers one input in ~35 s and 128 inputs in ~78 s, so the
old hardcoded 30 s request timeout failed the embed stage on every run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from graphify_mesh.sync import embedding
from graphify_mesh.sync.config import EMBED_DEFAULT_REQUEST_TIMEOUT, Settings
from graphify_mesh.sync.pipeline import run

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
FAKE_GRAPHIFY = FIXTURES_DIR / "fake_graphify" / "graphify"
ENV_VAR = "GRAPHIFY_MESH_EMBED_TIMEOUT"
KEY = "repo.a\x1fsrc/w.py\x1fWidget"


def _settings(tmp_path: Path, **overrides) -> Settings:
    mesh_root = tmp_path / "mesh"
    return Settings.from_env(
        mesh_root=mesh_root,
        scan_roots=[tmp_path / "www"],
        registry_path=mesh_root / "bin" / "registry.json",
        graphify_bin=str(FAKE_GRAPHIFY),
        **overrides,
    )


def _graph() -> dict:
    return {"nodes": [{"id": "n1", "label": "Widget", "source_file": "src/w.py"}]}


class TestEmbedTimeoutSetting:
    def test_default_is_120_seconds(self, tmp_path, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        assert EMBED_DEFAULT_REQUEST_TIMEOUT == 120.0
        assert _settings(tmp_path).embed_request_timeout == 120.0

    def test_env_overrides(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "300.5")
        assert _settings(tmp_path).embed_request_timeout == 300.5

    @pytest.mark.parametrize("raw", ["abc", "0", "-1", "3601", "nan", "inf"])
    def test_invalid_values_fail_at_startup(self, tmp_path, monkeypatch, raw):
        monkeypatch.setenv(ENV_VAR, raw)
        with pytest.raises(ValueError, match=ENV_VAR):
            _settings(tmp_path)

    def test_upper_bound_is_accepted(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "3600")
        assert _settings(tmp_path).embed_request_timeout == 3600.0

    def test_stage_passes_the_setting_to_every_embed_call(self, tmp_path, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        seen: list[float] = []

        def recording(base_url, model, inputs, timeout=30.0):
            seen.append(timeout)
            return [[1.0] for _ in inputs]

        monkeypatch.setattr(embedding, "embed_batch", recording)
        result = embedding.run_embedding_stage(
            graph_paths_by_repo={},
            repo_roots_by_id={"repo.a": tmp_path},
            graphs_by_repo={"repo.a": _graph()},
            unchanged_repo_ids=set(),
            settings=_settings(tmp_path, embed_request_timeout=77.0),
            provisional_generation_id="gen-1",
            health_check=lambda base_url, timeout: True,
        )

        assert result.stats.embedded == 1
        assert seen == [77.0]


class TestRecipeStampV2:
    def test_stamp_version_is_2(self):
        assert embedding.RECIPE_STAMP_VERSION == 2
        assert embedding.EmbeddingRecipe(model="m", dim=0).identity()["stamp_version"] == 2

    def test_v1_stamped_shard_is_re_embedded_once(self, tmp_path, monkeypatch):
        """Shards published under stamp v1 hold vectors of the pre-21b2dfa input:
        the bump must re-embed them once, and the run after reuses the result."""
        monkeypatch.setattr(embedding, "RECIPE_STAMP_VERSION", 1)
        v1_stamp = embedding.recipe_stamp_for_model("m")
        monkeypatch.undo()
        current = embedding.recipe_stamp_for_model("m")
        assert v1_stamp != current

        calls = {"n": 0}

        def counting(base_url, model, inputs, timeout=30.0):
            calls["n"] += 1
            return [[2.0] for _ in inputs]

        monkeypatch.setattr(embedding, "embed_batch", counting)

        def shard_from(previous: embedding.RepoShard) -> embedding.RepoShard:
            return embedding.compute_repo_shard(
                "repo.a",
                _graph(),
                tmp_path,
                previous_shard=previous,
                base_url="https://host",
                model="m",
                stats=embedding.EmbeddingStats(),
                recipe_stamp=current,
            )

        old = shard_from(embedding.RepoShard.empty())
        old = embedding.RepoShard(
            entries=old.entries, vectors=old.vectors, recipe_stamp=v1_stamp, recipe=None
        )
        calls["n"] = 0

        rebuilt = shard_from(old)
        assert calls["n"] == 1, "v1-stamped shard must be re-embedded"
        shard_from(rebuilt)
        assert calls["n"] == 1, "the re-embedded shard is reused on the next run"


class TestStaleShardLog:
    def test_missing_shard_ids_are_told_apart_from_recipe_changes(self, tmp_path):
        gen = tmp_path / "gen"
        gen.mkdir()
        assert embedding.missing_shard_repo_ids(gen, {"repo.a"}) == {"repo.a"}
        assert embedding.missing_shard_repo_ids(None, {"repo.a"}) == {"repo.a"}
        (gen / "repo.a.meta.json").write_text("{}", encoding="utf-8")
        (gen / "repo.b.json").write_text("{}", encoding="utf-8")
        assert embedding.missing_shard_repo_ids(gen, {"repo.a", "repo.b", "repo.c"}) == {"repo.c"}

    def test_pipeline_logs_no_shard_not_recipe_differs(self, env, monkeypatch, caplog):
        env.add_repo(
            "example-org.styleguide",
            "example-org",
            "styleguide",
            "styleguide.example-org.dev.lo",
            "repo_a.json",
        )
        env.write_registry()
        monkeypatch.setattr(
            embedding,
            "embed_batch",
            lambda base_url, model, inputs, timeout=30.0: [[1.0, 0.0] for _ in inputs],
        )
        healthy = {"ollama_embed_health_check": lambda *a, **kw: True}
        # A first run without embedding publishes no shard at all.
        assert run(env.settings(skip_embedding=True)).published

        with caplog.at_level(logging.INFO, logger="graphify_mesh.sync"):
            assert run(env.settings(**healthy)).published

        messages = [r.getMessage() for r in caplog.records]
        assert not any("recipe differs" in m for m in messages), messages
        assert any("no published shard" in m and "example-org.styleguide" in m for m in messages), (
            messages
        )
