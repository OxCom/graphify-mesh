"""--skip-embedding publishes the previous shards under the new generation id.

It used to persist nothing: `embeddings/current` kept the previous generation
id, and the server (`server/store.py` `_load_embeddings_checked`) dropped the
whole vector channel on `embeddings_generation_mismatch`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from graphify_mesh.sync import embedding
from graphify_mesh.sync.pipeline import run

REPO = "example-org.styleguide"
SHARD_META = f"{REPO}.meta.json"


def _repo(env, monkeypatch) -> Path:
    root = env.add_repo(REPO, "example-org", "styleguide", "styleguide.example-org.dev.lo")
    env.write_registry()
    monkeypatch.setattr(
        embedding,
        "embed_batch",
        lambda base_url, model, inputs, timeout=30.0: [[1.0, 0.0] for _ in inputs],
    )
    return root


def _healthy() -> dict:
    return {"ollama_embed_health_check": lambda *a, **kw: True}


def _current(env) -> Path:
    return Path(os.path.realpath(env.settings().embeddings_current_symlink))


def _entries(gen_dir: Path) -> dict:
    return json.loads((gen_dir / SHARD_META).read_text(encoding="utf-8"))["entries"]


def test_skip_embedding_carries_the_shard_into_the_new_generation(env, monkeypatch):
    _repo(env, monkeypatch)
    first = run(env.settings(**_healthy()))
    assert first.published
    assert _current(env).name == first.generation_id
    vectors = _entries(_current(env))
    assert vectors

    skipped = run(env.settings(skip_embedding=True))

    assert skipped.published
    assert skipped.generation_id != first.generation_id
    assert _current(env).name == skipped.generation_id, "server would drop the vector channel"
    assert _entries(_current(env)) == vectors
    assert skipped.embedding_status == "skipped (--skip-embedding)"


def test_skip_embedding_with_nothing_published_still_matches_the_generation(env, monkeypatch):
    _repo(env, monkeypatch)

    skipped = run(env.settings(skip_embedding=True))

    assert skipped.published
    assert _current(env).name == skipped.generation_id
    assert not (_current(env) / SHARD_META).exists(), "no empty shard file is written"


def test_dry_run_persists_no_embeddings(env, monkeypatch):
    _repo(env, monkeypatch)
    first = run(env.settings(**_healthy()))

    run(env.settings(skip_embedding=True, dry_run=True))

    assert _current(env).name == first.generation_id


def test_degraded_run_also_matches_the_generation(env, monkeypatch):
    _repo(env, monkeypatch)
    run(env.settings(**_healthy()))
    vectors = _entries(_current(env))

    degraded = run(env.settings())

    assert degraded.published
    assert degraded.embedding_status == embedding.EMBED_DEGRADED
    assert _current(env).name == degraded.generation_id
    assert _entries(_current(env)) == vectors


def test_normal_run_after_a_skip_run_re_embeds_the_body_changed_in_the_skip_run(env, monkeypatch):
    # A skip run carries the old vectors forward and records the new source
    # digest, so the next normal run sees the repo as unchanged. Whole-shard
    # reuse must still notice the stale content hashes, or the old vectors
    # stay forever.
    root = _repo(env, monkeypatch)
    # Same nodes after the body edit: the key set alone cannot reveal the change.
    env.set_control(env.collection_path("example-org", "styleguide"), "noop")
    source = root / "src" / "alpha.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("class AlphaClass:\n    def run(self):\n        return 1\n", encoding="utf-8")
    first = run(env.settings(**_healthy()))
    assert first.published
    assert first.embedding_stats["embedded"] > 0

    source.write_text(
        "class AlphaClass:\n    def run(self):\n        return 22\n", encoding="utf-8"
    )
    skipped = run(env.settings(skip_embedding=True))
    assert skipped.published
    carried = _entries(_current(env))

    embedded_inputs: list[str] = []

    def counting_embed(base_url, model, inputs, timeout=30.0):
        embedded_inputs.extend(inputs)
        return [[0.0, 1.0] for _ in inputs]

    monkeypatch.setattr(embedding, "embed_batch", counting_embed)
    normal = run(env.settings(**_healthy()))

    assert normal.published
    assert normal.embedding_stats["reused_repos_unchanged"] == 0
    changed = {k for k, v in _entries(_current(env)).items() if v != carried.get(k)}
    assert changed, "content hashes must move to the new body"
    assert len(embedded_inputs) == len(changed)
    assert all("return 22" in text for text in embedded_inputs)

    embedded_inputs.clear()
    again = run(env.settings(**_healthy()))
    assert again.embedding_stats["reused_repos_unchanged"] == 1
    assert embedded_inputs == []
