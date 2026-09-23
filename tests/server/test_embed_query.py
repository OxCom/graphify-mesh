"""Query-time embedding guards: short timeout, negative cache, and the
classified failure reason that reaches the response's `degraded` field.

The failure these cover, measured 2026-09-23: a backend whose control plane
answered `/api/tags` 200 in 0.07 s while every `/api/embed` call hung to the
timeout, 7 attempts out of 7. Before the negative cache each query paid the
full timeout again, so the cost scaled with query count.
"""

from __future__ import annotations

import pytest

from graphify_mesh.server import embed_query as eq
from graphify_mesh.server.retrieval import _embeddings_degraded_marker


class _Recorder:
    """Stand-in for `sync.embedding.embed_batch` that counts calls."""

    def __init__(self, behavior):
        self.calls = 0
        self.behavior = behavior

    def __call__(self, base_url, model, inputs, timeout=None):
        self.calls += 1
        return self.behavior(inputs)


def _timeout(_inputs):
    raise RuntimeError(
        "embed_batch request to https://backend:11434/api/embed failed: "
        "<urlopen error timed out>"
    )


def _refused(_inputs):
    raise RuntimeError(
        "embed_batch request to https://backend:11434/api/embed failed: "
        "<urlopen error [Errno 111] Connection refused>"
    )


def _shape_mismatch(_inputs):
    raise RuntimeError(
        "embed_batch response shape mismatch from https://backend:11434/api/embed: "
        "expected 1 embeddings, got 0"
    )


@pytest.fixture()
def embedder(monkeypatch):
    def _make(behavior, cooldown=60.0):
        rec = _Recorder(behavior)
        monkeypatch.setattr(eq, "embed_batch", rec)
        fn = eq.make_embed_query_fn(
            base_url="https://backend:11434", model="m", timeout=1.0, cooldown=cooldown
        )
        return fn, rec

    return _make


def test_repeated_failures_hit_the_backend_once_per_cooldown(embedder):
    fn, rec = embedder(_timeout)

    assert [fn(f"query {i}") for i in range(10)] == [None] * 10
    # The whole point: 10 degraded queries, one timeout paid.
    assert rec.calls == 1


def test_cooldown_expiry_lets_the_backend_be_retried(embedder, monkeypatch):
    fn, rec = embedder(_timeout, cooldown=30.0)
    clock = {"t": 1000.0}
    monkeypatch.setattr(eq.time, "monotonic", lambda: clock["t"])

    assert fn("q") is None
    assert fn("q") is None
    assert rec.calls == 1

    clock["t"] += 31.0
    assert fn("q") is None
    assert rec.calls == 2


def test_recovery_clears_the_cooldown_and_reason(embedder, monkeypatch):
    state = {"down": True}

    def flaky(_inputs):
        if state["down"]:
            _timeout(_inputs)
        return [[0.1, 0.2]]

    fn, rec = embedder(flaky, cooldown=30.0)
    clock = {"t": 1000.0}
    monkeypatch.setattr(eq.time, "monotonic", lambda: clock["t"])

    assert fn("q") is None
    assert fn.failure_reason() == eq.REASON_TIMEOUT

    state["down"] = False
    clock["t"] += 31.0
    assert fn("q") == [0.1, 0.2]
    assert fn.last_failure_reason is None
    # A healthy backend must not be throttled by a stale cooldown.
    assert fn("q") == [0.1, 0.2]
    assert rec.calls == 3


@pytest.mark.parametrize(
    ("behavior", "expected"),
    [
        (_timeout, eq.REASON_TIMEOUT),
        (_refused, eq.REASON_UNREACHABLE),
        (_shape_mismatch, eq.REASON_BAD_RESPONSE),
    ],
)
def test_failure_reason_is_classified(embedder, behavior, expected):
    fn, _ = embedder(behavior)
    assert fn("q") is None
    assert fn.failure_reason() == expected


def test_blank_query_never_reaches_the_backend(embedder):
    fn, rec = embedder(_timeout)
    assert fn("   ") is None
    assert rec.calls == 0


def test_timeout_env_override_rejects_nonsense(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_MESH_QUERY_EMBED_TIMEOUT", "0")
    assert eq.make_embed_query_fn().timeout == eq.DEFAULT_QUERY_EMBED_TIMEOUT
    monkeypatch.setenv("GRAPHIFY_MESH_QUERY_EMBED_TIMEOUT", "not-a-number")
    assert eq.make_embed_query_fn().timeout == eq.DEFAULT_QUERY_EMBED_TIMEOUT
    monkeypatch.setenv("GRAPHIFY_MESH_QUERY_EMBED_TIMEOUT", "1.5")
    assert eq.make_embed_query_fn().timeout == 1.5


def test_default_timeout_is_short_enough_to_not_dominate_a_query():
    # A warm single-query embed measured 0.19-0.28 s; a 10 s budget meant a
    # wedged backend added 10 s to every degraded query.
    assert eq.DEFAULT_QUERY_EMBED_TIMEOUT <= 3.0


def test_degraded_marker_distinguishes_cause(embedder):
    fn, _ = embedder(_timeout)
    embeddings = {"repo.a": object()}

    # Nothing published: the bare marker `store` also uses for this condition.
    assert _embeddings_degraded_marker({}, fn) == "embeddings_unavailable"

    # Published vectors + a failed query embed: the reason is carried.
    assert fn("q") is None
    assert _embeddings_degraded_marker(embeddings, fn) == "embeddings_unavailable:timeout"

    # A plain callable exposes no reason, so the bare marker is used.
    assert _embeddings_degraded_marker(embeddings, lambda q: None) == "embeddings_unavailable"
