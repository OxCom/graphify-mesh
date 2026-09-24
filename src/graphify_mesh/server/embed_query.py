"""Query-time embedding wrapper (WS5): turns a raw query string into a
vector for `retrieval.vector_candidates`, reusing the exact `/api/embed`
contract and base_url/model config `graphify_mesh.sync.embedding` /
`graphify_mesh.sync.config` already established for the sync pipeline (C9) — the
query MUST be embedded with the same model/endpoint that produced the
published vectors, or cosine similarity is meaningless.

Never raises: any transport/shape failure degrades to `None` (the caller,
`retrieval.vector_candidates`, surfaces an `"embeddings_unavailable:<reason>"`
marker in the response's `degraded` field), mirroring the pipeline's own
"never crash on a down local service" convention (C9) — just applied at query
time instead of build time.

Cost of degrading (2026-09-23): a backend can accept TCP and answer its
control-plane routes while every inference request hangs. Measured against
the then-configured host: `/api/tags` 200 in 0.07 s, `/api/embed` timing out
on 7 of 7 attempts. A per-query blocking call with a long timeout therefore
costs the full timeout on EVERY query for as long as the backend stays in
that state. Two guards below bound that cost:

  * a short default timeout — a warm single-query embed measured 0.19-0.28 s
    against a local backend, so the budget here is headroom, not a limit on
    healthy operation;
  * a negative cache — one failure suppresses the next `FAILURE_COOLDOWN`
    seconds of calls in this process, so N degraded queries cost one timeout
    per cooldown window instead of N timeouts.

Neither guard is a health probe. A startup probe against `/api/tags` would
have reported this backend HEALTHY (it answered 200 in 0.07 s while every
embed hung), so the only honest signal is a real embed call's outcome, which
is what the negative cache records.
"""

from __future__ import annotations

import logging
import os
import threading
import time

from graphify_mesh.sync.config import EMBED_DEFAULT_BASE_URL, EMBED_DEFAULT_MODEL
from graphify_mesh.sync.embedding import embed_batch

log = logging.getLogger("graphify_mesh.server.embed_query")

# Kept short: a query-time embed call blocks one MCP `search`/`context_pack`
# invocation, unlike the sync pipeline's bulk embed calls which can afford a
# longer per-batch timeout. A healthy backend answers a single-input embed in
# well under a second (0.19-0.28 s measured warm), so this is ~10x headroom;
# what it actually bounds is the cost of a wedged backend.
DEFAULT_QUERY_EMBED_TIMEOUT = 2.5
# After a failure, skip the backend entirely for this long. Sized to be far
# longer than the timeout (so the amortized cost of a dead backend is near
# zero) but short enough that a backend coming back is picked up within a
# minute without restarting the daemon.
DEFAULT_FAILURE_COOLDOWN = 60.0

# Kept as a module constant for callers/tests that imported it before the
# env override existed.
QUERY_EMBED_TIMEOUT = DEFAULT_QUERY_EMBED_TIMEOUT

# Why the vector retriever contributed nothing. `retrieval` prefixes these
# with `embeddings_unavailable:` so a client can tell a wedged/unreachable
# backend apart from a misconfiguration without reading the server log.
REASON_TIMEOUT = "timeout"
REASON_UNREACHABLE = "unreachable"
REASON_BAD_RESPONSE = "bad_response"


def _positive_float_env(name: str, default: float) -> float:
    """Read a positive float from the environment, falling back to `default`
    on anything unparseable or non-positive. A bad value must not be able to
    turn the timeout into 0 (every query degrades) or a negative cooldown."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        log.warning("%s=%r is not a number, using %s", name, raw, default)
        return default
    if value <= 0:
        log.warning("%s=%r is not positive, using %s", name, raw, default)
        return default
    return value


def _classify(exc: Exception) -> str:
    """Map an `embed_batch` RuntimeError onto one of the REASON_* constants.

    `embed_batch` flattens every transport failure into one RuntimeError whose
    message carries the original exception's text, so the message is the only
    signal available without changing that function's contract."""
    text = str(exc).lower()
    if "timed out" in text or "timeout" in text:
        return REASON_TIMEOUT
    if "shape mismatch" in text or "not a json object" in text or "too large" in text:
        return REASON_BAD_RESPONSE
    return REASON_UNREACHABLE


class QueryEmbedder:
    """Callable matching `retrieval.EmbedQueryFn` (`str -> list[float] | None`),
    plus a `last_failure_reason` attribute the caller reads via `getattr` so
    plain-callable stand-ins in tests keep satisfying the same contract."""

    def __init__(self, base_url: str, model: str, timeout: float, cooldown: float):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.cooldown = cooldown
        self.last_failure_reason: str | None = None
        # Guards the cooldown deadline and the reason: the MCP server can
        # serve concurrent requests, and without this two queries racing on a
        # dead backend would each pay a full timeout.
        self._lock = threading.Lock()
        self._cooldown_until = 0.0

    def _cooling_down(self) -> bool:
        with self._lock:
            return time.monotonic() < self._cooldown_until

    def _record_failure(self, reason: str) -> None:
        with self._lock:
            self._cooldown_until = time.monotonic() + self.cooldown
            self.last_failure_reason = reason

    def _record_success(self) -> None:
        with self._lock:
            self._cooldown_until = 0.0
            self.last_failure_reason = None

    def __call__(self, query: str) -> list[float] | None:
        if not query or not query.strip():
            return None
        if self._cooling_down():
            # Deliberately silent: the failure that opened the cooldown was
            # already logged once. Logging here would restore exactly the
            # per-query noise the cooldown exists to remove.
            return None
        try:
            vectors = embed_batch(self.base_url, self.model, [query], timeout=self.timeout)
        except RuntimeError as exc:
            reason = _classify(exc)
            self._record_failure(reason)
            log.warning(
                "graphify-mesh: query embedding unavailable (%s), degrading to "
                "lexical+structural only for the next %.0fs: %s",
                reason,
                self.cooldown,
                exc,
            )
            return None
        if not vectors:
            self._record_failure(REASON_BAD_RESPONSE)
            log.warning(
                "graphify-mesh: query embedding returned no vector (%s), degrading to "
                "lexical+structural only for the next %.0fs",
                REASON_BAD_RESPONSE,
                self.cooldown,
            )
            return None
        self._record_success()
        return vectors[0]

    def failure_reason(self) -> str:
        """The reason to report when this embedder just declined to produce a
        vector. A suppressed call reports the failure that opened the cooldown
        rather than `cooling_down`: the cause is what the client needs, and
        the log line already distinguishes the attempt from the suppression."""
        with self._lock:
            return self.last_failure_reason or REASON_UNREACHABLE


def make_embed_query_fn(
    base_url: str | None = None,
    model: str | None = None,
    timeout: float | None = None,
    cooldown: float | None = None,
) -> QueryEmbedder:
    """Returns a `str -> list[float] | None` callable matching
    `retrieval.EmbedQueryFn`, bound to the resolved base_url/model (env
    override via the SAME `GRAPHIFY_MESH_OLLAMA_EMBED_*` variables the sync
    pipeline reads, so a query never drifts onto a different embedding
    space than the one the published vectors were built with).

    Timeout and cooldown are read from `GRAPHIFY_MESH_QUERY_EMBED_TIMEOUT` /
    `GRAPHIFY_MESH_QUERY_EMBED_COOLDOWN` so a wedged backend can be ridden out
    by editing the unit, without promoting a new build."""
    resolved_base_url = base_url or os.environ.get(
        "GRAPHIFY_MESH_OLLAMA_EMBED_BASE_URL", EMBED_DEFAULT_BASE_URL
    )
    resolved_model = model or os.environ.get(
        "GRAPHIFY_MESH_OLLAMA_EMBED_MODEL", EMBED_DEFAULT_MODEL
    )
    resolved_timeout = (
        timeout
        if timeout is not None
        else _positive_float_env("GRAPHIFY_MESH_QUERY_EMBED_TIMEOUT", DEFAULT_QUERY_EMBED_TIMEOUT)
    )
    resolved_cooldown = (
        cooldown
        if cooldown is not None
        else _positive_float_env("GRAPHIFY_MESH_QUERY_EMBED_COOLDOWN", DEFAULT_FAILURE_COOLDOWN)
    )
    return QueryEmbedder(resolved_base_url, resolved_model, resolved_timeout, resolved_cooldown)
