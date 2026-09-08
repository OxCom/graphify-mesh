"""WS3 embedding index: snippet builder, batched native `/api/embed` calls,
id-map + tombstones (C27), sharded/resumable storage, and GC.

Runs AFTER the WS2 naming/label stage and BEFORE WS4 overlay-resolve (plan
WS1 item 6 order: `... -> label -> embed changed -> overlay resolve -> ...`).

Design notes (read before changing anything here):

  * C6 (embedding input): every embedded node's input string is
    `label + bounded source snippet + path + community name` — see
    `build_embedding_input`. Nodes carry no context text by default, so the
    snippet is read directly from `source_file` on disk, bounded to a small
    line/char window (`SNIPPET_WINDOW_LINES`/`SNIPPET_MAX_CHARS`) so this
    never turns into an unbounded file read.

  * C9 (native contract): batched calls hit `{base_url}/api/embed` — the
    NATIVE Ollama endpoint, NOT the `/v1` OpenAI-compat surface the WS2
    naming stage's LLM calls use. Verified by a real curl during WS3 build
    (see config.py's EMBED_DEFAULT_* docstring for the exact request/response
    shape). Never reuse naming.py's `default_ollama_health_check` here — it
    targets `/v1/models`, a different API surface entirely; this module has
    its own `default_embed_health_check` against the native `/api/tags`.

  * C27 (id-map + tombstones): `build_id_map` rebuilds the id-map to the
    EXACT current node-key set every generation. A key that existed in the
    previous id-map as "active" but is absent from the current run's key set
    is marked "tombstoned" (with the generation_id it disappeared in) rather
    than silently dropped. Keys are `LogicalRef`-shaped
    (repo, source_file, qualified_label) — the same durable-reference shape
    WS4's overlay module already uses — never a raw graphify node id, which
    is not stable across generations.

  * Resumability / changed-only: the pipeline already knows, per repo, which
    repos were untouched this run (`decide_action` -> ACTION_SKIP, see
    sync_project.py). `run_embedding_stage` is handed that same set
    (`unchanged_repo_ids`) and reuses the ENTIRE previous shard for those
    repos without recomputing anything. For repos that did change, each
    node's embedding input is content-hashed; a node whose hash matches the
    previous published shard's entry is reused, otherwise it is
    (re)embedded. This deliberately reuses WS1's existing changed-detection
    signal instead of re-deriving one.

  * Skip heuristic: trivial one-line getters/setters (see `is_trivial_node`)
    are never embedded — the embedding call budget is spent on nodes likely
    to carry distinguishing semantic content. Skipped nodes carry no vector
    at all in the shard; `overlay_similar.py`'s fallback (exact label +
    same-community match) is the documented behavior for them (deliverable
    7) rather than leaving them silently unhandled.

  * Storage/GC: shards are staged into the per-run ephemeral tempdir during
    the pipeline run (`stage_embeddings`) and only copied into the permanent,
    untracked `settings.embeddings_dir/generations/<generation_id>/` tree
    (plus the `current` symlink flip and GC of old generations) once the
    surrounding pipeline run actually publishes (`persist_generation`) — the
    same "nothing durable happens unless publish happens" rule the rest of
    the pipeline already follows for naming/overlay artifacts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from graphify_mesh.sync.config import Settings, is_valid_http_base_url
from graphify_mesh.sync.overlay_refs import LogicalRef
from graphify_mesh.sync.publish import _fsync_dir
from graphify_mesh.sync.source_cache import get_source_lines
from graphify_mesh.sync.tls import ssl_context
from graphify_mesh.sync.vectors import RepoVectors

log = logging.getLogger("graphify_mesh.sync.embedding")

EMBED_HEALTHY = "ok"
EMBED_DEGRADED = "degraded"
# Health check passed (server was up), but a real embed_batch call failed
# mid-run (network blip, host went down partway through, etc). Distinct from
# EMBED_DEGRADED (never even attempted this run) so the generation manifest
# is honest about "some new vectors landed, then it degraded" vs "skipped
# entirely" — both are recoverable next run, but they're not the same event.
EMBED_PARTIAL = "partial"

# C6 snippet window: bounded read around the node's line (if known), or the
# file's head if no line is available. Kept small and fixed rather than
# reading whole files — nodes have no context text by default per C6, and a
# bounded window is the whole point of the constraint.
SNIPPET_WINDOW_LINES = 20
SNIPPET_MAX_CHARS = 800
# Hard cap on how many lines of a source file are ever scanned to locate the
# window, so a single pathologically large file can't blow up one embed run.
SNIPPET_READ_LINE_CAP = 4000

# C6 final embedding input: label + snippet + path + community name,
# truncated to this many chars as a last-resort guard regardless of how the
# pieces above were individually bounded.
EMBED_INPUT_MAX_CHARS = 3000

# Skip heuristic: trivial one-line accessors are not worth an embedding call.
# A node is "trivial" iff its label matches this accessor-naming pattern AND
# its snippet (once built) has at most TRIVIAL_MAX_SNIPPET_LINES non-blank
# lines — i.e. it really is a bare one-liner, not just an accessor-named
# method with a meaningful body.
TRIVIAL_LABEL_PATTERN = re.compile(r"^(get|set|is|has)[A-Z_]\w*$")
TRIVIAL_MAX_SNIPPET_LINES = 2

EMBED_BATCH_SIZE = 16
EMBED_REQUEST_TIMEOUT = 30.0
# One bounded retry per failed batch: a single transient Ollama blip used to
# degrade the current repo AND every repo after it to their previous shards
# (see run_embedding_stage's EMBED_PARTIAL fallback), discarding vectors the
# run had already computed. A short backoff + one retry absorbs the common
# one-off blip; a second consecutive failure still declares EMBED_PARTIAL.
EMBED_BATCH_RETRIES = 1
EMBED_RETRY_BACKOFF_SECONDS = 2.0

# Shard format v2: `<repo>.meta.json` (entries + dim + format marker) +
# `<repo>.npy` (float32 matrix, row i belongs to the key whose entry says
# `"row": i`). v1 (`<repo>.json` with an inline `embedding` list per entry)
# is still read for back-compat; SUPPORTED_SHARD_FORMATS is the meta.json
# `shard_format` allowlist read_previous_shard will accept before trusting a
# v2 shard's matrix at all.
SHARD_FORMAT_VERSION = 2
SUPPORTED_SHARD_FORMATS = frozenset({1, 2})
SHARD_META_SUFFIX = ".meta.json"
SHARD_MATRIX_SUFFIX = ".npy"

HealthCheckFn = Callable[[str, float], bool]


@dataclass
class EmbeddingRecipe:
    """C28: the exact recipe recorded in the generation manifest so a future
    generation (or a human debugging drift) can tell what produced a vector
    without re-deriving it from source."""

    model: str
    dim: int
    snippet_window_lines: int = SNIPPET_WINDOW_LINES
    snippet_max_chars: int = SNIPPET_MAX_CHARS
    input_max_chars: int = EMBED_INPUT_MAX_CHARS
    skip_heuristic: str = (
        f"trivial accessor skip: label matches {TRIVIAL_LABEL_PATTERN.pattern!r} "
        f"AND snippet has <= {TRIVIAL_MAX_SNIPPET_LINES} non-blank lines"
    )
    tokenizer_notes: str = (
        "no client-side tokenization/truncation beyond EMBED_INPUT_MAX_CHARS char cap; "
        "the /api/embed backend applies its own model-native tokenizer server-side"
    )

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "dim": self.dim,
            "snippet_window_lines": self.snippet_window_lines,
            "snippet_max_chars": self.snippet_max_chars,
            "input_max_chars": self.input_max_chars,
            "skip_heuristic": self.skip_heuristic,
            "tokenizer_notes": self.tokenizer_notes,
        }


@dataclass
class EmbeddingStats:
    embedded: int = 0
    skipped_trivial: int = 0
    reused: int = 0
    reused_repos_unchanged: int = 0
    tombstoned: int = 0

    def to_dict(self) -> dict:
        return {
            "embedded": self.embedded,
            "skipped_trivial": self.skipped_trivial,
            "reused": self.reused,
            "reused_repos_unchanged": self.reused_repos_unchanged,
            "tombstoned": self.tombstoned,
        }


@dataclass
class RepoShard:
    """In-RAM shard for one repo: durable entry bookkeeping (content_hash +
    which matrix row, if any, holds this key's vector) plus the actual
    vectors as a `RepoVectors` matrix. `entries[key]` is
    `{"content_hash": str | None, "row": int | None}` — `"row": None` means
    this key carries no vector at all (trivial-skip), matching v1's
    `embedding: None` marker."""

    entries: dict[str, dict] = field(default_factory=dict)
    vectors: RepoVectors = field(default_factory=RepoVectors.empty)

    @classmethod
    def empty(cls) -> RepoShard:
        return cls(entries={}, vectors=RepoVectors.empty())


@dataclass
class EmbeddingStageResult:
    status: str  # EMBED_HEALTHY | EMBED_DEGRADED | EMBED_PARTIAL
    # Vector-only view per repo (the `overlay_similar.py` seam) — a
    # `RepoVectors` matrix, not a raw `{key: [float, ...]}` mapping.
    vectors_by_repo: dict[str, RepoVectors] = field(default_factory=dict)
    # Full shard (entries + vectors) per repo — this is what actually gets
    # persisted to disk (`stage_embeddings`/`persist_generation`), so a
    # future run can resume from content_hash comparisons even for nodes
    # that carry no vector.
    shards_by_repo: dict[str, RepoShard] = field(default_factory=dict)
    stats: EmbeddingStats = field(default_factory=EmbeddingStats)
    recipe: EmbeddingRecipe | None = None
    id_map: dict[str, dict] = field(default_factory=dict)
    reason: str = ""


def key_to_ref(key: str) -> LogicalRef:
    """Inverse of `LogicalRef.to_key()` — splits an embedding shard/id-map
    key back into its logical-ref parts for building an OverlayEdge."""
    repo, source_file, label = key.split("\x1f", 2)
    return LogicalRef(repo=repo, source_file=source_file, qualified_label=label)


def node_key(repo_id: str, node: dict) -> str | None:
    """Durable logical key for a node, matching the `LogicalRef` shape WS4's
    overlay module already uses (C27) — never the raw per-repo graphify node
    id, which is not stable across generations."""
    source_file = node.get("source_file")
    label = node.get("label")
    if not source_file or not label:
        return None
    return LogicalRef(repo=repo_id, source_file=source_file, qualified_label=label).to_key()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# graphify writes a node's position as `source_location: "L34"`, never as a `line`
# field — so every consumer that asked for node["line"] received None, build_snippet
# fell back to `start = 0`, and every evidence card carried the first 20 lines of its
# file. For PHP and TypeScript that is the import block: ~800 characters of near-zero
# signal per card. Citations lost the position too and rendered as `path:?`, forcing a
# second locator call and making `grep -n` strictly cheaper than asking the graph.
# Measured consequence: 4 graph calls across 12 benchmark cells, 8 of which never
# called it at all. 4364 of this repo's 5499 nodes carry a usable source_location.
_SOURCE_LOCATION_LINE = re.compile(r"^L(\d+)$")


def node_line(node: dict) -> int | None:
    """The node's 1-based line, from `line` or from graphify's `source_location`."""
    value = node.get("line")
    if isinstance(value, int) and value > 0:
        return value
    match = _SOURCE_LOCATION_LINE.match(str(node.get("source_location") or "").strip())
    if not match:
        return None
    # "L0" is not a line: lines are 1-based, and 0 is falsy downstream, so it would slip
    # back into the head-of-file fallback this helper exists to remove.
    parsed = int(match.group(1))
    return parsed if parsed > 0 else None


def build_snippet(source_root: Path | None, source_file: str | None, line: int | None) -> str:
    """C6 bounded source snippet. Returns "" (not an error) whenever a
    snippet cannot be produced — missing root, missing file, unreadable file,
    or a `source_file` that escapes `source_root` (absolute path or `..`
    traversal; graph.json content is treated as hostile input, per the
    public-package threat model) — embedding still proceeds with
    label/path/community alone."""
    if source_root is None or not source_file:
        return ""
    if Path(source_file).is_absolute():
        return ""
    path = (source_root / source_file).resolve()
    try:
        path.relative_to(source_root.resolve())
    except ValueError:
        return ""
    if not path.is_file():
        return ""

    # Shared per-run line cache (source_cache.py): the embed stage and the
    # lexical-index stage both call build_snippet once per node, so the same
    # file used to be opened and line-scanned ~2x-per-node; now it is read
    # once (capped at SNIPPET_READ_LINE_CAP lines, same cap the old streaming
    # scan enforced) and every window slices out of the cached tuple.
    lines = get_source_lines(path, SNIPPET_READ_LINE_CAP)
    if lines is None:
        return ""

    half = SNIPPET_WINDOW_LINES // 2
    start = max(0, (line - 1 - half)) if line else 0
    end = start + SNIPPET_WINDOW_LINES

    snippet = "\n".join(lines[start:end])
    return snippet[:SNIPPET_MAX_CHARS]


def build_embedding_input(
    label: str, snippet: str, source_file: str, community_name: str | None
) -> str:
    """C6: label + bounded source snippet + path + community name,
    concatenated into a single embedding input string."""
    parts = [
        f"label: {label}",
        f"path: {source_file}",
        f"community: {community_name or 'unassigned'}",
    ]
    if snippet:
        parts.append(f"snippet:\n{snippet}")
    text = "\n".join(parts)
    return text[:EMBED_INPUT_MAX_CHARS]


def is_trivial_node(label: str, snippet: str) -> bool:
    """Skip heuristic: trivial one-line accessors are not embedded at all
    (deliverable 2). Documented fallback for these nodes when a caller wants
    "similar" results anyway lives in `overlay_similar.py` (exact label +
    same-community match, no embedding lookup — deliverable 7)."""
    if not TRIVIAL_LABEL_PATTERN.match(label or ""):
        return False
    non_blank = [ln for ln in snippet.splitlines() if ln.strip()]
    return len(non_blank) <= TRIVIAL_MAX_SNIPPET_LINES


def default_embed_health_check(base_url: str, timeout: float) -> bool:
    """GET `{base_url}/api/tags` — the cheapest native-endpoint call that
    confirms the Ollama host is reachable, without triggering an actual
    embedding computation. Any failure => unhealthy; never raises (C9's
    "never crash the pipeline on a down local service" pattern, mirroring
    naming.default_ollama_health_check, but against the native surface)."""
    url = base_url.rstrip("/") + "/api/tags"
    if not is_valid_http_base_url(url):
        log.warning("embed health check refused non-http(s) URL %s", url)
        return False
    req = urllib.request.Request(url)  # noqa: S310 - scheme validated above
    try:
        with urllib.request.urlopen(  # noqa: S310 - fixed internal endpoint
            req, timeout=timeout, context=ssl_context()
        ) as resp:
            status = getattr(resp, "status", resp.getcode())
            return 200 <= status < 300
    except Exception as exc:  # noqa: BLE001 - any failure => unhealthy, never crash the pipeline
        log.warning("embed health check failed for %s: %s", url, exc)
        return False


def embed_batch(
    base_url: str, model: str, inputs: list[str], timeout: float = EMBED_REQUEST_TIMEOUT
) -> list[list[float]]:
    """POST `{base_url}/api/embed` with `{"model": model, "input": inputs}`
    (native contract, verified via a real curl during WS3 build — NOT the
    `/v1/embeddings` OpenAI-compat shape, per C9). Returns one vector per
    input, in input order. Raises RuntimeError on any transport/shape
    failure — callers decide whether that means "degraded mode" or a hard
    failure; this function never silently returns partial/wrong-length
    results."""
    if not inputs:
        return []
    url = base_url.rstrip("/") + "/api/embed"
    if not is_valid_http_base_url(url):
        raise RuntimeError(f"embed_batch refused non-http(s) URL: {url!r}")
    payload = json.dumps({"model": model, "input": inputs}).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 - scheme validated above
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(  # noqa: S310 - fixed internal endpoint
            req, timeout=timeout, context=ssl_context()
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"embed_batch request to {url} failed: {exc}") from exc

    vectors = body.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(inputs):
        got_description = len(vectors) if isinstance(vectors, list) else repr(vectors)
        raise RuntimeError(
            f"embed_batch response shape mismatch from {url}: expected {len(inputs)} embeddings, "
            f"got {got_description}"
        )
    return vectors


def _embed_batch_with_retry(base_url: str, model: str, inputs: list[str]) -> list[list[float]]:
    """`embed_batch` with EMBED_BATCH_RETRIES bounded retries (short fixed
    backoff) before letting the failure propagate. Looked up via the module
    global so tests monkeypatching `embedding.embed_batch` keep working."""
    attempts = EMBED_BATCH_RETRIES + 1
    for attempt in range(1, attempts + 1):
        try:
            return embed_batch(base_url, model, inputs)
        except RuntimeError as exc:
            if attempt >= attempts:
                raise
            log.warning(
                "embed_batch attempt %d/%d failed (%s) — retrying after %.1fs backoff",
                attempt,
                attempts,
                exc,
                EMBED_RETRY_BACKOFF_SECONDS,
            )
            time.sleep(EMBED_RETRY_BACKOFF_SECONDS)
    # Unreachable: the loop either returns or re-raises on its last attempt.
    raise RuntimeError("embed_batch retry loop exited without a result")


def _iter_embeddable_nodes(repo_id: str, graph_data: dict, source_root: Path | None):
    """Yields (key, label, source_file, community_name, snippet, is_trivial)
    for every node in this repo's raw graph that has enough fields to build
    a logical key at all. Nodes without a resolvable key (no source_file or
    label — e.g. bare external/library nodes) are not embeddable and are
    skipped entirely, same as they're unindexable for overlay logical refs."""
    for node in graph_data.get("nodes", []):
        if not isinstance(node, dict):
            continue
        key = node_key(repo_id, node)
        if key is None:
            continue
        label = node["label"]
        source_file = node["source_file"]
        community_name = node.get("community_name")
        snippet = build_snippet(source_root, source_file, node_line(node))
        yield key, label, source_file, community_name, snippet, is_trivial_node(label, snippet)


def compute_repo_shard(
    repo_id: str,
    graph_data: dict,
    source_root: Path | None,
    previous_shard: RepoShard,
    base_url: str,
    model: str,
    stats: EmbeddingStats,
) -> RepoShard:
    """Builds this repo's shard for a repo that changed this run. Each
    `entries[key]` is `{"content_hash": ..., "row": int | None}` — `row`
    (assigned once the final `RepoVectors` matrix is built below) is `None`
    iff the node was skipped by the trivial heuristic, never that embedding
    failed silently (a real failure raises, per `embed_batch`)."""
    entries: dict[str, dict] = {}
    # key -> vector (list[float] for freshly-embedded, ndarray row for
    # reused-from-previous) collected before the final RepoVectors matrix is
    # built, so reused rows pass straight through without a list copy.
    vector_sources: dict[str, list[float] | np.ndarray] = {}
    to_embed_keys: list[str] = []
    to_embed_inputs: list[str] = []

    for key, label, source_file, community_name, snippet, trivial in _iter_embeddable_nodes(
        repo_id, graph_data, source_root
    ):
        if trivial:
            entries[key] = {"content_hash": None, "row": None}
            stats.skipped_trivial += 1
            continue

        input_text = build_embedding_input(label, snippet, source_file, community_name)
        digest = _content_hash(input_text)
        prev_entry = previous_shard.entries.get(key)
        prev_vector = previous_shard.vectors.get(key)
        if (
            prev_entry is not None
            and prev_entry.get("content_hash") == digest
            and prev_vector is not None
        ):
            entries[key] = {"content_hash": digest, "row": None}  # row filled in below
            vector_sources[key] = prev_vector
            stats.reused += 1
            continue

        to_embed_keys.append(key)
        to_embed_inputs.append(input_text)
        entries[key] = {"content_hash": digest, "row": None}  # placeholder, filled below

    total_batches = -(-len(to_embed_keys) // EMBED_BATCH_SIZE) if to_embed_keys else 0
    if total_batches:
        log.info(
            "%s: embedding %d nodes in %d batches ...", repo_id, len(to_embed_keys), total_batches
        )
    for batch_start in range(0, len(to_embed_keys), EMBED_BATCH_SIZE):
        batch_num = batch_start // EMBED_BATCH_SIZE + 1
        batch_keys = to_embed_keys[batch_start : batch_start + EMBED_BATCH_SIZE]
        batch_inputs = to_embed_inputs[batch_start : batch_start + EMBED_BATCH_SIZE]
        vectors = _embed_batch_with_retry(base_url, model, batch_inputs)
        for key, vector in zip(batch_keys, vectors, strict=True):
            vector_sources[key] = vector
            stats.embedded += 1
        log.info(
            "%s: batch %d/%d done (%d nodes embedded so far)",
            repo_id,
            batch_num,
            total_batches,
            stats.embedded,
        )

    repo_vectors = RepoVectors.from_mapping(vector_sources)
    for row, key in enumerate(repo_vectors.keys):
        entries[key]["row"] = row

    return RepoShard(entries=entries, vectors=repo_vectors)


def build_id_map(
    previous_id_map: dict[str, dict], current_keys: set[str], generation_id: str
) -> dict[str, dict]:
    """C27: rebuild the id-map to the EXACT current node-key set every
    generation. Keys present now are (re)marked "active"; keys that were
    "active" in the previous id-map but are absent now are marked
    "tombstoned" (never silently dropped) with the generation_id they
    disappeared in. A key already "tombstoned" stays tombstoned (its
    original tombstoned_at generation_id is preserved) unless it reappears,
    in which case it goes back to "active"."""
    new_map: dict[str, dict] = {}
    for key in current_keys:
        # Present now, regardless of prior status (including a previously
        # tombstoned key reappearing) -> active again, tombstone cleared.
        new_map[key] = {"status": "active", "generation_id": generation_id, "tombstoned_at": None}

    for key, prior in previous_id_map.items():
        if key in current_keys:
            continue
        if prior.get("status") == "tombstoned":
            new_map[key] = prior
            continue
        new_map[key] = {
            "status": "tombstoned",
            "generation_id": prior.get("generation_id"),
            "tombstoned_at": generation_id,
        }
    return new_map


def _validate_repo_id_for_filename(repo_id: str) -> None:
    """repo_id is validated at registry load (registry.REPO_ID_PATTERN), but
    shard paths are built from it in several places — keep a defense-in-depth
    check right where each filename is formed so a future caller with an
    unvalidated repo_id can never traverse out of the shard dir."""
    if "/" in repo_id or "\\" in repo_id or repo_id in ("", ".", "..") or repo_id.startswith("."):
        raise ValueError(f"unsafe repo_id for shard filename: {repo_id!r}")


def _shard_filename(repo_id: str) -> str:
    """v1 shard filename (`<repo>.json`, entries with inline `embedding`
    lists) — still used as the read-compat fallback in `read_previous_shard`."""
    _validate_repo_id_for_filename(repo_id)
    return f"{repo_id}.json"


def _shard_meta_filename(repo_id: str) -> str:
    """v2 shard meta filename (`<repo>.meta.json`)."""
    _validate_repo_id_for_filename(repo_id)
    return f"{repo_id}{SHARD_META_SUFFIX}"


def _shard_matrix_filename(repo_id: str) -> str:
    """v2 shard matrix filename (`<repo>.npy`)."""
    _validate_repo_id_for_filename(repo_id)
    return f"{repo_id}{SHARD_MATRIX_SUFFIX}"


def _load_shard_matrix(matrix_path: Path, repo_id: str) -> np.ndarray | None:
    """Loads the v2 matrix file mmap'd (read-only). Missing/corrupt matrix
    alongside a meta file is not a hard failure — it just means the previous
    shard is treated as empty, which forces a full re-embed this run (safe,
    only costs compute)."""
    if not matrix_path.is_file():
        log.warning(
            "%s: shard meta present but matrix file missing at %s — "
            "treating previous shard as empty",
            repo_id,
            matrix_path,
        )
        return None
    try:
        return np.load(matrix_path, mmap_mode="r")
    except (OSError, ValueError) as exc:
        log.warning(
            "%s: failed to load shard matrix %s (%s) — treating previous shard as empty",
            repo_id,
            matrix_path,
            exc,
        )
        return None


def _read_v1_shard(shard_path: Path) -> RepoShard:
    # Sync trusts its own artifacts here — this reads back a shard this same
    # pipeline wrote in a previous run. There is deliberately no size gate
    # (unlike the server's read-side `MAX_SHARD_BYTES`): a legitimately-grown
    # shard hitting a size ceiling would silently force a full re-embed,
    # which is the wrong failure mode for sync. Parse failure (corrupt JSON,
    # or invalid UTF-8 bytes on disk) instead degrades to an empty shard,
    # which safely triggers re-embed for just this repo.
    try:
        data = json.loads(shard_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return RepoShard.empty()

    raw_entries = data.get("entries", {})
    entries: dict[str, dict] = {}
    vector_sources: dict[str, list[float]] = {}
    for key, entry in raw_entries.items():
        entries[key] = {"content_hash": entry.get("content_hash"), "row": None}
        embedding_value = entry.get("embedding")
        if embedding_value:
            vector_sources[key] = embedding_value

    vectors = RepoVectors.from_mapping(vector_sources)
    for row, key in enumerate(vectors.keys):
        entries[key]["row"] = row
    return RepoShard(entries=entries, vectors=vectors)


def _validate_v2_shard(meta: dict, matrix: np.ndarray) -> str | None:
    """Validates a v2 shard's meta + matrix BEFORE any RepoVectors is built
    from them. Returns a human-readable reason string on the first violation
    found (guard-clause chain, cheapest checks first), or `None` if the
    shard is trustworthy. Never raises — every check here exists precisely
    because the input is untrusted (a previous run's artifact, possibly
    corrupted by disk issues, a partial write, or a hostile/malformed
    generation dir); callers must treat a non-None reason as "corrupt
    previous shard" and fall back to an empty RepoShard, which forces a safe
    re-embed rather than persisting or propagating bad data."""
    shard_format = meta.get("shard_format")
    # `x in a_frozenset_of_ints` calls hash(x) first — an unhashable JSON
    # value (a list or dict, from a malformed/hostile meta file) would raise
    # TypeError instead of failing this check cleanly. Guard the type
    # BEFORE the membership test (bool excluded: isinstance(True, int) is
    # True in Python, but a bool is never a valid format marker).
    if not isinstance(shard_format, int) or isinstance(shard_format, bool):
        return f"shard_format is not an int (got {shard_format!r})"
    if shard_format not in SUPPORTED_SHARD_FORMATS:
        return f"unsupported shard_format {shard_format!r}"

    entries = meta.get("entries")
    if not isinstance(entries, dict):
        return f"entries is not a dict (got {type(entries).__name__})"
    for key, entry in entries.items():
        if not isinstance(entry, dict):
            return f"entry {key!r} is not a dict (got {type(entry).__name__})"

    if matrix.ndim != 2:
        return f"matrix is not 2-D (ndim={matrix.ndim})"

    if matrix.dtype != np.float32:
        return f"matrix dtype is {matrix.dtype}, expected float32"

    dim = meta.get("dim")
    if dim != matrix.shape[1]:
        return f"meta dim {dim!r} does not match matrix width {matrix.shape[1]}"

    n_rows = matrix.shape[0]
    seen_rows: dict[int, str] = {}
    for key, entry in entries.items():
        row = entry.get("row")
        if row is None:
            continue
        if isinstance(row, bool) or not isinstance(row, int):
            return f"entry {key!r} has non-int row {row!r}"
        if row < 0 or row >= n_rows:
            return f"entry {key!r} has out-of-range row {row!r} (n_rows={n_rows})"
        if row in seen_rows:
            return f"duplicate row {row} claimed by both {seen_rows[row]!r} and {key!r}"
        seen_rows[row] = key

    return None


def _read_v2_shard(current_dir: Path, repo_id: str, meta_path: Path) -> RepoShard:
    # Same trust-your-own-artifacts stance as `_read_v1_shard` above: no
    # MAX_SHARD_BYTES-style size ceiling on the sync side, since this is our
    # own previous-run output, not untrusted input — gating on size here
    # would risk a silent full re-embed for a shard that simply grew.
    # Invalid UTF-8 bytes on disk degrade to an empty shard (forces re-embed)
    # rather than crashing the sync run.
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        log.warning(
            "%s: corrupt shard meta at %s — treating previous shard as empty",
            repo_id,
            meta_path,
        )
        return RepoShard.empty()

    if not isinstance(meta, dict):
        log.warning(
            "%s: shard meta at %s is not a JSON object (got %s) — treating previous shard as empty",
            repo_id,
            meta_path,
            type(meta).__name__,
        )
        return RepoShard.empty()

    matrix_path = current_dir / _shard_matrix_filename(repo_id)
    matrix = _load_shard_matrix(matrix_path, repo_id)
    if matrix is None:
        return RepoShard.empty()

    invalid_reason = _validate_v2_shard(meta, matrix)
    if invalid_reason is not None:
        log.warning(
            "%s: invalid v2 shard in %s (%s) — treating previous shard as empty",
            repo_id,
            meta_path,
            invalid_reason,
        )
        return RepoShard.empty()

    entries: dict[str, dict] = meta["entries"]
    n_rows = matrix.shape[0]
    keys_by_row: list[str | None] = [None] * n_rows
    for key, entry in entries.items():
        row = entry.get("row")
        if row is None:
            continue
        keys_by_row[row] = key

    if any(key is None for key in keys_by_row):
        log.warning(
            "%s: shard matrix has rows with no owning entry in %s"
            " — treating previous shard as empty",
            repo_id,
            meta_path,
        )
        return RepoShard.empty()

    # Every row was assigned a key above (the `any(... is None)` guard
    # returned early otherwise) — narrow `list[str | None]` to `list[str]`
    # explicitly so mypy sees what we already know at runtime.
    narrowed_keys: list[str] = [key for key in keys_by_row if key is not None]
    if len(narrowed_keys) != len(keys_by_row):
        # Unreachable by construction (the `any(... is None)` guard above
        # returned early) — kept as a guard clause instead of an assert so
        # -O builds and bandit S101 both stay honest.
        log.warning("%s: internal row/key narrowing mismatch — treating shard as empty", repo_id)
        return RepoShard.empty()

    vectors = RepoVectors.from_rows(narrowed_keys, matrix)
    return RepoShard(entries=entries, vectors=vectors)


def read_previous_shard(embeddings_current_dir: Path | None, repo_id: str) -> RepoShard:
    if embeddings_current_dir is None or not embeddings_current_dir.is_dir():
        return RepoShard.empty()

    meta_path = embeddings_current_dir / _shard_meta_filename(repo_id)
    if meta_path.is_file():
        return _read_v2_shard(embeddings_current_dir, repo_id, meta_path)

    v1_path = embeddings_current_dir / _shard_filename(repo_id)
    if not v1_path.is_file():
        return RepoShard.empty()
    return _read_v1_shard(v1_path)


def read_previous_id_map(embeddings_current_dir: Path | None) -> dict[str, dict]:
    if embeddings_current_dir is None or not embeddings_current_dir.is_dir():
        return {}
    id_map_path = embeddings_current_dir / "id-map.json"
    if not id_map_path.is_file():
        return {}
    try:
        return json.loads(id_map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def run_embedding_stage(
    graph_paths_by_repo: dict[str, Path],
    repo_roots_by_id: dict[str, Path],
    graphs_by_repo: dict[str, dict],
    unchanged_repo_ids: set[str],
    settings: Settings,
    provisional_generation_id: str,
    health_check: HealthCheckFn | None = None,
) -> EmbeddingStageResult:
    """Top-level WS3 embed-changed stage. `graphs_by_repo` is the same
    per-repo raw graph data overlay.py already loads (post per-project
    labeling, i.e. carrying per-repo `community_name`) — passed in rather
    than reloaded, so pipeline.py only reads each per-repo graph.json once.

    `provisional_generation_id` is a run-scoped identifier used only for the
    id-map's "tombstoned_at"/"generation_id" bookkeeping while staging;
    nothing durable is written until `persist_generation` is called after a
    successful publish (see module docstring)."""
    # URL scheme gate BEFORE any health check runs: a base URL that is not
    # plain http(s)-with-host (file://, gopher://, ...) fails this stage's
    # health-check path outright — degraded mode, zero requests attempted.
    degrade_reason = ""
    if not is_valid_http_base_url(settings.ollama_embed_base_url):
        degrade_reason = (
            f"invalid embed base URL {settings.ollama_embed_base_url!r}: "
            "scheme must be http or https with a non-empty host"
        )
    if not degrade_reason:
        check = health_check if health_check is not None else default_embed_health_check
        healthy = check(settings.ollama_embed_base_url, settings.ollama_embed_health_timeout)
        if not healthy:
            degrade_reason = "embed health check failed"

    previous_current_dir = (
        settings.embeddings_current_symlink
        if settings.embeddings_current_symlink.exists()
        else None
    )
    previous_id_map = read_previous_id_map(previous_current_dir)

    if degrade_reason:
        log.warning(
            "embed service unusable at %s (%s) — skipping embed-changed entirely (degraded mode); "
            "similar_approach falls back to the exact-match scorer this generation",
            settings.ollama_embed_base_url,
            degrade_reason,
        )
        # Degraded: carry forward whatever was last published, verbatim, so a
        # transient outage doesn't erase the whole index; nodes added since
        # then simply have no vector until a healthy run recomputes them.
        carried_forward_vectors: dict[str, RepoVectors] = {}
        shards_by_repo: dict[str, RepoShard] = {}
        for repo_id in graphs_by_repo:
            prev_shard = read_previous_shard(previous_current_dir, repo_id)
            shards_by_repo[repo_id] = prev_shard
            carried_forward_vectors[repo_id] = prev_shard.vectors
        return EmbeddingStageResult(
            status=EMBED_DEGRADED,
            vectors_by_repo=carried_forward_vectors,
            shards_by_repo=shards_by_repo,
            reason=degrade_reason,
            id_map=previous_id_map,
        )

    stats = EmbeddingStats()
    vectors_by_repo: dict[str, RepoVectors] = {}
    all_keys: set[str] = set()
    staged_shards: dict[str, RepoShard] = {}
    mid_run_failure: str | None = None

    def _stage_shard(staged_repo_id: str, staged_shard: RepoShard) -> None:
        staged_shards[staged_repo_id] = staged_shard
        all_keys.update(staged_shard.entries.keys())
        vectors_by_repo[staged_repo_id] = staged_shard.vectors

    total_repos = len(graphs_by_repo)
    repo_items = list(graphs_by_repo.items())
    for i, (repo_id, graph_data) in enumerate(repo_items, start=1):
        source_root = repo_roots_by_id.get(repo_id)
        previous_shard = read_previous_shard(previous_current_dir, repo_id)

        if repo_id in unchanged_repo_ids and previous_shard.entries:
            # WS1's own changed-detection (decide_action -> ACTION_SKIP)
            # already told us this repo is untouched this run — reuse its
            # whole previous shard rather than re-reading/re-hashing every
            # node in it.
            log.info("[%d/%d] %s: reusing previous shard (unchanged) ...", i, total_repos, repo_id)
            stats.reused_repos_unchanged += 1
            _stage_shard(repo_id, previous_shard)
            continue

        log.info("[%d/%d] %s: computing shard ...", i, total_repos, repo_id)
        try:
            shard = compute_repo_shard(
                repo_id,
                graph_data,
                source_root,
                previous_shard,
                settings.ollama_embed_base_url,
                settings.ollama_embed_model,
                stats,
            )
        except RuntimeError as exc:
            # A real embed_batch call failed AFTER the pre-flight health
            # check passed (and survived _embed_batch_with_retry's bounded
            # retry) — a mid-run network blip, the host going down partway
            # through, etc. This must NOT crash the whole pipeline and lose
            # already-completed repos' real vectors (or the naming stage's
            # result, which already ran and succeeded by this point) — fall
            # back to this repo's previous shard and every repo after it,
            # keep what's already staged, and report the run as PARTIAL
            # rather than letting the exception propagate out of the
            # pipeline.
            log.warning(
                "%s: embed_batch failed mid-run (%s) — falling back to previous shard for "
                "this repo and all %d remaining repo(s); "
                "already-embedded repos this run are kept",
                repo_id,
                exc,
                total_repos - i,
            )
            mid_run_failure = f"{repo_id}: {exc}"
            # The failing repo's previous shard is already in hand — reuse
            # the loaded object instead of re-reading it from disk.
            _stage_shard(repo_id, previous_shard)
            for fallback_repo_id, _ in repo_items[i:]:
                _stage_shard(
                    fallback_repo_id, read_previous_shard(previous_current_dir, fallback_repo_id)
                )
            break

        # Drop the previous shard's reference BEFORE staging the new one so
        # the previous and new shard never coexist longer than needed
        # (peak-RSS: compute_repo_shard has already taken everything it
        # reuses out of it by now).
        del previous_shard
        _stage_shard(repo_id, shard)

    id_map = build_id_map(previous_id_map, all_keys, provisional_generation_id)
    stats.tombstoned = sum(1 for v in id_map.values() if v.get("status") == "tombstoned")

    recipe = EmbeddingRecipe(model=settings.ollama_embed_model, dim=_infer_dim(vectors_by_repo))

    return EmbeddingStageResult(
        status=EMBED_PARTIAL if mid_run_failure else EMBED_HEALTHY,
        vectors_by_repo=vectors_by_repo,
        shards_by_repo=staged_shards,
        stats=stats,
        recipe=recipe,
        id_map=id_map,
        reason=mid_run_failure or "",
    )


def _infer_dim(vectors_by_repo: dict[str, RepoVectors]) -> int:
    from graphify_mesh.sync.config import EMBED_DEFAULT_DIM

    return max(
        (rv.dim for rv in vectors_by_repo.values() if len(rv)),
        default=EMBED_DEFAULT_DIM,
    )


def stage_embeddings(
    staging_root: Path, staged_shards_source: dict[str, RepoShard], id_map: dict[str, dict]
) -> Path:
    """Write this run's shards + id-map into the ephemeral per-run staging
    tempdir as shard format v2 (`<repo>.meta.json` + `<repo>.npy`). Nothing
    here touches the real, persistent embeddings_dir — see
    `persist_generation`."""
    out_dir = staging_root / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    for repo_id, shard in staged_shards_source.items():
        meta_path = out_dir / _shard_meta_filename(repo_id)
        matrix_path = out_dir / _shard_matrix_filename(repo_id)
        # json.dump streams straight to the file handle — json.dumps built
        # the whole serialized shard meta as one in-RAM string first, which
        # at large node counts briefly doubled this stage's footprint.
        with meta_path.open("w", encoding="utf-8") as meta_fh:
            json.dump(
                {
                    "repo_id": repo_id,
                    "shard_format": SHARD_FORMAT_VERSION,
                    "dim": shard.vectors.dim,
                    "entries": shard.entries,
                },
                meta_fh,
            )
        np.save(matrix_path, shard.vectors.matrix)
    with (out_dir / "id-map.json").open("w", encoding="utf-8") as id_map_fh:
        json.dump(id_map, id_map_fh, indent=2, sort_keys=True)
    return out_dir


def persist_generation(
    embeddings_dir: Path, generation_id: str, staged_dir: Path, keep: int
) -> None:
    """Called only after a successful publish (mirrors publish.flip_current):
    copies the staged shard files into
    `embeddings_dir/generations/<generation_id>/`, flips
    `embeddings_dir/current` to point at it, then GCs old generations,
    keeping only the last `keep`."""
    generations_dir = embeddings_dir / "generations"
    generations_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = generations_dir / generation_id
    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    shutil.copytree(staged_dir, gen_dir)

    current = embeddings_dir / "current"
    tmp_link = embeddings_dir / ".current.tmp"
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(gen_dir.resolve(), target_is_directory=True)
    os.rename(str(tmp_link), str(current))
    # Same durability rule as publish.flip_current: without the dir fsync a
    # crash right after rename can leave the flip unwritten on disk.
    _fsync_dir(embeddings_dir)

    gc_old_generations(generations_dir, keep, current=current)


def gc_old_generations(generations_dir: Path, keep: int, current: Path | None = None) -> list[str]:
    """Keep only the `keep` most-recently-created generation dirs (sorted by
    directory name, which is the pipeline's lexicographically-sortable
    timestamp-prefixed generation_id — same ordering convention as
    `publish.write_generation`). The generation `current` points at is
    always pinned regardless of sort order (mirrors
    publish.prune_old_generations: clock skew must never delete the live
    generation). Returns the list of removed generation_ids."""
    if not generations_dir.is_dir():
        return []
    current_name = None
    if current is not None and current.exists():
        current_name = os.path.basename(os.path.realpath(current))
    gen_dirs = sorted((p for p in generations_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
    to_remove = gen_dirs[:-keep] if keep > 0 else gen_dirs
    removed = []
    for gen_dir in to_remove:
        if gen_dir.name == current_name:
            continue
        shutil.rmtree(gen_dir, ignore_errors=True)
        removed.append(gen_dir.name)
    return removed
