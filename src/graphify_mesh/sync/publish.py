"""Versioned generation publish + atomic `current` symlink flip (WS1 item 8).

Publish never deletes a previous generation. If validation fails or the
stale-repo threshold is exceeded, the caller simply never calls `publish()`
— "rollback" is defined as `current` never having been moved, which is why
this module never touches `current` except in the final atomic rename.

WS6: for the manual rollback procedure (reverting to an OLDER generation
than whatever `current` presently points at, past the automatic cases
above), see `bin/graphify_mesh.sync/ROLLBACK.md`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
from datetime import date
from pathlib import Path

log = logging.getLogger("graphify_mesh.sync.publish")

# Name of a graphify pre-overwrite backup dir (`date.today().isoformat()`).
_DATED_SNAPSHOT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Every file the pipeline's full publish sequence writes into a generation
# dir. Completeness for pruning is manifest-aware (see `_is_incomplete` in
# `prune_old_generations`): the manifest's `artifact_sha256` map, not this
# tuple, decides which artifacts a given generation must carry.
EXPECTED_GENERATION_ARTIFACTS = (
    "global-graph.json",
    "cross-project-overlay.json",
    "lexical-index.json",
    "generation-manifest.json",
)


def make_generation_id(output_hash: str) -> str:
    # Second-precision timestamp + content-hash prefix is not sufficient to
    # guarantee uniqueness on its own: two runs within the same second that
    # merge to byte-identical output (e.g. nothing changed) would otherwise
    # collide on generation_id. Add a short random nonce.
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    nonce = os.urandom(3).hex()
    return f"{ts}-{output_hash[:8]}-{nonce}"


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_atomic(path: Path, data: dict, **encoder_kwargs) -> str:
    """Write via tmp-file + rename so an interrupted write can never leave a
    truncated .json in a generation dir. `current` is only flipped after all
    writes, so a partial file was never *served* — but a stranded generation
    dir with half a graph in it is a trap for the manual-rollback procedure
    (ROLLBACK.md points operators at old generation dirs directly).

    Streams `json.JSONEncoder.iterencode` chunks straight to the file — the
    whole serialized artifact (global-graph.json alone can be tens of MB,
    lexical-index.json 100+ MB) is never materialized as one string in RAM.
    Returns the sha256 hex digest of the raw bytes actually written to the
    tmp file (computed before the rename), so the caller can record
    per-artifact content hashes in the generation manifest."""
    tmp_path = path.with_name(path.name + ".tmp")
    hasher = hashlib.sha256()
    encoder = json.JSONEncoder(**encoder_kwargs)
    # fsync file DATA before rename: rename alone can become durable while
    # the contents are not (power loss), leaving an empty/garbage .json
    # behind an already-flipped `current`.
    with tmp_path.open("wb") as fh:
        for chunk in encoder.iterencode(data):
            raw = chunk.encode("utf-8")
            hasher.update(raw)
            fh.write(raw)
        fh.flush()
        os.fsync(fh.fileno())
    os.rename(str(tmp_path), str(path))
    return hasher.hexdigest()


def create_generation_dir(generations_dir: Path, generation_id: str) -> Path:
    gen_dir = generations_dir / generation_id
    gen_dir.mkdir(parents=True, exist_ok=False)
    return gen_dir


def write_global_graph(gen_dir: Path, graph_data: dict) -> str:
    """Stage global-graph.json into the generation dir. Compact separators —
    the merged graph is the largest artifact after the lexical index, and
    indentation roughly doubles its on-disk/serialization size for zero
    benefit (it is only ever read back by json.loads). Returns the sha256 of
    the bytes written, for the manifest's `artifact_sha256` map."""
    graph_path = gen_dir / "global-graph.json"
    digest = _write_json_atomic(graph_path, graph_data, separators=(",", ":"))
    _fsync_dir(gen_dir)
    return digest


def write_manifest(generations_dir: Path, gen_dir: Path, manifest: dict) -> None:
    """Stage generation-manifest.json LAST, after every other artifact in the
    generation dir has been written — the manifest's `artifact_sha256` map
    covers those files' bytes, so it can only be produced once they exist.
    The manifest is small; keep indent=2 + sort_keys for human debuggability."""
    manifest_path = gen_dir / "generation-manifest.json"
    _write_json_atomic(manifest_path, manifest, indent=2, sort_keys=True)
    _fsync_dir(gen_dir)
    _fsync_dir(generations_dir)


def write_generation(
    generations_dir: Path, generation_id: str, graph_data: dict, manifest: dict
) -> Path:
    """Compat entry point (pre-`artifact_sha256` publish shape): create the
    generation dir and stage global-graph.json + generation-manifest.json.
    The pipeline itself no longer uses this — it calls
    `create_generation_dir`/`write_global_graph`/`write_overlay`/
    `write_lexical_index` and then `write_manifest` LAST, so the manifest's
    `artifact_sha256` map can cover every other artifact's written bytes.
    Manifests written through this wrapper carry no `artifact_sha256`."""
    gen_dir = create_generation_dir(generations_dir, generation_id)
    write_global_graph(gen_dir, graph_data)
    write_manifest(generations_dir, gen_dir, manifest)
    return gen_dir


def write_overlay(gen_dir: Path, overlay_data: dict) -> str:
    """WS4: stage the cross-project overlay artifact inside the SAME
    generation dir as `global-graph.json`/`generation-manifest.json`, so it
    flips atomically with everything else on `flip_current` — but as an
    entirely separate file, never merged into `global-graph.json` (C5).
    Returns the sha256 of the bytes written (`artifact_sha256` manifest map).
    sort_keys is retained for deterministic artifact bytes — the overlay
    builder's output key order is owned by another module and is not
    guaranteed sorted."""
    overlay_path = gen_dir / "cross-project-overlay.json"
    digest = _write_json_atomic(overlay_path, overlay_data, separators=(",", ":"), sort_keys=True)
    _fsync_dir(gen_dir)
    return digest


def write_lexical_index(gen_dir: Path, lexical_data: dict) -> str:
    """WS5: stage the lexical-index bundle artifact inside the SAME
    generation dir as `global-graph.json`/`cross-project-overlay.json`, so it
    flips atomically with everything else on `flip_current`. Consumed
    read-only by the graphify-mesh companion MCP server, never written to at
    query time. Returns the sha256 of the bytes written (`artifact_sha256`
    manifest map). sort_keys retained for deterministic artifact bytes."""
    lexical_path = gen_dir / "lexical-index.json"
    digest = _write_json_atomic(lexical_path, lexical_data, separators=(",", ":"), sort_keys=True)
    _fsync_dir(gen_dir)
    return digest


def read_current_lexical_index(global_dir: Path) -> dict | None:
    current = global_dir / "current"
    if not current.exists():
        return None
    lexical_path = current / "lexical-index.json"
    if not lexical_path.exists():
        return None
    try:
        return json.loads(lexical_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_current_overlay(global_dir: Path) -> dict | None:
    current = global_dir / "current"
    if not current.exists():
        return None
    overlay_path = current / "cross-project-overlay.json"
    if not overlay_path.exists():
        return None
    try:
        return json.loads(overlay_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def flip_current(global_dir: Path, gen_dir: Path) -> None:
    """Atomically flip `current` -> gen_dir via rename of a temp symlink."""
    current = global_dir / "current"
    tmp_link = global_dir / ".current.tmp"
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(gen_dir.resolve(), target_is_directory=True)
    os.rename(str(tmp_link), str(current))
    _fsync_dir(global_dir)


def prune_old_generations(generations_dir: Path, current: Path, keep: int = 2) -> list[str]:
    """Delete all generation dirs under `generations_dir` except the one
    `current` points at and the `keep - 1` most recent others (by directory
    name, which sorts chronologically — `make_generation_id` is a
    zero-padded UTC timestamp prefix).

    Each generation is a full copy of the merged graph + overlay + lexical
    index (global-graph.json alone can be tens of MB, lexical-index.json can
    exceed 100MB) — with no GC, a generation this large published on every
    scheduled run (e.g. hourly) accumulates without bound. The write helpers
    never delete anything themselves (by design, so a crash mid-write never
    corrupts a previous good generation) — pruning only ever runs here,
    AFTER `flip_current` has already succeeded, so a crash during pruning
    can strand extra generation dirs (wasted disk) but can never remove the
    one `current` needs.

    Also removes generation dirs that never finished publishing — either a
    dangling `<name>/lexical-index.json.tmp` with no matching `.json` (the
    process was killed between `write_lexical_index`'s tmp-write and its
    rename), a dir missing the manifest or global-graph.json outright, or a
    dir whose manifest's `artifact_sha256` map names an artifact absent on
    disk. These were never `current` and are safe to delete
    outright, keep-count aside — and they must never occupy a keep slot that
    would otherwise protect a complete generation.
    """
    if not generations_dir.is_dir():
        return []
    current_name = os.path.basename(os.path.realpath(current)) if current.exists() else None
    all_names = sorted(p.name for p in generations_dir.iterdir() if p.is_dir())

    def _is_incomplete(name: str) -> bool:
        gen_dir = generations_dir / name
        if any(gen_dir.glob("*.tmp")):
            return True
        manifest_path = gen_dir / "generation-manifest.json"
        if not manifest_path.is_file():
            return True
        if not (gen_dir / "global-graph.json").is_file():
            return True
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return True
        artifact_sha256 = manifest.get("artifact_sha256") if isinstance(manifest, dict) else None
        if not isinstance(artifact_sha256, dict):
            # Legacy manifest (e.g. via `write_generation`) with no
            # artifact_sha256 map: graph + manifest present is complete.
            return False
        for artifact in artifact_sha256:
            if not (gen_dir / artifact).is_file():
                return True
        return False

    incomplete = [n for n in all_names if n != current_name and _is_incomplete(n)]
    complete = [n for n in all_names if n not in incomplete]
    # Keep the most recent `keep` complete generations (current is always
    # among the most recent, but pin it explicitly in case clock skew ever
    # makes it sort out of the tail).
    keep_set = set(complete[-keep:]) if keep > 0 else set()
    if current_name is not None:
        keep_set.add(current_name)
    to_remove = incomplete + [n for n in complete if n not in keep_set]

    removed = []
    for name in to_remove:
        target = generations_dir / name
        shutil.rmtree(target, ignore_errors=True)
        removed.append(name)
    return removed


def prune_dated_snapshots(
    collection_path: Path, keep_days: int, today: date | None = None
) -> list[str]:
    """Delete graphify's dated backup dirs under `collection_path` that are
    more than `keep_days` days old, always keeping the newest one.

    Upstream graphify's `backup_if_protected` copies graph.json,
    GRAPH_REPORT.md, manifest.json etc. into `<collection>/<YYYY-MM-DD>/`
    (local date) before overwriting a labeled graph, and nothing retained or
    pruned them. Only direct children that are real directories (never
    symlinks) named exactly `YYYY-MM-DD` with a valid calendar date count;
    every other entry (`cache`, generation ids, timestamped names) is never
    touched. A dir is old when `(today - dir_date).days > keep_days`.
    `keep_days <= 0` disables rotation. A failed delete is logged as a
    warning and skipped — wasted disk is acceptable, a failed run is not.
    Returns the names actually removed.
    """
    if keep_days <= 0 or not collection_path.is_dir():
        return []
    today = today or date.today()
    try:
        children = list(collection_path.iterdir())
    except OSError as exc:
        log.warning("publish: could not list %s for dated snapshots: %s", collection_path, exc)
        return []
    dated: dict[str, date] = {}
    for child in children:
        if not _DATED_SNAPSHOT_RE.match(child.name):
            continue
        if child.is_symlink() or not child.is_dir():
            continue
        try:
            dated[child.name] = date.fromisoformat(child.name)
        except ValueError:
            continue
    if not dated:
        return []
    newest = max(dated, key=dated.__getitem__)

    removed = []
    for name in sorted(dated):
        if name == newest or (today - dated[name]).days <= keep_days:
            continue
        try:
            shutil.rmtree(collection_path / name)
        except OSError as exc:
            log.warning(
                "publish: could not remove dated snapshot %s: %s", collection_path / name, exc
            )
            continue
        removed.append(name)
    return removed


def output_hash(graph_data: dict) -> str:
    """sha256 of the canonical (sort_keys) JSON serialization, streamed
    chunk-by-chunk into the hasher so the whole multi-MB string is never
    materialized. MUST stay byte-equivalent to the historical
    `json.dumps(graph_data, sort_keys=True)` — published manifests from older
    generations embed hashes produced by that implementation, and the MCP
    server recomputes this over global-graph.json to verify them. Separators
    are pinned to `(", ", ": ")` — json.dumps's defaults when indent is None
    (equality verified against json.dumps across unicode/float/inf inputs)."""
    h = hashlib.sha256()
    encoder = json.JSONEncoder(sort_keys=True, separators=(", ", ": "))
    for chunk in encoder.iterencode(graph_data):
        h.update(chunk.encode("utf-8"))
    return h.hexdigest()


def read_current_manifest(global_dir: Path) -> dict | None:
    current = global_dir / "current"
    if not current.exists():
        return None
    manifest_path = current / "generation-manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_current_global_graph(global_dir: Path) -> dict | None:
    """Load the previously PUBLISHED global graph (global-graph.json behind
    the `current` symlink), not just its manifest.

    Used by the WS2 naming stage's degraded-mode fallback (C23): when Ollama
    is down and this generation's fresh clustering/labeling is skipped
    entirely, `community_name` is restored per-node only from this
    previously-published GLOBAL artifact — never from the current merge's
    per-project inputs — so a transient outage can't leak a per-project
    community name into the global graph.
    """
    current = global_dir / "current"
    if not current.exists():
        return None
    graph_path = current / "global-graph.json"
    if not graph_path.exists():
        return None
    try:
        return json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
