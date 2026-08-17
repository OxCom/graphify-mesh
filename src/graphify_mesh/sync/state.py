"""Per-project source manifest state (untracked, under graphify/global/state/).

Used to decide `graphify update` (code-only change) vs `graphify extract`
(semantic/docs/config change) per WS1 item 2, and to detect dirty worktrees
(WS1 item 4) without ever running a mutating git command.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from graphify_mesh.sync.config import IGNORED_DIR_NAMES, categorize_file

log = logging.getLogger("graphify_mesh.sync")


@dataclass
class SourceDigest:
    code_hash: str
    semantic_hash: str
    file_count: int

    def to_dict(self) -> dict:
        return {
            "code_hash": self.code_hash,
            "semantic_hash": self.semantic_hash,
            "file_count": self.file_count,
        }


CONTENT_DIGEST_ENV = "GRAPHIFY_MESH_CONTENT_DIGEST"


def _use_content_digests() -> bool:
    """Whether the manifest digest keys on file CONTENT (default) or mtime.

    mtime is a proxy for "changed" that is wrong in the expensive direction here:
    a `git checkout`, an editor save-without-edit, or a regenerated lockfile bumps
    `st_mtime_ns` with byte-identical content, `semantic_hash` changes, and
    `decide_action` schedules a full non-deterministic `graphify extract` that can
    take 900-2700 s of shared-GPU time and whose output legitimately differs from
    the last run. Keying on content means an unchanged tree is genuinely unchanged
    and the LLM never re-runs.

    Set GRAPHIFY_MESH_CONTENT_DIGEST=0 to restore mtime behaviour (e.g. on a tree
    so large that hashing dominates, though measured cost here is ~66 MB per four
    repos, i.e. seconds against a multi-minute extract).
    """
    raw = os.environ.get(CONTENT_DIGEST_ENV)
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _change_marker(path: Path, stat: os.stat_result, content_digests: bool) -> str:
    """Per-file component of the manifest digest.

    Falls back to mtime when content is unreadable (races, permissions, a file
    deleted between walk and hash) so one bad file degrades that entry instead of
    aborting the whole manifest.
    """
    if not content_digests:
        return str(stat.st_mtime_ns)
    try:
        digest = file_content_hash(path)
    except OSError:
        # Unreadable (permissions, deleted between walk and hash, I/O error) or
        # a device/FIFO that cannot be slurped. One bad file must degrade its own
        # entry to the mtime marker, never abort the whole manifest — an aborted
        # manifest would look like "no files" and cascade into a bootstrap.
        return str(stat.st_mtime_ns)
    if digest is None:
        return str(stat.st_mtime_ns)
    return digest[:32]


def compute_source_manifest(root: Path) -> SourceDigest:
    """Walk root, hashing (relpath, size, content-digest) separately per category.

    Uses os.walk with in-place `dirnames` pruning so ignored trees
    (node_modules/.git/vendor/...) are never even descended into — the old
    `sorted(root.rglob("*"))` enumerated every one of their entries just to
    filter them out again. The candidate list is still sorted with Path
    ordering and every path still passes the exact same full-`parts` ignore
    filter, so entry order and content — and therefore the digest — are
    byte-identical to the rglob implementation for unchanged trees (state
    compatibility)."""
    code_entries: list[str] = []
    semantic_entries: list[str] = []
    count = 0
    if not root.is_dir():
        return SourceDigest(code_hash="empty", semantic_hash="empty", file_count=0)

    content_digests = _use_content_digests()

    candidates: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIR_NAMES)
        base = Path(dirpath)
        for filename in filenames:
            candidates.append(base / filename)

    for path in sorted(candidates):
        # Kept identical to the rglob-era filter (checked against the FULL
        # path parts, filename included): a *file* named e.g. `vendor`, or a
        # root that itself lives under an ignored dir, must keep hashing the
        # same as before.
        if any(part in IGNORED_DIR_NAMES for part in path.parts):
            continue
        if not path.is_file():
            continue
        category = categorize_file(path)
        if category == "ignore":
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        rel = str(path.relative_to(root))
        entry = f"{rel}:{stat.st_size}:{_change_marker(path, stat, content_digests)}"
        count += 1
        if category == "code":
            code_entries.append(entry)
        else:
            semantic_entries.append(entry)

    def _hash(entries: list[str]) -> str:
        h = hashlib.sha256()
        for e in entries:
            h.update(e.encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()[:16]

    return SourceDigest(
        code_hash=_hash(code_entries), semantic_hash=_hash(semantic_entries), file_count=count
    )


def load_state(state_path: Path) -> dict:
    """Load the per-project source-manifest state. A corrupt or unreadable
    state file (torn write after power loss, manual tampering) must never
    brick every subsequent sync run — treat it exactly like a missing file:
    start from empty state. Worst case is one full re-extract cycle, which
    is self-healing; an unhandled JSONDecodeError here would require manual
    cleanup before any run could succeed again."""
    if not state_path.exists():
        return {}
    try:
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "state file %s is unreadable/corrupt (%s) — starting from empty state; "
            "all projects will be treated as changed this run",
            state_path,
            exc,
        )
        return {}
    if not isinstance(loaded, dict):
        log.warning(
            "state file %s does not contain a JSON object — starting from empty state",
            state_path,
        )
        return {}
    return loaded


def save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    # fsync the file DATA before the rename: without it, a power loss can
    # make the rename durable while the contents are not, leaving a
    # truncated/empty state file behind (the exact corruption load_state
    # tolerates above — but better to not produce it in the first place).
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(state, indent=2, sort_keys=True))
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(state_path)


def is_worktree_dirty(root: Path) -> bool:
    """Read-only `git status --porcelain` check. Never mutates the target repo."""
    if not (root / ".git").exists():
        return False
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607 - system git from PATH, read-only status
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return bool(result.stdout.strip())


def file_content_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def graph_hash_and_counts(path: Path) -> tuple[str | None, tuple[int, int] | None]:
    """Read graph.json's bytes exactly once; derive both the content hash and
    the node/edge counts from the same buffer (the shrink-guard in
    sync_project.apply_action needs both, per side — reading the file twice
    doubled I/O on every update/extract). Returns (None, None) for a missing
    or unreadable file; (hash, None) when the bytes exist but are not valid
    JSON — mirroring file_content_hash/graph_node_edge_counts semantics."""
    if not path.exists():
        return None, None
    try:
        raw = path.read_bytes()
    except OSError:
        return None, None
    digest = hashlib.sha256(raw).hexdigest()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return digest, None
    counts = (len(data.get("nodes", [])), len(data.get("links", data.get("edges", []))))
    return digest, counts


def graph_node_edge_counts(path: Path) -> tuple[int, int] | None:
    """Cheap count via json.load — never used on real production-sized graphs
    from the exploring agent's context (subprocess/engine-internal only)."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return len(data.get("nodes", [])), len(data.get("links", data.get("edges", [])))
