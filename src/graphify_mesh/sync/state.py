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
import stat as stat_mod
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from graphify_mesh.sync.config import (
    CURRENT_CLASSIFICATION,
    ClassificationTable,
    categorize_file,
    is_sensitive_path,
)

log = logging.getLogger("graphify_mesh.sync")

# Reserved top-level state key holding the wall/monotonic clock baseline of the
# previous run. It is not a repo_id — readers of the state dict must skip it.
CLOCK_STATE_KEY = "__clock__"

# Largest ignore file read for the digest. A bigger one is skipped like a
# non-regular file: a cloned repo must not make this process buffer it whole.
MAX_IGNORE_FILE_BYTES = 1024 * 1024


@dataclass
class SourceDigest:
    code_hash: str
    semantic_hash: str
    file_count: int
    # Whether a .sql file is in the digest. Gates the SQL extractor-capability
    # stamp (pipeline._repo_versions); computed per walk, never serialized.
    has_sql: bool = field(default=False, compare=False)
    # Root-relative POSIX paths of the files hashed as semantic, sorted. The
    # partial-extract check looks each one up in graphify's manifest, which
    # keys in-root files the same way. Computed per walk, never serialized.
    semantic_paths: tuple[str, ...] = field(default=(), compare=False)

    @property
    def attempt_id(self) -> str:
        """Identity of one extract attempt over the whole source state.

        The `allow_shrink_once` token: it covers code AND semantic content, so a
        code-only change after approval yields a different id. Computed, never
        serialized, and always hex, including for the "empty" sentinel digest.
        """
        raw = f"{self.code_hash}\n{self.semantic_hash}".encode()
        return hashlib.sha256(raw).hexdigest()[:16]

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


BUILD_CONFIG_FILENAME = ".graphify_build.json"


@dataclass(frozen=True)
class IgnoreRules:
    """The corpus options graphify persists next to a repo's graph.

    `graphify update` re-applies them from <collection>/.graphify_build.json
    (graphify.watch._read_build_excludes / _read_build_gitignore), so the digest
    must apply the same ones to count only files graphify reads.
    """

    extra_excludes: tuple[str, ...] = ()
    gitignore: bool = True


def read_build_ignore_rules(collection_path: Path) -> IgnoreRules:
    """The persisted `--exclude` list and gitignore switch, defaults when absent.

    Same tolerance as upstream's readers: a missing, unreadable or malformed
    file means no extra excludes and VCS ignore files honoured.
    """
    path = collection_path / BUILD_CONFIG_FILENAME
    try:
        config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        config = {}
    if not isinstance(config, dict):
        config = {}
    raw = config.get("excludes")
    excludes = (
        tuple(item for item in raw if isinstance(item, str) and item)
        if isinstance(raw, list)
        else ()
    )
    gitignore = config.get("gitignore")
    return IgnoreRules(excludes, gitignore if isinstance(gitignore, bool) else True)


def _ignore_file_lines(path: Path) -> list[str]:
    """Patterns of one ignore file, read only when it is a regular, non-symlink file.

    Upstream `_read_ignore_text` follows symlinks and opens FIFOs: a cloned repo's
    `.gitignore -> /dev/zero` would hang or exhaust this process. git >= 2.32 also
    refuses symlinked ignore files. A skipped file keeps more files in the
    digest, never fewer, so the divergence costs at most a needless rebuild.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return []
    try:
        if not stat_mod.S_ISREG(os.fstat(fd).st_mode):
            return []
        with os.fdopen(fd, "rb", closefd=False) as fh:
            # One byte past the cap tells "at the cap" from "over it", and also
            # bounds a file that grows between fstat and read.
            raw = fh.read(MAX_IGNORE_FILE_BYTES + 1)
    except OSError:
        return []
    finally:
        os.close(fd)
    if len(raw) > MAX_IGNORE_FILE_BYTES:
        log.warning(
            "%s is larger than %d bytes; its ignore rules are not applied to the digest",
            path,
            MAX_IGNORE_FILE_BYTES,
        )
        return []
    from graphify.detect import _parse_gitignore_line

    lines = (_parse_gitignore_line(line) for line in _decode_ignore_text(raw).splitlines())
    return [line for line in lines if line]


def _decode_ignore_text(raw: bytes) -> str:
    """Upstream `_read_ignore_text`'s decoding: UTF-8, then a BOM'd UTF-16, then latin-1."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            pass
    return raw.decode("latin-1")


def _dir_ignore_patterns(directory: Path, gitignore: bool) -> list[tuple[Path, str]]:
    """upstream `_load_dir_own_ignore`, with the safe reader above."""
    names = (".gitignore", ".graphifyignore") if gitignore else (".graphifyignore",)
    return [(directory, line) for name in names for line in _ignore_file_lines(directory / name)]


def _tracked_path_keys(root: Path, vcs_root: Path) -> tuple[set[str], set[str]] | None:
    """upstream `_git_tracked_path_keys` with the repo's config disarmed.

    Upstream runs `git ls-files` in the scanned repo as is, and `ls-files` runs
    a repo-configured `core.fsmonitor` command (checked on git 2.53): in this
    unsandboxed process that is code execution from repo config. The same
    flags as is_worktree_dirty prevent it. None when git fails: the caller then
    drops .gitignore rules rather than ignore tracked files.
    """
    from graphify.detect import _is_regular_file, _path_identity

    if not (vcs_root / ".git").exists():
        # Another VCS marker: upstream applies .gitignore with no tracked exemption.
        return set(), set()
    try:
        # No shell; the only non-literal argument is the repo path, passed as one argv item.
        proc = subprocess.run(  # noqa: S603
            [  # noqa: S607 - system git from PATH, read-only listing
                "git",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.hooksPath=/dev/null",
                "-C",
                str(vcs_root),
                "ls-files",
                "-z",
                "--cached",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
            # GIT_DIR, GIT_WORK_TREE, GIT_INDEX_FILE or GIT_CONFIG_* from this
            # process would point git at another repo or re-arm the config
            # disarmed above, so every GIT_* variable is dropped.
            env={
                **{k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
                "GIT_OPTIONAL_LOCKS": "0",
            },
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    tracked_files: set[str] = set()
    tracked_dirs: set[str] = set()
    for raw in proc.stdout.split(b"\0"):
        if not raw:
            continue
        path = Path(os.path.abspath(vcs_root / os.fsdecode(raw)))
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if not _is_regular_file(path):
            continue
        tracked_files.add(_path_identity(path))
        parent = path.parent
        while parent != root:
            key = _path_identity(parent)
            if key in tracked_dirs:
                break
            tracked_dirs.add(key)
            parent = parent.parent
    return tracked_files, tracked_dirs


class GraphifyIgnoreRules:
    """Answers "would graphify's detect() leave this path out?" for one repo.

    A rebuild of upstream `graphify.detect.ignored_predicate` from its helpers,
    because calling it here is unsafe: it runs `git ls-files` with the repo's own
    config (fsmonitor code execution) and reads ignore files through symlinks and
    FIFOs. `detect()` itself is never called: it activates the repo's
    `.graphifyrc` language overrides process-wide. Deliberate differences, each
    keeping more files in the digest: no user-wide `core.excludesFile` (the
    child runs under a staging HOME and does not see it either), no symlinked or
    non-regular ignore files, and no .gitignore rules when git fails.

    `root` is the path the digest walks; paths are mapped onto its resolved form,
    which is what upstream anchors patterns at.
    """

    def __init__(self, root: Path, rules: IgnoreRules) -> None:
        from graphify import detect as upstream

        self._upstream = upstream
        self._walk_root = Path(os.path.abspath(root))
        self._root = root.resolve()
        ceiling = upstream._find_vcs_root(self._root) or self._root
        lineage = [self._root]
        while lineage[-1] != ceiling and lineage[-1].parent != lineage[-1]:
            lineage.append(lineage[-1].parent)
        self._gitignore = rules.gitignore
        self._patterns: list[tuple[Path, str]] = []
        self._explicit: list[tuple[Path, str]] = []
        if self._gitignore:
            info_exclude = upstream._git_info_exclude(ceiling)
            if info_exclude is not None:
                self._patterns.extend((ceiling, line) for line in _ignore_file_lines(info_exclude))
        for directory in reversed(lineage):
            self._patterns.extend(_dir_ignore_patterns(directory, self._gitignore))
            self._explicit.extend(_dir_ignore_patterns(directory, False))
        self._tracked: tuple[set[str], set[str]] = (set(), set())
        if self._gitignore and len(self._patterns) > len(self._explicit):
            tracked = _tracked_path_keys(self._root, ceiling)
            if tracked is None:
                log.warning(
                    "%s: git ls-files failed; .gitignore rules are not applied to its digest",
                    self._root,
                )
                self._gitignore = False
                self._patterns = list(self._explicit)
            else:
                self._tracked = tracked
        for pattern in rules.extra_excludes:
            line = upstream._parse_gitignore_line(pattern)
            if line:
                self._patterns.append((self._root, line))
                self._explicit.append((self._root, line))
        self._loaded: set[Path] = {self._root}
        self._linked: dict[Path, bool] = {}
        self._linked_root = getattr(upstream, "_linked_worktree_main_root", None)
        self._cache: dict[Path, bool] = {}
        self._explicit_cache: dict[Path, bool] = {}

    def ignores_file(self, path: Path) -> bool:
        return self._ignores(path, is_dir=False)

    def ignores_dir(self, path: Path) -> bool:
        """Whether detect() prunes this directory, so nothing under it is read."""
        return self._ignores(path, is_dir=True)

    def _ignores(self, path: Path, is_dir: bool) -> bool:
        upstream = self._upstream
        try:
            rel_parts = Path(os.path.abspath(path)).relative_to(self._walk_root).parts
        except ValueError:
            return False
        if not rel_parts:
            return False
        target = self._root.joinpath(*rel_parts)
        if not is_dir and (
            target.name in upstream._SKIP_FILES or upstream._is_installed_graphify_file(target)
        ):
            return True
        checked = rel_parts if is_dir else rel_parts[:-1]
        parent = self._root
        for part in checked:
            if upstream._is_noise_dir(part, parent):
                return True
            parent = parent / part
            # graphify < 0.9.97 has no linked-worktree prune (#4683); skipping keeps parity.
            if self._linked_root is None:
                continue
            if parent not in self._linked:
                main_root = self._linked_root(parent)
                self._linked[parent] = main_root is not None and upstream._resolves_under_root(
                    main_root, self._root
                )
            if self._linked[parent]:
                return True
        ancestor = self._root
        for part in rel_parts[:-1]:
            ancestor = ancestor / part
            if ancestor not in self._loaded:
                self._loaded.add(ancestor)
                self._patterns.extend(_dir_ignore_patterns(ancestor, self._gitignore))
                self._explicit.extend(_dir_ignore_patterns(ancestor, False))
        return bool(
            upstream._is_scan_ignored(
                target,
                self._root,
                self._patterns,
                self._explicit,
                self._tracked[0],
                self._tracked[1],
                cache=self._cache,
                explicit_cache=self._explicit_cache,
            )
        )


def compute_source_manifest(
    root: Path,
    table: ClassificationTable = CURRENT_CLASSIFICATION,
    ignore_rules: IgnoreRules | None = None,
) -> SourceDigest:
    """Walk root, hashing (relpath, size, content-digest) separately per category.

    `table` picks the file classification; a legacy table reproduces the hashes
    stored by an older CLASSIFICATION_VERSION (see decide_action's migration).
    One-table form of compute_source_manifests."""
    return compute_source_manifests(root, (table,), ignore_rules)[0]


def compute_source_manifests(
    root: Path,
    tables: Sequence[ClassificationTable],
    ignore_rules: IgnoreRules | None = None,
) -> tuple[SourceDigest, ...]:
    """One digest per table in `tables`, all taken from a single walk of root.

    Migration compares the digest under the current table with the one under
    the table the stored state was written with. Two separate walks could see
    different file contents (a document edited between them), and the repo
    would be stamped as migrated without the extract that edit needs. Here each
    file is stat'ed and read at most once and that one snapshot feeds every
    table.

    Uses os.walk with in-place `dirnames` pruning so ignored trees
    (node_modules/.git/vendor/...) are never even descended into. A dir is
    pruned only when every table prunes it; each table then applies its own
    full-`parts` filter to every candidate. The candidate list is sorted with
    Path ordering, so entry order and content — and therefore the digest — are
    byte-identical to the rglob-era implementation for unchanged trees (state
    compatibility).

    A table with `honours_ignore_rules` also drops every file graphify's ignore
    rules exclude (GraphifyIgnoreRules), with `ignore_rules` holding the repo's
    persisted build options; None means graphify's defaults. A table with
    `drops_sensitive_paths` drops every file graphify skips as a likely secret
    (config.is_sensitive_path)."""
    if not root.is_dir():
        empty = SourceDigest(code_hash="empty", semantic_hash="empty", file_count=0)
        return tuple(empty for _ in tables)

    content_digests = _use_content_digests()
    honours = [table.honours_ignore_rules for table in tables]
    ignore = GraphifyIgnoreRules(root, ignore_rules or IgnoreRules()) if any(honours) else None
    # graphify scans root.resolve() and tests the absolute path for secrets.
    resolved_root = root.resolve()

    candidates: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        kept: list[str] = []
        for d in sorted(dirnames):
            if all(t.prunes(d) for t in tables):
                continue
            # A dir graphify prunes holds no file any honouring table counts;
            # descending it only pays when some table still reads it (migration).
            if (
                ignore is not None
                and all(t.prunes(d) or t.honours_ignore_rules for t in tables)
                and ignore.ignores_dir(base / d)
            ):
                continue
            kept.append(d)
        dirnames[:] = kept
        for filename in filenames:
            candidates.append(base / filename)

    code_entries: list[list[str]] = [[] for _ in tables]
    semantic_entries: list[list[str]] = [[] for _ in tables]
    counts = [0 for _ in tables]
    has_sql = [False for _ in tables]
    semantic_paths: list[list[str]] = [[] for _ in tables]
    for path in sorted(candidates):
        # Kept identical to the rglob-era filter (checked against the FULL
        # path parts, filename included): a *file* named e.g. `vendor`, or a
        # root that itself lives under an ignored dir, must keep hashing the
        # same as before.
        categories: list[str | None] = [
            None if any(table.prunes(part) for part in path.parts) else categorize_file(path, table)
            for table in tables
        ]
        if ignore is not None and any(
            honour and category not in (None, "ignore")
            for honour, category in zip(honours, categories, strict=True)
        ):
            if ignore.ignores_file(path):
                categories = [
                    None if honour else category
                    for honour, category in zip(honours, categories, strict=True)
                ]
        if any(
            table.drops_sensitive_paths and category not in (None, "ignore")
            for table, category in zip(tables, categories, strict=True)
        ) and is_sensitive_path(resolved_root / path.relative_to(root)):
            categories = [
                None if table.drops_sensitive_paths else category
                for table, category in zip(tables, categories, strict=True)
            ]
        if all(category in (None, "ignore") for category in categories):
            continue
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        rel = str(path.relative_to(root))
        entry = f"{rel}:{stat.st_size}:{_change_marker(path, stat, content_digests)}"
        for i, category in enumerate(categories):
            if category == "code":
                code_entries[i].append(entry)
            elif category == "semantic":
                semantic_entries[i].append(entry)
                semantic_paths[i].append(path.relative_to(root).as_posix())
            else:
                continue
            counts[i] += 1
            if category == "code" and path.suffix.lower() == ".sql":
                has_sql[i] = True

    def _hash(entries: list[str]) -> str:
        h = hashlib.sha256()
        for e in entries:
            h.update(e.encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()[:16]

    return tuple(
        SourceDigest(
            code_hash=_hash(code_entries[i]),
            semantic_hash=_hash(semantic_entries[i]),
            file_count=counts[i],
            has_sql=has_sql[i],
            semantic_paths=tuple(semantic_paths[i]),
        )
        for i in range(len(tables))
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
            # -c flags first: the scanned repo's own config is untrusted input, and
            # core.fsmonitor/hooksPath would run its commands in this process's context.
            [  # noqa: S607 - system git from PATH, read-only status
                "git",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.hooksPath=/dev/null",
                "status",
                "--porcelain",
            ],
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
