"""Reuse of `graphify merge-graphs` output when its inputs have not changed.

The merge re-reads, re-composes and re-serializes every per-repo graph on
each run (about 20 s for the fleet), although its output is a pure function
of what it reads. An entry holds the raw bytes of `merged-graph.json` under
`<cache_dir>/merge/<key>.bin` (about 93 MB for the fleet), streamed in and
out so the payload is never held in RAM.

The key covers everything the child reads: the ordered input paths (repo
tags derive from them) and each file's content, the installed graphify
(version and wheel RECORD), the networkx and Python versions, and every
variable of the environment the child receives. The installed packages are
read in this process, so a key is produced only when the child provably runs
them: GRAPHIFY_BIN's shebang interpreter is this interpreter in this
environment, graphify is a non-editable install, and this process imports it
from that install. Otherwise there is no key and the merge always runs.
Writes are spooled through `PendingCacheWrites` and land only after publish.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from graphify_mesh.sync import graphify_cli, repo_tags
from graphify_mesh.sync.stage_cache import (
    CACHE_FORMAT,
    PendingCacheWrites,
    copy_entry_payload,
    gc_entries,
)

log = logging.getLogger(__name__)

MERGE_CACHE_SUBDIR = "merge"
# Bump when the key or the payload changes meaning.
MERGE_CACHE_RECIPE = "merge-v2"
# HOME is a fresh, empty per-run staging dir (`run_merge_graphs`): its path
# changes every run and its content never does, so the key records a constant.
_STAGING_HOME = "<per-run empty staging home>"


def _shebang_interpreter(graphify_bin: str) -> str | None:
    """Interpreter path from GRAPHIFY_BIN's `#!` line, or None when the value is
    not a single script with a direct interpreter path (`env` lookups and
    multi-word commands resolve through PATH and are not pinned)."""
    argv = graphify_cli.resolve_bin_argv(graphify_bin)
    if len(argv) != 1:
        return None
    script = argv[0] if os.sep in argv[0] else shutil.which(argv[0])
    if script is None:
        return None
    try:
        with open(script, "rb") as fh:
            first = fh.readline(512)
    except OSError:
        return None
    if not first.startswith(b"#!"):
        return None
    parts = first[2:].decode("utf-8", "replace").split()
    if not parts or os.path.basename(parts[0]) == "env":
        return None
    return parts[0]


@lru_cache(maxsize=8)
def child_runs_this_interpreter(graphify_bin: str) -> bool:
    """Whether GRAPHIFY_BIN runs on this interpreter in this environment.

    Same realpath alone is not enough: every venv's `python` is a symlink to
    the base interpreter, so two venvs share it. The environment is pinned by
    the interpreter's prefix (`<prefix>/bin/python`), which must be sys.prefix.
    """
    interpreter = _shebang_interpreter(graphify_bin)
    if interpreter is None:
        return False
    if os.path.realpath(interpreter) != os.path.realpath(sys.executable):
        return False
    prefix = Path(os.path.abspath(interpreter)).parent.parent
    return os.path.realpath(prefix) == os.path.realpath(sys.prefix)


@lru_cache(maxsize=1)
def _installed_graphify() -> tuple[str, str] | None:
    """(version, sha256 of RECORD) of the graphify install this process
    imports, or None when that cannot be pinned: no wheel metadata, an
    editable install (its RECORD does not change when its source does), or a
    `graphify` module imported from somewhere else than that install."""
    try:
        dist = importlib.metadata.distribution(repo_tags.GRAPHIFY_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return None
    record = dist.read_text("RECORD")
    if not record:
        return None
    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        try:
            dir_info = json.loads(direct_url).get("dir_info") or {}
        except (json.JSONDecodeError, AttributeError):
            return None
        if not isinstance(dir_info, dict) or dir_info.get("editable"):
            return None
    spec = importlib.util.find_spec("graphify")
    if spec is None or spec.origin is None:
        return None
    installed = Path(str(dist.locate_file("graphify/__init__.py")))
    if os.path.realpath(spec.origin) != os.path.realpath(installed):
        return None
    return dist.version, hashlib.sha256(record.encode("utf-8")).hexdigest()


def _networkx_version() -> str:
    try:
        return importlib.metadata.version("networkx")
    except importlib.metadata.PackageNotFoundError:
        return "absent"


@dataclass(frozen=True)
class InputSnapshot:
    """Content hash plus (size, mtime_ns, inode) of each input, in order."""

    hashes: tuple[str, ...]
    stats: tuple[tuple[int, int, int], ...]


def snapshot_inputs(graph_paths: list[Path]) -> InputSnapshot | None:
    """None if an input cannot be read."""
    hashes = []
    stats = []
    for path in graph_paths:
        try:
            with path.open("rb") as fh:
                st = os.fstat(fh.fileno())
                h = hashlib.sha256()
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
        except OSError:
            return None
        hashes.append(h.hexdigest())
        stats.append((st.st_size, st.st_mtime_ns, st.st_ino))
    return InputSnapshot(tuple(hashes), tuple(stats))


def merge_key(
    graph_paths: list[Path],
    inputs: InputSnapshot | None,
    graphify_bin: str,
    child_env: dict[str, str],
) -> str | None:
    """None when the inputs or the producer cannot be pinned: no reuse then.
    `child_env` is the env the merge child receives
    (`graphify_cli.merge_graphs_env`); its values are hashed, never logged."""
    if inputs is None or not child_runs_this_interpreter(graphify_bin):
        return None
    installed = _installed_graphify()
    if installed is None:
        return None
    # A binary that reports another version than the install hashed above is
    # not running it, whatever its shebang says.
    if repo_tags.probed_binary_version(graphify_bin) != installed[0]:
        return None
    header = (
        MERGE_CACHE_RECIPE,
        str(CACHE_FORMAT),
        installed[0],
        installed[1],
        _networkx_version(),
        sys.version,
        os.path.realpath(sys.executable),
        os.path.realpath(sys.prefix),
    )
    h = hashlib.sha256("\x00".join(header).encode("utf-8", "surrogatepass"))
    for name in sorted(child_env):
        value = _STAGING_HOME if name == "HOME" else child_env[name]
        h.update(b"\x00env\x00" + name.encode("utf-8", "surrogatepass"))
        h.update(b"=" + value.encode("utf-8", "surrogatepass"))
    for path, digest in zip(graph_paths, inputs.hashes, strict=True):
        h.update(b"\x00input\x00" + str(path).encode("utf-8", "surrogatepass"))
        h.update(b"\x00" + digest.encode("ascii"))
    return h.hexdigest()


class MergeCache:
    """Cache source for the merge stage.

    After publish and `pending.flush()`, call `collect_garbage()` to drop
    entries this run did not use.
    """

    def __init__(self, cache_dir: Path, pending: PendingCacheWrites) -> None:
        self.dir = cache_dir / MERGE_CACHE_SUBDIR
        self.pending = pending
        self.used_keys: set[str] = set()
        self.consulted = False

    def restore(self, key: str, dest: Path) -> bool:
        """Stream the entry into `dest`; False on a miss or a bad entry."""
        self.consulted = True
        self.used_keys.add(key)
        return copy_entry_payload(self.dir, key, dest)

    def stage(self, key: str, source: Path) -> None:
        self.used_keys.add(key)
        self.pending.add_file(self.dir, key, source)

    def collect_garbage(self) -> int:
        """Remove entries this run did not read or stage. A run that computed
        no key keeps the cache as it is."""
        if not self.consulted:
            return 0
        return gc_entries(self.dir, self.used_keys)
