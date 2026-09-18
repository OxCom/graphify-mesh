"""Shared bounded cache of source-file line lists for snippet building.

Both snippet consumers (`embedding.build_snippet`, used per-node by the WS3
embed stage AND per-node again by the WS1.6 lexical-index stage via
`lexical_index.build_lexical_index`) used to open and line-scan the same
source file once per node, per stage — a file contributing 50 nodes was
opened ~100 times per sync run. This module reads a file's lines ONCE (up to
the caller's line cap, mirroring `SNIPPET_READ_LINE_CAP`) and lets every
snippet window slice out of the cached tuple instead.

Memory bounds (all four are hard caps, so a pathological repo can't blow RSS):

  * per file: at most ``line_cap`` lines are ever read or stored — the same
    cap the old streaming reader enforced, so cached content is exactly the
    prefix the per-call scan used to see;
  * per line: at most ``SOURCE_CACHE_MAX_LINE_CHARS`` characters are kept;
    a longer line is truncated to that prefix rather than dropping the file,
    since snippet windows only ever show a small slice of a line anyway;
  * per file total: reading stops once ``SOURCE_CACHE_MAX_FILE_CHARS``
    characters have been kept, so a newline-free blob costs a bounded amount
    even below the line cap;
  * total: at most ``SOURCE_CACHE_MAX_FILES`` files are resident at once
    (``functools.lru_cache`` eviction).

Callers keep full responsibility for path validation (absolute-path and
``..``-traversal rejection against ``source_root``) BEFORE consulting the
cache — the cache is keyed on the already-resolved path and never resolves
or validates anything itself.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Upper bound on distinct files resident in the cache at once. 256 files x
# SNIPPET_READ_LINE_CAP (4000) lines is a small, fixed ceiling regardless of
# how many nodes or repos a run touches.
SOURCE_CACHE_MAX_FILES = 256

# Upper bound on the characters kept for a single line. Snippet windows show a
# short slice of a line, so a minified bundle or a generated blob on one line
# is truncated to this prefix instead of being cached whole.
SOURCE_CACHE_MAX_LINE_CHARS = 4096

# Upper bound on the characters kept for one file. Reading stops as soon as it
# is reached, so a newline-free file costs a bounded amount even when it stays
# under the line cap.
SOURCE_CACHE_MAX_FILE_CHARS = 1_000_000

# Size of one read from disk. Bounds the transient buffer, which is why the
# reader chunks instead of iterating lines: line iteration would materialize a
# 10 MB line in full before any cap could apply.
SOURCE_CACHE_READ_CHUNK_CHARS = 65536


@lru_cache(maxsize=SOURCE_CACHE_MAX_FILES)
def _read_capped_lines(
    path_str: str, st_mtime_ns: int, st_size: int, line_cap: int
) -> tuple[str, ...] | None:
    """Reads up to ``line_cap`` newline-stripped lines of ``path_str``.

    Keyed on file identity + version (``st_mtime_ns``/``st_size`` from the
    caller's ``os.stat``) so a long-lived process (the MCP server) never
    serves pre-edit lines after the file changes on disk.

    A line longer than ``SOURCE_CACHE_MAX_LINE_CHARS`` is stored truncated to
    that prefix and its tail is discarded; reading stops once
    ``SOURCE_CACHE_MAX_FILE_CHARS`` characters have been kept.

    Encoding/error handling is byte-for-byte the old per-call reader's:
    ``utf-8`` with ``errors="replace"``, each line stripped of its trailing
    newline. Returns ``None`` (cached, like any other result) when the file
    cannot be read — callers treat that exactly like the old reader's OSError
    path (empty snippet, never an exception)."""
    lines: list[str] = []
    total_chars = 0
    buf = ""
    # True while the tail of an over-long line is being discarded up to its
    # newline, so the discarded part never accumulates in `buf`.
    dropping = False

    def _room() -> bool:
        return len(lines) < line_cap and total_chars < SOURCE_CACHE_MAX_FILE_CHARS

    def _keep() -> int:
        """Characters the next appended line may keep. The file allowance is
        the binding one near the end of the budget, so clamping to it is what
        keeps the retained total at or below SOURCE_CACHE_MAX_FILE_CHARS."""
        return min(SOURCE_CACHE_MAX_LINE_CHARS, SOURCE_CACHE_MAX_FILE_CHARS - total_chars)

    try:
        with open(path_str, encoding="utf-8", errors="replace") as fh:
            while _room():
                chunk = fh.read(SOURCE_CACHE_READ_CHUNK_CHARS)
                if not chunk:
                    break
                buf += chunk
                while _room():
                    newline_at = buf.find("\n")
                    if dropping:
                        if newline_at < 0:
                            buf = ""
                            break
                        buf = buf[newline_at + 1 :]
                        dropping = False
                        continue
                    if newline_at < 0:
                        if len(buf) > SOURCE_CACHE_MAX_LINE_CHARS:
                            kept = buf[: _keep()]
                            lines.append(kept)
                            total_chars += len(kept)
                            buf = ""
                            dropping = True
                            continue
                        break
                    line = buf[:newline_at][: _keep()]
                    lines.append(line)
                    total_chars += len(line)
                    buf = buf[newline_at + 1 :]
            # A final line without a trailing newline is kept, as before.
            if buf and not dropping and _room():
                lines.append(buf[: _keep()])
    except OSError:
        return None
    return tuple(lines)


def get_source_lines(path: Path, line_cap: int) -> tuple[str, ...] | None:
    """Cached line list for an already-validated, already-resolved source
    path. ``None`` means the file was unreadable (missing, permissions, ...)
    — the caller degrades to an empty snippet, same as before the cache.

    One ``os.stat`` per call keys the cache on file version; a failed stat
    returns the unreadable sentinel WITHOUT touching the cache, so no bogus
    key is ever cached for a missing file."""
    path_str = str(path)
    try:
        stat_result = os.stat(path_str)
    except OSError:
        return None
    return _read_capped_lines(path_str, stat_result.st_mtime_ns, stat_result.st_size, line_cap)


def clear_source_cache() -> None:
    """Drops every cached file. Exposed for tests and for long-lived callers
    that want a fresh view of the filesystem between logical runs."""
    _read_capped_lines.cache_clear()
