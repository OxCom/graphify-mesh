"""Runtime assertion of the ACTUAL graphify clustering backend (C25).

graphify's `cluster.py` `_partition()` picks Leiden or Louvain purely as a
function of what is importable in the interpreter that runs it: it tries
`graspologic_native`, then `graspologic.partition.leiden`, and falls back to
`nx.community.louvain_communities` on ImportError. Nothing reports which one
ran. The naming stage clusters in process, so that interpreter is this one —
this module answers for it, compares the answer against the pinned constant in
config.py, and hard-fails (raises) on any disagreement.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass

from graphify_mesh.sync.config import PINNED_CLUSTERING_BACKEND

LEIDEN_BACKEND = "leiden"
LOUVAIN_BACKEND = "louvain"


class BackendMismatchError(RuntimeError):
    """The clustering backend that would actually run differs from the one
    pinned in config.py. This is a hard failure — the naming stage must never
    run against an unexpectedly different (and un-reviewed) clustering
    algorithm."""


@dataclass
class BackendCheckResult:
    backend: str
    matches_pinned: bool
    interpreter: str


def _graspologic_importable() -> bool:
    """Whether Leiden is available to THIS interpreter.

    Both spellings are checked because upstream tries `graspologic_native`
    first and only then `graspologic.partition.leiden`
    (`graphify/cluster.py`), and on Python >= 3.13 the `leiden` extra installs
    ONLY `graspologic-native` — a check for `graspologic` alone would certify
    Louvain while Leiden actually runs.

    `find_spec` introspects without importing, so neither package is executed
    by the check itself.
    """
    for module in ("graspologic_native", "graspologic"):
        try:
            if importlib.util.find_spec(module) is not None:
                return True
        except (ImportError, ValueError):
            continue
    return False


def assert_pinned_backend(graphify_bin: str | None = None) -> BackendCheckResult:
    """Hard-fail unless the clustering backend that will actually run matches
    `PINNED_CLUSTERING_BACKEND`.

    The naming stage now clusters in process, so the interpreter that decides
    Leiden vs Louvain is this one. `graphify_bin` is accepted and ignored: the
    merge stage still passes it, and dropping the parameter would be a
    signature break for no gain.

    Callers (`naming.run_naming`) must let this propagate uncaught — a mismatch
    blocks the naming stage and, by not being swallowed anywhere in
    pipeline.py, blocks publish end-to-end.
    """
    backend = LEIDEN_BACKEND if _graspologic_importable() else LOUVAIN_BACKEND
    matches = backend == PINNED_CLUSTERING_BACKEND
    if not matches:
        raise BackendMismatchError(
            f"clustering backend mismatch: this interpreter would use {backend!r}, "
            f"but config pins {PINNED_CLUSTERING_BACKEND!r}. Install or remove graspologic "
            f"in {sys.executable}, or change PINNED_CLUSTERING_BACKEND."
        )
    return BackendCheckResult(backend=backend, matches_pinned=matches, interpreter=sys.executable)
