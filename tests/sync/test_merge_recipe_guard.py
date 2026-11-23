"""Pins the source of the code that decides the merge cache key and whether a
key exists at all. With the stage cache on, an unchanged fleet's merge output
is served from disk, so a change here is invisible until MERGE_CACHE_RECIPE is
bumped. The payload itself is upstream's output; the key pins that producer."""

from __future__ import annotations

import hashlib
import inspect

from graphify_mesh.sync import graphify_cli, merge_cache

GUARDED = (
    # Key, its inputs and the producer pinning.
    merge_cache.merge_key,
    merge_cache.snapshot_inputs,
    merge_cache.InputSnapshot,
    merge_cache.child_runs_this_interpreter,
    merge_cache._shebang_interpreter,
    merge_cache._installed_graphify,
    merge_cache._networkx_version,
    # The child's argv and environment, which the key must describe.
    graphify_cli.run_merge_graphs,
    graphify_cli.merge_graphs_env,
    graphify_cli.child_env,
    graphify_cli._finish_child_env,
    graphify_cli._allowed_parent_env,
    graphify_cli._child_env_allows,
    graphify_cli._child_env_extra,
    graphify_cli._isolated_home_env,
    graphify_cli.resolve_bin_argv,
)
GUARDED_VALUES = (
    merge_cache.MERGE_CACHE_RECIPE,
    merge_cache._STAGING_HOME,
    sorted(graphify_cli.CHILD_ENV_PASSTHROUGH),
    graphify_cli.CHILD_BACKEND_ENV_PREFIXES,
    graphify_cli.CHILD_ENV_EXTRA_VAR,
)
PINNED_SOURCE_SHA256 = "0ae14d13df0e6b525f08cf0f45cbb163b19a72188945e4044200eccd8e048d5c"


def _source_sha256() -> str:
    h = hashlib.sha256()
    for obj in GUARDED:
        h.update(inspect.getsource(obj).encode("utf-8"))
        h.update(b"\0")
    for value in GUARDED_VALUES:
        h.update(repr(value).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def test_merge_key_source_is_pinned():
    actual = _source_sha256()
    assert actual == PINNED_SOURCE_SHA256, (
        "code that shapes the merge cache key, or the merge child's argv or env, "
        "changed. A comment- or docstring-only change: update PINNED_SOURCE_SHA256 "
        "only. A semantic change: bump MERGE_CACHE_RECIPE in merge_cache.py "
        "(invalidates every cached merge) and update PINNED_SOURCE_SHA256. "
        f"New value: {actual!r}"
    )
