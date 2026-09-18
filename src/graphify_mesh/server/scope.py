"""WS5 scope contract: every tool takes `scope: current|all|repo:<id>`.

Fail-closed by design (plan WS5 bullet 1 / requirement 2): an omitted or
`"current"` scope resolves the client's cwd against `registry.json` to a
single `repo_id`. If no registered repo's root is an ancestor of cwd, this
raises `ScopeResolutionError` — it NEVER silently falls back to a global/
unscoped search. The only ways to get cross-repo results are `scope="all"`,
an explicit `scope="repo:<id>"`, or the dedicated `cross_project` tool.

Scope filtering must be applied to the CANDIDATE SET before ranking, not as
a post-filter of an already-ranked global top-K — see `retrieval.py`'s
`gather_candidates`, which takes the resolved repo_id set and filters at
candidate-generation time. Doing this only as a final filter step was an
explicitly named failure mode in the plan (a large repo's global top-K can
crowd out every current-project hit before the filter ever runs).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path


class ScopeResolutionError(ValueError):
    """Raised whenever scope cannot be resolved to a concrete, known repo_id
    set. Callers must surface this as a hard tool error, never swallow it
    into an unscoped/global fallback."""


@dataclass(frozen=True)
class RegistryEntry:
    repo_id: str
    root: Path
    enabled: bool


@dataclass(frozen=True)
class ScopeDecision:
    mode: str  # "repo" | "all"
    # Always a concrete set of enabled repo_ids, for mode "repo" and "all"
    # alike: `None` (no filter at all) would serve repos the registry has
    # since disabled but the published generation still carries.
    repo_ids: frozenset[str]


# Parsed-registry cache keyed per path on (st_mtime_ns, st_size, st_ino):
# registry.json is consulted on EVERY tool call, and re-reading + re-parsing +
# Path.resolve() per call is pure waste when the file has not changed. mtime
# alone is NOT a safe key — mtime-preserving overwrites (rsync -t, cp -p,
# tar extraction) would keep serving stale scope authorization — so size and
# inode are part of the signature too. A missing/unreadable registry is never
# cached, so the fail-closed behavior (no entries -> ScopeResolutionError
# downstream) is preserved and recovers the instant the file (re)appears.
_registry_cache: dict[str, tuple[tuple[int, int, int], list[RegistryEntry]]] = {}
_registry_cache_lock = threading.Lock()


def load_registry_entries(registry_path: Path) -> list[RegistryEntry]:
    if not registry_path.is_file():
        return []
    try:
        st = registry_path.stat()
    except OSError:
        return []
    signature = (st.st_mtime_ns, st.st_size, st.st_ino)
    cache_key = str(registry_path)
    cached = _registry_cache.get(cache_key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    with _registry_cache_lock:
        cached = _registry_cache.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached[1]
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        disabled = set(data.get("disabled", []))
        entries = []
        for repo in data.get("repos", []):
            if not isinstance(repo, dict) or "repo_id" not in repo or "root" not in repo:
                continue
            entries.append(
                RegistryEntry(
                    repo_id=repo["repo_id"],
                    root=Path(repo["root"]).resolve(),
                    enabled=bool(repo.get("enabled", True)) and repo["repo_id"] not in disabled,
                )
            )
        _registry_cache[cache_key] = (signature, entries)
        return entries


def _match_cwd(cwd: Path, entries: list[RegistryEntry]) -> str | None:
    """Longest-matching (most specific) registered repo root that is an
    ancestor of (or equal to) `cwd`. Deterministic tie-break: if two roots
    tie on path length (should not happen with well-formed registry data),
    the alphabetically-first repo_id wins."""
    resolved_cwd = cwd.resolve()
    candidates = []
    for entry in entries:
        if not entry.enabled:
            continue
        try:
            resolved_cwd.relative_to(entry.root)
        except ValueError:
            continue
        candidates.append(entry)
    if not candidates:
        return None
    candidates.sort(key=lambda e: (-len(str(e.root)), e.repo_id))
    return candidates[0].repo_id


def _enabled_repo_ids(entries: list[RegistryEntry]) -> frozenset[str]:
    return frozenset(e.repo_id for e in entries if e.enabled)


def resolve_scope(scope: str | None, cwd: Path, entries: list[RegistryEntry]) -> ScopeDecision:
    """Fail-closed scope resolution. `scope` is the raw tool argument:
    None/""/"current" -> resolve cwd; "all" -> every enabled repo; "repo:<id>"
    -> explicit single repo, validated against the registry. Any other shape
    raises. "all" with no enabled repo raises rather than resolving to an
    unfiltered search."""
    if scope is None or scope in ("", "current"):
        repo_id = _match_cwd(cwd, entries)
        if repo_id is None:
            raise ScopeResolutionError(
                f"cannot resolve implicit scope: cwd {cwd} does not match any registered, "
                "enabled repo root in registry.json — pass scope='all' or scope='repo:<id>' "
                "explicitly (fail-closed: never silently falls back to a global search)"
            )
        return ScopeDecision(mode="repo", repo_ids=frozenset({repo_id}))

    if scope == "all":
        enabled = _enabled_repo_ids(entries)
        if not enabled:
            raise ScopeResolutionError(
                "scope='all' resolved to no repos: registry.json lists no enabled repo "
                "(fail-closed: an empty registry never means 'search everything')"
            )
        return ScopeDecision(mode="all", repo_ids=enabled)

    if scope.startswith("repo:"):
        repo_id = scope[len("repo:") :]
        known = _enabled_repo_ids(entries)
        if repo_id not in known:
            raise ScopeResolutionError(
                f"unknown or disabled repo_id {repo_id!r} in scope={scope!r}"
            )
        return ScopeDecision(mode="repo", repo_ids=frozenset({repo_id}))

    raise ScopeResolutionError(
        f"invalid scope {scope!r}: expected 'current', 'all', or 'repo:<id>'"
    )


def resolve_repo_list(repos: list[str] | None, entries: list[RegistryEntry]) -> frozenset[str]:
    """For `cross_project(repos=...)`: an explicit repo list is validated
    against the registry (unknown repo_id is a hard error, per fail-closed
    convention); `None`/empty means "all registered, enabled repos" — the
    breadth is safe here (unlike the implicit-scope case above) because
    calling `cross_project` at all is itself the explicit cross-repo opt-in.

    "Enabled" is enforced by returning the enabled repo_id set rather than
    `None`. `None` means "no filter at all" downstream, which served every
    repo the published generation still carried, including one the registry
    had since disabled.

    A registry with nothing enabled raises `ScopeResolutionError`, the same
    answer `resolve_scope('all')` gives to the same registry state. Returning
    an empty set instead made `cross_project` answer "no hits" for a query
    that was never run, hiding an empty or fully disabled registry behind a
    result that reads like a search miss."""
    if not repos:
        enabled = _enabled_repo_ids(entries)
        if not enabled:
            raise ScopeResolutionError(
                "cross_project resolved to no repos: registry.json lists no enabled repo "
                "(fail-closed: an empty registry never means 'search everything')"
            )
        return enabled
    known = _enabled_repo_ids(entries)
    unknown = set(repos) - known
    if unknown:
        raise ScopeResolutionError(
            f"unknown or disabled repo_id(s) in cross_project repos=: {sorted(unknown)}"
        )
    return frozenset(repos)
