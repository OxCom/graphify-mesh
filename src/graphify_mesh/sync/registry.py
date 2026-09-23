"""Registry (repo-identity map) load/parse.

Registry is the source of truth for repo identity. A stray `.graphify_root`
file on disk (some of which just contain "." junk, per plan M7) must never
override it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# repo_id is used as a path component (embedding shard filenames, staging
# dirs) — the first character must be alphanumeric so "." / ".." / hidden
# names are impossible, and no separator characters are in the set.
REPO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# A single-use, per-repo shrink authorization: the `semantic_hash` of the exact
# attempt the guard refused (state.compute_source_manifest emits a 16-char sha256
# prefix). The operator copies it out of the refusal line into the repo's registry
# entry, which authorizes THAT shrink for THAT repo — `--allow-shrink` disarms the
# guard for every repo in the run, which is how four repos went through at once on
# 2026-09-22 when only one deletion was intended. The token is matched against the
# refused digest and consumed in per-repo state, so it never authorizes a second
# shrink.
SHRINK_GRANT_PATTERN = re.compile(r"^[0-9a-f]{8,64}$")


@dataclass
class RepoEntry:
    repo_id: str
    root: Path
    collection_path: Path
    enabled: bool = True
    allow_shrink_once: str | None = None


@dataclass
class Registry:
    repos: list[RepoEntry] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)
    external_roots: list[str] = field(default_factory=list)

    def by_collection_path(self) -> dict[str, RepoEntry]:
        return {str(r.collection_path): r for r in self.repos}

    def by_repo_id(self) -> dict[str, RepoEntry]:
        return {r.repo_id: r for r in self.repos}


def _require_non_empty_str(entry: dict, index: int, key: str) -> str:
    value = entry.get(key)
    if isinstance(value, str) and value:
        return value
    raise ValueError(
        f"registry repos[{index}]: missing or invalid {key!r} "
        f"(expected non-empty string, got {value!r})"
    )


def _optional_bool(entry: dict, index: int, key: str, default: bool) -> bool:
    value = entry.get(key, default)
    if isinstance(value, bool):
        return value
    raise ValueError(f"registry repos[{index}]: invalid {key!r} (expected bool, got {value!r})")


def _optional_shrink_grant(entry: dict, index: int) -> str | None:
    """Parse `allow_shrink_once`, rejecting anything that is not a digest token.

    Absent, null and empty all mean "guard armed" — an older registry, or one
    written by a tool that does not know the key, keeps the guard on. A present
    but malformed value is a hard error rather than a silent "armed": the operator
    who typed it believes a shrink is authorized, and a silent refusal would send
    them looking in the wrong place. A garbage token can never authorize anything
    either way, since acceptance requires an exact match against the refused
    digest.
    """
    value = entry.get("allow_shrink_once")
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not SHRINK_GRANT_PATTERN.fullmatch(value):
        raise ValueError(
            f"registry repos[{index}]: invalid 'allow_shrink_once' {value!r} "
            f"(expected the refused source digest, matching {SHRINK_GRANT_PATTERN.pattern})"
        )
    return value


def _str_list(raw: dict, key: str) -> list[str]:
    value = raw.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"registry {key!r} must be a list of strings, got {value!r}")
    for i, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(f"registry {key}[{i}]: expected non-empty string, got {item!r}")
    return list(value)


def _parse_entry(entry, index: int) -> RepoEntry:
    if not isinstance(entry, dict):
        raise ValueError(f"registry repos[{index}]: expected object, got {type(entry).__name__}")
    repo_id = _require_non_empty_str(entry, index, "repo_id")
    if not REPO_ID_PATTERN.fullmatch(repo_id):
        raise ValueError(
            f"registry repos[{index}]: invalid repo_id {repo_id!r} "
            f"(must match {REPO_ID_PATTERN.pattern} — repo_id is used as a filename)"
        )
    return RepoEntry(
        repo_id=repo_id,
        root=Path(_require_non_empty_str(entry, index, "root")),
        collection_path=Path(_require_non_empty_str(entry, index, "collection_path")),
        enabled=_optional_bool(entry, index, "enabled", True),
        allow_shrink_once=_optional_shrink_grant(entry, index),
    )


def load_registry(path: Path) -> Registry:
    """Load and validate the registry file.

    The registry is semi-trusted (org-owned) but the package is public, so a
    malformed registry must fail loudly at load time with the entry index and
    field named — never surface later as a KeyError/AttributeError somewhere
    downstream in the pipeline.
    """
    if not path.exists():
        return Registry()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"registry root: expected JSON object, got {type(raw).__name__}")
    repos_raw = raw.get("repos", [])
    if not isinstance(repos_raw, list):
        raise ValueError(f"registry 'repos': expected list, got {type(repos_raw).__name__}")
    repos = [_parse_entry(entry, index) for index, entry in enumerate(repos_raw)]
    seen: set[str] = set()
    for entry in repos:
        if entry.repo_id in seen:
            raise ValueError(
                f"registry: duplicate repo_id {entry.repo_id!r} — "
                f"by_repo_id() would silently keep only the last one"
            )
        seen.add(entry.repo_id)
    return Registry(
        repos=repos,
        disabled=_str_list(raw, "disabled"),
        external_roots=_str_list(raw, "external_roots"),
    )


def registry_hash(path: Path) -> str:
    """Stable content hash of the registry file, for the generation manifest (C28)."""
    import hashlib

    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
