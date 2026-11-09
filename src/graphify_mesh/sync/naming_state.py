"""Mesh-owned naming state.

Replaces graphify's `.graphify_labels.json` / `.graphify_labels.json.sig`
sidecars, which this package no longer has a reason to share now that the
naming stage calls graphify's clustering and labeling in process. Owning the
file means the reuse key, the model that produced a name, and whether a name is
provisional are all explicit and versioned.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from graphify_mesh.sync.validate import PLACEHOLDER_RE

log = logging.getLogger("graphify_mesh.sync.naming_state")

SCHEMA_VERSION = 1
STATE_FILENAME = "naming-state.json"


@dataclass
class CommunityEntry:
    sig: str
    name: str
    provisional: bool = False
    # The LLM label the grounding check refused for this membership signature
    # (no word of it occurs in the members). `name` is then a provisional hub
    # name. The model repeats the label for the same members, so it is not asked
    # again until the signature or the labeling recipe changes.
    ungrounded: str = ""


@dataclass
class RepoEntry:
    """One repo's partition under `cluster_scope=repo`, in repo-local ids.

    `fingerprint` covers the repo's nodes and its internal edges — everything
    its clustering and labeling read — so an equal fingerprint lets the next
    run reuse the partition and names without clustering or an LLM call."""

    fingerprint: str
    assignments: dict[str, int] = field(default_factory=dict)
    communities: dict[str, CommunityEntry] = field(default_factory=dict)


@dataclass
class NamingState:
    merged_fingerprint: str
    clustering: dict = field(default_factory=dict)
    labeling: dict = field(default_factory=dict)
    communities: dict[str, CommunityEntry] = field(default_factory=dict)
    assignments: dict[str, int] = field(default_factory=dict)
    # Absent (empty) under cluster_scope=global and in files written before it.
    # The schema version stays 1: older code ignores the key and still loads.
    repos: dict[str, RepoEntry] = field(default_factory=dict)


def _entries(raw: dict | None) -> dict[str, CommunityEntry]:
    return {
        str(cid): CommunityEntry(
            sig=str(entry["sig"]),
            name=str(entry["name"]),
            # `rejected` came from the retired cross-repo uniqueness rule, which
            # refused legitimately shared names: such a hub name is asked again.
            provisional=bool(entry.get("provisional", False) or entry.get("rejected", False)),
            ungrounded=str(entry.get("ungrounded") or ""),
        )
        for cid, entry in (raw or {}).items()
    }


def _dump_entries(entries: dict[str, CommunityEntry]) -> dict:
    return {
        cid: {
            "sig": e.sig,
            "name": e.name,
            "provisional": e.provisional,
            **({"ungrounded": e.ungrounded} if e.ungrounded else {}),
        }
        for cid, e in entries.items()
    }


def load(naming_dir: Path) -> NamingState | None:
    """Parsed state, or None when absent, unreadable, or written by a schema
    this version does not understand. None always means "run the full stage",
    never an error: a lost state file costs one relabeling run, not a failure."""
    path = naming_dir / STATE_FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        return None
    try:
        repos = {
            str(repo): RepoEntry(
                fingerprint=str(entry["fingerprint"]),
                assignments={str(k): int(v) for k, v in (entry.get("assignments") or {}).items()},
                communities=_entries(entry.get("communities")),
            )
            for repo, entry in (raw.get("repos") or {}).items()
        }
        return NamingState(
            merged_fingerprint=str(raw.get("merged_fingerprint", "")),
            clustering=dict(raw.get("clustering") or {}),
            labeling=dict(raw.get("labeling") or {}),
            communities=_entries(raw.get("communities")),
            assignments={str(k): int(v) for k, v in (raw.get("assignments") or {}).items()},
            repos=repos,
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def save(naming_dir: Path, state: NamingState) -> None:
    """Atomic write: a crash mid-write must not leave a half-parsed state that
    the next run would treat as authoritative."""
    naming_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "merged_fingerprint": state.merged_fingerprint,
        "clustering": state.clustering,
        "labeling": state.labeling,
        "communities": _dump_entries(state.communities),
        "assignments": state.assignments,
    }
    if state.repos:
        payload["repos"] = {
            repo: {
                "fingerprint": entry.fingerprint,
                "assignments": entry.assignments,
                "communities": _dump_entries(entry.communities),
            }
            for repo, entry in state.repos.items()
        }
    fd, tmp_name = tempfile.mkstemp(dir=str(naming_dir), prefix=".naming-state-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, naming_dir / STATE_FILENAME)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def seed_from_graphify_sidecars(naming_dir: Path) -> dict[str, CommunityEntry]:
    """One-time migration off graphify's sidecars.

    Only cids present in BOTH files are taken: a name whose membership signature
    is unknown cannot be matched to a community later, and carrying it under a
    guessed signature would attach an old name to a different community.
    """
    out_dir = naming_dir / "graphify-out"
    labels = _read_json_object(out_dir / ".graphify_labels.json")
    sigs = _read_json_object(out_dir / ".graphify_labels.json.sig")
    seeded: dict[str, CommunityEntry] = {}
    for cid, name in labels.items():
        sig = sigs.get(cid)
        if not isinstance(sig, str) or not isinstance(name, str) or not name:
            continue
        if PLACEHOLDER_RE.match(name):
            # A legacy `Community 7` is not a name; carrying it would block
            # publish at validate.py:207 on every future run.
            continue
        # provisional=True on purpose: upstream's own cluster-only writes hub
        # fallbacks into these sidecars, so a matching signature does not prove
        # an LLM produced the name. The text is kept as the outage fallback and
        # the next healthy run relabels it.
        seeded[str(cid)] = CommunityEntry(sig=sig, name=name, provisional=True)
    if seeded:
        log.info("naming: seeded %d community name(s) from graphify sidecars", len(seeded))
    return seeded


def names_by_sig(state: NamingState | None) -> dict[str, tuple[str, bool]]:
    """`{membership_sig: (name, provisional)}` — names travel by membership, not
    by community id, because ids renumber when membership shifts."""
    if state is None:
        return {}
    return {e.sig: (e.name, e.provisional) for e in state.communities.values()}


def _read_json_object(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}
