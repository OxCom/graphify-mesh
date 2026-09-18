"""Scan-integrity guarantees of discovery: registry `root` containment, and
the scan_incomplete signal that keeps a failed scan from looking like a
removed repo.

Mirrors the conventions of tests/sync/test_discovery.py and
tests/sync/test_discovery_depth.py (the `env` fixture, `Env.add_repo`).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from graphify_mesh.sync.discovery import (
    ScanError,
    assert_registry_containment,
    discover_filesystem,
    reconcile,
)
from graphify_mesh.sync.registry import load_registry

# ---------------------------------------------------------------------------
# assert_registry_containment: entry.root
# ---------------------------------------------------------------------------


def test_assert_registry_containment_root_outside_approved_raises(env, tmp_path):
    # collection_path is perfectly legal; only `root` escapes. The pipeline
    # stat-walks `root` and passes it to `graphify update|extract`, so this
    # entry alone would walk a tree nobody approved.
    collection = env.collection_path("c", "rootesc")
    collection.mkdir(parents=True)
    outside = tmp_path / "outside-project"
    outside.mkdir()

    env._repos.append(
        {
            "repo_id": "c.rootesc",
            "root": str(outside),
            "collection_path": str(collection),
            "enabled": True,
        }
    )
    env.write_registry()
    registry = load_registry(env.registry_path)

    with pytest.raises(ValueError, match="c.rootesc"):
        assert_registry_containment(registry, [env.scan_root])


def test_assert_registry_containment_root_at_filesystem_root_raises(env):
    collection = env.collection_path("c", "slash")
    collection.mkdir(parents=True)
    env._repos.append(
        {"repo_id": "c.slash", "root": "/", "collection_path": str(collection), "enabled": True}
    )
    env.write_registry()
    registry = load_registry(env.registry_path)

    with pytest.raises(ValueError, match="root"):
        assert_registry_containment(registry, [env.scan_root])


def test_assert_registry_containment_disabled_entry_with_bad_root_passes(env, tmp_path):
    collection = env.collection_path("c", "off")
    collection.mkdir(parents=True)
    outside = tmp_path / "outside-disabled"
    outside.mkdir()
    env._repos.append(
        {
            "repo_id": "c.off",
            "root": str(outside),
            "collection_path": str(collection),
            "enabled": False,
        }
    )
    env.write_registry()
    registry = load_registry(env.registry_path)

    # Disabled entries are never handed to the pipeline, so they are not gated.
    assert_registry_containment(registry, [env.scan_root])


# ---------------------------------------------------------------------------
# scan_incomplete
# ---------------------------------------------------------------------------


def test_unreadable_scan_subtree_marks_scan_incomplete_and_spares_repo(env):
    if os.geteuid() == 0:
        pytest.skip("chmod 000 has no effect for root")

    root = env.add_repo("example-org.locked", "example-org", "locked", "locked/project")
    env.write_registry()

    scan_errors: list[str] = []
    try:
        root.parent.chmod(0o000)
        discovered = discover_filesystem([env.scan_root], [env.scan_root], scan_errors=scan_errors)
    finally:
        root.parent.chmod(0o755)

    registry = load_registry(env.registry_path)
    report = reconcile(discovered, registry, env.mesh_root, scan_errors=scan_errors)

    assert scan_errors
    assert report.scan_incomplete is True
    assert report.scan_errors == scan_errors
    # The link was never seen because the directory could not be listed —
    # that is not evidence the project is gone.
    assert "example-org.locked" not in report.removed
    assert "example-org.locked" in report.registered
    assert report.to_dict()["scan_incomplete"] is True
    assert report.to_dict()["scan_errors"] == scan_errors


def test_missing_collection_still_reported_missing_under_incomplete_scan(env):
    if os.geteuid() == 0:
        pytest.skip("chmod 000 has no effect for root")

    root = env.add_repo(
        "example-org.gone",
        "example-org",
        "gone",
        "locked/project",
        make_collection=False,
    )
    env.write_registry()

    scan_errors: list[str] = []
    try:
        root.parent.chmod(0o000)
        discovered = discover_filesystem([env.scan_root], [env.scan_root], scan_errors=scan_errors)
    finally:
        root.parent.chmod(0o755)

    registry = load_registry(env.registry_path)
    report = reconcile(discovered, registry, env.mesh_root, scan_errors=scan_errors)

    assert report.scan_incomplete is True
    # collection_path genuinely does not exist: a confirmed fact, still reported.
    assert "example-org.gone" in report.missing
    assert "example-org.gone" not in report.removed


def test_clean_scan_still_reports_genuinely_removed_repo(env):
    env.add_repo(
        "example-org.styleguide", "example-org", "styleguide", "styleguide.example-org.dev.lo"
    )
    env.write_registry()

    shutil.rmtree(env.scan_root / "styleguide.example-org.dev.lo")

    scan_errors: list[str] = []
    discovered = discover_filesystem([env.scan_root], [env.scan_root], scan_errors=scan_errors)
    registry = load_registry(env.registry_path)
    report = reconcile(discovered, registry, env.mesh_root, scan_errors=scan_errors)

    assert scan_errors == []
    assert report.scan_incomplete is False
    assert "example-org.styleguide" in report.removed
    assert "example-org.styleguide" not in report.registered


def test_reconcile_without_scan_errors_argument_keeps_old_behavior(env):
    env.add_repo("example-org.plain", "example-org", "plain", "plain.example-org.dev.lo")
    env.write_registry()
    shutil.rmtree(env.scan_root / "plain.example-org.dev.lo")

    discovered = discover_filesystem([env.scan_root], [env.scan_root])
    registry = load_registry(env.registry_path)
    report = reconcile(discovered, registry, env.mesh_root)

    assert report.scan_incomplete is False
    assert report.scan_errors == []
    assert "example-org.plain" in report.removed


def _add_repo_under(env, repo_id: str, scan_root, root_name: str, product: str, sub: str):
    """Register a repo whose project dir lives under an arbitrary scan root.

    Env.add_repo always builds under `env.scan_root`; the per-root suppression
    tests need a second, unrelated scan root.
    """
    collection = env.collection_path(product, sub)
    collection.mkdir(parents=True, exist_ok=True)
    root = scan_root / root_name
    root.mkdir(parents=True, exist_ok=True)
    (root / "graphify-out").symlink_to(collection, target_is_directory=True)
    env._repos.append(
        {
            "repo_id": repo_id,
            "root": str(root),
            "collection_path": str(collection),
            "enabled": True,
        }
    )
    return root


def test_scan_error_under_one_root_does_not_spare_repo_under_another_root(env, tmp_path):
    """The suppression is proportional to the damage.

    An unreadable directory under scan root A hides only what lives under it.
    A repo under the unrelated scan root B was scanned completely, so its
    missing link is a confirmed removal and must still be classified.
    """
    if os.geteuid() == 0:
        pytest.skip("chmod 000 has no effect for root")

    root_b = tmp_path / "www-b"
    root_b.mkdir()

    locked_root = env.add_repo("example-org.locked", "example-org", "locked", "locked/project")
    gone_root = _add_repo_under(
        env, "example-org.gone-b", root_b, "gone-project", "example-org", "goneb"
    )
    env.write_registry()

    # Genuinely removed from the fully readable root B.
    shutil.rmtree(gone_root)

    scan_errors: list[str] = []
    try:
        locked_root.parent.chmod(0o000)
        discovered = discover_filesystem(
            [env.scan_root, root_b], [env.scan_root, root_b], scan_errors=scan_errors
        )
    finally:
        locked_root.parent.chmod(0o755)

    registry = load_registry(env.registry_path)
    report = reconcile(discovered, registry, env.mesh_root, scan_errors=scan_errors)

    assert report.scan_incomplete is True
    # Under the errored subtree: absence proves nothing.
    assert "example-org.locked" not in report.removed
    assert "example-org.locked" in report.registered
    # Under an unrelated root the scan read completely: absence is a fact.
    assert "example-org.gone-b" in report.removed
    assert "example-org.gone-b" not in report.registered


def test_scan_error_paths_are_structured_not_parsed_from_messages(env):
    if os.geteuid() == 0:
        pytest.skip("chmod 000 has no effect for root")

    root = env.add_repo("example-org.locked2", "example-org", "locked2", "locked/project")
    env.write_registry()

    scan_errors: list[str] = []
    try:
        root.parent.chmod(0o000)
        discover_filesystem([env.scan_root], [env.scan_root], scan_errors=scan_errors)
    finally:
        root.parent.chmod(0o755)

    assert scan_errors
    # Every recorded error names the failing path as an attribute, while still
    # being an ordinary string for logging and JSON.
    for error in scan_errors:
        assert isinstance(error, ScanError)
        assert isinstance(error, str)
        assert isinstance(error.path, Path)
    assert any(error.path == root.parent for error in scan_errors)


def test_unlocatable_scan_errors_suppress_removed_for_every_repo(env):
    """An error carrying no path cannot be localized, so it fails closed."""
    env.add_repo("example-org.plain2", "example-org", "plain2", "plain2.example-org.dev.lo")
    env.write_registry()
    shutil.rmtree(env.scan_root / "plain2.example-org.dev.lo")

    discovered = discover_filesystem([env.scan_root], [env.scan_root])
    registry = load_registry(env.registry_path)
    report = reconcile(
        discovered, registry, env.mesh_root, scan_errors=["opaque failure, no path recorded"]
    )

    assert report.scan_incomplete is True
    assert "example-org.plain2" not in report.removed
    assert "example-org.plain2" in report.registered
