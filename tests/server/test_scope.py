from __future__ import annotations

from pathlib import Path

import pytest
from conftest import registry_repo, write_registry

from graphify_mesh.server.scope import (
    ScopeResolutionError,
    load_registry_entries,
    resolve_repo_list,
    resolve_scope,
)


def test_fail_closed_no_cwd_match_raises_not_silent_global(tmp_path):
    registry_path = tmp_path / "registry.json"
    registered_root = tmp_path / "www" / "known-repo"
    registered_root.mkdir(parents=True)
    write_registry(registry_path, [registry_repo("known.repo", registered_root)])
    entries = load_registry_entries(registry_path)

    unregistered_cwd = tmp_path / "www" / "unregistered-repo"
    unregistered_cwd.mkdir(parents=True)

    with pytest.raises(ScopeResolutionError):
        resolve_scope(None, unregistered_cwd, entries)
    with pytest.raises(ScopeResolutionError):
        resolve_scope("current", unregistered_cwd, entries)


def test_scope_all_resolves_to_the_enabled_repo_ids_not_an_unfiltered_search(tmp_path):
    """`repo_ids=None` means "no filter" downstream, so it served a repo the
    registry had disabled but the published generation still carried."""
    registry_path = tmp_path / "registry.json"
    live, dead = tmp_path / "live", tmp_path / "dead"
    live.mkdir()
    dead.mkdir()
    write_registry(
        registry_path,
        [registry_repo("live.repo", live), registry_repo("dead.repo", dead, enabled=False)],
    )
    entries = load_registry_entries(registry_path)

    decision = resolve_scope("all", tmp_path, entries)
    assert decision.mode == "all"
    assert decision.repo_ids == frozenset({"live.repo"})


def test_scope_all_honours_the_registry_disabled_list(tmp_path):
    registry_path = tmp_path / "registry.json"
    root = tmp_path / "acme"
    root.mkdir()
    write_registry(registry_path, [registry_repo("acme.project", root)], disabled=["acme.project"])
    entries = load_registry_entries(registry_path)

    with pytest.raises(ScopeResolutionError):
        resolve_scope("all", tmp_path, entries)


def test_scope_all_with_no_enabled_repo_fails_closed(tmp_path):
    entries = load_registry_entries(tmp_path / "does-not-exist.json")
    with pytest.raises(ScopeResolutionError):
        resolve_scope("all", tmp_path, entries)


def test_scope_current_resolves_to_longest_matching_registered_root(tmp_path):
    registry_path = tmp_path / "registry.json"
    root = tmp_path / "www" / "acme-project"
    nested_cwd = root / "src" / "deep"
    nested_cwd.mkdir(parents=True)
    write_registry(registry_path, [registry_repo("acme.project", root)])
    entries = load_registry_entries(registry_path)

    decision = resolve_scope(None, nested_cwd, entries)
    assert decision.mode == "repo"
    assert decision.repo_ids == frozenset({"acme.project"})


def test_scope_repo_explicit_rejects_unknown_repo_id(tmp_path):
    registry_path = tmp_path / "registry.json"
    root = tmp_path / "www" / "acme-project"
    root.mkdir(parents=True)
    write_registry(registry_path, [registry_repo("acme.project", root)])
    entries = load_registry_entries(registry_path)

    with pytest.raises(ScopeResolutionError):
        resolve_scope("repo:does.not.exist", tmp_path, entries)

    decision = resolve_scope("repo:acme.project", tmp_path, entries)
    assert decision.repo_ids == frozenset({"acme.project"})


def test_scope_ignores_disabled_repo_for_cwd_match(tmp_path):
    registry_path = tmp_path / "registry.json"
    root = tmp_path / "www" / "disabled-project"
    root.mkdir(parents=True)
    write_registry(registry_path, [registry_repo("disabled.project", root, enabled=False)])
    entries = load_registry_entries(registry_path)

    with pytest.raises(ScopeResolutionError):
        resolve_scope(None, root, entries)


def test_invalid_scope_string_raises():
    with pytest.raises(ScopeResolutionError):
        resolve_scope("bogus-scope-value", Path("/tmp"), [])


def test_resolve_repo_list_none_means_all_registered_enabled(tmp_path):
    """An omitted/empty `repos` must resolve to the ENABLED repo_ids. It used
    to return `None` — no filter — so `cross_project` answered with hits from
    a disabled repo still present in the published generation."""
    registry_path = tmp_path / "registry.json"
    live, dead = tmp_path / "live", tmp_path / "dead"
    live.mkdir()
    dead.mkdir()
    write_registry(
        registry_path,
        [registry_repo("live.repo", live), registry_repo("dead.repo", dead, enabled=False)],
    )
    entries = load_registry_entries(registry_path)

    assert resolve_repo_list(None, entries) == frozenset({"live.repo"})
    assert resolve_repo_list([], entries) == frozenset({"live.repo"})


def test_resolve_repo_list_rejects_a_disabled_repo_id_explicitly_named(tmp_path):
    registry_path = tmp_path / "registry.json"
    dead = tmp_path / "dead"
    dead.mkdir()
    write_registry(registry_path, [registry_repo("dead.repo", dead, enabled=False)])
    entries = load_registry_entries(registry_path)

    with pytest.raises(ScopeResolutionError):
        resolve_repo_list(["dead.repo"], entries)


def test_resolve_repo_list_rejects_unknown_repo_id(tmp_path):
    registry_path = tmp_path / "registry.json"
    root = tmp_path / "www" / "acme-project"
    root.mkdir(parents=True)
    write_registry(registry_path, [registry_repo("acme.project", root)])
    entries = load_registry_entries(registry_path)

    with pytest.raises(ScopeResolutionError):
        resolve_repo_list(["acme.project", "unknown.repo"], entries)

    assert resolve_repo_list(["acme.project"], entries) == frozenset({"acme.project"})


def test_resolve_repo_list_raises_when_nothing_is_enabled(tmp_path):
    """Same registry state, same answer as `resolve_scope('all')`: an empty
    frozenset made `cross_project` look like a search that found nothing."""
    registry_path = tmp_path / "registry.json"
    dead = tmp_path / "dead"
    dead.mkdir()
    write_registry(registry_path, [registry_repo("dead.repo", dead, enabled=False)])
    entries = load_registry_entries(registry_path)

    with pytest.raises(ScopeResolutionError) as excinfo:
        resolve_scope("all", dead, entries)
    all_message = str(excinfo.value)

    for repos in (None, []):
        with pytest.raises(ScopeResolutionError) as excinfo:
            resolve_repo_list(repos, entries)
        assert "cross_project" in str(excinfo.value)
    assert "no enabled repo" in all_message


def test_resolve_repo_list_raises_on_an_empty_registry(tmp_path):
    registry_path = tmp_path / "registry.json"
    write_registry(registry_path, [])
    entries = load_registry_entries(registry_path)

    with pytest.raises(ScopeResolutionError):
        resolve_repo_list(None, entries)
