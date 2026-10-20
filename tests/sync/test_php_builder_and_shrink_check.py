"""graphify-php builder dispatch and the attributed shrink guard (2026-10-13).

Two defects drove this:

1. Mesh `update` ran plain `graphify update` for every repo, which never loads
   graphify-php's resolvers, so PHP repos lost their resolver edges on each run
   (cem.messages 5766 -> 4101 edges, cem.hub 39279 -> 29824). A registry
   `builder: graphify-php` now routes `update` through graphify-php-build.
2. The per-repo guard only compared counts. `graphify-shrink-check` attributes
   loss to source changes; its verdict wins when it answers, and the count floor
   stays as the fallback when it is absent or errors.
"""

from __future__ import annotations

import json
import logging
import site
import stat
import sys
import sysconfig
from pathlib import Path

import pytest

from graphify_mesh.sync import graphify_cli, repo_tags, sync_project, validate
from graphify_mesh.sync.cli import build_overrides, parse_args
from graphify_mesh.sync.config import Settings
from graphify_mesh.sync.graphify_cli import probe_php_build_version
from graphify_mesh.sync.pipeline import (
    _explained_shrink_total,
    _producer_changed,
    _producer_upgraded,
    _repo_versions,
    run,
)
from graphify_mesh.sync.registry import load_registry
from graphify_mesh.sync.state import SourceDigest
from graphify_mesh.sync.sync_project import (
    ACTION_EXTRACT,
    ACTION_SKIP,
    ACTION_UPDATE,
    STATUS_FAILED,
    STATUS_SHRINK_REFUSED,
    STATUS_UPDATED,
    ProjectOutcome,
    apply_action,
    decide_action,
    producer_for,
    refusal_key,
)

DIGEST = SourceDigest(code_hash="c1", semantic_hash="s1", file_count=1)


def _graph(nodes: int, edges: int) -> dict:
    return {
        "nodes": [{"id": f"n{i}"} for i in range(nodes)],
        "links": [{"source": "n0", "target": f"n{i % max(nodes, 1)}"} for i in range(edges)],
    }


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _executable(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _collection(tmp_path: Path, nodes: int = 10, edges: int = 10) -> Path:
    collection = tmp_path / "coll"
    collection.mkdir(parents=True)
    _write_json(collection / "graph.json", _graph(nodes, edges))
    _write_json(collection / "manifest.json", {"files": {"a.php": "old"}})
    return collection


def _writing_invoker(nodes: int, edges: int, manifest: dict | None = None, calls=None):
    """An invoker that rewrites graph.json (and manifest.json) like a real child."""

    def invoker(graphify_bin, root, collection_path, staging_home, **kwargs):
        if calls is not None:
            calls.append(kwargs)
        _write_json(collection_path / "graph.json", _graph(nodes, edges))
        if manifest is not None:
            _write_json(collection_path / "manifest.json", manifest)
        return graphify_cli.CliResult(0, "", "")

    return invoker


def _apply(tmp_path, collection, **kwargs):
    return apply_action(
        "cem.messages",
        "graphify",
        tmp_path / "repo",
        collection,
        kwargs.pop("action", ACTION_UPDATE),
        DIGEST,
        tmp_path / "home",
        shrink_tolerance=kwargs.pop("shrink_tolerance", 0.1),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# registry `builder`
# ---------------------------------------------------------------------------


def _registry(tmp_path: Path, **extra) -> Path:
    path = tmp_path / "registry.json"
    entry = {"repo_id": "cem.hub", "root": "/r", "collection_path": "/c", **extra}
    _write_json(path, {"repos": [entry]})
    return path


def test_builder_defaults_to_graphify(tmp_path):
    assert load_registry(_registry(tmp_path)).repos[0].builder == "graphify"


def test_builder_accepts_graphify_php(tmp_path):
    reg = load_registry(_registry(tmp_path, builder="graphify-php"))
    assert reg.repos[0].builder == "graphify-php"


@pytest.mark.parametrize("value", ["graphify_php", "php", "", 1, None])
def test_builder_rejects_unknown_values(tmp_path, value):
    with pytest.raises(ValueError, match=r"repos\[0\]: invalid 'builder'"):
        load_registry(_registry(tmp_path, builder=value))


# ---------------------------------------------------------------------------
# run_php_update
# ---------------------------------------------------------------------------


def test_run_php_update_argv_and_env(tmp_path, monkeypatch):
    monkeypatch.delenv(graphify_cli.CHILD_SANDBOX_VAR, raising=False)
    seen: dict = {}

    def fake_run(argv, cwd, env, timeout=None):
        seen.update(argv=argv, env=env)
        return graphify_cli.CliResult(0, "", "")

    monkeypatch.setattr(graphify_cli, "_run", fake_run)
    collection = tmp_path / "coll"
    graphify_cli.run_php_update(
        "graphify",
        tmp_path / "repo",
        collection,
        tmp_path / "home",
        php_build_bin="graphify-php-build",
    )
    assert seen["argv"] == [
        "graphify-php-build",
        "--graph",
        str(collection / "graph.json"),
        "--",
        "update",
        str(tmp_path / "repo"),
    ]
    assert seen["env"]["GRAPHIFY_MESH_MANAGED"] == "1"
    assert seen["env"]["HOME"] == str(tmp_path / "home")
    # Both must land in writable binds: under bwrap the cwd and /tmp are read-only.
    assert seen["env"]["GRAPHIFY_PHP_STATUS"] == str(collection / ".graphify-php-status.json")
    assert seen["env"]["TMPDIR"] == str(tmp_path / "home" / "tmp")
    assert (tmp_path / "home" / "tmp").is_dir()


def test_run_php_update_child_sees_graphify_php_vars_not_mesh_secrets(tmp_path, monkeypatch):
    monkeypatch.delenv(graphify_cli.CHILD_SANDBOX_VAR, raising=False)
    monkeypatch.setenv("GRAPHIFY_PHP_EXTERNAL_STUBS", "1")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "do-not-leak")
    dump = tmp_path / "env.txt"
    script = _executable(tmp_path / "php-build", f'#!/bin/sh\nenv > "{dump}"\n')

    result = graphify_cli.run_php_update(
        "graphify", tmp_path, tmp_path / "coll", tmp_path / "home", php_build_bin=str(script)
    )

    assert result.ok, result.stderr
    child_env = dict(line.split("=", 1) for line in dump.read_text().splitlines() if "=" in line)
    assert child_env["GRAPHIFY_PHP_EXTERNAL_STUBS"] == "1"
    assert child_env["GRAPHIFY_MESH_MANAGED"] == "1"
    assert "GRAPHIFY_MESH_HTTP_TOKEN" not in child_env


def test_run_php_update_child_imports_editable_install_from_real_user_site(tmp_path, monkeypatch):
    """graphify_php is an editable install (a `.pth` in the real user site) and the
    wrapper's interpreter computes its user site from HOME. With HOME on the staging
    dir the import failed under bwrap (2026-10-13); PYTHONUSERBASE must point the
    child back at the parent's real user base."""
    monkeypatch.delenv(graphify_cli.CHILD_SANDBOX_VAR, raising=False)
    # site caches the user base on first use, so the parent's view is patched
    # rather than its environment.
    user_base = tmp_path / "real-local"
    monkeypatch.setattr(site, "getuserbase", lambda: str(user_base))
    user_site = Path(sysconfig.get_path("purelib", "posix_user", {"userbase": str(user_base)}))
    user_site.mkdir(parents=True)
    src = tmp_path / "editable-src"
    src.mkdir()
    (src / "fake_graphify_php.py").write_text("MARK = 'loaded'\n", encoding="utf-8")
    (user_site / "__editable__.fake_graphify_php.pth").write_text(f"{src}\n", encoding="utf-8")
    out = tmp_path / "out.txt"
    # The production wrapper runs on /usr/bin/python3. A venv interpreter never reads the
    # user site, so under a venv pytest the child must use the base interpreter instead.
    interpreter = getattr(sys, "_base_executable", sys.executable)
    script = _executable(
        tmp_path / "php-build",
        f"#!{interpreter}\n"
        "import fake_graphify_php, pathlib\n"
        f"pathlib.Path({str(out)!r}).write_text(fake_graphify_php.MARK)\n",
    )

    result = graphify_cli.run_php_update(
        "graphify", tmp_path, tmp_path / "coll", tmp_path / "home", php_build_bin=str(script)
    )

    assert result.ok, result.stderr
    assert out.read_text() == "loaded"


# ---------------------------------------------------------------------------
# dispatch by builder
# ---------------------------------------------------------------------------


def _forbidden(*_a, **_kw):
    raise AssertionError("wrong invoker")


def test_php_builder_update_uses_php_invoker(tmp_path, monkeypatch):
    collection = _collection(tmp_path)
    calls: list = []
    monkeypatch.setitem(
        sync_project._PHP_INVOKERS, ACTION_UPDATE, _writing_invoker(12, 12, calls=calls)
    )
    monkeypatch.setitem(sync_project._INVOKERS, ACTION_UPDATE, _forbidden)

    outcome = _apply(tmp_path, collection, builder="graphify-php", php_build_bin="/opt/pb")

    assert outcome.status == STATUS_UPDATED
    assert calls[0]["php_build_bin"] == "/opt/pb"


def test_graphify_builder_update_uses_plain_invoker(tmp_path, monkeypatch):
    collection = _collection(tmp_path)
    monkeypatch.setitem(sync_project._PHP_INVOKERS, ACTION_UPDATE, _forbidden)
    monkeypatch.setitem(sync_project._INVOKERS, ACTION_UPDATE, _writing_invoker(12, 12))
    assert _apply(tmp_path, collection).status == STATUS_UPDATED


def test_php_builder_extract_stays_on_plain_graphify(tmp_path, monkeypatch):
    collection = _collection(tmp_path)
    monkeypatch.setitem(sync_project._PHP_INVOKERS, ACTION_UPDATE, _forbidden)
    monkeypatch.setitem(sync_project._INVOKERS, ACTION_EXTRACT, _writing_invoker(12, 12))
    outcome = _apply(tmp_path, collection, action=ACTION_EXTRACT, builder="graphify-php")
    assert outcome.status == STATUS_UPDATED


@pytest.mark.parametrize("code", [2, 3])
def test_php_resolver_failure_is_failed_with_stderr_tail(tmp_path, monkeypatch, code):
    collection = _collection(tmp_path)
    old = (collection / "graph.json").read_bytes()

    def failing(*_a, **_kw):
        _write_json(collection / "graph.json", _graph(1, 0))
        return graphify_cli.CliResult(code, "", "warning\nresolver laravel-routes failed")

    monkeypatch.setitem(sync_project._PHP_INVOKERS, ACTION_UPDATE, failing)
    outcome = _apply(tmp_path, collection, builder="graphify-php")

    assert outcome.status == STATUS_FAILED
    assert "expected resolver failed" in outcome.reason
    assert "resolver laravel-routes failed" in outcome.reason
    assert (collection / "graph.json").read_bytes() == old


def test_producer_for():
    assert producer_for(ACTION_UPDATE, "graphify") == "update"
    assert producer_for(ACTION_UPDATE, "graphify-php") == "php-update"
    assert producer_for(ACTION_EXTRACT, "graphify-php") == "extract"


# ---------------------------------------------------------------------------
# manifest snapshot + restore
# ---------------------------------------------------------------------------


def test_refusal_restores_graph_and_manifest(tmp_path, monkeypatch):
    collection = _collection(tmp_path)
    old_graph = (collection / "graph.json").read_bytes()
    old_manifest = (collection / "manifest.json").read_bytes()
    monkeypatch.setitem(
        sync_project._INVOKERS, ACTION_UPDATE, _writing_invoker(2, 2, {"files": {}})
    )

    outcome = _apply(tmp_path, collection, shrink_policy="count")

    assert outcome.status == STATUS_SHRINK_REFUSED
    assert (collection / "graph.json").read_bytes() == old_graph
    assert (collection / "manifest.json").read_bytes() == old_manifest


def test_refusal_removes_a_manifest_the_refused_run_created(tmp_path, monkeypatch):
    collection = _collection(tmp_path)
    (collection / "manifest.json").unlink()
    monkeypatch.setitem(
        sync_project._INVOKERS, ACTION_UPDATE, _writing_invoker(2, 2, {"files": {}})
    )

    outcome = _apply(tmp_path, collection, shrink_policy="count")

    assert outcome.status == STATUS_SHRINK_REFUSED
    assert not (collection / "manifest.json").exists()


def test_accepted_update_keeps_new_manifest(tmp_path, monkeypatch):
    collection = _collection(tmp_path)
    monkeypatch.setitem(
        sync_project._INVOKERS, ACTION_UPDATE, _writing_invoker(12, 12, {"files": {"b": 1}})
    )
    assert _apply(tmp_path, collection, shrink_policy="count").status == STATUS_UPDATED
    assert json.loads((collection / "manifest.json").read_text()) == {"files": {"b": 1}}


# ---------------------------------------------------------------------------
# graphify-shrink-check integration
# ---------------------------------------------------------------------------


def _report(verdict: str, **extra) -> dict:
    base = {
        "verdict": verdict,
        "policy": "attributed",
        "tolerance": 0.1,
        "nodes": {"old": 10, "new": 5, "lost": 5, "orphan": 1, "explained": 3},
        "edges": {"old": 10, "new": 5, "lost": 5, "explained": 5},
        "reasons": [],
        "files": [],
    }
    base.update(extra)
    return base


def _fake_check(tmp_path: Path, report: dict | None, code: int) -> tuple[Path, Path]:
    """A graphify-shrink-check double that logs its argv and prints `report`."""
    argv_log = tmp_path / "check-argv.json"
    out = tmp_path / "check-out.json"
    out.write_text(json.dumps(report) if report is not None else "not json", encoding="utf-8")
    script = _executable(
        tmp_path / "fake-shrink-check",
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"json.dump(sys.argv[1:], open({str(argv_log)!r}, 'w'))\n"
        f"sys.stdout.write(open({str(out)!r}).read())\n"
        "sys.stderr.write('boom from the tool\\n')\n"
        f"sys.exit({code})\n",
    )
    return script, argv_log


def test_check_accept_overrides_the_count_floor(tmp_path, monkeypatch):
    collection = _collection(tmp_path)
    script, argv_log = _fake_check(tmp_path, _report("accept"), 0)
    monkeypatch.setitem(
        sync_project._INVOKERS, ACTION_UPDATE, _writing_invoker(5, 5, {"files": {}})
    )

    outcome = _apply(
        tmp_path,
        collection,
        shrink_policy="attributed",
        shrink_check_bin=str(script),
        producer_changed=True,
    )

    assert outcome.status == STATUS_UPDATED
    # nodes: min(lost 5, explained 3 + orphan 1); edges: min(lost 5, explained 5)
    assert outcome.explained_shrink == (4, 5)
    argv = json.loads(argv_log.read_text())
    assert argv[argv.index("--policy") + 1] == "attributed"
    assert argv[argv.index("--tolerance") + 1] == "0.1"
    assert argv[argv.index("--root") + 1] == str(tmp_path / "repo")
    assert argv[argv.index("--new-graph") + 1] == str(collection / "graph.json")
    assert argv[argv.index("--new-manifest") + 1] == str(collection / "manifest.json")
    assert argv[argv.index("--old-graph") + 1] != str(collection / "graph.json")
    assert "--producer-changed" in argv


def test_check_refuse_restores_and_names_top_files(tmp_path, monkeypatch):
    collection = _collection(tmp_path)
    old_manifest = (collection / "manifest.json").read_bytes()
    files = [{"file": f"src/F{i}.php", "class": "resolver", "lost": i} for i in range(7)]
    report = _report("refuse", reasons=["resolver edges lost: 40"], files=files)
    script, argv_log = _fake_check(tmp_path, report, 1)
    # The graph GROWS, so the count floor alone would have accepted it.
    monkeypatch.setitem(
        sync_project._INVOKERS, ACTION_UPDATE, _writing_invoker(20, 20, {"files": {}})
    )

    outcome = _apply(tmp_path, collection, shrink_policy="attributed", shrink_check_bin=str(script))

    assert outcome.status == STATUS_SHRINK_REFUSED
    assert "resolver edges lost: 40" in outcome.reason
    assert "src/F6.php:resolver:6" in outcome.reason
    assert "src/F2.php:resolver:2" in outcome.reason
    assert "src/F1.php" not in outcome.reason  # top 5 only
    assert f'"allow_shrink_once": "{DIGEST.attempt_id}"' in outcome.reason
    assert outcome.refused_manifest == DIGEST
    assert (collection / "manifest.json").read_bytes() == old_manifest
    assert "--producer-changed" not in json.loads(argv_log.read_text())


def test_check_refuse_still_honours_the_shrink_grant(tmp_path, monkeypatch):
    collection = _collection(tmp_path)
    script, _ = _fake_check(tmp_path, _report("refuse", reasons=["x"]), 1)
    monkeypatch.setitem(
        sync_project._INVOKERS, ACTION_UPDATE, _writing_invoker(20, 20, {"files": {}})
    )
    outcome = _apply(
        tmp_path,
        collection,
        shrink_policy="attributed",
        shrink_check_bin=str(script),
        shrink_grant=DIGEST.attempt_id,
    )
    assert outcome.status == STATUS_UPDATED
    assert outcome.consumed_shrink_grant == DIGEST.attempt_id


@pytest.mark.parametrize(
    "authorization",
    [{"allow_shrink": True}, {"shrink_grant": DIGEST.attempt_id}],
    ids=["allow-shrink", "grant"],
)
def test_operator_accepted_shrink_is_explained_in_full(tmp_path, monkeypatch, authorization):
    """The global guard subtracts explained_shrink. A shrink the operator
    accepted outright carried (0, 0), so the global guard refused it again."""
    collection = _collection(tmp_path, nodes=10, edges=10)
    monkeypatch.setitem(
        sync_project._INVOKERS, ACTION_UPDATE, _writing_invoker(6, 7, {"files": {}})
    )

    outcome = _apply(tmp_path, collection, shrink_policy="count", **authorization)

    assert outcome.status == STATUS_UPDATED
    assert outcome.explained_shrink == (4, 3)


@pytest.mark.parametrize(
    ("report", "code"),
    [(_report("accept"), 2), (None, 0), ({"nodes": {}}, 0)],
    ids=["exit-2", "unparseable", "no-verdict"],
)
def test_check_error_falls_back_to_count_and_warns_once(
    tmp_path, monkeypatch, caplog, report, code
):
    script, _ = _fake_check(tmp_path, report, code)
    caplog.set_level(logging.WARNING, logger="graphify_mesh.sync")
    outcomes = []
    for i in range(2):
        collection = _collection(tmp_path / str(i))
        monkeypatch.setitem(
            sync_project._INVOKERS, ACTION_UPDATE, _writing_invoker(5, 5, {"files": {}})
        )
        outcomes.append(
            _apply(tmp_path, collection, shrink_policy="attributed", shrink_check_bin=str(script))
        )
    # 10 -> 5 is below the 10% count floor, so the fallback refuses.
    assert [o.status for o in outcomes] == [STATUS_SHRINK_REFUSED] * 2
    assert "cli reported success but node/edge counts did not grow" in outcomes[0].reason
    warnings = [r for r in caplog.records if "using the count floor" in r.getMessage()]
    assert len(warnings) == 1


@pytest.mark.parametrize(
    ("report", "code"),
    [(_report("accept"), 1), (None, 1), (_report("refuse", reasons=["edges lost"]), 0)],
    ids=["exit-1-accept-json", "exit-1-unparseable", "exit-0-refuse-json"],
)
def test_check_refusal_on_either_channel_fails_closed(tmp_path, monkeypatch, caplog, report, code):
    """An exit code and verdict that disagree used to fall back to the count
    floor, which accepts a growing graph. Any refusal signal now refuses."""
    caplog.set_level(logging.WARNING, logger="graphify_mesh.sync")
    collection = _collection(tmp_path)
    old_graph = (collection / "graph.json").read_bytes()
    script, _ = _fake_check(tmp_path, report, code)
    # The graph GROWS, so the count floor alone would have accepted it.
    monkeypatch.setitem(
        sync_project._INVOKERS, ACTION_UPDATE, _writing_invoker(20, 20, {"files": {}})
    )

    outcome = _apply(tmp_path, collection, shrink_policy="attributed", shrink_check_bin=str(script))

    assert outcome.status == STATUS_SHRINK_REFUSED
    assert "graphify-shrink-check refused" in outcome.reason
    assert outcome.explained_shrink == (0, 0)
    assert (collection / "graph.json").read_bytes() == old_graph
    assert not any("— using the count floor" in r.getMessage() for r in caplog.records)
    assert any("refusing, not using the count floor" in r.getMessage() for r in caplog.records)


def test_missing_check_binary_falls_back_to_count(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="graphify_mesh.sync")
    collection = _collection(tmp_path)
    monkeypatch.setitem(
        sync_project._INVOKERS, ACTION_UPDATE, _writing_invoker(12, 12, {"files": {}})
    )
    outcome = _apply(
        tmp_path,
        collection,
        shrink_policy="attributed",
        shrink_check_bin="graphify-shrink-check-not-installed",
    )
    assert outcome.status == STATUS_UPDATED
    assert any("not found on PATH" in r.getMessage() for r in caplog.records)


def test_count_policy_never_runs_the_tool(tmp_path, monkeypatch):
    collection = _collection(tmp_path)
    script, argv_log = _fake_check(tmp_path, _report("accept"), 0)
    monkeypatch.setitem(
        sync_project._INVOKERS, ACTION_UPDATE, _writing_invoker(12, 12, {"files": {}})
    )
    _apply(tmp_path, collection, shrink_policy="count", shrink_check_bin=str(script))
    assert not argv_log.exists()


def test_check_env_drops_mesh_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "do-not-leak")
    collection = _collection(tmp_path)
    dump = tmp_path / "check-env.json"
    script = _executable(
        tmp_path / "env-check",
        "#!/usr/bin/env python3\n"
        "import json, os\n"
        f"json.dump(dict(os.environ), open({str(dump)!r}, 'w'))\n"
        "print(json.dumps({'verdict': 'accept'}))\n",
    )
    monkeypatch.setitem(
        sync_project._INVOKERS, ACTION_UPDATE, _writing_invoker(12, 12, {"files": {}})
    )
    _apply(tmp_path, collection, shrink_policy="attributed", shrink_check_bin=str(script))
    assert "GRAPHIFY_MESH_HTTP_TOKEN" not in json.loads(dump.read_text())


# ---------------------------------------------------------------------------
# refusal hold keyed by refused_key
# ---------------------------------------------------------------------------

PRIOR = {"code_hash": "c0", "semantic_hash": "s0"}


def _key(digest: SourceDigest, policy: str = "attributed", producer: str = "update") -> str:
    return refusal_key(digest, producer, policy)


def test_ast_only_holds_identical_refused_update():
    current = SourceDigest("c1", "s1", 1)
    prior = dict(PRIOR, refused_key=_key(current), refusal_streak=1)
    assert decide_action(prior, current, True, ast_only=True, refusal_key=_key(current)) == (
        ACTION_SKIP
    )


def test_ast_only_hold_releases_on_policy_or_producer_change():
    current = SourceDigest("c1", "s1", 1)
    prior = dict(PRIOR, refused_key=_key(current), refusal_streak=1)
    for key in (_key(current, policy="count"), _key(current, producer="php-update")):
        assert decide_action(prior, current, True, ast_only=True, refusal_key=key) == (
            ACTION_UPDATE
        )


def test_ast_only_hold_releases_on_code_change():
    refused = SourceDigest("c1", "s1", 1)
    fixed = SourceDigest("c2", "s1", 1)
    prior = dict(PRIOR, refused_key=_key(refused), refusal_streak=1)
    assert decide_action(prior, fixed, True, ast_only=True, refusal_key=_key(fixed)) == (
        ACTION_UPDATE
    )


def test_ast_only_hold_yields_to_matching_grant():
    current = SourceDigest("c1", "s1", 1)
    prior = dict(PRIOR, refused_key=_key(current), refusal_streak=1)
    assert (
        decide_action(
            prior,
            current,
            True,
            shrink_grant=current.attempt_id,
            ast_only=True,
            refusal_key=_key(current),
        )
        == ACTION_UPDATE
    )


def test_refusal_key_includes_known_graphify_version():
    current = SourceDigest("c1", "s1", 1)
    assert refusal_key(current, "update", "attributed", "0.9.66").endswith(
        ":update@0.9.66:attributed"
    )
    assert refusal_key(current, "update", "attributed", None) == _key(current)
    assert refusal_key(current, "update", "attributed", "0.9.66") != refusal_key(
        current, "update", "attributed", "0.9.89"
    )


@pytest.mark.parametrize(
    ("prior", "producer", "version", "changed"),
    [
        ({"producer": "update", "graphify_version": "0.9.66"}, "update", "0.9.89", True),
        ({"producer": "update", "graphify_version": "0.9.89"}, "update", "0.9.89", False),
        ({"producer": "update"}, "update", "0.9.89", True),
        ({"producer": "update", "graphify_version": "0.9.66"}, "update", None, False),
        ({"producer": "update", "graphify_version": "0.9.89"}, "php-update", "0.9.89", True),
        (None, "update", "0.9.89", False),
        ({"code_hash": "c0"}, "update", "0.9.89", True),
        ({"code_hash": "c0"}, "update", None, False),
    ],
    ids=[
        "upgrade",
        "same",
        "prior-unrecorded",
        "current-unknown",
        "label",
        "no-state",
        "pre-producer-state",
        "pre-producer-state-version-unknown",
    ],
)
def test_producer_changed(prior, producer, version, changed):
    assert _producer_changed(prior, producer, version) is changed


def test_code_only_update_is_held_too():
    current = SourceDigest("c1", "s0", 1)
    prior = dict(PRIOR, refused_key=_key(current), refusal_streak=1)
    assert decide_action(prior, current, True, refusal_key=_key(current)) == ACTION_SKIP


@pytest.mark.parametrize(
    "shrink_grant", [None, "0123456789abcdef"], ids=["update-hold", "foreign-grant"]
)
def test_run_level_allow_shrink_releases_the_update_hold(shrink_grant):
    """--allow-shrink never reached decide_action, so a held repo stayed SKIP and
    the flag accepted nothing for it (2026-10-13)."""
    current = SourceDigest("c1", "s1", 1)
    prior = dict(PRIOR, refused_key=_key(current), refusal_streak=1)
    kwargs = {"shrink_grant": shrink_grant, "refusal_key": _key(current)}
    assert decide_action(prior, current, True, ast_only=True, **kwargs) == ACTION_SKIP
    assert (
        decide_action(prior, current, True, ast_only=True, allow_shrink=True, **kwargs)
        == ACTION_UPDATE
    )


def test_run_level_allow_shrink_releases_the_extract_hold():
    current = SourceDigest("c0", "s1", 1)
    prior = dict(PRIOR, refused_semantic_hash="s1", refusal_streak=99)
    assert decide_action(prior, current, True) == ACTION_SKIP
    assert decide_action(prior, current, True, allow_shrink=True) == ACTION_EXTRACT


# ---------------------------------------------------------------------------
# pipeline: builder threading, producer state, refused_key
# ---------------------------------------------------------------------------

REPO_ID = "example-org.php"


@pytest.fixture
def php_repo(env):
    root = env.add_repo(REPO_ID, "example-org", "php", "php.example-org.dev.lo", "repo_a.json")
    env.write_registry()
    return env, root


def _set_builder(env, builder: str) -> None:
    payload = json.loads(env.registry_path.read_text())
    payload["repos"][0]["builder"] = builder
    env.registry_path.write_text(json.dumps(payload), encoding="utf-8")


def _recording_apply(calls: list, status: str = STATUS_UPDATED):
    def fake(repo_id, graphify_bin, root, collection_path, action, current_manifest, *a, **kw):
        calls.append({"action": action, **kw})
        if status == STATUS_SHRINK_REFUSED:
            return ProjectOutcome(repo_id, action, status, refused_manifest=current_manifest)
        return ProjectOutcome(repo_id, action, status, new_manifest=current_manifest)

    return fake


def _state(env) -> dict:
    return json.loads(env.settings().state_path.read_text())[REPO_ID]


def test_pipeline_threads_builder_and_records_producer(php_repo, monkeypatch):
    env, root = php_repo
    calls: list = []
    monkeypatch.setattr("graphify_mesh.sync.pipeline.apply_action", _recording_apply(calls))

    run(env.settings(shrink_policy="count"))
    assert calls[-1]["builder"] == "graphify"
    assert calls[-1]["producer_changed"] is False  # no prior producer
    assert _state(env)["producer"] == "update"

    _set_builder(env, "graphify-php")
    (root / "app.php").write_text("<?php\n", encoding="utf-8")
    run(env.settings(shrink_policy="count", php_build_bin="/opt/pb"))
    assert calls[-1]["action"] == ACTION_UPDATE
    assert calls[-1]["builder"] == "graphify-php"
    assert calls[-1]["php_build_bin"] == "/opt/pb"
    assert calls[-1]["producer_changed"] is True
    assert _state(env)["producer"] == "php-update"


def test_pipeline_ast_only_refusal_is_held_by_refused_key(php_repo, monkeypatch):
    env, root = php_repo
    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: None)
    calls: list = []
    monkeypatch.setattr(
        "graphify_mesh.sync.pipeline.apply_action",
        _recording_apply(calls, status=STATUS_SHRINK_REFUSED),
    )
    settings = env.settings(ast_only=True, shrink_policy="attributed")

    run(settings)
    state = _state(env)
    assert state["refused_key"].endswith(":update:attributed")
    assert len(calls) == 1

    report = run(env.settings(ast_only=True, shrink_policy="attributed"))
    assert len(calls) == 1, "identical refused update re-ran"
    assert {a["repo_id"]: a for a in report.project_actions}[REPO_ID]["action"] == ACTION_SKIP

    run(env.settings(ast_only=True, shrink_policy="count"))
    assert len(calls) == 2, "a policy change must retry the refused source once"


def test_pipeline_allow_shrink_reruns_a_held_update(php_repo, monkeypatch):
    env, _ = php_repo
    calls: list = []
    monkeypatch.setattr(
        "graphify_mesh.sync.pipeline.apply_action",
        _recording_apply(calls, status=STATUS_SHRINK_REFUSED),
    )
    run(env.settings(ast_only=True, shrink_policy="attributed"))
    run(env.settings(ast_only=True, shrink_policy="attributed"))
    assert len(calls) == 1, "precondition: the refused update is held"

    run(env.settings(ast_only=True, shrink_policy="attributed", allow_shrink=True))
    assert len(calls) == 2, "--allow-shrink must re-run the held update"
    assert calls[-1]["allow_shrink"] is True


def test_pipeline_records_graphify_version_and_flags_an_upgrade(php_repo, monkeypatch):
    env, root = php_repo
    calls: list = []
    monkeypatch.setattr("graphify_mesh.sync.pipeline.apply_action", _recording_apply(calls))
    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.66")
    run(env.settings(shrink_policy="count"))
    assert _state(env)["graphify_version"] == "0.9.66"

    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.89")
    (root / "app.php").write_text("<?php\n", encoding="utf-8")
    run(env.settings(shrink_policy="count"))
    assert calls[-1]["producer_changed"] is True, "an upgrade must loosen the shrink check"
    assert _state(env)["graphify_version"] == "0.9.89"


def test_pipeline_graphify_upgrade_retries_a_held_update(php_repo, monkeypatch):
    env, _ = php_repo
    calls: list = []
    monkeypatch.setattr(
        "graphify_mesh.sync.pipeline.apply_action",
        _recording_apply(calls, status=STATUS_SHRINK_REFUSED),
    )
    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.66")
    run(env.settings(ast_only=True, shrink_policy="attributed"))
    assert _state(env)["refused_key"].endswith(":update@0.9.66:attributed")
    run(env.settings(ast_only=True, shrink_policy="attributed"))
    assert len(calls) == 1, "precondition: the refused update is held"

    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.89")
    run(env.settings(ast_only=True, shrink_policy="attributed"))
    assert len(calls) == 2, "a graphify upgrade must retry the held update once"


# ---------------------------------------------------------------------------
# global guard allowed delta
# ---------------------------------------------------------------------------


def test_global_guard_subtracts_explained_shrink():
    refused = validate.validate_shrink_guard((80, 80), (100, 100), False, 0.1)
    assert not refused.ok
    allowed = validate.validate_shrink_guard((80, 80), (100, 100), False, 0.1, (15, 15))
    assert allowed.ok


def test_global_guard_still_refuses_unexplained_remainder():
    result = validate.validate_shrink_guard((50, 100), (100, 100), False, 0.1, (10, 0))
    assert not result.ok
    assert "(10 explained by source changes)" in result.errors[0]


def test_run_all_passes_allowed_shrink():
    data = {"nodes": [{"id": "a::1", "community_name": "x"}], "links": []}
    result = validate.run_all(data, (2, 0), False, True, 0.0, allowed_shrink=(1, 0))
    assert not any("shrink-guard" in e for e in result.errors)


def test_explained_shrink_total_ignores_unaccepted_outcomes():
    outcomes = [
        ProjectOutcome("a", ACTION_UPDATE, STATUS_UPDATED, explained_shrink=(3, 4)),
        ProjectOutcome("b", ACTION_UPDATE, STATUS_UPDATED, explained_shrink=(1, 0)),
        ProjectOutcome("c", ACTION_UPDATE, STATUS_SHRINK_REFUSED, explained_shrink=(9, 9)),
    ]
    assert _explained_shrink_total(outcomes) == (4, 4)


# ---------------------------------------------------------------------------
# settings + CLI
# ---------------------------------------------------------------------------


def _settings(tmp_path) -> Settings:
    return Settings.from_env(
        mesh_root=tmp_path, scan_roots=[tmp_path], registry_path=tmp_path / "r.json"
    )


def test_settings_defaults(tmp_path, monkeypatch):
    for var in (
        "GRAPHIFY_MESH_SHRINK_POLICY",
        "GRAPHIFY_MESH_SHRINK_CHECK_BIN",
        "GRAPHIFY_MESH_PHP_BUILD_BIN",
    ):
        monkeypatch.delenv(var, raising=False)
    s = _settings(tmp_path)
    assert (s.shrink_policy, s.shrink_check_bin, s.php_build_bin) == (
        "attributed",
        "graphify-shrink-check",
        "graphify-php-build",
    )


def test_settings_read_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHIFY_MESH_SHRINK_POLICY", "connected")
    monkeypatch.setenv("GRAPHIFY_MESH_SHRINK_CHECK_BIN", "/opt/check")
    monkeypatch.setenv("GRAPHIFY_MESH_PHP_BUILD_BIN", "/opt/pb")
    s = _settings(tmp_path)
    assert (s.shrink_policy, s.shrink_check_bin, s.php_build_bin) == (
        "connected",
        "/opt/check",
        "/opt/pb",
    )


def test_unknown_policy_falls_back_to_default(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHIFY_MESH_SHRINK_POLICY", "loose")
    assert _settings(tmp_path).shrink_policy == "attributed"


def test_cli_shrink_policy_is_explicit_only():
    assert "shrink_policy" not in build_overrides(parse_args(["--once"]))
    overrides = build_overrides(parse_args(["--once", "--shrink-policy", "count"]))
    assert overrides["shrink_policy"] == "count"
    with pytest.raises(SystemExit):
        parse_args(["--once", "--shrink-policy", "loose"])


def test_no_snapshot_files_leak(tmp_path, monkeypatch):
    import tempfile

    before = set(Path(tempfile.gettempdir()).glob("graphify-mesh-sync-snapshot-*"))
    collection = _collection(tmp_path)
    monkeypatch.setitem(
        sync_project._INVOKERS, ACTION_UPDATE, _writing_invoker(2, 2, {"files": {}})
    )
    _apply(tmp_path, collection, shrink_policy="count")
    after = set(Path(tempfile.gettempdir()).glob("graphify-mesh-sync-snapshot-*"))
    assert after == before


# ---------------------------------------------------------------------------
# rebuild on producer upgrade (graphify, graphify-php)
# ---------------------------------------------------------------------------

PHP_V1 = {"graphify_php": "0.1.0", "graphify": "0.9.89"}
PHP_V2 = {"graphify_php": "0.2.0", "graphify": "0.9.89"}


@pytest.mark.parametrize(
    ("prior", "versions", "changed"),
    [
        ({"producer": "php-update", "producer_versions": PHP_V1}, PHP_V1, False),
        ({"producer": "php-update", "producer_versions": PHP_V1}, PHP_V2, True),
        ({"producer": "php-update"}, PHP_V2, False),
        ({"producer": "php-update", "producer_versions": PHP_V1}, {"graphify_php": None}, False),
        ({"producer": "php-update", "producer_versions": PHP_V1}, None, False),
        ({"producer": "php-update", "producer_versions": "junk"}, PHP_V2, False),
    ],
    ids=["same", "different", "prior-absent", "current-unknown", "no-current", "prior-garbage"],
)
def test_producer_changed_compares_producer_versions(prior, versions, changed):
    assert _producer_changed(prior, "php-update", None, versions) is changed


@pytest.mark.parametrize(
    ("prior", "graphify", "versions", "upgraded"),
    [
        ({"graphify_version": "0.9.66"}, "0.9.89", None, True),
        ({"graphify_version": "0.9.89"}, "0.9.89", None, False),
        ({}, "0.9.89", None, False),
        ({"graphify_version": "0.9.66"}, None, None, False),
        ({"producer": "extract", "graphify_version": "0.9.89"}, "0.9.89", None, False),
        ({"producer_versions": PHP_V1, "graphify_version": "0.9.89"}, "0.9.89", PHP_V2, True),
        ({"graphify_version": "0.9.89"}, "0.9.89", PHP_V2, False),
        (None, "0.9.89", PHP_V2, False),
    ],
    ids=[
        "graphify-upgrade",
        "same",
        "prior-unrecorded",
        "current-unknown",
        "label-only",
        "plugin-upgrade",
        "plugin-prior-absent",
        "no-state",
    ],
)
def test_producer_upgraded(prior, graphify, versions, upgraded):
    """Narrower than _producer_changed: only two known, different versions rebuild."""
    assert _producer_upgraded(prior, graphify, versions) is upgraded


UNCHANGED = SourceDigest("c0", "s0", 1)


def test_unchanged_source_with_producer_upgrade_updates():
    assert decide_action(PRIOR, UNCHANGED, True) == ACTION_SKIP
    assert decide_action(PRIOR, UNCHANGED, True, producer_changed=True) == ACTION_UPDATE
    assert decide_action(None, UNCHANGED, False, producer_changed=True) != ACTION_SKIP


def test_producer_upgrade_update_is_held_by_its_refusal_key():
    prior = dict(PRIOR, refused_key=_key(UNCHANGED), refusal_streak=1)
    kwargs = {"refusal_key": _key(UNCHANGED), "producer_changed": True}
    assert decide_action(prior, UNCHANGED, True, **kwargs) == ACTION_SKIP
    assert decide_action(prior, UNCHANGED, True, allow_shrink=True, **kwargs) == ACTION_UPDATE


def _probe_with(monkeypatch, result: graphify_cli.CliResult, seen: dict | None = None):
    def fake_run(argv, cwd, env, timeout=None):
        if seen is not None:
            seen.update(argv=argv, cwd=cwd, timeout=timeout, empty=not any(Path(cwd).iterdir()))
        return result

    monkeypatch.setattr(graphify_cli, "_run", fake_run)
    return probe_php_build_version("/opt/pb")


def test_probe_parses_the_version_line(monkeypatch):
    seen: dict = {}
    out = graphify_cli.CliResult(0, "graphify-php 0.1.0 (graphify 0.9.89)\n", "")
    assert _probe_with(monkeypatch, out, seen) == {"graphify_php": "0.1.0", "graphify": "0.9.89"}
    assert seen["argv"] == ["/opt/pb", "--version"]
    assert seen["empty"], "the probe must run in an empty scratch cwd"
    assert seen["timeout"] == graphify_cli.PHP_VERSION_PROBE_TIMEOUT_SECONDS


def test_probe_maps_unknown_to_none(monkeypatch):
    out = graphify_cli.CliResult(0, "graphify-php unknown (graphify 0.9.89)\n", "")
    assert _probe_with(monkeypatch, out) == {"graphify_php": None, "graphify": "0.9.89"}


@pytest.mark.parametrize(
    "result",
    [
        graphify_cli.CliResult(1, "graphify-php 0.1.0 (graphify 0.9.89)\n", ""),
        graphify_cli.CliResult(124, "", "timeout: ..."),
        graphify_cli.CliResult(127, "", "exec error: ..."),
        graphify_cli.CliResult(0, "graphify 0.9.89\n", ""),
        graphify_cli.CliResult(0, "graphify-php 0.1.0 (graphify 0.9.89)\nextra\n", ""),
        graphify_cli.CliResult(0, "graphify-php $(x) (graphify 0.9.89)\n", ""),
        graphify_cli.CliResult(0, "", ""),
    ],
    ids=["non-zero", "timeout", "missing", "old-wrapper", "two-lines", "garbage", "empty"],
)
def test_probe_returns_none_on_failure(monkeypatch, result):
    assert _probe_with(monkeypatch, result) is None


def test_probe_empty_binary_runs_nothing(monkeypatch):
    monkeypatch.setattr(graphify_cli, "_run", lambda *a, **k: pytest.fail("ran"))
    assert probe_php_build_version("  ") is None


def test_probe_runs_a_real_stub_and_times_out(tmp_path, monkeypatch):
    ok = _executable(tmp_path / "pb-ok", "#!/bin/sh\necho 'graphify-php 1.2.3 (graphify 4.5.6)'\n")
    assert probe_php_build_version(str(ok)) == {"graphify_php": "1.2.3", "graphify": "4.5.6"}

    slow = _executable(tmp_path / "pb-slow", "#!/bin/sh\nsleep 5\n")
    monkeypatch.setattr(graphify_cli, "PHP_VERSION_PROBE_TIMEOUT_SECONDS", 1)
    assert probe_php_build_version(str(slow)) is None


def _probe_returns(monkeypatch, versions, calls: list | None = None):
    def fake(php_build_bin):
        if calls is not None:
            calls.append(php_build_bin)
        return versions

    monkeypatch.setattr(graphify_cli, "probe_php_build_version", fake)


def test_pipeline_unchanged_source_rebuilds_on_graphify_upgrade(php_repo, monkeypatch):
    env, _ = php_repo
    calls: list = []
    probes: list = []
    _probe_returns(monkeypatch, PHP_V1, probes)
    monkeypatch.setattr("graphify_mesh.sync.pipeline.apply_action", _recording_apply(calls))
    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.66")
    run(env.settings(shrink_policy="count"))
    run(env.settings(shrink_policy="count"))
    assert len(calls) == 1, "precondition: an unchanged source is skipped"
    assert probes == [], "no graphify-php repo, so no probe"
    assert "producer_versions" not in _state(env)

    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.89")
    report = run(env.settings(shrink_policy="count"))
    assert len(calls) == 2
    assert calls[-1]["action"] == ACTION_UPDATE
    assert {a["repo_id"]: a for a in report.project_actions}[REPO_ID]["action"] == ACTION_UPDATE
    assert _state(env)["graphify_version"] == "0.9.89"

    run(env.settings(shrink_policy="count"))
    assert len(calls) == 2, "the upgrade rebuilds once, not every run"


def test_pipeline_unchanged_source_rebuilds_on_plugin_upgrade(php_repo, monkeypatch):
    env, _ = php_repo
    _set_builder(env, "graphify-php")
    calls: list = []
    monkeypatch.setattr("graphify_mesh.sync.pipeline.apply_action", _recording_apply(calls))
    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.66")
    _probe_returns(monkeypatch, PHP_V1)
    run(env.settings(shrink_policy="count"))
    state = _state(env)
    assert state["producer_versions"] == PHP_V1
    assert state["graphify_version"] == "0.9.89", "the wrapper's graphify, not this process's"
    run(env.settings(shrink_policy="count"))
    assert len(calls) == 1, "precondition: same plugin, unchanged source is skipped"

    _probe_returns(monkeypatch, PHP_V2)
    run(env.settings(shrink_policy="count"))
    assert len(calls) == 2
    assert calls[-1]["action"] == ACTION_UPDATE
    assert calls[-1]["producer_changed"] is True
    assert _state(env)["producer_versions"] == PHP_V2


def test_pipeline_without_plugin_info_skips_as_today(php_repo, monkeypatch):
    env, _ = php_repo
    _set_builder(env, "graphify-php")
    calls: list = []
    monkeypatch.setattr("graphify_mesh.sync.pipeline.apply_action", _recording_apply(calls))
    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.89")
    run(env.settings(shrink_policy="count"))  # conftest stub: probe failed
    state = _state(env)
    assert "producer_versions" not in state
    assert state["graphify_version"] == "0.9.89"

    run(env.settings(shrink_policy="count"))
    assert len(calls) == 1

    _probe_returns(monkeypatch, PHP_V2)
    run(env.settings(shrink_policy="count"))
    assert len(calls) == 1, "state without producer_versions must not rebuild once"


def test_pipeline_held_plugin_upgrade_retries_on_the_next_plugin(php_repo, monkeypatch):
    env, _ = php_repo
    _set_builder(env, "graphify-php")
    calls: list = []
    monkeypatch.setattr("graphify_mesh.sync.pipeline.apply_action", _recording_apply(calls))
    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: None)
    _probe_returns(monkeypatch, PHP_V1)
    run(env.settings(shrink_policy="attributed"))

    monkeypatch.setattr(
        "graphify_mesh.sync.pipeline.apply_action",
        _recording_apply(calls, status=STATUS_SHRINK_REFUSED),
    )
    _probe_returns(monkeypatch, PHP_V2)
    run(env.settings(shrink_policy="attributed"))
    assert len(calls) == 2
    assert _state(env)["refused_key"].endswith(":php-update@0.9.89+graphify-php-0.2.0:attributed")
    run(env.settings(shrink_policy="attributed"))
    assert len(calls) == 2, "the refused upgrade update is held"

    _probe_returns(monkeypatch, {"graphify_php": "0.3.0", "graphify": "0.9.89"})
    run(env.settings(shrink_policy="attributed"))
    assert len(calls) == 3, "a further plugin upgrade retries the held update"


def test_producer_change_update_keeps_manifest_semantics(tmp_path, monkeypatch):
    """No manifest is moved aside for a producer-change update: `graphify update`
    re-extracts every file without reading it. A refusal still restores both files."""
    collection = _collection(tmp_path)
    old_manifest = (collection / "manifest.json").read_bytes()
    seen: list = []

    def invoker(graphify_bin, root, collection_path, staging_home, **kwargs):
        seen.append((collection_path / "manifest.json").read_bytes())
        _write_json(collection_path / "graph.json", _graph(2, 2))
        _write_json(collection_path / "manifest.json", {"files": {}})
        return graphify_cli.CliResult(0, "", "")

    monkeypatch.setitem(sync_project._INVOKERS, ACTION_UPDATE, invoker)
    outcome = _apply(tmp_path, collection, shrink_policy="count", producer_changed=True)

    assert seen == [old_manifest]
    assert outcome.status == STATUS_SHRINK_REFUSED
    assert (collection / "manifest.json").read_bytes() == old_manifest


# ---------------------------------------------------------------------------
# fix round 1: probe failure and "unknown" keep the last recorded versions
# ---------------------------------------------------------------------------


ACCEPTED_V1 = {"producer_versions": PHP_V1, "graphify_version": "0.9.89"}
REFUSED_V2 = {
    **ACCEPTED_V1,
    "refused_producer_versions": PHP_V2,
    "refused_graphify_version": "0.9.91",
}


@pytest.mark.parametrize(
    ("builder", "probed", "prior", "expected"),
    [
        ("graphify", PHP_V2, ACCEPTED_V1, ("0.9.66", None, "0.9.66", None)),
        ("graphify-php", PHP_V2, ACCEPTED_V1, ("0.9.89", PHP_V2, "0.9.89", PHP_V2)),
        ("graphify-php", None, None, ("0.9.66", None, "0.9.66", None)),
        ("graphify-php", None, ACCEPTED_V1, ("0.9.89", PHP_V1, "0.9.89", PHP_V1)),
        ("graphify-php", None, REFUSED_V2, ("0.9.89", PHP_V1, "0.9.91", PHP_V2)),
        ("graphify-php", PHP_V2, REFUSED_V2, ("0.9.89", PHP_V2, "0.9.89", PHP_V2)),
        (
            "graphify-php",
            {"graphify_php": "0.2.0", "graphify": None},
            {"graphify_version": "0.9.89"},
            ("0.9.89", {"graphify_php": "0.2.0"}, "0.9.89", {"graphify_php": "0.2.0"}),
        ),
        (
            "graphify-php",
            {"graphify_php": "0.2.0", "graphify": None},
            {},
            (None, {"graphify_php": "0.2.0"}, None, {"graphify_php": "0.2.0"}),
        ),
        (
            "graphify-php",
            None,
            {"producer_versions": {"graphify_php": 7}},
            ("0.9.66", None, "0.9.66", None),
        ),
    ],
    ids=[
        "plain-builder",
        "probed",
        "probe-failed-no-state",
        "probe-failed-keeps-accepted",
        "probe-failed-key-keeps-refused",
        "probe-ok-overrides-refused",
        "graphify-unknown-keeps-recorded",
        "graphify-unknown-records-none",
        "garbage-prior",
    ],
)
def test_repo_versions(builder, probed, prior, expected):
    """Recorded versions never take a refused overlay; key versions prefer it."""
    assert tuple(_repo_versions(builder, "0.9.66", probed, prior)) == expected


def test_pipeline_held_refusal_survives_a_probe_failure(php_repo, monkeypatch):
    env, root = php_repo
    calls: list = []
    held = _refuse_under_v2(env, root, monkeypatch, calls)
    assert held["refused_producer_versions"] == PHP_V2

    _probe_returns(monkeypatch, None)
    report = run(env.settings(ast_only=True, shrink_policy="attributed"))
    assert len(calls) == 2, "a probe failure released the held refusal"
    assert {a["repo_id"]: a for a in report.project_actions}[REPO_ID]["action"] == ACTION_SKIP
    assert _state(env)["refused_key"] == held["refused_key"]

    _probe_returns(monkeypatch, PHP_V2)
    run(env.settings(ast_only=True, shrink_policy="attributed"))
    assert len(calls) == 2, "the probe recovering must not release it either"


def test_pipeline_warns_once_when_the_probe_fails(php_repo, monkeypatch, caplog):
    env, _ = php_repo
    monkeypatch.setattr("graphify_mesh.sync.pipeline.apply_action", _recording_apply([]))
    caplog.set_level(logging.WARNING, logger="graphify_mesh.sync")
    run(env.settings(shrink_policy="count"))
    assert not [r for r in caplog.records if "--version gave no usable answer" in r.message]

    _set_builder(env, "graphify-php")
    caplog.clear()
    run(env.settings(shrink_policy="count", php_build_bin="/opt/pb"))
    warned = [r for r in caplog.records if "--version gave no usable answer" in r.message]
    assert len(warned) == 1
    assert warned[0].message.startswith("/opt/pb ")


def test_pipeline_graphify_unknown_keeps_the_recorded_version(php_repo, monkeypatch):
    env, root = php_repo
    _set_builder(env, "graphify-php")
    calls: list = []
    monkeypatch.setattr("graphify_mesh.sync.pipeline.apply_action", _recording_apply(calls))
    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.66")
    _probe_returns(monkeypatch, PHP_V1)
    run(env.settings(shrink_policy="count"))
    assert _state(env)["graphify_version"] == "0.9.89"

    _probe_returns(monkeypatch, {"graphify_php": "0.1.0", "graphify": None})
    (root / "app.php").write_text("<?php\n", encoding="utf-8")
    run(env.settings(shrink_policy="count"))
    assert len(calls) == 2
    assert _state(env)["graphify_version"] == "0.9.89", "this process's graphify was recorded"

    _probe_returns(monkeypatch, PHP_V1)
    run(env.settings(shrink_policy="count"))
    assert len(calls) == 2, "the recovered probe read as a graphify upgrade"


# ---------------------------------------------------------------------------
# fix round 2: key graphify from state, accepted versions without refused overlay
# ---------------------------------------------------------------------------


def _refuse_under_v2(env, root, monkeypatch, calls: list) -> dict:
    """Accept under PHP_V1, then refuse a code-change update under PHP_V2.

    A code change, so the next run reaches the refusal hold: an unchanged
    source with no probed upgrade is skipped before the key is consulted.
    """
    _set_builder(env, "graphify-php")
    monkeypatch.setattr("graphify_mesh.sync.pipeline.apply_action", _recording_apply(calls))
    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.66")
    _probe_returns(monkeypatch, PHP_V1)
    run(env.settings(ast_only=True, shrink_policy="attributed"))
    monkeypatch.setattr(
        "graphify_mesh.sync.pipeline.apply_action",
        _recording_apply(calls, status=STATUS_SHRINK_REFUSED),
    )
    _probe_returns(monkeypatch, PHP_V2)
    (root / "app.php").write_text("<?php\n// refused\n", encoding="utf-8")
    run(env.settings(ast_only=True, shrink_policy="attributed"))
    assert len(calls) == 2
    return _state(env)


def test_held_refusal_survives_total_probe_failure_with_another_graphify(php_repo, monkeypatch):
    """In-process graphify 0.9.66, wrapper's 0.9.89: the key must not take 0.9.66."""
    env, root = php_repo
    calls: list = []
    held = _refuse_under_v2(env, root, monkeypatch, calls)
    assert held["refused_graphify_version"] == "0.9.89"
    assert held["refused_key"].endswith(":php-update@0.9.89+graphify-php-0.2.0:attributed")

    _probe_returns(monkeypatch, None)
    report = run(env.settings(ast_only=True, shrink_policy="attributed"))
    assert len(calls) == 2, "a total probe failure released the held refusal"
    assert {a["repo_id"]: a for a in report.project_actions}[REPO_ID]["action"] == ACTION_SKIP
    assert _state(env)["refused_key"] == held["refused_key"]


def test_unchanged_repo_is_not_rebuilt_by_a_total_probe_failure(php_repo, monkeypatch):
    env, _ = php_repo
    _set_builder(env, "graphify-php")
    calls: list = []
    monkeypatch.setattr("graphify_mesh.sync.pipeline.apply_action", _recording_apply(calls))
    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.66")
    _probe_returns(monkeypatch, PHP_V1)
    run(env.settings(shrink_policy="count"))
    _probe_returns(monkeypatch, None)
    run(env.settings(shrink_policy="count"))
    _probe_returns(monkeypatch, PHP_V1)
    run(env.settings(shrink_policy="count"))
    assert len(calls) == 1
    assert _state(env)["graphify_version"] == "0.9.89"


def test_accept_during_probe_outage_records_no_refused_overlay(php_repo, monkeypatch):
    env, root = php_repo
    calls: list = []
    _refuse_under_v2(env, root, monkeypatch, calls)

    monkeypatch.setattr("graphify_mesh.sync.pipeline.apply_action", _recording_apply(calls))
    _probe_returns(monkeypatch, None)
    (root / "app.php").write_text("<?php\n// accepted\n", encoding="utf-8")
    run(env.settings(ast_only=True, shrink_policy="attributed"))
    assert len(calls) == 3
    state = _state(env)
    assert state["producer_versions"] == PHP_V1, "the refused attempt's versions were recorded"
    assert state["graphify_version"] == "0.9.89"
    assert "refused_producer_versions" not in state
    assert "refused_graphify_version" not in state
