"""`graphify update --force` after graphify 0.9.102 made `update` incremental.

0.9.102 re-extracts only files whose content hash differs from manifest.json,
and that hash does not cover the extractor version. A producer upgrade on an
unchanged source therefore changed nothing: graph byte-identical, NOOP, new
version recorded, upgrade never applied. `--force` is graphify's full rebuild.
It also waives graphify's own shrink guard, whose refusal (exit 1, "Refusing
to overwrite —") is now held like the mesh's own and released the same way.
"""

from __future__ import annotations

import json
import logging
import re
import stat
import sys
from pathlib import Path

import pytest

from graphify_mesh.sync import clustering, graphify_cli, repo_tags, sync_project
from graphify_mesh.sync.pipeline import _producer_changed, run
from graphify_mesh.sync.state import SourceDigest, load_state, save_state
from graphify_mesh.sync.sync_project import (
    ACTION_SKIP,
    ACTION_UPDATE,
    STATUS_FAILED,
    STATUS_SHRINK_REFUSED,
    STATUS_UPDATED,
    apply_action,
)

FAKE_GRAPHIFY = Path(__file__).resolve().parents[1] / "fixtures" / "fake_graphify" / "graphify"
REPO_ID = "example-org.svc"
TOKEN_RE = re.compile(r'"allow_shrink_once": "([0-9a-f]+)"')


@pytest.fixture
def repo(env, monkeypatch):
    root = env.add_repo(REPO_ID, "example-org", "svc", "svc.example-org.dev.lo", "repo_a.json")
    env.write_registry()
    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.89")
    return env, root, env.collection_path("example-org", "svc")


def _settings(env, **overrides):
    return env.settings(shrink_policy="count", **overrides)


def _updates(env) -> list[dict]:
    return [r for r in env.read_argv_log() if r["cmd"] == "update"]


def _row(report) -> dict:
    return {a["repo_id"]: a for a in report.project_actions}[REPO_ID]


def _node_count(collection: Path) -> int:
    return len(json.loads((collection / "graph.json").read_text())["nodes"])


def _set_registry(env, **fields) -> None:
    payload = json.loads(env.registry_path.read_text())
    payload["repos"][0].update(fields)
    env.registry_path.write_text(json.dumps(payload), encoding="utf-8")


def test_producer_upgrade_forces_a_full_update(repo, monkeypatch):
    env, _root, collection = repo
    env.set_control(collection, "incremental")
    run(_settings(env))
    assert "--force" not in _updates(env)[-1]["argv"], "no prior producer: nothing to force"
    before = _node_count(collection)

    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.103")
    report = run(_settings(env))

    assert "--force" in _updates(env)[-1]["argv"]
    assert _row(report)["status"] == STATUS_UPDATED
    assert _node_count(collection) == before + 1, "the upgrade must reach unchanged files"


def test_ordinary_code_change_updates_without_force(repo):
    env, root, collection = repo
    run(_settings(env))
    (root / "app.php").write_text("<?php\n", encoding="utf-8")

    report = run(_settings(env))

    assert _row(report)["action"] == ACTION_UPDATE
    assert "--force" not in _updates(env)[-1]["argv"]
    assert len(_updates(env)) == 2


def _state(env) -> dict:
    return load_state(_settings(env).state_path)[REPO_ID]


def _rewrite_state(env, **fields) -> None:
    """Set (or, with None, drop) fields of this repo's accepted state."""
    path = _settings(env).state_path
    state = load_state(path)
    for key, value in fields.items():
        if value is None:
            state[REPO_ID].pop(key, None)
        else:
            state[REPO_ID][key] = value
    save_state(path, state)


def test_producer_label_switch_alone_does_not_force(repo):
    """Accepted by extract, now a code edit: an ordinary update. `--force` here
    disabled graphify's own provenance and semantic shrink guards."""
    env, root, _collection = repo
    run(_settings(env))
    _rewrite_state(env, producer="extract")
    assert _producer_changed(_state(env), "update", "0.9.89"), "precondition: label differs"
    (root / "app.php").write_text("<?php\n", encoding="utf-8")

    report = run(_settings(env))

    assert _row(report)["action"] == ACTION_UPDATE
    assert "--force" not in _updates(env)[-1]["argv"]


def test_unrecorded_graphify_version_forces_once(repo):
    env, _root, _collection = repo
    run(_settings(env))
    _rewrite_state(env, graphify_version=None)

    report = run(_settings(env))

    assert _row(report)["action"] == ACTION_UPDATE, "unchanged source still rebuilds once"
    assert "--force" in _updates(env)[-1]["argv"]
    assert _state(env)["graphify_version"] == "0.9.89"
    calls = len(_updates(env))
    assert _row(run(_settings(env)))["action"] == ACTION_SKIP
    assert len(_updates(env)) == calls


def test_php_invoker_gets_force_after_the_separator_on_producer_change(repo, monkeypatch, tmp_path):
    env, root, collection = repo
    env.set_control(collection, "incremental")
    _set_registry(env, builder="graphify-php")
    wrapper_log = tmp_path / "php-build-argv.jsonl"
    # Stands in for graphify-php-build: records its argv and hands what follows
    # `--` to the fake graphify, as the real wrapper passes it through.
    wrapper = tmp_path / "graphify-php-build"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"with open({str(wrapper_log)!r}, 'a') as fh:\n"
        "    fh.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "rest = sys.argv[sys.argv.index('--') + 1:]\n"
        f"os.execv({str(FAKE_GRAPHIFY)!r}, [{str(FAKE_GRAPHIFY)!r}, *rest])\n",
        encoding="utf-8",
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    versions = {"graphify_php": "1.0.0", "graphify": "0.9.89"}
    monkeypatch.setattr(graphify_cli, "probe_php_build_version", lambda _bin: dict(versions))
    run(_settings(env, php_build_bin=str(wrapper)))
    before = _node_count(collection)

    versions["graphify"] = "0.9.103"
    report = run(_settings(env, php_build_bin=str(wrapper)))

    argv = json.loads(wrapper_log.read_text().splitlines()[-1])
    assert argv[argv.index("--") + 1 :] == ["update", str(root), "--force"]
    assert _row(report)["status"] == STATUS_UPDATED
    assert _node_count(collection) == before + 1


def test_php_wrapper_exit_4_is_a_failed_wiring_fault_not_a_hold(repo, monkeypatch, tmp_path):
    env, root, collection = repo
    _set_registry(env, builder="graphify-php")
    wrapper = tmp_path / "graphify-php-build"
    wrapper.write_text(
        f"#!{sys.executable}\nimport sys\nsys.stderr.write('shrink guard refused')\nsys.exit(4)\n",
        encoding="utf-8",
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(
        graphify_cli,
        "probe_php_build_version",
        lambda _bin: {"graphify_php": "1.0.0", "graphify": "0.9.89"},
    )
    run(_settings(env))
    before = _node_count(collection)
    (root / "app.php").write_text("<?php\n", encoding="utf-8")

    row = _row(run(_settings(env, php_build_bin=str(wrapper))))

    assert row["status"] == STATUS_FAILED
    assert "wiring fault" in row["reason"]
    assert "GRAPHIFY_MESH_MANAGED" in row["reason"]
    assert _node_count(collection) == before


def _refused_by_graphify(env, root, collection):
    """Accepted state first, then a code change graphify's own guard refuses."""
    run(_settings(env))
    env.set_control(collection, "graphify_refuse")
    (root / "app.php").write_text("<?php\n", encoding="utf-8")
    return run(_settings(env))


def test_graphify_refusal_is_a_shrink_refusal_and_is_held(repo):
    env, root, collection = repo
    report = _refused_by_graphify(env, root, collection)

    assert _row(report)["status"] == STATUS_SHRINK_REFUSED
    assert TOKEN_RE.search(_row(report)["reason"]), _row(report)["reason"]
    report = run(_settings(env))
    assert _row(report)["action"] == ACTION_SKIP, "the identical refused update must be held"
    assert len(_updates(env)) == 2


def test_allow_shrink_releases_graphify_refusal_with_force(repo):
    env, root, collection = repo
    _refused_by_graphify(env, root, collection)
    before = _node_count(collection)

    report = run(_settings(env, allow_shrink=True))

    assert "--force" in _updates(env)[-1]["argv"]
    assert _row(report)["status"] == STATUS_UPDATED
    assert _node_count(collection) == before - 1


def test_grant_releases_graphify_refusal_with_force(repo):
    env, root, collection = repo
    report = _refused_by_graphify(env, root, collection)
    _set_registry(env, allow_shrink_once=TOKEN_RE.search(_row(report)["reason"]).group(1))

    report = run(_settings(env))

    assert "--force" in _updates(env)[-1]["argv"]
    assert _row(report)["status"] == STATUS_UPDATED


def test_allow_shrink_forces_only_the_repo_whose_refusal_it_releases(repo):
    env, root, collection = repo
    other_root = env.add_repo(
        "example-org.other", "example-org", "other", "other.example-org.dev.lo", "repo_b.json"
    )
    env.write_registry()
    _refused_by_graphify(env, root, collection)
    (other_root / "app.php").write_text("<?php\n", encoding="utf-8")
    seen = len(_updates(env))

    run(_settings(env, allow_shrink=True))

    force_by_root = {
        arg: "--force" in r["argv"]
        for r in _updates(env)[seen:]
        for arg in r["argv"]
        if arg in (str(root), str(other_root))
    }
    assert force_by_root == {str(root): True, str(other_root): False}


def test_allow_shrink_alone_does_not_force_an_ordinary_update(repo):
    env, root, _collection = repo
    run(_settings(env))
    (root / "app.php").write_text("<?php\n", encoding="utf-8")

    report = run(_settings(env, allow_shrink=True))

    assert _row(report)["action"] == ACTION_UPDATE
    assert "--force" not in _updates(env)[-1]["argv"], "no refusal was held: nothing to release"


def test_allow_shrink_does_not_force_a_refusal_below_the_hold_limit(repo):
    """A matching refused_key with a streak under the limit is not held, so the
    update runs anyway and --allow-shrink has nothing to release."""
    env, root, collection = repo
    _refused_by_graphify(env, root, collection)
    _rewrite_state(env, refusal_streak=0)

    run(_settings(env, allow_shrink=True))

    assert "--force" not in _updates(env)[-1]["argv"]


def test_grant_is_spent_by_an_accepted_forced_update_without_shrink(repo):
    """Under `--force` graphify's guard never refuses; the mesh guard can accept
    with no shrink. The grant must still be spent, or it authorizes a later shrink."""
    env, root, collection = repo
    report = _refused_by_graphify(env, root, collection)
    match = TOKEN_RE.search(_row(report)["reason"])
    assert match is not None
    token = match.group(1)
    _set_registry(env, allow_shrink_once=token)
    env.set_control(collection, "normal")
    before = _node_count(collection)

    report = run(_settings(env))

    assert "--force" in _updates(env)[-1]["argv"]
    assert _row(report)["status"] == STATUS_UPDATED
    assert _node_count(collection) == before + 1, "accepted without a shrink"
    assert _state(env)["consumed_shrink_grant"] == token


def _collection(tmp_path: Path) -> Path:
    collection = tmp_path / "coll"
    collection.mkdir()
    (collection / "graph.json").write_text(
        json.dumps({"nodes": [{"id": f"n{i}"} for i in range(10)], "links": []}),
        encoding="utf-8",
    )
    return collection


@pytest.mark.parametrize(
    ("stderr", "status"),
    [
        (
            "[graphify] WARNING: Refusing to overwrite — saved semantic "
            "node(s) are missing from the new graph: n3.",
            STATUS_SHRINK_REFUSED,
        ),
        ("Refusing to overwrite; delete the file and run a", STATUS_FAILED),
        ("Traceback: boom", STATUS_FAILED),
    ],
)
def test_only_graphify_shrink_refusal_is_classified_as_refused(
    tmp_path, monkeypatch, stderr, status
):
    collection = _collection(tmp_path)
    old = (collection / "graph.json").read_bytes()
    monkeypatch.setitem(
        sync_project._INVOKERS,
        ACTION_UPDATE,
        lambda *_a, **_kw: graphify_cli.CliResult(1, "", stderr),
    )
    digest = SourceDigest(code_hash="c1", semantic_hash="s1", file_count=1)

    outcome = apply_action(
        REPO_ID, "graphify", tmp_path, collection, ACTION_UPDATE, digest, tmp_path / "home"
    )

    assert outcome.status == status
    assert (outcome.refused_manifest is not None) == (status == STATUS_SHRINK_REFUSED)
    assert (collection / "graph.json").read_bytes() == old


@pytest.fixture
def captured(monkeypatch):
    monkeypatch.delenv(graphify_cli.CHILD_SANDBOX_VAR, raising=False)
    seen: list[list[str]] = []

    def fake_run(argv, cwd, env, timeout=None):
        seen.append(argv)
        return graphify_cli.CliResult(0, "", "")

    monkeypatch.setattr(graphify_cli, "_run", fake_run)
    return seen


def test_run_update_appends_force_only_when_asked(captured, tmp_path):
    graphify_cli.run_update("graphify", tmp_path, tmp_path / "coll", tmp_path / "home")
    graphify_cli.run_update("graphify", tmp_path, tmp_path / "coll", tmp_path / "home", force=True)
    assert captured[0][-2:] == ["update", str(tmp_path)]
    assert captured[1][-3:] == ["update", str(tmp_path), "--force"]


def test_run_php_update_passes_force_after_the_separator(captured, tmp_path):
    graphify_cli.run_php_update(
        "graphify", tmp_path, tmp_path / "coll", tmp_path / "home", force=True
    )
    argv = captured[0]
    assert argv[argv.index("--") :] == ["--", "update", str(tmp_path), "--force"]


def test_graphify_force_never_reaches_the_child(repo, monkeypatch):
    env, _root, _collection = repo
    monkeypatch.setenv("GRAPHIFY_FORCE", "1")
    run(_settings(env))
    assert _updates(env), "precondition: the fake was invoked"
    assert all(r["graphify_force_env"] is None for r in _updates(env))


def test_graphify_force_is_denied_even_when_declared_extra():
    assert not graphify_cli._child_env_allows("GRAPHIFY_FORCE", frozenset({"GRAPHIFY_FORCE"}))


def test_dry_run_makes_no_labeling_call(repo, monkeypatch, caplog):
    env, _root, _collection = repo
    calls: list[int] = []

    def recording_llm_names(G, communities, *, backend, model, max_concurrency=1):
        calls.append(len(communities))
        return {cid: f"Community Name {cid}" for cid in communities}, "llm"

    monkeypatch.setattr(clustering, "llm_names", recording_llm_names)
    caplog.set_level(logging.INFO, logger="graphify_mesh.sync")

    run(_settings(env, dry_run=True, ollama_health_check=lambda *_a: True))

    assert calls == []
    assert re.search(r"would label [1-9]\d* communities", caplog.text), caplog.text
    run(_settings(env, ollama_health_check=lambda *_a: True))
    assert calls, "control: a real run with a healthy backend does reach the labeler"


def test_dry_run_reports_no_outage_and_probes_no_backend(repo, caplog):
    env, _root, _collection = repo
    probes: list[str] = []

    def recording_health_check(base_url, *_a):
        probes.append(base_url)
        return True

    caplog.set_level(logging.INFO, logger="graphify_mesh.sync")
    report = run(_settings(env, dry_run=True, ollama_health_check=recording_health_check))

    assert probes == []
    assert "ollama unhealthy" not in caplog.text
    assert "ollama unreachable" not in caplog.text
    assert "unhealthy warning is expected" not in caplog.text
    assert "naming: dry run — labeling skipped" in caplog.text
    assert report.labeling_reason.startswith("dry run")
