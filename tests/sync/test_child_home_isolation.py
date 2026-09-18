"""Child-process HOME isolation and the opt-in bubblewrap sandbox.

`update` and `extract` are the two calls that parse untrusted repository
source, so they must not inherit the operator's real HOME: upstream resolves
custom LLM providers from `~/.graphify/providers.json`, and a provider's
`base_url` decides where the parsed corpus goes.

The sandbox tests assert the emitted argv, including the order of its
arguments. Order is the property, not decoration: a `--bind` that lands before
the `--tmpfs` over the same path is not a bind at all, and a `--bind` after it
undoes the mask.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from graphify_mesh.sync import graphify_cli, pipeline


@pytest.fixture
def captured(monkeypatch):
    """Replace `_run` with a recorder so no child is ever spawned."""
    calls: list[dict] = []

    def fake_run(argv, cwd, env, timeout=None):
        calls.append({"argv": argv, "cwd": cwd, "env": env})
        return graphify_cli.CliResult(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(graphify_cli, "_run", fake_run)
    return calls


@pytest.fixture(autouse=True)
def _fresh_log_once():
    """Each test gets a clean once-per-process log ledger, or only the first
    test in the file would ever see a warning."""
    graphify_cli.reset_log_once()
    yield
    graphify_cli.reset_log_once()


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Sandbox on, with a stub `bwrap` that just execs its payload."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    bwrap = fake_bin / "bwrap"
    bwrap.write_text('#!/bin/sh\nexec "$@"\n', encoding="utf-8")
    bwrap.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    monkeypatch.setenv(graphify_cli.CHILD_SANDBOX_VAR, "1")
    return bwrap


def _binds(argv: list[str], flag: str) -> list[str]:
    return [argv[i + 1] for i, tok in enumerate(argv) if tok == flag]


def test_run_update_uses_staging_home(captured, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "real-home"))
    staging_home = tmp_path / "staging" / "home"
    graphify_cli.run_update("graphify", tmp_path / "repo", tmp_path / "coll", staging_home)

    env = captured[0]["env"]
    assert env["HOME"] == str(staging_home)
    assert env["HOME"] != str(tmp_path / "real-home")
    assert staging_home.is_dir()
    assert env["PYTHONPATH"]


def test_run_extract_uses_staging_home(captured, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "real-home"))
    staging_home = tmp_path / "staging" / "home"
    graphify_cli.run_extract("graphify", tmp_path / "repo", tmp_path / "coll", staging_home)

    env = captured[0]["env"]
    assert env["HOME"] == str(staging_home)
    assert env["HOME"] != str(tmp_path / "real-home")
    assert staging_home.is_dir()


def test_merge_cluster_label_isolation_unchanged(captured, tmp_path):
    staging_home = tmp_path / "merge-home"
    graphify_cli.run_merge_graphs(
        "graphify", [tmp_path / "a.json"], tmp_path / "out.json", staging_home
    )
    graphify_cli.run_cluster_only("graphify", tmp_path, staging_home)
    graphify_cli.run_label("graphify", tmp_path, staging_home, "ollama", "m")

    assert [c["env"]["HOME"] for c in captured] == [str(staging_home)] * 3


def test_project_staging_home_is_unique_per_repo(tmp_path):
    a = pipeline.project_staging_home(tmp_path, "example-org.aaa")
    b = pipeline.project_staging_home(tmp_path, "example-org.zzz")
    assert a != b
    # Sanitisation alone would map these two onto the same directory name; the
    # repo_id hash is what keeps them apart.
    assert pipeline.project_staging_home(tmp_path, "a.b") != pipeline.project_staging_home(
        tmp_path, "a-b"
    )
    assert pipeline.project_staging_home(tmp_path, "a/../b").parent == tmp_path / "project-homes"


def test_project_staging_home_is_stable(tmp_path):
    assert pipeline.project_staging_home(tmp_path, "x.y") == pipeline.project_staging_home(
        tmp_path, "x.y"
    )


def test_pipeline_gives_concurrent_repos_distinct_homes(env, monkeypatch):
    """The pipeline, not the test, must pick a different HOME per repo.

    Driven through `pipeline.run` so the staging home under assertion is the one
    the run computed. The previous version of this test called
    `project_staging_home` itself in a loop and passed with every repo sharing
    one directory.
    """
    env.add_repo("example-org.aaa", "example-org", "aaa", "aaa.example-org.dev.lo")
    env.add_repo("example-org.zzz", "example-org", "zzz", "zzz.example-org.dev.lo")
    env.write_registry()
    settings = env.settings()

    seen: dict[str, Path] = {}
    real_apply = pipeline.apply_action

    def recording_apply(
        repo_id, graphify_bin, root, collection_path, action, manifest, staging_home, **kw
    ):
        seen[repo_id] = Path(staging_home)
        return real_apply(
            repo_id, graphify_bin, root, collection_path, action, manifest, staging_home, **kw
        )

    monkeypatch.setattr(pipeline, "apply_action", recording_apply)
    pipeline.run(settings)

    assert set(seen) == {"example-org.aaa", "example-org.zzz"}
    assert len(set(seen.values())) == 2, "two repos shared one staging HOME"
    for home in seen.values():
        assert home.parent.name == "project-homes"


def test_sandbox_off_leaves_argv_unchanged(captured, tmp_path, monkeypatch):
    monkeypatch.delenv(graphify_cli.CHILD_SANDBOX_VAR, raising=False)
    graphify_cli.run_update("graphify", tmp_path, tmp_path / "coll", tmp_path / "home")
    assert captured[0]["argv"] == ["graphify", "update", str(tmp_path)]


def test_sandbox_on_wraps_argv(captured, sandbox, tmp_path):
    staging_home = tmp_path / "home"
    staging_home.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    collection = tmp_path / "mesh" / "coll"
    collection.mkdir(parents=True)
    graphify_cli.run_extract("graphify", repo, collection, staging_home)

    argv = captured[0]["argv"]
    assert argv[0] == str(sandbox)
    assert argv[1:4] == ["--ro-bind", "/", "/"]
    assert "--proc" in argv
    writable = set(_binds(argv, "--bind"))
    assert {str(repo), str(collection), str(staging_home)} <= writable
    # The real command still follows the wrapper unchanged.
    assert argv[argv.index("graphify") :] == [
        "graphify",
        "extract",
        str(repo),
        "--backend",
        "ollama",
        "--force",
        "--max-concurrency",
        "1",
    ]


def test_tmpfs_home_precedes_every_writable_bind(captured, sandbox, tmp_path):
    """The isolation is the ordering. A `--bind` before the `--tmpfs` over the
    home directory is erased by it; the masks must all come first."""
    repo = tmp_path / "repo"
    repo.mkdir()
    graphify_cli.run_update("graphify", repo, tmp_path / "coll", tmp_path / "home")

    argv = captured[0]["argv"]
    last_tmpfs = max(i for i, tok in enumerate(argv) if tok == "--tmpfs")
    first_bind = min(i for i, tok in enumerate(argv) if tok == "--bind")
    assert argv.index("--tmpfs") < first_bind
    assert last_tmpfs < first_bind
    assert argv.index("--ro-bind") == 1, "the read-only root must be the first mount"
    assert argv[argv.index("--tmpfs") + 1] == str(Path.home().resolve())


def test_sandbox_on_without_bwrap_fails_closed(captured, tmp_path, monkeypatch):
    monkeypatch.setenv(graphify_cli.CHILD_SANDBOX_VAR, "yes")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(graphify_cli.ChildSandboxUnavailable):
        graphify_cli.run_update("graphify", tmp_path, tmp_path / "coll", tmp_path / "home")
    assert captured == []


# --- 1. the scanned repository must not steer a writable bind ----------------


def test_graphify_out_symlink_cannot_steer_a_writable_bind(captured, sandbox, tmp_path):
    """A `graphify-out` symlink pointing at the operator's home used to produce
    `--tmpfs $HOME ... --bind $HOME $HOME`, handing the child back the very
    directory the tmpfs masks. The bind target now comes from the registry."""
    repo = tmp_path / "repo"
    repo.mkdir()
    victim = tmp_path / "victim-home"
    (victim / ".ssh").mkdir(parents=True)
    (repo / "graphify-out").symlink_to(victim, target_is_directory=True)
    collection = tmp_path / "mesh" / "coll"
    collection.mkdir(parents=True)

    graphify_cli.run_extract("graphify", repo, collection, tmp_path / "home")

    argv = captured[0]["argv"]
    assert str(victim) not in argv
    assert str(victim) not in set(_binds(argv, "--bind"))
    assert str(collection) in set(_binds(argv, "--bind"))


def test_write_path_outside_approved_roots_raises(captured, sandbox, tmp_path):
    approved = tmp_path / "approved"
    approved.mkdir()
    repo = approved / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    policy = graphify_cli.SandboxPolicy(containment_roots=(approved,))

    with pytest.raises(graphify_cli.SandboxContainmentError) as excinfo:
        graphify_cli.run_extract("graphify", repo, outside, approved / "home", policy=policy)

    assert str(outside) in str(excinfo.value)
    assert captured == []


def test_containment_rejects_a_symlinked_collection_path(captured, sandbox, tmp_path):
    """Containment is applied to the RESOLVED path, so a contained symlink
    pointing out of the approved roots is refused, not followed."""
    approved = tmp_path / "approved"
    approved.mkdir()
    repo = approved / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sneaky = approved / "coll"
    sneaky.symlink_to(outside, target_is_directory=True)
    policy = graphify_cli.SandboxPolicy(containment_roots=(approved,))

    with pytest.raises(graphify_cli.SandboxContainmentError):
        graphify_cli.run_extract("graphify", repo, sneaky, approved / "home", policy=policy)


def test_containment_raises_for_a_missing_path_too(captured, sandbox, tmp_path):
    """An escaping path must raise, not be dropped for not existing yet."""
    approved = tmp_path / "approved"
    approved.mkdir()
    policy = graphify_cli.SandboxPolicy(containment_roots=(approved,))
    with pytest.raises(graphify_cli.SandboxContainmentError):
        graphify_cli.run_extract(
            "graphify",
            approved / "repo",
            tmp_path / "gone" / "coll",
            approved / "home",
            policy=policy,
        )


def test_pipeline_policy_covers_approved_roots_mesh_root_and_staging(env, tmp_path):
    env.add_repo("example-org.aaa", "example-org", "aaa", "aaa.example-org.dev.lo")
    env.write_registry()
    settings = env.settings()

    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    policy = pipeline.child_sandbox_policy(settings, staging_root)

    assert env.scan_root.resolve() in policy.containment_roots
    assert settings.mesh_root.resolve() in policy.containment_roots
    # The staging root is a mkdtemp under TMPDIR, under none of the configured
    # roots. Leaving it out refuses the engine's own staging HOME.
    assert staging_root.resolve() in policy.containment_roots
    assert policy.masked_paths == (settings.mesh_root / "bin",)


def test_missing_write_path_is_debug_logged(sandbox, tmp_path, caplog):
    absent = tmp_path / "not-there"
    with caplog.at_level(logging.DEBUG, logger="graphify_mesh.sync"):
        graphify_cli._sandbox_argv(["/bin/true"], [absent], None)
    assert any("does not exist yet" in r.getMessage() for r in caplog.records)


# --- 2. the tmpfs home must also remove agent/socket access ------------------


def test_runtime_dir_is_masked(captured, sandbox, tmp_path, monkeypatch):
    """`--ro-bind / /` leaves /run/user/<uid> reachable, and read-only does not
    stop `connect()` on the ssh-agent or D-Bus socket underneath it."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    repo = tmp_path / "repo"
    repo.mkdir()

    graphify_cli.run_extract("graphify", repo, tmp_path / "coll", tmp_path / "home")

    assert str(runtime) in _binds(captured[0]["argv"], "--tmpfs")


def test_runtime_dir_prefers_xdg_then_falls_back_to_run_user_uid(monkeypatch, tmp_path):
    runtime = tmp_path / "rt"
    runtime.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    assert graphify_cli._runtime_dir() == runtime

    # Unset is the systemd case: the variable is absent but /run/user/<uid>
    # exists anyway. A uid with no runtime directory yields None rather than a
    # bind on a path that is not there.
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    monkeypatch.setattr(graphify_cli.os, "getuid", lambda: 999999)
    assert graphify_cli._runtime_dir() is None


def test_missing_runtime_dir_is_not_masked(captured, sandbox, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "no-such-runtime"))
    graphify_cli.run_update("graphify", tmp_path, tmp_path / "coll", tmp_path / "home")
    masked = _binds(captured[0]["argv"], "--tmpfs")
    assert str(tmp_path / "no-such-runtime") not in masked


# --- 3. the env allowlist must not be bypassable from disk -------------------


def test_policy_masked_paths_are_tmpfs(captured, sandbox, tmp_path):
    mesh_bin = tmp_path / "mesh" / "bin"
    mesh_bin.mkdir(parents=True)
    (mesh_bin / "graphify-mesh-sync.env").write_text("GRAPHIFY_MESH_HTTP_TOKEN=s\n")
    policy = graphify_cli.SandboxPolicy(masked_paths=(mesh_bin,))

    graphify_cli.run_extract(
        "graphify", tmp_path, tmp_path / "coll", tmp_path / "home", policy=policy
    )

    assert str(mesh_bin) in _binds(captured[0]["argv"], "--tmpfs")


def test_operator_can_extend_the_masked_set(captured, sandbox, tmp_path, monkeypatch):
    extra = tmp_path / "secrets"
    extra.mkdir()
    monkeypatch.setenv(graphify_cli.CHILD_MASK_PATHS_VAR, f"{extra},  ")

    graphify_cli.run_update("graphify", tmp_path, tmp_path / "coll", tmp_path / "home")

    assert str(extra) in _binds(captured[0]["argv"], "--tmpfs")


def test_relative_mask_entry_is_refused(captured, sandbox, tmp_path, monkeypatch):
    monkeypatch.setenv(graphify_cli.CHILD_MASK_PATHS_VAR, "relative/path")
    graphify_cli.run_update("graphify", tmp_path, tmp_path / "coll", tmp_path / "home")
    assert "relative/path" not in _binds(captured[0]["argv"], "--tmpfs")


# --- 4. process-namespace and device gaps ------------------------------------


def test_sandbox_unshares_pid_and_dies_with_parent(captured, sandbox, tmp_path):
    graphify_cli.run_update("graphify", tmp_path, tmp_path / "coll", tmp_path / "home")
    argv = captured[0]["argv"]
    for flag in ("--unshare-pid", "--die-with-parent", "--new-session"):
        assert flag in argv, flag


def test_sandbox_uses_a_private_dev(captured, sandbox, tmp_path):
    """The real /dev hands over a `/dev/shm` shared with every process of this
    UID. `--dev` gives the child its own."""
    graphify_cli.run_update("graphify", tmp_path, tmp_path / "coll", tmp_path / "home")
    argv = captured[0]["argv"]
    assert "--dev-bind" not in argv
    assert argv[argv.index("--dev") + 1] == "/dev"


# --- 5. the sandbox switch must not fail open --------------------------------


@pytest.mark.parametrize("value", ["1", "on", "true", "yes", "Y", "Enabled", "TRUE"])
def test_recognised_and_unrecognised_truthy_values_enable_the_sandbox(value, monkeypatch):
    monkeypatch.setenv(graphify_cli.CHILD_SANDBOX_VAR, value)
    assert graphify_cli._child_sandbox_enabled() is True


@pytest.mark.parametrize("value", ["0", "off", "false", "no", "", "  "])
def test_recognised_falsy_values_disable_the_sandbox(value, monkeypatch):
    monkeypatch.setenv(graphify_cli.CHILD_SANDBOX_VAR, value)
    assert graphify_cli._child_sandbox_enabled() is False


def test_unset_disables_the_sandbox(monkeypatch):
    monkeypatch.delenv(graphify_cli.CHILD_SANDBOX_VAR, raising=False)
    assert graphify_cli._child_sandbox_enabled() is False


def test_unparsable_value_warns_and_fails_closed(monkeypatch, caplog):
    monkeypatch.setenv(graphify_cli.CHILD_SANDBOX_VAR, "2")
    with caplog.at_level(logging.WARNING, logger="graphify_mesh.sync"):
        assert graphify_cli._child_sandbox_enabled() is True
    assert any("not a recognised on/off value" in r.message for r in caplog.records)


# --- 6. the pipx layout must survive the tmpfs -------------------------------


def _pipx_layout(home: Path) -> Path:
    """`pipx install graphifyy` as docs/setup.md recommends it: a console
    script in the venv's bin/, reached through a symlink in ~/.local/bin, with
    the package under the venv's lib/."""
    venv = home / ".local" / "pipx" / "venvs" / "graphifyy"
    (venv / "bin").mkdir(parents=True)
    (venv / "lib" / "python3.12" / "site-packages" / "graphify").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\nversion = 3.12.0\n", encoding="utf-8")
    script = venv / "bin" / "graphify"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(0o755)
    shim_dir = home / ".local" / "bin"
    shim_dir.mkdir(parents=True)
    shim = shim_dir / "graphify"
    shim.symlink_to(script)
    return shim


def test_pipx_venv_root_is_rebound_readonly(captured, sandbox, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    shim = _pipx_layout(home)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", f"{shim.parent}:{os.environ['PATH']}")

    graphify_cli.run_update(str(shim), tmp_path, tmp_path / "coll", tmp_path / "staging")

    readable = _binds(captured[0]["argv"], "--ro-bind")
    venv = home / ".local" / "pipx" / "venvs" / "graphifyy"
    assert str(venv) in readable, "the venv root carries lib/ and pyvenv.cfg"
    assert str(shim.parent) in readable, "argv carries the symlink path, so bind it too"


def test_venv_root_detection_stops_at_pyvenv_cfg(tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("", encoding="utf-8")
    assert graphify_cli._venv_root(venv / "bin" / "graphify") == venv
    assert graphify_cli._venv_root(tmp_path / "plain" / "bin" / "graphify") is None


# --- 7. diagnostics fire once per process, not once per child ----------------


def test_dropped_backend_variables_are_logged_once(tmp_path, caplog, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.delenv(graphify_cli.CHILD_SANDBOX_VAR, raising=False)
    with caplog.at_level(logging.INFO, logger="graphify_mesh.sync"):
        for _ in range(3):
            graphify_cli._run(["/bin/true"], cwd=None, env=None)
    hits = [r for r in caplog.records if "backend variables set here" in r.msg]
    assert len(hits) == 1


def test_child_env_extra_rejection_is_logged_once(caplog, monkeypatch):
    monkeypatch.setenv(graphify_cli.CHILD_ENV_EXTRA_VAR, "GRAPHIFY_MESH_HTTP_TOKEN")
    with caplog.at_level(logging.WARNING, logger="graphify_mesh.sync"):
        for _ in range(3):
            graphify_cli._child_env_extra()
    hits = [r for r in caplog.records if "refusing to pass them to the child" in r.msg]
    assert len(hits) == 1


def test_a_real_sandboxed_run_publishes(env, monkeypatch, tmp_path):
    """The whole pipeline, sandbox on, through a `bwrap` stub that execs its
    payload.

    This is the check that catches a containment rule which refuses the engine's
    own paths: the per-repo staging HOME lives under a `mkdtemp` staging root
    that is under no configured root, so a policy built from the approved roots
    and the mesh root alone raises `SandboxContainmentError` on every launch,
    and nothing in the argv-level tests would notice.
    """
    fake_bin = tmp_path / "sandbox-bin"
    fake_bin.mkdir()
    bwrap = fake_bin / "bwrap"
    # Consume bwrap's own options up to the first non-option argument, then exec
    # the rest: the stub has to behave like bwrap enough to run the payload.
    bwrap.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "argv = sys.argv[1:]\n"
        "takes_two = {'--ro-bind', '--bind', '--dev-bind'}\n"
        "takes_one = {'--tmpfs', '--dev', '--proc'}\n"
        "i = 0\n"
        "while i < len(argv) and argv[i].startswith('--'):\n"
        "    if argv[i] in takes_two:\n"
        "        i += 3\n"
        "    elif argv[i] in takes_one:\n"
        "        i += 2\n"
        "    else:\n"
        "        i += 1\n"
        "os.execvp(argv[i], argv[i:])\n",
        encoding="utf-8",
    )
    bwrap.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    monkeypatch.setenv(graphify_cli.CHILD_SANDBOX_VAR, "1")

    env.add_repo("example-org.aaa", "example-org", "aaa", "aaa.example-org.dev.lo")
    env.write_registry()
    report = pipeline.run(env.settings())

    assert report.published, report.errors
    rows = {row["repo_id"]: row["status"] for row in report.project_actions}
    assert rows["example-org.aaa"] in {"updated", "unchanged", "noop"}, rows
