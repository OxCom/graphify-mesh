"""An extract that leaves an edited document unstamped does not advance state.

graphify 0.9.103 keeps a truncated or hollow LLM result out of the manifest:
the document's `semantic_hash` is cleared to "" (cli.py `_stamped_manifest_files`
with `partial_source_files`, detect.py `save_manifest(clear_semantic=...)`) and
the command still exits 0. The mesh used to accept that run and advance its
semantic hash, so the edit stayed unindexed until the next document change.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from graphify_mesh.sync import graphify_cli, pipeline, sync_project
from graphify_mesh.sync.pipeline import run
from graphify_mesh.sync.state import (
    CONTENT_DIGEST_ENV,
    SourceDigest,
    compute_source_manifest,
    load_state,
)
from graphify_mesh.sync.sync_project import ProjectOutcome

DIGEST = SourceDigest(code_hash="c2", semantic_hash="s2", file_count=3)


def _row(ast_hash: str, semantic_hash: str) -> dict:
    return {"mtime": 1.0, "seen": 1.0, "ast_hash": ast_hash, "semantic_hash": semantic_hash}


def _setup(tmp_path: Path, before: dict) -> Path:
    collection = tmp_path / "coll"
    collection.mkdir()
    graph = {"nodes": [{"id": "a"}], "links": []}
    (collection / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (collection / "manifest.json").write_text(json.dumps(before), encoding="utf-8")
    return collection


def _invoker(after: dict):
    def invoke(_bin, _root, collection_path, _home, **_kwargs):
        graph = {"nodes": [{"id": "a"}, {"id": "b"}], "links": []}
        (collection_path / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
        (collection_path / "manifest.json").write_text(json.dumps(after), encoding="utf-8")
        return graphify_cli.CliResult(returncode=0, stdout="", stderr="")

    return invoke


def _run(
    tmp_path,
    monkeypatch,
    before,
    after,
    action=sync_project.ACTION_EXTRACT,
    digest=DIGEST,
    retried=frozenset(),
):
    collection = _setup(tmp_path, before)
    monkeypatch.setitem(sync_project._INVOKERS, action, _invoker(after))
    outcome = sync_project.apply_action(
        "example-org.gw",
        "graphify",
        tmp_path / "repo",
        collection,
        action,
        digest,
        tmp_path / "home",
        retried_documents=retried,
    )
    return outcome, collection


def test_newly_unstamped_document_holds_state_and_keeps_graph(tmp_path, monkeypatch, caplog):
    before = {"README.md": _row("h1", "h1"), "src/a.ts": _row("x", "x")}
    after = {"README.md": _row("h1", ""), "src/a.ts": _row("x", "x")}
    with caplog.at_level(logging.WARNING, logger="graphify_mesh.sync"):
        outcome, collection = _run(tmp_path, monkeypatch, before, after)

    assert outcome.status == sync_project.STATUS_UPDATED
    assert outcome.new_manifest is None
    assert outcome.refused_manifest is None
    assert "README.md" in outcome.reason
    assert outcome.graph_content_hash is not None
    graph = json.loads((collection / "graph.json").read_text(encoding="utf-8"))
    assert len(graph["nodes"]) == 2
    manifest = json.loads((collection / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["README.md"]["semantic_hash"] == ""
    assert any("README.md" in r.getMessage() for r in caplog.records)


def test_retry_that_stays_unstamped_is_accepted_with_warning(tmp_path, monkeypatch, caplog):
    before = {"README.md": _row("h1", "")}
    after = {"README.md": _row("h1", "")}
    with caplog.at_level(logging.WARNING, logger="graphify_mesh.sync"):
        outcome, _ = _run(tmp_path, monkeypatch, before, after)

    assert outcome.new_manifest == DIGEST
    assert "README.md" in outcome.reason
    assert "still unstamped" in outcome.reason
    assert any("still unstamped" in r.getMessage() for r in caplog.records)


def test_fully_stamped_extract_is_accepted_silently(tmp_path, monkeypatch):
    before = {"README.md": _row("h1", "h1")}
    after = {"README.md": _row("h2", "h2")}
    outcome, _ = _run(tmp_path, monkeypatch, before, after)

    assert outcome.new_manifest == DIGEST
    assert outcome.reason == ""


def test_code_rows_never_hold(tmp_path, monkeypatch):
    before = {"src/a.php": _row("x", "x"), "conf/b.yaml": _row("y", "y")}
    after = {"src/a.php": _row("x2", ""), "conf/b.yaml": _row("", "")}
    outcome, _ = _run(tmp_path, monkeypatch, before, after)

    assert outcome.new_manifest == DIGEST
    assert outcome.reason == ""


def test_ast_update_clearing_a_document_hash_is_not_a_partial(tmp_path, monkeypatch):
    # `graphify update` writes kind="ast" and drops semantic_hash of an edited doc.
    before = {"README.md": _row("h1", "h1")}
    after = {"README.md": _row("h2", "")}
    outcome, _ = _run(tmp_path, monkeypatch, before, after, action=sync_project.ACTION_UPDATE)

    assert outcome.new_manifest == DIGEST


def test_unreadable_manifest_falls_back_to_accept(tmp_path, monkeypatch):
    collection = _setup(tmp_path, {"README.md": _row("h1", "h1")})

    def invoke(_bin, _root, collection_path, _home, **_kwargs):
        graph = {"nodes": [{"id": "a"}, {"id": "b"}], "links": []}
        (collection_path / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
        (collection_path / "manifest.json").write_text("{not json", encoding="utf-8")
        return graphify_cli.CliResult(returncode=0, stdout="", stderr="")

    monkeypatch.setitem(sync_project._INVOKERS, sync_project.ACTION_EXTRACT, invoke)
    outcome = sync_project.apply_action(
        "example-org.gw",
        "graphify",
        tmp_path / "repo",
        collection,
        sync_project.ACTION_EXTRACT,
        DIGEST,
        tmp_path / "home",
    )

    assert outcome.new_manifest == DIGEST


def test_held_state_makes_the_next_run_extract_again():
    prior = {"code_hash": "c1", "semantic_hash": "s1", "file_count": 3}
    action = sync_project.decide_action(prior, DIGEST, has_graph=True)
    assert action == sync_project.ACTION_EXTRACT


NEW_DOC = SourceDigest(
    code_hash="c2",
    semantic_hash="s2",
    file_count=3,
    semantic_paths=("README.md", "docs/new.md", "conf/app.yaml"),
)


def test_new_document_without_a_row_holds_once(tmp_path, monkeypatch):
    """graphify writes no row for a new document whose LLM result was partial."""
    before = {"README.md": _row("h1", "h1")}
    after = {"README.md": _row("h1", "h1")}
    outcome, _ = _run(tmp_path, monkeypatch, before, after, digest=NEW_DOC)

    assert outcome.new_manifest is None
    assert "docs/new.md" in outcome.reason
    assert outcome.unstamped_documents == ("docs/new.md",)


def test_retried_new_document_is_accepted_with_warning(tmp_path, monkeypatch, caplog):
    before = {"README.md": _row("h1", "h1")}
    after = {"README.md": _row("h1", "h1")}
    with caplog.at_level(logging.WARNING, logger="graphify_mesh.sync"):
        outcome, _ = _run(
            tmp_path,
            monkeypatch,
            before,
            after,
            digest=NEW_DOC,
            retried=frozenset({"docs/new.md"}),
        )

    assert outcome.new_manifest == NEW_DOC
    assert "still unstamped" in outcome.reason
    assert outcome.unstamped_documents == ("docs/new.md",), "kept so it stays bounded"
    assert any("docs/new.md" in r.getMessage() for r in caplog.records)


def test_new_document_with_an_empty_row_holds_once(tmp_path, monkeypatch):
    before = {"README.md": _row("h1", "h1")}
    after = {"README.md": _row("h1", "h1"), "docs/new.md": _row("n", "")}
    outcome, _ = _run(tmp_path, monkeypatch, before, after, digest=NEW_DOC)

    assert outcome.new_manifest is None
    assert outcome.unstamped_documents == ("docs/new.md",)


def test_stamped_new_document_is_accepted(tmp_path, monkeypatch):
    before = {"README.md": _row("h1", "h1")}
    after = {"README.md": _row("h1", "h1"), "docs/new.md": _row("n", "n")}
    outcome, _ = _run(tmp_path, monkeypatch, before, after, digest=NEW_DOC)

    assert outcome.new_manifest == NEW_DOC
    assert outcome.unstamped_documents == ()


def test_absolute_manifest_keys_under_the_root_match_the_digest(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    before = {str(root / "README.md"): _row("h1", "h1")}
    after = {
        str(root / "README.md"): _row("h1", "h1"),
        str(root / "docs" / "new.md"): _row("n", "n"),
    }
    outcome, _ = _run(tmp_path, monkeypatch, before, after, digest=NEW_DOC)

    assert outcome.new_manifest == NEW_DOC


def test_partial_document_already_unstamped_and_retried_is_accepted(tmp_path, monkeypatch):
    """A stamped document that comes back "" holds once; with its path in the
    retried set the same result is accepted."""
    before = {"README.md": _row("h1", "h1")}
    after = {"README.md": _row("h1", ""), "docs/new.md": _row("n", "n")}
    held, _ = _run(tmp_path, monkeypatch, before, after, digest=NEW_DOC)
    assert held.unstamped_documents == ("README.md",)

    (tmp_path / "second").mkdir()
    accepted, _ = _run(
        tmp_path / "second",
        monkeypatch,
        before,
        after,
        digest=NEW_DOC,
        retried=frozenset({"README.md"}),
    )
    assert accepted.new_manifest == NEW_DOC


def test_update_never_checks_documents(tmp_path, monkeypatch):
    outcome, _ = _run(
        tmp_path, monkeypatch, {}, {}, action=sync_project.ACTION_UPDATE, digest=NEW_DOC
    )
    assert outcome.new_manifest == NEW_DOC
    assert outcome.unstamped_documents is None


def test_digest_lists_semantic_paths_relative_and_posix(tmp_path):
    (tmp_path / "guide").mkdir()
    (tmp_path / "guide" / "intro.md").write_text("# g\n", encoding="utf-8")
    (tmp_path / "m.py").write_text("x = 1\n", encoding="utf-8")
    digest = compute_source_manifest(tmp_path)
    assert digest.semantic_paths == ("guide/intro.md",)
    assert "m.py" not in digest.semantic_paths
    assert digest == SourceDigest(digest.code_hash, digest.semantic_hash, digest.file_count)


REPO = "example-org.gw"


def test_pipeline_gives_each_unstamped_document_one_retry(env, monkeypatch):
    monkeypatch.delenv(CONTENT_DIGEST_ENV, raising=False)
    root = env.add_repo(REPO, "example-org", "gw", "gw.example-org.dev.lo")
    env.write_registry()
    (root / "README.md").write_text("# a\n", encoding="utf-8")
    run(env.settings(shrink_policy="count"))
    state_path = env.settings().state_path
    seeded = load_state(state_path)[REPO]
    (root / "README.md").write_text("# edited\n", encoding="utf-8")

    seen: list = []

    def fake_apply(repo_id, _bin, _root, _coll, action, manifest, _home, **kwargs):
        retried = kwargs["retried_documents"]
        seen.append((action, retried, manifest.semantic_paths))
        return ProjectOutcome(
            repo_id,
            action,
            sync_project.STATUS_UPDATED,
            new_manifest=manifest if retried else None,
            unstamped_documents=("README.md",),
        )

    monkeypatch.setattr(pipeline, "apply_action", fake_apply)

    run(env.settings(shrink_policy="count"))
    held = load_state(state_path)[REPO]
    assert held["semantic_hash"] == seeded["semantic_hash"], "accepted state held"
    assert held[sync_project.UNSTAMPED_DOCUMENTS_KEY] == ["README.md"]

    run(env.settings(shrink_policy="count"))
    accepted = load_state(state_path)[REPO]
    assert [s[0] for s in seen] == [sync_project.ACTION_EXTRACT] * 2
    assert seen[0][1] == frozenset() and seen[1][1] == frozenset({"README.md"})
    assert "README.md" in seen[0][2]
    assert accepted["semantic_hash"] != seeded["semantic_hash"]
    assert accepted[sync_project.UNSTAMPED_DOCUMENTS_KEY] == ["README.md"]

    run(env.settings(shrink_policy="count"))
    assert len(seen) == 2, "no third extract: the source is accepted"
