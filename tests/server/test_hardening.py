"""Robustness/hardening contract: oversized stdin lines, crashing handlers,
unexpected tool exceptions, and hostile tool arguments must all degrade
gracefully — the serve loop and the server process always survive."""

from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import pytest
from conftest import build_generation, fake_embed_query_fn, make_node, registry_repo, write_registry

from graphify_mesh.server import stdio_guard
from graphify_mesh.server.config import ServerConfig
from graphify_mesh.server.retrieval import exact_alias_hits, lexical_candidates
from graphify_mesh.server.server import MAX_K, MAX_TOKEN_BUDGET, GraphifyMeshServer


def _server(tmp_path: Path, cwd: Path) -> GraphifyMeshServer:
    config = ServerConfig.from_env(
        mesh_root=tmp_path, registry_path=tmp_path / "bin" / "registry.json"
    )
    return GraphifyMeshServer(config, cwd=cwd, embed_query_fn=fake_embed_query_fn())


def _search_server(tmp_path, monkeypatch) -> GraphifyMeshServer:
    """Server with a registered repo + synthetic generation so a VALID
    search succeeds — lets tests prove the server survives after a bad call."""
    root = tmp_path / "acme"
    root.mkdir(parents=True)
    write_registry(tmp_path / "bin" / "registry.json", [registry_repo("acme.repo", root)])
    generation = build_generation(
        [make_node("acme.repo", "OrderService", "src/order.py", node_id="n1")]
    )
    server = _server(tmp_path, root)
    monkeypatch.setattr(server, "_generation", lambda: generation)
    return server


# --- stdio_guard.capped_stdin: oversized-line cap ---------------------------
#
# The size cap protocol.py used to provide moved to stdio_guard (see that
# module's docstring): the SDK's own stdio reader has no bound at all.


def _drain(async_file) -> list[str]:
    async def _collect() -> list[str]:
        collected: list[str] = []
        it: AsyncIterator[str] = async_file.__aiter__()
        async for line in it:
            collected.append(line)
        return collected

    return anyio.run(_collect)


def test_oversized_line_is_skipped_and_loop_keeps_serving(monkeypatch):
    monkeypatch.setattr(stdio_guard, "MAX_LINE_BYTES", 64)
    giant = "x" * 500  # one newline-less-within-cap giant line
    good = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    stream = io.StringIO(giant + "\n" + good + "\n")
    lines = _drain(stdio_guard.capped_stdin(stream))
    assert [json.loads(line) for line in lines] == [{"jsonrpc": "2.0", "id": 1, "method": "ping"}]


def test_oversized_line_without_trailing_newline_at_eof(monkeypatch):
    monkeypatch.setattr(stdio_guard, "MAX_LINE_BYTES", 64)
    stream = io.StringIO("y" * 500)  # oversized AND no newline before EOF
    assert _drain(stdio_guard.capped_stdin(stream)) == []


def test_line_at_cap_is_still_parsed(monkeypatch):
    monkeypatch.setattr(stdio_guard, "MAX_LINE_BYTES", 4096)
    message = {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {"pad": "z" * 4000}}
    line = json.dumps(message)
    assert len(line) <= 4096
    stream = io.StringIO(line + "\n")
    lines = _drain(stdio_guard.capped_stdin(stream))
    assert [json.loads(line) for line in lines] == [message]


# --- handler exceptions: a raise ABOVE the tool layer (inside the SDK's own
# request/notification dispatch) is exercised at the real-subprocess level
# now, in tests/server/test_stdio_e2e.py:
# test_request_handler_exception_above_the_tool_layer_gets_error_and_loop_survives
# and test_notification_handler_exception_still_gets_no_response — that
# dispatch lives in mcp.server.lowlevel.Server, not in this package's code
# anymore. `GraphifyMeshServer.call_tool`'s own try/except below is a
# narrower, separate guarantee: "no exception text leaks" for a raise INSIDE
# a tool handler specifically (call_tool sanitizes to "internal error"; the
# SDK's own dispatch, verified in the tests above, does not sanitize a
# raising handler's message — it only guarantees no traceback/path leak and
# that the transport loop survives).


# --- call_tool: unexpected exception -----------------------------------------


def test_call_tool_unexpected_exception_returns_generic_internal_error(monkeypatch):
    server = _server(Path("/tmp"), Path("/tmp"))

    def explode(arguments: dict) -> dict:
        raise RuntimeError("stacktrace /etc/passwd secret")

    monkeypatch.setattr(server, "tool_search", explode)
    result = server.call_tool("search", {"q": "x"})
    assert result["isError"] is True
    assert result["content"][0]["text"] == "internal error"


def test_call_tool_non_serializable_result_hits_generic_branch(monkeypatch):
    server = _server(Path("/tmp"), Path("/tmp"))
    monkeypatch.setattr(server, "tool_search", lambda arguments: {"bad": object()})
    result = server.call_tool("search", {"q": "x"})
    assert result["isError"] is True
    assert result["content"][0]["text"] == "internal error"


# --- argument validation: k ---------------------------------------------------


@pytest.mark.parametrize("bad_k", [0, -5, MAX_K + 1, 10**9, "abc", 1.5, True])
def test_search_rejects_invalid_k_and_server_survives(tmp_path, monkeypatch, bad_k):
    server = _search_server(tmp_path, monkeypatch)
    result = server.call_tool("search", {"q": "OrderService", "k": bad_k})
    assert result["isError"] is True
    assert "'k' must be" in result["content"][0]["text"]

    # Server survives: a subsequent valid call succeeds.
    ok = server.call_tool("search", {"q": "OrderService", "k": 5})
    assert ok["isError"] is False


@pytest.mark.parametrize("tool", ["cross_project", "find_similar"])
def test_other_k_tools_reject_invalid_k(tmp_path, monkeypatch, tool):
    server = _search_server(tmp_path, monkeypatch)
    arguments = {"q": "x", "node": "x", "k": "abc"}
    result = server.call_tool(tool, arguments)
    assert result["isError"] is True
    assert "'k' must be" in result["content"][0]["text"]


# --- argument validation: token_budget ---------------------------------------


@pytest.mark.parametrize("bad_budget", [0, -1, MAX_TOKEN_BUDGET + 1, "lots", True])
def test_context_pack_rejects_invalid_token_budget(tmp_path, monkeypatch, bad_budget):
    server = _search_server(tmp_path, monkeypatch)
    result = server.call_tool("context_pack", {"goal": "g", "token_budget": bad_budget})
    assert result["isError"] is True
    assert "'token_budget' must be" in result["content"][0]["text"]


def test_context_pack_accepts_valid_token_budget(tmp_path, monkeypatch):
    server = _search_server(tmp_path, monkeypatch)
    result = server.call_tool("context_pack", {"goal": "OrderService", "token_budget": 500})
    assert result["isError"] is False


# --- retrieval: malformed lexical-index entry ---------------------------------


def test_lexical_entry_missing_key_is_tolerated():
    generation = build_generation(
        [
            make_node("acme.repo", "OrderService", "src/order.py", node_id="n1"),
            make_node("acme.repo", "OrderRepository", "src/order_repo.py", node_id="n2"),
        ]
    )
    # Corrupt every posting list with a malformed entry — schema_version 3
    # entries are int-packed `(doc_id << FIELD_PACK_BITS) | field_index`
    # values, so a stray list here is the wrong shape and must be skipped.
    for entries in generation.lexical.get("postings", {}).values():
        entries.append(["acme.repo"])

    ranked = lexical_candidates("OrderService", generation.lexical, None)
    assert ranked  # real entries still rank; malformed one is skipped


def test_lexical_candidates_and_alias_hits_still_serve_v2_generation():
    """A published v2 generation must keep serving after deploy (publish can
    lag behind package upgrade — e.g. shrink-guard blocking)."""
    v2 = {
        "schema_version": 2,
        "field_boosts": {"label": 3.0, "path": 1.5, "snippet": 1.0},
        "postings": {"alphaclass": [["repoA", "keyA1", "label"]]},
        "doc_freq": {"global": {"alphaclass": 1}, "per_repo": {}},
        "alias_exact": {"alphaclass": [["repoA", "keyA1"]]},
        "document_count": 1,
    }
    assert exact_alias_hits("AlphaClass", v2, None) == ["keyA1"]
    assert lexical_candidates("AlphaClass", v2, None) == ["keyA1"]


def test_search_tool_survives_malformed_lexical_entry(tmp_path, monkeypatch):
    server = _search_server(tmp_path, monkeypatch)
    generation = server._generation()
    for entries in generation.lexical.get("postings", {}).values():
        entries.append(["acme.repo"])  # malformed: wrong length, no key

    result = server.call_tool("search", {"q": "OrderService"})
    assert result["isError"] is False


# --- constant alignment --------------------------------------------------------


def test_max_k_is_the_retrieval_clamp_constant():
    from graphify_mesh.server import ranking

    assert MAX_K == ranking.MAX_K


# --- project_map: registered repo_ids only -----------------------------------


@pytest.mark.parametrize("bad_repo", [None, "", 5, ["repo.a"], {"repo": "a"}])
def test_project_map_rejects_non_string_repo(tmp_path, monkeypatch, bad_repo):
    server = _search_server(tmp_path, monkeypatch)
    with pytest.raises(Exception, match="non-empty string"):
        server.tool_project_map({"repo": bad_repo})


def test_project_map_fails_closed_for_unregistered_repo(tmp_path, monkeypatch):
    """A repo present in the generation but absent from (or removed from)
    the registry must not resolve — project_map serves REGISTERED ids only."""
    root = tmp_path / "acme"
    root.mkdir(parents=True)
    write_registry(tmp_path / "bin" / "registry.json", [registry_repo("acme.repo", root)])
    generation = build_generation(
        [make_node("ghost.repo", "GhostService", "src/ghost.py", node_id="g1")]
    )
    server = _server(tmp_path, root)
    monkeypatch.setattr(server, "_generation", lambda: generation)

    result = server.tool_project_map({"repo": "ghost.repo"})
    assert result["resolved"] is False
    assert "repo_not_registered" in result["degraded"]


def test_project_map_resolves_registered_repo(tmp_path, monkeypatch):
    server = _search_server(tmp_path, monkeypatch)
    result = server.tool_project_map({"repo": "acme.repo"})
    assert result["resolved"] is True
    assert result["node_count"] == 1
