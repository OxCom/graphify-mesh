from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest
from starlette.testclient import TestClient

from graphify_mesh.server.http_app import build_http_app

HEADERS_OK = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


@pytest.fixture
def client(mesh_server, http_config):
    app = build_http_app(mesh_server, http_config)
    return TestClient(app, base_url="http://127.0.0.1:19744")


def test_missing_token_is_401(client, http_config):
    with client as c:
        response = c.post(http_config.http_path, headers=HEADERS_OK, json=_initialize())
    assert response.status_code == 401
    assert "token" not in response.text.lower() or http_config.http_token not in response.text


def test_wrong_token_is_401(client, http_config):
    with client as c:
        response = c.post(
            http_config.http_path,
            headers={**HEADERS_OK, "Authorization": "Bearer nope"},
            json=_initialize(),
        )
    assert response.status_code == 401


def test_correct_token_lists_the_five_tools(client, http_config, mesh_server):
    with client as c:
        c.post(http_config.http_path, headers=_auth(http_config), json=_initialize())
        response = c.post(http_config.http_path, headers=_auth(http_config), json=_tools_list())
    assert response.status_code == 200
    names = [t["name"] for t in _result(response)["tools"]]
    assert names == [s["name"] for s in mesh_server.tool_schemas()]


def test_foreign_host_header_is_rejected(client, http_config):
    with client as c:
        response = c.post(
            http_config.http_path,
            headers={**_auth(http_config), "Host": "evil.example.com"},
            json=_initialize(),
        )
    assert response.status_code in (400, 401, 403, 421)


def test_tool_call_carries_cwd_through(client, http_config, registered_repo_root):
    with client as c:
        c.post(http_config.http_path, headers=_auth(http_config), json=_initialize())
        response = c.post(
            http_config.http_path,
            headers=_auth(http_config),
            json=_tools_call("search", {"q": "anything", "cwd": str(registered_repo_root)}),
        )
    assert response.status_code == 200
    assert _result(response)["isError"] is False


def test_no_session_id_is_issued_in_stateless_mode(client, http_config):
    with client as c:
        response = c.post(http_config.http_path, headers=_auth(http_config), json=_initialize())
    assert "mcp-session-id" not in {k.lower() for k in response.headers}


def test_token_is_never_logged(caplog, mesh_server, http_config):
    build_http_app(mesh_server, http_config)
    assert http_config.http_token not in caplog.text


def test_rejected_requests_never_log_the_token(caplog, mesh_server):
    """The security requirement is about the REJECTION branch, not just
    construction time: a wrong-token request and a no-token request both
    reach `_TokenAuthMiddleware`'s `log.warning` call, so this drives both
    and checks the captured record, not just `caplog.text` as a whole —
    a distinctive token value makes the substring check meaningful."""
    token = "distinctive-parity-secret-do-not-log-9e41ab"
    config = replace(mesh_server.config, transport="http", http_token=token)
    app = build_http_app(mesh_server, config)
    with caplog.at_level("WARNING", logger="graphify_mesh.server.http_app"):
        with TestClient(app, base_url=f"http://{config.http_host}:{config.http_port}") as c:
            wrong = c.post(
                config.http_path,
                headers={**HEADERS_OK, "Authorization": "Bearer nope"},
                json=_initialize(),
            )
            missing = c.post(config.http_path, headers=HEADERS_OK, json=_initialize())

    assert wrong.status_code == 401
    assert missing.status_code == 401
    assert token not in caplog.text

    rejection_records = [r for r in caplog.records if r.name == "graphify_mesh.server.http_app"]
    assert len(rejection_records) == 2
    for record in rejection_records:
        assert token not in record.getMessage()
        assert record.getMessage().startswith("rejected unauthorized request from ")


def test_lifespan_starts_and_stops_the_session_manager(client, caplog):
    """Guards against silently regressing to a client that never runs the
    ASGI lifespan (e.g. `httpx.ASGITransport`, which does not) — the fix
    this test was added for. `StreamableHTTPSessionManager.run()` logs one
    INFO line on entry and one on exit; both must appear, in order, bracketing
    the `with client:` block, and neither before nor after it."""
    with caplog.at_level("INFO", logger="mcp.server.streamable_http_manager"):
        with client:
            messages_inside = [r.message for r in caplog.records]
            assert any("session manager started" in m.lower() for m in messages_inside)
            assert not any("shutting down" in m.lower() for m in messages_inside)
    messages_after = [r.message for r in caplog.records]
    assert any("shutting down" in m.lower() for m in messages_after)


def _auth(config) -> dict:
    return {**HEADERS_OK, "Authorization": f"Bearer {config.http_token}"}


def _initialize() -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "0.0.0"},
        },
    }


def _tools_list() -> dict:
    return {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}


def _tools_call(name: str, arguments: dict) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


def _result(response: httpx.Response) -> dict:
    body = response.text
    if body.lstrip().startswith("data:"):
        lines = [line for line in body.splitlines() if line.startswith("data:")]
        body = lines[-1][len("data:") :].strip()
    payload = json.loads(body)
    return payload["result"]


# --- incoming-message size bound (stdio had one, HTTP did not state one) ---


def test_the_session_manager_states_the_shared_message_cap(mesh_server, http_config):
    """Both transports must state the same number in one place: relying on
    the SDK's own default leaves the HTTP bound version-dependent and
    different from stdio's."""
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    from graphify_mesh.server.frames import MAX_MESSAGE_BYTES

    managers: list[StreamableHTTPSessionManager] = []
    original = StreamableHTTPSessionManager.__init__

    def recording_init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        managers.append(self)

    StreamableHTTPSessionManager.__init__ = recording_init  # type: ignore[method-assign]
    try:
        build_http_app(mesh_server, http_config)
    finally:
        StreamableHTTPSessionManager.__init__ = original  # type: ignore[method-assign]

    assert len(managers) == 1
    assert managers[0].max_request_body_size == MAX_MESSAGE_BYTES


def test_oversized_body_is_rejected_on_received_bytes_not_content_length(client, http_config):
    """No `Content-Length` to trust: the cap has to count what arrives. The
    generator yields more than the cap in chunks, so a server that only
    checks the header would buffer all of it."""
    from graphify_mesh.server.frames import MAX_MESSAGE_BYTES

    chunk = b"x" * (1024 * 1024)
    chunks_needed = MAX_MESSAGE_BYTES // len(chunk) + 4

    def body_chunks():
        yield b'{"jsonrpc":"2.0","id":1,"method":"ping","params":{"pad":"'
        for _ in range(chunks_needed):
            yield chunk

    with client as c:
        response = c.post(http_config.http_path, headers=_auth(http_config), content=body_chunks())
    assert response.status_code == 413
    assert http_config.http_token not in response.text


def test_the_shared_cap_is_four_mebibytes(client, http_config):
    """Pins the number itself, not just that both transports agree on one.
    Lowering it is a deliberate decision (the shared daemon's memory is a
    machine-wide resource); raising it must be one too."""
    from graphify_mesh.server.frames import MAX_MESSAGE_BYTES

    assert MAX_MESSAGE_BYTES == 4 * 1024 * 1024


def test_a_body_under_the_shared_cap_is_still_served(client, http_config, registered_repo_root):
    """A legitimate frame well inside the cap must not be rejected by either
    the guard or the session manager below it."""
    from graphify_mesh.server.frames import MAX_MESSAGE_BYTES

    padding = "p" * (MAX_MESSAGE_BYTES // 2)
    with client as c:
        c.post(http_config.http_path, headers=_auth(http_config), json=_initialize())
        response = c.post(
            http_config.http_path,
            headers=_auth(http_config),
            json=_tools_call("search", {"q": padding, "cwd": str(registered_repo_root)}),
        )
    assert response.status_code == 200


# --- malformed frames get a generic error, never the SDK's detail ---


def test_unparseable_body_gets_a_generic_parse_error(client, http_config, caplog):
    with caplog.at_level("WARNING", logger="graphify_mesh.server.http_app"):
        with client as c:
            response = c.post(http_config.http_path, headers=_auth(http_config), content=b"{")
    payload = json.loads(response.text)
    assert payload["error"]["code"] == -32700
    assert payload["error"]["message"] == "parse error"
    assert "Expecting property name" not in response.text
    assert http_config.http_token not in caplog.text


def test_batch_array_gets_a_generic_invalid_request(client, http_config):
    with client as c:
        response = c.post(
            http_config.http_path,
            headers=_auth(http_config),
            json=[{"jsonrpc": "2.0", "id": 1, "method": "ping"}],
        )
    payload = json.loads(response.text)
    assert payload["error"]["code"] == -32600
    assert "validation error" not in response.text.lower()
    assert "pydantic" not in response.text.lower()


def test_invalid_envelope_never_echoes_the_offending_input(client, http_config):
    with client as c:
        response = c.post(
            http_config.http_path,
            headers=_auth(http_config),
            json={
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": ["s3cret-marker"],
            },
        )
    payload = json.loads(response.text)
    assert payload["id"] == 7
    assert payload["error"]["code"] == -32602
    assert "s3cret-marker" not in response.text
    assert "pydantic" not in response.text.lower()


def test_unauthorized_is_still_answered_before_any_body_parsing(client, http_config):
    """The token gate stays outermost: a request with no token and a
    malformed body must get the 401, not a JSON-RPC parse error."""
    with client as c:
        response = c.post(http_config.http_path, headers=HEADERS_OK, content=b"{")
    assert response.status_code == 401
    assert response.text == '{"error": "unauthorized"}'


def test_explicit_null_params_reaches_the_sdk(client, http_config):
    """Transport parity for the `params: null` frame the SDK's own models
    accept: it must not be rejected at the HTTP frame guard either."""
    with client as c:
        c.post(http_config.http_path, headers=_auth(http_config), json=_initialize())
        response = c.post(
            http_config.http_path,
            headers=_auth(http_config),
            json={"jsonrpc": "2.0", "id": 5, "method": "tools/list", "params": None},
        )
    assert response.status_code == 200
    assert "tools" in _result(response)
