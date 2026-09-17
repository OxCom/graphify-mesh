from __future__ import annotations

import pytest

from graphify_mesh.server.server import main


def test_no_arguments_runs_stdio(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_ROOT", str(tmp_path))
    monkeypatch.delenv("GRAPHIFY_MESH_TRANSPORT", raising=False)
    calls: list[str] = []
    monkeypatch.setattr(
        "graphify_mesh.server.server.serve_stdio", lambda mesh: calls.append("stdio")
    )
    assert main([]) == 0
    assert calls == ["stdio"]


def test_transport_http_starts_the_daemon(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_ROOT", str(tmp_path))
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "s3cret")
    seen: list[tuple[str, int, str]] = []

    def fake_serve_http(mesh, config):
        seen.append((config.http_host, config.http_port, config.http_path))

    # Patched on http_app, not server: server.py imports http_app lazily
    # inside main() to avoid a circular import with test_http_app.py's
    # import order (http_app -> sdk_app -> server), so `main` always reads
    # http_app.serve_http fresh at call time.
    monkeypatch.setattr("graphify_mesh.server.http_app.serve_http", fake_serve_http)
    assert main(["--transport", "http", "--port", "20005"]) == 0
    assert seen == [("127.0.0.1", 20005, "/mcp")]


def test_http_without_token_exits_2_and_says_why(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GRAPHIFY_MESH_ROOT", str(tmp_path))
    monkeypatch.delenv("GRAPHIFY_MESH_HTTP_TOKEN", raising=False)
    assert main(["--transport", "http"]) == 2
    assert "GRAPHIFY_MESH_HTTP_TOKEN" in capsys.readouterr().err


def test_public_bind_without_opt_in_exits_2(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GRAPHIFY_MESH_ROOT", str(tmp_path))
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "s3cret")
    assert main(["--transport", "http", "--host", "0.0.0.0"]) == 2  # noqa: S104
    assert "allow-public-bind" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["search", "context_pack"])
def test_cwd_description_no_longer_claims_a_proxy_injects_it(mesh_server, name):
    schema = next(s for s in mesh_server.tool_schemas() if s["name"] == name)
    description = schema["inputSchema"]["properties"]["cwd"]["description"]
    assert "proxy" not in description.lower()
    assert "absolute" in description.lower()
