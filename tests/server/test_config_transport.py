from __future__ import annotations

import pytest

from graphify_mesh.server.config import (
    DEFAULT_HTTP_HOST,
    DEFAULT_HTTP_PATH,
    DEFAULT_HTTP_PORT,
    ConfigError,
    ServerConfig,
)


def test_defaults_are_stdio(monkeypatch, tmp_path):
    for var in (
        "GRAPHIFY_MESH_TRANSPORT",
        "GRAPHIFY_MESH_HTTP_HOST",
        "GRAPHIFY_MESH_HTTP_PORT",
        "GRAPHIFY_MESH_HTTP_PATH",
        "GRAPHIFY_MESH_HTTP_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    config = ServerConfig.from_env(mesh_root=tmp_path)
    assert config.transport == "stdio"
    assert config.http_host == DEFAULT_HTTP_HOST
    assert config.http_port == DEFAULT_HTTP_PORT
    assert config.http_path == DEFAULT_HTTP_PATH
    assert config.http_token is None


def test_env_selects_http_and_port(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_TRANSPORT", "http")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_PORT", "20001")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "s3cret")
    config = ServerConfig.from_env(mesh_root=tmp_path)
    assert config.transport == "http"
    assert config.http_port == 20001
    assert config.http_token == "s3cret"


def test_explicit_argument_beats_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_TRANSPORT", "http")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_PORT", "20001")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "s3cret")
    config = ServerConfig.from_env(mesh_root=tmp_path, http_port=20002)
    assert config.http_port == 20002


def test_http_without_token_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_TRANSPORT", "http")
    monkeypatch.delenv("GRAPHIFY_MESH_HTTP_TOKEN", raising=False)
    with pytest.raises(ConfigError, match="GRAPHIFY_MESH_HTTP_TOKEN"):
        ServerConfig.from_env(mesh_root=tmp_path)


def test_blank_token_counts_as_absent(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_TRANSPORT", "http")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "   ")
    with pytest.raises(ConfigError, match="GRAPHIFY_MESH_HTTP_TOKEN"):
        ServerConfig.from_env(mesh_root=tmp_path)


def test_stdio_ignores_a_missing_token(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_TRANSPORT", "stdio")
    monkeypatch.delenv("GRAPHIFY_MESH_HTTP_TOKEN", raising=False)
    assert ServerConfig.from_env(mesh_root=tmp_path).transport == "stdio"


@pytest.mark.parametrize("value", ["sse", "HTTP2", ""])
def test_unknown_transport_is_refused(monkeypatch, tmp_path, value):
    monkeypatch.setenv("GRAPHIFY_MESH_TRANSPORT", value)
    with pytest.raises(ConfigError):
        ServerConfig.from_env(mesh_root=tmp_path)


@pytest.mark.parametrize("value", ["0", "-1", "65536", "abc", "80.5"])
def test_invalid_port_is_refused(monkeypatch, tmp_path, value):
    monkeypatch.setenv("GRAPHIFY_MESH_TRANSPORT", "http")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "s3cret")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_PORT", value)
    with pytest.raises(ConfigError):
        ServerConfig.from_env(mesh_root=tmp_path)


def test_path_must_start_with_slash(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_TRANSPORT", "http")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "s3cret")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_PATH", "mcp")
    with pytest.raises(ConfigError, match="path"):
        ServerConfig.from_env(mesh_root=tmp_path)


def test_public_bind_requires_opt_in(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_TRANSPORT", "http")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "s3cret")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_HOST", "0.0.0.0")  # noqa: S104
    with pytest.raises(ConfigError, match="allow-public-bind"):
        ServerConfig.from_env(mesh_root=tmp_path)
    config = ServerConfig.from_env(mesh_root=tmp_path, allow_public_bind=True)
    assert config.http_host == "0.0.0.0"  # noqa: S104


def test_token_does_not_leak_in_repr(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_TRANSPORT", "http")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_TOKEN", "super_secret_token_value")
    monkeypatch.setenv("GRAPHIFY_MESH_HTTP_HOST", "127.0.0.1")
    config = ServerConfig.from_env(mesh_root=tmp_path)
    config_repr = repr(config)
    # Token must not appear in repr
    assert "super_secret_token_value" not in config_repr
    # Other fields should still be visible for debugging
    assert "http_host" in config_repr
    assert "127.0.0.1" in config_repr
    assert "http" in config_repr  # transport
