"""SSRF guard contract: only http/https base URLs with a host may ever reach
urllib in the naming/embedding backends — file://, gopher:// etc. must fail
the health-check path without any request being attempted."""

from __future__ import annotations

import pytest

from graphify_mesh.sync.config import (
    API_TIMEOUT_MAX_SECONDS,
    OLLAMA_DEFAULT_API_TIMEOUT,
    _api_timeout_from_env,
    is_valid_http_base_url,
)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "gopher://host:70/x",
        "ftp://host/",
        "unix:///var/run/x.sock",
        "http://",  # no host
        "https://",
        "not-a-url",
        "",
        "//host-without-scheme",
        "javascript:alert(1)",
    ],
)
def test_invalid_base_urls_rejected(url):
    assert is_valid_http_base_url(url) is False


@pytest.mark.parametrize(
    "url",
    ["http://localhost:11434", "https://ollama.internal:11434", "http://127.0.0.1:11434/v1"],
)
def test_http_https_with_host_accepted(url):
    assert is_valid_http_base_url(url) is True


# ---------------------------------------------------------------------------
# GRAPHIFY_MESH_OLLAMA_API_TIMEOUT
# ---------------------------------------------------------------------------


def test_ollama_api_timeout_defaults_to_the_declared_constant(monkeypatch):
    monkeypatch.delenv("GRAPHIFY_MESH_OLLAMA_API_TIMEOUT", raising=False)
    assert (
        _api_timeout_from_env("GRAPHIFY_MESH_OLLAMA_API_TIMEOUT", OLLAMA_DEFAULT_API_TIMEOUT)
        == OLLAMA_DEFAULT_API_TIMEOUT
    )
    assert OLLAMA_DEFAULT_API_TIMEOUT == 180.0


def test_ollama_api_timeout_accepts_a_value_up_to_the_ceiling(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_MESH_OLLAMA_API_TIMEOUT", str(API_TIMEOUT_MAX_SECONDS))
    assert (
        _api_timeout_from_env("GRAPHIFY_MESH_OLLAMA_API_TIMEOUT", OLLAMA_DEFAULT_API_TIMEOUT)
        == API_TIMEOUT_MAX_SECONDS
    )


@pytest.mark.parametrize("raw", ["not-a-number", "0", "-5", "3601"])
def test_ollama_api_timeout_rejects_bad_values_naming_the_variable(monkeypatch, raw):
    """Fail at startup with the variable named, rather than hanging (or raising
    a bare float() traceback) in the middle of a 75-90 minute pipeline."""
    monkeypatch.setenv("GRAPHIFY_MESH_OLLAMA_API_TIMEOUT", raw)
    with pytest.raises(ValueError, match="GRAPHIFY_MESH_OLLAMA_API_TIMEOUT"):
        _api_timeout_from_env("GRAPHIFY_MESH_OLLAMA_API_TIMEOUT", OLLAMA_DEFAULT_API_TIMEOUT)
