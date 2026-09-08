"""Tests for the outbound-TLS mode knob (graphify_mesh.sync.tls)."""

from __future__ import annotations

import ssl
import sys

import pytest

from graphify_mesh.sync.tls import (
    TLS_INSECURE,
    TLS_RELAXED,
    TLS_STRICT,
    resolve_tls_mode,
    ssl_context,
)


def test_default_mode_is_strict():
    assert resolve_tls_mode({}) == TLS_STRICT


@pytest.mark.parametrize("value", ["relaxed", "RELAXED", " relaxed "])
def test_mode_is_case_and_space_insensitive(value):
    assert resolve_tls_mode({"GRAPHIFY_MESH_TLS_MODE": value}) == TLS_RELAXED


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_insecure_shorthand(value):
    assert resolve_tls_mode({"GRAPHIFY_MESH_TLS_INSECURE": value}) == TLS_INSECURE


@pytest.mark.parametrize("value", ["0", "false", "", "no"])
def test_insecure_shorthand_falsy_values_keep_strict(value):
    assert resolve_tls_mode({"GRAPHIFY_MESH_TLS_INSECURE": value}) == TLS_STRICT


def test_explicit_mode_wins_over_shorthand():
    env = {"GRAPHIFY_MESH_TLS_MODE": "relaxed", "GRAPHIFY_MESH_TLS_INSECURE": "1"}
    assert resolve_tls_mode(env) == TLS_RELAXED


def test_unknown_mode_raises_naming_the_var():
    with pytest.raises(ValueError, match="GRAPHIFY_MESH_TLS_MODE"):
        resolve_tls_mode({"GRAPHIFY_MESH_TLS_MODE": "kinda-secure"})


def test_strict_context_keeps_every_check():
    context = ssl_context({})
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    # "strict" is defined as the stock default context, so it must carry
    # whatever this interpreter's defaults are — no more, no less.
    assert context.verify_flags == ssl.create_default_context().verify_flags


@pytest.mark.skipif(
    sys.version_info < (3, 13),
    reason="create_default_context() only enables VERIFY_X509_STRICT from 3.13 on",
)
def test_strict_context_carries_the_strict_profile_on_313_plus():
    assert ssl_context({}).verify_flags & ssl.VERIFY_X509_STRICT


def test_relaxed_context_drops_only_the_strict_profile():
    context = ssl_context({"GRAPHIFY_MESH_TLS_MODE": "relaxed"})
    # The point of "relaxed": chain and hostname verification survive.
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert not context.verify_flags & ssl.VERIFY_X509_STRICT
    # And nothing else in the flag set moves.
    expected = ssl.create_default_context().verify_flags & ~ssl.VERIFY_X509_STRICT
    assert context.verify_flags == expected


def test_insecure_context_disables_verification_and_warns(caplog):
    with caplog.at_level("WARNING", logger="graphify_mesh.sync.tls"):
        context = ssl_context({"GRAPHIFY_MESH_TLS_MODE": "insecure"})
    assert context.verify_mode == ssl.CERT_NONE
    assert context.check_hostname is False
    assert "NOT verified" in caplog.text
