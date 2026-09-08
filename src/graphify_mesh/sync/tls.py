"""TLS context selection for the sync pipeline's outbound HTTPS calls.

Why this exists
---------------
On hosts behind a TLS-intercepting appliance (corporate "SSL decryption"), every
outbound HTTPS request is re-signed by an internal CA. Those generated
certificates frequently omit extensions that the strict X.509 profile requires
— most commonly the Authority Key Identifier. Python 3.13+ enables
``ssl.VERIFY_X509_STRICT`` in ``ssl.create_default_context()``, so such a chain
is rejected with::

    [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
    Missing Authority Key Identifier

even though ``curl`` and Node accept it against the same CA store, and even
though the interception CA is installed and trusted system-wide. No environment
variable turns that flag off, which is why the pipeline needs its own knob: the
health probes would otherwise report a perfectly reachable backend as unhealthy
and the pipeline would run permanently degraded.

The knob
--------
``GRAPHIFY_MESH_TLS_MODE`` (default ``strict``):

``strict``
    Stock ``ssl.create_default_context()``. Full chain verification, hostname
    check, and the strict extension profile. Use this everywhere the chain is
    not rewritten.

``relaxed``
    Clears ``VERIFY_X509_STRICT`` only. Chain verification against the system
    CA store and hostname checking stay ON — this accepts a trusted-but-sloppy
    interception certificate, and nothing else. This is the setting for a
    corporate-proxied host.

``insecure``
    No verification at all: ``check_hostname = False``,
    ``verify_mode = CERT_NONE``. The ``curl -k`` equivalent. Any certificate is
    accepted, including one from an active attacker, so the connection gives no
    authenticity guarantee. Only for a diagnostic run on a network you trust.

``GRAPHIFY_MESH_TLS_INSECURE=1`` is accepted as a shorthand for
``GRAPHIFY_MESH_TLS_MODE=insecure``; an explicit ``GRAPHIFY_MESH_TLS_MODE``
wins over it.

An unknown mode value raises ``ValueError`` at call time rather than silently
falling back — a typo in the mode must not quietly re-enable a check the
operator meant to drop, nor quietly drop one they meant to keep.

The context is only meaningful for ``https://`` URLs; ``urlopen`` ignores it for
plain HTTP, so callers can pass it unconditionally.
"""

from __future__ import annotations

import logging
import os
import ssl

log = logging.getLogger("graphify_mesh.sync.tls")

TLS_MODE_ENV = "GRAPHIFY_MESH_TLS_MODE"
TLS_INSECURE_ENV = "GRAPHIFY_MESH_TLS_INSECURE"

TLS_STRICT = "strict"
TLS_RELAXED = "relaxed"
TLS_INSECURE = "insecure"

TLS_MODES = (TLS_STRICT, TLS_RELAXED, TLS_INSECURE)

# Truthy spellings accepted for the boolean shorthand. Anything else (including
# "0", "false", "") means "not set".
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def resolve_tls_mode(env: dict[str, str] | None = None) -> str:
    """Return the configured TLS mode, one of TLS_MODES.

    Reads os.environ by default; `env` is for tests. Raises ValueError on an
    unrecognized GRAPHIFY_MESH_TLS_MODE value.
    """
    source = os.environ if env is None else env
    raw = (source.get(TLS_MODE_ENV) or "").strip().lower()
    if raw:
        if raw not in TLS_MODES:
            raise ValueError(f"{TLS_MODE_ENV} must be one of {', '.join(TLS_MODES)}, got {raw!r}")
        return raw
    if (source.get(TLS_INSECURE_ENV) or "").strip().lower() in _TRUTHY:
        return TLS_INSECURE
    return TLS_STRICT


def ssl_context(env: dict[str, str] | None = None) -> ssl.SSLContext:
    """Build the SSLContext for outbound pipeline HTTPS calls.

    Pass the result as `urlopen(..., context=...)`. Safe to call per request:
    building a context is cheap next to the request itself, and doing so keeps
    the mode live (a systemd restart with a changed EnvironmentFile takes effect
    without any cached state).
    """
    mode = resolve_tls_mode(env)
    context = ssl.create_default_context()
    if mode == TLS_STRICT:
        return context
    if mode == TLS_RELAXED:
        # Chain + hostname verification stay on; only the strict
        # extension-presence profile is dropped.
        context.verify_flags &= ~ssl.VERIFY_X509_STRICT
        return context
    # insecure
    log.warning(
        "%s=insecure: TLS certificates are NOT verified for outbound pipeline "
        "calls (curl -k equivalent). Use 'relaxed' instead unless this is a "
        "one-off diagnostic run.",
        TLS_MODE_ENV,
    )
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context
