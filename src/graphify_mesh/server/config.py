"""Resolved runtime configuration for one graphify-mesh server process.

Mirrors `graphify_mesh.sync.config.Settings`'s "all paths configurable"
convention so tests never touch a real filesystem tree; the default below is
a placeholder — override via GRAPHIFY_MESH_ROOT for your environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 19744
DEFAULT_HTTP_PATH = "/mcp"

_PUBLIC_BIND_HOSTS = frozenset({"0.0.0.0", "::", ""})  # noqa: S104


class ConfigError(ValueError):
    """Invalid transport configuration. Raised at startup, never per request."""


@dataclass
class ServerConfig:
    mesh_root: Path
    registry_path: Path
    transport: str = "stdio"
    http_host: str = DEFAULT_HTTP_HOST
    http_port: int = DEFAULT_HTTP_PORT
    http_path: str = DEFAULT_HTTP_PATH
    http_token: str | None = field(default=None, repr=False)
    allow_public_bind: bool = False

    @property
    def global_dir(self) -> Path:
        return self.mesh_root / "graphify" / "global"

    @property
    def current_symlink(self) -> Path:
        return self.global_dir / "current"

    @property
    def embeddings_current_symlink(self) -> Path:
        return self.global_dir / "embeddings" / "current"

    @classmethod
    def from_env(
        cls,
        mesh_root: Path | None = None,
        registry_path: Path | None = None,
        transport: str | None = None,
        http_host: str | None = None,
        http_port: int | None = None,
        http_path: str | None = None,
        allow_public_bind: bool | None = None,
    ) -> ServerConfig:
        # Defaults to the current working directory (no machine-specific
        # path); set GRAPHIFY_MESH_ROOT for a real deployment.
        resolved_mesh_root = Path(
            mesh_root or os.environ.get("GRAPHIFY_MESH_ROOT") or Path.cwd()
        ).resolve()
        resolved_registry = Path(
            registry_path
            or os.environ.get(
                "GRAPHIFY_MESH_REGISTRY", str(resolved_mesh_root / "bin" / "registry.json")
            )
        ).resolve()

        # Resolve transport (explicit argument > environment > default)
        if transport is not None:
            resolved_transport = transport
        elif "GRAPHIFY_MESH_TRANSPORT" in os.environ:
            resolved_transport = os.environ["GRAPHIFY_MESH_TRANSPORT"]
        else:
            resolved_transport = "stdio"

        if resolved_transport not in ("stdio", "http"):
            raise ConfigError(
                f"Invalid transport '{resolved_transport}'; must be 'stdio' or 'http'"
            )

        # Resolve HTTP host (explicit argument > environment > default)
        resolved_http_host = (
            http_host or os.environ.get("GRAPHIFY_MESH_HTTP_HOST") or DEFAULT_HTTP_HOST
        )

        # Resolve HTTP port (explicit argument > environment > default)
        resolved_http_port: int
        if http_port is not None:
            resolved_http_port = http_port
        else:
            port_str = os.environ.get("GRAPHIFY_MESH_HTTP_PORT")
            if port_str:
                try:
                    resolved_http_port = int(port_str)
                except ValueError as err:
                    raise ConfigError(
                        f"GRAPHIFY_MESH_HTTP_PORT must be an integer, got '{port_str}'"
                    ) from err
            else:
                resolved_http_port = DEFAULT_HTTP_PORT

        # Validate port is in valid range
        if resolved_http_port < 1 or resolved_http_port > 65535:
            raise ConfigError(
                f"HTTP port must be between 1 and 65535, got {resolved_http_port}"  # noqa: E501
            )

        # Resolve HTTP path (explicit argument > environment > default)
        resolved_http_path = (
            http_path or os.environ.get("GRAPHIFY_MESH_HTTP_PATH") or DEFAULT_HTTP_PATH
        )

        # Validate path starts with /
        if not resolved_http_path.startswith("/"):
            raise ConfigError(f"HTTP path must start with '/', got '{resolved_http_path}'")

        # Resolve allow_public_bind (explicit argument > environment > default)
        resolved_allow_public_bind: bool
        if allow_public_bind is not None:
            resolved_allow_public_bind = allow_public_bind
        else:
            env_val = os.environ.get("GRAPHIFY_MESH_ALLOW_PUBLIC_BIND", "").lower()
            resolved_allow_public_bind = env_val in ("1", "true", "yes")

        # Check for public bind without opt-in
        if resolved_http_host in _PUBLIC_BIND_HOSTS and not resolved_allow_public_bind:
            raise ConfigError(
                f"Cannot bind to {resolved_http_host} without --allow-public-bind flag"
            )

        # Resolve HTTP token (explicit argument > environment > default)
        resolved_http_token: str | None = None
        if resolved_transport == "http":
            token = (os.environ.get("GRAPHIFY_MESH_HTTP_TOKEN") or "").strip()
            if not token:
                raise ConfigError(
                    "HTTP transport requires a bearer token: set GRAPHIFY_MESH_HTTP_TOKEN "
                    "(a loopback TCP port is reachable by every local process, unlike the "
                    "unix socket it replaces)"
                )
            resolved_http_token = token

        return cls(
            mesh_root=resolved_mesh_root,
            registry_path=resolved_registry,
            transport=resolved_transport,
            http_host=resolved_http_host,
            http_port=resolved_http_port,
            http_path=resolved_http_path,
            http_token=resolved_http_token,
            allow_public_bind=resolved_allow_public_bind,
        )
