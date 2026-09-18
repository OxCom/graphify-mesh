"""`graphify-mesh` stdio MCP server (WS5 deliverable 2): wires the 5 hybrid/
cross-project/evidence tools (`search`, `cross_project`, `find_similar`,
`project_map`, `context_pack`) onto the MCP SDK's stdio transport
(`mcp.server.stdio.stdio_server`), wrapped by `stdio_guard.capped_stdin` for
the line-size cap the SDK's own reader does not provide.

Scope resolution (`scope.py`) runs fresh on every `search`/`context_pack`
call against `registry.json`. `scope='current'` resolves the CLIENT's cwd:
the optional per-call `cwd` argument when given, otherwise this process's
cwd. The per-call form is what makes a shared daemon serving many sessions
work — one process per client session ("stdio per-session") leaves `cwd`
out and keeps the original behavior. Either way an unregistered directory
fails closed (see C26 in `graphify_mesh.server/__init__.py` for why this is
not an arbitrary-path cache).

Every tool call is wrapped so `ScopeResolutionError` and
`GenerationUnavailableError` degrade to an MCP tool-error result
(`isError: true` in the tool result, not a transport-level crash or a
JSON-RPC protocol error) — a client always gets a structured response it
can read, never a dead pipe.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from graphify_mesh.server import context_pack as context_pack_mod
from graphify_mesh.server import project_map as project_map_mod
from graphify_mesh.server import ranking
from graphify_mesh.server import similar as similar_mod
from graphify_mesh.server.config import ConfigError, ServerConfig
from graphify_mesh.server.embed_query import make_embed_query_fn
from graphify_mesh.server.retrieval import Hit, rank
from graphify_mesh.server.scope import (
    ScopeResolutionError,
    load_registry_entries,
    resolve_repo_list,
    resolve_scope,
)
from graphify_mesh.server.store import Generation, GenerationStore, GenerationUnavailableError
from graphify_mesh.sync.embedding import node_line
from graphify_mesh.sync.perms import audit_config_permissions

log = logging.getLogger("graphify_mesh.server.server")

SERVER_NAME = "graphify-mesh"

DEFAULT_TOKEN_BUDGET = 2000

# Upper bound for client-supplied `token_budget`: 50x the default. Large
# enough for any legitimate evidence pack, small enough that a hostile value
# can't drive unbounded work.
MAX_TOKEN_BUDGET = 50 * DEFAULT_TOKEN_BUDGET

# Single source of truth for the `k` ceiling: the same constant
# `retrieval.rank` clamps against (retrieval.py) — validation here and the
# clamp there can never drift apart.
MAX_K = ranking.MAX_K


class ToolError(RuntimeError):
    """Any tool-execution failure that should degrade to an MCP
    `isError: true` result rather than a JSON-RPC protocol error or a
    crashed process."""


def _validate_k(arguments: dict) -> int:
    """Client-supplied `k` must be a real int (bools excluded) in
    [1, MAX_K]. Anything else is a ToolError, never a crash or a silent
    coercion."""
    k = arguments.get("k", ranking.DEFAULT_K)
    if isinstance(k, bool) or not isinstance(k, int):
        raise ToolError(f"'k' must be an integer between 1 and {MAX_K}")
    if k < 1 or k > MAX_K:
        raise ToolError(f"'k' must be between 1 and {MAX_K}, got {k}")
    return k


def _validate_token_budget(arguments: dict) -> int:
    """Client-supplied `token_budget` must be a real int (bools excluded)
    in [1, MAX_TOKEN_BUDGET]."""
    budget = arguments.get("token_budget", DEFAULT_TOKEN_BUDGET)
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise ToolError(f"'token_budget' must be an integer between 1 and {MAX_TOKEN_BUDGET}")
    if budget < 1 or budget > MAX_TOKEN_BUDGET:
        raise ToolError(f"'token_budget' must be between 1 and {MAX_TOKEN_BUDGET}, got {budget}")
    return budget


def _validate_str(arguments: dict, key: str, default: str = "") -> str:
    """Client-supplied string argument: absent -> default; present but not a
    string -> ToolError (never an AttributeError deep inside a retriever)."""
    value = arguments.get(key, default)
    if not isinstance(value, str):
        raise ToolError(f"'{key}' must be a string")
    return value


def _validate_scope(arguments: dict) -> str | None:
    """`scope` is optional; when present it must be a string — the actual
    current|all|repo:<id> grammar is enforced by `scope.resolve_scope`."""
    scope = arguments.get("scope")
    if scope is not None and not isinstance(scope, str):
        raise ToolError("'scope' must be a string ('current', 'all', or 'repo:<id>')")
    return scope


def _validate_cwd(arguments: dict) -> Path | None:
    """`cwd` is an optional per-call client working directory. The server is
    run as ONE shared daemon for many sessions (see module docstring), so
    `Path.cwd()` is the daemon's directory and is meaningless for
    `scope='current'`; nothing injects the caller's directory automatically,
    so passing `cwd` on every call is the caller's own contract to honor —
    one session can move between projects. It is only ever matched against
    registered roots in registry.json — an unregistered path still fails
    closed."""
    cwd = arguments.get("cwd")
    if cwd is None:
        return None
    if not isinstance(cwd, str) or not cwd:
        raise ToolError("'cwd' must be a non-empty string (an absolute client directory)")
    path = Path(cwd)
    if not path.is_absolute():
        raise ToolError("'cwd' must be an absolute path")
    return path


def _validate_repos(arguments: dict) -> list[str] | None:
    """`repos` is optional; when present it must be a list of strings — the
    repo_ids themselves are validated against the registry downstream."""
    repos = arguments.get("repos")
    if repos is None:
        return None
    if not isinstance(repos, list):
        raise ToolError("'repos' must be a list of strings (registered repo_ids)")
    if any(not isinstance(repo, str) for repo in repos):
        raise ToolError("'repos' must be a list of strings (registered repo_ids)")
    return repos


def _citation(repo: str, source_file: str, node: dict) -> str:
    line = node_line(node)
    return f"[{repo}:{source_file}:{line if line is not None else '?'}]"


def _hit_to_dict(hit: Hit, generation: Generation) -> dict:
    node = generation.node_by_id.get(hit.node_id, {})
    return {
        "key": hit.key,
        "repo": hit.repo,
        "label": hit.label,
        "source_file": hit.source_file,
        "citation": _citation(hit.repo, hit.source_file, node),
        "community_name": hit.community_name,
        "degree": hit.degree,
        "score": hit.score,
        "match_type": hit.match_type,
        "deprecated": hit.deprecated,
    }


class GraphifyMeshServer:
    def __init__(
        self, config: ServerConfig, cwd: Path | None = None, embed_query_fn: Callable | None = None
    ):
        self.config = config
        self.store = GenerationStore(config)
        self._cwd_override = cwd
        self.embed_query_fn = embed_query_fn or make_embed_query_fn()

    @property
    def cwd(self) -> Path:
        return self._cwd_override if self._cwd_override is not None else Path.cwd()

    def _scope_cwd(self, arguments: dict, scope: str | None) -> Path:
        """The client's directory for `scope='current'`: the per-call `cwd`
        argument when the caller sent one, otherwise this process's.

        The fallback holds for the stdio transport only, where one process
        serves one client session and the process directory therefore is the
        caller's. Under the shared HTTP daemon it is the daemon's own
        directory, which may itself sit inside some registered project and
        would then answer an implicit-scope call with THAT project's results
        instead of the caller's. So an HTTP call that leaves `cwd` out while
        the scope is implicit (absent, "" or "current") is refused, naming
        the argument it must send."""
        call_cwd = _validate_cwd(arguments)
        if call_cwd is not None:
            return call_cwd
        if self.config.transport == "http" and (scope is None or scope in ("", "current")):
            raise ToolError(
                "'cwd' is required on this call: the shared HTTP daemon cannot infer the "
                "caller's project directory, so scope='current' has nothing to resolve "
                "against. Pass 'cwd' (an absolute client directory) or an explicit scope "
                "('all' or 'repo:<id>')."
            )
        return self.cwd

    def _registry_entries(self):
        return load_registry_entries(self.config.registry_path)

    # --- tool implementations ------------------------------------------

    def tool_search(self, arguments: dict) -> dict:
        query = _validate_str(arguments, "q")
        k = _validate_k(arguments)
        scope = _validate_scope(arguments)
        scope_cwd = self._scope_cwd(arguments, scope)
        entries = self._registry_entries()
        try:
            decision = resolve_scope(scope, scope_cwd, entries)
        except ScopeResolutionError as exc:
            raise ToolError(str(exc)) from exc
        generation = self._generation()
        ranked = rank(query, generation, decision.repo_ids, k, self.embed_query_fn)
        return {
            "hits": [_hit_to_dict(h, generation) for h in ranked.hits],
            "degraded": ranked.degraded,
            "scope_mode": decision.mode,
        }

    def tool_cross_project(self, arguments: dict) -> dict:
        query = _validate_str(arguments, "q")
        k = _validate_k(arguments)
        repos = _validate_repos(arguments)
        entries = self._registry_entries()
        try:
            repo_filter = resolve_repo_list(repos, entries)
        except ScopeResolutionError as exc:
            raise ToolError(str(exc)) from exc
        generation = self._generation()
        ranked = rank(query, generation, repo_filter, k, self.embed_query_fn)
        return {
            "hits": [_hit_to_dict(h, generation) for h in ranked.hits],
            "degraded": ranked.degraded,
        }

    def tool_find_similar(self, arguments: dict) -> dict:
        node = _validate_str(arguments, "node")
        k = _validate_k(arguments)
        cross_repo_only = arguments.get("cross_repo_only", False)
        if not isinstance(cross_repo_only, bool):
            raise ToolError("'cross_repo_only' must be a boolean")
        generation = self._generation()
        # `similar.py` works purely off the published generation and never
        # reads the registry, so the enabled-repo set is handed down from
        # here. It has to reach candidate SELECTION, not the returned hits: a
        # post-filter can only shrink an already-truncated list, so k=1 with a
        # disabled top neighbor returned nothing instead of the enabled
        # runner-up.
        enabled = frozenset(entry.repo_id for entry in self._registry_entries() if entry.enabled)
        result = similar_mod.find_similar(
            node, generation, k, cross_repo_only, enabled_repos=enabled
        )
        return {
            "resolved": result.resolved,
            "hits": [_hit_to_dict(h, generation) for h in result.hits],
            "degraded": list(result.degraded),
        }

    def tool_project_map(self, arguments: dict) -> dict:
        repo = arguments.get("repo")
        if not isinstance(repo, str) or not repo:
            raise ToolError("'repo' must be a non-empty string (a registered repo_id)")
        # Contract: project_map serves REGISTERED, ENABLED repo_ids only. A
        # repo that is present in the current generation but has since been
        # removed from the registry, or disabled in it, fails closed like an
        # unknown one.
        registered = {entry.repo_id for entry in self._registry_entries() if entry.enabled}
        if repo not in registered:
            return {
                "resolved": False,
                "repo": repo,
                "node_count": 0,
                "community_breakdown": {},
                "top_hubs": [],
                "degraded": ["repo_not_registered"],
            }
        generation = self._generation()
        result = project_map_mod.project_map(repo, generation)
        return {
            "resolved": result.resolved,
            "repo": result.repo,
            "node_count": result.node_count,
            "community_breakdown": result.community_breakdown,
            "top_hubs": result.top_hubs,
            "degraded": result.degraded,
        }

    def tool_context_pack(self, arguments: dict) -> dict:
        goal = _validate_str(arguments, "goal")
        token_budget = _validate_token_budget(arguments)
        scope = _validate_scope(arguments)
        scope_cwd = self._scope_cwd(arguments, scope)
        entries = self._registry_entries()
        try:
            decision = resolve_scope(scope, scope_cwd, entries)
        except ScopeResolutionError as exc:
            raise ToolError(str(exc)) from exc
        generation = self._generation()
        result = context_pack_mod.build_context_pack(
            goal, generation, decision.repo_ids, entries, token_budget, self.embed_query_fn
        )
        return {
            "goal": result.goal,
            "cards": [
                {
                    "citation": c.citation,
                    "repo": c.repo,
                    "label": c.label,
                    "community_name": c.community_name,
                    "confidence": c.confidence,
                    "snippet": c.snippet,
                    "snippet_source": c.snippet_source,
                    "snippet_stale": c.snippet_stale,
                    "score": c.score,
                }
                for c in result.cards
            ],
            "truncated": result.truncated,
            "degraded": result.degraded,
        }

    def _generation(self) -> Generation:
        try:
            return self.store.generation
        except GenerationUnavailableError as exc:
            raise ToolError(str(exc)) from exc

    def _merge_store_degraded(self, result: dict) -> dict:
        """Store-level degraded markers (e.g.
        "reload_rejected_previous_generation_still_serving",
        "embeddings_generation_mismatch") are merged into EVERY tool
        response's `degraded` list — this is the single place they surface
        to clients; nothing reads them out of the manifest."""
        if not isinstance(result, dict):
            return result
        merged = set(self.store.degraded)
        merged.update(result.get("degraded", []))
        result["degraded"] = sorted(merged)
        return result

    # --- MCP wiring -------------------------------------------------------

    TOOLS: dict[str, str] = {
        "search": "tool_search",
        "cross_project": "tool_cross_project",
        "find_similar": "tool_find_similar",
        "project_map": "tool_project_map",
        "context_pack": "tool_context_pack",
    }

    def tool_schemas(self) -> list[dict]:
        return [
            {
                "name": "search",
                "description": (
                    "Hybrid lexical+vector+structural search within a scope (current project "
                    "by default). Fails closed if scope='current' can't be resolved against "
                    "registry.json."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string"},
                        "scope": {
                            "type": "string",
                            "description": "'current' (default), 'all', or 'repo:<id>'",
                        },
                        "k": {"type": "integer", "default": ranking.DEFAULT_K},
                        "cwd": {
                            "type": "string",
                            "description": (
                                "Absolute path of the project directory this call is "
                                "about, used to resolve scope='current'. Pass it on "
                                "every call — one session can move between projects. "
                                "Omit it only with an explicit scope ('all' or "
                                "'repo:<id>'); a directory outside registry.json is "
                                "refused."
                            ),
                        },
                    },
                    "required": ["q"],
                },
            },
            {
                "name": "cross_project",
                "description": (
                    "Explicit cross-repo hybrid search, optionally restricted to a repo_id list."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string"},
                        "repos": {"type": "array", "items": {"type": "string"}},
                        "k": {"type": "integer", "default": ranking.DEFAULT_K},
                    },
                    "required": ["q"],
                },
            },
            {
                "name": "find_similar",
                "description": (
                    "Cross-project (and optionally same-project) structurally/semantically "
                    "similar nodes to a given node/label."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "node": {"type": "string"},
                        "k": {"type": "integer", "default": ranking.DEFAULT_K},
                        "cross_repo_only": {"type": "boolean", "default": False},
                    },
                    "required": ["node"],
                },
            },
            {
                "name": "project_map",
                "description": (
                    "Structural overview of one registered repo in the current generation: "
                    "node count, community breakdown, top hub nodes."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {"repo": {"type": "string"}},
                    "required": ["repo"],
                },
            },
            {
                "name": "context_pack",
                "description": (
                    "Evidence cards (citations, snippets, confidence) for a goal, truncated "
                    "to a token budget without ever splitting a card."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string"},
                        "scope": {"type": "string"},
                        "token_budget": {"type": "integer", "default": DEFAULT_TOKEN_BUDGET},
                        "cwd": {
                            "type": "string",
                            "description": (
                                "Absolute path of the project directory this call is "
                                "about, used to resolve scope='current'. Pass it on "
                                "every call — one session can move between projects. "
                                "Omit it only with an explicit scope ('all' or "
                                "'repo:<id>'); a directory outside registry.json is "
                                "refused."
                            ),
                        },
                    },
                    "required": ["goal"],
                },
            },
        ]

    def call_tool(self, name: str, arguments: dict) -> dict:
        method_name = self.TOOLS.get(name)
        if method_name is None:
            return {
                "content": [{"type": "text", "text": f"unknown tool: {name!r}"}],
                "isError": True,
            }
        method = getattr(self, method_name)
        try:
            result = method(arguments or {})
            result = self._merge_store_degraded(result)
            return {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False}
        except ToolError as exc:
            return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
        except Exception:
            # Includes non-JSON-serializable results from json.dumps above.
            # Traceback to stderr only; the client sees a generic message —
            # never exception text, paths, or stack frames.
            log.exception("tool %r raised an unexpected exception", name)
            return {"content": [{"type": "text", "text": "internal error"}], "isError": True}

    # --- JSON-RPC method dispatch ------------------------------------------


def build_server() -> GraphifyMeshServer:
    config = ServerConfig.from_env()
    return GraphifyMeshServer(config)


def serve_stdio(mesh: GraphifyMeshServer) -> None:
    """Blocking stdio transport. Returns on stdin EOF — the clean-exit
    contract that keeps closed sessions from leaking resident processes."""
    import anyio
    from mcp.server.stdio import stdio_server

    from graphify_mesh.server.sdk_app import build_sdk_server
    from graphify_mesh.server.stdio_guard import capped_stdin, wire_stdout

    sdk = build_sdk_server(mesh)

    async def _run() -> None:
        # ONE writer for both frame sources. `stdio_guard`'s error frames and
        # the SDK's responses go through the same Python object, so its
        # buffer lock serializes them; two separate `dup()`s of fd 1 left
        # that resting on POSIX PIPE_BUF and a response over 4096 bytes could
        # have an error frame spliced into it. `wire_stdout` also takes over
        # the fd-1-to-stderr diversion the SDK skips once `stdout` is passed
        # explicitly.
        with wire_stdout() as writer:
            async with stdio_server(
                stdin=capped_stdin(out_stream=writer),
                stdout=anyio.wrap_file(writer),
            ) as (read_stream, write_stream):
                await sdk.run(read_stream, write_stream, sdk.create_initialization_options())

    anyio.run(_run)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    # Deferred: http_app imports GraphifyMeshServer from this module, so a
    # module-level import here would deadlock whichever module is imported
    # first (see tests/server/test_http_app.py, which imports http_app
    # before server). Safe at call time: by then both modules are fully
    # loaded.
    from graphify_mesh.server import http_app

    parser = argparse.ArgumentParser(
        prog="graphify-mesh-server",
        description=(
            "MCP server for the merged global graph. Default transport is stdio "
            "(one process per client session). --transport http runs one shared "
            "daemon for every local agent; it requires GRAPHIFY_MESH_HTTP_TOKEN."
        ),
    )
    parser.add_argument("--transport", choices=("stdio", "http"), default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--path", default=None)
    parser.add_argument("--allow-public-bind", action="store_true", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        config = ServerConfig.from_env(
            transport=args.transport,
            http_host=args.host,
            http_port=args.port,
            http_path=args.path,
            allow_public_bind=args.allow_public_bind,
        )
    except ConfigError as exc:
        print(f"graphify-mesh-server: {exc}", file=sys.stderr)
        return 2

    # The server trusts the registry for scope authorization, so a
    # group-writable registry is as much its problem as the sync engine's.
    # The env-file half is deliberately skipped: the daemon's own secret
    # lives in a file this package cannot locate.
    audit_config_permissions(config.registry_path, check_env_files=False)

    mesh = GraphifyMeshServer(config)
    if config.transport == "http":
        http_app.serve_http(mesh, config)
    else:
        serve_stdio(mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
