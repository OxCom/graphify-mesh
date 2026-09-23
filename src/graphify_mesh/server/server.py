"""`graphify-mesh` stdio MCP server (WS5 deliverable 2): wires the 6 hybrid/
cross-project/evidence/traversal tools (`search`, `cross_project`,
`find_similar`, `project_map`, `context_pack`, `neighbors`) onto the MCP SDK's stdio transport
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
from graphify_mesh.server import traverse as traverse_mod
from graphify_mesh.server.citation import citation
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


def _validate_relations(arguments: dict) -> list[str]:
    """`relation` is one relation name or a list of 1..MAX_RELATIONS of them,
    each a non-empty string of at most MAX_RELATION_LENGTH characters.
    Returned sorted and deduplicated so the traversal and its payload are
    order-independent."""
    value = arguments.get("relation")
    names = [value] if isinstance(value, str) else value
    limit = traverse_mod.MAX_RELATIONS
    length = traverse_mod.MAX_RELATION_LENGTH
    message = (
        f"'relation' must be a non-empty string or a list of 1..{limit} non-empty strings "
        f"of at most {length} characters"
    )
    if not isinstance(names, list) or not 1 <= len(names) <= limit:
        raise ToolError(message)
    if any(not isinstance(name, str) or not name or len(name) > length for name in names):
        raise ToolError(message)
    return sorted(set(names))


def _validate_direction(arguments: dict) -> str:
    direction = arguments.get("direction")
    if direction not in traverse_mod.DIRECTIONS:
        raise ToolError(f"'direction' must be one of {', '.join(traverse_mod.DIRECTIONS)}")
    return direction


def _validate_depth(arguments: dict) -> int:
    """Real int (bools excluded) in [1, traverse.MAX_DEPTH]."""
    depth = arguments.get("depth", traverse_mod.DEFAULT_DEPTH)
    maximum = traverse_mod.MAX_DEPTH
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= maximum:
        raise ToolError(f"'depth' must be an integer between 1 and {maximum}")
    return depth


def _hit_to_dict(hit: Hit, generation: Generation) -> dict:
    node = generation.node_by_id.get(hit.node_id, {})
    return {
        "key": hit.key,
        "repo": hit.repo,
        "label": hit.label,
        "source_file": hit.source_file,
        "citation": citation(hit.repo, hit.source_file, node),
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
                "communities_total": 0,
                "communities_omitted": 0,
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
            "communities_total": result.communities_total,
            "communities_omitted": result.communities_omitted,
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

    def tool_neighbors(self, arguments: dict) -> dict:
        node = arguments.get("node")
        if not isinstance(node, str) or not node:
            raise ToolError("'node' must be a non-empty string (a key, node id, or label)")
        repo = arguments.get("repo")
        if not isinstance(repo, str) or not repo:
            raise ToolError("'repo' must be a non-empty string (a registered repo_id)")
        relations = _validate_relations(arguments)
        direction = _validate_direction(arguments)
        depth = _validate_depth(arguments)
        include_inferred = arguments.get("include_inferred", False)
        if not isinstance(include_inferred, bool):
            raise ToolError("'include_inferred' must be a boolean")
        # Same registered-and-enabled gate as project_map, checked before
        # the generation is touched.
        registered = {entry.repo_id for entry in self._registry_entries() if entry.enabled}
        if repo not in registered:
            return traverse_mod.NeighborsResult(
                resolved=False,
                repo=repo,
                relations=relations,
                direction=direction,
                depth=depth,
                include_inferred=include_inferred,
                degraded=["repo_not_registered"],
            ).as_dict()
        generation = self._generation()
        unknown = [name for name in relations if name not in generation.relations]
        if unknown:
            raise ToolError(
                f"unknown relation(s) {', '.join(unknown)}; relations in the current "
                f"generation: {', '.join(sorted(generation.relations)) or '(none)'}"
            )
        return traverse_mod.neighbors(
            node, repo, relations, direction, depth, include_inferred, generation
        ).as_dict()

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
        "neighbors": "tool_neighbors",
    }

    def tool_schemas(self) -> list[dict]:
        return [
            {
                "name": "search",
                "description": (
                    "Find WHERE something lives in an indexed repository when you do not "
                    "already know the file: a mechanism, a responsibility, an entry point, "
                    "the code behind a behaviour. Answers 'roughly where does X happen' and "
                    "'what handles Y' across a whole repo at once, including files a keyword "
                    "search misses because the name never appears in them. Returns candidate "
                    "places with repo, path, line and label — verify them in the source. Not "
                    "for rosters, counts or literal config values. Scope defaults to the "
                    "current project and fails closed outside a registered repo root."
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
                    "Answer a question that spans repositories you do not have open: which "
                    "projects use a dependency, who consumes an endpoint or queue, where a "
                    "service is deployed, what else implements a pattern. Searches every "
                    "indexed repository at once, so it reaches code outside the working "
                    "directory that no local search can see. Returns candidate repositories "
                    "and places to open, not a final roster."
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
                    "Find an existing implementation to copy or stay consistent with: 'where "
                    "else is this done', 'has someone already solved this', 'what is the "
                    "analogous handler in another service'. Takes a node or label you already "
                    "have and returns structurally and semantically similar code elsewhere, "
                    "across projects by default."
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
                    "Get oriented in an unfamiliar repository before reading it: what the "
                    "main parts are, which nodes everything depends on, how big it is. Use "
                    "when you have just been pointed at a repo you do not know and need its "
                    "shape. Explicit request only — it is an overview, never evidence for a "
                    "specific claim."
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
                    "Use when a search has stalled: several rounds of grep and reading have "
                    "not reduced the task to named files and symbols. Takes the task goal and "
                    "returns a budgeted set of evidence cards — citation, snippet and "
                    "confidence per card — so one call replaces the next several searches. "
                    "Works within one repo or across all of them."
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
            {
                "name": "neighbors",
                "description": (
                    "Exact, COMPLETE traversal over chosen relation types inside one "
                    "registered repo: every node reachable over the indexed edges of those "
                    "relations within `depth`, not a ranked top-k like search. Use it for "
                    "'all subclasses of X' (relation=inherits, direction=in) or 'everything "
                    "that imports M' (relation=imports, direction=in). `complete` is false "
                    "only when the node cap truncated the result. calls/indirect_call edges "
                    "are incomplete (calls through typed properties / DI are not extracted), "
                    "so a missing call edge is not evidence of no call; those results come "
                    "back with reliability='partial'."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "node": {
                            "type": "string",
                            "description": (
                                "Durable key, graph node id, or label (exact, then "
                                "case-insensitive, then a bare method name as '.name()'). "
                                "Several label matches all become seeds."
                            ),
                        },
                        "repo": {"type": "string", "description": "Registered repo_id."},
                        "relation": {
                            "anyOf": [
                                {"type": "string"},
                                {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1,
                                    "maxItems": traverse_mod.MAX_RELATIONS,
                                },
                            ],
                            "description": (
                                "Relation name(s), e.g. inherits, implements, imports, "
                                "calls. An unknown name is an error listing the known ones."
                            ),
                        },
                        "direction": {
                            "type": "string",
                            "enum": list(traverse_mod.DIRECTIONS),
                            "description": (
                                "'in': edges whose TARGET is the current node (for "
                                "inherits: its subclasses; for imports: its importers). "
                                "'out': edges whose SOURCE is the current node (for "
                                "inherits: its parents). 'both': either."
                            ),
                        },
                        "depth": {
                            "type": "integer",
                            "default": traverse_mod.DEFAULT_DEPTH,
                            "minimum": 1,
                            "maximum": traverse_mod.MAX_DEPTH,
                        },
                        "include_inferred": {
                            "type": "boolean",
                            "default": False,
                            "description": "Also follow INFERRED-confidence edges.",
                        },
                    },
                    "required": ["node", "repo", "relation", "direction"],
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
