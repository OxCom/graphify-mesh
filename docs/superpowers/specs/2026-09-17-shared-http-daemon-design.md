# Shared HTTP daemon for the graphify-mesh MCP server

Date: 2026-09-17
Status: approved design, not yet implemented

## Problem

`graphify-mesh-server` speaks stdio and states "one process per client session". Every client
that spawns it by command gets a private backend that loads the whole merged generation —
graph, overlay, lexical index and embedding shards — into its own memory.

On this machine the duplication is already worked around outside the package: `~/.claude.json`
points Claude Code at `~/.claude/mcp-proxies/graphify-mesh-filter.mjs`, a Node stdio shim that
forwards to one shared daemon (`claude-mcp-graphify-mesh.service`, running
`mcp-broker.mjs` around the stdio binary). That keeps memory bounded but leaves a per-session
Node process in front of every window, and it only works for clients wired to that shim.

The package itself must offer the shared mode, so a client can reach one instance by URL with
no shim of any kind.

## Goal

One `graphify-mesh` process serving every local agent, reachable by URL, with the existing
per-session stdio mode kept intact and remaining the default.

## Non-goals

- Codex's configuration. It keeps spawning its own stdio backend; parked by explicit decision.
- A tool allowlist. Considered and dropped: the server exposes 5 tools and all 5 are needed for
  retrieval, so a filter would ship empty. Revisit if the tool count grows.
- Remote or multi-machine access. The daemon binds loopback.
- Server-initiated notifications (for example "a new generation was published"). Stateless
  transport has nowhere to push them; adding them later means moving to stateful sessions.

## Decisions

| Area | Decision |
|---|---|
| Shared transport | Streamable HTTP via the `mcp` SDK, with starlette and uvicorn |
| Own JSON-RPC transport | `server/protocol.py` is removed; stdio also runs through the SDK |
| Dependency status | `mcp`, `starlette`, `uvicorn` become required, not an extra |
| Mode selection | One console script. No flags means stdio; `--transport http` starts the daemon |
| Session state | Stateless, `json_response=True`. No SSE, no session reaper |
| `cwd` | Per call, supplied by the model. A PreToolUse hook fills it only when the field is absent |
| Concurrency | Reads run in parallel; generation reload is serialized |
| Auth | Bearer token required in HTTP mode; the daemon refuses to start without one |
| Bind | `127.0.0.1` with Host/Origin checking |
| Default port | 19744 |
| Port and token home | `~/.config/claude-mcp/graphify-mesh.env` |
| Node broker for this backend | Retired once the HTTP path is tested |

## Architecture

### Mode selection

`graphify_mesh.server.server:main` grows an argument parser:

```text
graphify-mesh-server                                  # stdio (default, unchanged behavior)
graphify-mesh-server --transport http                 # shared daemon
    [--host 127.0.0.1] [--port 19744] [--path /mcp]
```

Every flag has an environment equivalent, resolved the way the rest of this package resolves
configuration — flag wins over environment, environment wins over default:

| Variable | Flag | Default |
|---|---|---|
| `GRAPHIFY_MESH_TRANSPORT` | `--transport` | `stdio` |
| `GRAPHIFY_MESH_HTTP_HOST` | `--host` | `127.0.0.1` |
| `GRAPHIFY_MESH_HTTP_PORT` | `--port` | `19744` |
| `GRAPHIFY_MESH_HTTP_PATH` | `--path` | `/mcp` |
| `GRAPHIFY_MESH_HTTP_TOKEN` | — | none; required for `--transport http` |

The new fields join `ServerConfig` (`server/config.py`) so tests construct them directly and no
machine-specific value is hardcoded.

### Tool wiring

`GraphifyMeshServer.tool_schemas` and `GraphifyMeshServer.call_tool` stay the single source of
truth for what the 5 tools are and what they do. Both transports register the SDK's low-level
`Server` against those two methods, so the tool surface cannot drift between modes. The
`ToolError` to `isError: true` degradation that `call_tool` already performs is preserved
verbatim: a scope or generation failure must reach the client as a readable tool result, never
as a transport error or a dead connection.

### Port 19744

Checked on this machine: absent from `/etc/services`, not listening, and below the ephemeral
range (`/proc/sys/net/ipv4/ip_local_port_range` starts at 32768), so a random client port
cannot take it. Ports in use here at design time: 22, 53, 80, 443, 631, 3111-3113, 3306, 6379,
8700-8705, 8900-8904, 11434, 11435, 24282-24286.

### `cwd` and scope resolution

`scope='current'` resolves against the caller's directory, and a shared daemon's own directory
is its systemd unit's `$HOME`, which matches no registered root. The `cwd` tool argument
already exists for this (`server/server.py:110`, schemas at `server/server.py:325`) and is only
ever matched against registered roots in `registry.json`, so an unregistered path still fails
closed.

Two changes:

1. The model supplies `cwd` per call. One session moves between projects, so a value pinned at
   session start would silently answer for the wrong project. The tool descriptions stop saying
   "injected by the session proxy" and state that the caller passes its absolute project
   directory.
2. A PreToolUse hook on `mcp__graphify-mesh__*` in Claude Code fills `cwd` with
   `$CLAUDE_PROJECT_DIR` **only when the argument is missing**. A value the model supplied is
   never overwritten, or switching projects mid-session would break again. Claude Code 2.1.274
   supports this: `updatedInput` is present in the binary, and two hooks already match that tool
   prefix (`~/.claude/settings.json:158`, `:234`).

A client without that hook that omits `cwd` gets the existing fail-closed error. The error text
must name the fix: pass an absolute `cwd`, or use `scope='repo:<id>'`.

### Concurrency

Requirement: reads run in parallel, state changes queue.

Today the two are fused. `GenerationStore.generation` (`server/store.py:605`) calls
`ensure_fresh()` on every read, and `ensure_fresh` (`:453`) mutates `_generation`,
`_manifest_mtime`, `_current_target` and `degraded`. The split:

- `ensure_fresh` keeps doing the cheap `_stat_signature()` comparison under a shared read lock.
  Unchanged signature returns immediately, which is the common case.
- A changed signature takes the exclusive write lock, re-checks the signature under it (another
  thread may have reloaded already), then runs `_try_reload`. `_try_reload` already loads from
  the captured realpath rather than the live `current` symlink, so a publish flipping `current`
  mid-load cannot mix two generations.
- The returned `Generation` is treated as immutable. `build_indexes()` runs once during load,
  before the object is published to readers.

Two module-level caches on the read path also need guarding. Neither corrupts data under the
GIL, but both can build twice and waste the work:

- `server/scope.py:52` `_registry_cache`
- `server/similar.py:49` `_per_generation_cache`, used at `:98` and `:122`

Each gets its own lock around the build-and-store step, with the fast hit path lock-free.

The rest of the read path was audited and writes only to local variables: `retrieval.py`,
`ranking.py`, `context_pack.py`, `project_map.py`, `lexical_read.py`.

### Security

The current shared daemon is reachable only through a unix socket in a `0700` runtime directory.
A loopback TCP port is reachable by every local user and process on the machine, and what it
serves is the merged graph of every repository in `registry.json`. The HTTP mode therefore:

- **Requires a token.** No `GRAPHIFY_MESH_HTTP_TOKEN` means the daemon logs the reason and exits
  nonzero. An empty or whitespace-only value counts as absent, so a blank line in the env file
  cannot silently disable authentication. Comparison is constant-time. stdio mode needs no
  token and ignores the variable.
- **Binds loopback** and validates `Host`/`Origin` against the bind address plus the localhost
  aliases, which is the MCP spec's DNS-rebinding requirement.
- **Refuses a wildcard bind by default.** `--host 0.0.0.0` requires an explicit opt-in flag, and
  the daemon still requires the token.
- **Never logs the token, headers or request bodies.** Request logging records method name and
  outcome only.

Token and port live in `~/.config/claude-mcp/graphify-mesh.env`, which the systemd unit already
reads through `EnvironmentFile`. They are never committed and never written into the unit file,
where `systemctl cat` would expose the token to any local user.

### What the removal of `protocol.py` must not lose

`server/protocol.py` carries four behaviors that tests in `tests/server/test_protocol.py`,
`tests/server/test_hardening.py` and `tests/server/test_stdio_e2e.py` pin:

1. Unparseable JSON answers `-32700` with `id: null` instead of hanging the client.
2. A payload that is not a single JSON-RPC object, including a batch array, answers `-32600`.
3. A line over `MAX_LINE_BYTES` is drained rather than buffered without bound.
4. A handler exception logs the traceback to stderr and returns a generic `-32603` — never
   exception text, paths or stack frames — and notifications get no response at all.

Unknown until checked against the installed SDK: which of these it enforces itself. Each one
the SDK does not enforce keeps a guard in this package, and the tests are rewritten against the
new entry point rather than deleted. A protection dropped because the transport changed is a
regression, not a simplification.

## Files

In this repository:

- `src/graphify_mesh/server/server.py` — argument parser, transport dispatch, SDK registration,
  tool-description wording for `cwd`.
- `src/graphify_mesh/server/http_app.py` (new) — the ASGI app: session manager, token
  middleware, Host/Origin settings. Split from the blocking `uvicorn.run` call so tests can
  drive it in process.
- `src/graphify_mesh/server/config.py` — transport, host, port, path, token fields.
- `src/graphify_mesh/server/store.py` — read/write lock split.
- `src/graphify_mesh/server/scope.py`, `similar.py` — cache locks.
- `src/graphify_mesh/server/protocol.py` — removed.
- `pyproject.toml` — `mcp`, `starlette`, `uvicorn` as required dependencies.
- `Dockerfile.test` — rebuilt for the new dependencies.
- `tests/server/`, `tests/hardening/` — see Testing.
- `docs/mcp-server.md`, `docs/configuration.md`, `docs/architecture.md`, `docs/setup.md`,
  `CLAUDE.md` — the "one process per client session" and "stdlib only, no mcp SDK" statements
  stop being true and must be rewritten, not patched around.

Outside the repository, in order, after the tests pass:

- `~/.config/claude-mcp/graphify-mesh.env` — port and generated token.
- `~/.config/systemd/user/claude-mcp-graphify-mesh.service` — `ExecStart` becomes
  `graphify-mesh-server --transport http`, dropping node from the path.
- `~/.claude/hooks/` — the new `cwd`-filling hook, plus its `PreToolUse` entry in
  `~/.claude/settings.json`.
- `~/.claude.json` — the `graphify-mesh` entry becomes `type: "http"` with the URL and the
  `Authorization` header.
- `~/.claude/mcp-proxies/graphify-mesh-filter.mjs` — deleted. The shared `mcp-proxies/lib/`
  files stay: swarmvault, serena and agentmemory still use them.

The daemon runs from a host install, so the host needs `mcp`, `starlette` and `uvicorn`
installed for the runtime (not for tests, which stay in Docker). That install command goes to
the operator to run.

## Testing

- Transport parity: the same `tools/list` and `tools/call` request produces the same result over
  stdio and over HTTP.
- HTTP auth: no token refuses to start; wrong token gets 401; correct token succeeds; the token
  never appears in logs.
- Host/Origin rejection for a foreign `Host` header.
- `cwd`: present resolves `scope='current'`; absent fails closed with an error naming the fix;
  a `cwd` outside `registry.json` fails closed.
- Concurrency: parallel reads proceed while a reload is in flight and no reader observes a
  half-built `Generation`; a signature change reloads exactly once under N concurrent readers.
- The four `protocol.py` protections, re-pinned against the SDK entry point.
- Everything runs in the Docker image, per the repository's standing rule against host
  installs for test dependencies.

## Migration and rollback

Order matters: the repository work and its tests come first, the host switch second, and the
switch is one commit's worth of configuration.

Rollback is restoring two files — the unit's `ExecStart` and the `.claude.json` entry — plus
`graphify-mesh-filter.mjs` from git history. The stdio mode stays shipped and default, so the
old broker path keeps working for as long as those files exist.

Restarting the unit drops graphify-mesh tools from every open session for a few seconds.

## Risks

1. **The SDK's guarantees are assumed, not yet read.** The four protections above and the exact
   stateless-mode response shape need checking against the installed version before the tests
   are written.
2. **`mcp` becomes a required dependency** of a package that was deliberately stdlib-only on the
   server side. This is a real trade accepted for one transport implementation instead of two.
3. **The `cwd` hook is Claude Code only.** Any other client that omits `cwd` sees fail-closed
   errors until it passes the argument itself.
4. **A loopback port is a wider attack surface than a `0700` unix socket**, mitigated by the
   mandatory token and the Host/Origin check, not eliminated.
