# MCP server

`graphify-mesh-server` is one console script with two transports, both
registered once against the `mcp` SDK's low-level `Server`
(`server/sdk_app.py`) from the same `tool_schemas()` / `call_tool()` pair, so
the 5-tool surface cannot drift between modes. Both the `list_tools` and
`call_tool` handlers catch any unexpected exception, log the traceback to
stderr, and return a generic error with no exception text to the client.

- **stdio** (default, no flags): one process per client session, unchanged
  behavior. Exits cleanly the moment the client closes stdin.
- **`--transport http`**: one shared daemon serving every local agent over
  streamable HTTP, reachable by URL instead of a spawned process per session.

The advertised server name is **`graphify-mesh`**.

## Transport selection

| Flag | Env | Default | Meaning |
|------|-----|---------|---------|
| `--transport {stdio,http}` | `GRAPHIFY_MESH_TRANSPORT` | `stdio` | Which transport to run. |
| `--host HOST` | `GRAPHIFY_MESH_HTTP_HOST` | `127.0.0.1` | HTTP bind address. |
| `--port PORT` | `GRAPHIFY_MESH_HTTP_PORT` | `19744` | HTTP bind port. |
| `--path PATH` | `GRAPHIFY_MESH_HTTP_PATH` | `/mcp` | HTTP mount path. |
| `--allow-public-bind` | `GRAPHIFY_MESH_ALLOW_PUBLIC_BIND` | off | Required to bind any non-loopback host, wildcards and concrete interface addresses alike. See `configuration.md` for what counts as loopback. |
| — | `GRAPHIFY_MESH_HTTP_TOKEN` | none | Bearer token; required when `transport=http`, minimum 32 characters. |

A flag beats its environment variable, which beats the default. stdio mode
ignores the HTTP variables and the token entirely.

## Protocol

**stdio**: newline-delimited JSON-RPC 2.0 objects on stdin/stdout. Methods:
`initialize`, `tools/list`, `tools/call`. The client starts the process,
exchanges messages, then closes stdin; the server exits with code 0.

`server/stdio_guard.py` restores three protections the SDK's own stdin reader
does not provide: a line over 4 MiB (`server/frames.py:MAX_MESSAGE_BYTES`) is
drained in bounded chunks rather than buffered whole; a line that is not
parseable JSON, or not a single JSON object (including a batch array), is
answered with the exact JSON-RPC `-32700` or `-32600` before it reaches the
SDK; and a JSON object that is not a legal JSON-RPC 2.0 envelope is answered
`-32602` for a params-shape error and `-32600` otherwise, correlated with the
request id when the frame carries a legal one. Without that third check such
a frame reached the SDK's pydantic validation, which raises instead of
answering the request, leaving the client to wait for its own timeout. An
unrecognized JSON-RPC method answers `-32601`.

Minimal handshake:

```json
{"jsonrpc": "2.0", "id": 1, "method": "initialize"}
{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
{"jsonrpc": "2.0", "id": 3, "method": "tools/call",
 "params": {"name": "search", "arguments": {"q": "auth guard", "scope": "current"}}}
```

**HTTP**: streamable HTTP at `http://<host>:<port><path>` (default
`http://127.0.0.1:19744/mcp`), stateless (`json_response=True`,
`stateless=True`, no sessions, no SSE, no `Mcp-Session-Id`). Every request
needs `Authorization: Bearer <token>`; a missing or wrong token gets `401`
with body exactly `{"error": "unauthorized"}`. The token is never logged.
`Host`/`Origin` are validated against the bind address and the localhost
aliases (MCP's DNS-rebinding protection) when the bind is a loopback one; a
non-loopback bind accepts any `Host`, since it is reachable under many names
and the bearer token is the real gate. Both the bind permission and this
branch decide on the same `is_loopback_bind` rule, so they cannot disagree.

Past the token gate, `http_app.py`'s frame guard applies the same two rules
stdio applies, so a client sees one protocol whichever transport it speaks: a
request body is capped at `MAX_MESSAGE_BYTES` counted on bytes actually
received (a chunked client has no `Content-Length` to trust), answered `413`
with `{"error": "request body too large"}` and the rest of the body never
read; and a malformed frame gets the generic `-32700`/`-32600`/`-32602` error
with HTTP `400`, with the parse or validation detail logged instead of
returned. The SDK's own errors quote the offending input, which is what the
guard replaces. The same cap is passed to the session manager as
`max_request_body_size`, so both layers state one number.

### Inbound memory

A body being read is held whole in this one shared process, so the guard
bounds how many it reads at once: an `anyio.Semaphore` of
`http_app.BODY_BUFFER_SLOTS` (8), taken before the first byte and handed back
the moment the body reaches the layer below. That caps the guard's own share
of request bodies at 8 x 4 MiB = **32 MiB**.

Reaching the bound queues, it never rejects. A request waits for a slot,
typically for microseconds, and an operator sees latency on a burst rather
than an error; a client sitting on an idle keep-alive connection holds no slot
and is unaffected. uvicorn's `limit_concurrency` is deliberately NOT used: it
answers HTTP `503` once the number of open **connections** reaches the limit,
which on a daemon whose whole purpose is many long-lived local clients would
reject a client with nothing in flight.

Below the guard the session manager keeps its own buffer and copy of each body
it is handed, and both layers parse the JSON. The parsed Python objects are
the larger cost and are not a fixed multiple of the byte cap: a 4 MiB
document of many small values expands several times over as Python objects,
while a 4 MiB single string barely expands. The guard drops its own parsed
object and its buffer before dispatching downstream, so the copies below it
are the ones that live for the call. Nothing in this package bounds how many
requests are in that state at once, so total inbound memory is not a single
number — `BODY_BUFFER_SLOTS` x 4 MiB is the bound on the guard's share, not on
the daemon's.

If no consistent generation has been published yet, tool calls fail closed
(an error result) on either transport, rather than serving a partial or stale
graph.

## Concurrency (HTTP mode)

Reads run in parallel; a generation reload takes an exclusive lock
(`server/rwlock.py`, writer-preferring, so a steady stream of reads cannot
postpone a reload indefinitely). The two read-path caches — the registry
cache in `server/scope.py` and the per-generation index cache in
`server/similar.py` — are each guarded by their own lock, with the fast hit
path lock-free.

## The 6 tools

### `search`
Hybrid lexical + vector + structural search within a **scope**. Scope defaults
to the current project and only widens when asked. Fails closed if
`scope="current"` cannot be resolved against `registry.json`.

| Arg | Type | Default | Notes |
|-----|------|---------|-------|
| `q` | string | — (required) | Query text. |
| `scope` | string | `current` | `current`, `all`, or `repo:<id>`. `all` means every **enabled** repo in `registry.json`, not "no filter": a repo disabled since the generation was published is not served. With no enabled repo at all, `all` raises rather than searching everything. |
| `k` | integer | ranking default | Max results. |
| `cwd` | string | — | "Absolute path of the project directory this call is about, used to resolve scope='current'. Pass it on every call — one session can move between projects. Omit it only with an explicit scope ('all' or 'repo:<id>'); a directory outside registry.json is refused." |

Each `search` hit carries a `match_type`: `exact` when the whole query is an
exact alias of the node (FQCN, label, bare method name, file basename), then
`anchor`, then `fused` for the ranked hybrid results. A query token whose alias
(`t` or `.t()`) names 1 to 10 nodes makes those nodes anchor candidates, each
scored by the idf of the other query tokens found in its own fields and in its
depth-1 EXTRACTED neighbours. At most two candidates are pinned, each only if it
scores at least 10 and at least twice the next candidate; otherwise no `anchor`
hit appears and ranking is unchanged. The idf is corpus-wide, not per scope.
`cross_project` and `context_pack` rank through the same code, so anchors apply
to them as well.

Hits from `search`, `cross_project` and `find_similar`, and `context_pack`
cards, carry no `community_name`: the labels proved unreliable and cost payload
without adding evidence.

### `cross_project`
Explicit cross-repo hybrid search, optionally restricted to a list of repos.

| Arg | Type | Default | Notes |
|-----|------|---------|-------|
| `q` | string | — (required) | Query text. |
| `repos` | string[] | all enabled repos | Restrict to these `repo_id`s. Omitted means every enabled repo, never an unfiltered search; an unknown or disabled id is a hard error, and a registry with no enabled repo at all is an error too rather than an empty result. |
| `k` | integer | ranking default | Max results. |

### `find_similar`
Structurally / semantically similar nodes to a given node or label; can be
restricted to cross-repo matches only.

| Arg | Type | Default | Notes |
|-----|------|---------|-------|
| `node` | string | — (required) | Node id or label. |
| `k` | integer | ranking default | Max results. |
| `cross_repo_only` | boolean | `false` | Exclude same-repo matches. |

### `project_map`
Structural overview of one **registered** repo in the current generation: node
count, community breakdown, top hub nodes. Takes a `repo_id` that must resolve
against `registry.json` — never an arbitrary on-disk path.

`community_breakdown` keeps only communities of 2 or more nodes, and at most
the 25 largest (`COMMUNITY_MIN_SIZE`, `COMMUNITY_BREAKDOWN_LIMIT` in
`server/project_map.py`). `communities_total` and `communities_omitted` report
what the trim dropped. Community labels are orientation hints, not evidence:
on cem.hub, `UserRoleEnum`, `AbstractCronJob` and `LockWeeks` share one
unrelated label, and the untrimmed list was 128 entries, mostly singletons.
`top_hubs` is unchanged.

| Arg | Type | Notes |
|-----|------|-------|
| `repo` | string (required) | A registered `repo_id`. |

### `neighbors`
Exact traversal over chosen relation types inside one registered repo. It
returns **every** node reachable over the indexed edges of those relations
within `depth`, not a ranked top-k. This is the tool for roster questions that
have a structural answer: all subclasses of a base class, all implementors of
an interface, all importers of a module. On cem.hub, subclasses of
`AbstractCronJob` return the same 19 classes as a grep for `extends`.

| Arg | Type | Default | Notes |
|-----|------|---------|-------|
| `node` | string | — (required) | Durable key, graph node id, or label. Labels match exactly, then case-insensitively, then as a bare method name (`reminderSubmitWeek` matches `.reminderSubmitWeek()`). Several matches all become seeds. |
| `repo` | string | — (required) | A registered, enabled `repo_id`. The traversal never leaves it. |
| `relation` | string or string[] | — (required) | 1 to 16 relation names, e.g. `inherits`, `implements`, `imports`, `calls`. A name absent from the generation is an error that lists the names present. |
| `direction` | string | — (required) | `in`: edges whose target is the current node (subclasses for `inherits`, importers for `imports`). `out`: edges whose source is the current node (parents for `inherits`). `both`: either. |
| `depth` | integer | `1` | 1 to 16 hops. |
| `include_inferred` | boolean | `false` | Also follow non-EXTRACTED (INFERRED and AMBIGUOUS) edges. By default only EXTRACTED edges are followed. |

The payload lists `seeds` and `nodes`. Each node carries `depth`, a citation,
and `via`, the edges that reached it from the previous hop. `complete` is true
unless the 2000-node cap truncated the result (`truncated`). The cap counts
seeds and nodes together: a label matching more than 2000 nodes returns the
first 2000 seeds by key, no nodes, and `truncated: true`.
`frontier_exhausted` says the walk ran out of new nodes before the depth limit.
Nodes without a durable key are skipped and counted in `skipped_unkeyed`.

`reliability` is `exact` unless the request followed `calls`,
`indirect_call`, or non-EXTRACTED edges; then it is `partial` and `notes` says why.
Call edges are incomplete: the extractor does not resolve method calls
through typed properties, which is how dependency injection calls look. On
cem.hub only 554 of 2797 PHP methods have any incoming call edge. A missing
call edge is not evidence that no call exists.

### `context_pack`
Evidence cards (citations, snippets, confidence) for a goal, truncated to a
token budget without ever splitting a card mid-way.

| Arg | Type | Default | Notes |
|-----|------|---------|-------|
| `goal` | string | — (required) | What you are trying to do. |
| `scope` | string | — | Same scope grammar as `search`. |
| `token_budget` | integer | server default | Hard cap on returned card volume. |
| `cwd` | string | — | Same contract as `search`'s `cwd`: "Absolute path of the project directory this call is about, used to resolve scope='current'. Pass it on every call — one session can move between projects. Omit it only with an explicit scope ('all' or 'repo:<id>'); a directory outside registry.json is refused." |

## Registering with a client

Stdio (unchanged, one process per client session):

```json
{
  "mcpServers": {
    "graphify-mesh": {
      "command": "graphify-mesh-server",
      "env": { "GRAPHIFY_MESH_ROOT": "/path/to/your/workspace/graph-mesh" }
    }
  }
}
```

HTTP (one shared daemon, started separately — see
[`setup.md`](setup.md#8-running-the-shared-http-daemon) for the operator
walkthrough):

```json
{
  "mcpServers": {
    "graphify-mesh": {
      "type": "http",
      "url": "http://127.0.0.1:19744/mcp",
      "headers": { "Authorization": "Bearer <token>" }
    }
  }
}
```

Every call to `search` or `context_pack` against an HTTP daemon must pass
`cwd`: with one process serving many sessions there is no per-session cwd for
`scope='current'` to fall back to, and an absent `cwd` fails closed with an
error naming the fix (pass `cwd`, or use `scope='repo:<id>'`).

You can also invoke the module directly (useful before an install, with the
package on `PYTHONPATH`):

```bash
PYTHONPATH=src python -m graphify_mesh.server.server
PYTHONPATH=src python -m graphify_mesh.server.server --transport http
```
