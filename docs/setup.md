# Setup

Step-by-step: get from zero to a merged, queryable global graph across your own
repos. Every path below is a placeholder — substitute your own.

## 1. Prerequisites

- **Python 3.11+**
- **The upstream `graphify` CLI/library.** graphify-mesh shells out to the
  `graphify` binary and imports `graphify.build` at merge time. It's published
  on PyPI as `graphifyy` and is a declared dependency of this package, so
  `pip install graphify-mesh` (step 2 below) pulls it in automatically — no
  separate install needed for the library import to work.

  You do still want the standalone `graphify` **command** on `PATH` if you
  plan to run ad-hoc queries yourself (outside the sync engine), or if the
  sync engine runs in an environment where its own venv's `bin/` isn't on
  `PATH` (e.g. a systemd service — see `GRAPHIFY_BIN` below):

  ```bash
  pipx install graphifyy      # provides the `graphify` command, isolated
  graphify --help             # confirm it is on PATH
  ```

  If `graphify` is not on `PATH` for the environment that runs the sync (e.g. a
  systemd service), set `GRAPHIFY_BIN` to its absolute path.

- **(Optional) An Ollama host** for the community-naming and embedding stages.
  Both stages degrade gracefully when Ollama is unreachable (communities keep
  placeholder names; search falls back to lexical + structural only).

## 2. Install graphify-mesh

```bash
pip install graphify-mesh
# or, from a checkout:
pip install -e .
```

This provides three console scripts: `graphify-mesh-sync`,
`graphify-mesh-server`, and `graphify-mesh-reap`.

### Installing from the project's own package index

graphify-mesh is published to a PEP 503 "simple" index. The release workflow
(`.github/workflows/release.yml`) stores the index on the `gh-pages` branch.
The Pages workflow (`.github/workflows/pages.yml`) deploys the docs landing
page and that index together. After publishing the index, the release workflow
finishes; its successful completion triggers the Pages workflow on the default
branch. Pages environment protection therefore sees `main`, not a release tag.
The repository's Pages source must be set to "GitHub Actions". Install a
specific version with:

```bash
pip install \
  --index-url https://oxcom.github.io/graphify-mesh/simple/ \
  graphify-mesh==0.1.0
```

The exact URL scheme is:

- **Index root:** `https://oxcom.github.io/graphify-mesh/simple/`
- **Project page:** `https://oxcom.github.io/graphify-mesh/simple/graphify-mesh/`
- Each release's wheel and sdist are linked from the project page, pointing at
  the GitHub Release asset download URLs.

## 3. Point `graphify` at each repo

For every repo you want in the mesh, run `graphify` once so it produces a
`graphify-out/graph.json`, and make sure that output is reachable from one of
your scan roots (a `graphify-out` symlink per checkout is the convention).
Example layout:

```
/path/to/your/workspace/checkouts/
    backend-a/graphify-out    -> ../../graph-mesh/graphify/example-org/backend-a
    frontend-b/graphify-out   -> ../../graph-mesh/graphify/example-org/frontend-b
```

## 4. Write your registry.json

Copy the example and edit it to describe *your* repos:

```bash
cp examples/registry.example.json \
   /path/to/your/workspace/graph-mesh/bin/registry.json
$EDITOR /path/to/your/workspace/graph-mesh/bin/registry.json
```

Each entry needs a stable `repo_id`, the checkout `root`, and the
`collection_path` where that repo's `graph.json` lives. Keep `bin/registry.json`
and any `*.env` file beside it out of group and other reach (`0644` for the
registry, `0600` for the env file) — the engine warns at startup when they are
looser, and never changes them for you. See
[`configuration.md`](configuration.md#registryjson) for the full schema.

## 5. First dry run

A dry run prints every action and writes nothing outside a private staging
directory — safe to run anywhere:

```bash
graphify-mesh-sync --once --dry-run \
  --mesh-root  /path/to/your/workspace/graph-mesh \
  --scan-root  /path/to/your/workspace/checkouts \
  --scan-root  /path/to/your/workspace/other-checkouts \
  --scan-depth 4
```

Review the JSON report: `reconciliation`, `project_actions`, `stale_repos`,
`merge_ok`, and `validation_ok` tell you what a real run would do.

## 6. First real run

```bash
graphify-mesh-sync --once \
  --mesh-root  /path/to/your/workspace/graph-mesh \
  --scan-root  /path/to/your/workspace/checkouts \
  --scan-root  /path/to/your/workspace/other-checkouts \
  --scan-depth 4
```

On success this publishes a new immutable generation and flips
`<mesh_root>/graphify/global/current` to point at it.

You can also drive everything from env vars instead of flags — see
[`configuration.md`](configuration.md). To keep the graph fresh automatically,
set up a scheduled sync — full step-by-step walkthrough (systemd timers, cron
alternative, troubleshooting) in
[`keeping-sync-up-to-date.md`](keeping-sync-up-to-date.md).

## 7. Register the MCP server

`graphify-mesh-server` defaults to stdio, one process per client session.
Register it with your MCP-capable client:

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

See [`mcp-server.md`](mcp-server.md) for the tools and protocol details.

## 8. Running the shared HTTP daemon

`--transport http` runs one `graphify-mesh-server` process serving every
local agent over a URL instead of spawning a process per client session.
stdio keeps working and stays the default; this is an opt-in alternative for
a machine running many agent sessions against the same generation.

### Env file

Put the port and a generated token in `~/.config/claude-mcp/graphify-mesh.env`
(never in the systemd unit file, which `systemctl cat` would expose to any
local user):

```bash
GRAPHIFY_MESH_HTTP_PORT=19744
GRAPHIFY_MESH_HTTP_TOKEN=<generate one, do not reuse a token from elsewhere>
```

An empty or whitespace-only `GRAPHIFY_MESH_HTTP_TOKEN` counts as absent: the
daemon exits `2` with the reason on stderr instead of starting unauthenticated.

### systemd unit

Point the unit's `ExecStart` at the HTTP transport and have it read the env
file above via `EnvironmentFile`:

```ini
[Service]
EnvironmentFile=%h/.config/claude-mcp/graphify-mesh.env
Environment=GRAPHIFY_MESH_ROOT=/path/to/your/workspace/graph-mesh
ExecStart=graphify-mesh-server --transport http
```

Restarting the unit drops graphify-mesh tools from every open session for a
few seconds while it comes back up.

### Client entry

Point the client at the daemon's URL with the bearer token in the
`Authorization` header:

```json
{
  "mcpServers": {
    "graphify-mesh": {
      "type": "http",
      "url": "http://127.0.0.1:19744/mcp",
      "headers": { "Authorization": "Bearer <token from the env file>" }
    }
  }
}
```

Every `search`/`context_pack` call must pass its own absolute `cwd`: the
shared daemon has no per-session directory to fall back to, and an
unregistered or absent `cwd` fails closed.

### PreToolUse hook: filling cwd automatically

A client that calls `search` or `context_pack` with `scope: "current"` and no
`cwd` argument gets a `cwd` resolved against the daemon's own process
directory, which matches no registered repository. The call fails closed. A
Claude Code `PreToolUse` hook can fill `cwd` before the call reaches the
server, so the model does not have to pass it on every call from a
registered repository.

`examples/hooks/graphify-mesh-cwd.py` is that hook: stdlib-only Python, no
dependencies. Copy it somewhere Claude Code can execute:

```bash
mkdir -p ~/.claude/hooks
cp examples/hooks/graphify-mesh-cwd.py ~/.claude/hooks/graphify-mesh-cwd.py
chmod +x ~/.claude/hooks/graphify-mesh-cwd.py
```

Add a `PreToolUse` entry to `~/.claude/settings.json` matching this server's
tools:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "mcp__graphify-mesh__*",
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude/hooks/graphify-mesh-cwd.py"
          }
        ]
      }
    ]
  }
}
```

Verify from a registered repository: call `search` with `scope: "current"`
and no `cwd` argument. A working hook returns results scoped to that repo
instead of the closed-scope error.

The hook fills `cwd` only when the argument is absent. A call that already
carries `cwd` is passed through untouched, including an invalid one such as
`""` or a relative path: the server rejects those, and a hook that replaced
them with the session directory would turn a call that should fail into a
call answered for a different repository.

This hook is Claude-Code-specific: it relies on Claude Code's `PreToolUse`
stdin/stdout contract. A different MCP client needs its own equivalent, or
its calls must pass `cwd` themselves, or use an explicit `scope` such as
`repo:<id>` instead of `current`.

### Rollback

Restore two things: the unit's `ExecStart` back to the stdio command, and the
client's `mcpServers` entry back to the `command`/stdio form. stdio never
stopped working, so rollback is configuration-only, no data migration.
