# Keeping the mesh up to date

How to configure recurring re-indexing so the published global graph keeps
tracking your repos as they change. Every path below is a placeholder —
substitute your own.

This guide assumes you have already completed [`setup.md`](setup.md): the
package is installed, `registry.json` describes your repos, and at least one
manual `graphify-mesh-sync --once` run has published a generation.

## How freshness works

You never re-index anything by hand. Each `graphify-mesh-sync --once` run is a
complete refresh cycle:

1. **Discovery** re-scans the configured scan roots for `graphify-out` links and
   reconciles them against `registry.json` — new repos are picked up, removed
   ones flagged as stale.
2. **Per-project sync** diffs each repo's source digest against saved state
   and re-runs the upstream `graphify` extraction only for repos that actually
   changed (`update` for AST-only changes, `extract` for semantic changes,
   `noop` otherwise). Unchanged repos cost almost nothing.
3. **Merge → name → embed → overlay → validate → publish** rebuilds the global
   graph from the per-repo graphs and atomically flips the `current` symlink.
   Readers (the MCP server) never see a half-written generation.

So "keeping sync up to date" means exactly one thing: **run
`graphify-mesh-sync --once` on a schedule**. There is no daemon mode — `--once`
is the only supported mode, and a whole-transaction lock guarantees an
overlapping run exits cleanly instead of corrupting anything, so an aggressive
schedule is safe.

## Step-by-step: scheduled sync with systemd (recommended)

Ready-made example units live in [`examples/systemd/`](../examples/systemd/).
The walkthrough below installs them as **user** units (no root needed).

### Step 1 — create the environment file

The service reads its configuration from a systemd `EnvironmentFile` (plain
`KEY=VALUE` lines, *not* sourced by a shell):

```bash
cp examples/systemd/graphify-mesh-sync.env.example \
   /path/to/your/workspace/graphify-mesh-sync.env
$EDITOR /path/to/your/workspace/graphify-mesh-sync.env
```

Set at minimum:

```dosini
# Where the engine publishes graphify/global/<generation>/ trees.
GRAPHIFY_MESH_ROOT=/path/to/your/workspace/graph-mesh
# Colon-separated roots scanned for per-repo graphify-out links.
GRAPHIFY_MESH_SCAN_ROOTS=/path/to/your/workspace/checkouts:/path/to/your/workspace/other-checkouts
# Optional; defaults to the resolved scan roots above (whether they came
# from GRAPHIFY_MESH_SCAN_ROOTS, the legacy GRAPHIFY_MESH_SCAN_ROOT, or the
# current working directory). The legacy single-path GRAPHIFY_MESH_SCAN_ROOT
# remains supported as a fallback for scan roots too.
GRAPHIFY_MESH_APPROVED_ROOTS=/path/to/your/workspace/checkouts:/path/to/your/workspace/other-checkouts
# Defaults to <GRAPHIFY_MESH_ROOT>/bin/registry.json — set only if elsewhere.
GRAPHIFY_MESH_REGISTRY=/path/to/your/workspace/graph-mesh/bin/registry.json
```

Optional but recommended if you run Ollama (community naming + embeddings —
both degrade gracefully when unreachable):

```dosini
GRAPHIFY_MESH_OLLAMA_BASE_URL=http://localhost:11434/v1
GRAPHIFY_MESH_OLLAMA_API_KEY=dummy
GRAPHIFY_MESH_OLLAMA_MODEL=qwen2.5-coder:14b
GRAPHIFY_MESH_OLLAMA_EMBED_BASE_URL=http://localhost:11434
GRAPHIFY_MESH_OLLAMA_EMBED_MODEL=qwen3-embedding:0.6b
```

And the upstream-CLI knobs:

```dosini
GRAPHIFY_NO_BACKUP=1
# Absolute path if `graphify` is not on PATH for the service environment.
GRAPHIFY_BIN=/path/to/your/venv/bin/graphify
```

> **Security:** if the env file holds a real `GRAPHIFY_MESH_OLLAMA_API_KEY`,
> make it unreadable to others:
>
> ```bash
> chmod 0600 /path/to/your/workspace/graphify-mesh-sync.env
> ```

Full variable reference: [`configuration.md`](configuration.md).

### Step 2 — copy and edit the service + timer units

```bash
mkdir -p ~/.config/systemd/user
cp examples/systemd/graphify-mesh-sync.{service,timer} ~/.config/systemd/user/
$EDITOR ~/.config/systemd/user/graphify-mesh-sync.service
```

Edit every `/path/to/...` placeholder in the service file:

- `WorkingDirectory=` — your workspace directory.
- `EnvironmentFile=` — the env file from step 1.
- `ExecStart=` — the absolute path to `graphify-mesh-sync` inside your
  venv/pipx install (systemd does not inherit your shell's `PATH`), e.g.
  `/path/to/your/venv/bin/graphify-mesh-sync --once`.
- `ReadWritePaths=` — **uncomment it** and point it at your mesh root. The
  example units use `ProtectSystem=strict`, which makes the whole filesystem
  read-only for the service; without this line the first write fails.
- `BindReadOnlyPaths=` — **system unit only.** Uncomment both lines if `graphify-mesh-sync` is
  installed under your home directory, and point them at the directory holding
  the console script and the directory holding its `site-packages`. The unit
  sets `ProtectHome=tmpfs`, which replaces the home directory with an empty
  tmpfs so the sync engine and the `graphify` children it spawns on untrusted
  repository source cannot read `~/.aws/credentials`, `~/.ssh` or `~/.config`.
  A pipx or venv install under `~/.local` disappears along with the rest:
  without the binary bind the service fails to start with status `203/EXEC`
  (`ExecStart` not found), and without the `site-packages` bind it starts and
  dies with `ModuleNotFoundError: graphify_mesh`. An install outside the home
  directory (`/opt`, a system venv) needs neither line. A mesh root inside the
  home directory needs its own bind as well, so prefer one outside it.

> **What a user unit actually protects.** Measured on systemd 259, a
> `systemctl --user` unit ignores `ProtectHome=`: the home directory, `~/.ssh`
> and `~/.aws/credentials` included, stays fully visible to the service and to
> every `graphify` child it spawns. `ProtectSystem=` and the other namespace
> directives in the example are in the same category. A user unit still gets
> the scheduling, journal logging, `MemoryMax=`, `TimeoutStartSec=`,
> `NoNewPrivileges=` and `RestrictAddressFamilies=` — that last one is a seccomp
> filter rather than a mount namespace, so it does apply in a user unit
> (measured: `systemd-run --user -p RestrictAddressFamilies=AF_UNIX` makes
> `socket(AF_INET)` fail with `EAFNOSUPPORT`). What a user unit does not get is
> filesystem isolation. Two ways to get it:
> install the unit as a **system** unit under `/etc/systemd/system/` with a
> dedicated `User=` (then the directives above apply and the
> `BindReadOnlyPaths=` lines matter), or keep the user unit and switch on the
> engine's own child sandbox with `GRAPHIFY_MESH_CHILD_SANDBOX=1`, which uses
> bubblewrap and works unprivileged. The sandbox covers the `graphify` children
> only, not the sync engine process itself.
>
> Neither route hides the unit's `EnvironmentFile=` from a `graphify` child: it
> is an ordinary file owned by the service UID, and the child runs as that UID.
> See [`configuration.md`](configuration.md) for what to do about it
> (`LoadCredential=`, or a file the child's UID cannot open).

Both `graphify` children that parse repository source (`update` and `extract`)
additionally run under a per-run, per-repo staging `HOME` inside the sync
engine's temporary directory. A custom `~/.graphify/providers.json` is
deliberately invisible to them; see
[`configuration.md`](configuration.md#child-process-home-isolation).

### Step 3 — choose the cadence

The example timer runs the sync on every quarter hour of the wall clock:

```dosini
[Timer]
OnCalendar=*:0/15
Persistent=true
AccuracySec=1min
RandomizedDelaySec=60
```

The monotonic bases (`OnBootSec=`, `OnUnitInactiveSec=`) are deliberately not
used: on a VM the operator suspends with host-side "save state",
`CLOCK_MONOTONIC` stops, so those timers pause for the whole saved state and
never catch up, while `Persistent=true` on an `OnCalendar=` timer fires the
missed elapse right after resume.

Tune the `OnCalendar=` interval to your repo count and how fresh you need the
graph.
Because unchanged repos are near-free (digest diff → `noop`) and overlapping
runs are lock-protected, erring on the frequent side is fine; the expensive
stages only run for repos that actually changed.

### Step 4 — enable and start

```bash
systemctl --user daemon-reload
systemctl --user enable --now graphify-mesh-sync.timer
```

If the machine is a server where you are not always logged in, allow your user
units to run without an active session:

```bash
loginctl enable-linger "$USER"
```

### Step 5 — verify

```bash
# Timer scheduled?
systemctl --user list-timers graphify-mesh-sync.timer

# Trigger one run right now and watch it:
systemctl --user start graphify-mesh-sync.service
journalctl --user -u graphify-mesh-sync.service -f
```

A successful run ends by publishing a new generation and flipping the symlink:

```bash
ls -l /path/to/your/workspace/graph-mesh/graphify/global/current
```

The `current` symlink's mtime/target changes on every publish. MCP clients pick
the new generation up automatically — `graphify-mesh-server` resolves `current`
per session, so no server restart is needed.

### Step 6 (optional) — the reaper timer

MCP stdio sessions occasionally leave orphaned `graphify.serve` / `graphify
extract` processes behind when a client dies without closing stdin. The reaper
cleans these up on its own timer, decoupled from the sync:

```bash
cp examples/systemd/reap-graphify-serve.{service,timer} ~/.config/systemd/user/
$EDITOR ~/.config/systemd/user/reap-graphify-serve.service   # fix /path/to/... placeholders
systemctl --user daemon-reload
systemctl --user enable --now reap-graphify-serve.timer
```

> **Security:** install this as a *user* unit as shown. As a system unit it
> runs as root and could kill any matching process on the host — if you must
> run it system-wide, scope it to a dedicated unprivileged `User=`.

## Alternative: plain cron

If systemd is unavailable, a crontab entry works — you just lose the
sandboxing, journal logging, and overlap accounting the units provide (the
engine's own transaction lock still prevents concurrent runs from clashing):

```cron
# m h dom mon dow command
*/15 * * * * . /path/to/your/workspace/graphify-mesh-sync.cron.env && /path/to/your/venv/bin/graphify-mesh-sync --once >> /path/to/your/workspace/graph-mesh/sync.log 2>&1
```

where `graphify-mesh-sync.cron.env` is a shell-sourceable variant of the env
file from step 1 (each line prefixed with `export `). Remember cron's `PATH` is
minimal: use absolute paths for both the console script and `GRAPHIFY_BIN`.

## Adding or removing repos later

The schedule handles content changes automatically; membership changes need two
manual touches:

1. **Add a repo** — run `graphify` once in the new checkout so it has a
   `graphify-out/graph.json` reachable under a configured scan root, then add
   its entry (`repo_id`, `root`, `collection_path`, `enabled: true`) to
   `registry.json`. The next scheduled run picks it up.
2. **Remove a repo** — set `enabled: false` (or delete the entry) in
   `registry.json`. The next run will drop it — see the shrink guard below.

No restart of anything is required; discovery re-reads `registry.json` every
run.

## Promoting a new version of the engine

If the scheduler imports the engine straight from a working tree — a
`PYTHONPATH=` line in the env file pointing at a checkout's `src/` — then every
save lands in the next scheduled run. A half-written module becomes a
`SyntaxError` inside a run that has already written state and staging, and the
transaction lock does not help: the hazard is an edit arriving *during* a run,
not two runs overlapping.

Point the scheduler at a promoted snapshot instead, and promote explicitly.

Create the snapshot directory and give the unit its own import path through a
drop-in, not through the env file:

```bash
mkdir -p /path/to/your/workspace/mesh-src
systemctl --user edit graphify-mesh-sync.service
# [Service]
# Environment=PYTHONPATH=/path/to/your/workspace/mesh-src
```

`systemctl --user edit` reloads on save; after any hand-edit of the drop-in, run
`systemctl --user daemon-reload` yourself. **Only then** remove the
`PYTHONPATH=` line from the env file. systemd lets `EnvironmentFile=` override
`Environment=`, so reversing the order leaves a window in which a run started in
between imports the installed package rather than the snapshot. Verify:

```bash
systemctl --user show graphify-mesh-sync.service -p Environment | tr ' ' '\n' | grep PYTHONPATH
```

The drop-in rather than the env file, because that file is dual-purpose: systemd
reads it and an operator sources it for manual runs. Repointing it would send
every interactive run and every `--dry-run` at the snapshot too, which is the
opposite of what a development loop needs. The cost is that a manual run against
the working tree now names its own path:

```bash
PYTHONPATH=/path/to/your/checkout/src \
  /path/to/your/venv/bin/graphify-mesh-sync --once --dry-run \
  --mesh-root /path/to/your/workspace/graph-mesh \
  --scan-root /path/to/your/workspace/checkouts
```

Promotion is then one command, run while the timer is stopped:

```bash
systemctl --user stop graphify-mesh-sync.timer
systemctl --user is-active graphify-mesh-sync.service   # must not print "activating"
rsync -a --delete /path/to/your/checkout/src/ /path/to/your/workspace/mesh-src/
systemctl --user start graphify-mesh-sync.timer
```

`--delete` is what makes the snapshot a copy rather than an accumulation: a
module deleted in the checkout must not stay importable in the snapshot.

The MCP server is a separate unit and usually runs the installed package with no
`PYTHONPATH`, so writer and reader are on different code by construction. With
the example server unit, a `PYTHONPATH=` line in the shared env file reaches the
server too. Put it in a drop-in of the sync unit instead, as shown above. The
server picks a new generation up on the symlink flip without a restart as long
as the generation format is unchanged. Restart it when a promotion changes
something it shares, such as a dependency version:

```bash
systemctl --user restart graphify-mesh-server.service
```

The promotion overwrites the only snapshot, so no previous one stays on disk.
To roll a promotion back, export the previous tag and sync it in while the
timer is stopped. Export rather than check out: a checkout carries uncommitted
edits into the snapshot.

```bash
rm -rf /path/to/your/workspace/mesh-src.tag && mkdir -p /path/to/your/workspace/mesh-src.tag
git -C /path/to/your/checkout archive <previous-tag> src | tar -x -C /path/to/your/workspace/mesh-src.tag
rsync -a --delete /path/to/your/workspace/mesh-src.tag/src/ /path/to/your/workspace/mesh-src/
```

If quick rollback matters, copy `mesh-src/` to `mesh-src.prev/` before each
promotion. Restore that copy with the timer stopped:

```bash
rsync -a --delete /path/to/your/workspace/mesh-src.prev/ /path/to/your/workspace/mesh-src/
```

To fall back to the installed package instead, remove the
`PYTHONPATH` override. `systemctl --user revert graphify-mesh-sync.service`
does that, but it removes every drop-in and override of the sync unit. To
remove only this one, delete its drop-in file
(`~/.config/systemd/user/graphify-mesh-sync.service.d/override.conf` when made
with `systemctl --user edit`) and run `systemctl --user daemon-reload`.
`publish.prune_old_generations` keeps the previous generation on disk
(`keep_structural_generations`, default 2), so repointing `current` at it is a
`ln -sfn` away. Do not put the `PYTHONPATH=` line in the shared env file: with
the example server unit, it would switch the server too.

## Running the shared MCP server with systemd

`graphify-mesh-server --transport http` runs one MCP server for every local
agent. Three example units in [`examples/systemd/`](../examples/systemd/) run
it and restart it when its configuration changes:

- `graphify-mesh-server.service` runs the daemon.
- `graphify-mesh-server-reload.path` watches the config files.
- `graphify-mesh-server-reload.service` logs the change and restarts the
  daemon.

The server reads two environment files. The first is the same
`graphify-mesh-sync.env` the sync unit reads. Query embeddings must use the
embedding model and settings the sync run used, and one shared file keeps them
equal. The second file, `graphify-mesh-server.env`, holds the token and any
server-only overrides, such as the port. The server file is read last, so a
key set in both takes the server file's value. The shared file also carries
`GRAPHIFY_MESH_OLLAMA_API_KEY` into the server, which does not use it. If that
matters, keep the key in a file only the sync unit reads, such as a second
`EnvironmentFile=` on the sync unit. The split keeps the token out of the sync
process environment, and lets you share or edit the sync env file without
exposing it. It is not isolation: the sync engine's `graphify` children run as
the same UID and can open the server file too. Real isolation means running
the sync unit under a different UID from the one that owns
`graphify-mesh-server.env`. The server reads the token only from its
environment; see [`configuration.md`](configuration.md).

Changing `GRAPHIFY_MESH_OLLAMA_EMBED_MODEL` or
`GRAPHIFY_MESH_OLLAMA_EMBED_BASE_URL` in the shared file restarts the server at
once. After a model change, queries use the new model while the published
vectors still come from the old one. Until the next sync publishes re-embedded
vectors, the vector part of search is wrong. A model with a different dimension
scores every vector as zero. A model with the same dimension compares vectors
from two different spaces. The `degraded` list reports neither case. Run a sync
right after the change, and do not rely on search before it publishes.

A base-URL change that serves the same model name does not trigger
re-embedding, and neither does a `GRAPHIFY_MESH_EMBED_NUM_CTX` change. The
embedding cache key covers the model and snippet settings, not the endpoint or
the context window. Unchanged nodes keep their old vectors, and only changed
nodes are re-embedded. No flag forces a full re-embed.

1. Create the server env file and fill in a generated token of at least 32
   characters:

   ```bash
   cp examples/systemd/graphify-mesh-server.env.example \
      /path/to/your/workspace/graphify-mesh-server.env
   chmod 600 /path/to/your/workspace/graphify-mesh-server.env
   python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
   ```

2. Copy the three units:

   ```bash
   cp examples/systemd/graphify-mesh-server.service \
      examples/systemd/graphify-mesh-server-reload.{path,service} \
      ~/.config/systemd/user/
   ```

3. Edit every `/path/to/...` placeholder. The two `EnvironmentFile=` paths in
   the server unit and the two env-file `PathChanged=` paths in the path unit
   must name the same files.

4. Enable the server and the watcher:

   ```bash
   systemctl --user daemon-reload && systemctl --user enable --now graphify-mesh-server.service graphify-mesh-server-reload.path
   ```

5. Verify the reload. Note the current PID, touch a watched file, then check
   the journal and the PID again:

   ```bash
   systemctl --user show -p MainPID graphify-mesh-server
   touch /path/to/your/workspace/graphify-mesh-server.env
   journalctl --user -u graphify-mesh-server-reload -u graphify-mesh-server -n 20
   systemctl --user show -p MainPID graphify-mesh-server
   ```

   The journal shows a line like `graphify-mesh config changed:
   /path/to/your/workspace/graphify-mesh-server.env; restarting
   graphify-mesh-server.service`, and `MainPID` has a new value. The reload
   unit waits 2 s before restarting, so allow for that. The path reads
   `unknown path` when the reload service is started by hand, or on systemd
   older than 252. Neither sets `TRIGGER_PATH`.

   If the watcher stopped (`systemctl --user is-failed
   graphify-mesh-server-reload.path` prints `failed`), reset and restart it:

   ```bash
   systemctl --user reset-failed graphify-mesh-server-reload.service graphify-mesh-server-reload.path && systemctl --user start graphify-mesh-server-reload.path
   ```

The watcher also covers the server unit file and its drop-in directory
(`graphify-mesh-server.service.d`), so the reload unit runs
`systemctl --user daemon-reload` before the restart. A drop-in change is
logged as the directory path, not the changed file. One changed path is
logged (with several changes, no guarantee which), never file contents.

The reload unit sleeps 2 s before it restarts the server. Saves that land in
that window merge into the same run and are read by the restart. An edit
landing after the restart has started can be missed until the next change, so
save again if the journal shows no second restart. The reload unit sets
`StartLimitIntervalSec=0`, so a burst of saves cannot trip systemd's default
start limit and stop the watcher. The server unit sets its own crash-loop
guard: `StartLimitIntervalSec=60` with `StartLimitBurst=5`. The default 10 s
window never trips with `RestartSec=2`, so the window is wider. The reload unit
runs `systemctl --user reset-failed` on the server before each restart, so
config-change restarts do not count against it.

The reload unit uses `systemctl restart`, which also starts a stopped server.
When you stop the server on purpose, stop the watcher too, or the next config
edit starts the server again. Stop the reload service as well: a reload job
already in its 2 s sleep restarts the server when the sleep ends.

```bash
systemctl --user stop graphify-mesh-server-reload.path graphify-mesh-server-reload.service graphify-mesh-server.service
```

A failed server config check, such as a blank or short token or a public bind
without the opt-in, makes the server exit `2`. `RestartPreventExitStatus=2`
stops systemd retrying it, so the unit fails once and visibly. The watcher
restarts it once the file is fixed. Other bad values, such as a non-integer
`GRAPHIFY_MESH_EMBED_NUM_CTX`, fail with a traceback and exit `1`. The unit
then restarts until it hits its start limit.

A restart drops requests in flight, and those requests fail. The HTTP
transport is stateless, so the next request succeeds once the server is back,
after a few seconds.

The sync service needs no reload unit. Each timer run starts a fresh process
that reads its env file again.

## Troubleshooting

- **Run fails validation with a stale-repo / shrink error.** The validator
  refuses to publish a generation meaningfully smaller than the previous one,
  as protection against accidentally wiping graphs (e.g. scan roots that
  failed to mount). If the shrink is intentional — you really removed repos —
  authorize it once by hand:

  ```bash
  graphify-mesh-sync --once --allow-shrink \
    --mesh-root /path/to/your/workspace/graph-mesh \
    --scan-root /path/to/your/workspace/checkouts \
    --scan-root /path/to/your/workspace/other-checkouts
  ```

  Do **not** put `--allow-shrink` in the scheduled unit; it would disable the
  guard permanently.

- **Communities keep placeholder names / search feels lexical-only.** Ollama
  was unreachable during the run. Fix the `GRAPHIFY_MESH_OLLAMA_*` values (note
  the asymmetry: the LLM base URL ends in `/v1`, the embed base URL does not)
  and let the next run re-name/re-embed — both stages only re-process what
  changed, so recovery is cheap.

- **`graphify: command not found` in the journal.** The service environment
  has no shell `PATH`. Set `GRAPHIFY_BIN` in the env file to the absolute
  binary path.

- **Service fails on first write with a read-only filesystem error.** You
  forgot to uncomment `ReadWritePaths=` (step 2) — `ProtectSystem=strict`
  blocks all writes outside the listed paths.

- **Two runs at once?** Safe. The whole-transaction lock makes the second run
  exit without touching anything.

- **Want to see what a run would do without publishing?** Dry run writes
  nothing outside a private staging dir:

  ```bash
  graphify-mesh-sync --once --dry-run \
    --mesh-root /path/to/your/workspace/graph-mesh \
    --scan-root /path/to/your/workspace/checkouts \
    --scan-depth 4
  ```
