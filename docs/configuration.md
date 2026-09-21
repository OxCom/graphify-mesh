# Configuration

Everything is configurable via CLI flags, `GRAPHIFY_MESH_*` environment
variables, or the `Settings` / `ServerConfig` dataclasses directly. No
machine-specific paths are baked into the package.

## Environment-variable prefix

This package uses the **`GRAPHIFY_MESH_`** prefix for all of its own
environment variables. (An earlier internal iteration used a different prefix;
it was renamed to `GRAPHIFY_MESH_` when the engine became a standalone generic
package so that nothing carries organization-specific naming.)

Two variables belong to the upstream `graphify` CLI, not to this package, and
keep their upstream names: `GRAPHIFY_BIN` and `GRAPHIFY_NO_BACKUP`.

## Environment variables

| Variable | Used by | Default | Meaning |
|----------|---------|---------|---------|
| `GRAPHIFY_MESH_ROOT` | sync + server | current working directory | Root that contains the `graphify/global/` tree this engine publishes into and serves from. |
| `GRAPHIFY_MESH_SCAN_ROOTS` | sync | current working directory | Colon-separated roots scanned for per-repo `graphify-out` symlinks/directories. Segments are stripped and empty segments dropped; if no usable segments remain, resolution falls through to `GRAPHIFY_MESH_SCAN_ROOT`, then the current working directory. |
| `GRAPHIFY_MESH_SCAN_ROOT` | sync | (unset) | Legacy single-path form, still honored after `GRAPHIFY_MESH_SCAN_ROOTS` for backward compatibility. A whitespace-only value is treated as unset; the final fallback is the current working directory. |
| `GRAPHIFY_MESH_APPROVED_ROOTS` | sync | resolved scan roots | Colon-separated roots trusted by the discovery path-traversal guard. Uses the same strip-and-drop-empty normalization as `GRAPHIFY_MESH_SCAN_ROOTS`; unset or empty means the resolved scan roots. |
| `GRAPHIFY_MESH_SCAN_DEPTH` | sync | `4` | Maximum nesting of a project directory below each scan root. Values below `1` or unparsable values degrade to `1`; values above `8` degrade to `8` rather than raising. |
| `GRAPHIFY_MESH_REGISTRY` | sync + server | `<root>/bin/registry.json` | Path to `registry.json`. |
| `GRAPHIFY_MESH_CONTENT_DIGEST` | sync (manifest) | `1` (on) | When on, the per-project manifest digest hashes file **content**; set to `0`/`false`/`no`/`off` to key on `(size, mtime_ns)` as before. mtime marks a repo changed after a checkout or a save-without-edit, which schedules a non-deterministic LLM `extract` for a byte-identical tree. Unreadable files fall back to the mtime marker for that entry only. |
| `GRAPHIFY_MESH_UNREFRESHED_THRESHOLD` | sync (publish gate) | `0.40` | Fraction of repos that may end in `failed` (refresh did not complete; last-good graph intact) before publish is refused. Separate from — and looser than — `STALE_PUBLISH_THRESHOLD`, which still gates `shrink_refused`/`bootstrap_failed`. Clamped to `(0, 1]`; invalid values fall back to the default. |
| `GRAPHIFY_MESH_CLI_TIMEOUT` | sync (subprocess) | `900` | Seconds any single `graphify` subprocess may run before it is killed and reported as returncode `124`. `extract` runs an LLM pass whose wall time scales with repo size and with load on the shared Ollama host, so a large repo can exceed the default; because a timeout never advances per-repo state, such a repo fails on every run and counts toward the stale-ratio publish gate forever. Clamped to `60`–`7200` (the ceiling stays under the unit's `TimeoutStartSec` so a hung subprocess cannot outlive its run); unparsable or out-of-range values fall back to the default. |
| `GRAPHIFY_MESH_SHRINK_TOLERANCE` | sync (per-repo guard) | `0.10` | Fraction a repo's node/edge counts may drop before the per-repo shrink guard refuses the result. `extract` re-derives entities with an LLM and is non-deterministic, so an unchanged repo varies a few percent per run; a strict guard turns that jitter into a permanent refusal loop, because a refusal does not advance per-repo state and the repo re-extracts every run. Counts below `old * (1 - tolerance)` (rounded up) are still refused, so material loss is caught. Graphs small enough that the rounding floor equals the old count get exact-match semantics. Set to `0` for the old absolute behaviour — appropriate only if every repo uses AST-only `update`, which is deterministic. Unparsable or out-of-range values (`<0`, `>=1`) fall back to the default rather than disabling the guard. |
| `GRAPHIFY_MESH_OLLAMA_BASE_URL` | sync (naming) | `http://localhost:11434/v1` | OpenAI-compatible `/v1` endpoint for the community-labeling LLM. |
| `GRAPHIFY_MESH_OLLAMA_API_KEY` | sync (naming) | `dummy` | API key sent to the `/v1` endpoint (Ollama ignores it, but the client requires one). |
| `GRAPHIFY_MESH_OLLAMA_MODEL` | sync (naming) | `qwen2.5-coder:14b` | Model for community labeling. |
| `GRAPHIFY_MESH_OLLAMA_HEALTH_TIMEOUT` | sync (naming) | `3.0` | Seconds to wait on the naming-stage health check before degrading. |
| `GRAPHIFY_MESH_OLLAMA_API_TIMEOUT` | sync (naming) | `180.0` | Seconds one community-labeling call may take. Exported to graphify as `GRAPHIFY_API_TIMEOUT` by `naming.bind_upstream_backend`, because labeling now runs in process and the old `GRAPHIFY_MESH_CLI_TIMEOUT` bound on the child is gone. Upstream's own default is 600 s per call with zero retries for ollama, and a full relabel of ~1100 communities is ~11 sequential batches, so a wedged backend (healthy `/models`, hung completions) could outlast the unit's `TimeoutStartSec` and be SIGKILLed past every cleanup. Must be a number, `> 0` and `<= 3600`; anything else raises at startup naming the variable. Distinct from `GRAPHIFY_MESH_OLLAMA_HEALTH_TIMEOUT`, which bounds the `/models` probe only. |
| `GRAPHIFY_MESH_OLLAMA_EMBED_BASE_URL` | sync (embed) | `http://localhost:11434` | **Native** `/api/embed` endpoint (no `/v1` suffix). |
| `GRAPHIFY_MESH_OLLAMA_EMBED_MODEL` | sync (embed) | `qwen3-embedding:0.6b` | Embedding model. |
| `GRAPHIFY_MESH_OLLAMA_EMBED_HEALTH_TIMEOUT` | sync (embed) | `3.0` | Seconds to wait on the embed-stage health check before degrading. |
| `GRAPHIFY_MESH_EXTRACT_HEALTH` | sync (extract guard) | (enabled) | Disable knob for the extract-backend health probe: `off`/`0`/`false`/`no` turns it off. Disabled — or an empty probe URL (see `GRAPHIFY_MESH_EXTRACT_HEALTH_URL`) — makes the whole infra-outage feature inert: no probes are ever attempted, every child failure stays plain `failed`, and no `infra_*` statuses (and thus none of the grace-window gate handling) can be produced — behavior identical to before the guard existed. |
| `GRAPHIFY_MESH_EXTRACT_HEALTH_URL` | sync (extract guard) | `OLLAMA_BASE_URL` | Base URL probed (GET `{base}/models`) before each ollama-backed `extract`/`bootstrap` child is spawned, and again after a child fails, to classify outages as `infra_skipped`/`infra_failed` instead of `failed`. The probe is tri-state: a connect error, DNS failure, timeout, or 5xx is an **outage**; any **4xx** (bad key, wrong path, TLS/gateway misconfig) means the *probe* is misconfigured, not the backend — logged once per run at ERROR (naming the URL and status) and failed **open**, so children spawn normally and real failures stay plain `failed`. Falls back to `OLLAMA_BASE_URL` — the env var the extract **child** itself resolves its endpoint from, deliberately not `GRAPHIFY_MESH_OLLAMA_BASE_URL` (this package's own naming-stage endpoint, which can be a different host) — only when this var is **entirely unset**; set-but-empty pins the URL to `""` and keeps the feature inert. Empty (however arrived at) means inert. |
| `GRAPHIFY_MESH_EXTRACT_HEALTH_API_KEY` | sync (extract guard) | `OLLAMA_API_KEY` (else empty) | Bearer key sent with the probe request. |
| `GRAPHIFY_MESH_EXTRACT_HEALTH_TIMEOUT` | sync (extract guard) | `5.0` | Seconds to wait on the extract-backend probe. Same validation as the other `*_HEALTH_TIMEOUT` vars: must be a number, `> 0` and `<= 300`, anything else raises at startup. |
| `GRAPHIFY_MESH_INFRA_GRACE_HOURS` | sync (publish gate) | `24.0` | Grace window during which `infra_*` repos are removed from **both** publish-gate ratios entirely — numerator **and** denominator, so in-grace repos never dilute the ratios computed over the repos that still count (`infra_since` — the start of the repo's outage — is recorded in per-repo state, preserved across runs, and cleared on the next successful refresh; if the host was suspended between two runs it is shifted forward by the frozen interval, so a grace window is never consumed by wall-clock time during which no retry could run). Past the grace they count toward the **unrefreshed** ratio, in both numerator and denominator (their last-good graph is old, never suspect); once any repo is past its grace, a total outage also stops short-circuiting as the zero-refresh noop and ends the run blocked on the unrefreshed threshold instead. `0` is allowed and means infra statuses count toward the gate immediately. Negative, non-finite (`nan`/`inf`), or unparseable values raise at startup, naming the variable. |
| `GRAPHIFY_MESH_EXTRACT_CONCURRENCY` | sync (extract) | `2` | Max concurrent per-repo `graphify extract`/`update` children (bounded thread pool). Every child's RSS lands in the same cgroup `MemoryMax` as the parent sync process (steady-state parent peak observed ~1.9G inside a 4G cgroup), so this is a tuned, hard-capped setting — never derive it from the repo count. `subprocess.run` releases the GIL, so a thread pool (not a process pool) is sufficient. Unparsable/zero/negative values fall back to the hard floor of `1` (fully sequential) rather than raising. |
| `GRAPHIFY_MESH_TLS_MODE` | sync (all outbound HTTPS) | `strict` | How outbound HTTPS certificates are verified on the naming/embed health probes and the embed calls. `strict`: stock `ssl.create_default_context()` — full chain + hostname verification plus the strict X.509 extension profile. `relaxed`: clears `VERIFY_X509_STRICT` **only**; chain verification against the system CA store and hostname checking stay on. Use it on a host behind a TLS-intercepting corporate proxy, where the re-signed certificate omits the Authority Key Identifier and Python 3.13+ therefore rejects a chain that `curl` and Node accept (`certificate verify failed: Missing Authority Key Identifier`) — without it the probes report a reachable backend as unhealthy and the pipeline runs permanently degraded. `insecure`: the `curl -k` equivalent — `check_hostname=False`, `verify_mode=CERT_NONE`, every certificate accepted including an attacker's, logged at WARNING on each context build; for a diagnostic run on a trusted network only. An unrecognized value raises at call time naming the variable. Note this covers only calls this package makes itself: the `graphify` **child** process resolves TLS on its own, so a proxied host needs the child pointed at an endpoint it can reach (e.g. a local HTTP shim). |
| `GRAPHIFY_MESH_TLS_INSECURE` | sync (all outbound HTTPS) | unset | Boolean shorthand for `GRAPHIFY_MESH_TLS_MODE=insecure` (`1`/`true`/`yes`/`on`). An explicit `GRAPHIFY_MESH_TLS_MODE` wins over it. |
| `GRAPHIFY_BIN` | sync | `graphify` | Name/path of the upstream `graphify` binary. |
| `GRAPHIFY_NO_BACKUP` | sync | (set to `1` on child calls) | Suppresses `graphify`'s dated backup dirs; the sync engine always sets this on the graphify subprocesses it spawns. |
| `GRAPHIFY_MESH_TRANSPORT` | server | `stdio` | `stdio` or `http`. A CLI `--transport` flag beats this; this beats the default. |
| `GRAPHIFY_MESH_HTTP_HOST` | server | `127.0.0.1` | Bind address for `--transport http`. |
| `GRAPHIFY_MESH_HTTP_PORT` | server | `19744` | Bind port for `--transport http`. |
| `GRAPHIFY_MESH_HTTP_PATH` | server | `/mcp` | Mount path for `--transport http`. |
| `GRAPHIFY_MESH_HTTP_TOKEN` | server | none | Bearer token required for `--transport http`. Blank or whitespace-only counts as absent, and a token shorter than 32 characters is refused: the daemon exits `2` with the reason on stderr rather than starting unauthenticated or with a guessable token. The error never echoes the token. stdio mode ignores this variable. |
| `GRAPHIFY_MESH_ALLOW_PUBLIC_BIND` | server | off (`0`/`false`/`no`) | Opt-in required to bind **any** non-loopback host with `--transport http`, not only the wildcards. Loopback means `127.0.0.0/8`, `::1`, the IPv4-mapped loopback forms and the `localhost` names; a concrete interface address such as `192.168.1.10`, and any hostname that neither is `localhost` nor parses as an IP, count as public. The guard never resolves DNS. Without the opt-in, `ServerConfig.from_env` raises `ConfigError` before the daemon starts. |
| `GRAPHIFY_MESH_CHILD_ENV_EXTRA` | sync | none | Comma-separated list of extra variable **names** to pass through to the `graphify` subprocesses. The child environment is an allowlist (`PATH`, `HOME`, `LANG`/`LC_*`, `NO_COLOR`, the proxy and CA variables, `PYTHONPATH`, the `OLLAMA_*` family, and `GRAPHIFY_*` minus `GRAPHIFY_MESH_*` — which covers `GRAPHIFY_OLLAMA_*`), so an operator-specific variable the upstream binary needs is declared here rather than by weakening the allowlist. The allowlist is narrow on purpose: the extraction backend is pinned to `ollama` by a literal at its call site (`sync/graphify_cli.py`'s `run_extract`), so no environment can select another backend and forwarding another vendor's credentials only widens what the child could leak. To run a non-Ollama backend you must therefore both make the backend selectable in the package and declare its variables here by name: the `OPENAI_*`, `ANTHROPIC_*`, `GEMINI_*`, `DEEPSEEK_*`, `AZURE_OPENAI_*` and `AWS_*` families, plus `GOOGLE_API_KEY`, `KIMI_BASE_URL`, `MOONSHOT_API_KEY`, `CLAUDE_CONFIG_DIR`, `CLAUDE_PROJECT_DIR`, `FALKORDB_PASSWORD` and `LOCALAPPDATA`, no longer pass by default. When one of those names is set in the parent environment and dropped, it is logged by name at info level **once per process**, with this variable named as the fix. Every dropped parent variable is logged by name (never by value) at debug level, also once per process: the dropped set is a property of the engine's environment, identical for every child a run spawns. A `GRAPHIFY_MESH_*` name listed here is rejected with a warning: this package's own config, tokens included, is never the child's business. |
| `GRAPHIFY_MESH_CHILD_SANDBOX` | sync | off | Set to `1`/`on`/`true`/`yes` to run every `graphify` child under [bubblewrap](https://github.com/containers/bubblewrap). `0`/`off`/`false`/`no` and an unset variable leave it off; **any other value warns and turns the sandbox ON**, because a value nobody recognises still means somebody asked for the sandbox, and the old parse silently produced an unsandboxed run for `enabled`, `Y` or `2`. The child keeps the filesystem readable and keeps network access (the extract backend is an HTTP service). What it loses: the operator's home directory and the session runtime directory (`XDG_RUNTIME_DIR`, else `/run/user/<uid>`) are each replaced by an empty tmpfs, so `~/.ssh`, `~/.aws/credentials` and `~/.config` are gone **and** the ssh-agent and D-Bus sockets underneath the runtime directory cannot be connected to — a read-only bind does not stop `connect()`, and a live agent signs with keys whose files the child never reads. It also gets its own PID namespace (`--unshare-pid`, so it cannot signal the sync engine or the mesh HTTP daemon), its own session (`--new-session`), a private minimal `/dev` (`--dev`, so `/dev/shm` is not shared with every process of that UID), and dies with its monitor (`--die-with-parent`, so a `GRAPHIFY_MESH_CLI_TIMEOUT` expiry ends the real work rather than only the wrapper). Writable: the repo `root`, the **registry-declared** `collection_path`, and the staging HOME — every one of them checked at launch to resolve under an approved root or the mesh root, and a path that does not raises rather than being bound. The `graphify-out` symlink inside the scanned repository is never resolved for this, so the repository cannot choose where a writable bind lands. Read-only: the interpreter's `site-packages`, the `graphify` binary's directory (both the literal path in `argv` and its resolved target, which differ for a pipx shim) and, when the binary sits in a virtualenv, the venv root holding `pyvenv.cfg` and `lib/` — a `pipx install graphifyy` or `pip install --user` copy lives under the home directory the tmpfs empties. `bwrap` is an OS package and is not installed by this project: with the sandbox on and `bwrap` missing from `PATH`, the first child launch raises `ChildSandboxUnavailable` rather than running unsandboxed. Off by default, so an existing deployment is unaffected. This is the only filesystem isolation available to a `systemctl --user` install, which ignores systemd's `ProtectHome=` (measured on systemd 259). |
| `GRAPHIFY_MESH_CHILD_MASK_PATHS` | sync | none | Comma-separated **absolute** directories to replace with an empty tmpfs inside the child sandbox, on top of the ones the engine derives (the home directory, the session runtime directory, and the mesh `bin/` directory). Use it for operator state the child has no business reading that does not sit under any of those. Relative entries are refused with a warning. Ignored when the sandbox is off. |

## Child-process HOME isolation

Every `graphify` subprocess the sync engine spawns to do work — `update`,
`extract`, `merge-graphs` — runs with `HOME` pointed at a directory inside the
run's own temporary staging tree, never at the operator's home. `merge-graphs`
uses one staging home per run; `update` and `extract` get one **per repo**,
because they run concurrently (`GRAPHIFY_MESH_EXTRACT_CONCURRENCY`) and upstream
writes caches under `HOME`. The whole tree is deleted when the run ends.

The naming stage spawns nothing. It clusters and labels in process through
graphify's Python API, so neither the staging `HOME` nor the child sandbox
applies to it; what it handles is a graph the engine has already merged and
owns, plus one HTTP client to the configured backend.

The clustering-backend check (`sync/backend.py`) is no longer a child process
either. It calls `importlib.util.find_spec` for `graspologic_native` and then
`graspologic` **in the sync interpreter**, because that is the interpreter whose
imports decide whether graphify clusters with Leiden or Louvain now that
clustering runs in process. `find_spec` introspects without importing, so
neither package executes. Both spellings are checked: on Python >= 3.13 the
`leiden` extra installs only `graspologic-native`, and a check for `graspologic`
alone would certify Louvain while Leiden actually ran. `GRAPHIFY_BIN` still
matters for `merge-graphs`, `update`, `extract` and the version-parity check;
`assert_pinned_backend` accepts and ignores it.

Consequences an operator should expect:

- A custom `~/.graphify/providers.json` is not visible to any of these children.
  This is deliberate: upstream resolves custom LLM providers from that file, and
  a provider's `base_url` decides where a parsed repository and the API key are
  sent. Configure the backend through the environment instead.
- Upstream's query log and hook caches land in the staging tree and vanish with
  it, rather than accumulating under `~/.cache`.
- `PYTHONPATH` is set explicitly for these children so a `pip install --user`
  copy of `graphifyy` stays importable under the substituted `HOME`.

Set `GRAPHIFY_MESH_CHILD_SANDBOX=1` to back this up at the OS level with
bubblewrap; see the table above.

The substitution only redirects code that goes through `Path.home()` or
`expanduser`. The child still runs as the same UID and can open any absolute
path the operator can, so the OS-level half of this control is the example
unit's `ProtectHome=tmpfs` in a **system** unit, or
`GRAPHIFY_MESH_CHILD_SANDBOX=1` in a user unit, where `ProtectHome=` has no
effect (see [`keeping-sync-up-to-date.md`](keeping-sync-up-to-date.md)).

### What the sandbox does not solve: a readable `EnvironmentFile`

`systemd`'s `EnvironmentFile=` is an ordinary file owned by the service UID. The
sync engine's children run as that same UID and `--ro-bind / /` keeps the whole
filesystem readable, so a child that can run code can read that file directly
and recover `GRAPHIFY_MESH_OLLAMA_API_KEY` and `GRAPHIFY_MESH_HTTP_TOKEN` — the
exact names the child environment allowlist withholds. Network access is shared,
so reading and exfiltrating is one step. This is not only a sandbox problem: the
file is readable by that UID whether the sandbox is on or off, and the allowlist
was never a control against a process that can open files.

What the sandbox does do is mask the mesh `bin/` directory, which is where the
layout in [`setup.md`](setup.md) puts `registry.json` and the environment file.
The child reads nothing in there. `GRAPHIFY_MESH_CHILD_MASK_PATHS` extends the
masked set for a deployment that keeps the file elsewhere.

That is a mitigation, not a fix. The fix is to stop the file being openable by
the child's UID at all:

- a systemd credential (`LoadCredential=` / `SetCredential=`), which puts the
  secret in a file readable only by the service's own process, or
- an environment file owned by a different UID, with the engine started by a
  wrapper that reads it and drops privileges.

Neither is something this package can arrange for an operator. Until one of
them is in place, treat every secret in that file as reachable by every
`graphify` child the engine spawns.

## CLI flags (`graphify-mesh-sync`)

| Flag | Meaning |
|------|---------|
| `--once` | Single run (the only supported mode; there is no daemon loop). |
| `--dry-run` | Print every action; write nothing outside a private staging dir. |
| `--mesh-root PATH` | Override `GRAPHIFY_MESH_ROOT`. |
| `--scan-root PATH` | Add a scan root; repeat the flag for multiple roots. Explicit CLI roots take precedence over `GRAPHIFY_MESH_SCAN_ROOTS`, then legacy `GRAPHIFY_MESH_SCAN_ROOT`, then the current working directory. If every `--scan-root` value given is empty or whitespace-only, they are dropped and resolution falls through to `GRAPHIFY_MESH_SCAN_ROOTS`/`GRAPHIFY_MESH_SCAN_ROOT`/cwd exactly as if no `--scan-root` had been passed. |
| `--scan-depth N` | Override `GRAPHIFY_MESH_SCAN_DEPTH`; values are clamped to `1`–`8`. |
| `--registry PATH` | Override `GRAPHIFY_MESH_REGISTRY`. |
| `--skip-labeling` / `--no-skip-labeling` | Skip / enforce the non-placeholder community-name check. |
| `--skip-embedding` | Log-skip the embedding stage. |
| `--allow-shrink` | Authorize publishing a smaller graph than the previous generation; also authorizes per-repo (per-project) shrink acceptance — a shrunken per-repo graph is accepted and state advances instead of being refused. |
| `--extract-concurrency N` | Override `GRAPHIFY_MESH_EXTRACT_CONCURRENCY` (default 2, floor 1). |
| `-v`, `--verbose` | Debug logging. |

## CLI flags (`graphify-mesh-server`)

| Flag | Meaning |
|------|---------|
| `--transport {stdio,http}` | Override `GRAPHIFY_MESH_TRANSPORT`. |
| `--host HOST` | Override `GRAPHIFY_MESH_HTTP_HOST`. |
| `--port PORT` | Override `GRAPHIFY_MESH_HTTP_PORT`. |
| `--path PATH` | Override `GRAPHIFY_MESH_HTTP_PATH`. |
| `--allow-public-bind` | Override `GRAPHIFY_MESH_ALLOW_PUBLIC_BIND` (opt-in only; no flag to force it off once the env var is set). |

A flag beats its environment variable, which beats the default. There is no
`--token` flag: the bearer token is env-only (`GRAPHIFY_MESH_HTTP_TOKEN`), so
it never appears in a process listing.

## Discovery behavior

Scan depth is the maximum nesting of the **project directory** below a scan
root: depth `1` checks immediate child directories, while the old fixed scan
was equivalent to depth `2`. Discovery walks each root in sorted DFS preorder.
Hidden directories and directories named in `IGNORED_DIR_NAMES` (including
`.git`, `node_modules`, `vendor`, `dist`, and `build`) are pruned completely,
so projects at or below those names are not discoverable — this is a
behavior change from the old fixed-depth-2 scan, not a security measure.
Separately, directory symlinks are not followed during the walk, so a
symlinked project directory is not discovered; that restriction **is**
deliberate security hardening (a symlinked project dir could otherwise be
used to escape the scanned tree).

Multiple roots are scanned in their configured order. Overlapping or nested
roots share a depth-budget-aware visited set so subtrees are not walked
redundantly, while repeated links are still returned for duplicate reporting
and reconciliation. Per-walk and per-candidate OS errors are logged and
skipped without aborting the remaining discovery run. Both a candidate project
directory and its resolved `graphify-out` target must resolve under at least
one approved root or the candidate is rejected and reported. When multiple
discovered links resolve to one registered repo, reconciliation prefers the
link whose project directory matches the already-registered root.

Each scan root that reaches the walk (i.e. it resolved, is a directory, and
was not already covered by a prior root at equal or greater depth) logs its
own summary line with its candidate-dir count and discovered-link count;
roots that fail to resolve/stat, aren't a directory, or are already covered
are logged with a warning and skipped before that summary is ever produced. Of
the roots that do get a summary: a root that yields zero candidate
directories logs a warning (an empty or fully-pruned tree); a root that has
candidates but discovers zero `graphify-out` links logs a separate warning —
this can mean none of the candidates were actually graphify projects, but it
can equally mean per-candidate filesystem errors (each individually logged
and skipped, e.g. a race with a live tree) silently ate what would otherwise
have been valid links. The warnings catch different misconfiguration cases
and are not mutually exclusive with a successful run
on other roots.

## `Settings` fields (sync)

`graphify_mesh.sync.config.Settings` — resolved runtime configuration for one
pipeline run. Notable fields and derived paths:

- `mesh_root`, `registry_path` — base locations.
- `scan_roots: list[Path]`, `approved_roots: list[Path]` — ordered discovery
  roots and roots trusted by the path-traversal guard. These replace the old
  singular `scan_root` and `approved_root` fields.
- `scan_depth: int` — maximum project-directory nesting below each scan root;
  defaults to `4` and is clamped to `1`–`8`, including when `Settings` is
  constructed directly instead of through the environment parser.
- `graphify_bin`, `stale_threshold`, `dry_run`, `skip_labeling`,
  `skip_embedding`, `allow_shrink` — run behavior.
- `ollama_*` / `ollama_embed_*` — naming and embedding endpoints, models, and
  health timeouts (plus test-only injectable health checks). `ollama_api_timeout`
  is the per-labeling-call ceiling, separate from `ollama_health_timeout`.
- `keep_embedding_generations` — how many published generations' embedding
  shards to keep on disk (older ones are GC'd at publish time). Default: `2`.
- `keep_structural_generations` — how many structural generation directories
  (`global-graph.json` + `cross-project-overlay.json` + `lexical-index.json`,
  tens to 100+ MB each) to keep under `<global_dir>/generations/`.
  Default: `2`. GC semantics: pruning runs only **after** `flip_current`
  succeeds (`publish.prune_old_generations`) — it keeps the generation
  `current` points at plus the most recent complete generations up to the
  keep count, and also removes never-published generation dirs left behind
  by a crash (a dangling `*.tmp` file with no matching `.json`). The
  `current` generation is never removed, even when it is older than the
  keep window. A crash during pruning can strand extra generation dirs
  (wasted disk) but can never remove the generation `current` needs.
- `extract_concurrency` — max concurrent per-repo `graphify extract`/`update`
  children on the bounded thread pool (default 2, hard floor 1). Every
  child's RSS lands in the same cgroup `MemoryMax` as the parent sync
  process, so this is a tuned cap, never `len(repos)`.
- `extract_health_url`, `extract_health_api_key`, `extract_health_timeout`,
  `extract_health_enabled` — the extract-backend infra probe (plus the
  test-only injectables `extract_health_check`, a healthy/outage boolean,
  and `extract_health_probe`, the full tri-state, which wins when both are
  set). Disabled or an empty URL makes the feature inert; see the
  `GRAPHIFY_MESH_EXTRACT_HEALTH*` rows above.
- `infra_grace_hours` — how long `infra_*` repos stay exempt from the
  publish gate before counting toward the unrefreshed ratio. Default:
  `24.0`; `0` means they count immediately.

Derived path properties (all under `mesh_root`):

| Property | Location |
|----------|----------|
| `global_dir` | `<mesh_root>/graphify/global` |
| `generations_dir` | `<global_dir>/generations` |
| `current_symlink` | `<global_dir>/current` |
| `status_path` | `<global_dir>/status.json` |
| `state_path` | `<global_dir>/state/source-manifests.json` |
| `lock_path` | `<global_dir>/.graphify-mesh-sync.lock` |
| `naming_dir` | `<global_dir>/naming` |
| `embeddings_dir` | `<global_dir>/embeddings` |
| `manual_relations_path` | `<mesh_root>/bin/manual-relations.json` |
| `manual_relations_schema_path` | `<mesh_root>/bin/manual-relations.schema.json` |

## `ServerConfig` fields (server)

`graphify_mesh.server.config.ServerConfig` — `mesh_root` and `registry_path`,
plus derived `global_dir`, `current_symlink`, and `embeddings_current_symlink`
resolved from `GRAPHIFY_MESH_ROOT` / `GRAPHIFY_MESH_REGISTRY`, plus the
transport fields: `transport` (`"stdio"` or `"http"`), `http_host`,
`http_port`, `http_path`, `http_token` (`str | None`), and
`allow_public_bind`. `ServerConfig.from_env` raises `ConfigError` at startup
for an invalid transport, an out-of-range port, a path not starting with
`/`, a public bind without `allow_public_bind`, or `transport="http"` with no
usable token — never per request.

## `registry.json`

Source of truth for which repos are in the mesh. See
`examples/registry.example.json`.

```json
{
  "repos": [
    {
      "repo_id": "example-org.backend-a",
      "root": "/path/to/your/checkouts/backend-a",
      "collection_path": "/path/to/your/graph-mesh/graphify/example-org/backend-a",
      "enabled": true
    }
  ],
  "disabled": [],
  "external_roots": []
}
```

| Field | Meaning |
|-------|---------|
| `repos[].repo_id` | Stable logical id; becomes the node-id prefix / `repo` attribute in the merged graph. Must match `^[A-Za-z0-9][A-Za-z0-9._-]*$` (it is used as a filename, e.g. embedding shards) and be unique — duplicates are a load-time error. |
| `repos[].root` | The repo's checkout directory. Resolved and required to land under an approved root (a scan root or an `external_roots` entry), same as `collection_path`: the pipeline stat-walks this path and hands it to `graphify update`/`extract`. |
| `repos[].collection_path` | Directory holding that repo's `graph.json`. Two enabled entries whose paths RESOLVE to the same directory (symlinks and `..` segments included) block the run: concurrent per-repo workers would otherwise snapshot, rewrite and roll back the same `graph.json`. |
| `repos[].enabled` | If `false`, the repo is skipped. |
| `disabled` | List of `repo_id`s to force-disable. |
| `external_roots` | Additional approved roots for symlink resolution. |

`GRAPHIFY_MESH_APPROVED_ROOTS` (the `Settings.approved_roots` value) supplies
the discovery guard's approved roots and defaults to the scan roots.
`registry.json`'s `external_roots` extends the same containment policy for
registry-declared `collection_path` values that legitimately live elsewhere;
use the environment variable for discovered project/link locations and this
registry field for additional registered collection locations.

### File permissions in the mesh `bin/` directory

The engine audits the modes of its own configuration files once at startup,
under `graphify-mesh-sync` (including `--dry-run`, so a dry run is a way to
check permissions without publishing) and under `graphify-mesh-server`. It
warns and nothing else: these files belong to the operator, so the engine
never changes a mode and never blocks a run over one.

| File | Expected mode | Warns when |
|-------|---------------|------------|
| Any `*.env` beside `registry.json` | `0600`, owned by the service user | group or other hold any permission bit |
| `registry.json` | not group- or other-writable, e.g. `0644` | group or other can write |
| `manual-relations.json` | not group- or other-writable, e.g. `0644` | group or other can write |

The env file holds `GRAPHIFY_MESH_OLLAMA_API_KEY` and
`GRAPHIFY_MESH_HTTP_TOKEN`, so anyone who can read it has those credentials.
systemd does not tell the process which file it parsed as `EnvironmentFile=`,
so the audit checks every `*.env` in the directory that holds `registry.json`
instead. It only stats those files; it never opens one.

`registry.json` carries more authority than a list of paths suggests. Its
`repos[].root`, `repos[].collection_path`, `disabled` and `external_roots`
decide which directories get stat-walked and handed to `graphify extract`,
and they also supply the child sandbox's containment roots and its writable
bind targets. Anyone who can write the registry can redirect extraction and
steer a writable bind, which is why group- or other-writable is a finding
while world-readable is not: the file holds paths, not secrets.

## `manual-relations.json`

Human-declared cross-project overlay edges the sync engine cannot infer,
validated against `examples/manual-relations.schema.json`. See
`examples/manual-relations.example.json`.

Top level is `{ "relations": [ ... ] }`. Each relation has:

| Field | Meaning |
|-------|---------|
| `type` | One of `depends_on`, `similar_approach`, `provides_api`, `consumes_api`. |
| `source`, `target` | A logical ref: `{ repo, source_file, qualified_label, signature? }`. Both must resolve against the current generation's per-repo graphs at load time — a dangling reference is a hard error. |
| `confidence` | Optional number in `[0, 1]`. |
| `evidence` | Optional human-readable justification string. |

## `naming-state.json`

`<global_dir>/naming/naming-state.json` — the naming stage's own state, written
and read only by this package. It replaces graphify's `.graphify_labels.json` /
`.graphify_labels.json.sig` sidecars.

```json
{
  "schema_version": 1,
  "merged_fingerprint": "<publish.output_hash of the stripped merged graph>",
  "clustering": {"backend": "louvain", "resolution": 1.0},
  "labeling": {
    "backend": "ollama",
    "model": "qwen2.5-coder:14b",
    "base_url": "http://localhost:11434/v1"
  },
  "communities": {
    "42": {"sig": "<sha256 of the sorted member ids>",
           "name": "Auth Token Flow",
           "provisional": false}
  },
  "assignments": {"example-org.backend-a::src/Auth.php:AuthService": 42}
}
```

- `merged_fingerprint` gates the run-level reuse branch: a matching fingerprint,
  matching recipes, assignments covering exactly the graph's node ids and no
  provisional community together mean the stage applies the stored result
  without clustering and without any backend call.
- `clustering` and `labeling` are the two recipes, compared separately.
  `clustering` gates whether the stored `assignments` may seed community-id
  stabilization; `labeling` gates whether stored names may be carried. The
  endpoint is part of the labeling recipe, because the same model on a different
  server is a different labeler.
- `communities` is keyed by community id, but names travel between runs by
  `sig`, since ids renumber whenever membership shifts. `provisional: true`
  means no LLM has named that community yet — a hub fallback from an outage, or
  an entry migrated from the old sidecars — and the next healthy run relabels
  exactly those.
- The file is written through a temp file and `os.replace`. It is read back only
  if `schema_version` matches; an absent, unreadable or unknown-version file is
  treated as "no state", never as an error.

Deleting it costs one relabeling run and nothing else: the next run re-clusters,
finds no carried names and labels every community from scratch. Community ids
renumber in that run, because id stabilization has nothing to stabilize against.

The stage no longer reads or writes `.graphify_labels.json`,
`.graphify_labels.json.sig`, `graphify-out/graph.json` or `GRAPH_REPORT.md` in
that directory. If they exist when `naming-state.json` does not, the first run
seeds names from the two label sidecars (only community ids present in both, and
marked `provisional`, since upstream wrote hub fallbacks into them too — on this
mesh's migration run 536 of 984 signatures carried). All four files are deleted
after the first fully successful run.

## `generation-manifest.json`

Written by the sync pipeline into every generation directory
(`<global_dir>/generations/<generation_id>/generation-manifest.json`) at
publish time, alongside `global-graph.json`, `cross-project-overlay.json`,
and `lexical-index.json`. Notable field:

| Field | Meaning |
|-------|---------|
| `artifact_sha256` | Map of `{"<artifact filename>": "<sha256 hex of the raw file bytes>"}` for the artifacts in the same generation directory. Written at publish time; the server uses it for cheap consistency verification of a generation (hash the raw bytes of each listed artifact and compare), falling back to the legacy canonical hash for older generations whose manifest predates this field. |
