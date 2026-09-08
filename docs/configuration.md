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
| `GRAPHIFY_MESH_OLLAMA_EMBED_BASE_URL` | sync (embed) | `http://localhost:11434` | **Native** `/api/embed` endpoint (no `/v1` suffix). |
| `GRAPHIFY_MESH_OLLAMA_EMBED_MODEL` | sync (embed) | `qwen3-embedding:0.6b` | Embedding model. |
| `GRAPHIFY_MESH_OLLAMA_EMBED_HEALTH_TIMEOUT` | sync (embed) | `3.0` | Seconds to wait on the embed-stage health check before degrading. |
| `GRAPHIFY_MESH_EXTRACT_HEALTH` | sync (extract guard) | (enabled) | Disable knob for the extract-backend health probe: `off`/`0`/`false`/`no` turns it off. Disabled — or an empty probe URL (see `GRAPHIFY_MESH_EXTRACT_HEALTH_URL`) — makes the whole infra-outage feature inert: no probes are ever attempted, every child failure stays plain `failed`, and no `infra_*` statuses (and thus none of the grace-window gate handling) can be produced — behavior identical to before the guard existed. |
| `GRAPHIFY_MESH_EXTRACT_HEALTH_URL` | sync (extract guard) | `OLLAMA_BASE_URL` | Base URL probed (GET `{base}/models`) before each ollama-backed `extract`/`bootstrap` child is spawned, and again after a child fails, to classify outages as `infra_skipped`/`infra_failed` instead of `failed`. The probe is tri-state: a connect error, DNS failure, timeout, or 5xx is an **outage**; any **4xx** (bad key, wrong path, TLS/gateway misconfig) means the *probe* is misconfigured, not the backend — logged once per run at ERROR (naming the URL and status) and failed **open**, so children spawn normally and real failures stay plain `failed`. Falls back to `OLLAMA_BASE_URL` — the env var the extract **child** itself resolves its endpoint from, deliberately not `GRAPHIFY_MESH_OLLAMA_BASE_URL` (this package's own naming-stage endpoint, which can be a different host) — only when this var is **entirely unset**; set-but-empty pins the URL to `""` and keeps the feature inert. Empty (however arrived at) means inert. |
| `GRAPHIFY_MESH_EXTRACT_HEALTH_API_KEY` | sync (extract guard) | `OLLAMA_API_KEY` (else empty) | Bearer key sent with the probe request. |
| `GRAPHIFY_MESH_EXTRACT_HEALTH_TIMEOUT` | sync (extract guard) | `5.0` | Seconds to wait on the extract-backend probe. Same validation as the other `*_HEALTH_TIMEOUT` vars: must be a number, `> 0` and `<= 300`, anything else raises at startup. |
| `GRAPHIFY_MESH_INFRA_GRACE_HOURS` | sync (publish gate) | `24.0` | Grace window during which `infra_*` repos are removed from **both** publish-gate ratios entirely — numerator **and** denominator, so in-grace repos never dilute the ratios computed over the repos that still count (`infra_since` — the start of the repo's outage — is recorded in per-repo state, preserved across runs, and cleared on the next successful refresh). Past the grace they count toward the **unrefreshed** ratio, in both numerator and denominator (their last-good graph is old, never suspect); once any repo is past its grace, a total outage also stops short-circuiting as the zero-refresh noop and ends the run blocked on the unrefreshed threshold instead. `0` is allowed and means infra statuses count toward the gate immediately. Negative, non-finite (`nan`/`inf`), or unparseable values raise at startup, naming the variable. |
| `GRAPHIFY_MESH_EXTRACT_CONCURRENCY` | sync (extract) | `2` | Max concurrent per-repo `graphify extract`/`update` children (bounded thread pool). Every child's RSS lands in the same cgroup `MemoryMax` as the parent sync process (steady-state parent peak observed ~1.9G inside a 4G cgroup), so this is a tuned, hard-capped setting — never derive it from the repo count. `subprocess.run` releases the GIL, so a thread pool (not a process pool) is sufficient. Unparsable/zero/negative values fall back to the hard floor of `1` (fully sequential) rather than raising. |
| `GRAPHIFY_MESH_TLS_MODE` | sync (all outbound HTTPS) | `strict` | How outbound HTTPS certificates are verified on the naming/embed health probes and the embed calls. `strict`: stock `ssl.create_default_context()` — full chain + hostname verification plus the strict X.509 extension profile. `relaxed`: clears `VERIFY_X509_STRICT` **only**; chain verification against the system CA store and hostname checking stay on. Use it on a host behind a TLS-intercepting corporate proxy, where the re-signed certificate omits the Authority Key Identifier and Python 3.13+ therefore rejects a chain that `curl` and Node accept (`certificate verify failed: Missing Authority Key Identifier`) — without it the probes report a reachable backend as unhealthy and the pipeline runs permanently degraded. `insecure`: the `curl -k` equivalent — `check_hostname=False`, `verify_mode=CERT_NONE`, every certificate accepted including an attacker's, logged at WARNING on each context build; for a diagnostic run on a trusted network only. An unrecognized value raises at call time naming the variable. Note this covers only calls this package makes itself: the `graphify` **child** process resolves TLS on its own, so a proxied host needs the child pointed at an endpoint it can reach (e.g. a local HTTP shim). |
| `GRAPHIFY_MESH_TLS_INSECURE` | sync (all outbound HTTPS) | unset | Boolean shorthand for `GRAPHIFY_MESH_TLS_MODE=insecure` (`1`/`true`/`yes`/`on`). An explicit `GRAPHIFY_MESH_TLS_MODE` wins over it. |
| `GRAPHIFY_BIN` | sync | `graphify` | Name/path of the upstream `graphify` binary. |
| `GRAPHIFY_NO_BACKUP` | sync | (set to `1` on child calls) | Suppresses `graphify`'s dated backup dirs; the sync engine always sets this on the graphify subprocesses it spawns. |

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
  health timeouts (plus test-only injectable health checks).
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
plus derived `global_dir`, `current_symlink`, and `embeddings_current_symlink`.
Resolved from `GRAPHIFY_MESH_ROOT` / `GRAPHIFY_MESH_REGISTRY`.

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
| `repos[].root` | The repo's checkout directory. |
| `repos[].collection_path` | Directory holding that repo's `graph.json`. |
| `repos[].enabled` | If `false`, the repo is skipped. |
| `disabled` | List of `repo_id`s to force-disable. |
| `external_roots` | Additional approved roots for symlink resolution. |

`GRAPHIFY_MESH_APPROVED_ROOTS` (the `Settings.approved_roots` value) supplies
the discovery guard's approved roots and defaults to the scan roots.
`registry.json`'s `external_roots` extends the same containment policy for
registry-declared `collection_path` values that legitimately live elsewhere;
use the environment variable for discovered project/link locations and this
registry field for additional registered collection locations.

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

## `generation-manifest.json`

Written by the sync pipeline into every generation directory
(`<global_dir>/generations/<generation_id>/generation-manifest.json`) at
publish time, alongside `global-graph.json`, `cross-project-overlay.json`,
and `lexical-index.json`. Notable field:

| Field | Meaning |
|-------|---------|
| `artifact_sha256` | Map of `{"<artifact filename>": "<sha256 hex of the raw file bytes>"}` for the artifacts in the same generation directory. Written at publish time; the server uses it for cheap consistency verification of a generation (hash the raw bytes of each listed artifact and compare), falling back to the legacy canonical hash for older generations whose manifest predates this field. |
