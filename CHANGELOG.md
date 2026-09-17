# Changelog

## Unreleased

### Features and behavior changes

- Added: optional per-call `cwd` argument on `search` and `context_pack`.
  `scope: "current"` resolves the CLIENT's directory, which is this argument
  when the caller sends one and the server process's cwd otherwise. The server
  documented "one process per client session"; run as one shared daemon for
  many sessions (a systemd user unit, cwd `$HOME`), `Path.cwd()` matched no
  registered repo root, so every implicit-scope call failed closed with
  "cannot resolve implicit scope". Resolution is unchanged otherwise: the
  directory is only matched against registered roots in `registry.json`, a
  relative or empty `cwd` is a tool error, and an unregistered one still fails
  closed rather than widening to a global search.
- Added: `graphify-mesh-server --transport http`, a shared streamable-HTTP
  daemon (stateless, `mcp` SDK, bearer token required via
  `GRAPHIFY_MESH_HTTP_TOKEN`, loopback bind with Host/Origin validation) so
  one process can serve every local agent instead of one process per client
  session. stdio stays the default and unchanged. With a shared daemon, `cwd`
  above stops being optional in practice: nothing injects the caller's
  directory automatically, so the caller passes it on every call. The tool
  descriptions for `search`/`context_pack` were reworded to say so, replacing
  the earlier "injected by the session proxy" wording.
- Changed: the server's own JSON-RPC transport (`server/protocol.py`) is
  removed; both stdio and HTTP now run through the `mcp` SDK's low-level
  `Server`, registered once from the same tool schemas/`call_tool` pair so
  the two transports cannot serve different tools. `mcp`, `starlette`, and
  `uvicorn` become required dependencies. `server/stdio_guard.py` keeps the
  line-size cap and the malformed-frame `-32700`/`-32600` responses the
  retired transport provided. An unrecognized JSON-RPC method now answers
  `-32602` instead of the retired dispatcher's `-32601` — accepted, since
  matching `-32601` would mean tracking the SDK's own valid-method set.
- Changed: generation reads and reloads run under a writer-preferring
  read/write lock (`server/rwlock.py`) instead of being fused, so concurrent
  reads in the shared daemon no longer serialize behind each other while a
  reload is in flight. The registry cache (`server/scope.py`) and the
  per-generation similarity index cache (`server/similar.py`) are now
  individually lock-guarded against duplicate concurrent builds.
- Fixed: in HTTP mode one slow tool call blocked every other client. The
  synchronous `call_tool` ran inline in the async handler, so a `search`
  waiting on the embedding endpoint held the event loop for that whole
  timeout — other clients' calls, initialization and even the `401` for an
  unauthenticated request could not progress. Tool execution now runs off the
  loop under a bounded limiter (`sdk_app.TOOL_WORKER_LIMIT`, 8 threads).
- Fixed: a stale-capture race in `GenerationStore.ensure_fresh` could move the
  store onto an OLDER generation. The double check under the write lock
  compared the signature captured BEFORE the wait, so a thread that waited
  through a publish another thread had already loaded concluded a reload of
  its captured generation was due. The decision now comes from a fresh stat
  taken under the write lock; `_try_reload` still reads every artifact from
  the one realpath it is handed, so a publish landing mid-load cannot mix two
  generations.
- Fixed: a JSON object that is not a legal JSON-RPC 2.0 envelope
  (`{"jsonrpc":"2.0","id":7,"method":"tools/call","params":[]}`) got no
  correlated response. It reached the SDK's envelope validation, which raises
  instead of answering request 7, so the client waited for its own timeout.
  Both transports now check the envelope themselves
  (`server/frames.py:envelope_error`) and answer `-32602` for a params-shape
  error, `-32600` otherwise, preserving the request id. The existing `-32700`
  and `-32600` answers are unchanged.
- Fixed: HTTP parse and envelope errors returned the SDK's `str(JSONDecode
  Error)` / `str(ValidationError)`, which quote validation details and
  excerpts of the input, where stdio answers generically. The HTTP frame guard
  now answers the same generic coded error and logs the detail instead. The
  token gate stays outermost and unchanged: an unauthenticated request is
  still answered before a byte of its body is read.
- Changed: the incoming-message cap is one number for both transports
  (`server/frames.py:MAX_MESSAGE_BYTES`, 4 MiB), enforced on bytes actually
  received. HTTP previously took whatever default the installed SDK had, so
  the two transports stated different limits and the HTTP one moved with the
  SDK; stdio previously capped a line at 10 MB. The lower ceiling is
  deliberate: the shared daemon is one process serving every local agent, so
  its memory is a machine-wide resource, and no legitimate JSON-RPC frame for
  these five tools approaches 4 MiB.
- Added: the HTTP frame guard bounds concurrent body buffering with an
  `anyio.Semaphore` of `http_app.BODY_BUFFER_SLOTS` (8), held only while it
  reads a body and released the moment the body reaches the layer below, so
  its share of request bodies is capped at 8 x 4 MiB = 32 MiB. It queues
  rather than rejecting, and an idle keep-alive connection holds no slot.
  uvicorn's `limit_concurrency` is not used: it answers `503` on a count of
  open CONNECTIONS, which would reject a long-lived local client with nothing
  in flight. The guard also drops its parsed object and hands its buffer down
  without copying it, so it no longer keeps three representations of a body
  alive across the downstream call. Total inbound memory is still not a single
  number: the parsed-object cost is not a fixed multiple of the byte cap, and
  nothing here bounds how many requests sit below the guard at once.
- Changed: the `mcp` floor is `>=1.30`, up from `>=1.12`. The HTTP adapter
  passes `max_request_body_size`, which older releases do not accept, and an
  SDK that raises `TypeError` before the port opens is worse than a resolver
  conflict. `session_idle_timeout=None` was dropped instead of pinned: the SDK
  documents it as unused in stateless mode. `tests/server/test_sdk_surface.py`
  pins both facts.
- Fixed: the `cwd` hook (`examples/hooks/graphify-mesh-cwd.py`) overwrote an
  explicitly supplied invalid `cwd` — `""`, whitespace or a number became the
  session directory, so a call that should have failed validation was answered
  for a different repository. It now fills `cwd` only when the key is absent,
  and still fails open on anything unexpected.

- Added: infra-outage classification for the extract backend. A health probe
  (GET `{base}/models`, the same endpoint the naming stage checks) now runs
  before each ollama-backed `extract`/`bootstrap` child is spawned — a
  per-launch re-probe, so both a start-of-run outage and mid-run recovery are
  caught — and again after a child fails. The probe is tri-state: a connect
  error, DNS failure, timeout, or 5xx is an **outage**; any **4xx** means the
  probe itself is misconfigured (bad key, wrong path, TLS/gateway) and fails
  **open** — logged once per run at ERROR naming the URL and status code,
  children spawn normally, real failures stay plain `failed`. Preflight-outage
  repos become `infra_skipped` without spawning the child; a failure with a
  down backend becomes `infra_failed` (`bootstrap_failed`/`shrink_refused` are
  never reclassified, and only genuine bootstrap failures are reported in
  `auto_add_failed`). Both statuses keep an existing last-good graph flowing
  to the merge (a first-time bootstrap has none and is omitted from the
  generation until the backend recovers) and are removed from the publish-gate
  ratios entirely — numerator **and** denominator, so in-grace repos never
  dilute the ratios for the repos that still count — for
  `GRAPHIFY_MESH_INFRA_GRACE_HOURS` (default `24`, finite and `>= 0`, measured
  from the outage start persisted as `infra_since`, cleared on the next
  successful refresh); past the grace they count toward the unrefreshed ratio.
  When an outage means not a single actionable repo was refreshed, the run is
  a noop: merge/naming/embedding/overlay/lexical-index/validate/publish are
  skipped with `publish_blocked_reason` "noop: infra outage — no repo
  refreshed, nothing to integrate" — unless a repo was removed (the removal
  must still be merged out and published) or any infra repo is past its grace
  window (a >grace total outage ends blocked on the unrefreshed threshold
  instead of looking like routine idling). When both publish gates trip in
  one run, `publish_blocked_reason` names both, joined with `"; "`.
  `status.json` now also reports `publish_blocking_repos`,
  `unrefreshed_repos`, `infra_repos_in_grace`, and `infra_repos_past_grace`.
  The probe URL falls back to `OLLAMA_BASE_URL` (the extract child's own
  endpoint env, deliberately not `GRAPHIFY_MESH_OLLAMA_BASE_URL`) only when
  `GRAPHIFY_MESH_EXTRACT_HEALTH_URL` is entirely unset — set-but-empty keeps
  the feature inert — with `_API_KEY`/`_TIMEOUT` companions; setting
  `GRAPHIFY_MESH_EXTRACT_HEALTH=off` (or an empty URL) disables the feature
  entirely, restoring the previous behavior where every failure stays
  `failed`. Motivation: a remote-Ollama blip marked 8/16 repos `failed` within
  seconds and blocked publish at 50% > 40% although the merged graph was
  complete — the healthy repos' work was discarded.
- Fixed: the manifest digest now keys on file **content** instead of
  `(size, mtime_ns)`. A `git checkout`, an editor save-without-edit, or a
  regenerated lockfile used to bump `semantic_hash` with byte-identical content,
  scheduling a full non-deterministic `graphify extract` whose output legitimately
  differs run to run. Measured cost is ~66 MB hashed per four repos — seconds
  against a multi-minute extract. `GRAPHIFY_MESH_CONTENT_DIGEST=0` restores the
  old mtime behaviour. An unreadable file degrades to the mtime marker for that
  entry instead of aborting the manifest.
- Fixed: a shrink refusal now advances *attempted* state (`refused_semantic_hash`
  + `refusal_streak`) while leaving accepted state and the last-good graph
  untouched. Previously a refusal recorded nothing, so `decide_action` saw a
  changed source forever and the repo re-extracted and was re-refused on every
  run — a permanent retry loop that also kept it in the stale bucket. After
  `REFUSAL_RETRY_LIMIT` refusals of the same digest, the repo holds its last-good
  graph until the source actually changes. The deterministic AST path stays
  available while the LLM path is suppressed.
- Changed: the publish gate no longer treats all stale statuses alike.
  `bootstrap_failed` (no graph) and `shrink_refused` (suspect graph) keep the
  `STALE_PUBLISH_THRESHOLD` (30%) gate; `failed` (refresh did not complete, e.g.
  an extract timeout, leaving valid last-good data) is gated separately at
  `UNREFRESHED_PUBLISH_THRESHOLD` (40%, env `GRAPHIFY_MESH_UNREFRESHED_THRESHOLD`).
  A couple of slow repos no longer veto a generation that is strictly newer for
  everyone else — an unpublished generation also freezes the embedding channel —
  while a systemic failure rate still blocks. `publish_blocking_repos` and
  `unrefreshed_repos` are reported separately so a refusal names its cause.
- Added: `GRAPHIFY_MESH_CLI_TIMEOUT` makes the per-subprocess timeout
  configurable (default unchanged at `900` s, clamped to `60`–`7200`). `extract`
  wall time scales with repo size and Ollama load, and a repo that needs longer
  failed with returncode `124` on every run without ever advancing state, so it
  counted toward the stale-ratio publish gate indefinitely.
- Changed: the per-repo shrink guard now tolerates a configurable fraction of
  loss instead of refusing any decrease. `extract` re-derives entities with an
  LLM and is non-deterministic, so an unchanged repo varies a few percent per
  run; because a refusal does not advance per-repo state, a strict guard made
  the repo re-extract and wobble again on every run, leaving it permanently
  stale and blocking publishes indefinitely. Counts below
  `old * (1 - GRAPHIFY_MESH_SHRINK_TOLERANCE)` (rounded up, default `0.10`) are
  still refused, so material loss is caught, and graphs small enough that the
  floor equals the old count keep exact-match semantics. Set the env var to `0`
  for the previous absolute behavior.
- Changed: `--allow-shrink` now also authorizes the per-repo shrink guard —
  an operator-authorized run accepts a legitimately smaller per-repo graph
  (status `updated`, state advanced) instead of refusing it on every sync
  run; without the flag, per-repo shrink refusal is unchanged.
- Added: filesystem discovery supports multiple ordered scan roots and a
  configurable project-directory nesting depth (default `4`, clamped to
  `1`–`8`). `--scan-root` is repeatable, `--scan-depth` sets the depth,
  `GRAPHIFY_MESH_SCAN_ROOTS` accepts a normalized colon-separated root list,
  `GRAPHIFY_MESH_APPROVED_ROOTS` configures trusted discovery roots, and
  `GRAPHIFY_MESH_SCAN_DEPTH` configures the depth. The legacy single-path
  `GRAPHIFY_MESH_SCAN_ROOT` remains a fallback; with no configured root,
  discovery still defaults to the current working directory.
- Breaking: `Settings.scan_root` and `Settings.approved_root` were renamed to
  `Settings.scan_roots` and `Settings.approved_roots` and now hold
  `list[Path]`; there is no backward-compatibility shim. `Settings` also gains
  `scan_depth`. `discover_filesystem` now takes plural `scan_roots` and
  `approved_roots` arguments plus `depth`, so direct callers must update.
- Changed: discovery now prunes hidden directories and directories named in
  `IGNORED_DIR_NAMES`, so projects at or below names such as `vendor`,
  `build`, `dist`, `node_modules`, and `.git` are no longer discoverable.
  Directory symlinks are no longer followed during the discovery walk, so a
  symlinked project directory is not discovered; this is security hardening.
  Deeper scans may surface new `unregistered_discovered` or duplicate report
  rows for reconciliation; these rows are report-only and are not errors.

## 0.0.6

### Performance

- Changed: the naming stage relabels only communities whose membership
  fingerprint changed, instead of re-labeling every community each run;
  per-project extraction runs in a bounded parallel pool — together roughly
  3x lower sync wall time.
- Changed: lexical index schema v3 and embedding vectors stored as `.npy`
  shards (memory-mappable by the server) to cut sync peak RSS.
- Changed: all large published artifacts (`global-graph.json`,
  `cross-project-overlay.json`, `lexical-index.json`) are written via streamed
  `JSONEncoder.iterencode` with compact separators instead of a whole-artifact
  `json.dumps` string — no full-artifact string in RAM and smaller files on
  disk; only `generation-manifest.json` keeps pretty-printing. `output_hash`
  streams the same encoding into the hasher (digest byte-identical to the
  previous implementation).
- Added: shared bounded source-line cache (`sync/source_cache.py`), keyed on
  path + mtime + size, used by every snippet builder — each source file is
  read at most once per pipeline run instead of once per node per stage.
- Changed: source-tree scans (`compute_source_manifest`, overlay API
  extraction) use a single pruned `os.walk` per repo instead of `rglob`
  passes that materialized ignored trees (`node_modules`, `.git`, vendor);
  manifest digests are byte-identical for unchanged trees. Overlay API
  extraction also shares file text between provider/consumer passes and
  finds enclosing classes via a precomputed position list + bisect instead
  of rescanning the file prefix per route.
- Changed: per-repo `graph.json` files are read and hashed once
  (hash + parse + node/edge counts from one buffer); publish reuses hashes
  computed during the per-project stage instead of re-reading every graph.
- Changed: server vector search takes per-repo `np.argpartition` shortlists
  instead of a full Python sort over every embedded node per query; MMR
  selection tracks running max-similarity incrementally (output verified
  identical); `find_similar` and its exact-label fallback use lazy
  per-generation indexes instead of full overlay/node scans per call;
  `context_pack` builds evidence cards lazily inside the token-budget loop
  (no snippet I/O past the cutoff); registry entries are cached keyed on
  mtime + size + inode.

### Memory

- Changed: `VectorShards.normalized()` no longer materializes a float32 copy
  of every mmap'd shard — the matrix stays memory-mapped and only per-row
  norms are cached (the copy previously multiplied across one server process
  per client session).
- Changed: the pipeline drops the pre-naming merged graph and the per-repo
  graph map as soon as their last consumers finish, instead of holding
  multiple whole-graph copies through embed/overlay/lexical; degraded naming
  restore mutates the graph in place instead of copying every node while the
  previous generation is also resident.

### Reliability and correctness

- Added: `generation-manifest.json` now records `artifact_sha256` (sha256 of
  each artifact's raw bytes, hashed as written) and `sync_started_at`. The
  server verifies artifacts by hashing raw bytes in one pass — an artifact
  present on disk without a hash entry is a consistency error — and falls
  back to the legacy canonical hash only for older generations. The manifest
  is written last, so an interrupted publish is always detectably incomplete.
- Fixed: the server read artifacts through the live `current` symlink, so a
  publish flipping mid-load could mix files from two generations; all
  artifacts are now read from the pinned resolved path with generation-id
  cross-checks for the overlay and embeddings (mismatched embeddings drop
  the vector channel with a degraded marker instead of serving wrong data).
- Fixed: store-level degraded markers (e.g. serving the previous generation
  after a rejected reload) were never surfaced; every tool response's
  `degraded` list now includes them.
- Fixed: JSON-RPC protocol gaps — unparseable input now gets `-32700`,
  non-object/batch messages `-32600`, mistyped `params`/tool arguments
  `-32602`/typed tool errors instead of generic internal errors, and
  notifications (including id-less known methods) never receive a response.
- Fixed: embedding batch failures retry once with backoff before degrading,
  and the fallback reuses the already-loaded previous shard; `--dry-run` no
  longer performs live embedding calls.
- Fixed: `prune_old_generations` completeness is manifest-aware — it removes
  generations that a crash left without declared artifacts, but no longer
  deletes legacy-shaped generations (graph + manifest only) that are valid
  rollback targets.
- Fixed: context-pack snippets are read from the live working tree; cards
  now carry `snippet_source`/`snippet_stale` provenance (staleness measured
  against `sync_started_at`) instead of presenting possibly-shifted lines as
  generation data.
- Fixed: validation error lists are capped (50 + summary line) so a broken
  merge cannot write millions of strings into `status.json`; a dead
  positional-identity branch in validation and a dead conditional in the
  state-advancement path were removed.

### Tooling and tests

- Added: tests for the transaction lock (cross-process contention, release on
  exception, CLI exit code 3), crash-mid-publish atomicity (before and inside
  the symlink flip), and the JSON-RPC error contract.
- Changed: CI uses pip caching, cancels superseded runs, measures coverage,
  and enforces `ruff format`; the release workflow no longer installs the
  upstream dependency from an unpinned git HEAD and no longer force-pushes
  over `gh-pages` history.
- Added: `dev` extra (`test` + `lint`); shared test fixtures consolidated
  under `tests/fixtures/`.

## 0.0.5

- Changed: package version is derived from git tags via `hatch-vcs` instead
  of a hand-maintained `__version__`.
- Added: release-time guard that rejects a release whose tag and resolved
  package version disagree.
- Changed: GitHub Actions steps bumped to current action versions.

## 0.0.4

- Fixed: `__version__` reported 0.0.2 regardless of the installed release.
- Fixed: unbounded disk growth — structural generation directories
  (`global-graph.json` + overlay + lexical-index, 200-500MB each) were never
  garbage-collected, accumulating one per sync run. Added
  `publish.prune_old_generations` (keeps last 2, matching the embeddings GC
  convention); it also removes generations left dangling by an interrupted
  publish.
- Fixed: excessive peak memory in `build_lexical_index` — postings were
  stored as `{"repo","key","field","weight"}` dicts (weight derivable from a
  3-entry constant) with the raw accumulation structure and the materialized
  output held simultaneously. Switched to compact `[repo, key, field]`
  arrays, dedup-during-accumulation, and pop-as-converted — ~43% lower peak
  memory and ~18-19% smaller on-disk index on fixture benchmarks.
  Lexical-index schema bumped to v2; the MCP server rejects a v1-shaped
  index instead of silently misreading it.
- Hardened: `_write_json_atomic` now fsyncs file data before the rename, so
  an interrupted write can't leave a durable-rename but garbage-content file
  behind an already-flipped `current`.

## 0.0.2

- Fixed: `--skip-labeling` never actually gated the naming stage — it only
  affected a log message and a validation bypass, so `graphify cluster-only`/
  `label` ran regardless of the flag. The skip path now bypasses the naming
  call entirely, guaranteeing zero network calls when passed.
- Fixed: a mid-run network failure (e.g. a DNS blip) during `graphify label`
  or an `/api/embed` batch call raised an uncaught exception and crashed the
  whole pipeline, discarding already-completed work. Both stages now degrade
  gracefully instead:
  - `naming.py`: falls back to `LABELING_DEGRADED`, keeping names already on
    disk rather than crashing.
  - `embedding.py`: new `EMBED_PARTIAL` status — repos already embedded this
    run keep their real vectors; the failed repo and any not-yet-reached
    repos fall back to their previous published shard.
- Fixed: `graphify cluster-only`/`label` can hit their own internal
  shrink-guard (e.g. after a malformed node causes a node-count mismatch),
  print "Done" and exit 0, yet silently write zero `community_name` values.
  `run_naming` now verifies real names actually landed before reporting
  `LABELING_OK`, degrading instead of returning a false success.
- Added: per-repo and per-batch progress logging for the sync/naming/
  embedding stages (previously silent for the full run duration).
- Added: regression tests for the three fixes above.

## 0.0.1

Initial extraction from knowledge-base's in-repo sync engine into a
standalone installable package.
