# Changelog

## Unreleased

### Features and behavior changes

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
