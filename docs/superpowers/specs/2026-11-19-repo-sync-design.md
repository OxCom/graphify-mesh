# `graphify-mesh-sync --repo <id>`: fast single-repo refresh

Status: draft for approval, 2026-11-19. No code before approval.

## Intent

After a `git checkout` in one repo, the published graph should reflect that
repo within 1 minute. Today the only path is a full sync: 4.5–8 min wall,
2.6–3.3 GB peak on a normal run, 1h01m after the test re-include of
2026-11-19. The hourly timer (`OnUnitInactiveSec=1h`,
`RandomizedDelaySec=300`) stays the main producer.

## Goals

- `graphify-mesh-sync --once --repo <id>` publishes a new generation in under
  60 s when only `<id>` changed.
- The result is identical to what a full sync would publish for the same inputs.
- Every stage reports wall time, so the budget is measured, not guessed.

## Non-goals

- No watcher and no git hook in this change. A hook can call `--repo` later.
- No change to search ranking or to what the graph contains.
- No parallel runs: one sync at a time, as today.

## Anti-goals

- Patching the previous generation in place. That breaks hard invariant 2
  (every generation rebuilt from empty) in `CLAUDE.md`.
- A `--repo` generation that differs from a full-sync generation for the same
  inputs. Readers would see the graph change on the next timer run with no
  source change.
- Loosening the shrink guard for the other 24 repos because one repo was
  selected.

## Approaches

1. **Per-repo caches plus a thin `--repo` flag (chosen).** Every stage still
   rebuilds the generation from empty, but reuses per-repo results keyed by
   that repo's input hash. `--repo` only limits which repos get the
   per-project `graphify` step; the other stages run as normal and hit their
   caches. The timer run gets the same speed-up.
2. **Patch the previous generation.** Swap one repo's slice in the published
   files. Fastest, but violates invariant 2, and the overlay and lexical index
   are cross-repo, so they would still need a rebuild.
3. **No flag, faster full run only.** Same caches as 1, no flag. A checkout
   still waits for every repo's digest walk and `graphify update` decision.

Option 1 was picked on 2026-11-19, starting with per-stage timing.

## Design

### Step 0: per-stage timing (first deliverable, ships alone)

`rss.StageRssTracker.mark()` (`src/graphify_mesh/sync/rss.py:57`) is already
called at every stage boundary (`pipeline.py:1623` extract, `:1684` merge,
`:1796` naming, `:1896` embedding, `:1916` overlay, `:1935` lexical_index,
`:1983`/`:2158` validate_publish). Add `wall_s` and `cpu_s` (from
`time.monotonic()` and `time.process_time()`, plus child CPU from
`os.times()`) to each mark, log them on the existing stage summary line and
keep them in `report.stage_rss`. Discovery gets its own mark.

Then take three measurements on the live corpus: one no-change run, one run
after a one-file edit in a mid-size repo, and one in cem.hub. Those numbers
decide the order of steps 2–5. A stage under 2 s gets no cache.

### Step 1: the `--repo` flag

- `cli.py` gains `--repo <id>`, repeatable. Each id must be an enabled
  registry entry; an unknown id exits 2 before taking the lock.
- Discovery and reconcile run as today: they are needed for the registry
  containment check and cost little. Step 0 checks this.
- The per-project step computes the source digest and runs `graphify` only
  for selected repos. Every other repo keeps its last accepted `graph.json`
  and state entry untouched, as if its digest had not changed. Unselected
  repos are therefore never stale merely because of `--repo`.
- Merge, recluster, naming, embedding, overlay, lexical index, validate and
  publish all run over all repos, as today.
- A `--repo a` run can publish while repo `b` has an unsynced change. That
  is intended: `b`'s state entry is not touched, so the next run still sees
  `b`'s digest as changed and syncs it. The run report lists such repos
  under `pending_repos` (digest differs, not selected). *Call made here.*

### Lock with the timer run

Same `transaction_lock` (`pipeline.py:972`, `LOCK_EX|LOCK_NB`). A `--repo`
run that finds the lock held exits 3, as today. It does not wait: a timer
run that holds the lock already picks up the checkout, or the next one does.
A caller that wants "eventually" can retry; the flag adds no queue.

### Shrink guard

The global guard (`validate.py:68`, `allow_shrink`) compares per-repo counts
to the previous generation. Unselected repos reuse their last graph, so their
counts do not move and the guard behaves as in a full run. The per-repo
`graphify-shrink-check` runs for the selected repo exactly as today, with
`--producer-changed` only on the existing triggers (graphify version change,
ignore-rule change). `--repo` grants no shrink.

### Stage caches (steps 2–5, order set by step 0)

Each cache key is a hash of that stage's per-repo input, so a cache hit
produces the same bytes as a recompute. Generations stay rebuilt from empty:
the cache supplies per-repo pieces, and the stage assembles them fresh.

- **Naming.** Cluster scope is already `repo` (`config.py:202`), and names
  are reused by membership signature. Cache per repo: subgraph hash → cluster
  assignment. Only changed repos recluster and call the LLM.
- **Embedding.** Shards are already per repo, but every run re-hashes every
  node snippet and `persist_generation` copies the whole tree
  (`embedding.py:1503`). Skip re-hashing for repos whose graph hash is
  unchanged, and hard-link unchanged shard files instead of copying them.
- **Overlay `similar_approach`.** The exact pass compares every repo pair
  (56 s measured, `embed_similarity.py:209`). Accepted edges cannot be cached
  per pair: `_mutual_from_candidates` (`embed_similarity.py:478`) keeps a
  global top-k per node across all repos, and `overlay_similar.py` caps each
  repo pair after a global sort. One changed repo can therefore add or evict
  edges between two unchanged repos. Cache instead, per repo pair keyed by
  both shard hashes, each node's top-k candidates within that pair. A node's
  global top-k is the top-k of the union of its per-pair lists, so the
  mutual filter and the per-pair cap then run globally over cached
  candidates, and only pairs touching a changed repo are compared (24 of 300
  for one repo). *Builder's call, made here:* this is exact only on the
  exact path. On the LSH path, `MAX_BUCKET_COMPARE_ROWS` truncation depends
  on every row in a bucket, so a run that truncates any bucket skips the
  cache and recomputes. Cache size is measured in step 0's runs.
- **Lexical index.** Built once per generation (`lexical_index.py:293`).
  Step 0 decides whether it needs a per-repo cache.
- **Merge.** `graphify merge-graphs` over all files. Step 0 decides; if it
  is slow, the option is a cached merged result keyed by the sorted list of
  per-repo graph hashes, invalidated by any change.

Caches live under `<global_dir>/cache/<stage>/`, are written only after a
successful publish (the same rule as embedding shards), and are GC'd with the
generation that last referenced them. A missing or corrupt cache entry means
recompute, never a failure.

## Testing

- Unit: `--repo` with an unknown id, a disabled id, and two ids.
- Pipeline: `--repo a` leaves repo `b`'s state entry and graph untouched;
  the lock held → exit 3.
- Equivalence: for a fixture mesh with the naming LLM stubbed, a warm full
  run with caches and a warm full run with caches disabled publish
  byte-identical `global-graph.json`, overlay and lexical index (modulo
  `created_at` and the generation id). The same holds for a `--repo` run
  against a full run after the same edit. Both start from the same
  `naming-state.json`. A cold run is excluded: names come from
  `naming-state.json` by membership signature, so empty naming state can
  name differently by design. *Call made here.*
- Naming with Ollama down: a `--repo` run gives the changed repo's new
  communities provisional hub names, as a full run would, and the next
  healthy run relabels them.
- Shrink: `--repo` on a repo that loses files is refused exactly like a full run.

## Left to the builder

- Cache file format and naming.
- Exact log line format for stage timing.
- Whether the per-pair overlay cache stores scores or only accepted edges,
  as long as equivalence holds.

## Open

- The 60 s target is unverified until step 0. If merge or lexical index alone
  exceeds it on the live corpus, this design reports that and stops for a
  decision rather than relaxing equivalence.
