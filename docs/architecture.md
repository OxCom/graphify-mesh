# Architecture

## The pipeline

`graphify-mesh-sync` runs a single ordered pipeline per invocation. Each stage
feeds the next; the whole thing is guarded by a whole-transaction lock so two
runs can never overlap.

```
discovery
   -> per-project sync            (decide update vs extract per repo)
   -> merge                       (graphify merge-graphs, from empty, sorted)
   -> recluster + repo-tag remap  (re-cluster the merged graph; rewrite
                                    graphify's auto tags to true repo_ids)
   -> name                        (label communities via the LLM backend)
   -> embed-changed               (embed only nodes whose content changed)
   -> overlay-resolve             (build depends_on / provides_api /
                                    consumes_api / similar_approach edges +
                                    validated manual relations)
   -> lexical-index               (tokenized postings for hybrid search)
   -> validate                    (structural + consistency gates)
   -> atomic publish              (write a new generation, flip `current`)
```

### Stage notes

- **discovery** walks the scan root for `graphify-out` symlinks, reconciles them
  against `registry.json` (auto-add / removed / renamed / broken), and rejects
  any symlink whose resolved target escapes the approved root.
- **per-project sync** compares a fresh per-repo source digest against saved
  state to decide `update` (AST-only) vs `extract` (semantic) vs `noop`, and
  refuses a per-repo result that shrank unexpectedly.
- **merge** always calls `graphify merge-graphs` with a deterministic,
  sorted-by-`repo_id` list of per-repo `graph.json` paths — **never** `graphify
  global add`. See the invariants below.
- **recluster + remap** re-runs community detection on the merged graph and
  rewrites graphify's auto-derived node-id tags (which collide across repos
  under the same product directory) to the true registry `repo_id`, before any
  downstream stage runs.
- **name** sends only communities whose membership fingerprint changed to the
  LLM backend, reusing prior labels otherwise. Degrades to placeholder names if
  the backend is unreachable.
- **embed-changed** embeds only nodes whose durable content key changed since
  the last generation; shards are persisted (and GC'd to the last N
  generations) only once publish actually happens.
- **overlay-resolve** produces the cross-project overlay artifact and never
  writes any of its edge types into the structural graph.
- **validate** blocks publish on structural problems, a too-high stale-repo
  ratio (unless `--allow-shrink`), or a failed backend-pin check.
- **atomic publish** writes `global-graph.json`, `cross-project-overlay.json`,
  and `lexical-index.json` into a fresh generation directory, then flips the
  `current` symlink so readers always see a complete, consistent generation.

## The two MCP servers

There are two MCP servers in play, and they are complementary:

- **graphify's own server** (`graphify.serve`) can be repointed at the published
  `<mesh_root>/graphify/global/current/global-graph.json` for the **structural
  / PR** tools it already understands: `god_nodes`, `shortest_path`,
  `get_pr_impact`, `get_community`, `get_neighbors`, `get_node`, `graph_stats`,
  `list_prs`, `triage_prs`. Repointing is a config change, not a rewrite.
- **graphify-mesh's server** (`graphify-mesh-server`) owns everything
  hybrid / cross-project / evidence-oriented: `search`, `cross_project`,
  `find_similar`, `project_map`, `context_pack`. These need scope resolution,
  the lexical index, the embedding index, and the cross-project overlay.

### Two transports, one tool surface

`graphify-mesh-server` runs as either transport, both registered once against
the `mcp` SDK's low-level `Server` (`server/sdk_app.py`) from the same
`GraphifyMeshServer.tool_schemas()` / `.call_tool()` pair, so the tool set
cannot drift between modes:

- **stdio** (default): one process per client session, unchanged behavior.
- **`--transport http`**: one shared daemon serving every local agent over
  stateless streamable HTTP, gated by a mandatory bearer token and Host/Origin
  validation. See [`mcp-server.md`](mcp-server.md) for the protocol detail and
  [`configuration.md`](configuration.md) for the flags and env vars.

A shared daemon has one process's `cwd`, not each caller's, so `scope='current'`
resolution depends on the caller passing its own absolute directory in the
per-call `cwd` tool argument instead of relying on the process's working
directory.

### Concurrency in the shared daemon

Reads run in parallel; a generation reload takes an exclusive lock
(`server/rwlock.py`, a writer-preferring read/write lock, so a steady stream
of reads cannot postpone a reload indefinitely). `GenerationStore` re-checks
the generation signature under the write lock before reloading, so two
threads racing a reload never load the same generation twice, and the loaded
`Generation` object is only published to readers once it is fully built. The
registry cache (`server/scope.py`) and the per-generation similarity index
cache (`server/similar.py`) are each guarded by their own lock, with the fast
cache-hit path lock-free.

### Why community names can legitimately differ between them

graphify's server reads a graph's `community_name` attribute verbatim.
graphify-mesh's naming stage **strips and relabels** community names during the
sync pipeline (`graphify_mesh.sync.naming`). So during any transitional period —
historical caches, or an upstream instance not yet repointed at the published
artifact — the same node may show a different community name in each server.
This is expected, not a bug to reconcile: the relabeling only runs in this
pipeline, never in graphify's own process.

## Constraints worth knowing

- **The structural graph never contains cross-repo edges.** `depends_on`,
  `provides_api`, `consumes_api`, and `similar_approach` live only in the
  separate `cross-project-overlay.json` artifact. A cross-project guess can
  therefore never be mistaken for a ground-truth extracted edge, and shortest
  paths over the structural graph stay honest.
- **Generations are rebuilt from empty each run, not incrementally merged.**
  `graphify merge-graphs` is stateless: it composes a brand-new graph from the
  input `graph.json` files every time. `graphify global add` is stateful and
  deduplicates "external" nodes by label against an accumulated global graph,
  remapping edges onto the first-added repo's copy — re-adding a repo out of
  order or pruning + re-adding one silently rewires or drops edges other repos
  depend on. Rebuilding from empty is what makes runs deterministic and
  repeatable and is the reason `graphify global add` is never invoked anywhere
  in this package.
- **Publish is atomic.** Readers only ever see a fully written, consistency-
  checked generation; a partially written generation is never pointed to by
  `current`.
