# Graphify Mesh

A scheduled **sync engine** plus a companion **MCP query server** that merges
many per-repo [`graphify`](https://github.com/Graphify-Labs/graphify) knowledge
graphs into one safe, repo-attributed, human-named **global graph** — with a
cross-project dependency / similarity / API overlay, and a query server that
respects *"search only in my current repo unless I ask for more."*

- **pip / PyPI name:** `graphify-mesh`
- **import package:** `graphify_mesh`
- **console scripts:** `graphify-mesh-sync`, `graphify-mesh-server`, `graphify-mesh-reap`
- **Graphify:** [github.com/Graphify-Labs/graphify](https://github.com/Graphify-Labs/graphify)
- **Docs:** see [Documentation](#documentation) below

---

## Why this exists

`graphify` is an AST + LLM code-knowledge-graph tool: point it at a repo and it
produces a per-project `graph.json`. That works great for a single repo. The
moment you have **many** repos — say a dozen or two: PHP/Symfony backends,
TS/React frontends, a couple of infra repos — and you want to ask questions
*across* them, a pile of independent `graph.json` files stops being enough.

This engine was extracted from a real setup with ~16 independent
`graphify`-indexed repos. A documented before/after baseline probe suite
surfaced nine concrete failures that this package fixes:

1. **No repo attribution.** Merge repos into one global graph naively and every
   result looks like it could be from any of them.
2. **Keyword-seed traps.** Naive query engines match literal common words
   ("Usage", "path", "Events") instead of concepts.
3. **Noise flooding.** Generic getters / loggers / setters dominate BFS
   neighborhoods and crowd out the actually-relevant hit.
4. **Inferred-edge pollution.** Heuristically-guessed `calls` / `similar_to`
   edges create nonsense neighbors and shortest paths, indistinguishable from
   ground-truth extracted edges.
5. **Community-id collisions after merging.** "What's in community 3?" returns
   nodes from six unrelated repos.
6. **No cross-project answers.** "Does repo A depend on repo B?", "which repos
   use a similar auth pattern?" are unanswerable without manual grep across
   checkouts.
7. **Stale data, no repeatable rebuild.** Merging graphs incrementally corrupts
   shared external-node identity over successive runs.
8. **No human-readable names.** Communities show up as "Community 47" instead of
   a real name.
9. **Deprecated / dead code not down-weighted.** A `DEPRECATED/` folder's stale
   client code dominates results for a still-maintained API.

graphify-mesh solves this **generically** for anyone with multiple
`graphify`-indexed repos who wants: a scheduled sync engine that merges them
safely, human-named communities, a cross-project dependency/similarity/API
overlay, and a query server that only widens scope when you ask it to.

---

## Architecture at a glance

```
  per-repo graphify-out/graph.json  (one per registered repo)
                 |
                 v
        +------------------+        graphify-mesh-sync  (scheduled, single run)
        |  sync pipeline    |
        |  update -> merge -> recluster+remap -> name ->
        |  embed-changed -> overlay-resolve -> lexical-index ->
        |  validate -> atomic publish
        +------------------+
                 |  publishes a new immutable "generation"
                 v
   <mesh_root>/graphify/global/current/  ->  generations/<id>/
       |-- global-graph.json          (structural graph, repo-attributed,
       |                                human-named communities, NO cross-repo edges)
       |-- cross-project-overlay.json  (depends_on / provides_api / consumes_api /
       |                                similar_approach -- a SEPARATE artifact)
       +-- lexical-index.json          (tokenized postings for hybrid search)
                 |
                 v
        +------------------+        graphify-mesh-server  (stdio, one per session,
        |  MCP query server |         or --transport http, one shared daemon)
        |                   |  tools: search, cross_project, find_similar,
        |                   |         project_map, context_pack,
        |                   |         neighbors
        +------------------+
```

Two hard invariants worth stating up front:

- **The structural graph never contains cross-repo edges.** Those live only in
  the separate overlay artifact, so a cross-project guess can never be confused
  with a ground-truth extracted edge.
- **Each generation is rebuilt from empty**, never incrementally merged, so
  shared external-node identity cannot corrupt over successive runs.

See [`docs/architecture.md`](docs/architecture.md) for the full pipeline.

---

## Quickstart

```bash
# 1. Install graphify-mesh (pulls in the upstream `graphifyy` library dependency
#    automatically):
pip install graphify-mesh          # or: pip install -e .  (from a checkout)

# 2. (Optional) Also install the standalone `graphify` CLI command, isolated,
#    if you want to run ad-hoc queries yourself outside the sync engine:
pipx install graphifyy

# 3. Describe your repos in a registry.json (see examples/registry.example.json):
cp examples/registry.example.json /path/to/your/workspace/graph-mesh/bin/registry.json
$EDITOR /path/to/your/workspace/graph-mesh/bin/registry.json

# 4. Dry run (writes nothing outside a private staging dir):
graphify-mesh-sync --once --dry-run \
  --mesh-root /path/to/your/workspace/graph-mesh \
  --scan-root /path/to/your/workspace/checkouts \
  --scan-root /path/to/your/workspace/other-checkouts

# 5. Real run:
graphify-mesh-sync --once \
  --mesh-root /path/to/your/workspace/graph-mesh \
  --scan-root /path/to/your/workspace/checkouts \
  --scan-depth 4

# 6. Register the MCP server with your MCP-capable client (stdio, default):
#    command: graphify-mesh-server
#    env:     GRAPHIFY_MESH_ROOT=/path/to/your/workspace/graph-mesh
#    Or run one shared daemon for every local agent: graphify-mesh-server --transport http
#    (needs GRAPHIFY_MESH_HTTP_TOKEN — see docs/setup.md)
```

Full walkthrough: [`docs/setup.md`](docs/setup.md).

---

## Documentation

| Doc | What it covers |
|-----|----------------|
| [`docs/setup.md`](docs/setup.md) | Install, configure a registry for *your* repos, first dry-run + real run, register the MCP server, install-from-index URL. |
| [`docs/keeping-sync-up-to-date.md`](docs/keeping-sync-up-to-date.md) | Scheduled re-indexing: step-by-step systemd timer setup (env file, units, cadence, reaper), cron alternative, adding/removing repos, troubleshooting. |
| [`docs/configuration.md`](docs/configuration.md) | Every `GRAPHIFY_MESH_*` env var, every `Settings` field, the `registry.json` and `manual-relations.json` schemas. |
| [`docs/architecture.md`](docs/architecture.md) | Pipeline stages, the two-MCP-server concept, and the structural-vs-overlay / rebuild-from-empty invariants. |
| [`docs/mcp-server.md`](docs/mcp-server.md) | The 6 tools, both transports (stdio and shared HTTP), how to register the server. |
| [`docs/publishing.md`](docs/publishing.md) | Why GitHub Packages does **not** host Python, and how the gh-pages PEP 503 "simple" index substitutes for it. |

## License

MIT — see [`LICENSE`](LICENSE).
