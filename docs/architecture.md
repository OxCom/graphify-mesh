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
   -> name                        (cluster + label in process, no CLI child)
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
  any symlink whose resolved target escapes the approved root — as well as any
  registry entry whose `root` or `collection_path` resolves outside it. Every OS
  error it would otherwise swallow is recorded, and a run that scanned
  incompletely reports `scan_incomplete` with the reasons. A repo is then never
  classified as *removed* on the strength of a link the scan may simply have
  failed to see: removal must be a confirmed fact, never an inference from a
  failed scan.
- **per-project sync** compares a fresh per-repo source digest against saved
  state to decide `update` (AST-only) vs `extract` (semantic) vs `noop`. The
  digest hashes code and semantic files separately, classified the way upstream
  `graphify.detect.classify_file` routes them (`sync/config.py`, table version
  `CLASSIFICATION_VERSION`): `.yaml/.yml/.json` and package manifests such as
  `pyproject.toml` are code, documents are semantic, and other `.toml/.ini/.cfg/.conf`
  files and upstream-skipped lockfiles are ignored. Each table also names the
  directories whose files never reach the digest: v1 keeps its original 13, and
  v2 adds every fixed-name entry of upstream `detect._SKIP_DIRS` plus the
  `.egg-info` suffix, so build output such as `.mypy_cache/**/*.meta.json` cannot
  churn the code hash. v3 keeps the v2 categories and also drops every file
  graphify's own ignore rules exclude: `.gitignore` (tracked files stay, as in
  git), `.graphifyignore`, the `--exclude` list and `gitignore` switch persisted
  in `<collection>/.graphify_build.json`, and upstream's noise-dir checks. v3
  also drops every file graphify skips as a likely secret before classification
  (`docs/secrets.md`, `api_token.txt`, `service-account.json`, non-source files
  under `secrets/`), through `config.is_sensitive_path`, a hand copy of
  `detect._is_sensitive` that `tests/compat/test_sensitive_paths_drift.py`
  compares with the installed graphify. Counting them used to hold one extract
  and a retry for a document graphify never admits. On four
  repo copies this cut the digest from 2852 files to 965, against 960 that
  graphify `detect()` reads. The mesh rebuilds upstream's `ignored_predicate`
  from its helpers (`state.GraphifyIgnoreRules`) and never calls `detect()`,
  which applies a repo's `.graphifyrc` language overrides process-wide. The
  upstream predicate is not called either: it runs `git ls-files` with the
  repo's own config, which executes a configured `core.fsmonitor` command, and it
  reads ignore files through symlinks and FIFOs. The rebuild disarms fsmonitor
  and hooks, reads only regular non-symlink ignore files, skips the user-wide
  `core.excludesFile`, and drops `.gitignore` rules when git fails. An ignore
  file over 1 MiB (`state.MAX_IGNORE_FILE_BYTES`) counts as unreadable: its rules
  are not applied and a warning names it. The git child gets the parent
  environment minus every `GIT_*` variable, because `GIT_DIR`, `GIT_WORK_TREE`,
  `GIT_INDEX_FILE` or `GIT_CONFIG_*` would point it at another repo or re-arm
  the disarmed config. Each
  difference keeps more files in the digest, never fewer. Per-repo state records the
  table version with each accepted digest, and under v3 the digest recipe
  (`digest_recipe`: table version plus the in-process graphify version), because
  the ignore rules come from the installed graphify. On a version mismatch the repo's digest
  is also computed under the table its state was written with
  (`CLASSIFICATION_TABLES`), in the same walk as the current digest
  (`compute_source_manifests`), so both describe one file snapshot: an
  unchanged legacy semantic hash means files only changed category, so the repo
  runs `update`, while a changed one is a real document edit and takes the normal
  `extract` path. A recorded version that is not an integer selects no table: the
  raw hashes decide, a warning names the repo, and an accepted run stamps the
  current version. A refused or failed run keeps the old version, so the migration
  retries, and the double classification repeats on every run, dry runs included,
  until one is accepted. A recipe change cannot be checked the same way: the old
  graphify is gone, so its digest cannot be recomputed. When the stored hashes
  no longer match, the repo runs `update`, never `extract`, and the new hashes are
  accepted. A document edited in that same run is therefore extracted only on its
  next change. Refusal fields are recorded and compared under the old table
  while migrating, so a held refusal stays held, and an `allow_shrink_once`
  token issued under the old table still names the same attempt. The active
  table is `Settings.classification_version` (`GRAPHIFY_MESH_CLASSIFICATION_VERSION`,
  default `CLASSIFICATION_VERSION` = 3). Selecting an older key of
  `CLASSIFICATION_TABLES` is the rollback switch: 1 restores the table from
  before graphify 0.9.103, without ignore rules. A switch in either direction
  goes through the same one-walk migration, so a repo runs `update` unless a
  real document changed.

  After an `extract`, the mesh checks graphify's `manifest.json` for documents
  the extract left unstamped (`sync_project._unstamped_documents`). graphify
  exits 0 on a truncated or empty LLM answer but writes `semantic_hash: ""` for
  that document, or no row for a new one, so its own next extract retries it. A
  semantic document of this run's digest (`SourceDigest.semantic_paths`) in that
  state keeps the new graph but holds the repo's accepted state, so the next run
  extracts again. Each path gets one retry: per-repo state records it under
  `unstamped_documents` (`UNSTAMPED_DOCUMENTS_KEY`), and a second miss is
  accepted with a warning. The per-project sync also refuses a per-repo result that shrank unexpectedly. A refusal restores the
  last-good `graph.json` and names, in its reason, the source digest of the
  attempt it refused. A deliberate deletion of functionality looks exactly like
  a broken extraction, so the digest is how one specific shrink gets approved:
  putting it in that repo's registry entry as
  `"allow_shrink_once": "<digest>"` authorizes that one attempt for that one
  repo, and nothing else. The token must equal the refused digest exactly, it
  overrides the refusal-retry hold so the repo re-extracts once, and it is spent
  in per-repo state (`consumed_shrink_grant`) the moment it is used — the
  registry is never rewritten by the sync engine, because the mesh server and
  `mesh-register.py` share that file. An absent, empty or spent key means the
  guard is armed; a malformed one fails the registry load. `--allow-shrink`
  still exists for the whole-run case and accepts every repo's shrink in that
  run, which is why the per-repo key is the default way to approve one. Whether a
  result "shrank unexpectedly" is decided by `graphify-shrink-check` under the
  default `attributed` policy, which accepts loss explained by source changes,
  and by the node/edge count floor when that tool is absent or errors
  (`GRAPHIFY_MESH_SHRINK_POLICY`). A refusal restores `manifest.json` together
  with `graph.json`, and a refused `update` is held while the attempt id,
  producer and policy stay the same. Since graphify 0.9.102 `update` is
  incremental, so the mesh adds `--force` (graphify's full rebuild, which also
  waives graphify's own shrink guard) in three cases only: the recorded graphify
  or graphify-php version differs from the installed one, or no graphify version
  is recorded yet; `--allow-shrink` releases this repo's held refusal; or an
  `allow_shrink_once` token names this attempt, and is then spent on any
  accepted outcome. A producer label switch (`extract` to `update`) never forces.
  The first case also covers an extractor capability: for a plain graphify repo
  whose digest holds a `.sql` file, `producer_versions.tree_sitter_sql` records
  whether the sync interpreter can import `tree_sitter_sql` (`present` or
  `absent`; state from before the stamp counts as `absent`). Installing or
  removing the `sql` extra changes no source digest, so this entry is what
  rebuilds those repos once. Repos without `.sql` files never carry it.
  graphify-php repos do not either: they build in the wrapper's interpreter,
  which the version probe does not ask about SQL.
  Repos with `builder: graphify-php` run
  `update` through `graphify-php-build` so the PHP resolver edges survive. That
  wrapper appends `--force` to every `update` itself, because an incremental
  update cannot re-resolve PHP calls from unchanged files, so the force decision
  above matters only for plain graphify repos; the mesh guard still judges PHP
  rebuilds.

  **Operator note, first run after deploying classification v2.** Every repo
  whose tree holds files that changed category (YAML, JSON, package manifests,
  lockfiles, files under newly pruned dirs) migrates: its walk classifies each
  file under both tables and the repo runs `update`, or `extract` when a
  document changed. Repos with no recorded graphify version also rebuild once
  with `--force`. If the fleet's total node or edge count shrinks across these
  rebuilds, validate's global guard can block publish; review the counts, then
  rerun with `--allow-shrink`.

  **Operator note, first run after deploying classification v3.** Every repo
  migrates from v2 the same way, in one walk under both tables: `update` when no
  document changed, `extract` when one did. An edit to a file only v2 counted (a
  gitignored `CLAUDE.md`, for example) in that window also extracts once. Deploy
  it in a run of its own, after a `--dry-run`, and not in the run that installs
  the `sql` extra. When the
  child exits
  non-zero, the outcome reason carries the **last** 2000 characters of its
  stderr and the whole stream is written to
  `<collection>/.graphify_sync_error.log`, overwritten per failure. It used to
  carry the first 300 characters, which is the wrong end: the CLI prints its
  warnings first and its traceback last, so a failing repo reported a
  semantic-cache warning and nothing about the cause.
- **merge** always calls `graphify merge-graphs` with a deterministic,
  sorted-by-`repo_id` list of per-repo `graph.json` paths — **never** `graphify
  global add`. See the invariants below. `merge-graphs` writes cross-repo edges
  of its own (`same_type_as` links marked `context="cross_repo"` between type
  declarations two repos share, plus cross-repo `calls` edges), so the pipeline
  strips every edge whose endpoints resolve to different repos right after the
  repo-tag remap and before reclustering. The count is logged and reported in
  `RunReport.stripped_cross_repo_edges` and `status.json`. Stripping is how
  invariant 1 holds; refusing the publish instead would turn the first type
  shared between two repos into a permanent outage.
- **recluster + remap** re-runs community detection on the merged graph and
  rewrites graphify's auto-derived node-id tags (which collide across repos
  under the same product directory) to the true registry `repo_id`, before any
  downstream stage runs.
- **name** clusters the merged graph, by default one repo at a time, and labels
  its communities inside the sync process, through graphify's Python API. Only communities that need a name
  reach the LLM backend; an unreachable backend degrades to deterministic hub
  names rather than failing. See [the naming stage](#the-naming-stage) below.
- **embed-changed** embeds only nodes whose durable content key changed since
  the last generation; shards are persisted only once publish actually happens,
  and older embedding generations are collected only after `current` has
  flipped, so a crash in that window cannot delete the vectors the live
  generation is served with. Shard format v3 adds a
  **recipe stamp** to `meta.json` — a digest of the embedding model and the
  snippet/input parameters that produced those vectors. Reuse requires the stamp
  to match as well as the node's content key, and a repo whose sources did not
  change at all is still re-embedded when the stamp differs, so changing the
  model can no longer leave old vectors published under a manifest advertising
  the new one. A v1 or v2 shard carries no stamp and is never reused.
  `RECIPE_STAMP_VERSION` (`sync/embedding.py`) is part of the stamp and is
  bumped when the embedding input changes while the stamped fields stay the
  same. It went from 1 to 2 because commit 21b2dfa changed
  `build_embedding_input` without a bump, so the first run after that upgrade
  re-embeds every repo once. For an unchanged repo the log separates the two
  causes: "no published shard exists" against "published shard recipe differs
  from the configured one" (`embedding.missing_shard_repo_ids`). Each embed
  request waits at most `GRAPHIFY_MESH_EMBED_TIMEOUT` seconds. When the stage
  does not run (`--skip-embedding`), `embedding.carry_forward_stage` restages
  the last published shards under the new generation, dropping any shard of
  another recipe, so the server keeps its vector channel. Before, nothing was
  persisted, `embeddings/current` kept the previous generation id, and the
  server dropped all vectors with `embeddings_generation_mismatch`. Nodes
  added since have no vector until a run embeds them. A skip run records the
  new source digest while carrying the old vectors, so whole-shard reuse for an
  unchanged repo also checks every entry's content hash against the current
  input (`embedding._shard_content_current`). On a mismatch the repo goes through
  `compute_repo_shard`, which re-embeds only the changed nodes. The check costs
  about 81 µs per key, mostly snippet path validation: 1.4 s for the largest
  repo (13,644 keys) and 4.4 s for all 25 registered repos, measured 2026-11-06.
- **overlay-resolve** produces the cross-project overlay artifact and never
  writes any of its edge types into the structural graph.
- **validate** blocks publish on structural problems, a too-high stale-repo
  ratio (unless `--allow-shrink`), or a failed backend-pin check. The
  cross-repo-edge check is on the node-id prefixes, not on the relation type:
  two endpoints whose `<repo_id>::` prefixes differ is an error for **every**
  relation, so an ordinary structural relation such as `calls` spanning two
  repos is refused exactly like an overlay relation would be. It is a backstop,
  not the primary mechanism: the merge stage already stripped upstream's
  cross-repo edges, so one reaching validate means a later stage wrote it. Repo removal
  auto-authorizes the shrink only when the scan was complete.
- **atomic publish** writes `global-graph.json`, `cross-project-overlay.json`,
  and `lexical-index.json` into a fresh generation directory, persists the
  embedding generation, and only then flips the `current` symlink, so readers
  always see a complete, consistent generation. The embedding step comes
  **before** the flip deliberately: with the graph going live first, a reader
  entering that window loaded the new graph against the previous embeddings
  generation and dropped its whole vector channel. The cost is one stranded
  embeddings directory if the run dies between the two steps, which the next
  run's GC collects. `status.json` is written through the same atomic
  tmp-file-and-rename path, and carries `scan_incomplete` / `scan_errors`.

  **Operator note, deploying lexical schema v4 (config-key dotted paths).**
  Restart the HTTP daemon together with the package upgrade. A daemon still
  running the old code supports lexical schema `{2, 3}` only
  (`server/store.py`, `SUPPORTED_LEXICAL_SCHEMA_VERSIONS`), so it rejects the
  first v4 generation and keeps serving the v3 generation it already holds in
  memory. Sync's generation GC keeps only the newest generations, so later runs
  delete the directories that in-memory generation came from. stdio servers pick
  up the new code with the next client session.

  Config keys are addressed by their dotted path (`config_keys.config_key_paths`).
  A key that itself contains `.`, such as the Kubernetes label
  `app.kubernetes.io/name`, can produce the same path as a nested key. Such
  nodes share one durable key; the lexical stage counts them in the
  `config_key_collisions` stat of `lexical_index_stats`.

  When several nodes share a durable key, `compute_repo_shard` embeds one
  representative: the lowest line, then the lowest node id. The stored content
  hash and the vector both come from that node. Before, every occurrence
  overwrote the entry and re-queued the key, so the stored hash and the vector
  could belong to different nodes and every duplicate re-embedded on every run.

## The naming stage

The naming stage runs entirely in the sync process. It stages no graph file and
spawns no `graphify` child: `graphify cluster-only` and `graphify label` rebuild
the graph through `graphify.build.build_from_json`, which merges nodes sharing
`(source_file, label)` and, in a second pass, nodes sharing a label alone.
Neither pass reads a node's `repo`, so on a graph spanning fourteen repositories
both collapse nodes across repository boundaries and rewrite labels into full
paths. The stage's three modules are `sync/naming.py` (policy),
`sync/clustering.py` (every call into graphify's Python API) and
`sync/naming_state.py` (the state file).

In order, for each run:

1. `backend.assert_pinned_backend()` resolves the clustering backend by
   `importlib.util.find_spec` in this interpreter and raises unless it matches
   `PINNED_CLUSTERING_BACKEND`. This is one of the two failures the stage lets
   propagate.
2. `clustering.graph_from_node_link` converts the merged dict to an `nx.Graph`
   with `networkx.readwrite.json_graph.node_link_graph`, which applies no merge
   heuristic, no label disambiguation and no hyperedge revalidation. It passes
   `graph={}`: networkx does not copy that mapping, so without the substitution
   anything upstream writes into `G.graph` would land in the mesh's own merged
   dict, where the integrity guard — which covers nodes and links — would not
   see it.
3. `clustering.cluster_graph` runs `graphify.cluster.cluster`, then
   `remap_communities_to_previous` against the previous run's assignments, so a
   community keeps its id across runs instead of renumbering. The previous
   assignments are used only when the stored clustering recipe (backend,
   resolution and, for `cluster_scope=repo`, the scope) still matches; a
   labeling-model change does not discard id history.

   Under `cluster_scope=repo`, the default, `clustering.repo_groups` splits the
   graph by node `repo` and each repo's subgraph is clustered on its own. A node
   without a `repo` joins the smallest repo_id among its neighbours, repeated
   until no stub can be placed; the rest form one unattributed group. Global
   community ids are repo-local ids plus an offset taken in sorted repo order.
   Each repo's fingerprint (its nodes and internal edges) is stored with its
   local partition in `naming-state.json` under `repos`. A repo whose
   fingerprint is unchanged reuses that partition and its names: it is not
   clustered and none of its communities reach the LLM, unless one is
   provisional. Measured on four repo copies: global clustering produced 113 new
   signatures after a one-repo edit, per-repo clustering 47, all in the edited
   repo. Splitting is valid because the structural graph has no cross-repo
   edges. `cluster_scope=global` keeps the whole-graph clustering and its
   pre-scope recipe, so it is the rollback switch.
4. `clustering.membership_sigs` hashes each community's sorted member ids. That
   signature, not the community id, is the key names are carried by.
5. Names are resolved against `naming-state.json`. A community whose signature
   carries a non-provisional name keeps it. Everything else goes to
   `clustering.llm_names` in a single `generate_community_labels` call, which
   upstream batches internally, and only when the backend is usable: `openai` importable, a valid base URL, an
   endpoint the TLS mode can actually reach, and a passing `/models` health
   probe. A returned name matching `validate.PLACEHOLDER_RE` (`^Community \d+$`)
   counts as no name.

   Under `cluster_scope=repo`, in a reclustered repo, a community whose
   signature changed keeps the previous name of its id when graphify's
   `label_still_applies` holds: the shared members are a strict majority of
   both the old and the new membership. A community whose stored entry holds
   an `ungrounded` label never carries a name this way.

   In both scopes a grounding check then judges each LLM label against the
   community's own members (`naming._is_grounded`). The label must share a word
   with a member's label or `source_file`. Words are casefolded, split on
   camelCase and non-alphanumerics, and lose a plural `s`/`ies`
   (`naming._grounding_tokens`). A word matches by equality, by substring of 4+
   characters, by a member-word prefix, by a 6-character stem, or by the
   label's initials ("Data Transfer Object" against `Dto`). A community with
   no usable member words accepts any label. The same name in several repos is
   allowed: the earlier rule that refused a label another repo already used was
   removed, because many names are legitimately shared ("doctrine migrations")
   and `find_similar` relies on that equality.

   A refused label leaves the community on the step 6 hub name, marked
   provisional, and stores the label in the state entry's `ungrounded` field
   (`naming_state.CommunityEntry.ungrounded`). The model returns the same label
   for the same members, so that membership signature is not asked again until
   it or the labeling recipe changes; `--relabel-all` asks it again. An
   ungrounded community does not count as degraded and does not block the
   run-level reuse shortcut. The run reports "N label(s) rejected as not
   grounded in member labels, hub names kept". A state entry written by the old
   rule with `rejected: true` loads as provisional, so the next healthy run asks
   for its name again.
6. Whatever is still unnamed takes a hub name from
   `graphify.cluster.label_communities_by_hub` — the label of the community's
   highest-degree member, `UserRepository` rather than `Community 42` — and is
   recorded `provisional: true`. The next healthy run relabels exactly those.
7. `naming.assert_only_community_attrs_added` compares a per-node digest and a
   per-edge digest taken before the stage against the graph it produced. A lost
   node, a rewritten attribute or a changed edge raises `NamingIntegrityError`
   and the run publishes nothing. The digests cost a few MB against a 42 MB
   graph, so no second copy of the graph is held. The guard also runs on the
   reuse path, which writes onto the graph too.
8. `naming_state.save` writes the new state through a temp file and
   `os.replace`, so a crash mid-write cannot leave a half-parsed state the next
   run would trust. The file holds only this run's communities and repos, so a
   signature or repo that left the graph leaves the state with it.

A run whose merged-graph fingerprint, recipes and assignments all match a fully
non-provisional state skips steps 2 to 6 entirely and applies the stored
assignments and names. Reuse needs all of it: the assignments must cover exactly
the graph's node ids, every assigned community must carry a non-empty name, and
no community may be provisional. A provisional community on the reuse path is
what would otherwise freeze an outage's hub names permanently, because an
unchanged graph would never get a second chance once the backend recovered.

Measured on the live mesh (33310 nodes, 1097 communities): clustering is
deterministic and takes about 4 s, the whole naming stage about 104 s. A full
relabel is about 11 sequential batched LLM calls — batch size 100, and upstream
forces ollama calls serial.

A dry run makes no LLM call. It seeds its staging naming directory with the real
`naming-state.json`, runs the stage as if the backend were down, so new
communities get deterministic hub names, and logs how many communities a real
run would label.

## The two MCP servers

There are two MCP servers in play, and they are complementary:

- **graphify's own server** (`graphify.serve`) can be repointed at the published
  `<mesh_root>/graphify/global/current/global-graph.json` for the **structural
  / PR** tools it already understands: `god_nodes`, `shortest_path`,
  `get_pr_impact`, `get_community`, `get_neighbors`, `get_node`, `graph_stats`,
  `list_prs`, `triage_prs`. Repointing is a config change, not a rewrite.
- **graphify-mesh's server** (`graphify-mesh-server`) owns everything
  hybrid / cross-project / evidence-oriented: `search`, `cross_project`,
  `find_similar`, `project_map`, `context_pack`, `neighbors`. These need scope
  resolution, the lexical index, the embedding index, and the cross-project
  overlay. `list_repos` reads only `registry.json` and the served manifest, and
  answers before the first publish.

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
`Generation` object is only published to readers once it is fully built.

`GenerationStore.generation` releases the read lock before it returns, so a
tool call is not held against a concurrent reload — a reload can swap the
store's generation while that call is still running. What makes this safe is
immutability, not the lock: a `Generation` is never mutated after
`build_indexes()`, so an in-flight call keeps reading the complete generation
it was handed, which by the time it finishes may have been superseded. The
rwlock guards the store's own fields only.

The
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
  separate `cross-project-overlay.json` artifact, and the cross-repo edges
  upstream's own merge emits are stripped before anything downstream sees
  them. A cross-project guess can
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
- **`graphify.build.build_from_json` is never called.** It merges nodes that
  share `(source_file, label)`, and then nodes that share a label, without
  regard to a node's `repo`, so on the merged graph it collapses nodes across
  repositories and rewrites their labels into full paths. Graphs are loaded with
  `networkx.node_link_graph`, which is what graphify's own read side does.
- **Publish is atomic.** Readers only ever see a fully written, consistency-
  checked generation; a partially written generation is never pointed to by
  `current`.
