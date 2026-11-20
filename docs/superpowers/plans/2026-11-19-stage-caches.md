# Stage caches (overlay, lexical index, embedding) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the `--repo` run (5m26s on 2026-11-19, overlay 190 s of it) and the full run by caching per-repo work in three stages, with output byte-identical to a cold run.

**Architecture:** Each cache stores per-repo or per-repo-pair pieces keyed by a hash of that piece's inputs. The stage still assembles a fresh generation from those pieces, so invariant 2 (rebuild from empty) holds. Caches are written only after a successful publish, live under `<global_dir>/cache/<stage>/`, and a missing or corrupt entry means recompute. One switch turns every cache off, which is what the equivalence tests compare against.

**Tech Stack:** Python 3.11+, numpy, pytest in Docker (`graphify-mesh-test` image).

**Spec:** `docs/superpowers/specs/2026-11-19-repo-sync-design.md`, section "Stage caches".

## Baseline (2026-11-19, live corpus, 25 repos, 70,179 vectors)

| stage | full timer run wall/cpu s | `--repo cryengine.hub`, no edit, wall/cpu s |
|---|---|---|
| discovery | 3.2 / 1.4 | 0.9 / 0.5 |
| extract | 511.7 / 238.8 | 8.1 / 6.1 |
| merge | 33.3 / 28.5 | 18.4 / 17.1 |
| naming | 336.6 / 27.5 | 10.4 / 10.0 |
| embedding | 23.1 / 11.8 | 24.1 / 19.8 |
| overlay | 299.5 / 645.0 | 189.8 / 595.8 |
| lexical_index | 17.8 / 17.5 | 51.3 / 30.8 |
| validate_publish | 7.1 / 7.1 | 22.0 / 14.3 |
| total, peak | 20m40s, 3.8 GB | 5m26s, 3.0 GB |

The memory cap is `MemoryMax=4G`. No cache may raise the peak.

## Global Constraints

- Git is read-only for agents. The user commits. "Commit" steps mean: list the files for the user.
- `.py` files are edited only through Serena, then `get_diagnostics_for_file`. Tests, ruff and mypy run in Docker only.
- Hard invariants 1–3 in `CLAUDE.md` stay untouched.
- **Equivalence:** with the same inputs and the same `naming-state.json`, a run with caches on and a run with caches off publish byte-identical `global-graph.json`, `cross-project-overlay.json`, `lexical-index.json` and embedding shard files (modulo `created_at` and the generation id).
- **Switch:** `GRAPHIFY_MESH_STAGE_CACHE` (`1` default, `0` off) → `Settings.stage_cache: bool`. Off means caches are neither read nor written.
- **Location:** `<global_dir>/cache/<stage>/`. Entries are written only after `publish.flip_current` succeeds, with an atomic tmp+rename. Each entry records `CACHE_FORMAT` and the code-side recipe it depends on. A mismatch or unreadable file means recompute, with a debug log line and never a failure.
- **GC:** after a successful publish, delete cache entries that no input of the just-published generation referenced.
- **Memory:** caches load per repo or per pair and are released before the next one. Nothing holds every pair's cache at once.
- Dry run never writes a cache.

## Review Focus

- **Ties in the overlay heap.** Today the heap order is `(score, -rank, other_key)` with a global row rank. Cached candidates must be re-ranked against the current row order, or one repo's size change reorders ties in other pairs. The equivalence test in Task 3 uses equal scores on purpose.
- **Lexical doc ids** are assigned in sorted-repo order, so one repo's size change shifts every later repo's ids. Assembly must offset cached local ids, and must not store global ids. Pinned in Task 4.
- **Snippets read live source.** A lexical or embedding cache key built only from `graph.json` misses a source edit that leaves the graph unchanged. The key must include the repo's accepted source digest. Pinned in Tasks 2 and 4.
- **Corpus crossing `EXACT_MAX_ROWS` = 80,000** (now 70,179) switches overlay to LSH, which has no cache. Expected: no cache read or write, and a log line saying why. Pinned in Task 3.
- **Hard-linked shards and GC.** When GC deletes an old generation, the link in the new one must survive. Nothing may write into a published shard file in place. Pinned in Task 2.

---

### Task 1: Cache switch, cache dir, write-after-publish and GC

**Route:** `Agent(subagent_type="cavecrew-builder")` (new `sync/stage_cache.py`, `config.py`)
**Anchors:** `src/graphify_mesh/sync/config.py:1065` (`Settings`), `config.py:1302` and `:1335` (`embeddings_dir`, `naming_dir` as the pattern), `src/graphify_mesh/sync/pipeline.py:2208` (GC after `flip_current`), `src/graphify_mesh/sync/naming_state.py:143` (atomic tmp write pattern).
**Accept:** `tests/sync/test_stage_cache.py` passes.

**Interfaces:**
- Produces: `Settings.stage_cache: bool`, `Settings.cache_dir -> Path` (`global_dir / "cache"`). In `sync/stage_cache.py`: `CACHE_FORMAT = 1`; `read_entry(dir: Path, key: str) -> bytes | None` (returns `None` on a missing, foreign-format or unreadable entry); `write_entry(dir: Path, key: str, payload: bytes) -> None` (atomic); `gc_entries(dir: Path, keep_keys: set[str]) -> int`; and `PendingCacheWrites`, which collects `(dir, key, payload)` during a run and is flushed by `flush()` only after publish.
- Tests: round trip; corrupt and foreign-format entries read as `None`; `gc_entries` keeps exactly `keep_keys`; `PendingCacheWrites.flush()` is never called on a dry run or a blocked publish (pipeline wiring test); `GRAPHIFY_MESH_STAGE_CACHE=0` yields `stage_cache is False`.

### Task 2: Embedding: input digest in the shard meta, skip re-hashing, hard-link unchanged shards

**Route:** `Agent(subagent_type="claude")`, standard model
**Anchors:** `src/graphify_mesh/sync/embedding.py:1307-1329` (reuse decision), `:636-660` (`_shard_content_current`), `:1425-1456` (`stage_embeddings`, `np.save` for every repo), `:1490-1510` (`persist_generation`, `copytree`, `_fsync_tree`), `:283` (`RepoShard.recipe_stamp`).
**Accept:** embedding tests plus the new tests pass, the full suite stays green, and the equivalence check from Task 5 holds for shards.

**Interfaces:**
- Produces: the shard `meta.json` gains `inputs_digest` (sha256 of `graph.json` bytes, the repo's accepted source digest and `recipe_stamp`) and `matrix_sha256` (sha256 of `keys` plus the matrix bytes, as written). `matrix_sha256` is what Task 3 keys on. The shard format stays 3, and readers ignore unknown meta keys (check `server/store.py:451`).
- Behaviour:
  - When the cache is on, the repo is unchanged and the previous meta's `inputs_digest` equals the current one, skip `_shard_content_current` and reuse the shard.
  - Otherwise, decide as today.
  - `persist_generation(..., reused_repo_ids)` hard-links `<repo>.npy` and `<repo>.meta.json` from the previous generation dir for reused repos, using `os.link` with a `shutil.copy2` fallback on `OSError` (for example `EXDEV`). It still fsyncs.
  - Staging no longer `np.save`s reused repos. Staging lives under `PrivateTmp` `/tmp`, a different filesystem, which is why the link happens in `persist_generation` and not in staging.
- Tests:
  - A reused shard is a hard link (`st_nlink == 2`), and its bytes equal a cold write.
  - After `gc_embedding_generations` deletes the old generation, the linked file still reads.
  - A source edit that leaves `graph.json` unchanged changes `inputs_digest` and re-hashes.
  - With the cache off, the bytes are identical and nothing is linked.

### Task 3: Overlay `similar_approach`: per-repo-pair candidate cache, exact path only

**Route:** `Agent(subagent_type="claude")`, most capable model (the tie-ordering and memory constraints are subtle)
**Anchors:** `src/graphify_mesh/sync/embed_similarity.py:198` (`BUCKET_CHUNK_ROWS`), `:218` (`EXACT_MAX_ROWS = 80_000`), `:267-330` (`_exact_pairs`, heaps at `:291`, push at `:318-324`), `:388-397` (exact vs LSH), `:478` (`_mutual_from_candidates`), `src/graphify_mesh/sync/overlay_similar.py:215-306` (`_embedding_pairs`, `CANDIDATE_FLOOR`, `MAX_EDGES_PER_REPO_PAIR = 2000`), `src/graphify_mesh/sync/vectors.py:86` (`RepoVectors`).
**Accept:** the new tests pass, `tests/sync/test_embed_similarity.py` and the overlay tests stay green, and the full suite stays green.

**Interfaces:**
- Consumes: `matrix_sha256` per repo (Task 2). When it is absent, compute it from `RepoVectors`.
- Produces: `_exact_pairs` takes an optional `pair_cache` (`read(repo_a, repo_b) -> PairCandidates | None`, `stage(repo_a, repo_b, PairCandidates)`). The pair key is the sha256 of `(matrix_sha256[a], matrix_sha256[b], top_k, effective threshold, CACHE_FORMAT)`.
- `PairCandidates` holds, for each side, compact numpy arrays `(local_row, other_local_row, score: float32)`. It stores this pair's top-k candidates per row at or above the threshold, with no global ranks.
- Algorithm (the part the signature does not fix):
  - For each sorted repo pair, read the cache or compute the pair's block product as today.
  - Push its candidates into the global per-key heaps using the current global rank of `other` for the `-rank` tie-break.
  - Release the pair's arrays before the next pair.
  - A key's global top-k equals the top-k of the union of its per-pair top-k lists, so `_mutual_from_candidates`, the per-pair cap and calibration run unchanged on the result.
- LSH path (rows > `EXACT_MAX_ROWS`): no cache read or write. Log `overlay: LSH path, pair cache not used`.
- Tests:
  - Cache on vs off gives an identical edge list on a 4-repo fixture with deliberate equal-score ties and one repo whose row count changes between runs.
  - A second run with one repo changed reads `n-1` pairs and computes only the pairs that touch the changed repo.
  - The LSH path writes no cache.
  - Memory, as a unit-level assertion: only one pair's arrays are alive at a time. Use a counting `pair_cache` stub.

### Task 3b: Exact-path row limit: configurable, warned, reported

Added on 2026-11-19 at the user's request ("да + предупреждение о лимите строк"), relayed by knowledge-base-83.

**Route:** `Agent(subagent_type="cavecrew-builder")` (`embed_similarity.py`, `config.py`; the report field goes in `pipeline.py`, so the builder may need the `claude` route if it refuses 3 files)
**Anchors:** `src/graphify_mesh/sync/embed_similarity.py:207-218` (`EXACT_MAX_ROWS = 80_000` and its comment), `:388-397` (exact vs LSH branch and its warning), `src/graphify_mesh/sync/config.py:1065` (`Settings`), `src/graphify_mesh/sync/pipeline.py` (`RunReport`, overlay stage ~1976-2000).
**Accept:** the new tests pass and the full suite stays green.

**Facts (verified 2026-11-19):**
- The limit bounds compute time, not memory. The comment justifies it by run time and LSH recall.
- The exact pass compares in 1024-row blocks (`BUCKET_CHUNK_ROWS`).
- At 70,179 rows the overlay stage grew the high-water RSS by 181 MB (full run) and 168 MB (`--repo` run).
- The full run's 3.8 GB peak is the cgroup total, `graphify` subprocesses included.

**Interfaces:**
- `GRAPHIFY_MESH_EXACT_MAX_ROWS` maps to `Settings.exact_max_rows: int`. The default stays `80_000` until Task 6 measures a cold exact pass. A value below 2 falls back to the default, with a warning.
- When `rows > 0.9 * exact_max_rows`, log a WARNING: `similar_approach: <rows> rows is <pct>% of EXACT_MAX_ROWS=<limit>; above it overlay falls back to LSH and the pair cache is not used`.
- `RunReport.overlay_rows: int` and `RunReport.overlay_exact_max_rows: int` are printed in the CLI JSON and written to `status.json`.
- The decision stays on **total rows**, not on this run's uncached work.
  - A cache-dependent decision would let a warm run take the exact path where a cold run on the same input takes LSH. They would then publish different overlays, which breaks the plan's equivalence constraint.
  - Rejected for that reason. The user can raise the limit instead.

**Tests:**
- The env var overrides the default, and a bad value falls back with a warning.
- The warning fires at 90% of the limit and not below it.
- The report fields carry the row count and the limit.
- At `rows == limit` the exact path runs; at `limit + 1` the LSH path runs.

**Raising the default:** decided in Task 6, not here.
- Cold exact cost grows with n². Scaling 190–300 s at 70k gives about 10–15 min at 120k.
- That cost is paid on every cold or invalidated run, including the first run after a graphify upgrade, which invalidates every shard.
- Task 6 measures the overlay stage's wall time and RSS growth on a cold run, then proposes a default to the user.

### Task 4: Lexical index: per-repo partial cache

**Route:** `Agent(subagent_type="claude")`, standard model
**Anchors:** `src/graphify_mesh/sync/lexical_index.py:293-419` (`build_lexical_index`, per-repo loop `:323-390`, output `:408-419`, `schema_version` 4), `src/graphify_mesh/sync/publish.py:150` (`write_lexical_index`), `src/graphify_mesh/sync/pipeline.py:2005-2024` (`graphs_by_repo` lifetime).
**Accept:** the new tests pass, the existing lexical tests stay green, and `lexical-index.json` is byte-identical with the cache on and off.

**Interfaces:**
- Produces: `build_lexical_index(..., partial_cache=None)`. A per-repo `RepoLexicalPartial` holds `documents` (local order), `postings` (term → sorted local doc ids), `alias_exact`, `node_id_index` (local ids), `stale_semantic` and the per-repo stat counters. The key is the sha256 of (`graph.json` bytes, accepted source digest, `manifest.json` bytes, lexical schema version, `CACHE_FORMAT`).
- Assembly: in sorted-repo order, offset each partial's local doc ids by the running document count, then merge into the global structures exactly as the fresh loop does today. The global stats line ("N durable keys shared by M nodes") is recomputed from the merged data.
- Tests:
  - Byte-identical JSON with the cache on and off, using 3 repos where the middle repo grows between runs (so the ids of later repos shift).
  - A source edit with an unchanged graph misses the cache.
  - With the cache on and nothing changed, no per-repo snippet is read: count `build_snippet` calls through a monkeypatch.

### Task 5: Pipeline wiring, end-to-end equivalence, docs

**Route:** `Agent(subagent_type="claude")`, standard model
**Anchors:** `src/graphify_mesh/sync/pipeline.py` (embedding stage ~1890-1930, overlay ~1976-2000, lexical ~2005-2024, publish/GC ~2200-2215), `tests/sync/test_repo_flag.py` (multi-repo fixture to reuse), `docs/architecture.md`, `docs/configuration.md`, `CHANGELOG.md`.
**Accept:**
- `tests/sync/test_stage_cache_equivalence.py` passes with the embed backend stubbed (reuse the existing fake embedder fixture) and naming stubbed.
- The full suite, ruff, ruff format --check and mypy are clean.

**Steps:**
- Wire `PendingCacheWrites` through the three stages and flush after `flip_current`. Run GC for each stage with the keys the published generation used.
- Equivalence tests:
  1. A cold run with the cache off, then a warm run with the cache on, after the same edit to one repo. The published files are byte-identical.
  2. A `--repo` run with the cache on against a full run with the cache off, after the same edit. The files are identical.
  3. A corrupt cache file is a recompute, not a failure.
  4. A blocked publish writes no cache.
- Docs:
  - `GRAPHIFY_MESH_STAGE_CACHE` in `docs/configuration.md`.
  - A short "Stage caches" paragraph in `docs/architecture.md` naming what each cache keys on and that it never changes output.
  - A CHANGELOG entry.

### Task 6: Live measurement (user-gated)

**Route:** main thread. The reinstall needs the user's approval in this session.
**Accept:** a per-stage table sent to the user and to knowledge-base-83.

- [ ] Reinstall after the user approves. Verify `graphifyy` 0.9.112.
- [ ] A full run via `systemctl --user start graphify-mesh-sync.service`. It warms the caches.
- [ ] `--repo cryengine.hub` with no edit, via `graphify-mesh-sync-repo@cryengine.hub.service`. Run it twice back to back, to settle the open question below.
- [ ] `--repo cryengine.hub` after a one-comment-line edit. `www-cryengine-dev-lo-91` makes the edit and reverts it. The target is `src/Controller/Admin/AbstractController.php` after line 1, and the pre-edit sha256 is `055ee528…cf2d4`.
- [ ] Cold overlay cost for the row-limit decision: run once with `GRAPHIFY_MESH_STAGE_CACHE=0` (the user approves the exact command), and record the overlay wall time, CPU time and `hwm_growth_kb`.
- [ ] Report the table and the peak memory against the 4 GB cap. Propose an `EXACT_MAX_ROWS` default from the cold overlay numbers, and let the user choose.

**Open question carried into Task 6:** in the first `--repo` run, lexical_index took 51 s wall but 31 s CPU, and validate_publish took 22 s wall but 14 s CPU. In the full run they took 18/18 s and 7/7 s, and both runs indexed the same 361 keys over 888 nodes. Wall well above CPU suggests waiting on I/O or memory pressure from other load, not `--repo`. The two back-to-back no-edit runs settle it.
