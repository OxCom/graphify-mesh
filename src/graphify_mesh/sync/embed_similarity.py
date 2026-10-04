"""WS3 ANN mutual-top-k similarity over the embedding index (C7).

C7 forbids all-pairs (O(n^2)) similarity computation. This module implements
a small, dependency-free approximate nearest-neighbor mechanism instead:
random-hyperplane locality-sensitive hashing (LSH) for cosine similarity.
Each vector is hashed to a bucket signature (one bit per hyperplane, sign of
the dot product); only vectors sharing a bucket are ever compared directly,
so the candidate-generation step is sub-quadratic instead of exhaustive.
Once two vectors ARE compared, the similarity score is exact cosine
similarity — LSH here only reduces *which pairs get compared*, it never
approximates the score itself. Hyperplanes are seeded deterministically
(`LSH_SEED`) so results are reproducible across runs and in tests.

Known limitation (documented, not hidden): a genuinely similar pair whose
vectors happen to sit near a hyperplane boundary can be hashed into
different buckets and therefore missed — this is the accepted approximate
nature of LSH. If node counts grow enough that this starts costing real
recall, swap in a real ANN library (faiss/hnswlib) behind the same
`mutual_top_k_pairs` signature; nothing downstream needs to change.

C7 also states the similarity threshold must be tuned on labeled pairs —
none exist yet (no labeled ground truth has been collected). Until they do,
`SIMILARITY_THRESHOLD_DEFAULT` below is a documented, untuned, best-guess
default; it is a named constant specifically so it's easy to find and
override once real tuning data exists (see `overlay_similar.py` callers).
"""

from __future__ import annotations

import heapq
import logging
import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from graphify_mesh.sync.vectors import RepoVectors

log = logging.getLogger(__name__)

# Untuned default (C7) — real tuning needs labeled similar/dissimilar node
# pairs, which do not exist yet. Revisit once WS7's eval harness collects
# some.
SIMILARITY_THRESHOLD_DEFAULT = 0.82

# The cosine scale is not comparable across languages, so a single absolute threshold
# admits a different share of each language's analogues. Measured 2026-09-30 on the live
# shards, using same-path copies — the identical file present in both repos — as each
# repo pair's own IDENTITY LEVEL, i.e. what "the same code" scores in that language mix:
#
#     php/php 0.9039   tsx/tsx 0.8772   php/tsx 0.8548
#     php/ts  0.8100   ts/tsx  0.7749   ts/ts   0.6865   php/py 0.6439
#
# Identical PHP scores 0.90; identical TypeScript scores 0.69. Against that spread the
# 0.82 constant admitted 11.7% of php/php analogues, 10.8% of tsx/tsx, 4.0% of php/tsx
# and 0.4% of ts/tsx — a thirtyfold bias, and zero for every Python or Go pair.
#
# Hence a threshold derived per repo pair rather than fixed: IDENTITY_RELATIVE of that
# pair's identity level. At 0.92 the admitted shares even out to 13.0 / 7.1 / 13.3 /
# 13.3 percent for those same four language mixes, and total edges go from 5536 to 7162.
# 0.88 and 0.90 were measured too and admit 2.1x and 1.7x the volume at the same evenness,
# so 0.92 is the conservative end of a flat region, not a knife edge.
IDENTITY_RELATIVE = 0.92

# Below this identity level the embedding cannot tell the identical file from an unrelated
# one for that pair, so a fraction of it would admit noise.
#
# Raised from 0.75 to 0.80 on 2026-09-30 after the groups just above the old floor were
# read by name. Live `launcher<->styleguide` `.tsx`, identity 0.76: of the analogues in
# the admitted band, `SocialLinksList` matched `FooterMenuItem` at 0.700, `api` matched
# `Story` at 0.707, a test mock matched a Storybook story at 0.731 — React file shape,
# not shared approach, with roughly two plausible pairs in sixteen sampled. The reason is
# dynamic range: at identity 0.76 the whole gap between "identical" and "same shape" is
# about 0.06, so no threshold placed inside it separates them. An absolute margin under
# the identity was measured as an alternative and behaves the same as the relative factor
# (at margin 0.07 the admitted shares are 16.6/14.1/17.0 against 16.7/13.5/16.4), because
# identity levels cluster in 0.75-0.90 where the two forms nearly coincide. So the fix is
# to decline the group, not to reshape the threshold inside it.
#
# Cost, measured: 28 -> 37 groups declined, 7246 -> 6624 edges, `.php/.php` untouched at
# 16.7%, `.tsx/.tsx` 13.5 -> 12.2%, `.ts/.ts` 16.4 -> 15.7%. 0.78 and 0.80 give identical
# results, so 0.80 sits mid-plateau rather than on an edge; 0.82 starts cutting real
# `.ts/.ts` analogues (16.4 -> 10.5%). Measured: the ratio of the
# analogue 90th percentile to the identity level is 0.91-0.97 for every pair above this
# floor and 1.06-1.17 for every pair below it — that is, under the floor the "analogues"
# outscore the copies. Those pairs get no edges, which is an honest nothing.
IDENTITY_FLOOR = 0.80

# Fewer same-path copies than this and the median is not a calibration sample; the pair
# falls back to SIMILARITY_THRESHOLD_DEFAULT with its known language bias. A pair of repos
# that shares no file at all cannot be calibrated this way — a real limit of the method.
MIN_CALIBRATION_COPIES = 10

# Analogues across languages are not calibratable the same way and get a flat value.
# Two attempts at a per-pair calibration for them were measured and rejected: same-path
# copies do not exist across languages by construction, and same-label pairs do not
# separate (the median same-label score sits at 0.84-1.13 of the 90th percentile of the
# rest — `handle()`, `execute()` and `create()` are everywhere with different bodies).
#
# What is measured is where the signal lies. Real cross-language findings sit at 0.70
# and above: `TimeService::formatSecondsToTime()` against the TypeScript
# `formatSecondsToTime()` at 0.8213 — the same function written twice, once per side of
# the stack — a REST controller against its frontend client at 0.8258, and the same S3
# bucket-config test scenarios in a PHP and a Python service at 0.7098. Noise sits
# clearly below: the best .go/.php pairs are 0.58-0.60 and match on a shared word.
#
# So this is an empirical constant, not a derived one, and it is the weakest part of the
# rule. It exists because cross-stack duplication is the most expensive kind — nobody
# sees it — and a flat threshold that finds it beats a principled one that does not.
# Set to 0.74 on 2026-09-30 after the band was read by name. Population of cross-language
# code candidates in the live mesh: 7 pairs at 0.80 and above, 44 in [0.75, 0.80), 176 in
# [0.70, 0.75), 551 in [0.65, 0.70). A threshold of 0.80 would keep seven pairs for the
# whole mesh and discard nearly every real backend/frontend correspondence — the value is
# not concentrated in the top percentiles here, because different syntax puts a ceiling on
# the cosine between two implementations of the same thing.
#
# What [0.75, 0.80) holds, sampled: `EventsApi.ts` against `NotificationsController.php`
# method by method, `Formatter::format()` against `index.ts::numberFormatPhp()` (the name
# says what it is), `formatSecondsToTime` on both sides of the stack, and the same status
# enum in PHP and TS. What [0.70, 0.75) holds: still mostly real — `getTopProducts()`
# against `TopProducts.ts::endpoint()`, `S3Api.ts::getChunkUploadUrl()` against
# `S3UploadService::getPartUploadUri()` — with noise appearing (a migration against a
# status constant, `runner.py` against `procgroup_posix.go`). 0.74 keeps the first band
# whole and the stronger half of the second.
CROSS_LANGUAGE_THRESHOLD = 0.74

# Extensions that count as one language for calibration purposes.
# One family, because for this purpose they are one language: measured 2026-09-30, .js/.ts
# admitted 40.7% and .tsx/.vue 32.4% when treated as cross-language against the flat
# threshold, against 13-17% for every genuinely distinct pair. A Vue single-file component
# is TypeScript or JavaScript with a template around it, and .js against .ts is the same
# language with types erased, so the flat cross-language value is far too permissive there.
LANGUAGE_FAMILIES = (
    frozenset({".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs", ".vue"}),
)

# Only code carries an "approach". Measured 2026-09-30: without this, 86 of 968 live
# `.json`/`.json` candidates were emitted as similar_approach, and in the test corpus 47
# of 58 — two services' `package.json` or `tsconfig.json` resembling each other says
# nothing a reader can act on, and it consumes the same budget as real analogues, which
# is the failure the same-path rule already fixed once. Documentation behaves the same
# way: `.mdx` against `.tsx` enters as cross-language and pairs Storybook docs with the
# components they document.
#
# Calibration still reads copies of ANY extension — a shared `composer.json` is a fine
# measuring stick for what "identical" scores in a repo pair. Only emission is restricted.
CODE_EXTENSIONS = frozenset(
    {
        ".php",
        ".py",
        ".go",
        ".rb",
        ".java",
        ".kt",
        ".cs",
        ".rs",
        ".swift",
        ".scala",
        ".ts",
        ".tsx",
        ".mts",
        ".cts",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".vue",
        ".c",
        ".h",
        ".cc",
        ".cpp",
        ".hpp",
        ".m",
        ".mm",
        ".pp",
        ".scss",
        ".css",
        ".sql",
    }
)

# Candidate generation floor. It must sit CLEARLY below every threshold the rules above
# can produce, for two reasons found by measurement on 2026-09-30. A floor equal to the
# lowest admission (IDENTITY_FLOOR * IDENTITY_RELATIVE = 0.69) left pairs near the
# identity floor admitting 72-98% of their candidates, because their threshold was the
# floor. And identity itself is a median over the copies that clear this floor, so a
# floor near the copy distribution censors it upward — live copy scores run down to
# 0.6915, which a 0.69 floor cut off.
CANDIDATE_FLOOR = 0.60

LSH_NUM_HYPERPLANES = 12
LSH_SEED = 1337

# Cap on rows per sims-matrix chunk inside one LSH bucket: bounds the
# transient b x chunk float32 product (a hot 4096-row bucket would
# otherwise allocate a 64MB b x b matrix in one shot).
BUCKET_CHUNK_ROWS = 1024

# Hard cap on how many rows of one LSH bucket take part in the pairwise
# comparison. A degenerate bucket (thousands of near-identical vectors hashed
# alike) costs O(rows^2) comparisons, so one such bucket can dominate an entire
# run. Rows past the cap are skipped in the deterministic row order built
# above, and the truncation is logged.
MAX_BUCKET_COMPARE_ROWS = 4096

# Above this many resident vectors the LSH path is used; at or below it every
# cross-repo pair is compared exactly. Measured on the live corpus 2026-09-30:
# 23 repos, 39650 vectors, and one exact pass over all 253 repo pairs took 56 s
# against a ~6 min full sync — a 15% cost for full recall.
#
# What that recall was: of 3686 cem.hub<->cem.messages pairs at cosine >= 0.82,
# only 904 (24.5%) shared an LSH bucket and were ever compared. The rest were
# discarded before the threshold, the top-k heaps or the per-repo-pair cap could
# see them. `JwtAuthenticator` <-> `JwtAuthenticator` scores 0.8792 and is mutual
# top-5 on both sides, and it was missed for exactly this reason. Theory agrees:
# with 12 hyperplanes two vectors at cosine 0.88 share a bucket 12.8% of the time.
EXACT_MAX_ROWS = 80_000


def cosine_similarity(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    # Mixed-dimension vectors (e.g. a shard embedded under two different
    # models) carry no comparable signal — score them 0 instead of raising.
    if len(a) != len(b):
        return 0.0
    vec_a = np.asarray(a, dtype=np.float32)
    vec_b = np.asarray(b, dtype=np.float32)
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def _planes_matrix(
    dim: int, num_planes: int = LSH_NUM_HYPERPLANES, seed: int = LSH_SEED
) -> np.ndarray:
    """Deterministic random hyperplanes for LSH bucketing.

    CRITICAL determinism constraint: hyperplane VALUES are generated via
    `random.Random(seed).gauss(0.0, 1.0)` in the exact loop order of the
    pre-numpy `_random_hyperplanes` (planes outer loop, dim inner loop),
    then converted with `np.asarray(..., dtype=np.float32)` — that part is
    still binding. The resulting bucket signatures are deterministic for
    identical inputs but are NOT guaranteed bit-identical to the retired
    pre-numpy scalar path (see the comment on `_bucket_signature_batch`).
    """
    rng = random.Random(seed)  # noqa: S311 - deterministic LSH hyperplanes, not cryptographic
    planes = [[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(num_planes)]
    return np.asarray(planes, dtype=np.float32)  # shape (num_planes, dim)


def _bucket_signature_batch(matrix: np.ndarray, planes: np.ndarray) -> list[str]:
    # Signatures are deterministic for identical inputs on the same platform
    # (same process or a fresh one), but are NOT guaranteed bit-identical to
    # the pre-numpy scalar implementation: dot products near a hyperplane
    # boundary can flip sign under batched/BLAS summation vs. Python's
    # sequential sum, regardless of dtype. LSH tolerates this by design — a
    # boundary flip only changes which candidate pairs get compared; see
    # the module docstring's known-limitation paragraph on boundary misses.
    # matrix (n, dim) @ planes.T (dim, p) -> (n, p); sign bit per plane.
    dots = matrix @ planes.T
    bits = dots >= 0
    return ["".join("1" if b else "0" for b in row) for row in bits]


def _exact_pairs(
    vectors_by_repo: dict[str, RepoVectors],
    repo_ids: list[str],
    keys: list[str],
    row_refs: list[tuple[str, int]],
    top_k: int,
    threshold: float,
) -> list[tuple[str, str, float]]:
    """Every cross-repo pair compared, no bucketing.

    Same contract as the LSH path: per-key top-k heaps ordered by
    (score, -generation_rank), then the mutual filter, then a globally
    descending sort. Only which pairs are *offered* to the heaps differs, and
    here it is all of them, so the result is the exact mutual top-k set.

    Deterministic: repos in sorted order, rows ascending inside each, which is
    the order `keys` and `row_refs` were built in.
    """
    offsets: dict[str, int] = {}
    cursor = 0
    for repo_id in repo_ids:
        offsets[repo_id] = cursor
        cursor += len(vectors_by_repo[repo_id].keys)

    candidates: dict[str, list[tuple[float, int, str]]] = {key: [] for key in keys}
    next_rank = 0
    for index_a, repo_a in enumerate(repo_ids):
        rows_a = vectors_by_repo[repo_a].normalized()
        count_a = len(vectors_by_repo[repo_a].keys)
        base_a = offsets[repo_a]
        for repo_b in repo_ids[index_a + 1 :]:
            rows_b = vectors_by_repo[repo_b].normalized()
            count_b = len(vectors_by_repo[repo_b].keys)
            base_b = offsets[repo_b]
            # Both sides are sliced, never materialized whole: `normalized()` is a
            # lazy view precisely so the mmap'd shard is not pulled into RAM, and
            # a `.T` on it would defeat that. Memory here is one chunk x chunk
            # block of float32.
            for start_a in range(0, count_a, BUCKET_CHUNK_ROWS):
                end_a = min(start_a + BUCKET_CHUNK_ROWS, count_a)
                block_a = rows_a[start_a:end_a]
                for start_b in range(0, count_b, BUCKET_CHUNK_ROWS):
                    end_b = min(start_b + BUCKET_CHUNK_ROWS, count_b)
                    sims = block_a @ rows_b[start_b:end_b].T
                    hit_rows, hit_cols = np.nonzero(sims >= threshold)
                    for local_row, local_col in zip(hit_rows, hit_cols, strict=True):
                        key_a = keys[base_a + start_a + int(local_row)]
                        key_b = keys[base_b + start_b + int(local_col)]
                        score = float(sims[local_row, local_col])
                        rank = next_rank
                        next_rank += 1
                        for owner_key, other_key in ((key_a, key_b), (key_b, key_a)):
                            heap = candidates[owner_key]
                            entry = (score, -rank, other_key)
                            if len(heap) < top_k:
                                heapq.heappush(heap, entry)
                            elif heap and entry > heap[0]:
                                heapq.heapreplace(heap, entry)

    return _mutual_from_candidates(candidates)


def mutual_top_k_pairs(
    vectors_by_repo: dict[str, RepoVectors],
    top_k: int,
    threshold: float = SIMILARITY_THRESHOLD_DEFAULT,
    num_planes: int = LSH_NUM_HYPERPLANES,
    seed: int = LSH_SEED,
) -> list[tuple[str, str, float]]:
    """Cross-repo-only mutual top-k similarity pairs.

    Args:
        vectors_by_repo: repo_id -> RepoVectors, the per-repo containers
            already resident from the embedding stage. `repo_id` is used to
            enforce the cross-repo-only constraint (WS4 ship order item 2)
            — same-repo pairs are never candidates regardless of
            similarity.
        top_k: cap on candidates kept per node before the mutual filter.
        threshold: minimum cosine similarity to be a candidate at all.

    Returns:
        List of (key_a, key_b, score) with key_a < key_b (deterministic
        ordering, no duplicate/reversed pairs), where each of key_a/key_b is
        in the other's top-k candidate list ("mutual" top-k) and the repos
        differ.
    """
    repo_ids = sorted(repo_id for repo_id, rv in vectors_by_repo.items() if len(rv))
    if not repo_ids:
        return []
    dim = vectors_by_repo[repo_ids[0]].dim

    # Repos whose dim differs from the first (sorted) repo's dim are dropped
    # entirely — same "scores zero everywhere" outcome as the old per-vector
    # drop, since cosine_similarity always rejected mixed dimensions.
    kept_repo_ids: list[str] = []
    for repo_id in repo_ids:
        rv = vectors_by_repo[repo_id]
        if rv.dim != dim:
            log.warning(
                "similar_approach: repo %s embedding dim %d != %d — dropped from ANN",
                repo_id,
                rv.dim,
                dim,
            )
            continue
        kept_repo_ids.append(repo_id)

    # Flattened row addressing: (repo_id, local_row) per global row, in
    # sorted-repo then sorted-key order — matches the old items-dict
    # insertion order exactly, so bucket contents and pair ordering are
    # stable across this refactor.
    row_refs: list[tuple[str, int]] = []
    keys: list[str] = []
    for repo_id in kept_repo_ids:
        rv = vectors_by_repo[repo_id]
        for local_row, key in enumerate(rv.keys):
            row_refs.append((repo_id, local_row))
            keys.append(key)
    if len(keys) < 2:
        return []

    if len(keys) <= EXACT_MAX_ROWS:
        return _exact_pairs(vectors_by_repo, kept_repo_ids, keys, row_refs, top_k, threshold)

    log.warning(
        "similar_approach: %d rows exceeds EXACT_MAX_ROWS=%d — falling back to LSH "
        "bucketing, which compares only vectors sharing a bucket and loses most "
        "true pairs at this threshold",
        len(keys),
        EXACT_MAX_ROWS,
    )
    planes = _planes_matrix(dim, num_planes, seed)
    # Signatures per repo straight off the RAW resident matrix — no combined
    # copy, and raw (not normalized) keeps bucket assignments identical to
    # the previous implementation.
    signatures: list[str] = []
    for repo_id in kept_repo_ids:
        signatures.extend(_bucket_signature_batch(vectors_by_repo[repo_id].matrix, planes))

    buckets: dict[str, list[int]] = {}
    for row_index, sig in enumerate(signatures):
        buckets.setdefault(sig, []).append(row_index)

    # Candidate generation: only within-bucket comparisons (ANN, not
    # all-pairs). Exact cosine + threshold filtering still applies to every
    # candidate pair actually compared, so scoring is never approximate.
    # Each key keeps at most `top_k` candidates in a min-heap ordered by
    # (score, -generation_rank), so a candidate only displaces the current
    # worst one when it scores strictly higher; on a tie the earlier-generated
    # candidate stays. That is exactly what the previous
    # `sort(key=score, reverse=True)[:top_k]` produced, since Python's sort is
    # stable. No margin above `top_k` is needed: the mutual filter below asks
    # only whether each key is inside the other's own top-k set, and a
    # candidate outside a key's top-k could never pass that test anyway.
    candidates: dict[str, list[tuple[float, int, str]]] = {key: [] for key in keys}
    next_rank = 0
    for sig, bucket_rows in buckets.items():
        if len(bucket_rows) < 2:
            continue
        if len(bucket_rows) > MAX_BUCKET_COMPARE_ROWS:
            log.warning(
                "similar_approach: LSH bucket %s holds %d rows — comparing only the first %d; "
                "candidate pairs among the remaining rows are skipped",
                sig,
                len(bucket_rows),
                MAX_BUCKET_COMPARE_ROWS,
            )
            bucket_rows = bucket_rows[:MAX_BUCKET_COMPARE_ROWS]
        # Gather normalized rows for this bucket only, from each repo's
        # cached normalized() matrix — the only allocation is bucket-sized.
        sub = np.stack(
            [vectors_by_repo[row_refs[r][0]].normalized()[row_refs[r][1]] for r in bucket_rows]
        )
        for chunk_start in range(0, len(bucket_rows), BUCKET_CHUNK_ROWS):
            chunk_end = min(chunk_start + BUCKET_CHUNK_ROWS, len(bucket_rows))
            sims = sub[chunk_start:chunk_end] @ sub.T
            for ci in range(chunk_end - chunk_start):
                i = chunk_start + ci
                row_a = bucket_rows[i]
                key_a = keys[row_a]
                repo_a = row_refs[row_a][0]
                for j in range(i + 1, len(bucket_rows)):
                    row_b = bucket_rows[j]
                    repo_b = row_refs[row_b][0]
                    if repo_a == repo_b:
                        continue
                    score = float(sims[ci, j])
                    if score < threshold:
                        continue
                    key_b = keys[row_b]
                    rank = next_rank
                    next_rank += 1
                    for owner_key, other_key in ((key_a, key_b), (key_b, key_a)):
                        heap = candidates[owner_key]
                        entry = (score, -rank, other_key)
                        if len(heap) < top_k:
                            heapq.heappush(heap, entry)
                        elif heap and entry > heap[0]:
                            heapq.heapreplace(heap, entry)

    # Descending score inside each key's kept set, ties broken by generation
    # order (ascending rank; the heap stores -rank). That is what the pre-heap
    # implementation emitted, because its stable `sort(key=score, reverse=True)`
    # ran on a generation-ordered list. Order matters downstream:
    # overlay_similar.py caps each repo pair at MAX_EDGES_PER_REPO_PAIR, so the
    # pairs that survive the cap must be the highest-scoring ones. For any
    # bucket that stays under MAX_BUCKET_COMPARE_ROWS this produces exactly the
    # pairs, scores and ordering the unbounded implementation produced.
    return _mutual_from_candidates(candidates)


def _mutual_from_candidates(
    candidates: dict[str, list[tuple[float, int, str]]],
) -> list[tuple[str, str, float]]:
    """Mutual top-k filter and global ordering, shared by both candidate paths."""
    kept: dict[str, list[tuple[str, float]]] = {}
    top_candidates: dict[str, set[str]] = {}
    for key, heap in candidates.items():
        ordered_entries = sorted(heap, key=lambda entry: (-entry[0], -entry[1]))
        kept[key] = [(other, score) for score, _neg_rank, other in ordered_entries]
        top_candidates[key] = {other for other, _ in kept[key]}

    seen: set[frozenset] = set()
    pairs: list[tuple[str, str, float]] = []
    for key_a, scored in kept.items():
        for key_b, score in scored:
            if key_b not in top_candidates[key_a] or key_a not in top_candidates[key_b]:
                continue
            pair_id = frozenset((key_a, key_b))
            if pair_id in seen:
                continue
            seen.add(pair_id)
            ordered = tuple(sorted((key_a, key_b)))
            pairs.append((ordered[0], ordered[1], score))

    # Globally descending, which is what the comment above promises and what the
    # cap downstream requires. Until 2026-09-30 this returned `pairs` in the order
    # they were appended: `kept` is keyed by durable key, so the loop walked the
    # keys in path order and the per-key sort only ordered scores WITHIN one key.
    # overlay_similar.py then truncated at MAX_EDGES_PER_REPO_PAIR = 50 in list
    # order, so each repo pair's budget went to whichever files sort first —
    # `.docker/...`, `.gitlab-ci.yml`, `bin/console`, `composer.json`. Measured on
    # the 20260930T061707Z generation: of 23 cem.hub nodes carrying a
    # similar_approach edge, exactly one was PHP, out of 4243 PHP nodes, and the
    # cem.hub<->cem.messages budget was spent at scores from 0.8355 while the true
    # top 50 floor at 0.9958. find_similar therefore had no code analogues to
    # return for any PHP or TS symbol and fell through to same-repo adjacency.
    # The tie-break on keys keeps the output deterministic across runs.
    pairs.sort(key=lambda pair: (-pair[2], pair[0], pair[1]))
    return pairs
