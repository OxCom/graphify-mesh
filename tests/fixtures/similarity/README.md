# Synthetic similarity corpus

Test data for `graphify_mesh.sync.embed_similarity` and `overlay_similar`, used by
`tests/sync/test_embed_similarity.py`. Everything here is synthetic: no repository
name, path, symbol or embedding from the source generation is present.

## How it was derived

`build_corpus.py` reads seven shards of a live embedding generation
(`<repo>.npy` + `<repo>.meta.json`, 1024-dim float32) and keeps only the endpoints
of a sample of pairs, chosen by running `mutual_top_k_pairs` (top_k 5, candidates
down to `CANDIDATE_FLOOR`) on each of six repo pairs. Candidates are grouped the
way `overlay_similar` groups them — copies (same path in both repos) per
extension, same-language analogues per extension, cross-language analogues — and
each group is sampled at evenly spaced quantiles of its score-sorted list, so the
fixture keeps each group's score distribution: the median of an extension's copies
is its identity level and drives the calibrated threshold, and a top-N sample
would put it near 1.0 for every language.

| situation | what it encodes | sampled per extension (copies + analogues), cross-language |
|---|---|---|
| `php_php_large` | php/php pair with the most analogues; `.php` identity ~0.87 | 30 + 40, 8 |
| `php_php_small` | php/php pair with far fewer analogues; `.php` identity ~0.91 | 30 + 25, 10 |
| `tsx_tsx` | tsx/tsx pair whose analogues top out around 0.92; `.tsx` identity ~0.81, just above the floor; plus 15 `.ts/.ts` analogues with no `.ts` copies (family line) | 30 + 27 (+12 `.ts/.tsx`), 0 |
| `cross_language` | php repo vs tsx repo: config copies only, code pairs all cross-language | every pair |
| `low_identity` | ts/vue repo vs tsx repo: `.tsx` and `.json` identity ~0.76, under `IDENTITY_FLOOR`, so every group of the pair is declined; plus 20 `.ts/.ts` analogues with no `.ts` copies (declined through the family) and 15 non-code pairs (never emitted) | 30 + 30 (+12 `.ts/.tsx`), 8 |
| `mixed_extensions` | two mixed repos sharing `.php` (identity ~0.91), `.ts` (~0.81) and `.tsx` (~0.83) copies: calibration per extension, stricter-of-two for `.ts/.tsx` | 30 + 22 (+12 mixed), 15 |

Repo pairs outside the situations (formed by the joint run) are recorded under
`other_repo_pairs`; one of them has too few copies of any family, which is the
constant-fallback case.

The joint run over the whole fixture forms extra pairs among the sampled nodes and
drops some mutuality, so the recorded counts differ from the sample sizes.

Then, in order:

1. **Anonymize.** Repo ids, paths and labels are replaced by order-preserving
   synthetic names (`r1.php-mixed`, `src/p0136.php`, `L0150`). The same live path
   maps to the same synthetic path, the extension is kept, and sorted key order is
   the live one, so key order versus score order is what it was.
2. **Project.** Vectors are rotated onto their own top 512 principal directions
   (exact SVD, not centred) and stored as float16. The generator prints the largest
   change in cosine over every threshold-passing cross-repo pair, after the float16
   round trip; `corpus.json` records it (`projection_max_abs_cosine_error`).
3. **Record.** `expected.json` holds what the module computes on the files as
   written, per repo pair: `calibration` (per extension: copies, identity level,
   line = upper median times `IDENTITY_RELATIVE`, whether the extension is code),
   `emission` (per sorted extension pair of a same-language code group: how it is
   admitted — `calibrated`, `family`, `constant`, or `declined` when the line it
   inherits is a floored one — and at which line, analogues
   available and emitted, the count at the absolute threshold, the weakest emitted
   analogue and how many copies outrank it), `cross_language` (candidates and how
   many clear `CROSS_LANGUAGE_THRESHOLD`) and `noncode_pairs`; the admitted share
   by extension pair over the whole fixture; the constants everything was computed
   under; and one true pair whose two vectors hash to different LSH buckets under
   `_planes_matrix(512)`. Nothing recorded depends on `MAX_EDGES_PER_REPO_PAIR`:
   the tests that exercise the cap lower it themselves.

## Files

- `corpus.json` — dim, on-disk dtype, and the sorted keys per synthetic repo
  (`repo\x1fpath\x1flabel`, row order of the matching `.npy`).
- `r1.php-mixed.npy` … `r7.tsx-kit.npy` — float16 matrices, one row per key.
- `expected.json` — the recorded measurements the tests pin.
- `build_corpus.py` — the generator.

## Regenerating

```bash
python tests/fixtures/similarity/build_corpus.py            # default source
python tests/fixtures/similarity/build_corpus.py --source <embeddings dir> --dim 512
```

Output is deterministic for the same source generation. Regenerate `expected.json`
together with the vectors, never by hand; `test_corpus_matches_its_recorded_shape`
fails when the two drift apart. A change to the threshold, `CANDIDATE_FLOOR`,
`IDENTITY_RELATIVE`, `IDENTITY_FLOOR`, `MIN_CALIBRATION_COPIES`,
`CROSS_LANGUAGE_THRESHOLD`, `LANGUAGE_FAMILIES` or `CODE_EXTENSIONS` also needs a
regeneration; a change to `MAX_EDGES_PER_REPO_PAIR` does not.
