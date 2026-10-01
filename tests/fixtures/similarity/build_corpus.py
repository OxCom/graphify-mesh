"""Rebuild the synthetic similarity corpus under tests/fixtures/similarity/.

Usage (from the repository root, host interpreter with numpy):

    python tests/fixtures/similarity/build_corpus.py [--source <embeddings dir>]

The source is the live embedding generation (`<repo>.npy` + `<repo>.meta.json`
per repo). Nothing from it survives verbatim: repo ids, paths and labels are
replaced by order-preserving synthetic names, and the vectors are projected
onto their own principal components at a lower dimension. Only the shape of
the corpus is kept — which pairs share a path, file extensions, the key order
relative to score order, and the pairwise cosines of the pairs the tests
assert on (to the tolerance printed at the end of a run).

Deterministic: the selection walks sorted inputs, the projection is an exact
SVD, and the output is byte-stable for the same source generation.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC_ROOT = HERE.parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graphify_mesh.sync import embed_similarity as es  # noqa: E402
from graphify_mesh.sync import overlay_similar as ov  # noqa: E402
from graphify_mesh.sync.vectors import RepoVectors  # noqa: E402

DEFAULT_SOURCE = "/var/www/knowledge-base/graphify/global/embeddings/current"
KEY_SEP = "\x1f"
TOP_K = 5
FIXTURE_DIM = 512

# Synthetic repo ids sort in the same order as the live ids they replace, so
# the key order of the corpus (repo, path, label) is the live one.
REPO_IDS = {
    "cem.hub": "r1.php-mixed",
    "cem.messages": "r2.php-service",
    "cem.styleguide": "r3.tsx-kit",
    "cryengine.cuas": "r4.php-app",
    "cryengine.hub": "r5.php-mixed",
    "cryengine.launcher": "r6.ts-app",
    "cryengine.styleguide": "r7.tsx-kit",
}

# One situation per live repo pair. Candidates are the pairwise mutual pairs down
# to CANDIDATE_FLOOR (what overlay_similar calibrates and admits from), grouped
# the way overlay_similar groups them: copies per extension, same-language
# analogues per extension, and cross-language analogues. Each group is sampled at
# evenly spaced quantiles of its score-sorted list, so the fixture keeps each
# group's score DISTRIBUTION — the median of an extension's copies is its identity
# level and drives the calibrated threshold, and a top-N sample would put it near
# 1.0 for every language. `extensions` lists the extensions sampled for copies
# and analogues; `None` keeps every pair of that kind.
SITUATIONS = [
    {
        "name": "php_php_large",
        "repos": ("cryengine.cuas", "cryengine.hub"),
        "extensions": (".php",),
        "copies": 30,
        "analogues": 40,
        "cross_language": 8,
    },
    {
        "name": "php_php_small",
        "repos": ("cem.hub", "cem.messages"),
        "extensions": (".php",),
        "copies": 30,
        "analogues": 25,
        "cross_language": 10,
    },
    {
        "name": "tsx_tsx",
        "repos": ("cem.styleguide", "cryengine.styleguide"),
        "extensions": (".tsx",),
        "copies": 30,
        "analogues": 27,
        "cross_language": 15,
        # .ts analogues with no .ts copies sampled: judged by the .tsx family line.
        "family_only": {".ts": 15},
    },
    {
        # php repo against a tsx repo: copies are config only, the code pairs
        # are all cross-language and judged by the flat threshold.
        "name": "cross_language",
        "repos": ("cem.messages", "cem.styleguide"),
        "extensions": (".yml",),
        "copies": None,
        "analogues": None,
        "cross_language": None,
    },
    {
        # ts/vue repo against a tsx repo: identity levels around 0.76, under
        # IDENTITY_FLOOR, so every group of the pair is declined. The .tsx
        # analogues here are React file shape, not shared approach.
        "name": "low_identity",
        "repos": ("cryengine.launcher", "cryengine.styleguide"),
        "extensions": (".tsx", ".json"),
        "copies": 30,
        "analogues": 30,
        "cross_language": 8,
        # .ts analogues with (nearly) no .ts copies: judged by the family line.
        "family_only": {".ts": 20},
        # .json analogues: their copies calibrate, they are never emitted.
        "noncode": 15,
    },
    {
        # Two mixed repos sharing php AND ts copies, with identity levels a
        # tenth apart: calibration must be per extension, not per repo pair.
        "name": "mixed_extensions",
        "repos": ("cem.hub", "cryengine.hub"),
        "extensions": (".php", ".ts", ".tsx"),
        "copies": 30,
        "analogues": 22,
        "cross_language": 15,
    },
]


def _quantile_sample(items: list, budget: int | None) -> list:
    if budget is None or budget >= len(items):
        return list(items)
    step = len(items) / budget
    return [items[int(index * step)] for index in range(budget)]


def _path_of(key: str) -> str:
    return key.split(KEY_SEP)[1]


def _ext_of(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[-1] if "." in name else ""


def _kind(key_a: str, key_b: str) -> tuple[str, str]:
    """overlay_similar's grouping of a candidate pair.

    ("copy", ext) — same path, calibrates ext's line;
    ("analogue", "ext_a/ext_b") — same-language code pair, sorted extensions;
    ("cross", "") — cross-language code pair, flat threshold;
    ("noncode", "") — one side outside CODE_EXTENSIONS, never emitted.
    """
    ext_a = ov._extension(_path_of(key_a))
    ext_b = ov._extension(_path_of(key_b))
    if _path_of(key_a) == _path_of(key_b):
        return ("copy", ext_a)
    if ext_a not in es.CODE_EXTENSIONS or ext_b not in es.CODE_EXTENSIONS:
        return ("noncode", "")
    if ov._same_language(ext_a, ext_b):
        return ("analogue", "/".join(sorted((ext_a, ext_b))))
    return ("cross", "")


def load_live(source: Path, repo: str) -> RepoVectors:
    meta = json.loads((source / f"{repo}.meta.json").read_text(encoding="utf-8"))
    matrix = np.load(source / f"{repo}.npy", mmap_mode="r")
    keys_by_row: list[str | None] = [None] * matrix.shape[0]
    for key, entry in meta["entries"].items():
        if entry.get("row") is not None:
            keys_by_row[entry["row"]] = key
    kept = [row for row, key in enumerate(keys_by_row) if key is not None]
    return RepoVectors.from_rows(
        [keys_by_row[row] for row in kept],  # type: ignore[misc]
        np.ascontiguousarray(matrix[kept], dtype=np.float32),
    )


def select_keys(live: dict[str, RepoVectors]) -> dict[str, set[str]]:
    """Endpoints of the sampled pairs, chosen per situation on the pairwise run."""
    selected: dict[str, set[str]] = {repo: set() for repo in live}
    for situation in SITUATIONS:
        repo_a, repo_b = situation["repos"]
        pairs = es.mutual_top_k_pairs(
            {repo_a: live[repo_a], repo_b: live[repo_b]},
            top_k=TOP_K,
            threshold=min(es.SIMILARITY_THRESHOLD_DEFAULT, es.CANDIDATE_FLOOR),
        )
        groups: dict[tuple[str, str], list] = collections.defaultdict(list)
        for pair in pairs:
            groups[_kind(pair[0], pair[1])].append(pair)
        chosen = []
        report = []
        for ext in situation["extensions"]:
            copies = _quantile_sample(groups[("copy", ext)], situation["copies"])
            same_ext = groups[("analogue", f"{ext}/{ext}")]
            analogues = _quantile_sample(same_ext, situation["analogues"])
            # Mixed-extension pairs inside the family (.ts/.tsx): a small sample,
            # enough for the stricter-of-two and order-independence tests.
            mixed = [
                p
                for (kind, pair_label), members in groups.items()
                if kind == "analogue"
                and ext in pair_label.split("/")
                and pair_label != f"{ext}/{ext}"
                for p in members
            ]
            chosen += copies + analogues + _quantile_sample(sorted(mixed, key=lambda p: -p[2]), 12)
            all_copies = groups[("copy", ext)]
            identity = all_copies[len(all_copies) // 2][2] if all_copies else None
            report.append(
                f"{ext}: copies {len(copies)}/{len(all_copies)} identity={identity} "
                f"analogues {len(analogues)}/{len(same_ext)} "
                f"mixed {min(12, len(mixed))}/{len(mixed)}"
            )
        for ext, budget in situation.get("family_only", {}).items():
            members = groups[("analogue", f"{ext}/{ext}")]
            chosen += _quantile_sample(members, budget)
            report.append(f"{ext} family-only analogues {min(budget, len(members))}/{len(members)}")
        if situation.get("noncode"):
            members = groups[("noncode", "")]
            chosen += _quantile_sample(members, situation["noncode"])
            report.append(f"noncode {min(situation['noncode'], len(members))}/{len(members)}")
        cross = _quantile_sample(groups[("cross", "")], situation["cross_language"])
        chosen += cross
        report.append(f"cross-language {len(cross)}/{len(groups[('cross', '')])}")
        print(f"{situation['name']}: {'; '.join(report)}")
        for key_a, key_b, _ in chosen:
            selected[key_a.split(KEY_SEP)[0]].add(key_a)
            selected[key_b.split(KEY_SEP)[0]].add(key_b)
    return selected


def anonymize(selected: dict[str, set[str]]) -> dict[str, str]:
    """Order-preserving rename: same live path -> same synthetic path, extension kept."""
    paths = sorted({_path_of(key) for keys in selected.values() for key in keys})
    labels = sorted({key.split(KEY_SEP)[2] for keys in selected.values() for key in keys})
    path_map = {}
    for index, path in enumerate(paths):
        ext = _ext_of(path)
        path_map[path] = f"src/p{index:04d}" + (f".{ext}" if ext else "")
    label_map = {label: f"L{index:04d}" for index, label in enumerate(labels)}
    key_map: dict[str, str] = {}
    for repo, keys in selected.items():
        for key in keys:
            _, path, label = key.split(KEY_SEP, 2)
            key_map[key] = KEY_SEP.join((REPO_IDS[repo], path_map[path], label_map[label]))
    # A collision here would merge two nodes; the maps are injective by
    # construction, this only proves it.
    assert len(set(key_map.values())) == len(key_map)
    return key_map


def project(stacked: np.ndarray, dim: int) -> np.ndarray:
    """Exact SVD onto the top `dim` principal directions of the corpus itself.

    Not centred: cosine is about direction from the origin, and centring would
    change every angle. The result is a rotation followed by a truncation.
    """
    _, _, vt = np.linalg.svd(stacked.astype(np.float64), full_matrices=False)
    return (stacked.astype(np.float64) @ vt[:dim].T).astype(np.float32)


def load_fixture(directory: Path) -> dict[str, RepoVectors]:
    """The loader the tests use; kept here so the generator verifies what they see."""
    manifest = json.loads((directory / "corpus.json").read_text(encoding="utf-8"))
    out = {}
    for repo_id, keys in manifest["keys"].items():
        matrix = np.load(directory / f"{repo_id}.npy").astype(np.float32)
        out[repo_id] = RepoVectors.from_rows(list(keys), matrix)
    return out


def _line_of(copies: list[float]) -> float | None:
    """A (repo pair, extension) group's line: None when uncalibrated, 2.0 when floored."""
    if len(copies) < es.MIN_CALIBRATION_COPIES:
        return None
    ordered = sorted(copies)
    identity = ordered[len(ordered) // 2]
    return 2.0 if identity < es.IDENTITY_FLOOR else identity * es.IDENTITY_RELATIVE


def _applied(
    lines: dict[str, float], ext_a: str, ext_b: str, threshold: float
) -> tuple[float, str]:
    """overlay_similar._calibrated_line restated: (applied threshold, how).

    `lines` maps extension -> line for one repo pair (floored groups carry 2.0).
    """
    own = [lines[ext] for ext in (ext_a, ext_b) if ext in lines]
    family = next((f for f in es.LANGUAGE_FAMILIES if ext_a in f or ext_b in f), frozenset())
    pooled = [line for ext, line in lines.items() if ext in family]
    if own:
        line, how = max(own), "calibrated"
    elif pooled:
        line, how = max(pooled), "family"
    else:
        return threshold, "constant"
    # A floored group carries 2.0, and the stricter-of rule makes it win: declined.
    return line, ("declined" if line >= 2.0 else how)


def expectations(corpus: dict[str, RepoVectors]) -> dict:
    """Everything the tests pin, computed on the files as written.

    Two candidate lists: at the default threshold, for the pair-level tests, and
    at CANDIDATE_FLOOR, which is what overlay_similar calibrates and admits from.
    The admission rule is repeated here (`_kind`, `_line_of`, `_applied`) so a
    change to that rule fails against these numbers.
    """
    threshold = es.SIMILARITY_THRESHOLD_DEFAULT
    pairs = es.mutual_top_k_pairs(corpus, top_k=TOP_K)
    candidates = es.mutual_top_k_pairs(
        corpus, top_k=TOP_K, threshold=min(threshold, es.CANDIDATE_FLOOR)
    )

    grouped: dict[tuple, list] = {}
    for key_a, key_b, score in candidates:
        repo_pair = tuple(sorted((key_a.split(KEY_SEP)[0], key_b.split(KEY_SEP)[0])))
        grouped.setdefault((repo_pair, *_kind(key_a, key_b)), []).append((key_a, key_b, score))

    def lines_for(repo_pair) -> dict[str, float]:
        out = {}
        for (rp, kind, ext), members in grouped.items():
            if rp == repo_pair and kind == "copy":
                line = _line_of([s for _, _, s in members])
                if line is not None:
                    out[ext] = line
        return out

    def describe(repo_pair) -> dict:
        lines = lines_for(repo_pair)
        calibration = {}
        for (rp, kind, ext), members in grouped.items():
            if rp != repo_pair or kind != "copy":
                continue
            ordered = sorted(s for _, _, s in members)
            calibration[ext] = {
                "copies": len(ordered),
                "identity_level": ordered[len(ordered) // 2],
                "line": lines.get(ext),
                "code": ext in es.CODE_EXTENSIONS,
            }
        emission = {}
        for (rp, kind, pair_label), members in grouped.items():
            if rp != repo_pair or kind != "analogue":
                continue
            ext_a, ext_b = pair_label.split("/")
            applied, how = _applied(lines, ext_a, ext_b, threshold)
            emitted = [p for p in members if p[2] >= applied]
            weakest = emitted[-1] if emitted else None
            same_ext_copies = [s for _, _, s in grouped.get((repo_pair, "copy", ext_a), [])]
            emission[pair_label] = {
                "analogues": len(members),
                "admission": how,
                "admission_threshold": applied,
                "emitted": len(emitted),
                "admitted_share": len(emitted) / len(members),
                "emitted_at_default_threshold": sum(1 for p in members if p[2] >= threshold),
                "best_analogue_score": members[0][2],
                "weakest_emitted_pair": list(weakest) if weakest else None,
                "copies_ranked_above_weakest_emitted": (
                    sum(1 for s in same_ext_copies if s > weakest[2])
                    if weakest
                    else len(same_ext_copies)
                ),
            }
        cross = grouped.get((repo_pair, "cross", ""), [])
        cross_emitted = [p for p in cross if p[2] >= es.CROSS_LANGUAGE_THRESHOLD]
        noncode = grouped.get((repo_pair, "noncode", ""), [])
        return {
            "repos": list(repo_pair),
            "candidates_total": sum(len(v) for (rp, _, _), v in grouped.items() if rp == repo_pair),
            "calibration": dict(sorted(calibration.items())),
            "emission": dict(sorted(emission.items())),
            "cross_language": {
                "candidates": len(cross),
                "emitted": len(cross_emitted),
                "best_score": cross[0][2] if cross else None,
                "first_rejected_score": next(
                    (p[2] for p in cross if p[2] < es.CROSS_LANGUAGE_THRESHOLD), None
                ),
            },
            "noncode_pairs": {
                "candidates": len(noncode),
                "above_any_line": sum(
                    1 for p in noncode if p[2] >= min(list(lines.values()) + [threshold])
                ),
            },
            "emitted": sum(e["emitted"] for e in emission.values()) + len(cross_emitted),
        }

    situations = {
        situation["name"]: describe(tuple(sorted(REPO_IDS[r] for r in situation["repos"])))
        for situation in SITUATIONS
    }
    all_repo_pairs = sorted({rp for (rp, _, _) in grouped})
    situation_pairs = {tuple(s["repos"]) for s in situations.values()}
    other_pairs = {
        "<->".join(rp): describe(rp) for rp in all_repo_pairs if rp not in situation_pairs
    }

    # Admitted share of available code analogues by extension pair over the whole
    # fixture, the shape overlay_similar's evenness claim is made in.
    by_ext_pair: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for rp in all_repo_pairs:
        lines = lines_for(rp)
        for (rp2, kind, _pair_label), members in grouped.items():
            if rp2 != rp or kind not in ("analogue", "cross"):
                continue
            for key_a, key_b, score in members:
                ext_a = ov._extension(_path_of(key_a))
                ext_b = ov._extension(_path_of(key_b))
                if kind == "analogue":
                    applied, _ = _applied(lines, ext_a, ext_b, threshold)
                else:
                    applied = es.CROSS_LANGUAGE_THRESHOLD
                label = "/".join(sorted((ext_a, ext_b)))
                by_ext_pair[label][0] += 1
                by_ext_pair[label][1] += int(score >= applied)
    share_by_extension_pair = {
        label: {"available": available, "admitted": admitted}
        for label, (available, admitted) in sorted(by_ext_pair.items())
    }

    # A true pair whose vectors hash to different LSH buckets under the module's
    # own planes: the reason the exact path exists.
    dim = next(iter(corpus.values())).dim
    planes = es._planes_matrix(dim)
    signature: dict[str, str] = {}
    for repo_id in sorted(corpus):
        rv = corpus[repo_id]
        for key, sig in zip(rv.keys, es._bucket_signature_batch(rv.matrix, planes), strict=True):
            signature[key] = sig
    split = [(a, b, s) for a, b, s in pairs if signature[a] != signature[b]]
    shared = len(pairs) - len(split)
    # Prefer a cross-path analogue as the example: that is the kind of pair the
    # live corpus lost, a copy at the same path would be found by other means.
    split_cross = [p for p in split if _path_of(p[0]) != _path_of(p[1])]
    lsh_missed = (split_cross or split or [None])[0]

    return {
        "top_k": TOP_K,
        "threshold": threshold,
        "candidate_floor": es.CANDIDATE_FLOOR,
        "identity_relative": es.IDENTITY_RELATIVE,
        "identity_floor": es.IDENTITY_FLOOR,
        "min_calibration_copies": es.MIN_CALIBRATION_COPIES,
        "cross_language_threshold": es.CROSS_LANGUAGE_THRESHOLD,
        "language_families": [sorted(family) for family in es.LANGUAGE_FAMILIES],
        "code_extensions": sorted(es.CODE_EXTENSIONS),
        "dim": dim,
        "rows": sum(len(rv) for rv in corpus.values()),
        "pairs_total": len(pairs),
        "candidates_total": len(candidates),
        "pairs_sharing_lsh_bucket": shared,
        "pairs_split_across_lsh_buckets": len(split),
        "lsh_missed_pair": list(lsh_missed) if lsh_missed else None,
        "share_by_extension_pair": share_by_extension_pair,
        "situations": situations,
        "other_repo_pairs": other_pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--dim", type=int, default=FIXTURE_DIM)
    args = parser.parse_args()

    live = {repo: load_live(args.source, repo) for repo in REPO_IDS}
    selected = select_keys(live)
    key_map = anonymize(selected)

    # Stack in synthetic key order (the order RepoVectors will hold them in).
    ordered = sorted(key_map.items(), key=lambda item: item[1])
    stacked = np.stack(
        [live[key.split(KEY_SEP)[0]].get(key) for key, _ in ordered]  # type: ignore[arg-type]
    )
    # float16 on disk halves the size; the round trip is part of the error
    # measured below, since the tests only ever see the reloaded values.
    reduced = project(stacked, args.dim).astype(np.float16).astype(np.float32)

    # Projection error on the pairs that matter: every cross-repo pair at or
    # above the threshold in the source dimension.
    unit_full = stacked / np.linalg.norm(stacked, axis=1, keepdims=True)
    unit_low = reduced / np.linalg.norm(reduced, axis=1, keepdims=True)
    repos_of = np.array([key.split(KEY_SEP)[0] for key, _ in ordered])
    full_cos = unit_full @ unit_full.T
    low_cos = unit_low @ unit_low.T
    cross_repo = repos_of[:, None] != repos_of[None, :]
    strong = (full_cos >= es.SIMILARITY_THRESHOLD_DEFAULT) & cross_repo
    max_err = float(np.abs(full_cos[strong] - low_cos[strong]).max())
    print(
        f"rows={len(ordered)} dim {stacked.shape[1]} -> {args.dim}: max |cos - cos'| on "
        f"{int(strong.sum()) // 2} threshold-passing cross-repo pairs = {max_err:.4f}"
    )

    manifest_keys: dict[str, list[str]] = collections.defaultdict(list)
    rows_by_repo: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    for (_, synthetic), row in zip(ordered, reduced, strict=True):
        repo_id = synthetic.split(KEY_SEP)[0]
        manifest_keys[repo_id].append(synthetic)
        rows_by_repo[repo_id].append(row)
    for repo_id, rows in rows_by_repo.items():
        np.save(HERE / f"{repo_id}.npy", np.stack(rows).astype(np.float16))
    manifest = {
        "dim": args.dim,
        "vector_dtype_on_disk": "float16",
        "key_separator": "\\x1f",
        "keys": {repo_id: manifest_keys[repo_id] for repo_id in sorted(manifest_keys)},
        "projection_max_abs_cosine_error": round(max_err, 5),
    }
    (HERE / "corpus.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    expected = expectations(load_fixture(HERE))
    (HERE / "expected.json").write_text(
        json.dumps(expected, indent=1, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(expected, indent=1, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
