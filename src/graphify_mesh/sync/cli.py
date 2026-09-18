"""graphify-mesh-sync — sync-pipeline entrypoint.

Discovers all registered/discoverable per-project graphify collections under
the configured scan roots, decides update vs extract per project, rebuilds the
global graph from empty via `graphify merge-graphs` (never `graphify global
add` — see graphify_mesh/sync/__init__.py for the merge-semantics evidence),
validates the result, and atomically publishes a new generation.

Intended to be invoked by a scheduler (e.g. systemd Type=oneshot on a timer;
see examples/systemd/graphify-mesh-sync.{service,timer}) — this is a single
run (`--once` is the only supported mode; there is no daemon loop).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from graphify_mesh.sync.config import (
    EXTRACT_MIN_CONCURRENCY,
    SCAN_MAX_DEPTH,
    SCAN_MIN_DEPTH,
    Settings,
)
from graphify_mesh.sync.locking import LockHeldError
from graphify_mesh.sync.perms import audit_config_permissions
from graphify_mesh.sync.pipeline import run


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print every action; write nothing outside a private staging dir.",
    )
    parser.add_argument(
        "--once", action="store_true", help="Single run (the only supported mode; no daemon loop)."
    )
    parser.add_argument(
        "--mesh-root", type=Path, default=None, help="Override the graph-mesh repo root (testing)."
    )
    parser.add_argument(
        "--scan-root",
        dest="scan_roots",
        # Deliberately str, not Path: Settings.from_env's shared
        # empty/whitespace-value normalization (_clean_root_args) needs the
        # raw string to tell an explicit `--scan-root ""` apart from a real
        # path — `Path("")` already collapses to `Path(".")`, which would
        # silently resolve to cwd instead of being dropped.
        type=str,
        action="append",
        default=None,
        help=(
            "Scan root for graphify-out discovery; repeatable. Default: "
            "GRAPHIFY_MESH_SCAN_ROOTS (colon-separated), then "
            "GRAPHIFY_MESH_SCAN_ROOT, then current working directory."
        ),
    )
    parser.add_argument(
        "--scan-depth",
        type=int,
        default=None,
        help=(
            "Max nesting of a project dir below a scan root (default 4, range "
            f"{SCAN_MIN_DEPTH}-{SCAN_MAX_DEPTH}). Default: GRAPHIFY_MESH_SCAN_DEPTH."
        ),
    )
    parser.add_argument(
        "--registry", type=Path, default=None, help="Override the registry.json path (testing)."
    )
    # WS2 (naming) and WS3 (embedding) are both wired into the pipeline, so
    # both stages run by default now — a real run is expected to name
    # communities and embed nodes. --skip-* remain as explicit opt-outs for
    # fast local-only runs or when the Ollama host is unreachable.
    parser.add_argument(
        "--skip-labeling",
        action="store_true",
        default=False,
        help="Skip the naming stage and its non-placeholder community_name check. Default: off.",
    )
    parser.add_argument(
        "--no-skip-labeling",
        dest="skip_labeling",
        action="store_false",
        help="(default) Enforce the naming stage and its community_name check.",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        default=False,
        help="Skip the embedding stage. Default: off.",
    )
    parser.add_argument(
        "--no-skip-embedding",
        dest="skip_embedding",
        action="store_false",
        help="(default) Run the embedding stage.",
    )
    parser.add_argument(
        "--allow-shrink",
        action="store_true",
        help="Explicitly authorize a smaller published graph than the previous generation.",
    )
    parser.add_argument(
        "--extract-concurrency",
        type=int,
        default=None,
        help=(
            "Max concurrent `graphify extract`/`update` children (default 2, floor 1). "
            "Each child's RSS lands in this process's cgroup MemoryMax, so raise with "
            "care; never derive from repo count."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    overrides: dict = {
        "dry_run": args.dry_run,
        "skip_labeling": args.skip_labeling,
        "skip_embedding": args.skip_embedding,
        "allow_shrink": args.allow_shrink,
    }
    if args.extract_concurrency is not None:
        # Same hard floor as the env-var path (_extract_concurrency_from_env):
        # a bad/too-low CLI value degrades to fully sequential rather than
        # reaching ThreadPoolExecutor(max_workers=0), which raises.
        overrides["extract_concurrency"] = max(args.extract_concurrency, EXTRACT_MIN_CONCURRENCY)
    if args.scan_depth is not None:
        # Same floor-and-ceiling clamp as the env-var path
        # (_scan_depth_from_env): a bad/too-low CLI value degrades to the
        # shallowest scan rather than a zero/negative-depth walk, and a
        # too-high value is capped rather than triggering an unbounded walk.
        overrides["scan_depth"] = min(max(args.scan_depth, SCAN_MIN_DEPTH), SCAN_MAX_DEPTH)

    settings = Settings.from_env(
        mesh_root=args.mesh_root,
        scan_roots=args.scan_roots,
        registry_path=args.registry,
        **overrides,
    )

    # Before the run, so it also fires under --dry-run: a dry run is then a
    # way to check the configuration's permissions without publishing.
    audit_config_permissions(settings.registry_path, settings.manual_relations_path)

    try:
        report = run(settings)
    except LockHeldError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    print(
        json.dumps(
            {
                "dry_run": report.dry_run,
                "reconciliation": report.reconciliation,
                "project_actions": report.project_actions,
                "stale_repos": report.stale_repos,
                "dirty_repos": report.dirty_repos,
                "merge_ok": report.merge_ok,
                "merge_error": report.merge_error,
                "validation_ok": report.validation_ok,
                "validation_errors": report.validation_errors,
                "published": report.published,
                "publish_blocked_reason": report.publish_blocked_reason,
                "generation_id": report.generation_id,
                "skipped_stages": report.skipped_stages,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
