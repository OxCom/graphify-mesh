"""`depends_on` overlay edges + manual relations.

Identity resolution: `registry.json` is the source of truth for "is this
dependency one of our registered repos". A registered repo's package identity
(composer `name` / npm `name`) is read fresh from its manifest file every run
— never cached across generations — and matched against what a *different*
registered repo's manifest declares as a dependency.

Example lockfile survey across a multi-repo set:
  - composer.json + composer.lock: PHP/Symfony backends (e.g. backend-a).
  - package.json + package-lock.json (npm): TS/React frontends (e.g.
    frontend-b, and a shared styleguide package).
  - package.json + pnpm-lock.yaml: a service that uses pnpm.
  - yarn.lock is also supported if present.
A common shape: a composer.json may require only third-party/framework
packages (no cross-repo edge), while npm does declare cross-repo edges — e.g.
`frontend-b` depends on `@example-org/styleguide` (another registered repo's
own package name). Both composer and npm identity resolution are implemented
regardless, since which ecosystem a given repo-to-repo relationship shows up
in is not guaranteed to stay fixed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from graphify_mesh.sync.overlay_refs import (
    LogicalRef,
    OverlayEdge,
    require_resolved,
)

log = logging.getLogger("graphify_mesh.sync.overlay_depends")

PROVENANCE_MANIFEST_LOCK = "EXTRACTED_CONFIG"
PROVENANCE_MANUAL = "MANUAL"

DEPENDENCY_KIND_RUNTIME = "runtime"
DEPENDENCY_KIND_DEV = "dev"

CONFIDENCE_RUNTIME = 0.95
CONFIDENCE_DEV = 0.8
CONFIDENCE_MANUAL = 1.0


def _read_json_safe(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not parse %s: %s", path, exc)
        return None


def _read_yaml_safe(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        log.warning("could not parse %s: %s", path, exc)
        return None


def build_package_identity_map(repo_roots: dict[str, Path]) -> dict[str, str]:
    """{package_name (composer or npm) -> repo_id}, read fresh from each
    registered repo's root every call — never persisted across generations."""
    mapping: dict[str, str] = {}
    for repo_id, root in sorted(repo_roots.items()):
        for manifest_name in ("composer.json", "package.json"):
            data = _read_json_safe(root / manifest_name)
            if not data:
                continue
            name = data.get("name")
            if not name or not isinstance(name, str):
                continue
            if name in mapping and mapping[name] != repo_id:
                log.warning(
                    "package identity %r claimed by both %r and %r; keeping first",
                    name,
                    mapping[name],
                    repo_id,
                )
                continue
            mapping[name] = repo_id
    return mapping


def _composer_lock_names(root: Path) -> tuple[set[str], set[str]]:
    """(runtime_locked_names, dev_locked_names) from composer.lock."""
    lock = _read_json_safe(root / "composer.lock")
    if not lock:
        return set(), set()
    runtime = {p["name"] for p in lock.get("packages", []) if isinstance(p, dict) and "name" in p}
    dev = {p["name"] for p in lock.get("packages-dev", []) if isinstance(p, dict) and "name" in p}
    return runtime, dev


def _npm_lock_names(root: Path) -> set[str]:
    """All package names present in package-lock.json, any dependency kind
    (lockfileVersion>=2 uses a flat `packages` map keyed by node_modules
    path; v1 uses a nested `dependencies` map)."""
    lock = _read_json_safe(root / "package-lock.json")
    if not lock:
        return set()
    names: set[str] = set()
    packages = lock.get("packages")
    if isinstance(packages, dict):
        for key in packages:
            if not key:
                continue
            marker = "node_modules/"
            idx = key.rfind(marker)
            if idx != -1:
                names.add(key[idx + len(marker) :])
    deps = lock.get("dependencies")
    if isinstance(deps, dict):
        names.update(deps.keys())
    return names


def _pnpm_lock_has_name(root: Path, name: str) -> bool:
    """pnpm-lock.yaml lists resolved packages under top-level `packages:`
    keyed by `/name@version` (or `name@version` in newer lockfile versions).
    A full pnpm-lock semantic model is out of scope here (WS4 is not meant to
    re-implement pnpm's resolver) — this is a presence check only: does the
    lockfile contain this exact package name as a resolved entry."""
    data = _read_yaml_safe(root / "pnpm-lock.yaml")
    if not isinstance(data, dict):
        return False
    packages = data.get("packages")
    if isinstance(packages, dict):
        for key in packages:
            if not isinstance(key, str):
                continue
            stripped = key.lstrip("/")
            if stripped == name or stripped.startswith(name + "@"):
                return True
    importers = data.get("importers")
    if isinstance(importers, dict):
        for importer in importers.values():
            if not isinstance(importer, dict):
                continue
            for section in ("dependencies", "devDependencies"):
                block = importer.get(section)
                if isinstance(block, dict) and name in block:
                    return True
    return False


def _manifest_file_for(root: Path) -> str | None:
    if (root / "composer.json").is_file():
        return "composer.json"
    if (root / "package.json").is_file():
        return "package.json"
    return None


def extract_depends_on_edges(
    repo_id: str,
    root: Path,
    package_identity_map: dict[str, str],
) -> list[OverlayEdge]:
    """Parse this repo's manifest(s) + matching lockfile(s), and emit a
    `depends_on` edge for every dependency that resolves (by package name) to
    a *different* registered repo. A manifest entry with no corresponding
    lockfile entry is not actually resolved/installed — skipped, not guessed."""
    edges: list[OverlayEdge] = []

    composer = _read_json_safe(root / "composer.json")
    if composer:
        runtime_locked, dev_locked = _composer_lock_names(root)
        for kind, section, confidence, locked_names in (
            (DEPENDENCY_KIND_RUNTIME, "require", CONFIDENCE_RUNTIME, runtime_locked),
            (DEPENDENCY_KIND_DEV, "require-dev", CONFIDENCE_DEV, dev_locked),
        ):
            for dep_name in composer.get(section, {}) or {}:
                target_repo = package_identity_map.get(dep_name)
                if target_repo is None or target_repo == repo_id:
                    continue
                if dep_name not in locked_names:
                    log.info(
                        "%s: composer dependency %r declared but not found in composer.lock; "
                        "skipping (no guess)",
                        repo_id,
                        dep_name,
                    )
                    continue
                edges.append(
                    OverlayEdge(
                        type="depends_on",
                        source=LogicalRef(
                            repo=repo_id, source_file="composer.json", qualified_label=dep_name
                        ),
                        target=LogicalRef(
                            repo=target_repo, source_file="composer.json", qualified_label=dep_name
                        ),
                        provenance=PROVENANCE_MANIFEST_LOCK,
                        confidence=confidence,
                        evidence=(
                            f"composer.json[{section}] {dep_name}, "
                            f"confirmed in composer.lock ({kind})"
                        ),
                    )
                )

    package_json = _read_json_safe(root / "package.json")
    if package_json:
        npm_lock_names = _npm_lock_names(root)
        has_pnpm_lock = (root / "pnpm-lock.yaml").is_file()
        for kind, section, confidence in (
            (DEPENDENCY_KIND_RUNTIME, "dependencies", CONFIDENCE_RUNTIME),
            (DEPENDENCY_KIND_DEV, "devDependencies", CONFIDENCE_DEV),
        ):
            for dep_name in package_json.get(section, {}) or {}:
                target_repo = package_identity_map.get(dep_name)
                if target_repo is None or target_repo == repo_id:
                    continue
                locked = dep_name in npm_lock_names or (
                    has_pnpm_lock and _pnpm_lock_has_name(root, dep_name)
                )
                if not locked:
                    log.info(
                        "%s: npm dependency %r declared but not found in any lockfile; "
                        "skipping (no guess)",
                        repo_id,
                        dep_name,
                    )
                    continue
                edges.append(
                    OverlayEdge(
                        type="depends_on",
                        source=LogicalRef(
                            repo=repo_id, source_file="package.json", qualified_label=dep_name
                        ),
                        target=LogicalRef(
                            repo=target_repo, source_file="package.json", qualified_label=dep_name
                        ),
                        provenance=PROVENANCE_MANIFEST_LOCK,
                        confidence=confidence,
                        evidence=(
                            f"package.json[{section}] {dep_name}, confirmed in lockfile ({kind})"
                        ),
                    )
                )

    return edges


def load_manual_relations(path: Path, schema: dict) -> list[dict]:
    """Load + jsonschema-validate `manual-relations.json`. Returns the raw
    `relations` list (empty if the file does not exist — manual relations are
    optional).

    Raises ValueError on malformed JSON or a schema violation. The raised
    message names the file and the failing JSON pointer only: jsonschema's own
    message embeds the offending instance, which reaches logs and `status.json`
    through the pipeline, so the full detail stays at warning level in the
    process log instead.
    """
    import jsonschema

    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log.warning("manual relations %s is not valid JSON: %s", path, exc)
        raise ValueError(
            f"manual relations file {path} is not valid JSON "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from exc
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as exc:
        log.warning("manual relations %s failed schema validation: %s", path, exc)
        pointer = getattr(exc, "json_path", None) or "/".join(
            str(part) for part in exc.absolute_path
        )
        raise ValueError(
            f"manual relations file {path} failed schema validation at {pointer or '$'}"
        ) from exc
    return data.get("relations", [])


def build_manual_relation_edges(
    raw_relations: list[dict], graphs_by_repo: dict[str, dict]
) -> list[OverlayEdge]:
    """Every manual relation's source/target logical ref must resolve
    against the current generation's per-repo graphs — a dangling ref is a
    hard error (raises DanglingReferenceError), not a warning, per plan WS4."""
    edges: list[OverlayEdge] = []
    index_cache: dict[str, dict] = {}
    for relation in raw_relations:
        source_ref = LogicalRef.from_dict(relation["source"])
        target_ref = LogicalRef.from_dict(relation["target"])
        require_resolved(
            source_ref,
            graphs_by_repo,
            context=f"manual relation source ({relation.get('type')})",
            index_cache=index_cache,
        )
        require_resolved(
            target_ref,
            graphs_by_repo,
            context=f"manual relation target ({relation.get('type')})",
            index_cache=index_cache,
        )
        edges.append(
            OverlayEdge(
                type=relation["type"],
                source=source_ref,
                target=target_ref,
                provenance=PROVENANCE_MANUAL,
                confidence=float(relation.get("confidence", CONFIDENCE_MANUAL)),
                evidence=relation.get("evidence", "manually declared relation"),
            )
        )
    return edges
