"""`provides_api` / `consumes_api` overlay edges.

Extractor survey across a representative multi-repo set:
  - Symfony `#[Route(...)]` PHP attributes are used across the composer-based
    backends (e.g. a controller such as
    src/Controller/Rest/NotificationsController.php —
    `#[Route('/api/v1/notifications/users/{user}/events', name: "...",
    requirements: [...], methods: ['GET'])]`, spanning multiple lines).
  - If no swagger/openapi spec file (`*.yaml`/`*.json` named swagger/openapi)
    is checked out at any registered repo root, a dedicated OpenAPI/swagger
    extractor has no input to exercise, so only the Symfony attribute
    extractor is implemented here. Nelmio `#[OA\\...]` attributes may exist
    alongside `#[Route]` in the same controllers but describe the same routes
    `#[Route]` already captures — not a distinct source of provider identity.

Canonical API identity = `service :: METHOD :: normalized-path-template`
(plan WS4 item 3). Provenance for every extracted-from-code edge is
`EXTRACTED_CONFIG`. Per plan: "only emit a consumes edge where the URL is
actually resolvable to a provides edge — no guessing/fuzzy matching across
repos" — implemented here as an exact string match on the normalized path
template between what a controller declares (provider) and what a literal
string in another repo's source references alongside an HTTP-call-shaped
expression (consumer). An unresolvable consumer produces zero edges, never a
best-effort guess. Because an edge always needs two parties, a provided
route with no matching consumer this run does not itself appear alone —
`provides_api` and `consumes_api` are emitted together, exactly once each,
only for a matched (provider, consumer) pair.

Performance note: provider and consumer extraction share a single
`os.walk` pass per repo and a single read per source file. The walk applies
graphify's own skip rules, the same ones `state.compute_source_manifests`
applies (pruned dir names, `.graphifyignore`/`.gitignore`, the persisted
`--exclude` list, likely-secret paths), so every file read here is also in
the repo's source digest. `collect_api_facts` walks each repo once per run
and can serve an unchanged repo from the stage cache (overlay_api_cache.py),
keyed on that digest, without walking it at all.
"""

from __future__ import annotations

import os
import re
import stat as stat_mod
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Protocol

from graphify_mesh.sync.config import CURRENT_CLASSIFICATION, categorize_file, is_sensitive_path
from graphify_mesh.sync.overlay_refs import LogicalRef, OverlayEdge
from graphify_mesh.sync.state import GraphifyIgnoreRules, IgnoreRules, content_marker

PROVENANCE_EXTRACTED_CONFIG = "EXTRACTED_CONFIG"
CONFIDENCE_API_MATCH = 0.85

_ROUTE_ATTR_RE = re.compile(r"#\[Route\((.*?)\)\]", re.DOTALL)
_PATH_LITERAL_RE = re.compile(r"""^\s*['"](/[^'"]*)['"]""", re.MULTILINE)
_METHODS_RE = re.compile(r"methods\s*:\s*\[([^\]]*)\]")
_METHOD_NAME_RE = re.compile(r"\bfunction\s+(\w+)\s*\(")
_CLASS_NAME_RE = re.compile(r"\bclass\s+(\w+)")
_METHOD_TOKEN_RE = re.compile(r"'([A-Za-z]+)'")

# Consumer-side: an HTTP-client-shaped call followed (within the same
# expression) by a string literal path. Kept intentionally narrow — this is
# not a general-purpose HTTP-call detector, only enough to anchor a literal
# path so it can be checked for an exact match against a known provider.
_CONSUMER_CALL_RE = re.compile(
    r"(?:->request\(\s*['\"](?P<method>[A-Za-z]+)['\"]\s*,\s*['\"](?P<path1>/[^'\"]*)['\"]"
    r"|->(?:get|post|put|patch|delete)\(\s*['\"](?P<path2>/[^'\"]*)['\"]"
    r"|(?:axios|fetch)\s*\(\s*['\"](?P<path3>/[^'\"]*)['\"])",
    re.IGNORECASE,
)

_PROVIDER_SUFFIX = ".php"
# Suffix order is load-bearing: consumer candidates are emitted grouped by
# suffix in this order (each group path-sorted), matching the historical
# per-suffix scan order so downstream edge ordering stays stable.
_SOURCE_SUFFIXES = (".php", ".ts", ".tsx", ".js", ".jsx")

_ConsumerCandidate = tuple[str, str, str, str]

# Bump when the walk's file selection or either extractor changes what it
# emits: it is part of the stage-cache key (overlay_api_cache.py).
API_FACTS_RECIPE = "api-facts-v3"


@dataclass(frozen=True)
class ApiProvider:
    repo: str
    http_method: str
    path_template: str
    source_file: str
    qualified_label: str


ApiFacts = tuple[list[ApiProvider], list[_ConsumerCandidate]]


@dataclass
class ApiScan:
    """What a cache needs to judge one scan's facts.

    `complete` is False after any walk, stat or read failure. `markers` maps the
    root-relative POSIX path of every file read to `state.content_marker` of
    the bytes the facts came from."""

    complete: bool = True
    markers: dict[str, str] = field(default_factory=dict)


def scanned_suffix(name: str) -> str | None:
    """The suffix a file is scanned under, or None. `PurePosixPath.suffix`, as
    the digest classifies: '.js' alone is a dotfile with no suffix."""
    suffix = PurePosixPath(name).suffix
    return suffix if suffix in _SOURCE_SUFFIXES else None


def _decode_text(data: bytes) -> str:
    """What `read_text(encoding="utf-8", errors="ignore")` returns for `data`:
    text mode also turns \\r\\n and a lone \\r into \\n."""
    return data.decode("utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")


class ApiFactsSource(Protocol):
    """Stage cache handed to `collect_api_facts`: `lookup` is None on a miss
    or for a repo it cannot key, and `store` keeps facts it can key and trust."""

    def lookup(self, repo_id: str, root: Path) -> ApiFacts | None: ...

    def store(self, repo_id: str, root: Path, facts: ApiFacts, scan: ApiScan) -> None: ...


def normalize_path_template(path: str) -> str:
    """Canonical form for matching: strip query string, force a single
    leading slash, drop a trailing slash (except root), replace every
    `{paramName}` placeholder with the generic `{param}` so provider and
    consumer only need to agree on shape, not parameter names."""
    path = path.split("?", 1)[0].strip()
    if not path.startswith("/"):
        path = "/" + path
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return re.sub(r"\{[^}/]+\}", "{param}", path)


def _walk_source_files(
    root: Path, ignore_rules: IgnoreRules | None = None
) -> tuple[dict[str, list[Path]], ApiScan]:
    """One os.walk pass collecting every suffix needed by provider and
    consumer extraction. Mirrors the file selection of
    `state.compute_source_manifests` under CURRENT_CLASSIFICATION: pruned dir
    names, `Path.suffix` classified as code, graphify's ignore rules
    (`ignore_rules` holds the repo's persisted build options, None means
    graphify's defaults) and likely-secret paths. Each file read is therefore
    in the source digest the stage-cache key is built from."""
    by_suffix: dict[str, list[Path]] = {suffix: [] for suffix in _SOURCE_SUFFIXES}
    scan = ApiScan()
    if not root.is_dir():
        return by_suffix, scan
    table = CURRENT_CLASSIFICATION
    ignore = GraphifyIgnoreRules(root, ignore_rules or IgnoreRules())
    resolved_root = root.resolve()

    def walk_failed(_exc: OSError) -> None:
        scan.complete = False

    for dirpath, dirnames, filenames in os.walk(root, onerror=walk_failed):
        base = Path(dirpath)
        dirnames[:] = [
            d for d in dirnames if not table.prunes(d) and not ignore.ignores_dir(base / d)
        ]
        for filename in filenames:
            suffix = scanned_suffix(filename)
            if suffix is None:
                continue
            path = base / filename
            # Same full-parts filter as the digest: a root under a pruned
            # dir counts nothing there, so nothing is read here.
            if (
                any(table.prunes(part) for part in path.parts)
                or categorize_file(path, table) != "code"
                or ignore.ignores_file(path)
                or is_sensitive_path(resolved_root / path.relative_to(root))
            ):
                continue
            try:
                st = path.stat()
            except FileNotFoundError:
                # A dangling link (the digest skips it too), or a file gone
                # mid-walk (the cache then sees other files than the digest).
                continue
            except OSError:
                scan.complete = False
                continue
            if not stat_mod.S_ISREG(st.st_mode):
                continue
            by_suffix[suffix].append(path)
    for paths in by_suffix.values():
        paths.sort()
    return by_suffix, scan


def _providers_from_text(repo_id: str, rel: str, text: str) -> list[ApiProvider]:
    providers: list[ApiProvider] = []
    # Class-declaration positions/names precomputed once per file; each
    # route then finds its enclosing class via bisect instead of re-scanning
    # the whole file prefix per route match.
    class_ends: list[int] = []
    class_names: list[str] = []
    for class_match in _CLASS_NAME_RE.finditer(text):
        class_ends.append(class_match.end())
        class_names.append(class_match.group(1))

    for match in _ROUTE_ATTR_RE.finditer(text):
        body = match.group(1)
        path_match = _PATH_LITERAL_RE.search(body)
        if not path_match:
            continue
        raw_path = path_match.group(1)
        methods_match = _METHODS_RE.search(body)
        http_methods = (
            _METHOD_TOKEN_RE.findall(methods_match.group(1)) if methods_match else ["GET"]
        )

        # Enclosing method name (qualified_label = Class::method), best
        # effort: nearest preceding `class` + nearest following `function`.
        class_idx = bisect_right(class_ends, match.start()) - 1
        class_name = "UnknownClass"
        if class_idx >= 0:
            class_name = class_names[class_idx]
        method_match = _METHOD_NAME_RE.search(text, match.end())
        method_name = method_match.group(1) if method_match else "unknownMethod"
        qualified_label = f"{class_name}::{method_name}"

        template = normalize_path_template(raw_path)
        for http_method in http_methods or ["GET"]:
            providers.append(
                ApiProvider(
                    repo=repo_id,
                    http_method=http_method.upper(),
                    path_template=template,
                    source_file=rel,
                    qualified_label=qualified_label,
                )
            )
    return providers


def _consumers_from_text(repo_id: str, rel: str, text: str) -> list[_ConsumerCandidate]:
    candidates: list[_ConsumerCandidate] = []
    for match in _CONSUMER_CALL_RE.finditer(text):
        raw_path = match.group("path1") or match.group("path2") or match.group("path3")
        if not raw_path:
            continue
        method = (match.group("method") or "GET").upper()
        candidates.append((repo_id, method, normalize_path_template(raw_path), rel))
    return candidates


def _scan_repo_api_facts(
    repo_id: str, root: Path, ignore_rules: IgnoreRules | None = None
) -> tuple[ApiFacts, ApiScan]:
    """Walk the repo once, read each source file at most once, and run both
    provider and consumer extraction over the shared text."""
    by_suffix, scan = _walk_source_files(root, ignore_rules)
    providers: list[ApiProvider] = []
    candidates: list[_ConsumerCandidate] = []
    for suffix in _SOURCE_SUFFIXES:
        for path in by_suffix[suffix]:
            # One read: the facts and the marker come from the same bytes.
            try:
                data = path.read_bytes()
            except OSError:
                scan.complete = False
                continue
            scan.markers[path.relative_to(root).as_posix()] = content_marker(data)
            text = _decode_text(data)
            rel = str(path.relative_to(root))
            if suffix == _PROVIDER_SUFFIX:
                providers.extend(_providers_from_text(repo_id, rel, text))
            candidates.extend(_consumers_from_text(repo_id, rel, text))
    return (providers, candidates), scan


def extract_repo_api_facts(
    repo_id: str, root: Path, ignore_rules: IgnoreRules | None = None
) -> ApiFacts:
    """Providers and consumer candidates of one repo, from one walk."""
    facts, _ = _scan_repo_api_facts(repo_id, root, ignore_rules)
    return facts


def collect_api_facts(
    repo_roots_by_id: dict[str, Path],
    ignore_rules_by_repo: dict[str, IgnoreRules] | None = None,
    api_cache: ApiFactsSource | None = None,
) -> tuple[dict[str, list[ApiProvider]], dict[str, list[_ConsumerCandidate]]]:
    """Providers and consumer candidates per repo, one walk per repo at most.

    Both dicts keep `repo_roots_by_id` order: the provider index keeps the
    first provider of a route, so the order decides which one an edge names."""
    rules = ignore_rules_by_repo or {}
    providers_by_repo: dict[str, list[ApiProvider]] = {}
    candidates_by_repo: dict[str, list[_ConsumerCandidate]] = {}
    for repo_id, root in repo_roots_by_id.items():
        facts = api_cache.lookup(repo_id, root) if api_cache is not None else None
        if facts is None:
            facts, scan = _scan_repo_api_facts(repo_id, root, rules.get(repo_id))
            if api_cache is not None:
                api_cache.store(repo_id, root, facts, scan)
        providers_by_repo[repo_id], candidates_by_repo[repo_id] = facts
    return providers_by_repo, candidates_by_repo


def extract_symfony_route_providers(
    repo_id: str, root: Path, ignore_rules: IgnoreRules | None = None
) -> list[ApiProvider]:
    """Grep-based Symfony `#[Route(...)]` attribute extractor. AST tooling
    is not available at this layer (no embeddings/AST pass over PHP in the
    sync pipeline) — grep is explicitly acceptable per plan WS4 item 3."""
    providers, _ = extract_repo_api_facts(repo_id, root, ignore_rules)
    return providers


def extract_consumer_literal_paths(
    repo_id: str, root: Path, ignore_rules: IgnoreRules | None = None
) -> list[_ConsumerCandidate]:
    """[(repo, http_method_or_GET, normalized_path_template, source_file)]
    for every literal path string found alongside an HTTP-call-shaped
    expression, across both PHP and JS/TS sources (a consumer calling
    another registered repo's API can live in either ecosystem)."""
    _, candidates = extract_repo_api_facts(repo_id, root, ignore_rules)
    return candidates


def match_provides_consumes_edges(
    providers_by_repo: dict[str, list[ApiProvider]],
    consumer_candidates_by_repo: dict[str, list[_ConsumerCandidate]],
) -> list[OverlayEdge]:
    """Emit a matched (provides_api, consumes_api) pair only where a
    consumer's exact (method, normalized path template) resolves to a
    *different* repo's declared provider. No match => zero edges for that
    candidate — never a fuzzy/best-effort guess."""
    provider_index: dict[tuple[str, str], ApiProvider] = {}
    for providers in providers_by_repo.values():
        for provider in providers:
            provider_index.setdefault((provider.http_method, provider.path_template), provider)

    edges: list[OverlayEdge] = []
    emitted_pairs: set[tuple] = set()
    for consumer_repo, candidates in consumer_candidates_by_repo.items():
        for method, template, source_file in ((c[1], c[2], c[3]) for c in candidates):
            matched = provider_index.get((method, template))
            if matched is None or matched.repo == consumer_repo:
                continue
            pair_key = (
                consumer_repo,
                source_file,
                matched.repo,
                matched.qualified_label,
                method,
                template,
            )
            if pair_key in emitted_pairs:
                continue
            emitted_pairs.add(pair_key)

            provider_ref = LogicalRef(
                repo=matched.repo,
                source_file=matched.source_file,
                qualified_label=matched.qualified_label,
            )
            consumer_ref = LogicalRef(
                repo=consumer_repo, source_file=source_file, qualified_label=f"{method} {template}"
            )
            evidence = f"{method} {template} matched: provider {matched.repo}:{matched.source_file}"
            edges.append(
                OverlayEdge(
                    type="provides_api",
                    source=provider_ref,
                    target=consumer_ref,
                    provenance=PROVENANCE_EXTRACTED_CONFIG,
                    confidence=CONFIDENCE_API_MATCH,
                    evidence=evidence,
                )
            )
            edges.append(
                OverlayEdge(
                    type="consumes_api",
                    source=consumer_ref,
                    target=provider_ref,
                    provenance=PROVENANCE_EXTRACTED_CONFIG,
                    confidence=CONFIDENCE_API_MATCH,
                    evidence=evidence,
                )
            )
    return edges
