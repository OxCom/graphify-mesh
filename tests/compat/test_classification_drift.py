"""Drift check: the hand-copied classification table against installed graphify.

sync/config.py copies upstream's file-type sets by hand and never imports them
at runtime (decision 2026-11-06, yaml-json-classification). This lane is what
notices an upstream change. A failure prints the differing suffixes; the fix is
a table edit plus a CLASSIFICATION_VERSION bump, never a test edit alone.

Sets are identical in graphify 0.9.95 (the declared floor) and 0.9.103, so no
per-version allowlist is needed. Add one here when they diverge.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graphify_mesh.sync import config

pytest.importorskip("graphify")
detect = pytest.importorskip("graphify.detect")
manifest_ingest = pytest.importorskip("graphify.manifest_ingest")

# Upstream routes these to an LLM or transcription path. Together with
# DOC_EXTENSIONS they are every suffix classify_file does not send to CODE.
_UPSTREAM_NON_CODE_NAMES = (
    "DOC_EXTENSIONS",
    "PAPER_EXTENSIONS",
    "IMAGE_EXTENSIONS",
    "OFFICE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "GOOGLE_WORKSPACE_EXTENSIONS",
)


def _lower(suffixes) -> set[str]:
    return {s.lower() for s in suffixes}


def _upstream_non_code() -> set[str]:
    found: set[str] = set()
    for name in _UPSTREAM_NON_CODE_NAMES:
        found |= _lower(getattr(detect, name))
    return found


def _show(label: str, items) -> str:
    return f"{label}: {sorted(items)}"


def test_every_upstream_code_suffix_is_mesh_code():
    missing = _lower(detect.CODE_EXTENSIONS) - config.CODE_EXTENSIONS
    assert not missing, _show("upstream code suffixes missing from mesh", missing)


def test_mesh_code_suffixes_are_all_upstream_code():
    extra = config.CODE_EXTENSIONS - _lower(detect.CODE_EXTENSIONS)
    assert not extra, _show("mesh code suffixes upstream does not treat as code", extra)


def test_mesh_semantic_suffixes_are_upstream_documents():
    upstream_docs = _lower(detect.DOC_EXTENSIONS)
    extra = config.SEMANTIC_EXTENSIONS - upstream_docs
    assert not extra, _show("mesh semantic suffixes not upstream documents", extra)


def test_semantic_and_code_suffixes_do_not_overlap():
    both = config.SEMANTIC_EXTENSIONS & config.CODE_EXTENSIONS
    assert not both, _show("suffixes in both mesh tables", both)


def test_deliberately_excluded_is_exactly_the_upstream_remainder():
    expected = _upstream_non_code() - config.SEMANTIC_EXTENSIONS
    actual = set(config.DELIBERATELY_EXCLUDED_DOC)
    assert actual == expected, (
        _show("upstream non-code suffixes not listed as excluded", expected - actual)
        + "\n"
        + _show("excluded suffixes upstream no longer routes", actual - expected)
    )


def test_ignored_filenames_equal_upstream_skip_files():
    """_SKIP_FILES holds only file names (lockfiles, .graphifyinclude), so the
    comparison needs no filtering."""
    upstream = set(detect._SKIP_FILES)
    ours = set(config.IGNORED_FILENAMES)
    assert ours == upstream, (
        _show("upstream skip files missing from mesh", upstream - ours)
        + "\n"
        + _show("mesh ignored names upstream does not skip", ours - upstream)
    )


def _is_glob(name: str) -> bool:
    return any(ch in name for ch in "*?[")


def test_v2_prunes_every_fixed_name_upstream_skip_dir():
    """Files under a dir graphify never reads must not move the source digest.
    Upstream's evidence-gated names (env, out, coverage) are not in _SKIP_DIRS."""
    upstream = {name for name in detect._SKIP_DIRS if not _is_glob(name)}
    missing = upstream - config.CURRENT_CLASSIFICATION.pruned_dir_names
    assert not missing, _show("upstream skip dirs v2 still hashes", missing)


def test_every_upstream_skip_dir_glob_is_a_mesh_suffix():
    """The only glob is "*.egg-info"; a new glob shape needs a matcher first."""
    globs = {name for name in detect._SKIP_DIRS if _is_glob(name)}
    unmatched = {
        g
        for g in globs
        if not (g.startswith("*") and not _is_glob(g[1:]))
        or g[1:] not in config.CURRENT_CLASSIFICATION.pruned_dir_suffixes
    }
    assert not unmatched, _show("upstream skip-dir globs the mesh does not match", unmatched)


def test_code_filenames_equal_upstream_package_manifests():
    upstream = {name.lower() for name in manifest_ingest.PACKAGE_MANIFEST_NAMES}
    ours = set(config.CODE_FILENAMES)
    assert ours == upstream, (
        _show("upstream manifests missing from mesh", upstream - ours)
        + "\n"
        + _show("mesh code filenames upstream does not parse", ours - upstream)
    )


# Every mesh table decision, plus names whose routing moved or is subtle.
PARITY_NAMES = [
    "m.py",
    "a.php",
    "page.blade.php",
    "b.ts",
    "d.js",
    "x.mjs",
    "run.sh",
    "schema.sql",
    "main.tf",
    "compose.yaml",
    "values.yml",
    "data.json",
    "README.md",
    "page.mdx",
    "guide.rst",
    "notes.txt",
    "index.html",
    "tpl.j2",
    "paper.pdf",
    "logo.png",
    "settings.toml",
    "setup.cfg",
    "app.ini",
    ".env.example",
    "pyproject.toml",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "apm.yml",
    "package-lock.json",
    "pnpm-lock.yaml",
    ".graphifyinclude",
    "Makefile",
]

_UPSTREAM_TO_MESH = {
    "code": "code",
    "document": "semantic",
    "paper": "semantic",
    "image": "semantic",
}


def _upstream_category(path: Path) -> str:
    if path.name in detect._SKIP_FILES:
        return "ignore"
    kind = detect.classify_file(path)
    if kind is None:
        return "ignore"
    return _UPSTREAM_TO_MESH.get(kind.value, "ignore")


@pytest.mark.parametrize("name", PARITY_NAMES)
def test_classify_file_parity(tmp_path, name):
    """Where the mesh deliberately differs from upstream (the excluded suffixes)
    the mesh category is 'ignore'; everywhere else it equals upstream's."""
    path = tmp_path / name
    path.write_text("x\n", encoding="utf-8")
    mesh = config.categorize_file(path)
    upstream = _upstream_category(path)
    if path.suffix.lower() in config.DELIBERATELY_EXCLUDED_DOC:
        assert mesh == "ignore", name
    else:
        assert mesh == upstream, f"{name}: mesh={mesh} upstream={upstream}"
