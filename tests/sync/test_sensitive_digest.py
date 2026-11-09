"""The v3 digest skips files graphify drops as sensitive (detect._is_sensitive).

graphify never admits notes/secrets.md, api_token.txt, ... into the corpus. A
digest that counted them changed on every such new file, held one extract, and
retried it (sync_project._unstamped_documents), all for nothing.
"""

from __future__ import annotations

from pathlib import Path

from graphify_mesh.sync.config import (
    CURRENT_CLASSIFICATION,
    LEGACY_CLASSIFICATION_V1,
    LEGACY_CLASSIFICATION_V2,
    is_sensitive_path,
)
from graphify_mesh.sync.state import compute_source_manifest, compute_source_manifests


def _write(root: Path, rel: str, text: str = "x\n") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    _write(root, "src/main.py", "print(1)\n")
    _write(root, "notes/guide.md", "# guide\n")
    return root


def test_new_sensitive_files_leave_the_current_digest_unchanged(tmp_path):
    root = _repo(tmp_path)
    before = compute_source_manifest(root)

    for rel in (
        "notes/secrets.md",
        "passwords.md",
        "api_token.txt",
        "service-account.json",
        "secrets/db.yaml",
        ".env",
    ):
        _write(root, rel)

    after = compute_source_manifest(root)
    assert after == before


def test_genuine_source_in_a_secrets_dir_still_counts(tmp_path):
    root = _repo(tmp_path)
    before = compute_source_manifest(root)

    _write(root, "secrets/handler.py", "def handle():\n    pass\n")

    assert compute_source_manifest(root).code_hash != before.code_hash


def test_topic_notes_about_tokens_still_count(tmp_path):
    root = _repo(tmp_path)
    _write(root, "notes/token-economics-of-recall.md", "# tokens\n")

    digest = compute_source_manifest(root)

    assert "notes/token-economics-of-recall.md" in digest.semantic_paths


def test_legacy_tables_keep_counting_sensitive_files(tmp_path):
    # v1/v2 reproduce digests stored before this rule; they must not change.
    root = _repo(tmp_path)
    _write(root, "notes/secrets.md")

    current, v2, v1 = compute_source_manifests(
        root, (CURRENT_CLASSIFICATION, LEGACY_CLASSIFICATION_V2, LEGACY_CLASSIFICATION_V1)
    )

    assert "notes/secrets.md" not in current.semantic_paths
    assert "notes/secrets.md" in v2.semantic_paths
    assert "notes/secrets.md" in v1.semantic_paths
    assert CURRENT_CLASSIFICATION.drops_sensitive_paths
    assert not LEGACY_CLASSIFICATION_V2.drops_sensitive_paths
    assert not LEGACY_CLASSIFICATION_V1.drops_sensitive_paths


def test_sensitive_parent_dir_above_the_root_counts_like_upstream(tmp_path):
    # Upstream checks the absolute path, so a repo living under a
    # credentials/ dir drops its non-source files.
    root = tmp_path / "credentials" / "repo"
    _write(root, "src/main.py", "print(1)\n")
    _write(root, "notes/guide.md", "# guide\n")

    digest = compute_source_manifest(root)

    assert digest.semantic_paths == ()
    assert digest.file_count == 1
    assert is_sensitive_path(root.resolve() / "notes" / "guide.md")
