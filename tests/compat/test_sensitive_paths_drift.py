"""Drift check: config.is_sensitive_path against upstream detect._is_sensitive.

The mesh copies graphify's secret-file rule by hand (never imported at
runtime, same policy as the classification tables). graphify drops these files
before classification, so a digest that counted them would hold an extract for
a document graphify never admits. A failure names the differing paths; the fix
is a rule edit in sync/config.py plus a CLASSIFICATION_VERSION bump.
"""

from __future__ import annotations

import pytest

from graphify_mesh.sync import config

pytest.importorskip("graphify")
detect = pytest.importorskip("graphify.detect")

SAMPLES = (
    "docs/secrets.md",
    "passwords.md",
    "api_token.txt",
    "service-account.json",
    "serviceaccount.json",
    "service_account.py",
    "secrets/db.json",
    "secrets/handler.py",
    "secrets/notes.md",
    "Secrets/notes.md",
    "credentials/config.yaml",
    "credentials/go.mod",
    ".secrets/readme.txt",
    ".ssh/config.py",
    ".aws/credentials",
    ".gnupg/pubring.kbx",
    ".env",
    ".env.local",
    ".env.example",
    ".envrc",
    ".envrc.sample",
    "server.pem",
    "certs/server.crt",
    "keys/deploy.KEY",
    "id_rsa",
    "id_ed25519.pub",
    "grid_rsa",
    "secring.gpg",
    ".npmrc",
    ".git-credentials",
    "token-economics-of-recall.md",
    "password-policy-discussion.md",
    "token.economics.notes.md",
    "TOKENS.md",
    "token.md",
    "token.txt",
    "tokens.txt",
    "tokenizer.py",
    "device_token.rb",
    "passwords_controller.rb",
    "credentials.json",
    "oauth_token.json",
    "secret_store.txt",
    "private_key.txt",
    "github-personal-access-token.md",
    "terraform.tfvars",
    "main.tf",
    "src/main.py",
    "README.md",
    "docs/guide.md",
    "config/app.yaml",
)


def test_mesh_sensitive_rule_matches_upstream(tmp_path):
    root = tmp_path / "repo"
    differ: list[str] = []
    for rel in SAMPLES:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")
        upstream = detect._is_sensitive(path)
        if config.is_sensitive_path(path) != upstream:
            differ.append(f"{rel} (upstream sensitive={upstream})")
    assert not differ, f"mesh and upstream disagree on: {differ}"


def test_samples_cover_both_outcomes(tmp_path):
    root = tmp_path / "repo"
    outcomes = {config.is_sensitive_path(root / rel) for rel in SAMPLES}
    assert outcomes == {True, False}
