"""Pins the source of the code that turns a repo tree into cached API facts and
their cache key. With the stage cache on, a repo whose source digest did not
change is served from disk, so a change here is invisible until
API_FACTS_RECIPE is bumped."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from graphify_mesh.sync import overlay_api, overlay_api_cache, state

GUARDED = (
    # File selection, both extractors and the path normalization they share.
    overlay_api._walk_source_files,
    overlay_api.extract_repo_api_facts,
    overlay_api._scan_repo_api_facts,
    overlay_api.scanned_suffix,
    overlay_api._decode_text,
    overlay_api._providers_from_text,
    overlay_api._consumers_from_text,
    overlay_api.normalize_path_template,
    overlay_api.ApiProvider,
    # The key and the payload round trip.
    overlay_api_cache.facts_key,
    overlay_api_cache._encode,
    overlay_api_cache._decode,
    # The marker a stored scan is checked with against the digest.
    state.content_marker,
    overlay_api_cache.ApiSourceId.from_digest,
)
GUARDED_VALUES = (
    overlay_api.API_FACTS_RECIPE,
    overlay_api._SOURCE_SUFFIXES,
    overlay_api._PROVIDER_SUFFIX,
    overlay_api._ROUTE_ATTR_RE.pattern,
    int(overlay_api._ROUTE_ATTR_RE.flags),
    overlay_api._PATH_LITERAL_RE.pattern,
    int(overlay_api._PATH_LITERAL_RE.flags),
    overlay_api._METHODS_RE.pattern,
    int(overlay_api._METHODS_RE.flags),
    overlay_api._METHOD_NAME_RE.pattern,
    int(overlay_api._METHOD_NAME_RE.flags),
    overlay_api._CLASS_NAME_RE.pattern,
    int(overlay_api._CLASS_NAME_RE.flags),
    overlay_api._METHOD_TOKEN_RE.pattern,
    int(overlay_api._METHOD_TOKEN_RE.flags),
    overlay_api._CONSUMER_CALL_RE.pattern,
    int(overlay_api._CONSUMER_CALL_RE.flags),
)
PINNED_SOURCE_SHA256 = "4ba469c5eb6884d1e8684a57872301e4e22872205fb14999e8efc622101b9b50"


def _source_sha256() -> str:
    h = hashlib.sha256()
    for obj in GUARDED:
        h.update(inspect.getsource(obj).encode("utf-8"))
        h.update(b"\0")
    for value in GUARDED_VALUES:
        h.update(repr(value).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def test_api_facts_source_is_pinned():
    actual = _source_sha256()
    assert actual == PINNED_SOURCE_SHA256, (
        "code that shapes cached overlay API facts or their key changed. "
        "A comment- or docstring-only change: update PINNED_SOURCE_SHA256 only. "
        "A semantic change: bump API_FACTS_RECIPE in overlay_api.py "
        "(invalidates every cached entry) and update PINNED_SOURCE_SHA256. "
        f"New value: {actual!r}"
    )


def test_key_moves_with_recipe(monkeypatch):
    before = overlay_api_cache.facts_key("r", Path("/src"), "attempt", "ignore")
    monkeypatch.setattr(overlay_api_cache, "API_FACTS_RECIPE", "api-facts-test")
    assert overlay_api_cache.facts_key("r", Path("/src"), "attempt", "ignore") != before
