from __future__ import annotations

import importlib
import sys

import pytest

from graphify_mesh.sync import backend as backend_mod
from graphify_mesh.sync.config import PINNED_CLUSTERING_BACKEND


def test_graspologic_absent_here_means_louvain():
    """Neither graspologic spelling is installed in this environment, so the
    check must answer Louvain — the same backend config pins."""
    assert backend_mod._graspologic_importable() is False
    assert PINNED_CLUSTERING_BACKEND == backend_mod.LOUVAIN_BACKEND

    result = backend_mod.assert_pinned_backend()

    assert result.backend == backend_mod.LOUVAIN_BACKEND
    assert result.matches_pinned


def test_native_only_leiden_install_is_detected(tmp_path, monkeypatch):
    """Python >= 3.13's `leiden` extra installs graspologic-native only.

    A stub package on sys.path rather than a patched `find_spec`: patching it
    process-wide breaks any import that happens inside the test body.
    """
    (tmp_path / "graspologic_native").mkdir()
    (tmp_path / "graspologic_native" / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    assert backend_mod._graspologic_importable() is True


def test_pure_python_graspologic_install_is_detected(tmp_path, monkeypatch):
    """The other spelling: upstream falls back to `graspologic.partition.leiden`
    when the native package is missing, so it counts as Leiden too."""
    (tmp_path / "graspologic").mkdir()
    (tmp_path / "graspologic" / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    assert backend_mod._graspologic_importable() is True


def test_assert_pinned_backend_answers_for_this_interpreter(monkeypatch):
    monkeypatch.setattr(backend_mod, "_graspologic_importable", lambda: False)
    monkeypatch.setattr(backend_mod, "PINNED_CLUSTERING_BACKEND", backend_mod.LOUVAIN_BACKEND)
    result = backend_mod.assert_pinned_backend()
    assert result.backend == backend_mod.LOUVAIN_BACKEND
    assert result.matches_pinned is True
    assert result.interpreter == sys.executable


def test_assert_pinned_backend_raises_when_actual_backend_differs(monkeypatch):
    monkeypatch.setattr(backend_mod, "_graspologic_importable", lambda: True)
    monkeypatch.setattr(backend_mod, "PINNED_CLUSTERING_BACKEND", backend_mod.LOUVAIN_BACKEND)
    with pytest.raises(backend_mod.BackendMismatchError):
        backend_mod.assert_pinned_backend()


def test_graphify_bin_argument_is_accepted_and_ignored(monkeypatch):
    """`pipeline.py` and `naming.py` still pass `settings.graphify_bin`; the
    value no longer decides anything but must not break the call."""
    monkeypatch.setattr(backend_mod, "_graspologic_importable", lambda: False)
    monkeypatch.setattr(backend_mod, "PINNED_CLUSTERING_BACKEND", backend_mod.LOUVAIN_BACKEND)

    with_bin = backend_mod.assert_pinned_backend("/nonexistent/graphify")
    without_bin = backend_mod.assert_pinned_backend()

    assert with_bin == without_bin
