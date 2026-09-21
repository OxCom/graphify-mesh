from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

_BIN_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_BIN_DIR) not in sys.path:
    sys.path.insert(0, str(_BIN_DIR))

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
FAKE_GRAPHIFY = FIXTURES_DIR / "fake_graphify" / "graphify"
GRAPHS_DIR = FIXTURES_DIR / "graphs"


class Env:
    """A fully assembled fake scan-root + graph-mesh tree for one test."""

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        # Mirror real production topology: graph-mesh lives *inside* the
        # scan root (graph-mesh lives inside it, as a sibling of the other project
        # not beside it — this matters for the approved-root containment
        # check in discovery.discover_filesystem.
        self.scan_root = tmp_path / "www"
        self.scan_roots = [self.scan_root]
        self.mesh_root = self.scan_root / "graph-mesh"
        self.registry_path = self.mesh_root / "bin" / "registry.json"
        self.control_path = tmp_path / "fake-graphify-control.json"
        self.call_log_path = tmp_path / "fake-graphify-call-log.jsonl"
        (self.mesh_root / "bin").mkdir(parents=True, exist_ok=True)
        self.scan_root.mkdir(parents=True, exist_ok=True)
        self._repos: list[dict] = []
        self._control: dict[str, dict] = {}
        self.control_path.write_text("{}", encoding="utf-8")

    def collection_path(self, product: str, sub: str) -> Path:
        return self.mesh_root / "graphify" / product / sub

    def add_repo(
        self,
        repo_id: str,
        product: str,
        sub: str,
        root_name: str,
        graph_fixture: str | None = "repo_a.json",
        enabled: bool = True,
        make_symlink: bool = True,
        make_collection: bool = True,
    ) -> Path:
        collection_path = self.collection_path(product, sub)
        if make_collection:
            collection_path.mkdir(parents=True, exist_ok=True)
            if graph_fixture is not None:
                shutil.copy2(GRAPHS_DIR / graph_fixture, collection_path / "graph.json")

        # `root_name` may contain "/" to describe arbitrary nesting depth
        # below the scan root (e.g. "workspace/deep/assets" for a depth-3
        # project dir).
        root = self.scan_root
        for segment in root_name.split("/"):
            root = root / segment
        root.mkdir(parents=True, exist_ok=True)
        if make_symlink:
            link = root / "graphify-out"
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(collection_path, target_is_directory=True)

        self._repos.append(
            {
                "repo_id": repo_id,
                "root": str(root),
                "collection_path": str(collection_path),
                "enabled": enabled,
            }
        )
        return root

    def write_registry(
        self, disabled: list[str] | None = None, external_roots: list[str] | None = None
    ) -> None:
        payload = {
            "repos": self._repos,
            "disabled": disabled or [],
            "external_roots": external_roots or [],
        }
        self.registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def set_control(self, collection_path: Path, mode: str, **extra) -> None:
        self._control[str(collection_path)] = {"mode": mode, **extra}
        self.control_path.write_text(json.dumps(self._control), encoding="utf-8")

    def read_call_log(self) -> list[dict]:
        """Every cluster-only/label invocation the fake graphify stub
        recorded this test, in call order. See FAKE_GRAPHIFY_CALL_LOG."""
        if not self.call_log_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.call_log_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def settings(self, **overrides):
        from graphify_mesh.sync.config import Settings

        # Tests must never hit the real network: default the WS2 naming
        # stage's Ollama health check to "unhealthy" (degraded mode) unless
        # a test explicitly overrides it with its own deterministic fake.
        overrides.setdefault("ollama_health_check", lambda *a, **kw: False)
        # Same for the WS3 embed stage's own (native-endpoint) health check
        # — defaults to unhealthy/degraded so no test ever silently reaches
        # the real Ollama host; tests exercising the healthy embed path must
        # inject their own deterministic fake health_check AND embed_batch.
        overrides.setdefault("ollama_embed_health_check", lambda *a, **kw: False)
        # And for the extract-backend infra probe: an empty URL keeps the
        # feature inert (no probes, failures stay plain `failed`) even when
        # the machine running the tests has OLLAMA_BASE_URL exported. Tests
        # exercising the guard override this with an explicit URL plus a
        # deterministic extract_health_check fake.
        overrides.setdefault("extract_health_url", "")
        return Settings.from_env(
            mesh_root=self.mesh_root,
            scan_roots=self.scan_roots,
            registry_path=self.registry_path,
            graphify_bin=str(FAKE_GRAPHIFY),
            **overrides,
        )


@pytest.fixture(autouse=True)
def _reset_child_log_once():
    """Several child-env diagnostics fire once per PROCESS, so without this the
    second test to trigger one would see no log line at all."""
    from graphify_mesh.sync import graphify_cli

    graphify_cli.reset_log_once()
    yield
    graphify_cli.reset_log_once()


@pytest.fixture(autouse=True)
def _fake_graphify_env_passthrough(monkeypatch):
    """The child env allowlist in graphify_cli drops everything it does not
    recognise, including the two variables the fake graphify binary is driven
    by. Declare them through the documented extension point so the fake still
    sees its control file and writes its call log."""
    monkeypatch.setenv(
        "GRAPHIFY_MESH_CHILD_ENV_EXTRA", "FAKE_GRAPHIFY_CONTROL,FAKE_GRAPHIFY_CALL_LOG"
    )


@pytest.fixture(autouse=True)
def _no_live_labeling(monkeypatch):
    """Every naming test gets a deterministic labeler.

    A test that wants the degraded path sets its own health check to False; a
    test that wants names gets them from here, not from a backend. Without this,
    the suite's behavior depends on whether an Ollama happens to be listening.

    The fake returns `Community Name <cid>`, which deliberately does NOT match
    validate.PLACEHOLDER_RE (`^Community \\d+$`) — a name that did would send
    every test down the provisional path.
    """

    def fake_llm_names(G, communities, *, backend, model):
        return {cid: f"Community Name {cid}" for cid in communities}, "llm"

    monkeypatch.setattr("graphify_mesh.sync.clustering.llm_names", fake_llm_names, raising=True)


@pytest.fixture()
def env(tmp_path, monkeypatch) -> Env:
    e = Env(tmp_path)
    monkeypatch.setenv("FAKE_GRAPHIFY_CONTROL", str(e.control_path))
    monkeypatch.setenv("FAKE_GRAPHIFY_CALL_LOG", str(e.call_log_path))
    monkeypatch.setenv("GRAPHIFY_BIN", str(FAKE_GRAPHIFY))
    yield e
