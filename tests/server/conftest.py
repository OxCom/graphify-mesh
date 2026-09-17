from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

_BIN_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_BIN_DIR) not in sys.path:
    sys.path.insert(0, str(_BIN_DIR))

from graphify_mesh.server.config import ServerConfig  # noqa: E402
from graphify_mesh.server.http_app import build_http_app  # noqa: E402
from graphify_mesh.server.server import GraphifyMeshServer  # noqa: E402
from graphify_mesh.server.store import Generation  # noqa: E402
from graphify_mesh.sync.embedding import node_key  # noqa: E402
from graphify_mesh.sync.lexical_index import build_lexical_index  # noqa: E402
from graphify_mesh.sync.vectors import RepoVectors  # noqa: E402


def make_node(repo, label, source_file, node_id=None, line=1, community_name=None, **extra) -> dict:
    node = {
        "id": node_id or f"{repo}:{label}",
        "repo": repo,
        "label": label,
        "source_file": source_file,
        "line": line,
        "community_name": community_name,
    }
    node.update(extra)
    return node


def make_link(src_id: str, dst_id: str, confidence: str = "EXTRACTED") -> dict:
    return {"source": src_id, "target": dst_id, "confidence": confidence}


def key_for(repo: str, node: dict) -> str:
    """Same durable logical key `graphify_mesh.sync.embedding.node_key` produces
    — used by tests to build embeddings dicts / assert on `Hit.key`."""
    return node_key(repo, node)


def build_generation(
    nodes: list[dict],
    links: list[dict] | None = None,
    overlay_edges: list[dict] | None = None,
    embeddings: dict[str, RepoVectors] | None = None,
    generation_id: str = "gen-test-1",
    manifest_extra: dict | None = None,
) -> Generation:
    """Builds a fully-indexed, in-memory `Generation` from synthetic nodes —
    no disk I/O, no real project data. The lexical index is built with the
    REAL `graphify_mesh.sync.lexical_index.build_lexical_index` (not a hand-rolled
    stub) so postings/alias_exact/documents/fields shapes are exactly what
    production code produces (current schema: v3 int-packed postings, no
    stored `doc_freq` — the reader derives it from distinct doc ids)."""
    graph = {"nodes": nodes, "links": links or []}
    graphs_by_repo: dict[str, dict] = {}
    for node in nodes:
        graphs_by_repo.setdefault(node["repo"], {"nodes": []})["nodes"].append(node)
    lexical_result = build_lexical_index(graphs_by_repo, {})
    manifest = {"generation_id": generation_id, **(manifest_extra or {})}
    generation = Generation(
        generation_id=generation_id,
        manifest=manifest,
        graph=graph,
        overlay={"edges": overlay_edges or []},
        lexical=lexical_result.data,
        embeddings=embeddings or {},
    )
    generation.build_indexes()
    return generation


def write_registry(path: Path, repos: list[dict], disabled: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"repos": repos, "disabled": disabled or [], "external_roots": []}),
        encoding="utf-8",
    )


def registry_repo(repo_id: str, root: Path, enabled: bool = True) -> dict:
    return {
        "repo_id": repo_id,
        "root": str(root),
        "collection_path": str(root / "graphify-out"),
        "enabled": enabled,
    }


def fake_embed_query_fn(vectors_by_query: dict[str, list[float]] | None = None):
    """Deterministic stand-in for `embed_query.make_embed_query_fn`'s
    returned callable: tests never touch the network. `None` in
    `vectors_by_query` (or query absent) simulates the degraded/unavailable
    path exactly like a real transport failure would."""
    table = vectors_by_query or {}

    def embed_query(query: str):
        return table.get(query)

    return embed_query


@pytest.fixture()
def gen_factory():
    return build_generation


@pytest.fixture()
def mesh_server(tmp_path: Path) -> GraphifyMeshServer:
    """A `GraphifyMeshServer` over a published synthetic generation, for a
    plain tool call with no HTTP/stdio transport involved. Registry has one
    repo, `known-repo`, rooted at `tmp_path`; the generation has one node in
    it. Reused by the SDK adapter tests and by later shared-HTTP-daemon tasks."""
    write_registry(tmp_path / "bin" / "registry.json", [registry_repo("known-repo", tmp_path)])
    generation = build_generation(
        [make_node("known-repo", "Widget", "src/widget.py", node_id="n1")]
    )
    config = ServerConfig.from_env(
        mesh_root=tmp_path, registry_path=tmp_path / "bin" / "registry.json"
    )
    server = GraphifyMeshServer(config, cwd=tmp_path, embed_query_fn=fake_embed_query_fn())
    server._generation = lambda: generation  # type: ignore[method-assign]
    return server


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture()
def registered_repo_root(tmp_path: Path) -> Path:
    """The root of `mesh_server`'s registered `known-repo` — same `tmp_path`
    that fixture registers, so a test can pass it back as the per-call
    `cwd` for `scope='current'` resolution."""
    return tmp_path


@pytest.fixture()
def http_config(tmp_path: Path) -> ServerConfig:
    """A `ServerConfig` for the streamable-HTTP transport, matching
    `mesh_server`'s mesh_root/registry_path so both fixtures describe the
    same mesh in a test that uses them together."""
    return ServerConfig(
        mesh_root=tmp_path,
        registry_path=tmp_path / "bin" / "registry.json",
        transport="http",
        http_token="test-token",
    )


@pytest.fixture()
def store_with_two_generations(tmp_path: Path):
    """A `GenerationStore` over a mesh root with one published generation,
    plus a `publish_next()` callable that writes a second generation and
    flips `current` to it — for exercising `ensure_fresh()`'s reload path
    under concurrency. Reuses `test_store._write_generation` (the real
    manifest/hash-consistent generation writer) instead of a second
    generation-building path."""
    from test_store import _write_generation  # tests/server/test_store.py

    from graphify_mesh.server.store import GenerationStore

    config = ServerConfig.from_env(
        mesh_root=tmp_path, registry_path=tmp_path / "bin" / "registry.json"
    )
    _write_generation(
        config.global_dir,
        "gen-1",
        [make_node("known-repo", "Widget", "src/widget.py", node_id="n1")],
    )
    store = GenerationStore(config)

    def publish_next() -> None:
        _write_generation(
            config.global_dir,
            "gen-2",
            [
                make_node("known-repo", "Widget", "src/widget.py", node_id="n1"),
                make_node("known-repo", "Gadget", "src/gadget.py", node_id="n2"),
            ],
        )

    return store, publish_next


@pytest.fixture()
def registry_file(tmp_path: Path) -> Path:
    """A small valid registry.json for cache concurrency tests."""
    registry_path = tmp_path / "registry.json"
    write_registry(registry_path, [registry_repo("test-repo", tmp_path)])
    return registry_path


@pytest.fixture()
def generation_fixture():
    """A loaded Generation for cache concurrency tests."""
    return build_generation([make_node("test-repo", "TestNode", "src/test.py", node_id="n1")])


# --- transport-parity fixtures -------------------------------------------
#
# Both `stdio_client` and `http_client` read the SAME on-disk generation
# under the SAME registry, written once by `parity_mesh_root` — never
# `mesh_server`'s in-memory monkeypatch, which only the in-process object it
# patches can see. That is what makes a mismatch between the two transports
# a real divergence rather than an artifact of different test data.


@pytest.fixture()
def parity_mesh_root(tmp_path: Path) -> Path:
    from test_store import _write_generation  # tests/server/test_store.py

    registry_path = tmp_path / "bin" / "registry.json"
    write_registry(registry_path, [registry_repo("known-repo", tmp_path)])
    config = ServerConfig.from_env(mesh_root=tmp_path, registry_path=registry_path)
    _write_generation(
        config.global_dir,
        "gen-parity-1",
        [
            make_node("known-repo", "Widget", "src/widget.py", node_id="n1"),
            make_node("known-repo", "Gadget", "src/gadget.py", node_id="n2"),
        ],
        links=[make_link("n1", "n2")],
    )
    return tmp_path


@pytest.fixture()
def registered_repo_id() -> str:
    """The repo_id `parity_mesh_root` registers — same one `mesh_server`
    uses, so a test can share literals across both fixture families."""
    return "known-repo"


class _StdioTestClient:
    """Wraps the real subprocess harness from `test_stdio_e2e` so a
    transport-blind test can call `tools_list()` / `call()` without caring
    that the transport underneath is a pipe to a child process."""

    def __init__(self, proc, send_fn) -> None:
        self._proc = proc
        self._send = send_fn
        self._next_id = 100

    def _id(self) -> int:
        self._next_id += 1
        return self._next_id

    async def tools_list(self) -> dict:
        resp = self._send(self._proc, {"jsonrpc": "2.0", "id": self._id(), "method": "tools/list"})
        return resp["result"]

    async def call(self, name: str, arguments: dict) -> dict:
        resp = self._send(
            self._proc,
            {
                "jsonrpc": "2.0",
                "id": self._id(),
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
        )
        return resp["result"]


@pytest.fixture()
def stdio_client(parity_mesh_root: Path):
    import test_stdio_e2e  # tests/server/test_stdio_e2e.py

    proc = test_stdio_e2e._spawn(parity_mesh_root)
    try:
        test_stdio_e2e._initialize(proc)
        yield _StdioTestClient(proc, test_stdio_e2e._send)
    finally:
        proc.stdin.close()
        proc.wait(timeout=5)


class _HttpTestClient:
    """Same two-method surface as `_StdioTestClient`, driven over
    `starlette.testclient.TestClient` (which runs the ASGI lifespan;
    `httpx.ASGITransport` does not — see `test_lifespan_starts_and_stops_the_session_manager`)."""

    def __init__(self, client: TestClient, path: str, headers: dict) -> None:
        self._client = client
        self._path = path
        self._headers = headers
        self._next_id = 100

    def _id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _post(self, payload: dict) -> dict:
        response = self._client.post(self._path, headers=self._headers, json=payload)
        body = response.text
        if body.lstrip().startswith("data:"):
            lines = [line for line in body.splitlines() if line.startswith("data:")]
            body = lines[-1][len("data:") :].strip()
        return json.loads(body)["result"]

    async def tools_list(self) -> dict:
        payload = {"jsonrpc": "2.0", "id": self._id(), "method": "tools/list", "params": {}}
        return self._post(payload)

    async def call(self, name: str, arguments: dict) -> dict:
        return self._post(
            {
                "jsonrpc": "2.0",
                "id": self._id(),
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )


@pytest.fixture()
def http_client(parity_mesh_root: Path):
    config = ServerConfig(
        mesh_root=parity_mesh_root,
        registry_path=parity_mesh_root / "bin" / "registry.json",
        transport="http",
        http_token="parity-test-shared-daemon-token-8f2c9a",  # noqa: S106
    )
    mesh = GraphifyMeshServer(config, cwd=parity_mesh_root, embed_query_fn=fake_embed_query_fn())
    app = build_http_app(mesh, config)
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.http_token}",
    }
    with TestClient(app, base_url=f"http://{config.http_host}:{config.http_port}") as c:
        c.post(
            config.http_path,
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "parity-test", "version": "0"},
                },
            },
        )
        yield _HttpTestClient(c, config.http_path, headers)
