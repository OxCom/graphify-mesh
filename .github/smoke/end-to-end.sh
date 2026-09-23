#!/usr/bin/env bash
#
# End-to-end smoke for graphify-mesh: build a two-repo fixture mesh, run the
# real sync pipeline against the real `graphify` binary, then boot the HTTP MCP
# server against what it published and talk MCP to it.
#
# Nothing here is faked. The unit suite points GRAPHIFY_BIN at
# tests/fixtures/fake_graphify/graphify, so until this script existed no CI job
# had ever executed `graphify merge-graphs` for real, nor opened the HTTP
# transport's socket. Those are exactly the two seams where a working unit suite
# still ships a broken release.
#
# Usage: bash .github/smoke/end-to-end.sh <workdir>
# The workdir is an argument rather than a mktemp so a developer can run this by
# hand exactly the way CI runs it and then poke at the tree it left behind.

set -euo pipefail

WORK="${1:?usage: end-to-end.sh <workdir>}"
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"

# Mirror the production topology the unit fixtures use (tests/sync/conftest.py):
# the mesh root lives *inside* the scan root, as a sibling of the scanned
# projects. Discovery's approved-root containment check is sensitive to this
# layout, so a smoke that flattened it would exercise a shape nobody deploys.
SCAN_ROOT="$WORK/www"
MESH_ROOT="$SCAN_ROOT/graph-mesh"
REGISTRY="$MESH_ROOT/bin/registry.json"

fail() { echo "::error::$*" >&2; exit 1; }
step() { echo; echo "=== $* ==="; }

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
# Each fake project needs real parseable source, not just a canned graph.json:
# on a repo it has no prior state for, the pipeline picks the `update` action
# and shells out to `graphify update <root>`, which re-derives the graph from
# the AST. Source that produced no nodes would trip the shrink guard and the
# run would refuse the repo instead of merging it.
make_repo() {
  local name="$1" prefix="$2"
  local root="$SCAN_ROOT/$name"
  local collection="$MESH_ROOT/graphify/smoke/$name"

  mkdir -p "$root/src" "$collection"
  cat >"$root/src/${prefix}_core.py" <<EOF
"""Fixture module for the ${name} smoke repo."""


class ${prefix^}Service:
    def handle(self, payload):
        return ${prefix}_normalize(payload)

    def describe(self):
        return "${prefix}"


def ${prefix}_normalize(payload):
    return {k: v for k, v in sorted(payload.items())}


def ${prefix}_entrypoint():
    return ${prefix^}Service().handle({})
EOF
  cat >"$root/src/${prefix}_support.py" <<EOF
"""Second fixture module so the repo has more than one source file."""

from ${prefix}_core import ${prefix^}Service


class ${prefix^}Cache:
    def __init__(self):
        self._store = {}

    def get(self, key):
        return self._store.get(key)


def ${prefix}_build_service():
    return ${prefix^}Service()
EOF

  # The graphify-out symlink is what discovery walks the scan root looking for,
  # and it points at the registry-declared collection path — the project dir
  # never owns its own output directory.
  ln -sfn "$collection" "$root/graphify-out"

  # Seed a graph so the repo is an `update` (AST-only) and not a `bootstrap`.
  # Bootstrap runs `graphify extract --backend ollama`, and neither the CI
  # runner nor the test container has an Ollama to reach.
  cp "$SEED_GRAPH" "$collection/graph.json"
}

step "Building the fixture mesh under $WORK"
SEED_GRAPH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/tests/fixtures/graphs/repo_a.json"
[ -f "$SEED_GRAPH" ] || fail "seed graph fixture not found at $SEED_GRAPH"

mkdir -p "$MESH_ROOT/bin"
make_repo repo-a alpha
make_repo repo-b beta

cat >"$REGISTRY" <<EOF
{
  "repos": [
    {
      "repo_id": "smoke.repo-a",
      "root": "$SCAN_ROOT/repo-a",
      "collection_path": "$MESH_ROOT/graphify/smoke/repo-a",
      "enabled": true
    },
    {
      "repo_id": "smoke.repo-b",
      "root": "$SCAN_ROOT/repo-b",
      "collection_path": "$MESH_ROOT/graphify/smoke/repo-b",
      "enabled": true
    }
  ],
  "disabled": [],
  "external_roots": []
}
EOF

# The registry is read by a process that also runs child binaries, so the sync
# entrypoint audits its permissions and complains about a group/world-writable
# config. A default umask on a CI runner is fine, but be explicit.
chmod 600 "$REGISTRY"

GLOBAL_DIR="$MESH_ROOT/graphify/global"
CURRENT="$GLOBAL_DIR/current"

# --skip-labeling / --skip-embedding: both stages talk to Ollama, which exists
# on neither the GitHub runner nor the test container. Everything this smoke
# asserts on — discovery, per-repo sync, merge, repo-tag remap, overlay,
# lexical index, validate, publish — runs regardless.
SYNC_ARGS=(
  --once
  --mesh-root "$MESH_ROOT"
  --scan-root "$SCAN_ROOT"
  --scan-depth 2
  --registry "$REGISTRY"
  --skip-labeling
  --skip-embedding
)

# ---------------------------------------------------------------------------
# Phase 1 — a dry run must write nothing into the mesh root
# ---------------------------------------------------------------------------
# Dry-run redirects the publish step (sync/pipeline.py) and the naming
# workspace (sync/config.py) into an ephemeral staging root. A dry run that
# leaves a generation behind is a real regression: it means somebody scheduled
# a "safe" preview that mutated the tree every agent reads from.
step "Phase 1: dry run"
graphify-mesh-sync "${SYNC_ARGS[@]}" --dry-run

if [ -e "$CURRENT" ]; then
  fail "dry run published a generation: $CURRENT exists"
fi
if [ -d "$GLOBAL_DIR/generations" ] && [ -n "$(ls -A "$GLOBAL_DIR/generations")" ]; then
  fail "dry run wrote into $GLOBAL_DIR/generations"
fi
echo "OK: no generation published by the dry run"

# ---------------------------------------------------------------------------
# Phase 2 — a real run publishes a complete, two-repo generation
# ---------------------------------------------------------------------------
step "Phase 2: real run"
graphify-mesh-sync "${SYNC_ARGS[@]}"

[ -L "$CURRENT" ] || fail "no current symlink at $CURRENT"
GEN="$(readlink -f "$CURRENT")"
[ -d "$GEN" ] || fail "current symlink does not resolve to a directory: $CURRENT"
echo "OK: current -> $GEN"

# Assert on the published artifacts rather than on the pipeline's own exit
# code: publish flips the symlink last, so a half-written generation is
# precisely the failure mode a green exit code would hide.
python3 - "$GEN" <<'PY'
import json, sys
from pathlib import Path

gen = Path(sys.argv[1])
docs = {}
for name in ("global-graph.json", "cross-project-overlay.json", "lexical-index.json"):
    path = gen / name
    if not path.is_file():
        sys.exit(f"::error::missing published file: {path}")
    try:
        docs[name] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        sys.exit(f"::error::{path} is not valid JSON: {err}")
    print(f"OK: {name} exists and parses")

# A merge that silently drops one input still produces a valid graph, so the
# only assertion worth making is per-repo attribution: every fixture repo must
# own at least one node in the merged structural graph, tagged with the
# repo_id the registry declared (not graphify's own auto tag, which the remap
# stage rewrites).
nodes = docs["global-graph.json"].get("nodes") or []
seen = {}
for node in nodes:
    repo = node.get("repo_id") or node.get("repo")
    if repo:
        seen[repo] = seen.get(repo, 0) + 1
print(f"node count={len(nodes)} repo attribution={seen}")
missing = [r for r in ("smoke.repo-a", "smoke.repo-b") if seen.get(r, 0) == 0]
if missing:
    sys.exit(f"::error::merged graph has no nodes for: {', '.join(missing)}")
print("OK: both fixture repos are represented in global-graph.json")
PY

# ---------------------------------------------------------------------------
# Phase 3 — the HTTP transport serves MCP, and refuses without the token
# ---------------------------------------------------------------------------
step "Phase 3: HTTP MCP server"

# Never a literal in the file, never echoed: this token is the only thing
# between the daemon and every other process on the machine.
GRAPHIFY_MESH_HTTP_TOKEN="$(openssl rand -hex 32)"
export GRAPHIFY_MESH_HTTP_TOKEN
HTTP_HOST=127.0.0.1
HTTP_PORT=19744
HTTP_PATH=/mcp
URL="http://${HTTP_HOST}:${HTTP_PORT}${HTTP_PATH}"

SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
# A failing assertion below must not leave a listening daemon behind for the
# rest of the job (or, locally, for the rest of the day).
trap cleanup EXIT

# The server has no --mesh-root/--registry flags; both come from the
# environment (src/graphify_mesh/server/config.py), and mesh_root otherwise
# defaults to the current working directory.
GRAPHIFY_MESH_ROOT="$MESH_ROOT" GRAPHIFY_MESH_REGISTRY="$REGISTRY" \
  graphify-mesh-server \
    --transport http --host "$HTTP_HOST" --port "$HTTP_PORT" --path "$HTTP_PATH" \
    >"$WORK/server.log" 2>&1 &
SERVER_PID=$!

# Wait for the port instead of sleeping a fixed interval, and surface the
# server's own log if it died during startup (a config error exits immediately).
for _ in $(seq 1 60); do
  if python3 -c "
import socket,sys
s=socket.socket()
s.settimeout(0.5)
sys.exit(0 if s.connect_ex(('$HTTP_HOST', $HTTP_PORT))==0 else 1)
"; then
    break
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || { cat "$WORK/server.log" >&2; fail "server exited during startup"; }
  sleep 0.5
done

# The MCP conversation and the negative check share one script because both
# need the same headers and the same streamable-HTTP framing. Python's urllib
# rather than curl: it is already a hard dependency of everything else here,
# and it can assert on the parsed body instead of on grep.
python3 - "$URL" <<'PY'
import json, os, sys, urllib.error, urllib.request

url = sys.argv[1]
token = os.environ["GRAPHIFY_MESH_HTTP_TOKEN"]
EXPECTED = {"search", "cross_project", "find_similar", "project_map", "neighbors", "context_pack"}


def post(payload, bearer):
    body = json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
        # The streamable-HTTP transport requires the client to declare it can
        # take either framing, even though this server is configured
        # json_response=True and never opens an SSE stream.
        "Accept": "application/json, text/event-stream",
    }
    if bearer is not None:
        headers["Authorization"] = f"Bearer {bearer}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, resp.read().decode()


def rpc(payload, bearer=token):
    status, text = post(payload, bearer)
    if status != 200:
        sys.exit(f"::error::{payload.get('method')} returned HTTP {status}: {text[:400]}")
    # json_response=True still permits an SSE-shaped body; unwrap it if present
    # so this assertion does not depend on which framing the SDK picked.
    if text.startswith("event:") or text.startswith("data:"):
        text = "\n".join(
            line[len("data:"):].strip()
            for line in text.splitlines()
            if line.startswith("data:")
        )
    return json.loads(text)


init = rpc({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "graphify-mesh-smoke", "version": "0"},
    },
})
if "error" in init:
    sys.exit(f"::error::initialize failed: {init['error']}")
print("OK: initialize ->", init["result"]["serverInfo"])

listed = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
if "error" in listed:
    sys.exit(f"::error::tools/list failed: {listed['error']}")
names = {t["name"] for t in listed["result"]["tools"]}
print("OK: tools/list ->", sorted(names))
if names != EXPECTED:
    sys.exit(f"::error::tool set mismatch: missing={EXPECTED - names} unexpected={names - EXPECTED}")

# The negative half. A smoke that only ever sends the right token proves the
# happy path and nothing about the thing the token exists for.
for label, bearer in (("no token", None), ("wrong token", "x" * 64)):
    try:
        status, text = post({"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}}, bearer)
    except urllib.error.HTTPError as err:
        print(f"OK: {label} refused with HTTP {err.code}")
        continue
    sys.exit(f"::error::{label} was accepted with HTTP {status}: {text[:200]}")
PY

step "All three phases passed"
