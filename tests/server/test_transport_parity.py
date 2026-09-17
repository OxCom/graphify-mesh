"""The two transports must answer the same request identically.

One registration path (sdk_app.build_sdk_server) is what makes this true;
this test is what keeps it true when someone adds a tool to one path only.

Both `stdio_client` and `http_client` (tests/server/conftest.py) read the
same on-disk generation under `parity_mesh_root` — stdio via a real
subprocess (reusing tests/server/test_stdio_e2e.py's harness), http via
`starlette.testclient.TestClient` driving `build_http_app` in process. A
difference here is a real transport divergence: fix the divergence, never
the assertion.
"""

from __future__ import annotations

import pytest


@pytest.mark.anyio
async def test_tools_list_is_identical(stdio_client, http_client):
    stdio_tools = await stdio_client.tools_list()
    http_tools = await http_client.tools_list()
    assert stdio_tools == http_tools


@pytest.mark.anyio
async def test_project_map_result_is_identical(stdio_client, http_client, registered_repo_id):
    args = {"repo": registered_repo_id}
    assert await stdio_client.call("project_map", args) == await http_client.call(
        "project_map", args
    )


@pytest.mark.anyio
async def test_project_map_unregistered_repo_is_identical(stdio_client, http_client):
    args = {"repo": "not-a-registered-repo"}
    assert await stdio_client.call("project_map", args) == await http_client.call(
        "project_map", args
    )


@pytest.mark.anyio
async def test_search_success_is_identical(stdio_client, http_client, registered_repo_id):
    args = {"q": "widget", "scope": "all", "k": 5}
    stdio_result = await stdio_client.call("search", args)
    http_result = await http_client.call("search", args)
    assert stdio_result["isError"] is False
    assert stdio_result == http_result


@pytest.mark.anyio
async def test_scope_failure_is_identical(stdio_client, http_client):
    args = {"q": "x", "cwd": "/definitely/not/registered"}
    stdio_result = await stdio_client.call("search", args)
    http_result = await http_client.call("search", args)
    assert stdio_result["isError"] is True
    assert stdio_result == http_result


@pytest.mark.anyio
async def test_cross_project_result_is_identical(stdio_client, http_client, registered_repo_id):
    args = {"q": "widget", "repos": [registered_repo_id]}
    stdio_result = await stdio_client.call("cross_project", args)
    http_result = await http_client.call("cross_project", args)
    assert stdio_result["isError"] is False
    assert stdio_result == http_result


@pytest.mark.anyio
async def test_find_similar_result_is_identical(stdio_client, http_client):
    args = {"node": "Widget", "k": 5}
    stdio_result = await stdio_client.call("find_similar", args)
    http_result = await http_client.call("find_similar", args)
    assert stdio_result["isError"] is False
    assert stdio_result == http_result


@pytest.mark.anyio
async def test_find_similar_unresolved_node_is_identical(stdio_client, http_client):
    args = {"node": "NoSuchNode"}
    stdio_result = await stdio_client.call("find_similar", args)
    http_result = await http_client.call("find_similar", args)
    assert stdio_result["isError"] is False
    assert stdio_result == http_result


@pytest.mark.anyio
async def test_context_pack_result_is_identical(stdio_client, http_client):
    args = {"goal": "widget", "scope": "all", "token_budget": 500}
    stdio_result = await stdio_client.call("context_pack", args)
    http_result = await http_client.call("context_pack", args)
    assert stdio_result["isError"] is False
    assert stdio_result == http_result
