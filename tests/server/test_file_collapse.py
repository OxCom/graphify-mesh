"""Per-file collapse of `retrieval.rank` (V3): one hit per (repo, source_file),
the file's best-ranked position kept, the list filled to k from further down,
exact/anchor precedence within a file, the file-named node as the file's
representative, test slots on the collapsed list, and the callers that opt out."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import (
    build_generation,
    fake_embed_query_fn,
    key_for,
    make_node,
    registry_repo,
    write_registry,
)
from test_anchors import _fillers
from test_server import _server

from graphify_mesh.server import anchors, retrieval
from graphify_mesh.server.context_pack import build_context_pack
from graphify_mesh.server.retrieval import rank

# Label words without digits: a digit glued to a word changes its token.
WORDS = ("Refund", "Draft", "Archive", "Notify", "Export")


def _files(result) -> list[tuple[str, str]]:
    return [(hit.repo, hit.source_file) for hit in result.hits]


def _crowded_generation() -> list[dict]:
    """Three nodes in each of two strongly matching files, then five files
    with one weaker match each."""
    nodes = []
    for f in ("alpha", "beta"):
        nodes += [
            make_node("repo.a", f".invoiceTotal{f}{i}()", f"src/{f}.py", node_id=f"{f}{i}")
            for i in range(3)
        ]
    nodes += [
        make_node("repo.a", f".invoice{word}()", f"src/weak{i}.py", node_id=f"w{i}")
        for i, word in enumerate(WORDS)
    ]
    return nodes + _fillers()


def test_one_hit_per_file_and_list_filled_to_k():
    gen = build_generation(_crowded_generation())

    result = rank("invoice total", gen, None, k=5, embed_query_fn=fake_embed_query_fn())

    files = _files(result)
    assert len(files) == 5
    assert len(set(files)) == 5
    # Both crowded files keep their place at the top; the rest is filled from
    # the weaker single-node files.
    assert {f for _repo, f in files[:2]} == {"src/alpha.py", "src/beta.py"}
    assert all(f.startswith("src/weak") for _repo, f in files[2:])


def test_collapse_off_keeps_several_nodes_of_one_file():
    gen = build_generation(_crowded_generation())

    result = rank(
        "invoice total",
        gen,
        None,
        k=5,
        embed_query_fn=fake_embed_query_fn(),
        collapse_files=False,
    )

    files = [f for _repo, f in _files(result)]
    assert len(files) == 5
    assert files.count("src/alpha.py") + files.count("src/beta.py") > 2


def test_collapsed_top_n_is_a_prefix_of_a_larger_k():
    gen = build_generation(_crowded_generation())
    embed = fake_embed_query_fn()

    small = rank("invoice total", gen, None, k=3, embed_query_fn=embed)
    large = rank("invoice total", gen, None, k=7, embed_query_fn=embed)

    assert [h.key for h in large.hits[:3]] == [h.key for h in small.hits]


HOLIDAYS = "src/Cron/PopulatePublicHolidays.php"


def test_file_node_gives_way_to_the_class_at_the_files_best_position():
    file_node = make_node("repo.a", "PopulatePublicHolidays.php", HOLIDAYS, node_id="f")
    cls = make_node("repo.a", "PopulatePublicHolidays", HOLIDAYS, node_id="c", line=12)
    other = make_node("repo.a", ".publicOther()", "src/Other.php", node_id="o")
    gen = build_generation([file_node, cls, other] + _fillers())
    query = "populate public holidays php"

    expanded = rank(
        query, gen, None, k=5, embed_query_fn=fake_embed_query_fn(), collapse_files=False
    )
    # Precondition: the file node outranks the class node of the same file.
    keys = [h.key for h in expanded.hits]
    assert keys[0] == key_for("repo.a", file_node)
    assert key_for("repo.a", cls) in keys

    result = rank(query, gen, None, k=5, embed_query_fn=fake_embed_query_fn())

    assert result.hits[0].key == key_for("repo.a", cls)
    assert [h.source_file for h in result.hits].count(HOLIDAYS) == 1


def test_a_path_qualified_file_node_gives_way_too():
    # graphify labels a file node with a path suffix when its basename is not
    # unique in the repo.
    file_node = make_node("repo.a", "Cron/PopulatePublicHolidays.php", HOLIDAYS, node_id="f")
    cls = make_node("repo.a", "PopulatePublicHolidays", HOLIDAYS, node_id="c", line=12)
    gen = build_generation([file_node, cls] + _fillers())
    query = "cron populate public holidays php"

    expanded = rank(
        query, gen, None, k=5, embed_query_fn=fake_embed_query_fn(), collapse_files=False
    )
    assert expanded.hits[0].key == key_for("repo.a", file_node)

    result = rank(query, gen, None, k=5, embed_query_fn=fake_embed_query_fn())

    assert result.hits[0].key == key_for("repo.a", cls)


def test_a_backslash_file_node_gives_way_too():
    windows_path = "src\\Cron\\PopulatePublicHolidays.php"
    file_node = make_node("repo.a", "Cron\\PopulatePublicHolidays.php", windows_path, node_id="f")
    cls = make_node("repo.a", "PopulatePublicHolidays", windows_path, node_id="c", line=12)
    gen = build_generation([file_node, cls] + _fillers())
    query = "cron populate public holidays php"

    expanded = rank(
        query, gen, None, k=5, embed_query_fn=fake_embed_query_fn(), collapse_files=False
    )
    assert expanded.hits[0].key == key_for("repo.a", file_node)

    result = rank(query, gen, None, k=5, embed_query_fn=fake_embed_query_fn())

    assert result.hits[0].key == key_for("repo.a", cls)


def test_method_keeps_its_file_over_the_class_named_after_the_file():
    lock = "src/Cron/LockWeeks.php"
    method = make_node("repo.a", ".lockWeeksExecute()", lock, node_id="m")
    cls = make_node("repo.a", "LockWeeks", lock, node_id="c")
    other = make_node("repo.a", ".lockOther()", "src/Other.php", node_id="o")
    gen = build_generation([method, cls, other] + _fillers())

    result = rank("lock weeks execute", gen, None, k=5, embed_query_fn=fake_embed_query_fn())

    assert result.hits[0].key == key_for("repo.a", method)
    assert [h.source_file for h in result.hits].count(lock) == 1


def test_a_lone_file_node_still_represents_its_file():
    file_node = make_node("repo.a", "PopulatePublicHolidays.php", HOLIDAYS, node_id="f")
    gen = build_generation([file_node] + _fillers())

    result = rank("populate public holidays", gen, None, k=5, embed_query_fn=fake_embed_query_fn())

    assert result.hits[0].key == key_for("repo.a", file_node)


def test_best_node_represents_a_file_with_no_file_named_node():
    nodes = [
        make_node("repo.a", ".invoiceTotalA()", "src/billing.py", node_id="a"),
        make_node("repo.a", "Helper", "src/billing.py", node_id="h"),
    ]
    gen = build_generation(nodes + _fillers())
    expanded = rank(
        "invoice total",
        gen,
        None,
        k=5,
        embed_query_fn=fake_embed_query_fn(),
        collapse_files=False,
    )

    result = rank("invoice total", gen, None, k=5, embed_query_fn=fake_embed_query_fn())

    billing = [h for h in result.hits if h.source_file == "src/billing.py"]
    assert len(billing) == 1
    assert billing[0].key == expanded.hits[0].key


def test_exact_hit_takes_its_file_and_drops_the_fused_siblings():
    exact = make_node("repo.a", "OrderService", "src/order.py", node_id="e")
    siblings = [
        make_node("repo.a", f".orderService{i}()", "src/order.py", node_id=f"s{i}")
        for i in range(3)
    ]
    elsewhere = make_node("repo.a", ".orderServiceElsewhere()", "src/other.py", node_id="x")
    gen = build_generation([exact, *siblings, elsewhere] + _fillers())

    result = rank("OrderService", gen, None, k=5, embed_query_fn=fake_embed_query_fn())

    assert result.hits[0].key == key_for("repo.a", exact)
    assert result.hits[0].match_type == "exact"
    assert [h.source_file for h in result.hits].count("src/order.py") == 1
    assert "src/other.py" in [h.source_file for h in result.hits]


def test_exact_hits_sharing_a_file_count_once_against_k():
    # Case variants share one normalized alias but are distinct logical keys.
    nodes = [
        make_node("repo.a", label, "src/order.py", node_id=f"e{i}", line=i + 1)
        for i, label in enumerate(("OrderService", "orderService", "Orderservice"))
    ]
    nodes.append(make_node("repo.a", "OrderService", "src/other.py", node_id="e9"))
    gen = build_generation(nodes + _fillers())
    expanded = rank(
        "OrderService",
        gen,
        None,
        k=3,
        embed_query_fn=fake_embed_query_fn(),
        collapse_files=False,
    )
    # Precondition: the three exact hits of src/order.py fill k=3 on their own.
    assert [h.source_file for h in expanded.hits] == ["src/order.py"] * 3

    result = rank("OrderService", gen, None, k=3, embed_query_fn=fake_embed_query_fn())

    files = [h.source_file for h in result.hits]
    assert len(files) == len(set(files))
    assert "src/order.py" in files and "src/other.py" in files


def test_anchor_hit_takes_its_file_over_a_fused_sibling(monkeypatch):
    anchor = make_node("repo.a", ".exportRows()", "src/Export.php", node_id="an")
    sibling = make_node("repo.a", ".exportRowsCsv()", "src/Export.php", node_id="sib")
    other = make_node("repo.a", ".exportOther()", "src/Other.php", node_id="o")
    gen = build_generation([anchor, sibling, other] + _fillers())
    anchor_key = key_for("repo.a", anchor)
    monkeypatch.setattr(anchors, "anchor_keys", lambda *a, **kw: [anchor_key])

    result = rank("export rows csv", gen, None, k=5, embed_query_fn=fake_embed_query_fn())

    export_hits = [h for h in result.hits if h.source_file == "src/Export.php"]
    assert [h.key for h in export_hits] == [anchor_key]
    assert export_hits[0].match_type == "anchor"


def test_same_path_in_two_repos_is_two_files():
    nodes = [
        make_node("repo.a", ".invoiceTotal()", "src/billing.py", node_id="a"),
        make_node("repo.b", ".invoiceTotal()", "src/billing.py", node_id="b"),
    ]
    gen = build_generation(nodes + _fillers("repo.a"))

    result = rank("invoice total", gen, None, k=5, embed_query_fn=fake_embed_query_fn())

    assert {("repo.a", "src/billing.py"), ("repo.b", "src/billing.py")} <= set(_files(result))


def test_reserved_test_slot_runs_on_the_collapsed_list():
    source = [
        make_node("repo.a", f".overtimeCalculation{i}()", f"src/Overtime{i}.php", node_id=f"s{i}")
        for i in range(30)
    ]
    # Two nodes in the same test file: the slot takes one, never both.
    tests = [
        make_node(
            "repo.a",
            f".testOvertimeCalculationRounding{i}()",
            "tests/php/OvertimeTest.php",
            node_id=f"t{i}",
        )
        for i in range(2)
    ]
    gen = build_generation(source + tests)

    result = rank(
        "overtime calculation rounding", gen, None, k=20, embed_query_fn=fake_embed_query_fn()
    )

    files = _files(result)
    assert len(files) == 20
    assert len(set(files)) == 20
    assert files[retrieval.TEST_RESULT_SLOT_EVERY - 1] == ("repo.a", "tests/php/OvertimeTest.php")


# --- callers --------------------------------------------------------------


def _billing_nodes() -> list[dict]:
    return [
        make_node("repo.a", f".orderTotal{i}()", "src/order.py", node_id=f"o{i}", line=i + 1)
        for i in range(3)
    ] + _fillers()


def test_context_pack_keeps_several_cards_per_file():
    gen = build_generation(_billing_nodes())

    result = build_context_pack(
        "order total", gen, None, [], token_budget=100_000, embed_query_fn=fake_embed_query_fn()
    )

    assert [c.source_file for c in result.cards].count("src/order.py") == 3


def _search_server(tmp_path: Path, monkeypatch):
    root = tmp_path / "acme"
    root.mkdir(parents=True)
    write_registry(tmp_path / "bin" / "registry.json", [registry_repo("repo.a", root)])
    server = _server(tmp_path, root)
    gen = build_generation(_billing_nodes())
    monkeypatch.setattr(server, "_generation", lambda: gen)
    return server, root


def _tool_hits(server, tool: str, arguments: dict) -> list[dict]:
    result = server.call_tool(tool, arguments)
    assert result["isError"] is False, result
    return json.loads(result["content"][0]["text"])["hits"]


def test_search_tool_collapses_by_default_and_can_opt_out(tmp_path, monkeypatch):
    server, root = _search_server(tmp_path, monkeypatch)
    args = {"q": "order total", "cwd": str(root)}

    collapsed = _tool_hits(server, "search", args)
    expanded = _tool_hits(server, "search", {**args, "collapse_files": False})

    assert [h["source_file"] for h in collapsed].count("src/order.py") == 1
    assert [h["source_file"] for h in expanded].count("src/order.py") == 3


def test_search_tool_rejects_a_non_boolean_collapse_files(tmp_path, monkeypatch):
    server, root = _search_server(tmp_path, monkeypatch)

    result = server.call_tool("search", {"q": "order", "cwd": str(root), "collapse_files": "no"})

    assert result["isError"] is True


def test_search_tool_schema_declares_collapse_files(tmp_path, monkeypatch):
    server, _root = _search_server(tmp_path, monkeypatch)
    schema = next(t for t in server.tool_schemas() if t["name"] == "search")

    assert schema["inputSchema"]["properties"]["collapse_files"] == {
        "type": "boolean",
        "default": True,
        "description": schema["inputSchema"]["properties"]["collapse_files"]["description"],
    }


def test_cross_project_keeps_several_hits_per_file(tmp_path, monkeypatch):
    server, _root = _search_server(tmp_path, monkeypatch)

    hits = _tool_hits(server, "cross_project", {"q": "order total"})

    assert [h["source_file"] for h in hits].count("src/order.py") == 3
