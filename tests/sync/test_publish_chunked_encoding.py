"""publish.iter_json_chunks must yield exactly the text iterencode yields.

Published hashes (output_hash, artifact_sha256) are recomputed by the server
and compared with older generations, so any byte difference is a failure.
"""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from graphify_mesh.sync import publish

_ENCODERS = [
    {"sort_keys": True, "separators": (", ", ": ")},  # output_hash
    {"separators": (",", ":")},  # global-graph.json
    {"separators": (",", ":"), "sort_keys": True},  # overlay, lexical index
    {},
    {"ensure_ascii": False, "sort_keys": True},
]


def _sample(n: int) -> dict:
    return {
        "nodes": [
            {"id": f"r\x1fn{i}", "label": f'Ünïcödé {i} \u2028 "q"', "w": i / 3, "z": None}
            for i in range(n)
        ],
        "links": [{"source": f"n{i}", "target": f"n{i + 1}", "ok": i % 2 == 0} for i in range(n)],
        "postings": {f"tok{i:03d}": list(range(i)) for i in range(n)},
        "tuple": tuple(range(n)),
        "empty_list": [],
        "empty_dict": {},
        "nested": [[{"a": [1, 2, {"b": {}}]}], []],
        "int_keys": {1: "one", 2: "two"},
        "floats": [0.1, 1e300, -0.0, float("inf"), float("-inf")],
        "zz": "last",
        "aa": "first",
    }


def _reference(encoder: json.JSONEncoder, data: object) -> str:
    return "".join(encoder.iterencode(data))


@pytest.mark.parametrize("kwargs", _ENCODERS)
@pytest.mark.parametrize("n", [0, 1, 3, 4, 7, 9])
def test_chunks_match_iterencode_across_batch_boundaries(monkeypatch, kwargs, n):
    # Batch 3 puts n = B, B + 1 and several full batches on both code paths.
    monkeypatch.setattr(publish, "_ENCODE_BATCH", 3)
    encoder = json.JSONEncoder(**kwargs)
    data = _sample(n)
    assert "".join(publish.iter_json_chunks(encoder, data)) == _reference(encoder, data)


@pytest.mark.parametrize("depth", [0, 1, 2, 5])
def test_chunks_match_at_every_split_depth(monkeypatch, depth):
    monkeypatch.setattr(publish, "_ENCODE_BATCH", 2)
    monkeypatch.setattr(publish, "_ENCODE_DEPTH", depth)
    encoder = json.JSONEncoder(sort_keys=True, separators=(", ", ": "))
    data = _sample(5)
    assert "".join(publish.iter_json_chunks(encoder, data)) == _reference(encoder, data)


@pytest.mark.parametrize("data", [[], {}, "s", 0, None, True, [[]], {"k": {}}, (1, 2)])
def test_chunks_match_for_scalars_and_empties(data):
    encoder = json.JSONEncoder(separators=(",", ":"))
    assert "".join(publish.iter_json_chunks(encoder, data)) == _reference(encoder, data)


def test_nan_rejected_like_iterencode():
    encoder = json.JSONEncoder(allow_nan=False)
    with pytest.raises(ValueError):
        "".join(encoder.iterencode({"x": [math.nan]}))
    with pytest.raises(ValueError):
        "".join(publish.iter_json_chunks(encoder, {"x": [math.nan]}))


def test_indented_encoder_keeps_iterencode():
    encoder = json.JSONEncoder(indent=2, sort_keys=True)
    data = _sample(4)
    assert "".join(publish.iter_json_chunks(encoder, data)) == _reference(encoder, data)


def test_chunks_are_bounded(monkeypatch):
    """No chunk carries the whole artifact: the point of streaming."""
    monkeypatch.setattr(publish, "_ENCODE_BATCH", 10)
    encoder = json.JSONEncoder(separators=(",", ":"))
    data = {"nodes": [{"id": f"n{i}", "label": "x" * 50} for i in range(1000)]}
    total = len(_reference(encoder, data))
    assert max(len(c) for c in publish.iter_json_chunks(encoder, data)) < total / 50


def test_dict_with_many_keys_and_large_values_is_split(monkeypatch):
    """More keys than a batch, each value large: still no chunk with two values."""
    monkeypatch.setattr(publish, "_ENCODE_BATCH", 3)
    encoder = json.JSONEncoder(separators=(",", ":"), sort_keys=True)
    data = {"postings": {f"k{i}": [{"v": "x" * 200}] * 2 for i in range(20)}}
    chunks = list(publish.iter_json_chunks(encoder, data))
    assert "".join(chunks) == _reference(encoder, data)
    assert max(len(c) for c in chunks) < 2 * len(json.dumps({"v": "x" * 200}))


def test_cyclic_input_raises_like_iterencode():
    for data in ([], {"k": []}):
        inner = data if isinstance(data, list) else data["k"]
        inner.append(data)
        encoder = json.JSONEncoder()
        with pytest.raises(ValueError, match="Circular reference"):
            "".join(encoder.iterencode(data))
        with pytest.raises(ValueError, match="Circular reference"):
            "".join(publish.iter_json_chunks(encoder, data))


class _Opaque:
    def __init__(self, n: int) -> None:
        self.n = n


@pytest.mark.parametrize("n", [1, 4, 9])
def test_default_hook_matches_iterencode(monkeypatch, n):
    monkeypatch.setattr(publish, "_ENCODE_BATCH", 3)
    encoder = json.JSONEncoder(
        default=lambda o: {"opaque": o.n} if isinstance(o, _Opaque) else sorted(o),
        sort_keys=True,
        separators=(",", ":"),
    )
    data = {
        "objs": [_Opaque(i) for i in range(n)],
        "sets": {f"s{i}": {i, i + 1} for i in range(n)},
        "top": _Opaque(-1),
    }
    assert "".join(publish.iter_json_chunks(encoder, data)) == _reference(encoder, data)


def test_default_hook_errors_like_iterencode():
    encoder = json.JSONEncoder()
    data = {"x": [object()]}
    with pytest.raises(TypeError):
        "".join(encoder.iterencode(data))
    with pytest.raises(TypeError):
        "".join(publish.iter_json_chunks(encoder, data))


def test_output_hash_matches_json_dumps(monkeypatch):
    monkeypatch.setattr(publish, "_ENCODE_BATCH", 3)
    data = _sample(8)
    expected = hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()
    assert publish.output_hash(data) == expected


def test_written_artifacts_are_byte_identical(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "_ENCODE_BATCH", 3)
    data = _sample(8)
    digest = publish.write_global_graph(tmp_path, data)
    raw = (tmp_path / "global-graph.json").read_bytes()
    assert raw == json.dumps(data, separators=(",", ":")).encode("utf-8")
    assert digest == hashlib.sha256(raw).hexdigest()
    publish.write_lexical_index(tmp_path, data)
    assert (tmp_path / "lexical-index.json").read_bytes() == json.dumps(
        data, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
