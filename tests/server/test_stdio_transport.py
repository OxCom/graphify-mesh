"""Unit tests for `stdio_guard.capped_stdin`: the size cap and the frame-shape
validation that `mcp.server.stdio.stdio_server`'s own reader does not provide
(see the module docstring for the evidence)."""

from __future__ import annotations

import io
import json
import os
import sys
import threading
from collections.abc import AsyncIterator

import anyio
import mcp.types as types
import pytest
from mcp.server.stdio import stdio_server
from mcp.shared.message import SessionMessage

from graphify_mesh.server.stdio_guard import MAX_LINE_BYTES, _CappedLineReader, capped_stdin


def test_oversized_line_is_dropped_not_buffered(caplog):
    huge = "x" * (MAX_LINE_BYTES + 10)
    stream = _text_stream(
        f"{huge}\n" + json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}) + "\n"
    )
    lines = list(_drain(capped_stdin(stream)))
    assert len(lines) == 1
    assert json.loads(lines[0])["method"] == "ping"
    assert any("oversized" in r.message for r in caplog.records)


def test_eof_ends_the_stream():
    assert list(_drain(capped_stdin(_text_stream("")))) == []


def test_blank_lines_are_skipped():
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    lines = list(_drain(capped_stdin(_text_stream(f"\n\n{payload}\n"))))
    assert len(lines) == 1


def test_unparseable_line_gets_parse_error_and_is_not_forwarded():
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    out = io.StringIO()
    lines = list(_drain(capped_stdin(_text_stream(f"{{not json\n{payload}\n"), out)))
    assert len(lines) == 1
    assert json.loads(lines[0])["method"] == "ping"
    response = json.loads(out.getvalue().strip())
    assert response == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": -32700, "message": "parse error"},
    }


def test_batch_array_gets_invalid_request_and_is_not_forwarded():
    out = io.StringIO()
    lines = list(_drain(capped_stdin(_text_stream('[{"jsonrpc": "2.0", "id": 1}]\n'), out)))
    assert lines == []
    response = json.loads(out.getvalue().strip())
    assert response["id"] is None
    assert response["error"]["code"] == -32600


# --- pinning the write-safety assumption stdio_guard's module docstring
# relies on: the guard's error write and the SDK's own response write only
# avoid interleaving because they end up calling .write() on the SAME
# underlying io.BufferedWriter. Nothing in mcp or the stdlib promises this
# will keep being true, so these tests exist to catch it breaking.


@pytest.mark.anyio
async def test_sdk_stdout_shares_our_buffer_or_frames_can_interleave(monkeypatch):
    """This is the assumption stdio_guard's error-write safety relies on:
    `mcp.server.stdio.stdio_server`'s default stdout must wrap whatever
    `sys.stdout.buffer` is at call time, the same object `capped_stdin`'s
    default `out_stream` (`sys.stdout`) writes through — that shared buffer
    is the only thing that makes the two writers' lines never interleave
    (see `stdio_guard.py`'s module docstring and
    `test_error_write_and_sdk_style_write_do_not_interleave_on_shared_buffer`
    below). If a future `mcp` release builds its stdout from something else
    (a different global, a cached handle, a socket), this test fails instead
    of corrupted frames showing up in production."""
    raw = io.BytesIO()
    fake_stdout = io.TextIOWrapper(raw, encoding="utf-8")
    monkeypatch.setattr(sys, "stdout", fake_stdout)

    async with stdio_server(stdin=capped_stdin(io.StringIO(""))) as (_read_stream, write_stream):
        # `types.JSONRPCMessage` is a union type alias, not a constructible
        # class, as of mcp 2.x — `SessionMessage.message` takes a union
        # member (e.g. `JSONRPCNotification`) directly.
        message = types.JSONRPCNotification(jsonrpc="2.0", method="probe")
        await write_stream.send(SessionMessage(message=message))
        await anyio.sleep(0.1)  # let stdio_server's stdout_writer task drain and flush
        written = raw.getvalue()
        await write_stream.aclose()

    # Proof the SDK actually wrote through OUR fake sys.stdout.buffer, not
    # some other stream it built or cached independently.
    assert b"probe" in written


def test_error_write_and_sdk_style_write_do_not_interleave_on_shared_buffer():
    """Scaled-down concurrency regression for the guarantee above: two
    threads, one driving the guard's own error-write path
    (`_CappedLineReader._write_error`) and one writing plain JSON lines the
    way `stdio_server`'s `stdout_writer` does, both through independent
    `TextIOWrapper`s over the SAME shared `io.BufferedWriter` — mirroring
    `capped_stdin`'s default `out_stream=sys.stdout` and the SDK's own
    `TextIOWrapper(sys.stdout.buffer, ...)`. Every line received must be a
    complete, parseable JSON object: a fragment or a merge of two lines
    means CPython's `io.BufferedWriter.write()` stopped serializing
    concurrent calls, exactly what this module's write-safety argument
    depends on."""
    read_fd, write_fd = os.pipe()
    raw_writer = os.fdopen(write_fd, "wb", buffering=0)
    shared_buffer = io.BufferedWriter(raw_writer)

    guard_stream = io.TextIOWrapper(shared_buffer, encoding="utf-8")
    sdk_style_stream = io.TextIOWrapper(shared_buffer, encoding="utf-8")
    reader = _CappedLineReader(stream=io.StringIO(""), out_stream=guard_stream)

    n_per_thread = 300
    received: list[bytes] = []

    def drain() -> None:
        rf = os.fdopen(read_fd, "rb", buffering=0)
        buf = b""
        while len(received) < n_per_thread * 2:
            chunk = rf.read(65536)
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                received.append(line)

    def write_errors() -> None:
        for _ in range(n_per_thread):
            reader._write_error(-32700, "parse error")

    def write_direct() -> None:
        for i in range(n_per_thread):
            sdk_style_stream.write(json.dumps({"jsonrpc": "2.0", "id": i, "result": {}}) + "\n")
            sdk_style_stream.flush()

    reader_thread = threading.Thread(target=drain)
    error_thread = threading.Thread(target=write_errors)
    direct_thread = threading.Thread(target=write_direct)
    reader_thread.start()
    error_thread.start()
    direct_thread.start()
    error_thread.join(timeout=5)
    direct_thread.join(timeout=5)
    shared_buffer.close()  # closes raw_writer too: signals EOF to the drain thread
    reader_thread.join(timeout=5)

    assert len(received) == n_per_thread * 2
    for line in received:
        parsed = json.loads(line)  # raises on a fragment or a merge of two lines
        assert isinstance(parsed, dict)


def _text_stream(text: str) -> io.StringIO:
    return io.StringIO(text)


def _drain(async_file) -> list[str]:
    async def _collect() -> list[str]:
        collected: list[str] = []
        it: AsyncIterator[str] = async_file.__aiter__()
        async for line in it:
            collected.append(line)
        return collected

    return anyio.run(_collect)


# --- envelope validation (the regression left by removing protocol.py) ---
#
# The dict check alone let a frame with legal JSON and an illegal envelope
# through to the SDK's pydantic validation, which raises instead of answering
# the request, so the client waited for its own timeout. These pin the coded,
# id-correlated answer the retired dispatcher gave for the same input.


def test_params_shape_error_gets_invalid_params_with_the_request_id():
    out = io.StringIO()
    frame = '{"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": []}\n'
    lines = list(_drain(capped_stdin(_text_stream(frame), out)))
    assert lines == []
    response = json.loads(out.getvalue().strip())
    assert response["id"] == 7
    assert response["error"]["code"] == -32602


def test_invalid_envelope_gets_invalid_request_with_the_request_id():
    out = io.StringIO()
    frame = '{"jsonrpc": "1.0", "id": "abc", "method": "ping"}\n'
    lines = list(_drain(capped_stdin(_text_stream(frame), out)))
    assert lines == []
    response = json.loads(out.getvalue().strip())
    assert response["id"] == "abc"
    assert response["error"]["code"] == -32600


def test_non_scalar_id_gets_invalid_request_with_a_null_id():
    out = io.StringIO()
    frame = '{"jsonrpc": "2.0", "id": [], "method": "ping"}\n'
    lines = list(_drain(capped_stdin(_text_stream(frame), out)))
    assert lines == []
    response = json.loads(out.getvalue().strip())
    assert response["id"] is None
    assert response["error"]["code"] == -32600


def test_missing_method_gets_invalid_request():
    out = io.StringIO()
    lines = list(_drain(capped_stdin(_text_stream('{"jsonrpc": "2.0", "id": 3}\n'), out)))
    assert lines == []
    assert json.loads(out.getvalue().strip())["error"]["code"] == -32600


def test_error_messages_never_quote_the_offending_frame():
    out = io.StringIO()
    frame = '{"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": ["s3cret-marker"]}\n'
    list(_drain(capped_stdin(_text_stream(frame), out)))
    assert "s3cret-marker" not in out.getvalue()


def test_notifications_and_client_responses_still_reach_the_sdk():
    """The envelope check must not reject the legal shapes: a notification
    (no id) and a client-to-server response (no method, carries result) —
    MCP sends the latter for sampling and roots."""
    notification = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
    response_frame = json.dumps({"jsonrpc": "2.0", "id": 4, "result": {"roots": []}})
    out = io.StringIO()
    lines = list(_drain(capped_stdin(_text_stream(f"{notification}\n{response_frame}\n"), out)))
    assert len(lines) == 2
    assert out.getvalue() == ""


def test_oversized_line_is_read_in_bounded_chunks_never_whole():
    """The cap's point is that an over-limit line is DRAINED, not buffered.
    This records every `readline` size the guard asks for: each one must be
    bounded by the cap, however long the line on the wire is. A guard that
    called `readline()` with no argument would still drop the line and still
    pass `test_oversized_line_is_dropped_not_buffered`, while having already
    pulled the whole thing into memory."""

    class _RecordingStream:
        def __init__(self, text: str) -> None:
            self._inner = io.StringIO(text)
            self.sizes: list[int] = []

        def readline(self, size: int = -1) -> str:
            self.sizes.append(size)
            return self._inner.readline(size)

    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    stream = _RecordingStream("x" * (MAX_LINE_BYTES * 3) + "\n" + payload + "\n")
    lines = list(_drain(capped_stdin(stream)))

    assert len(lines) == 1
    assert stream.sizes, "the guard never read through a bounded readline"
    assert all(0 < size <= MAX_LINE_BYTES + 1 for size in stream.sizes), stream.sizes
    assert len(stream.sizes) > 3  # the oversized line took several drain reads


def test_explicit_null_params_is_not_a_params_shape_error():
    """`params` is typed `dict | None` in the SDK's own models, so
    `{"jsonrpc":"2.0","id":1,"method":"ping","params":null}` validates there
    and a client whose serializer emits explicit `null` has always worked.
    The envelope check must not turn that into `-32602`."""
    out = io.StringIO()
    frame = '{"jsonrpc": "2.0", "id": 1, "method": "ping", "params": null}\n'
    lines = list(_drain(capped_stdin(_text_stream(frame), out)))
    assert len(lines) == 1
    assert out.getvalue() == ""
