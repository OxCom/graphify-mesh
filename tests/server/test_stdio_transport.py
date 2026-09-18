"""Unit tests for `stdio_guard.capped_stdin`: the size cap and the frame-shape
validation that `mcp.server.stdio.stdio_server`'s own reader does not provide
(see the module docstring for the evidence)."""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import sys
import threading
import types as types_module
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
from mcp.server.stdio import stdio_server

from graphify_mesh.server import server
from graphify_mesh.server.stdio_guard import (
    MAX_LINE_BYTES,
    _CappedLineReader,
    capped_stdin,
    wire_stdout,
)


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


# --- pinning the write path `serve_stdio` actually sets up: ONE Python
# writer shared by the guard's error frames and the SDK's responses. Two
# independent `os.dup()`s of fd 1 left atomicity resting on POSIX PIPE_BUF,
# so a response over 4096 bytes could have an error frame spliced into it.


def test_installed_sdk_still_accepts_an_explicit_stdout():
    """The whole shared-writer fix depends on this argument existing. An
    `mcp` release that drops it must fail here, not corrupt stdout."""
    assert "stdout" in inspect.signature(stdio_server).parameters


def test_serve_stdio_shares_one_writer_between_guard_and_sdk(monkeypatch):
    """`capped_stdin(out_stream=...)` and `stdio_server(stdout=...)` must be
    handed the SAME object: that object's buffer lock is what serializes the
    two writers. Records what `serve_stdio` passes, without touching real
    stdio."""
    recorded: dict[str, object] = {}

    @asynccontextmanager
    async def fake_stdio_server(stdin=None, stdout=None):
        recorded["stdin"] = stdin
        recorded["stdout"] = stdout
        yield None, None

    class _FakeSdk:
        def create_initialization_options(self):
            return {}

        async def run(self, read_stream, write_stream, options):
            return None

    monkeypatch.setattr("mcp.server.stdio.stdio_server", fake_stdio_server)
    monkeypatch.setattr("graphify_mesh.server.sdk_app.build_sdk_server", lambda mesh: _FakeSdk())
    server.serve_stdio(mesh=None)

    guard_reader = recorded["stdin"].wrapped  # type: ignore[union-attr]
    sdk_writer = recorded["stdout"].wrapped  # type: ignore[union-attr]
    assert guard_reader._out_stream is sdk_writer


def test_wire_stdout_diverts_fd1_to_stderr_and_restores_it(tmp_path, monkeypatch):
    """Passing `stdout` explicitly skips the SDK's own fd-1 claim, so
    `wire_stdout` does that job: a stray write to fd 1 must land on stderr
    while the connection is up, and fd 1 must be restored afterwards."""
    out_path, err_path = tmp_path / "wire.out", tmp_path / "wire.err"
    out_file = open(out_path, "wb", buffering=0)  # noqa: SIM115
    err_file = open(err_path, "wb", buffering=0)  # noqa: SIM115
    saved_fd1 = os.dup(1)
    try:
        os.dup2(out_file.fileno(), 1)
        monkeypatch.setattr(sys, "stdout", os.fdopen(os.dup(1), "w", buffering=1))
        monkeypatch.setattr(sys, "stderr", os.fdopen(os.dup(err_file.fileno()), "w", buffering=1))

        with wire_stdout() as writer:
            writer.write("wire\n")
            writer.flush()
            os.write(1, b"stray\n")
        os.write(1, b"after\n")
    finally:
        os.dup2(saved_fd1, 1)
        os.close(saved_fd1)
        out_file.close()
        err_file.close()

    assert out_path.read_bytes() == b"wire\nafter\n"
    assert b"stray" in err_path.read_bytes()


class _BodyBoom(RuntimeError):
    """Raised by a test body to take `wire_stdout`'s exception exit path."""


def _buffered_wire_case(tmp_path, monkeypatch, body):
    """Runs `body(writer)` inside `wire_stdout()` with a block-buffered
    `sys.stdout` sitting on fd 1, the way a real process has it. Returns the
    bytes that reached the wire and the bytes that reached stderr."""
    out_path, err_path = tmp_path / "buf.out", tmp_path / "buf.err"
    out_file = open(out_path, "wb", buffering=0)  # noqa: SIM115
    err_file = open(err_path, "wb", buffering=0)  # noqa: SIM115
    saved_fd1 = os.dup(1)
    try:
        os.dup2(out_file.fileno(), 1)
        # fd 1 itself, block-buffered: a `print` stays in the Python buffer.
        monkeypatch.setattr(sys, "stdout", open(1, "w", closefd=False))  # noqa: SIM115
        monkeypatch.setattr(sys, "stderr", os.fdopen(os.dup(err_file.fileno()), "w", buffering=1))
        try:
            with contextlib.suppress(_BodyBoom), wire_stdout() as writer:
                body(writer)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
    finally:
        os.dup2(saved_fd1, 1)
        os.close(saved_fd1)
        out_file.close()
        err_file.close()
    return out_path.read_bytes(), err_path.read_bytes()


def test_buffered_print_during_the_connection_never_reaches_the_wire(tmp_path, monkeypatch):
    """`os.write(1, ...)` bypasses Python buffering, so it cannot catch the
    real leak: an ordinary buffered `print` sits in `sys.stdout`'s buffer
    while fd 1 points at stderr, and lands on the protocol pipe the moment
    fd 1 is restored unless the buffer is drained first."""

    def body(writer):
        writer.write("wire\n")
        writer.flush()
        print("buffered-leak")  # no flush: stays in sys.stdout's buffer

    wire, err = _buffered_wire_case(tmp_path, monkeypatch, body)

    assert b"buffered-leak" not in wire
    assert wire == b"wire\n"
    assert b"buffered-leak" in err


def test_buffered_print_never_reaches_the_wire_when_the_body_raises(tmp_path, monkeypatch):
    """Same drain on the exception exit path, which is the one a crashing
    library takes."""

    def body(writer):
        writer.write("wire\n")
        writer.flush()
        print("buffered-leak")
        raise _BodyBoom("boom")

    wire, err = _buffered_wire_case(tmp_path, monkeypatch, body)

    assert b"buffered-leak" not in wire
    assert wire == b"wire\n"
    assert b"buffered-leak" in err


def test_error_write_and_sdk_write_do_not_interleave_on_the_shared_writer():
    """Concurrency regression for the shared writer: one thread drives the
    guard's error-write path (`_CappedLineReader._write_error`), the other
    writes response frames the way `stdio_server`'s `stdout_writer` does,
    both through the ONE writer `serve_stdio` hands to both. The response
    frames are deliberately far larger than `PIPE_BUF`, the bound the
    previous two-descriptor arrangement relied on. Every line received must
    be a complete, parseable JSON object."""
    read_fd, write_fd = os.pipe()
    shared_writer = os.fdopen(write_fd, "w", buffering=1)
    reader = _CappedLineReader(stream=io.StringIO(""), out_stream=shared_writer)

    n_per_thread = 200
    big_payload = "y" * 60_000  # >> PIPE_BUF (4096 on Linux)
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

    def write_responses() -> None:
        for i in range(n_per_thread):
            shared_writer.write(
                json.dumps({"jsonrpc": "2.0", "id": i, "result": {"blob": big_payload}}) + "\n"
            )
            shared_writer.flush()

    reader_thread = threading.Thread(target=drain)
    error_thread = threading.Thread(target=write_errors)
    response_thread = threading.Thread(target=write_responses)
    reader_thread.start()
    error_thread.start()
    response_thread.start()
    error_thread.join(timeout=30)
    response_thread.join(timeout=30)
    shared_writer.close()  # signals EOF to the drain thread
    reader_thread.join(timeout=30)

    assert len(received) == n_per_thread * 2
    for line in received:
        parsed = json.loads(line)  # raises on a fragment or a merge of two lines
        assert isinstance(parsed, dict)


# --- the cap counts BYTES, not decoded characters -------------------------


class _RecordingBytes(io.BytesIO):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.sizes: list[int] = []

    def readline(self, size: int = -1) -> bytes:  # type: ignore[override]
        self.sizes.append(size)
        return super().readline(size)


def test_multibyte_oversized_line_is_bounded_on_bytes_and_answered(monkeypatch):
    """A line of 4-byte UTF-8 characters whose CHARACTER count is under the
    cap but whose BYTE count is over it. Bounding decoded characters let this
    buffer four times the documented limit. It must be dropped, every read
    must stay within the byte bound, and the client must get `-32600` instead
    of waiting for its own timeout (the HTTP transport answers 413 here)."""
    char_count = MAX_LINE_BYTES // 4 + 10
    assert char_count < MAX_LINE_BYTES  # under the cap if characters were counted
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    data = ("\U0001f600" * char_count + "\n" + payload + "\n").encode("utf-8")
    assert len(data) > MAX_LINE_BYTES

    byte_stream = _RecordingBytes(data)
    monkeypatch.setattr(sys, "stdin", types_module.SimpleNamespace(buffer=byte_stream))
    out = io.StringIO()

    lines = list(_drain(capped_stdin(out_stream=out)))

    assert len(lines) == 1
    assert json.loads(lines[0])["method"] == "ping"
    assert all(0 < size <= MAX_LINE_BYTES + 1 for size in byte_stream.sizes), byte_stream.sizes
    response = json.loads(out.getvalue().strip())
    assert response["id"] is None
    assert response["error"]["code"] == -32600
    assert str(MAX_LINE_BYTES) in response["error"]["message"]


def test_oversized_line_answer_never_quotes_the_offending_frame(monkeypatch):
    byte_stream = io.BytesIO(b"s3cret-marker" * (MAX_LINE_BYTES // 10) + b"\n")
    monkeypatch.setattr(sys, "stdin", types_module.SimpleNamespace(buffer=byte_stream))
    out = io.StringIO()
    list(_drain(capped_stdin(out_stream=out)))
    assert "s3cret-marker" not in out.getvalue()


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
