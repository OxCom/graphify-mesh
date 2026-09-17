"""Size-capped, frame-validating line source for the SDK's stdio transport
(`mcp.server.stdio.stdio_server`).

Two protections the retired `protocol.py` provided do not survive the move
onto the SDK's own stdin reader, so this module restores them at the read
boundary, before a line ever reaches the SDK:

1. That reader is `async for line in stdin` over an `anyio.AsyncFile[str]`,
   whose `readline()` has no size bound (confirmed by reading the installed
   `mcp.server.stdio` source: `AsyncFile.__aiter__` loops on
   `self.readline()`, which calls the wrapped file's `readline()` with no
   arguments) — a single giant or newline-less line from a misbehaving
   client would buffer whole in memory. `_CappedLineReader` carries forward
   the drain logic that used to live in `protocol.py`
   (`_drain_line`/`_read_lines`).
2. A line that is not parseable JSON, or that parses to something other than
   a single JSON object (including a batch array), is fed by the SDK's own
   `stdin_reader` into the session as a bare `Exception` — the session logs
   it and emits a generic `notifications/message`, never the JSON-RPC
   `-32700`/`-32600` error the spec requires and `protocol.py` used to
   return. `_CappedLineReader.readline` intercepts both cases itself: it
   writes the coded error response directly and does not forward the line,
   so the SDK never sees it.
3. A line that IS a JSON object but not a legal JSON-RPC 2.0 envelope
   (`{"jsonrpc":"2.0","id":7,"method":"tools/call","params":[]}`) reaches the
   SDK's pydantic validation, which raises rather than answering request `7`,
   so the client waits for its own timeout. `frames.envelope_error` decides
   these, and the answer is `-32602` for a params-shape error, `-32600`
   otherwise — the codes `protocol.py` returned for the same input.

An unparseable, non-object or unusably-identified frame is answered with
`id: null`: there is no id to read, or none JSON-RPC 2.0 allows. A frame that
carries a legal id (case 3) is answered with that id, so the client can
correlate the error with its pending request instead of waiting.

Both error responses and the SDK's own responses (via
`mcp.server.stdio.stdio_server`'s default stdout, a fresh `TextIOWrapper`
over `sys.stdout.buffer`) end up writing to the *same* underlying
`sys.stdout.buffer` `BufferedWriter` object. CPython's `io.BufferedWriter`
serializes concurrent `write()` calls on one buffer with an internal lock
(each call's bytes are appended as a whole under that lock), so two writers
sharing the same buffer can never interleave *within* a line. This module
deliberately does not add a second, unrelated lock of its own — instead the
two things that guarantee are conditional on (CPython's `BufferedWriter`
locking its `write()` calls; the SDK's stdout and `capped_stdin`'s default
`out_stream` wrapping the identical `sys.stdout.buffer`) are each pinned by
a test in `tests/server/test_stdio_transport.py`, so a future CPython or
`mcp` release that breaks either one fails a test instead of silently
corrupting stdout:
`test_sdk_stdout_shares_our_buffer_or_frames_can_interleave` (the SDK still
shares our buffer) and
`test_error_write_and_sdk_style_write_do_not_interleave_on_shared_buffer`
(concurrent writers on that shared buffer still never interleave). Read
those two tests before touching this file's write path or `serve_stdio`'s
call to `stdio_server`.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import IO, cast

import anyio

from graphify_mesh.server.frames import (
    MAX_MESSAGE_BYTES,
    NOT_A_SINGLE_OBJECT,
    PARSE_ERROR,
    envelope_error,
    error_frame_text,
    frame_id,
)

log = logging.getLogger("graphify_mesh.server.stdio_guard")

# Hard per-line size cap. Without it, `readline()` buffers an entire line in
# memory. The number lives in `frames.MAX_MESSAGE_BYTES` so the HTTP
# transport states the same one.
MAX_LINE_BYTES = MAX_MESSAGE_BYTES


def _error_frame(code: int, message: str, request_id: str | int | None = None) -> str:
    return error_frame_text(code, message, request_id)


class _CappedLineReader:
    """Sync file-like object exposing only a no-argument `readline()`, the
    one method `anyio.AsyncFile.readline()` calls on the object it wraps.
    Oversized lines are drained in bounded chunks and dropped; blank lines
    are skipped; an unparseable or non-object/batch line gets its coded
    JSON-RPC error written directly to `out_stream` and is not returned —
    only complete, valid single-object candidate frames reach the SDK."""

    def __init__(self, stream: IO[str], out_stream: IO[str]) -> None:
        self._stream = stream
        self._out_stream = out_stream

    def _drain_rest(self) -> None:
        """Consume (and discard) the remainder of an oversized line in
        bounded chunks, stopping at the next newline or EOF. Never
        accumulates the data."""
        while True:
            chunk = self._stream.readline(MAX_LINE_BYTES + 1)
            if not chunk:
                return
            if chunk.endswith("\n"):
                return

    def _write_error(self, code: int, message: str, request_id: str | int | None = None) -> None:
        self._out_stream.write(_error_frame(code, message, request_id) + "\n")
        self._out_stream.flush()

    def readline(self, size: int = -1) -> str:  # noqa: ARG002 - AsyncFile calls with no args
        while True:
            raw_line = self._stream.readline(MAX_LINE_BYTES + 1)
            if not raw_line:
                return ""  # EOF: clean stop, matches the old serve()'s WS6 contract
            if len(raw_line) > MAX_LINE_BYTES and not raw_line.endswith("\n"):
                self._drain_rest()
                log.warning("dropped oversized stdin line (> %d bytes)", MAX_LINE_BYTES)
                continue
            if not raw_line.strip():
                continue
            try:
                parsed = json.loads(raw_line)
            except json.JSONDecodeError:
                self._write_error(*PARSE_ERROR)
                log.warning("dropped unparseable stdin line")
                continue
            if not isinstance(parsed, dict):
                self._write_error(*NOT_A_SINGLE_OBJECT)
                log.warning("dropped non-object/batch stdin line")
                continue
            # A legal-JSON frame with an illegal envelope used to reach the
            # SDK, whose pydantic validation raises instead of answering the
            # request — the client then waits for its own timeout. Answer it
            # here, correlated with the request id when the frame carries a
            # usable one.
            coded = envelope_error(parsed)
            if coded is not None:
                self._write_error(*coded, frame_id(parsed))
                log.warning("dropped malformed-envelope stdin line (code %d)", coded[0])
                continue
            return raw_line


def capped_stdin(
    stream: IO[str] | None = None, out_stream: IO[str] | None = None
) -> anyio.AsyncFile[str]:
    """Async-iterable text stream for `stdio_server(stdin=...)`: yields
    complete, valid-shaped lines; drains and drops any line over
    `MAX_LINE_BYTES` with a stderr warning instead of buffering it; answers
    an unparseable or non-object/batch line with the matching JSON-RPC error
    on `out_stream` (default `sys.stdout`, the same stream the SDK writes
    its own responses to) instead of forwarding it; skips blank lines; stops
    at EOF."""
    reader = _CappedLineReader(
        stream if stream is not None else sys.stdin,
        out_stream if out_stream is not None else sys.stdout,
    )
    # `_CappedLineReader` only implements the one method `anyio.AsyncFile`
    # actually calls (`readline()`) — not the full `IO[str]` surface — so the
    # cast documents an intentional, narrower structural fit rather than a
    # type error.
    return anyio.wrap_file(cast("IO[str]", reader))
