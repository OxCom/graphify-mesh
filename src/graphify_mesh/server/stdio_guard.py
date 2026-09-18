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

Both error responses and the SDK's own responses reach the wire through ONE
shared Python writer. `serve_stdio` builds it once with `wire_stdout()` and
hands the same object to `capped_stdin(out_stream=...)` and to
`stdio_server(stdout=...)`, so every frame from either writer goes through
that object's buffer and its lock serializes concurrent writes — a response
of any size can no longer have an error frame spliced into its middle. The
earlier arrangement gave each side its own `os.dup()` of fd 1, which left
atomicity resting on POSIX `PIPE_BUF` (4096 bytes on Linux) and so held only
for frames smaller than that.

`wire_stdout()` also keeps the stray-write protection the SDK gives up when
it is handed an explicit `stdout`: it points fd 1 at stderr for the
connection's duration, so a library that prints to the real stdout cannot
corrupt the protocol stream, and restores fd 1 on exit.
`test_serve_stdio_shares_one_writer_between_guard_and_sdk` and
`test_error_write_and_sdk_write_do_not_interleave_on_the_shared_writer` in
`tests/server/test_stdio_transport.py` pin both halves, so an `mcp` release
that drops the `stdout` argument fails a test instead of silently corrupting
stdout. Read them before touching this file's write path or `serve_stdio`'s
call to `stdio_server`.

Line size is capped on BYTES, not decoded characters. On the real stdin path
the reader reads `sys.stdin.buffer` and decodes after the bound, so a line of
4-byte UTF-8 characters can never buffer four times the cap. A caller-supplied
text `stream` (what the unit tests inject) has no byte view, so the cap there
counts characters — the bound that stream can actually enforce.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
from collections.abc import Iterator
from typing import IO, Final, cast

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

# Answer for a line over the cap. Silently dropping it left the client waiting
# for its own timeout while the HTTP transport answered 413 for the same input.
# The message states the limit and never quotes the offending frame.
LINE_TOO_LARGE: Final = (
    -32600,
    f"invalid request: message exceeds the {MAX_LINE_BYTES} byte limit",
)


def _error_frame(code: int, message: str, request_id: str | int | None = None) -> str:
    return error_frame_text(code, message, request_id)


class _CappedLineReader:
    """Sync file-like object exposing only a no-argument `readline()`, the
    one method `anyio.AsyncFile.readline()` calls on the object it wraps.
    Oversized lines are drained in bounded chunks and dropped; blank lines
    are skipped; an unparseable or non-object/batch line gets its coded
    JSON-RPC error written directly to `out_stream` and is not returned —
    only complete, valid single-object candidate frames reach the SDK."""

    def __init__(
        self,
        stream: IO[str] | None = None,
        out_stream: IO[str] | None = None,
        byte_stream: IO[bytes] | None = None,
    ) -> None:
        if (stream is None) == (byte_stream is None):
            raise ValueError("exactly one of stream= / byte_stream= must be given")
        self._stream = stream
        self._byte_stream = byte_stream
        self._out_stream = out_stream if out_stream is not None else sys.stdout

    def _read_bounded(self) -> tuple[str, int]:
        """Next line plus its size in the unit the cap is counted in. The
        byte path reads `MAX_LINE_BYTES + 1` BYTES and decodes afterwards, so
        a multibyte line cannot buffer more than the cap; a decode error can
        only come from a line that is already over the bound and about to be
        dropped, hence `errors="replace"`. The text path (an injected
        in-memory stream) has no byte view and counts characters."""
        if self._byte_stream is not None:
            raw = self._byte_stream.readline(MAX_LINE_BYTES + 1)
            return raw.decode("utf-8", errors="replace"), len(raw)
        text_stream = cast("IO[str]", self._stream)
        raw_text = text_stream.readline(MAX_LINE_BYTES + 1)
        return raw_text, len(raw_text)

    def _drain_rest(self) -> None:
        """Consume (and discard) the remainder of an oversized line in
        bounded chunks, stopping at the next newline or EOF. Never
        accumulates the data."""
        while True:
            chunk, _size = self._read_bounded()
            if not chunk:
                return
            if chunk.endswith("\n"):
                return

    def _write_error(self, code: int, message: str, request_id: str | int | None = None) -> None:
        self._out_stream.write(_error_frame(code, message, request_id) + "\n")
        self._out_stream.flush()

    def readline(self, size: int = -1) -> str:  # noqa: ARG002 - AsyncFile calls with no args
        while True:
            raw_line, size = self._read_bounded()
            if not raw_line:
                return ""  # EOF: clean stop, matches the old serve()'s WS6 contract
            if size > MAX_LINE_BYTES and not raw_line.endswith("\n"):
                self._drain_rest()
                self._write_error(*LINE_TOO_LARGE)
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


def _private_wire_stdout() -> IO[str]:
    """A private duplicate of the real stdout descriptor, taken now — before
    `mcp.server.stdio.stdio_server()` claims fd 1 for its own writes and
    `dup2()`s it to stderr for the connection's duration (its stray-write
    guard: PR #3117; confirmed by reading `_claim_fd`/`_open_stdout_diversion`
    in the installed `mcp.server.stdio`) — so a coded error frame written to
    `sys.stdout` at that later point would land on stderr, never the wire.
    Only the standalone default for a `capped_stdin()` call that supplies no
    `out_stream`: `serve_stdio` passes the shared writer from `wire_stdout()`
    instead, which is what makes the guard's frames and the SDK's share one
    buffer. Falls back to `sys.stdout` itself when it has no real OS
    descriptor to duplicate (e.g. a test's in-memory stand-in).
    """
    try:
        fd = os.dup(sys.stdout.fileno())
    except (AttributeError, OSError, ValueError):
        return sys.stdout
    return os.fdopen(fd, "w", buffering=1)


@contextlib.contextmanager
def wire_stdout() -> Iterator[IO[str]]:
    """The one writer for the whole stdio connection: both this module's
    error frames and the SDK's responses write through it, so its buffer
    lock — not POSIX `PIPE_BUF`, which only covers 4096 bytes — is what
    keeps a large response and an error frame from interleaving.

    Handing `stdio_server` an explicit `stdout` also skips the SDK's own
    fd-1 claim, so this does that claim's job instead: fd 1 points at stderr
    while the connection is up, which keeps a stray `print` from any library
    off the protocol stream, and is restored on exit. The diversion only
    holds for bytes that reach fd 1 while it is in place, so `sys.stdout` is
    flushed on both edges: on entry, and again in the `finally` before fd 1
    is restored. Skipping the exit flush would leave a buffered `print` from
    the connection sitting in `sys.stdout` until the restored fd 1 drains it
    onto the wire. Yields `sys.stdout` unchanged when there is no real
    descriptor to duplicate (an in-memory stand-in in tests), since there is
    nothing to divert there either.
    """
    try:
        wire_fd = os.dup(sys.stdout.fileno())
    except (AttributeError, OSError, ValueError):
        yield sys.stdout
        return

    writer = os.fdopen(wire_fd, "w", buffering=1)
    diverted = False
    try:
        try:
            sys.stdout.flush()
            os.dup2(sys.stderr.fileno(), 1)
            diverted = True
        except (AttributeError, OSError, ValueError):
            log.warning("could not divert fd 1 to stderr; stray stdout writes reach the wire")
        yield writer
    finally:
        try:
            writer.flush()
        except (OSError, ValueError):
            pass
        if diverted:
            # sys.stderr writes to fd 2 and is unaffected by the restore below.
            with contextlib.suppress(AttributeError, OSError, ValueError):
                sys.stdout.flush()
            with contextlib.suppress(OSError):
                os.dup2(wire_fd, 1)
        writer.close()


def capped_stdin(
    stream: IO[str] | None = None, out_stream: IO[str] | None = None
) -> anyio.AsyncFile[str]:
    """Async-iterable text stream for `stdio_server(stdin=...)`: yields
    complete, valid-shaped lines; drains any line over `MAX_LINE_BYTES` and
    answers it with `-32600` instead of buffering it; answers an unparseable
    or non-object/batch line with the matching JSON-RPC error on `out_stream`
    instead of forwarding it; skips blank lines; stops at EOF.

    With no `stream`, lines come from `sys.stdin.buffer` and the cap is
    counted on bytes; an injected text `stream` is read as text and its cap
    counts characters. `out_stream` defaults to a private duplicate of the
    real stdout descriptor (`_private_wire_stdout`); `serve_stdio` passes the
    connection's shared writer instead."""
    out = out_stream if out_stream is not None else _private_wire_stdout()
    if stream is not None:
        reader = _CappedLineReader(stream=stream, out_stream=out)
    else:
        binary = getattr(sys.stdin, "buffer", None)
        if binary is not None:
            reader = _CappedLineReader(byte_stream=binary, out_stream=out)
        else:
            # No byte view (a text stand-in installed as sys.stdin): the cap
            # falls back to characters, which is all such a stream can bound.
            reader = _CappedLineReader(stream=sys.stdin, out_stream=out)
    # `_CappedLineReader` only implements the one method `anyio.AsyncFile`
    # actually calls (`readline()`) — not the full `IO[str]` surface — so the
    # cast documents an intentional, narrower structural fit rather than a
    # type error.
    return anyio.wrap_file(cast("IO[str]", reader))
