"""JSON-RPC frame limits and envelope validation shared by both transports.

Two things must read the same way whichever transport a client speaks:

1. The incoming-message size bound. `MAX_MESSAGE_BYTES` is the single number
   both transports state — `stdio_guard.MAX_LINE_BYTES` per line, and the
   HTTP guard plus the SDK session manager per request body.
2. The error a malformed frame gets back. The SDK validates envelopes with
   pydantic and puts `str(ValidationError)` — which quotes the offending
   input — in the client-visible message, so both transports check the
   envelope themselves first and answer with a fixed generic message,
   logging the detail instead.

The checks here are deliberately narrower than the SDK's model validation:
they reject only what no legal JSON-RPC 2.0 frame can be, so a frame that
passes still reaches the SDK for full validation. A client-to-server
*response* (a `result`/`error` frame answering a server-initiated request,
which MCP uses for sampling and roots) carries no `method` and must pass.
"""

from __future__ import annotations

import json
from typing import Final

# Hard per-message size cap for both transports. The shared HTTP daemon is one
# process serving every local agent, so its memory is a machine-wide resource,
# and no legitimate JSON-RPC frame for these six tools comes close to 4 MiB —
# a lower ceiling is worth more here than matching stdio's earlier 10 MB.
# Over-limit input is rejected without ever being buffered whole.
MAX_MESSAGE_BYTES: Final = 4 * 1024 * 1024

PARSE_ERROR: Final = (-32700, "parse error")
NOT_A_SINGLE_OBJECT: Final = (
    -32600,
    "invalid request: expected a single JSON-RPC object (batch requests are not supported)",
)
INVALID_REQUEST: Final = (-32600, "invalid request")
INVALID_PARAMS: Final = (-32602, "invalid params")


def frame_id(parsed: object) -> str | int | None:
    """The request id to correlate an error response with, or `None` when the
    frame carries no usable id. Only the two types JSON-RPC 2.0 allows count:
    an id of any other shape is itself the defect being reported."""
    if not isinstance(parsed, dict):
        return None
    candidate = parsed.get("id")
    if isinstance(candidate, bool):  # `True` is an int in Python, not a JSON-RPC id
        return None
    if isinstance(candidate, str | int):
        return candidate
    return None


def envelope_error(parsed: object) -> tuple[int, str] | None:
    """`(code, message)` for a frame the SDK would reject with a detail-
    carrying validation error, or `None` when the frame is shaped like a
    legal JSON-RPC 2.0 message."""
    if not isinstance(parsed, dict):
        return NOT_A_SINGLE_OBJECT
    if parsed.get("jsonrpc") != "2.0":
        return INVALID_REQUEST

    raw_id = parsed.get("id")
    if "id" in parsed and (isinstance(raw_id, bool) or not isinstance(raw_id, str | int | None)):
        return INVALID_REQUEST

    if "method" not in parsed:
        # A response frame (result/error) answering a server-initiated
        # request is the one legal shape with no method.
        if "result" in parsed or "error" in parsed:
            return None
        return INVALID_REQUEST
    if not isinstance(parsed["method"], str) or not parsed["method"]:
        return INVALID_REQUEST

    params = parsed.get("params")
    if "params" in parsed and params is not None and not isinstance(params, dict):
        # JSON-RPC 2.0 also allows positional array params; MCP does not use
        # them, and the SDK models type `params` as `dict | None`, so an array
        # here is a params-shape error rather than an invalid envelope.
        # Explicit `null` is exempt: it is what that `| None` accepts, and a
        # client whose serializer emits it has always worked.
        return INVALID_PARAMS

    return None


def error_frame_text(code: int, message: str, request_id: str | int | None = None) -> str:
    return json.dumps(
        {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
    )
