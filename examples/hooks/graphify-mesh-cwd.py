#!/usr/bin/env python3
"""Claude Code PreToolUse hook: fills the graphify-mesh `cwd` tool argument.

The graphify-mesh HTTP daemon serves one process to every client session, so
it has no per-session directory of its own. `scope: "current"` on `search`
and `context_pack` resolves the caller's directory from the `cwd` argument;
without it the daemon falls back to its own process directory, which matches
no registered repository and fails closed. This hook fills `cwd` from the
session's project directory before the call reaches the server.

It fills `cwd` only when the key is absent, and never touches a `cwd` the
model already supplied — not even an empty or otherwise invalid one, which
the server must reject rather than have the hook silently redirect. One
session can move between projects over its lifetime, so a value pinned once
by a hook would keep answering for the wrong project after the model moves
on.

Claude-Code-specific: this relies on Claude Code's PreToolUse hook contract
(stdin JSON in, `hookSpecificOutput.updatedInput` JSON out). Another MCP
client needs its own equivalent, or callers must pass `cwd` themselves.

Install: see docs/setup.md, "PreToolUse hook: filling cwd automatically".
"""

import json
import os
import sys

TOOL_PREFIX = "mcp__graphify-mesh__"


def main() -> None:
    try:
        payload = json.load(sys.stdin)
        tool_name = payload["tool_name"]
        if not isinstance(tool_name, str) or not tool_name.startswith(TOOL_PREFIX):
            return

        tool_input = payload.get("tool_input")
        if not isinstance(tool_input, dict):
            return

        # Only an ABSENT key may be filled. A `cwd` that is present but
        # invalid ("", whitespace, null, a number) is the server's to reject:
        # replacing it with the session directory turns a call that should
        # have failed validation into a call that succeeds against a
        # different scope.
        if "cwd" in tool_input:
            return

        project_dir = payload.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
        if not isinstance(project_dir, str) or not project_dir:
            return
        project_dir = os.path.abspath(project_dir)

        updated_input = dict(tool_input)
        updated_input["cwd"] = project_dir

        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "updatedInput": updated_input,
                    }
                }
            )
        )
    except Exception:
        # Fail open: a hook bug must never block a tool call.
        return


if __name__ == "__main__":
    main()
