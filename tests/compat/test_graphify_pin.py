"""The graphifyy dependency stays an exact pin.

A new graphify version makes the next sync force-rebuild every repo with
--producer-changed (pipeline `_producer_upgraded`). With a range, a routine
`pipx install --force` moved 0.9.104 to 0.9.112 on 2026-11-19 and rebuilt all
25 repos unasked.
"""

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _graphifyy_spec() -> str:
    deps = tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
    matches = [d for d in deps if d.replace(" ", "").startswith("graphifyy")]
    assert len(matches) == 1, matches
    return matches[0].replace(" ", "")


def test_graphifyy_is_pinned_exactly():
    spec = _graphifyy_spec()
    _, sep, version = spec.partition("==")
    assert sep == "==", f"graphifyy must be pinned with ==, got {spec!r}"
    assert version and not any(op in version for op in ("<", ">", ",", "*", "~", "!")), spec
