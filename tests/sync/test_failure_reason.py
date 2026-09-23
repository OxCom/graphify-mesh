"""The failure reason must carry the END of stderr and persist the whole of it."""

from pathlib import Path

from graphify_mesh.sync.sync_project import (
    FAILURE_LOG_NAME,
    FAILURE_REASON_TAIL,
    _failure_reason,
)


def test_short_stderr_is_kept_whole(tmp_path: Path):
    reason = _failure_reason(1, "  boom  ", tmp_path)
    assert reason.startswith("exit=1: boom")
    assert (tmp_path / FAILURE_LOG_NAME).read_text(encoding="utf-8") == "boom"


def test_long_stderr_keeps_the_tail_not_the_head(tmp_path: Path):
    stderr = "W" * 5000 + "\nValueError: the real cause"
    reason = _failure_reason(1, stderr, tmp_path)
    assert "ValueError: the real cause" in reason
    assert reason.count("W") < 5000
    assert (tmp_path / FAILURE_LOG_NAME).read_text(encoding="utf-8") == stderr
    assert len(reason) < FAILURE_REASON_TAIL + 200


def test_empty_stderr_says_so(tmp_path: Path):
    assert _failure_reason(2, "   ", tmp_path) == "exit=2: no stderr"
    assert not (tmp_path / FAILURE_LOG_NAME).exists()


def test_unwritable_collection_still_reports(tmp_path: Path):
    missing = tmp_path / "gone"
    reason = _failure_reason(1, "ValueError: cause", missing)
    assert "ValueError: cause" in reason
    assert "full stderr" not in reason
