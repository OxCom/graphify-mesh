"""`ReadWriteLock.write()`'s gap: if `self._cond.wait()` itself raises before
the writer is granted, the `finally` decrements `_waiting_writers` but must
also `notify_all()` — otherwise a reader already parked in `wait()` (blocked
because `_waiting_writers > 0`) is never woken by that event, since nothing
else will ever notify it.

This drives that exact path: a writer genuinely parks in a real
`Condition.wait()` (mutex released, like any real contended writer), a
reader parks behind it the same way, then the writer's own parked `wait()`
call is made to raise (rather than return normally) via a targeted
`notify(n=1)` that wakes ONLY the writer (FIFO: it parked first) — never the
reader. The reader can only make progress if `write()`'s own cleanup path
notifies it.
"""

from __future__ import annotations

import threading
import time

from graphify_mesh.server.rwlock import ReadWriteLock


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_writer_wait_exception_still_wakes_a_blocked_reader():
    lock = ReadWriteLock()

    # A real reader holds the lock so the writer below has something to
    # wait on (`while self._writer or self._readers`).
    reader1_holding = threading.Event()
    release_reader1 = threading.Event()

    def reader1_body() -> None:
        with lock.read():
            reader1_holding.set()
            release_reader1.wait(timeout=5)

    reader1 = threading.Thread(target=reader1_body)
    reader1.start()
    assert reader1_holding.wait(timeout=2), "reader1 never acquired the read lock"

    # Make ONLY the writer thread's `cond.wait()` raise, and only after it
    # has genuinely parked (mutex released) like a real contended writer —
    # not an immediate raise, which would never release the mutex at all.
    original_wait = lock._cond.wait
    raise_for: dict[str, threading.Thread | None] = {"thread": None}

    def patched_wait(timeout=None):
        if threading.current_thread() is raise_for["thread"]:
            original_wait(timeout)
            raise RuntimeError("injected: cond.wait failed")
        return original_wait(timeout)

    lock._cond.wait = patched_wait  # type: ignore[method-assign]

    writer_error: list[BaseException] = []

    def writer_body() -> None:
        try:
            with lock.write():
                pass  # pragma: no cover - never granted in this test
        except RuntimeError as exc:
            writer_error.append(exc)

    writer = threading.Thread(target=writer_body)
    raise_for["thread"] = writer
    writer.start()

    # Writer must be genuinely parked (released the mutex, queued in the
    # condition's own waiter list) before the reader starts, so the two are
    # concurrently blocked exactly like real contention.
    assert _wait_until(lambda: len(lock._cond._waiters) == 1), "writer never parked"

    reader2_progressed = threading.Event()

    def reader2_body() -> None:
        with lock.read():
            reader2_progressed.set()

    reader2 = threading.Thread(target=reader2_body)
    reader2.start()
    assert _wait_until(lambda: len(lock._cond._waiters) == 2), "reader2 never parked"

    # Wake ONLY the writer (FIFO: it parked first) — never the reader. If
    # the reader ever makes progress from here, it can only be because
    # `write()`'s own exception-cleanup path notified it.
    with lock._cond:
        lock._cond.notify(n=1)

    writer.join(timeout=2)
    assert not writer.is_alive(), "writer thread never finished"
    assert len(writer_error) == 1
    assert lock._waiting_writers == 0

    assert reader2_progressed.wait(timeout=2), (
        "reader2 stayed parked after the writer's wait() raised: "
        "write()'s cleanup path did not wake it"
    )
    reader2.join(timeout=2)
    assert not reader2.is_alive()

    release_reader1.set()
    reader1.join(timeout=2)
