"""Writer-preferring read/write lock.

The shared daemon answers many reads against one loaded generation while a
sync publish occasionally replaces it. Reads must not serialize behind each
other, and the swap must not run while a read is in flight. Writer-preferring
matters because reads are frequent: a reader-preferring lock would let a
steady stream of queries postpone the reload indefinitely, and the daemon
would keep serving a generation the pipeline already replaced.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class ReadWriteLock:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        with self._cond:
            while self._writer or self._waiting_writers:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        with self._cond:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers:
                    self._cond.wait()
                self._writer = True
            finally:
                self._waiting_writers -= 1
                # If `wait()` itself raised before granting the writer (e.g. an
                # injected exception, or interpreter shutdown), threads already
                # blocked in `wait()` would otherwise never be woken by this
                # writer's exit — nothing else releases them. Notifying here
                # too is a no-op cost on the normal path (readers/writers just
                # re-check their own condition and wait again if not runnable).
                self._cond.notify_all()
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()
