from __future__ import annotations

import threading

from graphify_mesh.server.rwlock import ReadWriteLock


def test_readers_run_in_parallel():
    lock = ReadWriteLock()
    both_inside = threading.Barrier(2, timeout=5)

    def reader():
        with lock.read():
            both_inside.wait()  # raises BrokenBarrierError if readers serialize

    threads = [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive()


def test_writer_excludes_readers():
    lock = ReadWriteLock()
    order: list[str] = []
    writer_holding = threading.Event()
    release_writer = threading.Event()

    def writer():
        with lock.write():
            order.append("writer-in")
            writer_holding.set()
            release_writer.wait(timeout=5)
            order.append("writer-out")

    def reader():
        writer_holding.wait(timeout=5)
        with lock.read():
            order.append("reader-in")

    w = threading.Thread(target=writer)
    r = threading.Thread(target=reader)
    w.start()
    r.start()
    writer_holding.wait(timeout=5)
    release_writer.set()
    w.join(timeout=5)
    r.join(timeout=5)
    assert order == ["writer-in", "writer-out", "reader-in"]


def test_reload_happens_once_under_concurrent_readers(store_with_two_generations):
    store, publish_next = store_with_two_generations
    store.ensure_fresh()
    publish_next()

    reload_calls: list[str] = []
    original = store._try_reload

    def counting_reload(target, mtime):
        reload_calls.append(target)
        return original(target, mtime)

    store._try_reload = counting_reload  # type: ignore[method-assign]

    errors: list[BaseException] = []

    def hit():
        try:
            assert store.generation is not None
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=hit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert errors == []
    assert len(reload_calls) == 1


def test_readers_never_see_a_half_built_generation(store_with_two_generations):
    store, publish_next = store_with_two_generations
    store.ensure_fresh()
    publish_next()

    seen: list[int] = []

    def hit():
        gen = store.generation
        # build_indexes() ran before the generation was published to readers
        seen.append(len(gen.node_by_id))

    threads = [threading.Thread(target=hit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert seen and all(count == seen[0] for count in seen)
    assert seen[0] > 0


def test_a_stale_capture_never_reloads_over_a_newer_generation(store_with_two_generations):
    """The double check under the write lock has to ask the filesystem
    again, not compare the signature captured before the wait.

    Interleaving, forced with events rather than hoped for:
    thread A captures gen-1 and stops before it can take any lock; gen-2 is
    published; thread B loads gen-2; A resumes. A's captured signature now
    differs from what is loaded, and a check against the capture concludes
    that a reload of gen-1 is due — the store ends up on the OLDER
    generation while gen-2 is current on disk.
    """
    store, publish_next = store_with_two_generations

    a_captured = threading.Event()
    b_loaded = threading.Event()
    original_stat = store._stat_signature
    a_calls = {"n": 0}

    def gated_stat():
        if threading.current_thread().name == "A":
            a_calls["n"] += 1
            if a_calls["n"] == 1:
                captured = original_stat()
                a_captured.set()
                assert b_loaded.wait(timeout=10)
                return captured
        return original_stat()

    store._stat_signature = gated_stat  # type: ignore[method-assign]

    reloads: list[str] = []
    original_reload = store._try_reload

    def recording_reload(target, mtime):
        reloads.append(target)
        return original_reload(target, mtime)

    store._try_reload = recording_reload  # type: ignore[method-assign]

    ids: dict[str, str] = {}

    def thread_a():
        ids["a"] = store.generation.generation_id

    def thread_b():
        assert a_captured.wait(timeout=10)
        publish_next()
        ids["b"] = store.generation.generation_id
        b_loaded.set()

    a = threading.Thread(target=thread_a, name="A")
    b = threading.Thread(target=thread_b, name="B")
    a.start()
    b.start()
    for t in (a, b):
        t.join(timeout=15)
        assert not t.is_alive()

    assert ids["b"] == "gen-2"
    assert store.generation.generation_id == "gen-2"
    assert not any(target.endswith("gen-1") for target in reloads[1:]), (
        f"reloaded a stale generation after gen-2 was loaded: {reloads}"
    )
