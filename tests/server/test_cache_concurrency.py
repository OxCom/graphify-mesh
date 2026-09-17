from __future__ import annotations

import threading

from graphify_mesh.server import scope as scope_mod


def test_registry_is_parsed_once_under_concurrent_readers(registry_file, monkeypatch):
    parses: list[str] = []
    original = scope_mod.json.loads

    def counting_loads(text, *args, **kwargs):
        parses.append("x")
        return original(text, *args, **kwargs)

    monkeypatch.setattr(scope_mod.json, "loads", counting_loads)
    scope_mod._registry_cache.clear()

    results: list[int] = []

    def hit():
        results.append(len(scope_mod.load_registry_entries(registry_file)))

    threads = [threading.Thread(target=hit) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert results and all(n == results[0] for n in results)
    assert len(parses) == 1


def test_similar_index_is_built_once_under_concurrent_readers(generation_fixture):
    import time

    from graphify_mesh.server import similar as similar_mod

    builds: list[str] = []

    def counting_build(generation):
        builds.append("x")
        time.sleep(0.001)  # Increase race window
        return {}

    getter = similar_mod._per_generation_cache(counting_build)

    def hit():
        getter(generation_fixture)

    threads = [threading.Thread(target=hit) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(builds) == 1
