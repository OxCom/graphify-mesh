"""Stage RSS tracking: reads /proc/self/status (Linux-only project; the
sync service runs under systemd on this box).

Attribution logic is unit-tested against a monkeypatched `rss.snapshot` with
scripted values, so it is deterministic regardless of allocator behavior
(e.g. `bytearray(N)` may be satisfied by lazy anonymous mmap and never touch
pages, so real allocations do not reliably move VmHWM) and regardless of
whatever HWM the process already reached from earlier tests in the same
run. Only one integration test touches the real /proc/self/status, and it
asserts nothing about magnitude of growth — just the basic invariants.
"""

from graphify_mesh.sync import rss


def test_snapshot_reads_positive_values():
    snap = rss.snapshot()
    assert snap.vm_rss_kb > 0
    assert snap.vm_hwm_kb >= snap.vm_rss_kb


def test_tracker_attributes_hwm_growth_to_stage(monkeypatch):
    scripted = [
        rss.RssSnapshot(vm_rss_kb=100_000, vm_hwm_kb=100_000),  # __init__ baseline
        rss.RssSnapshot(vm_rss_kb=100_000, vm_hwm_kb=100_000),  # mark("stage-a")
        rss.RssSnapshot(vm_rss_kb=164_000, vm_hwm_kb=164_000),  # mark("stage-b")
    ]

    def fake_snapshot() -> rss.RssSnapshot:
        return scripted.pop(0)

    monkeypatch.setattr(rss, "snapshot", fake_snapshot)
    tracker = rss.StageRssTracker()
    tracker.mark("stage-a")
    tracker.mark("stage-b")

    result = tracker.to_dict()
    assert set(result) == {"stage-a", "stage-b"}
    assert result["stage-a"]["hwm_growth_kb"] == 0
    assert result["stage-b"]["hwm_growth_kb"] == 64_000
    for entry in result.values():
        assert entry["rss_kb_at_end"] > 0
        assert entry["hwm_kb_at_end"] >= entry["rss_kb_at_end"]


def test_tracker_growth_is_never_negative_when_hwm_is_flat(monkeypatch):
    # VmHWM never decreases in reality, but the tracker must not produce a
    # negative growth figure even if a snapshot ever reported a lower value
    # (e.g. a bogus/unparseable read defaulting to 0).
    scripted = [
        rss.RssSnapshot(vm_rss_kb=100_000, vm_hwm_kb=100_000),  # __init__ baseline
        rss.RssSnapshot(vm_rss_kb=100_000, vm_hwm_kb=100_000),  # mark("a")
        rss.RssSnapshot(vm_rss_kb=50_000, vm_hwm_kb=50_000),  # mark("b")
    ]

    def fake_snapshot() -> rss.RssSnapshot:
        return scripted.pop(0)

    monkeypatch.setattr(rss, "snapshot", fake_snapshot)
    tracker = rss.StageRssTracker()
    tracker.mark("a")
    tracker.mark("b")

    assert tracker.to_dict()["b"]["hwm_growth_kb"] == 0


def test_tracker_records_wall_and_cpu_per_stage(monkeypatch):
    clock = iter([10.0, 12.5, 20.0])  # __init__, mark a, mark b
    cpu = iter([1.0, 1.5, 4.0])
    monkeypatch.setattr(rss.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(rss, "_cpu_seconds", lambda: next(cpu))
    monkeypatch.setattr(rss, "snapshot", lambda: rss.RssSnapshot(1, 1))
    t = rss.StageRssTracker()
    t.mark("a")
    t.mark("b")
    d = t.to_dict()
    assert (d["a"]["wall_s"], d["a"]["cpu_s"]) == (2.5, 0.5)
    assert (d["b"]["wall_s"], d["b"]["cpu_s"]) == (7.5, 2.5)


def test_tracker_marks_are_ordered_and_repeat_safe():
    tracker = rss.StageRssTracker()
    tracker.mark("only")
    first = tracker.to_dict()["only"]["hwm_kb_at_end"]
    tracker.mark("only")  # re-mark overwrites, never raises
    assert tracker.to_dict()["only"]["hwm_kb_at_end"] >= first
