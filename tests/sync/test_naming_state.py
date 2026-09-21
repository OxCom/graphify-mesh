import json

from graphify_mesh.sync import naming_state


def test_save_then_load_round_trips(tmp_path):
    state = naming_state.NamingState(
        merged_fingerprint="abc123",
        clustering={"backend": "louvain", "resolution": 1.0},
        labeling={"backend": "ollama", "model": "qwen2.5-coder:7b"},
        communities={
            "0": naming_state.CommunityEntry(sig="s0", name="Auth", provisional=False),
            "1": naming_state.CommunityEntry(sig="s1", name="Orders", provisional=True),
        },
        assignments={"a::x": 0, "a::y": 1},
    )
    naming_state.save(tmp_path, state)
    loaded = naming_state.load(tmp_path)
    assert loaded == state
    on_disk = json.loads((tmp_path / naming_state.STATE_FILENAME).read_text())
    assert on_disk["schema_version"] == naming_state.SCHEMA_VERSION


def test_load_returns_none_for_missing_or_unreadable_state(tmp_path):
    assert naming_state.load(tmp_path) is None
    (tmp_path / naming_state.STATE_FILENAME).write_text("{ not json")
    assert naming_state.load(tmp_path) is None


def test_load_returns_none_on_unknown_schema_version(tmp_path):
    (tmp_path / naming_state.STATE_FILENAME).write_text(
        json.dumps({"schema_version": 99, "merged_fingerprint": "x"})
    )
    assert naming_state.load(tmp_path) is None


def test_seed_takes_only_cids_present_in_both_sidecars(tmp_path):
    out = tmp_path / "graphify-out"
    out.mkdir()
    (out / ".graphify_labels.json").write_text(
        json.dumps({"0": "Auth", "1": "Orders", "2": "Reports"})
    )
    (out / ".graphify_labels.json.sig").write_text(json.dumps({"0": "s0", "1": "s1"}))
    seeded = naming_state.seed_from_graphify_sidecars(tmp_path)
    assert set(seeded) == {"0", "1"}
    assert seeded["0"] == naming_state.CommunityEntry(sig="s0", name="Auth", provisional=True)


def test_names_by_sig_maps_signature_to_name_and_provisional_flag():
    state = naming_state.NamingState(
        merged_fingerprint="f",
        clustering={},
        labeling={},
        communities={
            "0": naming_state.CommunityEntry(sig="s0", name="Auth", provisional=False),
            "7": naming_state.CommunityEntry(sig="s7", name="Hub", provisional=True),
        },
        assignments={},
    )
    assert naming_state.names_by_sig(state) == {"s0": ("Auth", False), "s7": ("Hub", True)}
    assert naming_state.names_by_sig(None) == {}
