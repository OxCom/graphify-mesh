"""Infra-outage handling for the extract backend (remote Ollama host).

Incident being guarded against: a short remote-Ollama blip made 8/16 repos
`failed` within seconds; the unrefreshed gate blocked publish at 50% > 40%
although the merged graph was complete (every failed repo contributed its
restored last-good graph), so the healthy repos' work was discarded.

The fix: probe the backend around every ollama-backed child launch, classify
outage casualties as `infra_skipped`/`infra_failed`, exempt them from the
publish gate for a grace window, and skip the whole merge-to-publish tail
when an outage refreshed nothing at all.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error

import pytest

from graphify_mesh.sync import pipeline
from graphify_mesh.sync.config import (
    INFRA_GRACE_DEFAULT_HOURS,
    Settings,
    _extract_health_url_from_env,
    _read_infra_grace_hours,
)
from graphify_mesh.sync.pipeline import (
    INFRA_SKIPPED_PREFLIGHT_REASON,
    NOOP_INFRA_OUTAGE_REASON,
    PROBE_HEALTHY,
    PROBE_MISCONFIGURED,
    PROBE_OUTAGE,
    _guarded_apply_action,
    _within_infra_grace,
    default_extract_backend_probe,
)
from graphify_mesh.sync.state import load_state, save_state
from graphify_mesh.sync.sync_project import (
    ACTION_BOOTSTRAP,
    ACTION_EXTRACT,
    ACTION_UPDATE,
    STATUS_BOOTSTRAP_FAILED,
    STATUS_FAILED,
    STATUS_INFRA_FAILED,
    STATUS_INFRA_SKIPPED,
    STATUS_SHRINK_REFUSED,
    ProjectOutcome,
)

PROBE_URL = "http://probe.test/v1"


@pytest.fixture(autouse=True)
def _clean_probe_env(monkeypatch):
    """The probe settings default from the process environment — isolate the
    tests from whatever the machine running them has exported."""
    for var in (
        "GRAPHIFY_MESH_EXTRACT_HEALTH",
        "GRAPHIFY_MESH_EXTRACT_HEALTH_URL",
        "GRAPHIFY_MESH_EXTRACT_HEALTH_API_KEY",
        "GRAPHIFY_MESH_EXTRACT_HEALTH_TIMEOUT",
        "GRAPHIFY_MESH_INFRA_GRACE_HOURS",
        "OLLAMA_BASE_URL",
        "OLLAMA_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def _static_check(result: bool, calls: list | None = None):
    def check(url, key, timeout):
        if calls is not None:
            calls.append((url, key, timeout))
        return result

    return check


def _sequenced_check(results: list[bool], calls: list | None = None):
    """Returns results in order; the last one repeats."""
    remaining = list(results)

    def check(url, key, timeout):
        if calls is not None:
            calls.append((url, key, timeout))
        if len(remaining) > 1:
            return remaining.pop(0)
        return remaining[0]

    return check


def _bare_settings(tmp_path, **overrides) -> Settings:
    return Settings(
        mesh_root=tmp_path,
        scan_roots=[tmp_path],
        approved_roots=[tmp_path],
        registry_path=tmp_path / "registry.json",
        **overrides,
    )


def _outcome(status: str, action: str = ACTION_EXTRACT, reason: str = "exit=1: boom"):
    return ProjectOutcome("example-org.aaa", action, status, reason=reason)


class TestGuardedApplyAction:
    """Unit level: preflight skip, mid-run reclassification, never-reclassify."""

    def test_preflight_down_never_spawns_the_child(self, tmp_path, monkeypatch):
        calls = []

        def explode(*args, **kwargs):
            raise AssertionError("child must not be invoked while preflight is down")

        monkeypatch.setattr(pipeline, "apply_action", explode)
        settings = _bare_settings(
            tmp_path, extract_health_url=PROBE_URL, extract_health_check=_static_check(False, calls)
        )
        outcome = _guarded_apply_action(
            "example-org.aaa",
            "graphify",
            tmp_path,
            tmp_path,
            ACTION_EXTRACT,
            None,
            tmp_path,
            settings=settings,
        )
        assert outcome.status == STATUS_INFRA_SKIPPED
        assert outcome.reason == INFRA_SKIPPED_PREFLIGHT_REASON
        assert outcome.new_manifest is None  # state must not advance
        assert calls == [(PROBE_URL, "", 5.0)]

    def test_failed_with_probe_down_becomes_infra_failed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "apply_action", lambda *a, **k: _outcome(STATUS_FAILED))
        calls = []
        settings = _bare_settings(
            tmp_path,
            extract_health_url=PROBE_URL,
            extract_health_check=_sequenced_check([True, False], calls),
        )
        outcome = _guarded_apply_action(
            "example-org.aaa",
            "graphify",
            tmp_path,
            tmp_path,
            ACTION_EXTRACT,
            None,
            tmp_path,
            settings=settings,
        )
        assert outcome.status == STATUS_INFRA_FAILED
        assert outcome.reason == "exit=1: boom; backend probe failed"
        assert len(calls) == 2  # preflight + post-failure re-probe

    def test_failed_with_probe_up_stays_failed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "apply_action", lambda *a, **k: _outcome(STATUS_FAILED))
        settings = _bare_settings(
            tmp_path, extract_health_url=PROBE_URL, extract_health_check=_static_check(True)
        )
        outcome = _guarded_apply_action(
            "example-org.aaa",
            "graphify",
            tmp_path,
            tmp_path,
            ACTION_EXTRACT,
            None,
            tmp_path,
            settings=settings,
        )
        assert outcome.status == STATUS_FAILED
        assert outcome.reason == "exit=1: boom"

    def test_misconfigured_preflight_fails_open_and_spawns_the_child(self, tmp_path, monkeypatch):
        spawned = []

        def spawning_apply(*args, **kwargs):
            spawned.append(True)
            return _outcome(STATUS_FAILED)

        monkeypatch.setattr(pipeline, "apply_action", spawning_apply)
        probes = []

        def misconfigured_probe(url, key, timeout):
            probes.append((url, key, timeout))
            return PROBE_MISCONFIGURED

        settings = _bare_settings(
            tmp_path, extract_health_url=PROBE_URL, extract_health_probe=misconfigured_probe
        )
        outcome = _guarded_apply_action(
            "example-org.aaa",
            "graphify",
            tmp_path,
            tmp_path,
            ACTION_EXTRACT,
            None,
            tmp_path,
            settings=settings,
        )
        # Fail open: the child spawned normally, the failure stays plain
        # `failed` (guard disarmed — no post-failure re-probe either).
        assert spawned == [True]
        assert outcome.status == STATUS_FAILED
        assert outcome.reason == "exit=1: boom"
        assert probes == [(PROBE_URL, "", 5.0)]

    def test_tri_state_probe_di_wins_over_the_boolean_di(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "apply_action", lambda *a, **k: _outcome(STATUS_FAILED))
        bool_calls = []
        settings = _bare_settings(
            tmp_path,
            extract_health_url=PROBE_URL,
            extract_health_check=_static_check(True, bool_calls),
            extract_health_probe=lambda *a: PROBE_OUTAGE,
        )
        outcome = _guarded_apply_action(
            "example-org.aaa",
            "graphify",
            tmp_path,
            tmp_path,
            ACTION_EXTRACT,
            None,
            tmp_path,
            settings=settings,
        )
        assert outcome.status == STATUS_INFRA_SKIPPED
        assert bool_calls == []

    @pytest.mark.parametrize(
        ("status", "action"),
        [
            (STATUS_BOOTSTRAP_FAILED, ACTION_BOOTSTRAP),
            (STATUS_SHRINK_REFUSED, ACTION_EXTRACT),
        ],
    )
    def test_bootstrap_failed_and_shrink_refused_are_never_reclassified(
        self, tmp_path, monkeypatch, status, action
    ):
        monkeypatch.setattr(pipeline, "apply_action", lambda *a, **k: _outcome(status, action))
        calls = []
        settings = _bare_settings(
            tmp_path,
            extract_health_url=PROBE_URL,
            # Backend down after preflight: still no reclassification — these
            # verdicts are about the repo's own output, not reachability.
            extract_health_check=_sequenced_check([True, False], calls),
        )
        outcome = _guarded_apply_action(
            "example-org.aaa",
            "graphify",
            tmp_path,
            tmp_path,
            action,
            None,
            tmp_path,
            settings=settings,
        )
        assert outcome.status == status
        assert len(calls) == 1  # preflight only, no post-failure probe

    def test_update_action_bypasses_the_guard_entirely(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            pipeline, "apply_action", lambda *a, **k: _outcome("updated", ACTION_UPDATE, "")
        )
        calls = []
        settings = _bare_settings(
            tmp_path, extract_health_url=PROBE_URL, extract_health_check=_static_check(False, calls)
        )
        outcome = _guarded_apply_action(
            "example-org.aaa",
            "graphify",
            tmp_path,
            tmp_path,
            ACTION_UPDATE,
            None,
            tmp_path,
            settings=settings,
        )
        assert outcome.status == "updated"
        assert calls == []  # AST-only update never probes

    def test_inert_when_disabled_or_url_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "apply_action", lambda *a, **k: _outcome(STATUS_FAILED))
        calls = []
        for settings in (
            _bare_settings(
                tmp_path,
                extract_health_url=PROBE_URL,
                extract_health_enabled=False,
                extract_health_check=_static_check(False, calls),
            ),
            _bare_settings(
                tmp_path, extract_health_url="", extract_health_check=_static_check(False, calls)
            ),
        ):
            outcome = _guarded_apply_action(
                "example-org.aaa",
                "graphify",
                tmp_path,
                tmp_path,
                ACTION_EXTRACT,
                None,
                tmp_path,
                settings=settings,
            )
            assert outcome.status == STATUS_FAILED  # never infra_*
        assert calls == []


class TestStatusBuckets:
    def test_within_infra_grace_boundaries(self):
        now = time.time()
        assert _within_infra_grace(now, now, 24.0)
        assert _within_infra_grace(now, now - 23 * 3600.0, 24.0)
        assert not _within_infra_grace(now, now - 25 * 3600.0, 24.0)
        # 0 means: count toward the gate immediately, no exemption at all.
        assert not _within_infra_grace(now, now, 0.0)


def _setup_extract_and_update_repos(env):
    """Repo aaa gets a semantic touch (-> extract, ollama-backed) and repo
    bbb a code touch (-> update, AST-only) after an initial published run."""
    root_a = env.add_repo("example-org.aaa", "example-org", "aaa", "aaa.example-org.dev.lo")
    root_b = env.add_repo(
        "example-org.bbb", "example-org", "bbb", "bbb.example-org.dev.lo", "repo_b.json"
    )
    (root_b / "app.php").write_text("<?php echo 1;", encoding="utf-8")
    env.write_registry()
    first = pipeline.run(env.settings())
    assert first.published

    (root_a / "touched.md").write_text("# semantic touch\n", encoding="utf-8")
    (root_b / "app.php").write_text("<?php echo 2;", encoding="utf-8")
    return root_a, root_b


def _rows_by_repo(report) -> dict[str, dict]:
    return {row["repo_id"]: row for row in report.project_actions}


class TestPipelinePreflight:
    def test_preflight_down_skips_extract_but_update_still_runs(self, env, monkeypatch):
        _setup_extract_and_update_repos(env)
        state_before = load_state(env.settings().state_path)

        applied: list[str] = []
        real_apply = pipeline.apply_action

        def counting_apply(repo_id, *args, **kwargs):
            applied.append(repo_id)
            return real_apply(repo_id, *args, **kwargs)

        monkeypatch.setattr(pipeline, "apply_action", counting_apply)
        settings = env.settings(
            extract_health_url=PROBE_URL, extract_health_check=_static_check(False)
        )
        report = pipeline.run(settings)

        rows = _rows_by_repo(report)
        assert rows["example-org.aaa"]["status"] == STATUS_INFRA_SKIPPED
        assert rows["example-org.aaa"]["reason"] == INFRA_SKIPPED_PREFLIGHT_REASON
        # The ollama-backed child was never invoked; the AST-only update ran.
        assert applied == ["example-org.bbb"]
        assert rows["example-org.bbb"]["status"] == "updated"

        # State not advanced for the skipped repo (aside from infra_since).
        state_after = load_state(settings.state_path)
        assert (
            state_after["example-org.aaa"]["semantic_hash"]
            == state_before["example-org.aaa"]["semantic_hash"]
        )
        assert "infra_since" in state_after["example-org.aaa"]

        # Last-good graph still flowed to the merge, and — within grace —
        # the infra repo blocks nothing: the run publishes.
        assert report.merge_ok
        assert report.published
        assert report.unrefreshed_repos == []
        # stale_repos / degraded reporting includes infra repos like failed.
        assert "example-org.aaa" in report.stale_repos


class TestPipelineGate:
    def _setup_sixteen_repos(self, env):
        roots = {}
        for i in range(1, 9):
            rid = f"example-org.ok{i:02d}"
            roots[rid] = env.add_repo(rid, "example-org", f"ok{i:02d}", f"ok{i:02d}.dev.lo")
        for i in range(1, 9):
            rid = f"example-org.zz{i:02d}"
            roots[rid] = env.add_repo(
                rid, "example-org", f"zz{i:02d}", f"zz{i:02d}.dev.lo", "repo_b.json"
            )
        env.write_registry()
        first = pipeline.run(env.settings())
        assert first.published

        for root in roots.values():
            (root / "touched.md").write_text("# semantic touch\n", encoding="utf-8")
        for i in range(1, 9):
            env.set_control(env.collection_path("example-org", f"zz{i:02d}"), "fail")
        return roots

    def test_half_fleet_infra_failed_within_grace_still_publishes(self, env, monkeypatch):
        self._setup_sixteen_repos(env)

        # Model the incident: the backend blips exactly while a child runs —
        # preflights see it healthy, each failure's re-probe sees it down.
        # The flag is shared between the probe and apply_action wrappers and
        # both run inside the pool's worker threads, so guard it with a lock.
        flag = {"down": False}
        flag_lock = threading.Lock()

        def check(url, key, timeout):
            with flag_lock:
                healthy = not flag["down"]
                flag["down"] = False
            return healthy

        real_apply = pipeline.apply_action

        def blipping_apply(*args, **kwargs):
            outcome = real_apply(*args, **kwargs)
            if outcome.status == STATUS_FAILED:
                with flag_lock:
                    flag["down"] = True
            return outcome

        monkeypatch.setattr(pipeline, "apply_action", blipping_apply)
        settings = env.settings(extract_health_url=PROBE_URL, extract_health_check=check)
        settings.extract_concurrency = 1  # deterministic probe interleaving
        report = pipeline.run(settings)

        rows = _rows_by_repo(report)
        infra_rows = [r for r in rows.values() if r["status"] == STATUS_INFRA_FAILED]
        assert len(infra_rows) == 8
        assert all(r["reason"].endswith("; backend probe failed") for r in infra_rows)
        # Within grace: exempt from both ratios — 8/16 infra must NOT block.
        assert report.unrefreshed_repos == []
        assert report.publish_blocking_repos == []
        assert report.published
        assert len([r for r in report.stale_repos if r.startswith("example-org.zz")]) == 8

    def test_half_fleet_plain_failed_still_blocks(self, env):
        """Regression guard for the old behavior: with the probe inert the
        same 8/16 failures stay `failed` and trip the unrefreshed gate."""
        self._setup_sixteen_repos(env)
        report = pipeline.run(env.settings())  # conftest default: probe inert

        rows = _rows_by_repo(report)
        failed_rows = [r for r in rows.values() if r["status"] == STATUS_FAILED]
        assert len(failed_rows) == 8
        assert len(report.unrefreshed_repos) == 8
        assert not report.published
        assert "unrefreshed" in report.publish_blocked_reason

    def test_grace_expiry_counts_toward_unrefreshed_and_blocks(self, env):
        _setup_extract_and_update_repos(env)
        settings = env.settings(
            extract_health_url=PROBE_URL, extract_health_check=_static_check(False)
        )
        # Pre-seed an outage that started long before the grace window.
        state = load_state(settings.state_path)
        stale_since = time.time() - (INFRA_GRACE_DEFAULT_HOURS + 100) * 3600.0
        state["example-org.aaa"]["infra_since"] = stale_since
        save_state(settings.state_path, state)

        report = pipeline.run(settings)

        assert _rows_by_repo(report)["example-org.aaa"]["status"] == STATUS_INFRA_SKIPPED
        # Past the grace the repo counts toward the UNREFRESHED ratio (1/2 =
        # 50% > 40%), never the suspect one.
        assert report.unrefreshed_repos == ["example-org.aaa"]
        assert report.publish_blocking_repos == []
        assert not report.published
        assert "unrefreshed" in report.publish_blocked_reason
        # The original outage start is preserved, not reset by this run.
        assert load_state(settings.state_path)["example-org.aaa"]["infra_since"] == stale_since


class TestInfraSinceLifecycle:
    def test_set_once_preserved_across_runs_and_cleared_on_success(self, env):
        root_a = env.add_repo("example-org.aaa", "example-org", "aaa", "aaa.example-org.dev.lo")
        env.write_registry()
        first = pipeline.run(env.settings())
        assert first.published
        (root_a / "touched.md").write_text("# semantic touch\n", encoding="utf-8")

        down = env.settings(extract_health_url=PROBE_URL, extract_health_check=_static_check(False))
        pipeline.run(down)
        t1 = load_state(down.state_path)["example-org.aaa"]["infra_since"]
        assert t1 > 0

        # Second outage run: the state was not advanced, so the same extract
        # is re-attempted — and the ORIGINAL outage start must be preserved.
        pipeline.run(down)
        assert load_state(down.state_path)["example-org.aaa"]["infra_since"] == t1

        # Recovery: the extract succeeds and the successful refresh clears
        # infra_since (the state entry is replaced wholesale).
        recovered = pipeline.run(env.settings())
        assert recovered.published
        state = load_state(down.state_path)
        assert "infra_since" not in state["example-org.aaa"]


class TestZeroRefreshNoop:
    def test_all_actionable_infra_skips_merge_and_later_stages(self, env):
        root_a = env.add_repo("example-org.aaa", "example-org", "aaa", "aaa.example-org.dev.lo")
        env.add_repo(
            "example-org.bbb", "example-org", "bbb", "bbb.example-org.dev.lo", "repo_b.json"
        )
        env.write_registry()
        first = pipeline.run(env.settings())
        assert first.published
        # Only aaa changes; bbb stays unchanged (ACTION_SKIP) — an unchanged
        # repo neither counts as refreshed nor prevents the noop.
        (root_a / "touched.md").write_text("# semantic touch\n", encoding="utf-8")

        settings = env.settings(
            extract_health_url=PROBE_URL, extract_health_check=_static_check(False)
        )
        report = pipeline.run(settings)

        rows = _rows_by_repo(report)
        assert rows["example-org.aaa"]["status"] == STATUS_INFRA_SKIPPED
        assert rows["example-org.bbb"]["status"] == "unchanged"
        assert not report.published
        assert report.publish_blocked_reason == NOOP_INFRA_OUTAGE_REASON
        assert report.skipped_stages == [
            "merge",
            "naming",
            "embedding",
            "overlay",
            "lexical_index",
            "validate",
            "publish",
        ]
        assert not report.merge_ok  # merge never ran


class TestDisableKnob:
    def test_env_off_keeps_failures_failed_and_never_probes(self, env, monkeypatch):
        root_a = env.add_repo("example-org.aaa", "example-org", "aaa", "aaa.example-org.dev.lo")
        env.write_registry()
        first = pipeline.run(env.settings())
        assert first.published
        (root_a / "touched.md").write_text("# semantic touch\n", encoding="utf-8")
        env.set_control(env.collection_path("example-org", "aaa"), "fail")

        monkeypatch.setenv("GRAPHIFY_MESH_EXTRACT_HEALTH", "off")
        calls = []
        settings = env.settings(
            extract_health_url=PROBE_URL, extract_health_check=_static_check(False, calls)
        )
        assert settings.extract_health_enabled is False
        report = pipeline.run(settings)

        assert _rows_by_repo(report)["example-org.aaa"]["status"] == STATUS_FAILED
        assert calls == []
        assert "infra_since" not in load_state(settings.state_path).get("example-org.aaa", {})


class TestInfraGraceHoursParsing:
    ENV = "GRAPHIFY_MESH_INFRA_GRACE_HOURS"

    def test_default(self, monkeypatch):
        monkeypatch.delenv(self.ENV, raising=False)
        assert _read_infra_grace_hours(self.ENV, INFRA_GRACE_DEFAULT_HOURS) == 24.0

    def test_valid_override(self, monkeypatch):
        monkeypatch.setenv(self.ENV, "12")
        assert _read_infra_grace_hours(self.ENV, INFRA_GRACE_DEFAULT_HOURS) == 12.0

    def test_zero_is_allowed(self, monkeypatch):
        monkeypatch.setenv(self.ENV, "0")
        assert _read_infra_grace_hours(self.ENV, INFRA_GRACE_DEFAULT_HOURS) == 0.0

    @pytest.mark.parametrize("raw", ["-1", "abc"])
    def test_negative_or_unparseable_raises_naming_the_var(self, monkeypatch, raw):
        monkeypatch.setenv(self.ENV, raw)
        with pytest.raises(ValueError, match=self.ENV):
            _read_infra_grace_hours(self.ENV, INFRA_GRACE_DEFAULT_HOURS)

    @pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "Infinity"])
    def test_non_finite_raises_naming_the_var(self, monkeypatch, raw):
        # float() happily parses these, but a nan/inf grace window makes the
        # gate arithmetic silently nonsensical — must fail fast at startup.
        monkeypatch.setenv(self.ENV, raw)
        with pytest.raises(ValueError, match=self.ENV):
            _read_infra_grace_hours(self.ENV, INFRA_GRACE_DEFAULT_HOURS)

    def test_settings_field_defaults(self, env):
        assert env.settings().infra_grace_hours == INFRA_GRACE_DEFAULT_HOURS


class TestExtractHealthUrlOverride:
    def test_set_but_empty_override_is_inert_not_fallback(self, monkeypatch):
        # An operator's explicit "no probe URL" must yield "" (feature inert),
        # never silently fall back to probing OLLAMA_BASE_URL.
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://fallback.test")
        monkeypatch.setenv("GRAPHIFY_MESH_EXTRACT_HEALTH_URL", "")
        assert _extract_health_url_from_env() == ""

    def test_whitespace_only_override_is_inert_too(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://fallback.test")
        monkeypatch.setenv("GRAPHIFY_MESH_EXTRACT_HEALTH_URL", "   ")
        assert _extract_health_url_from_env() == ""

    def test_entirely_unset_override_falls_back_to_ollama_base_url(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://fallback.test")
        monkeypatch.delenv("GRAPHIFY_MESH_EXTRACT_HEALTH_URL", raising=False)
        assert _extract_health_url_from_env() == "http://fallback.test"


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(PROBE_URL + "/models", code, "boom", hdrs=None, fp=None)


class TestDefaultExtractBackendProbe:
    """HTTP contract + tri-state classification of the default probe."""

    def test_probe_http_contract(self, monkeypatch):
        captured = {}

        class FakeResponse:
            status = 200

            def getcode(self):
                return self.status

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

        def fake_urlopen(req, timeout=None, context=None):
            captured["url"] = req.full_url
            captured["auth"] = req.get_header("Authorization")
            captured["timeout"] = timeout
            return FakeResponse()

        monkeypatch.setattr(pipeline.urllib.request, "urlopen", fake_urlopen)
        state, detail = default_extract_backend_probe(PROBE_URL + "/", "sekret", 5.0)
        assert state == PROBE_HEALTHY
        assert detail == ""
        assert captured["url"] == PROBE_URL + "/models"
        assert captured["auth"] == "Bearer sekret"
        assert captured["timeout"] == 5.0

    @pytest.mark.parametrize(
        ("exc", "expected_state", "detail_fragment"),
        [
            (_http_error(401), PROBE_MISCONFIGURED, "401"),
            (_http_error(404), PROBE_MISCONFIGURED, "404"),
            (_http_error(500), PROBE_OUTAGE, "500"),
            (urllib.error.URLError(ConnectionRefusedError("refused")), PROBE_OUTAGE, "refused"),
            (TimeoutError("timed out"), PROBE_OUTAGE, "timed out"),
        ],
    )
    def test_classification(self, monkeypatch, exc, expected_state, detail_fragment):
        def raising_urlopen(req, timeout=None, context=None):
            raise exc

        monkeypatch.setattr(pipeline.urllib.request, "urlopen", raising_urlopen)
        state, detail = default_extract_backend_probe(PROBE_URL, "key", 5.0)
        assert state == expected_state
        assert detail_fragment in detail

    def test_non_http_url_is_misconfigured_without_any_request(self, monkeypatch):
        def exploding_urlopen(req, timeout=None, context=None):
            raise AssertionError("no request may be attempted for a non-http(s) URL")

        monkeypatch.setattr(pipeline.urllib.request, "urlopen", exploding_urlopen)
        state, detail = default_extract_backend_probe("ftp://probe.test", "key", 5.0)
        assert state == PROBE_MISCONFIGURED
        assert "ftp://probe.test" in detail


class TestMisconfiguredProbeFailsOpenInPipeline:
    def test_401_spawns_children_normally_and_logs_error_once_per_run(
        self, env, monkeypatch, caplog
    ):
        _root_a, root_b = _setup_extract_and_update_repos(env)
        # Both repos need an ollama-backed extract this run.
        (root_b / "touched.md").write_text("# semantic touch\n", encoding="utf-8")

        def raising_urlopen(req, timeout=None, context=None):
            raise _http_error(401)

        monkeypatch.setattr(pipeline.urllib.request, "urlopen", raising_urlopen)
        # No DI probe: the pipeline runs the real default_extract_backend_probe.
        settings = env.settings(extract_health_url=PROBE_URL)
        settings.extract_concurrency = 1
        with caplog.at_level(logging.ERROR, logger="graphify_mesh.sync"):
            report = pipeline.run(settings)

        # Fail open: both children spawned and succeeded — no infra_* status,
        # no infra_since, and the run publishes.
        rows = _rows_by_repo(report)
        assert rows["example-org.aaa"]["status"] == "updated"
        assert rows["example-org.bbb"]["status"] == "updated"
        assert report.published
        state = load_state(settings.state_path)
        assert "infra_since" not in state["example-org.aaa"]
        assert "infra_since" not in state["example-org.bbb"]
        # Two launches probed the misconfigured endpoint; ONE error line.
        errors = [r for r in caplog.records if "probe misconfigured" in r.getMessage()]
        assert len(errors) == 1
        assert PROBE_URL in errors[0].getMessage()
        assert "401" in errors[0].getMessage()

    def test_connection_error_is_still_an_outage(self, env, monkeypatch):
        _setup_extract_and_update_repos(env)

        def raising_urlopen(req, timeout=None, context=None):
            raise urllib.error.URLError(ConnectionRefusedError("refused"))

        monkeypatch.setattr(pipeline.urllib.request, "urlopen", raising_urlopen)
        settings = env.settings(extract_health_url=PROBE_URL)
        report = pipeline.run(settings)

        assert _rows_by_repo(report)["example-org.aaa"]["status"] == STATUS_INFRA_SKIPPED
        assert "infra_since" in load_state(settings.state_path)["example-org.aaa"]


class TestConcurrentClassification:
    def test_two_workers_classify_repos_independently(self, env, monkeypatch):
        """Two extract launches on a two-worker pool, held at the preflight by
        a barrier so both are provably in flight concurrently. Each repo's
        probe sequence is thread-local (a repo runs entirely inside one worker
        task), so the failing repo's re-probe cannot leak into the healthy
        repo's classification."""
        root_a = env.add_repo("example-org.aaa", "example-org", "aaa", "aaa.example-org.dev.lo")
        root_b = env.add_repo(
            "example-org.bbb", "example-org", "bbb", "bbb.example-org.dev.lo", "repo_b.json"
        )
        env.write_registry()
        first = pipeline.run(env.settings())
        assert first.published
        state_before = load_state(env.settings().state_path)

        (root_a / "touched.md").write_text("# semantic touch\n", encoding="utf-8")
        (root_b / "touched.md").write_text("# semantic touch\n", encoding="utf-8")
        env.set_control(env.collection_path("example-org", "aaa"), "fail")

        barrier = threading.Barrier(2, timeout=10)
        per_thread = threading.local()

        def per_repo_probe(url, key, timeout):
            launch_calls = getattr(per_thread, "calls", 0)
            per_thread.calls = launch_calls + 1
            if launch_calls == 0:
                # Preflight: both workers must reach it before either child
                # spawns — proves the two launches really ran concurrently.
                barrier.wait()
                return True
            # Post-failure re-probe (only aaa's worker gets here): down.
            return False

        settings = env.settings(extract_health_url=PROBE_URL, extract_health_check=per_repo_probe)
        settings.extract_concurrency = 2
        report = pipeline.run(settings)

        rows = _rows_by_repo(report)
        assert rows["example-org.aaa"]["status"] == STATUS_INFRA_FAILED
        assert rows["example-org.bbb"]["status"] == "updated"
        assert report.published  # one in-grace infra repo blocks nothing

        state_after = load_state(settings.state_path)
        # aaa: outage recorded, accepted state untouched.
        assert "infra_since" in state_after["example-org.aaa"]
        assert (
            state_after["example-org.aaa"]["semantic_hash"]
            == state_before["example-org.aaa"]["semantic_hash"]
        )
        # bbb: refreshed normally — state advanced, no outage marker.
        assert "infra_since" not in state_after["example-org.bbb"]
        assert (
            state_after["example-org.bbb"]["semantic_hash"]
            != state_before["example-org.bbb"]["semantic_hash"]
        )


class TestMidRunRecovery:
    def test_first_launch_probes_down_later_launch_probes_up_and_succeeds(self, env):
        root_a = env.add_repo("example-org.aaa", "example-org", "aaa", "aaa.example-org.dev.lo")
        root_b = env.add_repo(
            "example-org.bbb", "example-org", "bbb", "bbb.example-org.dev.lo", "repo_b.json"
        )
        env.write_registry()
        first = pipeline.run(env.settings())
        assert first.published

        (root_a / "touched.md").write_text("# semantic touch\n", encoding="utf-8")
        (root_b / "touched.md").write_text("# semantic touch\n", encoding="utf-8")

        # Per-launch re-probe: the backend is down for aaa's preflight and
        # back up by bbb's — mid-run recovery is caught, not snapshotted away.
        settings = env.settings(
            extract_health_url=PROBE_URL, extract_health_check=_sequenced_check([False, True])
        )
        settings.extract_concurrency = 1  # deterministic launch order
        report = pipeline.run(settings)

        rows = _rows_by_repo(report)
        assert rows["example-org.aaa"]["status"] == STATUS_INFRA_SKIPPED
        assert rows["example-org.bbb"]["status"] == "updated"
        assert report.published
        state = load_state(settings.state_path)
        assert "infra_since" in state["example-org.aaa"]
        assert "infra_since" not in state["example-org.bbb"]


class TestTotalOutagePastGrace:
    def test_all_repos_past_grace_blocks_with_unrefreshed_reason_not_noop(self, env):
        root_a = env.add_repo("example-org.aaa", "example-org", "aaa", "aaa.example-org.dev.lo")
        root_b = env.add_repo(
            "example-org.bbb", "example-org", "bbb", "bbb.example-org.dev.lo", "repo_b.json"
        )
        env.write_registry()
        first = pipeline.run(env.settings())
        assert first.published
        (root_a / "touched.md").write_text("# semantic touch\n", encoding="utf-8")
        (root_b / "touched.md").write_text("# semantic touch\n", encoding="utf-8")

        settings = env.settings(
            extract_health_url=PROBE_URL, extract_health_check=_static_check(False)
        )
        # Pre-seed an outage older than the grace window for BOTH repos.
        state = load_state(settings.state_path)
        stale_since = time.time() - (INFRA_GRACE_DEFAULT_HOURS + 100) * 3600.0
        state["example-org.aaa"]["infra_since"] = stale_since
        state["example-org.bbb"]["infra_since"] = stale_since
        save_state(settings.state_path, state)

        report = pipeline.run(settings)

        rows = _rows_by_repo(report)
        assert rows["example-org.aaa"]["status"] == STATUS_INFRA_SKIPPED
        assert rows["example-org.bbb"]["status"] == STATUS_INFRA_SKIPPED
        # A >grace total outage must NOT look like routine idling: the noop is
        # suppressed, the normal tail runs, and the run ends blocked on the
        # unrefreshed threshold.
        assert report.publish_blocked_reason != NOOP_INFRA_OUTAGE_REASON
        assert report.skipped_stages == []
        assert report.merge_ok
        assert not report.published
        assert "unrefreshed" in report.publish_blocked_reason
        assert report.unrefreshed_repos == ["example-org.aaa", "example-org.bbb"]
        assert report.infra_repos_past_grace == ["example-org.aaa", "example-org.bbb"]
        assert report.infra_repos_in_grace == []


class TestNoopSuppressedOnRemoval:
    def test_removed_repo_is_merged_out_even_when_every_actionable_repo_is_infra(self, env):
        root_a = env.add_repo("example-org.aaa", "example-org", "aaa", "aaa.example-org.dev.lo")
        root_c = env.add_repo(
            "example-org.ccc", "example-org", "ccc", "ccc.example-org.dev.lo", "repo_b.json"
        )
        env.write_registry()
        first = pipeline.run(env.settings())
        assert first.published

        # aaa needs an extract (probed, backend down) and ccc's graphify-out
        # symlink disappears — a removed repo that must still be merged OUT.
        (root_a / "touched.md").write_text("# semantic touch\n", encoding="utf-8")
        (root_c / "graphify-out").unlink()

        settings = env.settings(
            extract_health_url=PROBE_URL, extract_health_check=_static_check(False)
        )
        report = pipeline.run(settings)

        assert "example-org.ccc" in report.reconciliation["removed"]
        assert _rows_by_repo(report)["example-org.aaa"]["status"] == STATUS_INFRA_SKIPPED
        # Every actionable repo ended infra_*, yet the noop must NOT fire —
        # the removal itself is a change that has to publish.
        assert report.publish_blocked_reason != NOOP_INFRA_OUTAGE_REASON
        assert report.skipped_stages == []
        assert report.merge_ok
        assert report.published
        manifest = json.loads(
            (settings.current_symlink / "generation-manifest.json").read_text(encoding="utf-8")
        )
        assert "example-org.ccc" not in manifest["repo_input_hashes"]
        assert "example-org.aaa" in manifest["repo_input_hashes"]


class TestGateDenominator:
    def test_in_grace_infra_repos_leave_both_numerator_and_denominator(self, env):
        """Pins the arithmetic: 7 in-grace infra + 6 failed + 3 ok. With the
        old diluted denominator the unrefreshed ratio was 6/16 = 37.5% < 40%
        and the run published although 13/16 repos went unrefreshed; with the
        in-grace repos excluded it is 6/9 = 66.7% > 40% and must block."""
        roots = {}
        for i in range(1, 8):
            rid = f"example-org.inf{i:02d}"
            roots[rid] = env.add_repo(rid, "example-org", f"inf{i:02d}", f"inf{i:02d}.dev.lo")
        for i in range(1, 7):
            rid = f"example-org.mfl{i:02d}"
            roots[rid] = env.add_repo(
                rid, "example-org", f"mfl{i:02d}", f"mfl{i:02d}.dev.lo", "repo_b.json"
            )
        for i in range(1, 4):
            rid = f"example-org.okk{i:02d}"
            roots[rid] = env.add_repo(rid, "example-org", f"okk{i:02d}", f"okk{i:02d}.dev.lo")
        env.write_registry()
        first = pipeline.run(env.settings())
        assert first.published

        for root in roots.values():
            (root / "touched.md").write_text("# semantic touch\n", encoding="utf-8")
        for i in range(1, 7):
            env.set_control(env.collection_path("example-org", f"mfl{i:02d}"), "fail")

        # Launch order is registry order (sequential pool): the 7 inf* repos
        # probe first and each consumes one False (preflight down, one probe
        # per launch); every later probe — mfl* preflights AND their
        # post-failure re-probes, okk* preflights — sees a healthy backend,
        # so the mfl* failures stay plain `failed`.
        settings = env.settings(
            extract_health_url=PROBE_URL,
            extract_health_check=_sequenced_check([False] * 7 + [True]),
        )
        settings.extract_concurrency = 1
        report = pipeline.run(settings)

        rows = _rows_by_repo(report)
        infra_rows = [r for r in rows.values() if r["status"] == STATUS_INFRA_SKIPPED]
        assert sorted(r["repo_id"] for r in infra_rows) == report.infra_repos_in_grace
        assert len(report.infra_repos_in_grace) == 7
        assert len(report.unrefreshed_repos) == 6
        assert report.publish_blocking_repos == []
        assert not report.published
        assert "unrefreshed" in report.publish_blocked_reason
        # The reason names the excluded-denominator count, not the fleet size.
        assert "6/9" in report.publish_blocked_reason


class TestDualGateReason:
    def test_reason_names_both_gates_when_both_trip(self, env):
        # 2 bootstrap failures (suspect gate: 2/4 = 50% > 30%) plus 2 plain
        # failures (unrefreshed gate: 2/4 = 50% > 40%) in the same run.
        for sub in ("bfa", "bfb"):
            env.add_repo(f"example-org.{sub}", "example-org", sub, f"{sub}.dev.lo", None)
            env.set_control(env.collection_path("example-org", sub), "fail")
        for sub in ("fla", "flb"):
            env.add_repo(f"example-org.{sub}", "example-org", sub, f"{sub}.dev.lo")
            env.set_control(env.collection_path("example-org", sub), "fail")
        env.write_registry()

        report = pipeline.run(env.settings())

        assert not report.published
        assert report.publish_blocking_repos == ["example-org.bfa", "example-org.bfb"]
        assert report.unrefreshed_repos == ["example-org.fla", "example-org.flb"]
        assert "suspect" in report.publish_blocked_reason
        assert "unrefreshed" in report.publish_blocked_reason
        assert "; " in report.publish_blocked_reason
