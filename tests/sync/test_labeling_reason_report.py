"""naming_result.reason reaches the run report and the naming log line.

`labeling` alone said "ok" when labels were rejected as not grounded in member
labels and hub names were kept; the reason is what tells the operator why.
"""

from __future__ import annotations

import dataclasses
import logging

from graphify_mesh.sync import naming, pipeline, repo_tags

REASON = "2 label(s) rejected as not grounded in member labels, hub names kept"


def test_reason_in_report_and_log(env, monkeypatch, caplog):
    monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.103")
    env.add_repo("example-org.svc", "example-org", "svc", "svc.example-org.dev.lo")
    env.write_registry()
    real = naming.run_naming

    def with_reason(*args, **kwargs):
        return dataclasses.replace(real(*args, **kwargs), reason=REASON)

    monkeypatch.setattr(pipeline.naming, "run_naming", with_reason)

    with caplog.at_level(logging.INFO, logger="graphify_mesh.sync"):
        report = pipeline.run(env.settings(shrink_policy="count"))

    assert report.labeling_reason == REASON
    assert any(
        r.getMessage().startswith("naming: ") and REASON in r.getMessage() for r in caplog.records
    )
