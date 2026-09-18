"""Version parity between the imported graphify and the merge binary.

`compute_tag_to_repo_id` derives the auto-tag -> repo_id map from
`graphify.build.distinct_repo_tags` in this interpreter, while the merge that
produced those tags ran `GRAPHIFY_BIN`, possibly from another environment. Two
versions that derive auto tags differently produce a map that silently matches
nothing, and the existing count check passes as long as the counts agree — so
the versions are compared before the map is built.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from graphify_mesh.sync import repo_tags
from graphify_mesh.sync.graphify_cli import CliResult

GRAPH_PATHS = [
    Path("/path/to/graph-mesh/graphify/example-org/backend-a/graph.json"),
    Path("/path/to/graph-mesh/graphify/example-org/frontend-b/graph.json"),
]
REPO_IDS = ["example-org.backend-a", "example-org.frontend-b"]


def _fake_run(stdout: str = "", stderr: str = "", returncode: int = 0):
    def run(argv, cwd, env, timeout=None):
        return CliResult(returncode=returncode, stdout=stdout, stderr=stderr)

    return run


@pytest.fixture(autouse=True)
def _forget_probed_binaries():
    # The parity check remembers which binaries it already compared, so each
    # test must start from an unprobed state.
    repo_tags.reset_version_parity_cache()
    yield
    repo_tags.reset_version_parity_cache()


class TestVersionParity:
    def test_matching_versions_build_the_map(self, monkeypatch):
        monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.56")
        monkeypatch.setattr(repo_tags, "_run", _fake_run(stdout="graphify 0.9.56\n"))

        mapping = repo_tags.compute_tag_to_repo_id(GRAPH_PATHS, REPO_IDS, graphify_bin="graphify")

        assert set(mapping.values()) == set(REPO_IDS)

    def test_differing_versions_raise_naming_both(self, monkeypatch):
        monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.56")
        monkeypatch.setattr(repo_tags, "_run", _fake_run(stdout="graphify 0.9.63\n"))

        with pytest.raises(ValueError) as excinfo:
            repo_tags.compute_tag_to_repo_id(GRAPH_PATHS, REPO_IDS, graphify_bin="graphify")

        message = str(excinfo.value)
        assert "0.9.56" in message
        assert "0.9.63" in message

    @pytest.mark.parametrize(
        ("returncode", "stdout", "stderr"),
        [
            (2, "", "error: no such option: --version\n"),  # flag not supported
            (0, "graphify\n", ""),  # banner carries no version number
        ],
    )
    def test_unusable_version_output_logs_and_continues(
        self, monkeypatch, caplog, returncode, stdout, stderr
    ):
        monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.56")
        monkeypatch.setattr(
            repo_tags, "_run", _fake_run(stdout=stdout, stderr=stderr, returncode=returncode)
        )

        with caplog.at_level(logging.WARNING, logger="graphify_mesh.sync"):
            mapping = repo_tags.compute_tag_to_repo_id(
                GRAPH_PATHS, REPO_IDS, graphify_bin="graphify"
            )

        assert set(mapping.values()) == set(REPO_IDS)
        assert "did not report a usable version" in caplog.text

    def test_no_binary_path_skips_the_check_and_logs(self, monkeypatch, caplog):
        def explode(*args, **kwargs):
            raise AssertionError("the binary must not be probed when graphify_bin is None")

        monkeypatch.setattr(repo_tags, "_run", explode)

        with caplog.at_level(logging.WARNING, logger="graphify_mesh.sync"):
            mapping = repo_tags.compute_tag_to_repo_id(GRAPH_PATHS, REPO_IDS)

        assert set(mapping.values()) == set(REPO_IDS)
        assert "no graphify binary path" in caplog.text

    def test_missing_package_metadata_skips_the_check(self, monkeypatch, caplog):
        monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: None)

        def explode(*args, **kwargs):
            raise AssertionError("the binary must not be probed without an in-process version")

        monkeypatch.setattr(repo_tags, "_run", explode)

        with caplog.at_level(logging.WARNING, logger="graphify_mesh.sync"):
            repo_tags.compute_tag_to_repo_id(GRAPH_PATHS, REPO_IDS, graphify_bin="graphify")

        assert "carries no installed metadata" in caplog.text

    def test_count_check_still_runs_after_a_matching_version(self, monkeypatch):
        monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.56")
        monkeypatch.setattr(repo_tags, "_run", _fake_run(stdout="graphify 0.9.56\n"))

        with pytest.raises(ValueError, match="repo-tag count mismatch"):
            repo_tags.compute_tag_to_repo_id([GRAPH_PATHS[0]], REPO_IDS, graphify_bin="graphify")


class TestBinaryVersionProbe:
    def test_parses_the_version_out_of_the_banner(self, monkeypatch):
        monkeypatch.setattr(repo_tags, "_run", _fake_run(stdout="graphify 0.9.56\n"))
        assert repo_tags._binary_graphify_version("graphify") == "0.9.56"

    def test_reads_the_banner_from_stderr_too(self, monkeypatch):
        monkeypatch.setattr(repo_tags, "_run", _fake_run(stderr="graphify, version 1.2.3\n"))
        assert repo_tags._binary_graphify_version("graphify") == "1.2.3"

    def test_empty_binary_value_probes_nothing(self, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("an empty GRAPHIFY_BIN must not reach subprocess")

        monkeypatch.setattr(repo_tags, "_run", explode)
        assert repo_tags._binary_graphify_version("   ") is None


class TestReleaseComparison:
    """Only the release parts decide a mismatch. A dev install, a local build
    tag or a post-release of the same release is the same graphify as far as
    `distinct_repo_tags` is concerned, and blocking every run over a suffix
    made the check unusable against a real install."""

    @pytest.mark.parametrize(
        ("in_process", "from_binary"),
        [
            ("0.9.64.dev0", "0.9.64"),
            ("0.9.64", "0.9.64.dev3"),
            ("0.9.64+g1a2b3c", "0.9.64"),
            ("0.9.64.post1", "0.9.64"),
        ],
    )
    def test_same_release_with_a_different_suffix_is_not_a_mismatch(
        self, monkeypatch, in_process, from_binary
    ):
        monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: in_process)
        monkeypatch.setattr(repo_tags, "_run", _fake_run(stdout=f"graphify {from_binary}\n"))

        mapping = repo_tags.compute_tag_to_repo_id(GRAPH_PATHS, REPO_IDS, graphify_bin="graphify")

        assert set(mapping.values()) == set(REPO_IDS)

    @pytest.mark.parametrize(
        ("in_process", "from_binary"),
        [
            ("0.9.63", "0.9.64"),
            ("0.9.64.dev0", "0.9.63"),
            ("1.0.0", "0.9.64"),
        ],
    )
    def test_differing_releases_still_raise(self, monkeypatch, in_process, from_binary):
        monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: in_process)
        monkeypatch.setattr(repo_tags, "_run", _fake_run(stdout=f"graphify {from_binary}\n"))

        with pytest.raises(ValueError, match="graphify version mismatch"):
            repo_tags.compute_tag_to_repo_id(GRAPH_PATHS, REPO_IDS, graphify_bin="graphify")

    def test_unparseable_version_warns_instead_of_blocking(self, monkeypatch, caplog):
        monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "not.a.version")
        monkeypatch.setattr(repo_tags, "_run", _fake_run(stdout="graphify 0.9.64\n"))

        with caplog.at_level(logging.WARNING, logger="graphify_mesh.sync"):
            mapping = repo_tags.compute_tag_to_repo_id(
                GRAPH_PATHS, REPO_IDS, graphify_bin="graphify"
            )

        assert set(mapping.values()) == set(REPO_IDS)
        assert "not comparable" in caplog.text

    def test_banner_version_is_read_next_to_the_package_name(self, monkeypatch):
        """A banner that prints the interpreter first must not be compared
        against the interpreter's version."""
        monkeypatch.setattr(repo_tags, "_run", _fake_run(stdout="Python 3.11.2, graphify 0.9.64\n"))

        assert repo_tags._binary_graphify_version("graphify") == "0.9.64"


class TestProbeMemo:
    def test_a_binary_already_compared_is_not_probed_again(self, monkeypatch):
        """The pipeline runs the check before the merge and the tag map is
        built after it; the second call must cost no subprocess."""
        monkeypatch.setattr(repo_tags, "_in_process_graphify_version", lambda: "0.9.64")
        calls = []

        def counting_run(argv, cwd, env, timeout=None):
            calls.append(argv)
            return CliResult(returncode=0, stdout="graphify 0.9.64\n", stderr="")

        monkeypatch.setattr(repo_tags, "_run", counting_run)

        repo_tags.check_graphify_version_parity("graphify")
        repo_tags.compute_tag_to_repo_id(GRAPH_PATHS, REPO_IDS, graphify_bin="graphify")

        assert len(calls) == 1
