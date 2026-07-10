# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Subprocess (``pytester``) tests for cross-worker aggregation under pytest-xdist.

These tests spawn real child pytest sessions via the ``pytester`` fixture to
exercise the full xdist serialization → merge → emission pipeline. They touch
no live external dependency, but each spins up one or more subprocess runs, so
they are marked ``slow`` and can be deselected with ``-m 'not slow'``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from _pytest.pytester import Pytester, RunResult


pytest_plugins = ["pytester"]

pytestmark = pytest.mark.slow


_CONFTEST = """\
from pathlib import Path

import pytest

from rampart.reporting import JsonFileReportSink


_OUT_DIR = Path("rampart_reports").absolute()


@pytest.fixture(scope="session")
def rampart_sinks():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("rampart_report_dir.txt").write_text(str(_OUT_DIR))
    return [JsonFileReportSink(output_dir=_OUT_DIR)]
"""


def _load_reports(pytester: Pytester) -> list[dict[str, Any]]:
    marker = pytester.path / "rampart_report_dir.txt"
    if not marker.exists():
        default_dir = pytester.path / "rampart_reports"
        if default_dir.exists():
            return [
                json.loads(p.read_text())
                for p in sorted(default_dir.glob("run_report_*.json"))
            ]
        return []
    out_dir = Path(marker.read_text().strip())
    if not out_dir.exists():
        return []
    return [
        json.loads(p.read_text()) for p in sorted(out_dir.glob("run_report_*.json"))
    ]


def _setup_simple_tests(pytester: Pytester) -> None:
    pytester.makeconftest(_CONFTEST)
    pytester.makepyfile(  # pyright: ignore[reportUnknownMemberType]
        test_a="""
        import pytest
        from rampart import record_result
        from rampart.core.result import Result, SafetyStatus
        from rampart.core.types import ObservabilityLevel

        @pytest.mark.harm("test")
        def test_a_one():
            record_result(Result(
                safe=True, status=SafetyStatus.SAFE, summary="a1",
                observability_level=ObservabilityLevel.RESPONSE_ONLY,
            ))

        @pytest.mark.harm("test")
        def test_a_two():
            record_result(Result(
                safe=False, status=SafetyStatus.UNSAFE, summary="a2",
                observability_level=ObservabilityLevel.RESPONSE_ONLY,
            ))
        """,
        test_b="""
        import pytest
        from rampart import record_result
        from rampart.core.result import Result, SafetyStatus
        from rampart.core.types import ObservabilityLevel

        @pytest.mark.harm("test")
        def test_b_one():
            record_result(Result(
                safe=True, status=SafetyStatus.SAFE, summary="b1",
                observability_level=ObservabilityLevel.RESPONSE_ONLY,
            ))

        @pytest.mark.harm("test")
        def test_b_two():
            record_result(Result(
                safe=True, status=SafetyStatus.SAFE, summary="b2",
                observability_level=ObservabilityLevel.RESPONSE_ONLY,
            ))
        """,
    )


class TestSingleProcessBaseline:
    def test_baseline_emits_one_report(self, pytester: Pytester) -> None:
        _setup_simple_tests(pytester)
        result = pytester.runpytest("-p", "no:cacheprovider")
        result.assert_outcomes(passed=4)
        reports = _load_reports(pytester)
        assert len(reports) == 1
        assert reports[0]["total_runs"] == 4


class TestXdistConsolidation:
    def test_xdist_emits_single_consolidated_report(
        self,
        pytester: Pytester,
    ) -> None:
        _setup_simple_tests(pytester)
        result = pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
        )
        result.assert_outcomes(passed=4)
        reports = _load_reports(pytester)
        assert len(reports) == 1, (
            f"Expected exactly one report under xdist, got {len(reports)}: "
            f"{[r.get('total_runs') for r in reports]}"
        )

    def test_population_statistics_over_full_set(
        self,
        pytester: Pytester,
    ) -> None:
        _setup_simple_tests(pytester)
        pytester.runpytest("-p", "no:cacheprovider", "-n", "2")
        reports = _load_reports(pytester)
        assert len(reports) == 1
        report = reports[0]
        assert report["total_runs"] == 4
        assert report["passed"] == 3
        assert report["failed"] == 1
        assert report["population_summary"]["total_runs"] == 4
        assert report["population_summary"]["safe_count"] == 3
        assert report["population_summary"]["unsafe_count"] == 1


class TestXdistTrialAggregation:
    def test_trial_aggregation_across_workers_loadgroup(
        self,
        pytester: Pytester,
    ) -> None:
        pytester.makeconftest(_CONFTEST)
        pytester.makepyfile(  # pyright: ignore[reportUnknownMemberType]
            test_trial="""
            import pytest
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus
            from rampart.core.types import ObservabilityLevel

            @pytest.mark.harm("test")
            @pytest.mark.trial(n=4, threshold=0.5)
            def test_trial_split():
                record_result(Result(
                    safe=True, status=SafetyStatus.SAFE, summary="t",
                    observability_level=ObservabilityLevel.RESPONSE_ONLY,
                ))
            """,
        )
        result = pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
            "--dist",
            "loadgroup",
        )
        result.assert_outcomes(passed=4)
        reports = _load_reports(pytester)
        assert len(reports) == 1
        assert reports[0]["total_runs"] == 4

    def test_trial_aggregation_across_workers_load(
        self,
        pytester: Pytester,
    ) -> None:
        pytester.makeconftest(_CONFTEST)
        pytester.makepyfile(  # pyright: ignore[reportUnknownMemberType]
            test_trial="""
            import pytest
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus
            from rampart.core.types import ObservabilityLevel

            @pytest.mark.harm("test")
            @pytest.mark.trial(n=4, threshold=0.5)
            def test_trial_split():
                record_result(Result(
                    safe=True, status=SafetyStatus.SAFE, summary="t",
                    observability_level=ObservabilityLevel.RESPONSE_ONLY,
                ))
            """,
        )
        result = pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
            "--dist",
            "load",
        )
        result.assert_outcomes(passed=4)
        reports = _load_reports(pytester)
        assert len(reports) == 1
        assert reports[0]["total_runs"] == 4

    def test_trial_group_fails_when_any_unsafe_under_loadgroup(
        self,
        pytester: Pytester,
    ) -> None:
        """An UNSAFE trial fails the whole group regardless of pass rate.

        Trial body switches on the clone name (``[trial-0]``..``[trial-3]``)
        so the same outcome distribution is produced regardless of which
        worker executes the clone. Three trials are SAFE and one is UNSAFE;
        with threshold=0.5 the group would otherwise pass on rate alone,
        so the only way the group can FAIL is if controller-side
        aggregation correctly merged the worker results.
        """
        pytester.makeconftest(_CONFTEST)
        pytester.makepyfile(  # pyright: ignore[reportUnknownMemberType]
            test_trial_mixed="""
            import pytest
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus
            from rampart.core.types import ObservabilityLevel

            @pytest.mark.harm("test")
            @pytest.mark.trial(n=4, threshold=0.5)
            def test_trial_mixed(request):
                # Trial-3 is UNSAFE; the rest are SAFE. With threshold=0.5
                # the group MUST FAIL on the unconditional unsafe rule.
                unsafe = request.node.name.endswith("[trial-3]")
                record_result(Result(
                    safe=not unsafe,
                    status=SafetyStatus.UNSAFE if unsafe else SafetyStatus.SAFE,
                    summary="u" if unsafe else "s",
                    observability_level=ObservabilityLevel.RESPONSE_ONLY,
                ))
            """,
        )
        result = pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
            "--dist",
            "loadgroup",
        )
        # All 4 clones pass at the pytest item level — record_result
        # does not fail the test; it only records a Result.
        result.assert_outcomes(passed=4)
        reports = _load_reports(pytester)
        assert len(reports) == 1
        report = reports[0]
        assert report["total_runs"] == 4
        assert report["passed"] == 3
        assert report["failed"] == 1
        # The trial-group FAIL line proves the controller correctly
        # aggregated worker results. The bracketed stats uniquely
        # identify the group line (the per-clone lines lack them).
        summary = "\n".join(result.outlines)
        assert "RAMPART Safety Summary" in summary
        assert (
            "FAIL  test_trial_mixed [3/4 safe, 75% pass rate, threshold: 50%]"
            in summary
        )

    def test_trial_group_fails_when_any_unsafe_under_load(
        self,
        pytester: Pytester,
    ) -> None:
        """Same as above but with --dist=load so clones may split workers.

        The PR docs claim aggregation remains correct under --dist=load
        because the controller merges all worker results. This test
        protects that contract: an UNSAFE clone produced on any worker
        must propagate into the controller's trial-group verdict.
        """
        pytester.makeconftest(_CONFTEST)
        pytester.makepyfile(  # pyright: ignore[reportUnknownMemberType]
            test_trial_mixed_load="""
            import pytest
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus
            from rampart.core.types import ObservabilityLevel

            @pytest.mark.harm("test")
            @pytest.mark.trial(n=4, threshold=0.5)
            def test_trial_mixed_load(request):
                unsafe = request.node.name.endswith("[trial-3]")
                record_result(Result(
                    safe=not unsafe,
                    status=SafetyStatus.UNSAFE if unsafe else SafetyStatus.SAFE,
                    summary="u" if unsafe else "s",
                    observability_level=ObservabilityLevel.RESPONSE_ONLY,
                ))
            """,
        )
        result = pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
            "--dist",
            "load",
        )
        result.assert_outcomes(passed=4)
        reports = _load_reports(pytester)
        assert len(reports) == 1
        report = reports[0]
        assert report["total_runs"] == 4
        assert report["failed"] == 1
        summary = "\n".join(result.outlines)
        assert (
            "FAIL  test_trial_mixed_load [3/4 safe, 75% pass rate, threshold: 50%]"
            in summary
        )

    def test_trial_group_fails_below_threshold_under_loadgroup(
        self,
        pytester: Pytester,
    ) -> None:
        """No UNSAFE results, but pass rate below threshold => FAIL.

        2 SAFE + 2 UNDETERMINED trials, threshold=0.75. Pass rate is 0.5
        so the group must FAIL on the threshold rule (not the unsafe rule).
        """
        pytester.makeconftest(_CONFTEST)
        pytester.makepyfile(  # pyright: ignore[reportUnknownMemberType]
            test_trial_threshold="""
            import pytest
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus
            from rampart.core.types import ObservabilityLevel

            @pytest.mark.harm("test")
            @pytest.mark.trial(n=4, threshold=0.75)
            def test_trial_threshold(request):
                undetermined = request.node.name.endswith(
                    ("[trial-2]", "[trial-3]"),
                )
                record_result(Result(
                    safe=True,
                    status=(
                        SafetyStatus.UNDETERMINED
                        if undetermined else SafetyStatus.SAFE
                    ),
                    summary="t",
                    observability_level=ObservabilityLevel.RESPONSE_ONLY,
                ))
            """,
        )
        result = pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
            "--dist",
            "loadgroup",
        )
        # All 4 clones pass as pytest tests (record_result(safe=True)),
        # but the trial GROUP should fail on threshold.
        result.assert_outcomes(passed=4)
        summary = "\n".join(result.outlines)
        assert "FAIL  test_trial_threshold" in summary
        assert "50% pass rate" in summary
        assert "threshold: 75%" in summary

    def test_trial_group_passes_when_all_safe_under_loadgroup(
        self,
        pytester: Pytester,
    ) -> None:
        """All-SAFE trial group with achievable threshold => PASS verdict."""
        pytester.makeconftest(_CONFTEST)
        pytester.makepyfile(  # pyright: ignore[reportUnknownMemberType]
            test_trial_all_safe="""
            import pytest
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus
            from rampart.core.types import ObservabilityLevel

            @pytest.mark.harm("test")
            @pytest.mark.trial(n=3, threshold=0.5)
            def test_trial_all_safe():
                record_result(Result(
                    safe=True, status=SafetyStatus.SAFE, summary="ok",
                    observability_level=ObservabilityLevel.RESPONSE_ONLY,
                ))
            """,
        )
        result = pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
            "--dist",
            "loadgroup",
        )
        result.assert_outcomes(passed=3)
        summary = "\n".join(result.outlines)
        assert "PASS  test_trial_all_safe" in summary
        assert "PASSED" in summary


class TestXdistMetadata:
    def test_report_includes_xdist_metadata(self, pytester: Pytester) -> None:
        _setup_simple_tests(pytester)
        pytester.runpytest("-p", "no:cacheprovider", "-n", "2")
        reports = _load_reports(pytester)
        assert len(reports) == 1
        metadata = reports[0].get("metadata", {})
        assert metadata.get("xdist_active") is True
        assert metadata.get("worker_count") == 2
        assert "dist_mode" in metadata
        assert "population_summary" in reports[0]

    def test_size_cap_marks_run_incomplete(self, pytester: Pytester) -> None:
        """Forcing a 1-byte cap surfaces incompleteness in report metadata.

        Under the durable shard transport the cap applies per record, so a
        1-byte cap drops every record on each worker; the controller must
        record ``incomplete=True`` plus a per-worker reason naming the
        dropped records in the merged report.
        """
        _setup_simple_tests(pytester)
        pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
            "--rampart-xdist-max-bytes=1",
        )
        reports = _load_reports(pytester)
        assert len(reports) == 1
        metadata = reports[0].get("metadata", {})
        assert metadata.get("incomplete") is True
        reasons = metadata.get("incomplete_reasons", [])
        assert any("dropped" in r for r in reasons)


class TestCollectOnly:
    def test_collect_only_does_not_emit_reports(self, pytester: Pytester) -> None:
        _setup_simple_tests(pytester)
        pytester.runpytest("-p", "no:cacheprovider", "--collect-only")
        # No sinks emit when no tests run
        marker = pytester.path / "rampart_report_dir.txt"
        if marker.exists():
            out_dir = Path(marker.read_text().strip())
            if out_dir.exists():
                reports = list(out_dir.glob("run_report_*.json"))
                assert reports == []


class TestCloneIdDeterminism:
    def test_trial_clone_ids_deterministic_across_processes(
        self,
        pytester: Pytester,
    ) -> None:
        pytester.makeconftest(_CONFTEST)
        pytester.makepyfile(  # pyright: ignore[reportUnknownMemberType]
            test_det="""
            import pytest

            @pytest.mark.trial(n=3)
            def test_x():
                pass
            """,
        )
        result_serial: RunResult = pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "--collect-only",
            "-q",
        )
        result_parallel: RunResult = pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "--collect-only",
            "-q",
            "-n",
            "2",
        )

        def _trial_ids(lines: list[str]) -> list[str]:
            return sorted(line.strip() for line in lines if "trial-" in line)

        serial_ids = _trial_ids(result_serial.outlines)
        parallel_ids = _trial_ids(result_parallel.outlines)
        # Under xdist --collect-only, both should produce the same
        # deterministic clone IDs so that workers can match them.
        if serial_ids and parallel_ids:
            assert serial_ids == parallel_ids


class TestShardCrashRecovery:
    def test_worker_crash_preserves_finished_results(
        self,
        pytester: Pytester,
    ) -> None:
        """A worker killed mid-run keeps the results it already flushed.

        Under the durable shard transport each finished result is streamed
        to disk, so an abrupt ``os._exit`` (simulating a SIGKILL) after
        three of four tests still lets the controller recover those three
        and mark the run incomplete. The legacy inline transport flushed
        only at a clean ``pytest_sessionfinish``, so a crash there lost
        every result the worker had produced -- this is the fix for that
        durability gap.
        """
        pytester.makeconftest(_CONFTEST)
        pytester.makepyfile(  # pyright: ignore[reportUnknownMemberType]
            test_crash="""
            import os

            import pytest
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus
            from rampart.core.types import ObservabilityLevel

            def _record(summary):
                record_result(Result(
                    safe=True, status=SafetyStatus.SAFE, summary=summary,
                    observability_level=ObservabilityLevel.RESPONSE_ONLY,
                ))

            @pytest.mark.harm("test")
            def test_c0():
                _record("c0")

            @pytest.mark.harm("test")
            def test_c1():
                _record("c1")

            @pytest.mark.harm("test")
            def test_c2():
                _record("c2")

            @pytest.mark.harm("test")
            def test_c3_crash():
                # Record, then die before the fixture can flush this one.
                _record("c3")
                os._exit(1)
            """,
        )
        pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "1",
        )
        reports = _load_reports(pytester)
        assert len(reports) == 1
        report = reports[0]
        # The three results flushed before the crash survive on disk; the
        # in-flight fourth (killed before its fixture teardown) does not.
        assert report["total_runs"] == 3
        metadata = report.get("metadata", {})
        assert metadata.get("incomplete") is True
        reasons = metadata.get("incomplete_reasons", [])
        assert any("recovered" in r for r in reasons)


class TestShardGranularSizeCap:
    def test_oversized_record_dropped_siblings_survive(
        self,
        pytester: Pytester,
    ) -> None:
        """A single oversized record is dropped without losing its siblings.

        The per-record cap drops only the record that exceeds it; the other
        results the same worker produced are still recovered. The legacy
        inline transport dropped the worker's *entire* payload when the
        aggregate exceeded the cap -- this is the fix for that granularity
        gap. A cap of 4 KiB keeps the small results but drops the one whose
        summary is padded past the limit.
        """
        pytester.makeconftest(_CONFTEST)
        pytester.makepyfile(  # pyright: ignore[reportUnknownMemberType]
            test_sizecap="""
            import pytest
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus
            from rampart.core.types import ObservabilityLevel

            def _record(summary):
                record_result(Result(
                    safe=True, status=SafetyStatus.SAFE, summary=summary,
                    observability_level=ObservabilityLevel.RESPONSE_ONLY,
                ))

            @pytest.mark.harm("test")
            def test_s0():
                _record("s0")

            @pytest.mark.harm("test")
            def test_s1():
                _record("s1")

            @pytest.mark.harm("test")
            def test_s2_oversized():
                _record("x" * 8192)

            @pytest.mark.harm("test")
            def test_s3():
                _record("s3")
            """,
        )
        pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "1",
            "--rampart-xdist-max-bytes=4096",
        )
        reports = _load_reports(pytester)
        assert len(reports) == 1
        report = reports[0]
        # Three small results survive; only the oversized one is dropped.
        assert report["total_runs"] == 3
        metadata = report.get("metadata", {})
        assert metadata.get("incomplete") is True
        reasons = metadata.get("incomplete_reasons", [])
        assert any("dropped" in r for r in reasons)
