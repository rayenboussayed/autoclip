"""Offline tests for the suite's own hooks: opt-in gating, GitHub annotations and the e2e session folder."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from .. import conftest
from ..e2e import conftest as e2e_conftest
from ..e2e import harness
from ..e2e.harness import E2ERun


@dataclass
class FakeItem:
    """The only part of a pytest item the gating hook touches."""

    keywords: set[str]
    markers: list[Any] = field(default_factory=list)

    def add_marker(self, marker: Any) -> None:
        """Record a marker exactly like ``pytest.Item`` would."""
        self.markers.append(marker)


@dataclass
class FakeReport:
    """The only part of a pytest report the annotation and recording hooks touch."""

    failed: bool
    when: str
    location: tuple[str, int | None, str]
    longrepr: str
    nodeid: str
    outcome: str = "failed"


def gate(items: list[FakeItem]) -> None:
    """Run the collection hook over fake items."""
    conftest.pytest_collection_modifyitems(cast(pytest.Config, object()), cast(list[pytest.Item], items))


def annotate(report: FakeReport) -> None:
    """Run the root annotation hook over a fake report."""
    conftest.pytest_runtest_logreport(cast(pytest.TestReport, report))


def record(report: FakeReport) -> None:
    """Run the e2e recording hook over a fake report."""
    e2e_conftest.pytest_runtest_logreport(cast(pytest.TestReport, report))


def error_lines(output: str) -> list[str]:
    """The annotation commands in captured stdout (the leading blank line separates them from progress)."""
    return [line for line in output.splitlines() if line.startswith("::error")]


@pytest.mark.parametrize(
    ("env", "e2e_skipped", "network_skipped"),
    [
        ({}, True, True),
        ({"AUTOCLIP_NETWORK_TESTS": "1"}, True, False),
        ({"AUTOCLIP_E2E_TESTS": "1"}, False, False),
        ({"AUTOCLIP_E2E_TESTS": "1", "AUTOCLIP_NETWORK_TESTS": "0"}, False, False),
        ({"AUTOCLIP_NETWORK_TESTS": "true"}, True, True),
        ({"AUTOCLIP_E2E_TESTS": "yes"}, True, True),
        ({"AUTOCLIP_NETWORK_TESTS": "0"}, True, True),
    ],
)
def test_gating_skips_e2e_and_network_unless_opted_in(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], e2e_skipped: bool, network_skipped: bool
) -> None:
    monkeypatch.delenv("AUTOCLIP_E2E_TESTS", raising=False)
    monkeypatch.delenv("AUTOCLIP_NETWORK_TESTS", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    e2e = FakeItem({"e2e", "network"})
    network = FakeItem({"network"})
    plain = FakeItem(set())
    gate([e2e, network, plain])
    assert bool(e2e.markers) is e2e_skipped
    assert bool(network.markers) is network_skipped
    assert plain.markers == []
    if e2e_skipped:
        assert "AUTOCLIP_E2E_TESTS=1" in str(e2e.markers[0].kwargs["reason"])
    if network_skipped:
        assert "AUTOCLIP_NETWORK_TESTS=1" in str(network.markers[0].kwargs["reason"])


def test_e2e_tests_enable_the_network_marker_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOCLIP_E2E_TESTS", "1")
    monkeypatch.delenv("AUTOCLIP_NETWORK_TESTS", raising=False)
    network = FakeItem({"network"})
    gate([network])
    assert network.markers == []


def test_annotation_escapes_github_commands_and_clamps_the_line(capsys: pytest.CaptureFixture[str]) -> None:
    conftest._annotation("tests/unit/test_x.py", 0, "100%\r\nnext é")
    output = capsys.readouterr().out
    assert output == "\n::error file=tests/unit/test_x.py,line=1::100%25%0D%0Anext \\xe9\n"
    conftest._annotation("a b.py", 42, "ok")
    assert error_lines(capsys.readouterr().out) == ["::error file=a b.py,line=42::ok"]


def test_annotations_are_only_emitted_for_failures_on_github(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    report = FakeReport(False, "call", ("tests/unit/test_x.py", 3, "test_x"), "boom", "tests/unit/test_x.py::test_x")
    annotate(report)
    assert capsys.readouterr().out == ""

    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    annotate(FakeReport(True, "call", ("tests/unit/test_x.py", 3, "test_x"), "boom", "tests/unit/test_x.py::test_x"))
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("GITHUB_ACTIONS", "false")
    annotate(FakeReport(True, "call", ("tests/unit/test_x.py", 3, "test_x"), "boom", "tests/unit/test_x.py::test_x"))
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    annotate(FakeReport(False, "call", ("tests/unit/test_x.py", 3, "test_x"), "boom", "tests/unit/test_x.py::test_x"))
    assert capsys.readouterr().out == ""


def test_a_failure_is_annotated_with_a_summary_and_the_full_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    report = FakeReport(True, "call", ("tests/unit/test_x.py", 4, "test_x"), "line one\n\nlast line\n\n", "tests/unit/test_x.py::test_x")
    annotate(report)
    lines = error_lines(capsys.readouterr().out)
    assert lines == [
        "::error file=tests/unit/test_x.py,line=5::tests/unit/test_x.py::test_x (call): last line",
        "::error file=tests/unit/test_x.py,line=5::line one%0A%0Alast line%0A%0A",
    ]


def test_a_report_without_a_line_starts_at_line_one(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    annotate(FakeReport(True, "setup", ("tests/unit/test_x.py", None, "test_x"), "boom", "tests/unit/test_x.py::test_x"))
    assert error_lines(capsys.readouterr().out)[0].startswith("::error file=tests/unit/test_x.py,line=1::")


def test_long_messages_are_truncated_per_annotation(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    text = "x" * 5000
    annotate(FakeReport(True, "call", ("tests/unit/test_x.py", 0, "test_x"), text, "tests/unit/test_x.py::test_x"))
    first, second = error_lines(capsys.readouterr().out)
    prefix = "::error file=tests/unit/test_x.py,line=1::tests/unit/test_x.py::test_x (call): "
    assert first == prefix + "x" * 200
    assert second == "::error file=tests/unit/test_x.py,line=1::" + "x" * 4000


def test_an_empty_traceback_still_produces_two_annotations(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    annotate(FakeReport(True, "teardown", ("tests/unit/test_x.py", 7, "test_x"), "", "tests/unit/test_x.py::test_x"))
    assert error_lines(capsys.readouterr().out) == [
        "::error file=tests/unit/test_x.py,line=8::tests/unit/test_x.py::test_x (teardown): ",
        "::error file=tests/unit/test_x.py,line=8::",
    ]


def test_the_e2e_run_fixture_creates_the_folder_environment_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOCLIP_E2E_RESULTS", str(tmp_path / "results"))
    fixture = cast(Any, e2e_conftest.e2e_run).__wrapped__()
    run = cast(E2ERun, next(fixture))
    try:
        assert run.directory.parent == (tmp_path / "results").resolve()
        assert e2e_conftest._HOLDER.run is run
        environment = json.loads((run.directory / "environment.json").read_text(encoding="utf-8"))
        assert environment["directory"] == str(run.directory)
        assert environment["flags"]["AUTOCLIP_E2E_RESULTS"] == str(tmp_path / "results")
    finally:
        with pytest.raises(StopIteration):
            next(fixture)
    assert e2e_conftest._HOLDER.run is None
    summary = json.loads((run.directory / "run.json").read_text(encoding="utf-8"))
    assert summary["directory"] == str(run.directory)
    assert summary["cases"] == [] and summary["outcomes"] == {}


def test_the_e2e_recording_hook_ignores_reports_between_runs(tmp_path: Path) -> None:
    report = FakeReport(True, "call", ("tests/e2e/test_cli_e2e.py", 3, "test_x"), "boom", "tests/e2e/test_cli_e2e.py::test_x")
    e2e_conftest._HOLDER.run = None
    record(report)
    run = E2ERun(tmp_path, "now")
    e2e_conftest._HOLDER.run = run
    try:
        record(report)
        record(
            FakeReport(
                False, "setup", ("tests/e2e/test_cli_e2e.py", 3, "test_x"), "", "tests/e2e/test_cli_e2e.py::test_x", outcome="passed"
            )
        )
    finally:
        e2e_conftest._HOLDER.run = None
    assert run.outcomes == {"tests/e2e/test_cli_e2e.py::test_x": {"call": "failed", "setup": "passed"}}


def test_the_manifest_and_bbc_fixtures_share_the_manifest_and_honour_the_override(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = cast(Any, e2e_conftest.manifest).__wrapped__()
    assert len(entries) == 10
    monkeypatch.delenv("AUTOCLIP_E2E_URL", raising=False)
    assert cast(Any, e2e_conftest.bbc_entry).__wrapped__(entries) == harness.choose_bbc_entry(entries)
    monkeypatch.setenv("AUTOCLIP_E2E_URL", "https://youtu.be/custom")
    overridden = cast(Any, e2e_conftest.bbc_entry).__wrapped__(entries)
    assert overridden.url == "https://youtu.be/custom"
