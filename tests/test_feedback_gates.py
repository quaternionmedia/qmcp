"""The feedback loop reads the harness's suite as pytest reports it, not as uv does.

`gate_findings` runs the suite under `uv run` and reads its output. uv writes
lines of its own to stderr -- "Installed 1 package in 12ms" when it syncs
first -- unless it is run `--quiet`; `tests/test_selfcheck.py` shows that on
the real uv. Here uv is stood in for by that one behaviour, so the reading is
checked without running a suite inside the suite.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import qmcp.feedback as feedback

SUMMARY = "1662 passed, 11 skipped in 200.01s"


def _uv(returncode: int, stdout: str):
    """A `subprocess.run` that answers as `uv run` does after a sync."""
    def run(argv, **kwargs):
        stderr = "" if "--quiet" in argv else "Installed 1 package in 12ms\n"
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr=stderr)
    return run


def test_a_passing_suite_is_summarised_by_pytests_own_line(monkeypatch) -> None:
    """Mutation seen red: `--quiet` removed from the argv -- the summary read
    "Installed 1 package in 12ms"."""
    monkeypatch.setattr(feedback.subprocess, "run", _uv(0, f"....\n{SUMMARY}\n"))

    (finding,) = feedback.gate_findings(Path("."))

    assert (finding.name, finding.state, finding.detail) == ("suite", "pass", SUMMARY)


def test_a_failing_suite_names_each_failure(monkeypatch) -> None:
    out = ("FAILED tests/test_hitl.py::test_expiry - AssertionError\n"
           "1 failed, 10 passed in 2.00s\n")
    monkeypatch.setattr(feedback.subprocess, "run", _uv(1, out))

    findings = feedback.gate_findings(Path("."))

    assert [(f.name, f.state) for f in findings] == [("test_expiry - AssertionError", "fail")]
