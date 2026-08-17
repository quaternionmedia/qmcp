"""The local demo runs, and it demonstrates the thing it claims to.

WHY A DEMO NEEDS A TEST. This demo was previously a sequence typed into a
session and pasted into a handoff page. Nobody else could reproduce it, and a
result nobody can reproduce is the same defect as a hand-run check reported as
CI. These tests run the real module, so `278 passed` covers the demo too and a
change that breaks it is caught by the suite rather than by someone trying it.

WHY THE ASSERTIONS ARE ABOUT THE AUDIT LOG. Recording an invocation is the code
path that was broken on `main` -- `ToolInvocation.execution_id` was `UUID NOT
NULL` against an insert supplying no id, and every call raised
`sqlite3.IntegrityError`. A test asserting only that the demo exits zero would
pass against a harness that recorded nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.demo_agent_harness import TOOLS_EXPECTED, main, run  # noqa: E402


def test_the_demo_records_every_call_it_makes():
    """Three calls in, three rows in the audit log. This is the demo's claim."""
    result = run(out=lambda *a: None)
    assert result["invocations_before"] == 0
    assert result["invocations_after"] == 3


def test_no_call_returns_an_error():
    result = run(out=lambda *a: None)
    assert result["errors"] == [None, None, None]


def test_every_recorded_invocation_succeeded():
    """Status comes from the audit record. `ToolInvokeResponse` has no status
    field, and reading one off it yields None for a call that worked."""
    result = run(out=lambda *a: None)
    assert [row["status"] for row in result["recorded"]] == ["success"] * 3


def test_the_harness_registers_the_four_tools():
    assert set(run(out=lambda *a: None)["tools"]) == set(TOOLS_EXPECTED)


def test_the_demo_starts_from_an_empty_database_every_run():
    """A demo that accumulated rows across runs would drift into passing for a
    reason that has nothing to do with the code -- and would mean it is writing
    somewhere permanent."""
    first = run(out=lambda *a: None)
    second = run(out=lambda *a: None)
    assert first["invocations_before"] == second["invocations_before"] == 0


def test_the_demo_leaves_the_working_database_alone(tmp_path):
    """`./qmcp.db` is the operator's. It must be byte-identical afterwards, and
    it must not be created if it was not there."""
    working = _REPO_ROOT / "qmcp.db"
    before = working.read_bytes() if working.is_file() else None
    run(out=lambda *a: None)
    after = working.read_bytes() if working.is_file() else None
    assert before == after


def test_the_demo_prints_what_it_established():
    """A demo whose output a reader cannot follow is a test with extra steps."""
    lines: list[str] = []
    run(out=lines.append)
    printed = "\n".join(lines)
    assert "tools registered" in printed
    assert "audit log" in printed
    assert "0 -> 3" in printed


def test_the_demo_exits_zero():
    assert main() == 0
