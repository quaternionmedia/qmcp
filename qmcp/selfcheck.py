"""qmcp validating its own repository, recorded the way it records anything else.

    qmcp selfcheck                 # run the checks, show what they found
    qmcp selfcheck --deltas        # the failures, as units of work
    qmcp selfcheck --json          # the run, as data

The checks are the gates this project is held to. Each is a real subprocess
against this working tree, and its start, duration, exit status and error text
are written to the database as a `ToolInvocation` -- the row the server writes
for a tool invoked over HTTP, and the row `qmcp dashboard` reads.

`to_delta` emits a delta for a failing check only. Its phase comes from facts,
as in `qmcp/cookbook/delta.py`:

    brainstorm      a check failed and nobody has been asked about it
    planning        a human was asked and answered

Nothing here reaches `implementation` or beyond: running a check again cannot
establish that somebody acted on it
(`governance/qm/records/DRAFT-a-disagreement-is-a-delta.md` 4).

What this cannot see: whether a check is the right check, or whether a passing
gate enforces anything. A check that reports success while enforcing nothing
reports success here too.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from qmcp.addresses import invocation_address
from qmcp.dashboard import DEFAULT_PROJECT
from qmcp.db.models import HumanRequest, HumanRequestStatus, InvocationStatus, ToolInvocation

# Where a check's captured output goes, so a later check can read it. The tag
# gate reads a test run rather than running one, so it can be pointed at a run
# somebody else captured.
CAPTURE = "selfcheck-suite.txt"

DELTA_SCHEMA = 1

BRAINSTORM = "brainstorm"
PLANNING = "planning"


@dataclass(frozen=True)
class Check:
    """One gate this repository is held to, and how to run it."""

    name: str
    what: str
    argv: tuple[str, ...]
    # Written to this path rather than to the terminal, when a later check
    # needs the text rather than the exit status.
    capture: str | None = None


@dataclass(frozen=True)
class Finding:
    """What one check established, and the row that proves it ran."""

    check: str
    what: str
    ok: bool
    detail: str
    invocation_id: str
    address: str
    duration_ms: int


def checks(capture_dir: Path) -> tuple[Check, ...]:
    """The gates, in the order a release would meet them.

    `tag-claims` comes after `suite` and reads what `suite` wrote: it judges
    a captured run rather than running one, so it cannot be satisfied by a run
    nobody kept.
    """
    gate = "governance/qm/project-seed/ci/check_tag_claims.py"
    return (
        Check(
            name="suite",
            what="every test this project ships",
            argv=("pytest", "-q"),
            capture=str(capture_dir / CAPTURE),
        ),
        Check(
            name="tag-claims",
            what="whether that run supports a release claim",
            argv=("python", gate, "--test-output", str(capture_dir / CAPTURE)),
        ),
        Check(
            name="walkthrough",
            what="whether the executable pages still execute",
            argv=("pytest", "walkthrough", "-q"),
        ),
    )


def last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "(no output)"


def run_check(check: Check, repo: Path, project: str) -> tuple[Finding, ToolInvocation]:
    """Run one check for real and build the invocation row that records it.

    The row is built and not committed here. Persisting is the caller's, so a
    caller can run the checks against a scratch database, or against none.
    """
    invocation = ToolInvocation(
        tool_name=check.name,
        input_params={"argv": list(check.argv), "repo": repo.name},
    )
    started = time.perf_counter()
    completed = subprocess.run(
        ["uv", "run", *check.argv], cwd=repo,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    elapsed = int((time.perf_counter() - started) * 1000)

    output = (completed.stdout or "") + (completed.stderr or "")
    if check.capture:
        Path(check.capture).write_text(output, encoding="utf-8", newline="\n")

    # The exit status, taken from the process rather than from its text, so a
    # wording change cannot silently disable the check.
    ok = completed.returncode == 0

    invocation.status = InvocationStatus.SUCCESS if ok else InvocationStatus.FAILED
    invocation.duration_ms = elapsed
    invocation.completed_at = datetime.now(UTC)
    invocation.result = {"exit_code": completed.returncode}
    if not ok:
        invocation.error = last_line(output)

    finding = Finding(
        check=check.name,
        what=check.what,
        ok=ok,
        detail=last_line(output),
        invocation_id=invocation.id,
        address=invocation_address(invocation.id, project),
        duration_ms=elapsed,
    )
    return finding, invocation


def delta_name(finding: Finding) -> str:
    """The delta's identity, and it is the check rather than the run.

    Re-running a failing check finds the same delta rather than opening a
    second one, as `DRAFT-a-disagreement-is-a-delta.md` 2 rules for
    divergence.
    """
    return f"{finding.check}-does-not-pass"


def phase_of(finding: Finding, answered: bool) -> str:
    """Where this finding stands, from facts alone.

    `brainstorm` until a human has been asked and has answered, then
    `planning`. Nothing goes further: re-running does not establish that
    anybody acted.
    """
    return PLANNING if answered else BRAINSTORM


def to_delta(finding: Finding, project: str, *, answered: bool = False) -> dict:
    """One failure as a delta the control panel ingests without translating.

    The keys inside `delta` are dossier's column names, so the consumer
    writes `ProjectDelta(**payload["delta"])`. A link points back at the
    invocation that found the failure.
    """
    return {
        "schema": DELTA_SCHEMA,
        "project": project,
        "delta": {
            "name": delta_name(finding),
            "title": f"{finding.check}: {finding.what}",
            "description": finding.detail,
            "phase": phase_of(finding, answered),
            "delta_type": "chore",
            "priority": "high",
        },
        "links": [
            {"link_type": "invocation", "target_id": None,
             "target_name": finding.invocation_id},
            {"link_type": "address", "target_id": None,
             "target_name": f"{project}/delta/{delta_name(finding)}"},
        ],
    }


def ask(finding: Finding, project: str) -> HumanRequest:
    """The question a failing check raises, as a row in the human-in-the-loop queue.

    A failing gate is not a decision. Whether to fix it, wait, or accept it is a
    person's, and this is where that gets asked rather than assumed.
    """
    return HumanRequest(
        id=f"selfcheck-{finding.check}",
        request_type="decision",
        prompt=(f"{finding.check} does not pass: {finding.detail}. "
                f"Fix it, accept it, or defer it?"),
        options=["fix", "accept", "defer"],
        context={"address": finding.address, "project": project,
                 "delta": delta_name(finding)},
        status=HumanRequestStatus.PENDING,
    )


def render(findings: list[Finding], project: str = DEFAULT_PROJECT) -> str:
    """The run, for a terminal. Reads the findings and runs nothing."""
    out = [f"qmcp selfcheck  {project}", ""]
    for finding in findings:
        mark = "[ok]" if finding.ok else "[!!]"
        out.append(f"  {mark} {finding.check:<14} {finding.what}")
        out.append(f"       {finding.duration_ms:>6}ms  {finding.address}")
        out.append(f"       {finding.detail}")
        out.append("")

    failed = [f for f in findings if not f.ok]
    if not failed:
        out += ["  Every check passed. No unit of work follows from a green gate.",
                ""]
        return "\n".join(out)

    out.append(f"  {len(failed)} check(s) did not pass, and each is a unit of work:")
    for finding in failed:
        out.append(f"    {delta_name(finding)}  ->  {project}/delta/{delta_name(finding)}")
    out += [
        "",
        "  They open at `brainstorm`. A failing gate is a thing somebody noticed,",
        "  not a thing anybody decided -- so the phase says noticed, and moves",
        "  when a human answers the question in the queue.",
        "",
        "  `qmcp selfcheck --deltas` emits them for `dossier deltas ingest`.",
    ]
    return "\n".join(out)
