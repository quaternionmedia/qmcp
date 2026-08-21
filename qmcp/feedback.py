"""The loop that improves the loop: diagnose, record, and ratchet.

    uv run qmcp feedback run

**THIS IS A PIPELINE TOPOLOGY, AND IT IS THE FIRST ONE IN THE REGISTRY THAT
RUNS.** `BaseTopology.run` raises `NotImplementedError("Topology runtime is not
implemented yet")` and every registered topology inherits it, so the manager has
so far been a vocabulary and a schema rather than a runtime. Nothing here
changes that for the other seven: this subclass implements `run` for one shape,
which is what the base class being abstract is for.

**THE STAGES ARE THE TWO SELF-CHECKS, AND THEY LOOK DIFFERENT WAYS.**
`qmcp.selfcheck` runs the gates -- the suite, the tag claims, the walkthroughs
-- and answers "does this pass". `dossier.diagnostics` reads the pair for the
defect classes a green suite does not see, and answers "is a passing gate
enforcing anything". Its own docstring is where that question is asked, and
running the two together is the answer.

**THE RATCHET IS THE POINT, AND IT IS THE ONLY PART THAT IMPROVES ANYTHING.**
Running checks repeatedly finds recurrences, which is worth something and is not
improvement. What makes the next run better than this one is that a defect
nobody could have caught becomes a check that would have caught it. So a run
reports `unratcheted`: failures with no corresponding check, which is the queue
of work that would make the loop stronger rather than merely busier.

**IT DOES NOT SPEND, SCHEDULE ITSELF, OR CLOSE ANYTHING.** No agent is
dispatched: the two stages are subprocesses this project already runs, and the
model on this machine is local. Nothing here writes a cron entry --
`governance/qm/records/DRAFT-no-unattended-spending.md`, and a loop that ran
itself would be exactly the unattended thing that record is about. A failure
becomes a delta at `brainstorm` and stops; convergence is a person's.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qmcp.agentframework.models.enums import TopologyType
from qmcp.agentframework.topologies import BaseTopology, topology

# The two stages, and which question each answers. Named here rather than
# configured, because a feedback loop whose stages are a setting is one that can
# be quietly reduced to the stage that always passes.
STAGES = (
    ("gates", "does this pass"),
    ("diagnostics", "is a passing gate enforcing anything"),
)

BRAINSTORM = "brainstorm"


@dataclass(frozen=True)
class Finding:
    """One thing a stage reported, in the vocabulary both stages share."""

    stage: str
    name: str
    state: str
    detail: str = ""

    @property
    def is_failure(self) -> bool:
        return self.state in ("fail", "failed", "error")

    @property
    def is_unknown(self) -> bool:
        return self.state == "unknown"


@dataclass
class Run:
    """One pass of the loop, and what it leaves behind."""

    findings: list[Finding] = field(default_factory=list)
    ran: list[str] = field(default_factory=list)
    """Stages that actually executed. A stage missing from here did not run,
    which is not the same as a stage that ran and found nothing."""

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.is_failure]

    @property
    def unknowns(self) -> list[Finding]:
        return [f for f in self.findings if f.is_unknown]

    @property
    def is_clean(self) -> bool:
        """Both stages ran, neither failed, nothing was unknown."""
        return (len(self.ran) == len(STAGES)
                and not self.failures and not self.unknowns)

    def deltas(self, project: str) -> list[dict[str, Any]]:
        """Failures as units of work. Passing checks are not work.

        `brainstorm` and no further, for the same reason `qmcp.selfcheck` stops
        there: running a check again cannot establish that anybody acted on it.
        """
        return [
            {
                "schema": 1,
                "project": project,
                "perspective": "qmcp/feedback",
                "delta": {
                    "name": f"feedback-{f.stage}-{f.name}",
                    "title": f"{f.stage}: {f.name}",
                    "description": f.detail,
                    "phase": BRAINSTORM,
                    "delta_type": "defect",
                    "priority": "medium",
                },
            }
            for f in self.failures
        ]

    def summary(self) -> str:
        if not self.ran:
            return "no stage ran, which is the finding"
        missing = [name for name, _ in STAGES if name not in self.ran]
        parts = [f"{len(self.ran)} of {len(STAGES)} stages ran"]
        if missing:
            parts.append(f"{', '.join(missing)} did not")
        parts.append(f"{len(self.findings)} checks")
        if self.failures:
            parts.append(f"{len(self.failures)} failed")
        if self.unknowns:
            parts.append(f"{len(self.unknowns)} unknown")
        if self.is_clean:
            parts.append("clean")
        return ", ".join(parts)


def unratcheted(run: Run, known_classes: set[str]) -> list[Finding]:
    """Failures that no check was written for.

    **THIS IS THE IMPROVEMENT, AND EVERYTHING ELSE IS OBSERVATION.** A failure
    that a check already covers is a recurrence: unwelcome, and the loop is
    working. A failure that nothing covers is a hole -- somebody found it by
    hand, or a person hit it -- and closing it means writing the check, not
    just the fix. That queue is what makes the next run stronger than this one.

    `known_classes` is the set of check names, so this is the failures whose
    name nothing recognises.
    """
    return [f for f in run.failures if f.name not in known_classes]


def _read(argv: list[str], cwd: Path, timeout: float = 900.0) -> tuple[int, str]:
    try:
        done = subprocess.run(argv, cwd=str(cwd), capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, f"{type(exc).__name__}: {exc}"
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def diagnostics_findings(panel: Path) -> list[Finding]:
    """The panel's inward checks, read as findings.

    Run as a subprocess in the panel's own environment rather than imported.
    The harness does not depend on the panel and must not start: importing
    across would make the seam a requirement, and the panel's virtualenv is
    where its dependencies are.
    """
    script = ("import json;"
              "from dossier.diagnostics import run;"
              "print(json.dumps([{'name': r.name, 'state': r.state,"
              " 'detail': r.detail} for r in run().results]))")
    code, out = _read(["uv", "run", "--no-sync", "python", "-c", script], panel)
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("["):
            try:
                rows = json.loads(line)
            except ValueError:
                break
            return [Finding("diagnostics", r["name"], r["state"],
                            r.get("detail", "")) for r in rows]
    return [Finding("diagnostics", "could-not-run", "unknown",
                    f"exit {code}: {out.strip().splitlines()[-1] if out.strip() else ''}")]


def gate_findings(harness: Path) -> list[Finding]:
    """The harness's gates, read as findings."""
    code, out = _read(["uv", "run", "pytest", "-q", "--no-header"], harness)
    if code == 0:
        return [Finding("gates", "suite", "pass",
                        out.strip().splitlines()[-1] if out.strip() else "")]
    failed = [line for line in out.splitlines() if line.startswith("FAILED")]
    if not failed:
        return [Finding("gates", "suite", "fail", f"exit {code}")]
    return [Finding("gates", line.split("::")[-1][:60], "fail", line[:120])
            for line in failed]


@topology
class FeedbackTopology(BaseTopology):
    """The improvement loop, as a pipeline the manager knows about.

    Registered so it is discoverable beside the other seven, and implemented so
    that at least one of them does something. The `run` here takes paths rather
    than agents: no agent is dispatched, which is a fact about this loop worth
    it being awkward to hide.
    """

    topology_type = TopologyType.PIPELINE

    def __init__(self, *args, **kwargs) -> None:  # noqa: D107
        # Constructed directly in the plain case, and through the registry in
        # the configured one. Both are allowed; only the first is used today.
        if args or kwargs:
            super().__init__(*args, **kwargs)

    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """One pass. Returns what it found rather than acting on it."""
        harness = Path(input_data.get("harness") or ".")
        panel = Path(input_data["panel"])
        found = once(harness=harness, panel=panel)
        return {
            "summary": found.summary(),
            "ran": list(found.ran),
            "failures": [f.name for f in found.failures],
            "unknowns": [f.name for f in found.unknowns],
        }


def once(harness: Path, panel: Path) -> Run:
    """Run both stages. A stage that cannot run is absent, never assumed green.

    The stages are independent on purpose: the diagnostics are worth having
    when the suite is red, and that is exactly when somebody wants them.
    """
    found = Run()

    if panel.is_dir():
        found.findings.extend(diagnostics_findings(panel))
        found.ran.append("diagnostics")
    if harness.is_dir():
        found.findings.extend(gate_findings(harness))
        found.ran.append("gates")
    return found


# --- intake: an export becoming threads, then deltas --------------------------
#
# **NAMED `intake` BECAUSE IT IS ONE PIPELINE AND NOT FOUR COMMANDS.** Unpack,
# index, read, relate. Each stage's output is the next one's input and no stage
# can be usefully run alone -- which is what makes it a pipeline in the
# manager's vocabulary rather than a set of tools somebody sequences by hand.
#
# THE FIRST STAGE IS A PERSON'S AND IS NOT HERE. Somebody asks the service for
# an export, waits for the mail, and downloads it. That is a human step by
# construction; `intake` begins at a path on disk.

INTAKE_STAGES = (
    ("unpack", "an export archive becomes one file per conversation"),
    ("index", "each conversation gets an identity and a digest"),
    ("read", "turns are parsed, and a conversation that will not parse is "
             "named rather than skipped"),
    ("relate", "each thread is placed against the projects it is about"),
)


@dataclass
class Intake:
    """What one import did, stage by stage."""

    source: str
    stages: dict[str, str] = field(default_factory=dict)
    written: int = 0
    unchanged: int = 0
    unreadable: list[str] = field(default_factory=list)
    indexed: int | None = None
    """Threads in the archive afterwards. `None` when the harness did not say,
    which is not zero."""

    @property
    def ran(self) -> list[str]:
        return [name for name, _ in INTAKE_STAGES if name in self.stages]

    def summary(self) -> str:
        parts = [f"{self.source}: {self.written} new"]
        if self.unchanged:
            parts.append(f"{self.unchanged} unchanged")
        if self.unreadable:
            parts.append(f"{len(self.unreadable)} unreadable")
        parts.append(f"archive holds "
                     f"{'unknown' if self.indexed is None else self.indexed}")
        return ", ".join(parts)
