"""The loop that improves the loop: diagnose, record, and ratchet.

    uv run qmcp feedback run

That command is claimed and not routed: `qmcp.cli` has no `feedback` group
(`tests/test_declared_commands.py`, `ONLY_CLAIMED`). Call `once` directly.

`FeedbackTopology` is a pipeline topology that implements `run`, for this one
shape; `BaseTopology.run` raises `NotImplementedError`.

The two stages look different ways. `gates` runs the harness's suite and
answers "does this pass". `diagnostics` runs `dossier.diagnostics` in the
panel's environment and answers "is a passing gate enforcing anything".

`unratcheted` lists failures that no check covers: the queue of checks to
write, as distinct from recurrences a check already catches.

It does not spend, schedule itself or close anything. The stages are
subprocesses this project already runs, nothing writes a cron entry
(`governance/qm/records/DRAFT-no-unattended-spending.md`), and a failure
becomes a delta at `brainstorm` and stops there.

`docs/self-checks.md` describes the loop and `intake`.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qmcp.agentframework.models.enums import TopologyType
from qmcp.agentframework.topologies import BaseTopology, topology

# The two stages, and which question each answers. Fixed here rather than
# configured, so the loop cannot be reduced to the stage that always passes.
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

        `brainstorm` and no further, as in `qmcp.selfcheck`: running a check
        again cannot establish that anybody acted on it.
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

    A failure a check already covers is a recurrence. A failure nothing
    covers is a hole, closed by writing the check as well as the fix.

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

    Run as a subprocess in the panel's own environment rather than imported:
    the harness does not depend on the panel, and the panel's virtualenv holds
    its dependencies.
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
    """The harness's gates, read as findings.

    `--quiet` keeps uv's own lines -- "Installed 1 package" after a sync -- out
    of the output, where the last of them would stand as the suite's summary.
    """
    code, out = _read(["uv", "run", "--quiet", "pytest", "-q", "--no-header"], harness)
    if code == 0:
        return [Finding("gates", "suite", "pass",
                        out.strip().splitlines()[-1] if out.strip() else "")]
    failed = [line for line in out.splitlines() if line.startswith("FAILED")]
    if not failed:
        return [Finding("gates", "suite", "fail", f"exit {code}")]
    return [Finding("gates", line.split("::")[-1][:60], "fail", line[:120])
            for line in failed]


class FeedbackTopology(BaseTopology):
    """The improvement loop, as a pipeline the manager knows about.

    Not registered. `TopologyRegistry` is keyed by `TopologyType` and
    `register` replaces silently, so a second class claiming `PIPELINE` would
    win or lose by import order. This is a concrete pipeline that runs, not
    the generic one; `qmcp.orchestration` says the same in the plane.
    """

    topology_type = TopologyType.PIPELINE

    def __init__(self, *args, **kwargs) -> None:  # noqa: D107
        # Constructed directly with no arguments, or with a topology's
        # arguments; the first is the one in use.
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

    The stages are independent, so the diagnostics run when the suite is red.
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
# One pipeline of four stages: unpack, index, read, relate. Each stage's output
# is the next one's input.
#
# The step before it is a person's: requesting the export from the service and
# downloading it. `intake` begins at a path on disk.

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
