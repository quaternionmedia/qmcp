"""One workflow step, expressed as a delta, and rebuilt from one.

    from qmcp.cookbook.delta import to_delta, from_delta

    delta = to_delta(step, result, project=identity.this_project())
    step  = from_delta(delta, output_type=ChangeSummary)

WHAT THIS IS FOR. dossier plans work as `ProjectDelta` rows -- discrete units
moving through brainstorm, planning, implementation, review, documentation,
complete. qmcp runs work as `AgentStep`s in a pipeline. They are the same thing
seen from two ends: a unit of work with a lifecycle and an audit trail. This is
the correspondence, written once, in both directions, with a test that a step
survives the round trip.

THE SEAM IS A SCHEMA, NOT AN IMPORT. Nothing here imports dossier, and nothing
here should. dossier's `ProjectDelta` lives on an unmerged branch, dossier is
not a qmcp dependency, and coupling the two would mean neither ships without the
other -- which is the opposite of interchangeable. What crosses is a plain dict
whose keys are dossier's column names, so the consumer writes

    ProjectDelta(**delta["delta"], project_id=resolved)

and nothing translates in between. `project_id` is the one required column not
in the row, deliberately: it is an integer primary key in dossier's database,
and qmcp cannot know it. The `project` key beside the row carries `owner/repo`,
which is what the consumer resolves. `SCHEMA` is the promise's version, and a
change to any key is a change to it.

WHAT DOES NOT SURVIVE THE ROUND TRIP, and why it cannot. `output_type` is a
Python class; a delta is data. A row in a database cannot carry a Pydantic model
any more than a plan can carry the code that fulfils it -- so `from_delta`
requires the caller to supply it. That is the modularity rather than a
limitation of it: the delta pins *what the step is and where it stands*, and any
step satisfying that identity can be swapped in behind it. `retries` is dropped
for the same reason in miniature: it is a runtime knob, not a fact about the
work.

WHAT THIS CANNOT SEE. Whether the delta is true. It reads a step's declaration
and, if given one, its result -- not whether the work was any good, whether the
review that ran was a real review, or whether a step that produced output
produced correct output. `phase` here is derived from execution facts and never
from qmcp import identity
from judgement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from qmcp.cookbook.steps import AgentStep, StepResult

SCHEMA = 1

# dossier's `DeltaPhase` values, spelled out rather than imported. If dossier
# renames one, this list stops matching and the test that pins these names is
# where it is noticed -- which is the point of writing them down.
PLANNING = "planning"
IMPLEMENTATION = "implementation"
REVIEW = "review"
COMPLETE = "complete"

# `DeltaLink.link_type` for the audit record a step leaves behind in qmcp. The
# other side of the join is `ToolInvocation.id`, which is why the value goes in
# `target_name` -- it is a UUID string, and `target_id` is an integer column.
INVOCATION_LINK = "invocation"


@dataclass(frozen=True)
class DeltaIdentity:
    """The part of a step a delta pins: what it is, not how it runs."""

    name: str
    system_prompt: str
    mcp_tool: str | None
    mcp_criteria: tuple[str, ...]


def phase_of(step: AgentStep, result: StepResult | None) -> str:
    """Where this step stands, from execution facts alone.

    Four states, and each is a fact somebody can check:

      planning        declared, never run
      implementation  ran, and a review it declared has not happened
      review          ran, and the review it declared did happen
      complete        ran, and declared no review, so nothing is outstanding

    `complete` is the only one that could flatter, and it is reachable only for
    a step that never asked to be reviewed. A step that asked and did not get it
    stays in implementation, because the outstanding thing is outstanding.
    """
    if result is None:
        return PLANNING
    if step.mcp_tool is None:
        return COMPLETE
    return REVIEW if result.mcp_invocation_id else IMPLEMENTATION


def title_of(step: AgentStep) -> str:
    """A human-readable title from the step's identifier.

    `dossier`'s `name` is the short id and `title` is for reading. Deriving one
    from the other beats leaving it empty, and a caller with a better title
    passes it.
    """
    return step.name.replace("_", " ").replace("-", " ").strip().capitalize()


def to_delta(
    step: AgentStep,
    result: StepResult | None = None,
    *,
    project: str | None = None,
    title: str | None = None,
    delta_type: str = "feature",
    priority: str = "medium",
) -> dict[str, Any]:
    """This step as a delta record dossier can ingest without translation.

    `delta` holds `ProjectDelta` columns; `links` holds `DeltaLink` rows. They
    are separate because they are separate tables, and flattening them would
    make the consumer undo the flattening.
    """
    links: list[dict[str, Any]] = []
    if result is not None and result.mcp_invocation_id:
        links.append({
            "link_type": INVOCATION_LINK,
            "target_id": None,
            "target_name": result.mcp_invocation_id,
        })

    return {
        "schema": SCHEMA,
        "project": project,
        "delta": {
            "name": step.name,
            "title": title or title_of(step),
            "description": step.system_prompt,
            "phase": phase_of(step, result),
            "delta_type": delta_type,
            "priority": priority,
        },
        "links": links,
        # Not columns on any dossier table: what qmcp needs to rebuild the step.
        # Kept beside the row rather than inside it, so a consumer writing
        # `ProjectDelta(**delta["delta"])` is never handed a field it has no
        # column for.
        "step": {
            "mcp_tool": step.mcp_tool,
            "mcp_criteria": list(step.mcp_criteria),
        },
    }


def identity_of(step: AgentStep) -> DeltaIdentity:
    """What a delta pins about a step. Two steps equal here are interchangeable."""
    return DeltaIdentity(
        name=step.name,
        system_prompt=step.system_prompt,
        mcp_tool=step.mcp_tool,
        mcp_criteria=tuple(step.mcp_criteria),
    )


def from_delta(
    delta: dict[str, Any],
    output_type: type[BaseModel],
    *,
    retries: int = 3,
) -> AgentStep:
    """Rebuild the step this delta describes.

    `output_type` is required and is not optional-with-a-default on purpose. A
    default would let a caller reconstruct a step that returns the wrong shape
    and never notice, and the whole value of the round trip is that the thing
    coming back is the thing that went in.
    """
    if delta.get("schema") != SCHEMA:
        raise ValueError(
            f"delta schema {delta.get('schema')!r}, this build speaks {SCHEMA}. "
            f"Refusing rather than guessing which keys moved."
        )
    row = delta["delta"]
    extra = delta.get("step") or {}
    return AgentStep(
        name=row["name"],
        system_prompt=row["description"] or "",
        output_type=output_type,
        mcp_tool=extra.get("mcp_tool"),
        mcp_criteria=list(extra.get("mcp_criteria") or []),
        retries=retries,
    )


def invocation_ids(delta: dict[str, Any]) -> list[str]:
    """The qmcp audit records this delta points at.

    The join dossier cannot make on its own: a delta row names the invocation,
    and the invocation lives in qmcp's `ToolInvocation` table.
    """
    return [
        link["target_name"]
        for link in delta.get("links") or []
        if link.get("link_type") == INVOCATION_LINK and link.get("target_name")
    ]
