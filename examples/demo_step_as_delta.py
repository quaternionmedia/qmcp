#!/usr/bin/env python3
"""Local demo: one workflow step, reconciled with one dossier delta.

    uv run python examples/demo_step_as_delta.py

WHAT IT SHOWS. The `summarizer` step of `CHANGE_IMPACT_PIPELINE` -- a real step
from a real flow -- expressed as a delta row dossier can insert, moved through
its phases by execution facts, rebuilt from the delta, and then *swapped* for a
different implementation carrying the same delta identity.

THE POINT IS THE SWAP. A mapping that only round-trips proves the two shapes
match. Interchangeable means more: any step with the same delta identity can
stand in for the one the delta describes, because what the delta pins is the
work and not the code. The demo builds a second step with a different
`output_type` and shows the identity is unchanged.

WHAT IT DOES NOT DO. Run an agent. Every step here is *declared* and its phase
comes from execution facts the demo supplies, because running one needs a local
LLM this machine does not have. The mapping is what is demonstrated; the agent
is not, and a phase printed below is derived rather than earned.

WHAT IT CANNOT REACH. dossier. Nothing here imports it -- see the module
docstring of `qmcp/cookbook/delta.py`. The demo prints the row and asserts its
shape; inserting it is dossier's side of the seam, and dossier's `ProjectDelta`
is still on an unmerged branch.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from qmcp.cookbook.delta import (
    from_delta,
    identity_of,
    invocation_ids,
    phase_of,
    to_delta,
)
from qmcp.cookbook.steps import StepResult

PROJECT = "quaternionmedia/qmcp"


class TerseSummary(BaseModel):
    """A different output shape, for the swap. The delta cannot tell."""

    headline: str = ""


def the_step():
    """The `summarizer` step, read off the committed pipeline rather than retyped."""
    from qmcp.cookbook.change_impact import CHANGE_IMPACT_PIPELINE

    return CHANGE_IMPACT_PIPELINE.steps[0]


def run(out=print) -> dict[str, Any]:
    step = the_step()
    out(f"step             {step.name}  (from CHANGE_IMPACT_PIPELINE)")
    out(f"output type      {step.output_type.__name__}")
    out("")

    planned = to_delta(step, None, project=PROJECT)
    out(f"as a delta       phase={planned['delta']['phase']}  "
        f"type={planned['delta']['delta_type']}  links={len(planned['links'])}")
    out(f"row dossier gets {json.dumps(planned['delta'], indent=None)}")
    out("")

    # The same step, now with an execution behind it. The step declares no
    # review, so nothing is outstanding and the phase is complete.
    ran = to_delta(step, StepResult(name=step.name, output={"themes": ["auth"]}),
                   project=PROJECT)
    out(f"after running    phase={ran['delta']['phase']}")

    # A reviewed variant: declare a reviewer, and record the invocation it left.
    reviewed_step = type(step)(
        name=step.name, system_prompt=step.system_prompt,
        output_type=step.output_type, mcp_tool="reviewer",
        mcp_criteria=["risk", "completeness"],
    )
    reviewed = to_delta(
        reviewed_step,
        StepResult(name=step.name, output={"themes": ["auth"]},
                   mcp_invocation_id="b9532e20-d725-4589-a055-477d4e947b8d"),
        project=PROJECT,
    )
    out(f"after review     phase={reviewed['delta']['phase']}  "
        f"invocation={invocation_ids(reviewed)}")

    # And the one that must not flatter: a review declared and not performed.
    outstanding = phase_of(reviewed_step, StepResult(name=step.name, output={}))
    out(f"review missing   phase={outstanding}  (not complete: something is outstanding)")
    out("")

    rebuilt = from_delta(reviewed, output_type=step.output_type)
    out(f"rebuilt from it  {rebuilt.name}  tool={rebuilt.mcp_tool}  "
        f"criteria={rebuilt.mcp_criteria}")
    out(f"identity matches {identity_of(rebuilt) == identity_of(reviewed_step)}")

    swapped = from_delta(reviewed, output_type=TerseSummary)
    out(f"swapped output   {swapped.output_type.__name__}  "
        f"identity still matches {identity_of(swapped) == identity_of(reviewed_step)}")

    return {
        "planned": planned,
        "ran": ran,
        "reviewed": reviewed,
        "outstanding_phase": outstanding,
        "rebuilt_matches": identity_of(rebuilt) == identity_of(reviewed_step),
        "swapped_matches": identity_of(swapped) == identity_of(reviewed_step),
        "swapped_output_type": swapped.output_type,
    }


def main() -> int:
    findings = run()
    print()
    print("The delta pins the work, not the code: two steps with different "
          "output types")
    print("carry the same delta identity, so either can stand behind it.")
    return 0 if findings["rebuilt_matches"] and findings["swapped_matches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
