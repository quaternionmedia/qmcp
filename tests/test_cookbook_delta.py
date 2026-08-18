"""A workflow step and a delta are the same thing, and the round trip proves it.

WHAT IS ACTUALLY UNDER TEST. Not that the mapping runs -- that a step which goes
out as a delta comes back as the same step. A mapping tested one direction at a
time passes while quietly losing a field, and the field it loses is the one
nobody thought to assert.

THE OTHER HALF IS THE COUPLING GUARD. `test_the_seam_does_not_import_dossier` is
the reason this file exists as much as the round trip is: dossier's
`ProjectDelta` is on an unmerged branch, dossier is not a qmcp dependency, and
an import would mean neither project ships without the other. The seam is a
schema.
"""

from __future__ import annotations

from pydantic import BaseModel

from qmcp.cookbook.delta import (
    COMPLETE,
    IMPLEMENTATION,
    INVOCATION_LINK,
    PLANNING,
    REVIEW,
    SCHEMA,
    from_delta,
    identity_of,
    invocation_ids,
    phase_of,
    title_of,
    to_delta,
)
from qmcp.cookbook.steps import AgentStep, StepResult

import pytest


class Summary(BaseModel):
    text: str = ""


class Other(BaseModel):
    value: int = 0


def step(**kw) -> AgentStep:
    base = {
        "name": "summarize_changes",
        "system_prompt": "Summarize the changes in this diff.",
        "output_type": Summary,
    }
    base.update(kw)
    return AgentStep(**base)


def result(invocation: str | None = None) -> StepResult:
    return StepResult(name="summarize_changes", output={"text": "ok"},
                      mcp_invocation_id=invocation)


# --- the round trip ----------------------------------------------------------


def test_a_step_survives_the_round_trip():
    """The whole claim: out as a delta, back as the same step."""
    original = step(mcp_tool="reviewer", mcp_criteria=["clarity", "accuracy"])
    rebuilt = from_delta(to_delta(original), output_type=Summary)
    assert identity_of(rebuilt) == identity_of(original)


def test_a_step_with_no_review_survives_too():
    original = step()
    rebuilt = from_delta(to_delta(original), output_type=Summary)
    assert identity_of(rebuilt) == identity_of(original)


def test_the_round_trip_carries_the_prompt_verbatim():
    """`description` is the prompt. A truncated one would still round-trip
    structurally and would rebuild a step that does something else."""
    prompt = "Summarize.\n\nRules:\n- be exact\n- cite line numbers"
    rebuilt = from_delta(to_delta(step(system_prompt=prompt)), output_type=Summary)
    assert rebuilt.system_prompt == prompt


def test_the_criteria_survive_as_a_list_not_a_string():
    rebuilt = from_delta(
        to_delta(step(mcp_tool="reviewer", mcp_criteria=["a", "b"])),
        output_type=Summary,
    )
    assert rebuilt.mcp_criteria == ["a", "b"]


def test_output_type_is_supplied_by_the_caller_and_not_carried():
    """A delta is data; an output type is code. The seam pins what the step is,
    and the caller brings the class -- which is what makes steps swappable."""
    rebuilt = from_delta(to_delta(step()), output_type=Other)
    assert rebuilt.output_type is Other


def test_from_delta_requires_an_output_type():
    with pytest.raises(TypeError):
        from_delta(to_delta(step()))  # type: ignore[call-arg]


# --- the phase is derived from execution facts -------------------------------


def test_a_step_that_never_ran_is_planning():
    assert phase_of(step(), None) == PLANNING


def test_a_step_that_ran_and_asked_for_no_review_is_complete():
    assert phase_of(step(), result()) == COMPLETE


def test_a_step_whose_declared_review_happened_is_in_review():
    assert phase_of(step(mcp_tool="reviewer"), result("inv-1")) == REVIEW


def test_a_step_whose_declared_review_did_not_happen_stays_in_implementation():
    """The one that must not flatter. Something outstanding is outstanding, and
    reporting it complete would mark unreviewed work as finished."""
    assert phase_of(step(mcp_tool="reviewer"), result(None)) == IMPLEMENTATION


def test_the_phase_reaches_the_delta_row():
    assert to_delta(step(), result())["delta"]["phase"] == COMPLETE


# --- the shape dossier ingests -----------------------------------------------


def test_the_delta_row_holds_only_project_delta_columns():
    """Every key here is a real `ProjectDelta` column, verified against the
    model on dossier's `feature/delta-entity-type` branch. A stray key is a
    validation error in the consumer, which is why the step-only fields live
    beside the row rather than inside it.

    `project_id` is absent on purpose and is the consumer's to supply: it is
    required, has no default, and is an integer primary key qmcp cannot know.
    `project` beside the row carries `owner/repo` for it to resolve.
    """
    columns = {"name", "title", "description", "phase", "delta_type", "priority"}
    assert set(to_delta(step())["delta"]) == columns
    assert "project_id" not in to_delta(step())["delta"]
    assert "project" in to_delta(step(), project="quaternionmedia/qmcp")


def test_the_step_fields_are_kept_out_of_the_row():
    payload = to_delta(step(mcp_tool="reviewer"))
    assert "mcp_tool" not in payload["delta"]
    assert payload["step"]["mcp_tool"] == "reviewer"


def test_an_invocation_becomes_a_link_row_not_a_column():
    payload = to_delta(step(mcp_tool="reviewer"), result("inv-42"))
    assert payload["links"] == [
        {"link_type": INVOCATION_LINK, "target_id": None, "target_name": "inv-42"}
    ]


def test_the_invocation_id_goes_in_target_name_because_it_is_not_an_integer():
    """`DeltaLink.target_id` is an integer column and an invocation id is a
    UUID string. Putting it in the wrong column fails at insert, in dossier,
    where nobody would look for a qmcp defect."""
    payload = to_delta(step(mcp_tool="reviewer"), result("b9532e20-d725-4589"))
    assert payload["links"][0]["target_id"] is None
    assert isinstance(payload["links"][0]["target_name"], str)


def test_a_step_that_was_never_reviewed_produces_no_links():
    assert to_delta(step(), result())["links"] == []


def test_the_audit_join_is_readable_back_off_the_delta():
    payload = to_delta(step(mcp_tool="reviewer"), result("inv-7"))
    assert invocation_ids(payload) == ["inv-7"]


def test_links_of_other_types_are_not_read_as_invocations():
    payload = to_delta(step())
    payload["links"] = [{"link_type": "branch", "target_name": "evolve/x"}]
    assert invocation_ids(payload) == []


def test_the_phase_values_are_dossiers_own_spellings():
    """Written down rather than imported, so a rename in dossier is noticed
    here instead of silently producing rows with an invalid enum value."""
    assert (PLANNING, IMPLEMENTATION, REVIEW, COMPLETE) == (
        "planning", "implementation", "review", "complete")


# --- versioning --------------------------------------------------------------


def test_the_payload_declares_its_schema():
    assert to_delta(step())["schema"] == SCHEMA


def test_a_payload_from_another_schema_is_refused_rather_than_guessed():
    payload = to_delta(step())
    payload["schema"] = SCHEMA + 1
    with pytest.raises(ValueError, match="Refusing"):
        from_delta(payload, output_type=Summary)


def test_the_project_is_carried_when_given_and_null_when_not():
    assert to_delta(step(), project="quaternionmedia/qmcp")["project"] == "quaternionmedia/qmcp"
    assert to_delta(step())["project"] is None


def test_a_title_is_derived_when_none_is_given_and_kept_when_one_is():
    assert title_of(step()) == "Summarize changes"
    assert to_delta(step(), title="Summarise the diff")["delta"]["title"] == "Summarise the diff"


# --- the coupling guard ------------------------------------------------------


def test_the_seam_does_not_import_dossier():
    """Asserted on the source, because the failure is the import existing.

    dossier's `ProjectDelta` is on an unmerged branch and dossier is not a qmcp
    dependency. An import here would mean neither project ships without the
    other, which is the opposite of what this module is for.
    """
    from pathlib import Path

    import qmcp.cookbook.delta as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    )
    assert "dossier" not in code
