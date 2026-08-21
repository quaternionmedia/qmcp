"""One description of a topology, for windows that draw it differently.

THE TEST WORTH READING IS THE FIRST ONE. The description must hold nothing that
belongs to a terminal or to a browser, or the other window has to undo it.
"""

from __future__ import annotations

import json

import pytest

from qmcp.agentframework.models.enums import TopologyType
from qmcp.orchestration import REFUSED
from qmcp.topology_view import (
    BLACK_BOX,
    FLOWS,
    LEVELS,
    PARTS,
    REFUSAL,
    as_payload,
    gallery,
    view_of,
)


def test_the_description_holds_nothing_a_renderer_owns():
    """THE ONE THAT MATTERS.

    No coordinates, no glyphs, no colours, no widths. The moment one appears
    the description is a picture of one window and the other has to undo it.

    Mutation: add an `x`/`y` or a colour to `Box` and this fails.
    """
    payload = as_payload(view_of(TopologyType.DELEGATION, FLOWS))
    text = json.dumps(payload).lower()
    for owned in ("colour", "color", '"x"', '"y"', "width", "pixel", "glyph",
                  "font", "#"):
        assert owned not in text, f"the description carries {owned!r}"


def test_every_level_describes_the_same_topology():
    """One view at three distances, not three views."""
    for level in LEVELS:
        assert view_of(TopologyType.DELEGATION, level).topology == "delegation"


def test_a_black_box_hides_the_parts_and_keeps_the_edges():
    """Level 0 answers "what goes in and what comes out" and nothing else."""
    view = view_of(TopologyType.PIPELINE, BLACK_BOX)
    kinds = {b.kind for b in view.boxes}
    assert "gate" not in kinds
    assert len(view.boxes) < len(view_of(TopologyType.PIPELINE, FLOWS).boxes)
    assert view.box("box").note.endswith("parts inside")


def test_the_black_box_is_folded_from_the_flow_rather_than_written_twice():
    """A summary written separately can disagree with what it summarises.

    Mutation: hard-code a level-0 shape and this stops being true the moment
    the level-2 shape changes.
    """
    flow = view_of(TopologyType.PIPELINE, FLOWS)
    boxed = view_of(TopologyType.PIPELINE, BLACK_BOX)
    inside = len(flow.boxes) - len([b for b in flow.boxes
                                    if b.kind in ("input", "output")])
    assert f"{inside} parts inside" == boxed.box("box").note


def test_the_parts_level_says_nothing_about_order():
    """Level 1 is what it is made of. Order is level 2's answer."""
    assert view_of(TopologyType.CROSS_CHECK, PARTS).arrows == ()


def test_an_unknown_level_is_refused_rather_than_guessed():
    with pytest.raises(ValueError):
        view_of(TopologyType.PIPELINE, 7)


# --- what the plane says travels with the shape -------------------------------


def test_a_refused_topology_is_in_the_gallery_and_marked():
    """A gallery is where somebody chooses, and a chooser needs to see the one
    they must not use.

    Mutation: filter refused topologies out and this fails.
    """
    found = {v.topology: v for v in gallery(BLACK_BOX)}
    assert "council" in found
    assert found["council"].is_refused
    assert found["council"].status == REFUSED


def test_the_refusal_is_on_the_arrow_that_carries_it():
    """`council`'s arbiter deciding is the refused act, so the arrow from the
    arbiter is the refused one -- not the whole shape greyed."""
    view = view_of(TopologyType.COUNCIL, FLOWS)
    refused = [a for a in view.arrows if a.kind == REFUSAL]
    assert len(refused) == 1
    assert refused[0].frm == "arbiter"


def test_spending_and_deciding_travel_with_the_view():
    view = view_of(TopologyType.DEBATE, BLACK_BOX)
    assert "spends" in view.marks and "decides" in view.marks


def test_a_running_topology_carries_no_alarming_marks():
    view = view_of(TopologyType.DELEGATION, BLACK_BOX)
    assert view.marks == ()
    assert not view.is_refused


# --- the gallery --------------------------------------------------------------


def test_the_gallery_holds_every_described_shape():
    names = {v.topology for v in gallery()}
    for expected in ("pipeline", "delegation", "crosscheck", "ensemble",
                     "debate", "chain", "compound", "council"):
        assert expected in names


def test_a_name_with_no_shape_is_drawn_as_an_empty_box_not_omitted():
    """`mesh`, `star` and `ring` are in the vocabulary and nothing describes
    them. A gallery that dropped them would hide that the names exist.

    Mutation: return None for an undescribed topology and this fails.
    """
    view = view_of(TopologyType.MESH, FLOWS)
    assert view.topology == "mesh"
    assert "no shape described" in view.caption


# --- the payload --------------------------------------------------------------


def test_the_payload_is_json_and_flat():
    """Both windows read this over a seam; neither imports this module."""
    payload = as_payload(view_of(TopologyType.COUNCIL, FLOWS))
    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped["topology"] == "council"
    assert round_tripped["status"] == REFUSED
    assert any(a["kind"] == REFUSAL for a in round_tripped["arrows"])


def test_a_count_is_carried_when_it_is_known_and_absent_when_it_is_not():
    """Nine council members is a fact; the number of pipeline stages is not."""
    council = as_payload(view_of(TopologyType.COUNCIL, FLOWS))
    members = next(b for b in council["boxes"] if b["id"] == "members")
    assert members["count"] == 9

    pipeline = as_payload(view_of(TopologyType.PIPELINE, FLOWS))
    assert all(b["count"] is None for b in pipeline["boxes"])
