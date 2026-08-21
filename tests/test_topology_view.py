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


# --- one channel, one axis ----------------------------------------------------


def test_strength_and_measured_never_share_a_channel():
    """THE RULE THE WHOLE ENCODING TURNS ON.

    An unmeasured edge is not a weak edge: one is an absence of evidence, the
    other is evidence of absence. On one scale, "nobody looked" renders as "we
    looked and it is negligible" -- a claim nobody made.

    Mutation: map `measured` onto `line_weight` and this fails.
    """
    from qmcp.topology_view import channels_for

    assert not (set(channels_for("strength")) & set(channels_for("measured")))
    assert channels_for("strength") and channels_for("measured")


def test_every_channel_carries_exactly_one_axis():
    """Two axes in one channel is the collision above, generalised. A window
    reading this cannot know which of the two a given line means."""
    from qmcp.topology_view import ENCODING

    seen = [c.channel for c in ENCODING]
    assert len(seen) == len(set(seen)), f"a channel appears twice: {seen}"


def test_a_categorical_axis_declares_its_values():
    """A continuous channel given a categorical axis invents an ordering; the
    reverse throws a magnitude away. The scale says which this is."""
    from qmcp.topology_view import ENCODING

    for channel in ENCODING:
        if channel.scale == "categorical":
            assert channel.values, f"{channel.channel} names no values"
        else:
            assert not channel.values


def test_the_encoding_travels_as_data_so_both_windows_read_the_same_one():
    """A renderer that picked its own mapping would mean thickness is strength
    in one window and something else in the other."""
    import json

    from qmcp.topology_view import encoding_payload

    payload = json.loads(json.dumps(encoding_payload()))
    axes = {c["axis"] for c in payload}
    assert {"strength", "measured", "relation_kind", "box_kind"} <= axes


def test_an_unmeasured_edge_carries_no_weight_rather_than_zero():
    """`None` and `0.0` are different claims about the same edge.

    Mutation: default `weight` to 0.0 and this fails.
    """
    from qmcp.topology_view import Arrow, from_relations

    assert Arrow("a", "b").weight is None
    view = from_relations("x", [{"target": "o/r/delta/w", "weight": None,
                                 "evidence": [{}]}])
    assert view.arrows[0].weight is None


def test_a_measured_weight_survives_the_payload_as_a_number():
    from qmcp.topology_view import as_payload, from_relations

    view = from_relations("x", [{"target": "o/r/delta/w", "weight": 0.42,
                                 "relation": "part-of",
                                 "evidence": [{"basis": "27 of 63 turns"}]}])
    arrow = as_payload(view)["arrows"][0]
    assert arrow["weight"] == 0.42
    assert "27 of 63" in arrow["basis"], "the weight arrived without its basis"


def test_a_shape_arrow_has_no_weight_because_a_shape_has_no_strength():
    """"Stage to stage" in a pipeline is the shape, not a measurement of it."""
    from qmcp.agentframework.models.enums import TopologyType
    from qmcp.topology_view import FLOWS, view_of

    for arrow in view_of(TopologyType.PIPELINE, FLOWS).arrows:
        assert arrow.weight is None
