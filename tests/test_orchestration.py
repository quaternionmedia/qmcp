"""What each topology would do, and which pairings this organisation refuses.

THE TEST WORTH READING IS THE REFUSAL ONE. A machine reaching a verdict on
whether to ratify is not a machine doing a person's job badly -- it is a verdict
that asserts something nobody asserted, and it is indistinguishable afterwards
from one somebody made.
"""

from __future__ import annotations

import pytest

from qmcp.agentframework.models.enums import TopologyType
from qmcp.agentframework.topologies import TopologyRegistry
from qmcp.orchestration import (
    ATTESTED,
    BRAINSTORM,
    PLANE,
    REFUSED,
    RUNS,
    by_type,
    cross_check,
    delegate,
    refuses,
    render,
    undeclared,
    unregistered_types,
)


# --- the refusal --------------------------------------------------------------


def test_a_deciding_topology_is_refused_an_attested_act():
    """THE ONE THAT MATTERS.

    `debate` is a perfectly good shape and it ends in something choosing.
    Pointed at ratification it would produce a verdict nobody made.

    Mutation: drop the `decides` condition and this fails.
    """
    why = refuses(TopologyType.DEBATE, "ratify a record")
    assert why is not None
    assert "person" in why


def test_the_same_topology_is_allowed_an_ordinary_question():
    """The refusal is a property of the pairing, not of the shape. A deciding
    topology pointed at something nobody's constitution reserves is fine.

    Mutation: refuse deciding topologies outright and this fails.
    """
    assert refuses(TopologyType.DEBATE, "which colour for the chart") is None


def test_a_reporting_topology_is_allowed_an_attested_act():
    """Reporting on ratification is not ratifying. A cross-check that told
    somebody what three checkers thought has performed no attested act."""
    assert refuses(TopologyType.CROSS_CHECK, "ratify a record") is None


def test_a_refused_topology_is_refused_whatever_it_is_pointed_at():
    """Council's arbiter makes the final decision by construction, so the
    shape carries the problem rather than the pairing."""
    assert refuses(TopologyType.COUNCIL, "anything at all") is not None
    assert refuses(TopologyType.COUNCIL, "ratify a record") is not None


def test_an_undeclared_topology_is_refused_rather_than_allowed():
    """Default closed. A topology nobody declared is one nothing knows the
    cost or the authority of, and letting it through because no rule matched
    is how an orchestration plane becomes decorative.

    Mutation: return None for unknown topologies and this fails.
    """
    why = refuses(TopologyType.MESH, "anything")
    assert why is not None and "no declared capability" in why


def test_every_attested_act_is_a_real_sentence():
    """Restated from the corpus, so at least check they are not empty."""
    assert len(ATTESTED) >= 5
    for act in ATTESTED:
        assert act and act[0].islower(), act


# --- the declarations themselves ----------------------------------------------


def test_every_registered_topology_declares_what_it_would_do():
    """A topology that ran with its cost and its authority unstated is the
    whole failure this module exists against.

    Mutation: register a topology without a capability and this fails.
    """
    assert undeclared() == [], undeclared()


def test_the_declarations_cover_the_registry_in_both_directions():
    assert set(by_type()) == set(TopologyRegistry._topologies)


def test_a_brainstorm_is_named_as_one_rather_than_left_silent():
    """A stub that raises reads as debt. Saying `brainstorm` says it is a
    proposal somebody wrote before anybody needed it, which is what it is."""
    kinds = {c.status for c in PLANE}
    assert BRAINSTORM in kinds
    assert RUNS in kinds
    for entry in PLANE:
        assert entry.status in (RUNS, BRAINSTORM, REFUSED)


def test_every_declaration_says_why():
    """A status without a reason is a verdict."""
    for entry in PLANE:
        assert len(entry.why) > 40, entry.topology


def test_only_the_shapes_that_are_implemented_claim_to_run():
    """`can_run` is what a caller checks. It must not be true for a shape
    whose `run` still raises."""
    import asyncio

    for entry in PLANE:
        if not entry.can_run:
            continue
        cls = TopologyRegistry.get(entry.topology)
        assert cls is not None
        # The three that run are the three implemented in this module or in
        # `qmcp.feedback`; a stub's `run` raises NotImplementedError.
        assert cls.run is not __import__(
            "qmcp.agentframework.topologies", fromlist=["x"]).BaseTopology.run, (
            f"{entry.topology.value} claims to run but inherits the stub")


def test_names_in_the_vocabulary_with_no_implementation_are_reported():
    """`mesh`, `star` and `ring` are in the enum with no class. Reported
    rather than removed: a smaller brainstorm is still somebody's intent."""
    absent = unregistered_types()
    assert "mesh" in absent
    assert render().count("implemented by nothing") == 1


# --- delegation ---------------------------------------------------------------


def test_work_goes_to_the_worker_for_its_shape():
    items = [{"shape": "a"}, {"shape": "b"}, {"shape": "a"}]
    routed = delegate(items, {"a": lambda i: "did a", "b": lambda i: "did b"})
    assert [r.result for r in routed] == ["did a", "did b", "did a"]
    assert all(r.taken for r in routed)


def test_a_shape_with_no_worker_is_named_rather_than_dropped():
    """Dropping it leaves a run looking finished with its work untouched.

    Mutation: skip unrouted items and this fails.
    """
    routed = delegate([{"shape": "a"}, {"shape": "?"}], {"a": lambda i: 1})
    assert len(routed) == 2
    unrouted = [r for r in routed if not r.taken]
    assert len(unrouted) == 1 and "?" in unrouted[0].worker


def test_a_worker_that_raises_loses_one_item_and_not_the_rest():
    def explodes(item):
        raise RuntimeError("nope")

    routed = delegate([{"shape": "x"}, {"shape": "ok"}],
                      {"x": explodes, "ok": lambda i: "fine"})
    assert "RuntimeError" in str(routed[0].result)
    assert routed[1].result == "fine"


def test_the_shape_can_be_read_however_the_caller_keeps_it():
    routed = delegate(["alpha", "beta"], {"a": lambda i: i.upper()},
                      shape_of=lambda item: item[0])
    assert routed[0].result == "ALPHA"
    assert routed[1].taken is False


# --- cross-check --------------------------------------------------------------


def test_independent_checkers_are_counted_and_not_resolved():
    """It reports. What to do about the count is somebody else's.

    Mutation: return a single boolean verdict and this fails -- the split
    below stops being visible.
    """
    found = cross_check("the claim", [lambda c: True, lambda c: False,
                                      lambda c: True])
    assert found.agreed == 2
    assert found.majority is True
    assert found.unanimous is False
    assert found.is_split is True


def test_unanimous_is_not_the_same_as_a_majority():
    found = cross_check("x", [lambda c: True, lambda c: True])
    assert found.unanimous and found.majority and not found.is_split


def test_a_checker_that_raises_counts_against_rather_than_vanishing():
    """A check that could not be made is not agreement. Dropping it would let
    one broken checker raise everybody else's share of the vote.

    Mutation: skip raising checkers and this fails -- two of two becomes
    unanimous where it should be one of two.
    """
    def explodes(claim):
        raise ValueError("could not tell")

    found = cross_check("x", [lambda c: True, explodes])
    assert found.agreed == 1
    assert len(found.verdicts) == 2
    assert found.unanimous is False
    assert "ValueError" in found.reasons[1]


def test_a_checker_may_give_a_reason_with_its_verdict():
    found = cross_check("x", [lambda c: (False, "the port disagrees")])
    assert found.verdicts == (False,)
    assert found.reasons[0] == "the port disagrees"


def test_no_checkers_is_not_agreement():
    """Nobody looked. Reporting that as unanimous would be the worst available
    answer."""
    found = cross_check("x", [])
    assert found.unanimous is False
    assert found.majority is False
    assert found.is_split is False


# --- the runtimes actually run ------------------------------------------------


@pytest.mark.asyncio
async def test_the_delegation_topology_runs_through_the_manager():
    cls = TopologyRegistry.get(TopologyType.DELEGATION)
    out = await cls().run({"items": [{"shape": "a"}, {"shape": "?"}],
                           "workers": {"a": lambda i: 1}})
    assert out["routed"] == 1
    assert len(out["unrouted"]) == 1


@pytest.mark.asyncio
async def test_the_crosscheck_topology_runs_through_the_manager():
    cls = TopologyRegistry.get(TopologyType.CROSS_CHECK)
    out = await cls().run({"claim": "x",
                           "checkers": [lambda c: True, lambda c: False]})
    assert out["agreed"] == 1 and out["of"] == 2
    assert out["split"] is True
