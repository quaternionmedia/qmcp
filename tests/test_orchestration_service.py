"""The orchestration plane over HTTP, and the parameter it must not have.

THE TEST WORTH READING IS THE ONE ABOUT `person`. A window can supply workers,
a budget, a model and a build, and can never supply a person's judgement by
query string. `qmcp.orchestration.unmet` holds that line in code; these routes
must not open a door round it.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from qmcp import orchestration as plane
from qmcp import orchestration_service as service


@pytest.fixture()
def served() -> TestClient:
    app = FastAPI()
    service.register(app)
    return TestClient(app)


# --- the plane -----------------------------------------------------------------


def test_the_plane_is_served_as_the_module_declares_it(served):
    """Every capability, every field, read off `PLANE` and nothing added.

    Mutation: drop `why` from `capability_payload` and this fails.
    """
    body = served.get("/v1/orchestration/plane").json()
    assert body["schema"] == 1
    assert [c["topology"] for c in body["capabilities"]] == [
        c.topology.value for c in plane.PLANE]
    for served_row, declared in zip(body["capabilities"], plane.PLANE):
        assert served_row["status"] == declared.status
        assert served_row["spends"] is declared.spends
        assert served_row["decides"] is declared.decides
        assert served_row["why"] == declared.why
        assert [n["key"] for n in served_row["needs"]] == [
            n.key for n in declared.needs]
        for need in served_row["needs"]:
            assert set(need) == {"key", "because", "supplied_by"}


def test_the_vocabularies_ride_along(served):
    """A window building a legend needs the words, not a guess at them."""
    body = served.get("/v1/orchestration/plane").json()
    assert body["needs"] == list(plane.NEEDS)
    assert body["attested"] == list(plane.ATTESTED)
    assert set(body["statuses"]) == {plane.RUNS, plane.BRAINSTORM, plane.REFUSED}


def test_the_drift_reports_are_the_modules_own(served):
    """Three reports the module already computes, served rather than recomputed.

    Mutation: return `[]` for `stubs` and this fails -- the registry holds
    stubs today, and a plane that hid them would be lying over HTTP.
    """
    drift = served.get("/v1/orchestration/plane").json()["drift"]
    assert drift["undeclared"] == plane.undeclared()
    assert drift["stubs"] == plane.stubs()
    assert drift["unregistered_types"] == plane.unregistered_types()
    assert drift["stubs"], "the registry holds stubs, and the route says none"


def test_the_plane_names_the_command_that_prints_the_same_thing(served):
    body = served.get("/v1/orchestration/plane").json()
    assert "uv run qmcp orchestration plane" in body["reading"]["declared_not_discovered"]


# --- runnable ------------------------------------------------------------------


def test_an_empty_hand_runs_nothing_and_says_what_each_shape_wants(served):
    body = served.get("/v1/orchestration/runnable").json()
    assert body["runnable"] == []
    assert body["have"] == {"workers": 0, "budget": 0, "model": False,
                            "built": None}
    for shape in body["shapes"]:
        assert shape["runnable"] is False
        assert shape["unmet"], f"{shape['topology']} is short of nothing yet not runnable"


def test_two_workers_make_the_two_reporting_shapes_runnable(served):
    """The journey `walkthrough/06` walks, over HTTP.

    Mutation: pass `workers` through as a budget and this fails.
    """
    body = served.get("/v1/orchestration/runnable?workers=2").json()
    assert body["runnable"] == [t.value for t in plane.runnable_now(workers=2)]
    assert "delegation" in body["runnable"]
    by_name = {s["topology"]: s for s in body["shapes"]}
    assert by_name["delegation"]["runnable"] is True
    assert by_name["delegation"]["unmet"] == []
    assert [n["key"] for n in by_name["ensemble"]["unmet"]] == ["build", "budget"]


def test_built_overrides_the_declared_status(served):
    body = served.get("/v1/orchestration/runnable?built=true&budget=5").json()
    by_name = {s["topology"]: s for s in body["shapes"]}
    assert by_name["ensemble"]["runnable"] is True
    assert body["have"]["built"] is True


def test_person_is_never_supplied_and_there_is_no_parameter_for_it(served):
    """THE ONE THAT MATTERS.

    Give the route everything a caller can have, plus a `person` parameter
    it does not declare, and council is still short of a person. The route
    has no such parameter, and this checks the absence rather than trusting
    the docstring.

    Mutation: add `person: bool = Query(False)` and pass it to `unmet` (which
    would also need a branch) and this fails.
    """
    import inspect

    body = served.get(
        "/v1/orchestration/runnable?workers=9&budget=99&model=true&built=true"
        "&person=true").json()
    by_name = {s["topology"]: s for s in body["shapes"]}
    assert [n["key"] for n in by_name["council"]["unmet"]] == [plane.PERSON]
    assert "council" not in body["runnable"]
    assert body["never_supplied"]["key"] == plane.PERSON

    assert "person" not in inspect.signature(service.runnable_payload).parameters
    served_params = {
        p.name
        for route in served.app.routes
        if getattr(route, "path", "") == "/v1/orchestration/runnable"
        for p in route.dependant.query_params
    }
    assert "person" not in served_params, served_params


def test_the_hand_is_bounded_rather_than_trusted(served):
    assert served.get("/v1/orchestration/runnable?workers=-1").status_code == 422
    assert served.get("/v1/orchestration/runnable?budget=-5").status_code == 422
