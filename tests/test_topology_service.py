"""The topology over HTTP, and the line the readings must not cross.

THE TEST WORTH READING IS THE FIRST. The topology *shapes* are this harness's
own vocabulary and name nobody. The *readings* are derived from a person's
conversations. They are two classes of route and one of them is loopback-only,
which is a decision that has to be enforced rather than remembered.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from qmcp import topology_service as ts
from qmcp import topology_view as tv


@pytest.fixture()
def shapes_only() -> TestClient:
    """A server with the shapes registered and no readings -- what an
    off-loopback bind produces."""
    app = FastAPI()
    ts.register(app)
    return TestClient(app)


@pytest.fixture()
def everything(tmp_path) -> TestClient:
    app = FastAPI()
    ts.register(app)
    ts.register_readings(app, tmp_path)
    return TestClient(app)


# --- the boundary --------------------------------------------------------------


def test_readings_are_absent_rather_than_refused_when_only_shapes_are_served():
    """THE ONE THAT MATTERS.

    Off loopback the readings route must not exist. A 403 would be a refusal,
    and a refusal tells a caller the archive is on this machine -- which is
    itself the fact being protected. `qmcp.threads.service` is registered the
    same way and for the same reason.

    Mutation: register the readings unconditionally and this fails.
    """
    app = FastAPI()
    ts.register(app)
    client = TestClient(app)

    assert client.get("/v1/topology").status_code == 200
    assert client.get("/v1/topology/relations/dossier").status_code == 404

    served = {r.path for r in app.routes if hasattr(r, "path")}
    assert not any("relations" in path for path in served), (
        "a readings route is registered on a server that only serves shapes")


def test_the_two_registrations_are_separate_functions():
    """A caller must not be able to serve the readings by accident. One
    function that took a flag would put the decision at the call site of
    whoever forgot it.
    """
    assert callable(ts.register) and callable(ts.register_readings)
    import inspect

    assert "root" not in inspect.signature(ts.register).parameters, (
        "the shapes route takes no archive, so it cannot read one")


def test_the_server_registers_readings_only_on_loopback():
    """The wiring, not just the module. `create_app` is read because the
    decision lives there and this module must not second-guess it.

    **THE STRUCTURE, NOT THE TEXT.** The first version split the file on the
    first occurrence of the name and asserted `is_loopback` came before it --
    which the *import* line always precedes, so the test failed on correct
    code. Text order is not nesting. This walks the tree and asks whether the
    call sits inside an `if is_loopback(...)` body, which is the actual claim.

    Mutation: move the call out of the guard and this fails.
    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(ts.__file__).parent.joinpath("server.py")
                     .read_text(encoding="utf-8"))

    guarded = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.Call)
                and getattr(test.func, "id", "") == "is_loopback"):
            continue
        for inner in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if isinstance(inner, ast.Call) and \
                    getattr(inner.func, "id", "") == "register_topology_readings":
                guarded = True

    assert guarded, "the readings are registered outside the loopback guard"


# --- the shapes ----------------------------------------------------------------


def test_every_topology_is_listed_with_the_encoding(shapes_only):
    body = shapes_only.get("/v1/topology").json()
    assert body["topologies"], "a listing with no topologies asserts nothing"
    assert {c["channel"] for c in body["encoding"]} == {
        "line_weight", "line_style", "line_colour", "node_shape"}


def test_a_named_topology_comes_back_as_a_payload(shapes_only):
    listed = shapes_only.get("/v1/topology").json()["topologies"]
    kind = listed[0]["topology"]

    body = shapes_only.get(f"/v1/topology/shape/{kind}").json()
    assert body["payload"]["topology"] == kind
    assert set(body["payload"]) >= {"topology", "level", "caption", "status",
                                    "marks", "boxes", "arrows"}
    for arrow in body["payload"]["arrows"]:
        assert set(arrow) == {"from", "to", "label", "kind", "weight", "basis"}


def test_an_unknown_topology_is_a_404_naming_where_to_look(shapes_only):
    answer = shapes_only.get("/v1/topology/shape/nonsense")
    assert answer.status_code == 404
    assert "GET /v1/topology" in answer.json()["detail"]


def test_the_level_is_bounded_rather_than_trusted(shapes_only):
    assert shapes_only.get("/v1/topology/shape/delegation?level=99"
                           ).status_code == 422
    assert shapes_only.get("/v1/topology/shape/delegation?level=-1"
                           ).status_code == 422


def test_the_encoding_can_be_fetched_without_a_view(shapes_only):
    """A window checking which mapping it must honour should not have to fetch
    a topology it does not want."""
    body = shapes_only.get("/v1/topology/encoding").json()
    assert body["encoding"] == tv.encoding_payload()


def test_the_shape_route_cannot_be_confused_for_the_readings_route(shapes_only):
    """`/shape/` is in the path so a topology named `relations` could never
    collide with a subject named `relations`.

    Mutation: serve shapes at `/v1/topology/{kind}` and this fails.
    """
    served = {r.path for r in shapes_only.app.routes if hasattr(r, "path")}
    assert "/v1/topology/shape/{kind}" in served


# --- the readings --------------------------------------------------------------


def test_an_absent_archive_is_an_absent_answer_rather_than_no_relations(everything):
    """**NOT AN EMPTY LIST.** A subject with no relations and an archive that
    could not be read are opposite facts, and a 200 with `[]` states the first
    while meaning the second.

    Mutation: return an empty payload instead of 404 and this fails.
    """
    answer = everything.get("/v1/topology/relations/dossier")
    assert answer.status_code == 404
    assert "absent answer" in answer.json()["detail"]


def test_a_subject_that_is_not_a_repository_name_is_refused(everything):
    assert everything.get(
        "/v1/topology/relations/" + "x" * 500).status_code in (400, 404)


def test_min_share_is_bounded(everything):
    assert everything.get(
        "/v1/topology/relations/dossier?min_share=3").status_code == 422


def test_a_survey_counts_threads_and_relations_separately(tmp_path):
    """Zero relations and zero threads are different answers, and a single
    number cannot carry both."""
    relations, surveyed = ts._survey(tmp_path, "dossier")
    assert relations == [] and surveyed == 0


def test_a_survey_authorises_no_spend():
    """These sources read local files. The budget is stated rather than
    defaulted, because an amount is what somebody consents to.

    Mutation: raise the budget and this fails.
    """
    assert ts.SURVEY_BUDGET == 0


# --- the governed seam, served beside the vocabulary ---------------------------


def test_the_governed_seam_is_in_the_gallery(shapes_only):
    """A shape a front end cannot list is a shape nobody will draw.

    The reason `council` is listed rather than dropped, in the other
    direction: the gallery is where somebody choosing a shape looks.
    """
    listed = shapes_only.get("/v1/topology").json()["topologies"]
    names = [entry["topology"] for entry in listed]

    assert "governed" in names
    assert names.count("governed") == 1


def test_the_governed_seam_is_served_off_loopback(shapes_only):
    """It is a shape. It names nobody and holds nothing personal."""
    response = shapes_only.get("/v1/topology/shape/governed")

    assert response.status_code == 200
    assert response.json()["payload"]["topology"] == "governed"


def test_the_governed_payload_declares_that_it_spends(shapes_only):
    """The plane's declarations are part of the picture, over HTTP too."""
    payload = shapes_only.get(
        "/v1/topology/shape/governed?level=2").json()["payload"]

    assert "spends" in payload["marks"]
    assert any(arrow["kind"] == "refusal" for arrow in payload["arrows"])


def test_the_governed_seam_answers_at_every_level(shapes_only):
    for level in (0, 1, 2):
        response = shapes_only.get(f"/v1/topology/shape/governed?level={level}")
        assert response.status_code == 200
        assert response.json()["payload"]["level"] == level


def test_an_unknown_shape_is_still_a_404(shapes_only):
    """Adding a name outside the enum must not make every name resolve."""
    assert shapes_only.get("/v1/topology/shape/govern").status_code == 404
