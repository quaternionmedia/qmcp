"""Saved topology designs over HTTP, and the line between designing and running.

THE TEST WORTH READING IS THE COUNCIL ONE. The plane refuses to run a council
because its arbiter decides. Saving one is allowed, and the response carries the
refusal and says the design is kept: a store that refused the save would hide
the rule at the moment somebody was choosing a shape.

Every test here runs against its own SQLite file. The configured database
holds somebody's human-in-the-loop queue, and `AGENTS.md` records what a test
that writes into it does to another session's work.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from qmcp import identity
from qmcp import orchestration as plane
from qmcp import topology_designs as designs
from qmcp.addresses import parse
from qmcp.agentframework.models.entities.topologies import Topology, config_class_for
from qmcp.agentframework.models.enums import TopologyType

PROJECT = "quaternionmedia/qmcp"


@pytest.fixture()
def served(tmp_path) -> TestClient:
    app = FastAPI()
    designs.register(app, sessions=designs.sessions_at(tmp_path / "designs.db"),
                     project=PROJECT)
    return TestClient(app)


def saved(client: TestClient, name: str = "two-checkers",
          kind: str = "crosscheck", **config) -> dict:
    answer = client.post("/v1/topologies", json={
        "name": name, "description": f"a {kind}", "topology_type": kind,
        "config": config})
    assert answer.status_code == 201, answer.json()
    return answer.json()


# --- saving --------------------------------------------------------------------


def test_a_design_is_saved_validated_and_addressed(served):
    body = saved(served, num_checkers=2)
    assert body["schema"] == 1
    assert body["id"] == 1
    assert body["name"] == "two-checkers"
    assert body["topology_type"] == "crosscheck"
    assert body["address"] == f"{PROJECT}/topology/two-checkers"
    assert parse(body["address"]).kind == "topology"


def test_the_stored_config_is_the_effective_one(served):
    """Defaults filled, so what a window reads back is the shape as the
    harness reads it, and `get_typed_config` on the row round-trips.

    Mutation: store `body.get("config")` raw instead of the validated dump
    and this fails -- the defaults are missing.
    """
    body = saved(served, num_checkers=2)
    effective = config_class_for(TopologyType.CROSS_CHECK).model_validate(
        {"num_checkers": 2}).model_dump(mode="json")
    assert body["config"] == effective
    assert set(body["config"]) > {"num_checkers"}


def test_an_invalid_config_is_refused_with_the_field_named(served):
    """THE RED CASE. 422, and the detail says which field and why, so a
    window can put the message beside the input.

    Mutation: skip `validated_config` on save and this fails.
    """
    answer = served.post("/v1/topologies", json={
        "name": "one-checker", "description": "x", "topology_type": "crosscheck",
        "config": {"num_checkers": 1}})
    assert answer.status_code == 422
    detail = answer.json()["detail"]
    assert detail["where"] == "config"
    assert detail["errors"][0]["loc"] == ["num_checkers"]
    assert "greater than or equal to 2" in detail["errors"][0]["msg"]
    assert served.get("/v1/topologies").json()["count"] == 0


def test_a_required_config_field_that_is_missing_is_named(served):
    answer = served.post("/v1/topologies", json={
        "name": "hubless", "description": "x", "topology_type": "star",
        "config": {}})
    assert answer.status_code == 422
    assert answer.json()["detail"]["errors"][0]["loc"] == ["hub_agent_name"]


def test_an_unknown_kind_is_refused_naming_the_kinds(served):
    answer = served.post("/v1/topologies", json={
        "name": "x", "description": "x", "topology_type": "hexagon"})
    assert answer.status_code == 422
    detail = answer.json()["detail"]
    assert detail["where"] == "topology_type"
    for kind in designs.kinds():
        assert kind in detail["errors"][0]["msg"]


def test_a_bad_name_is_refused_by_the_models_own_validator(served):
    """A table model skips validation on construction. The route must not.

    Mutation: build the row with `Topology(**fields)` instead of
    `Topology.model_validate` and this fails -- the bad name is stored.
    """
    answer = served.post("/v1/topologies", json={
        "name": "Bad Name!", "description": "x", "topology_type": "pipeline"})
    assert answer.status_code == 422
    assert answer.json()["detail"]["where"] == "design"
    assert answer.json()["detail"]["errors"][0]["loc"] == ["name"]


def test_a_name_collision_is_a_409_pointing_at_put(served):
    """The name is the address. Two designs at one address would be one
    design with two histories.

    Mutation: drop the `taken` check and this fails with a 500 from the
    unique index, which is the wrong shape for a caller's mistake.
    """
    saved(served, num_checkers=2)
    again = served.post("/v1/topologies", json={
        "name": "Two-Checkers", "description": "again",
        "topology_type": "crosscheck", "config": {"num_checkers": 3}})
    assert again.status_code == 409
    assert "PUT /v1/topologies/two-checkers" in again.json()["detail"]
    assert served.get("/v1/topologies").json()["count"] == 1


# --- the refused shape ---------------------------------------------------------


def test_a_refused_shape_saves_and_the_response_says_the_run_is_what_is_refused(served):
    """THE ONE THAT MATTERS.

    Designing is not an act; running is. The council saves, the plane's
    refusal comes back beside it, and a sentence says the design is kept.

    Mutation: return 4xx for a `REFUSED` capability and this fails.
    """
    body = saved(served, name="the-council", kind="council")
    capability = body["capability"]
    assert capability["status"] == plane.REFUSED
    assert capability["refusal"] == plane.refuses(TopologyType.COUNCIL, "")
    assert "arbiter" in capability["refusal"]
    assert "designing a shape is not an act" in capability["saved_anyway"]
    assert served.get("/v1/topologies/the-council").status_code == 200


def test_an_allowed_shape_carries_no_refusal_and_no_saved_anyway(served):
    body = saved(served, num_checkers=2)
    assert body["capability"]["refusal"] is None
    assert "saved_anyway" not in body["capability"]
    assert body["capability"]["status"] == plane.RUNS


def test_the_capability_block_is_the_planes_declaration(served):
    """Nothing here invents a verdict. Every field is read off the plane.

    Mutation: hard-code `decides=False` in `capability_block` and this fails
    on debate.
    """
    body = saved(served, name="a-debate", kind="debate")
    declared = plane.by_type()[TopologyType.DEBATE]
    capability = body["capability"]
    assert capability["declared"] is True
    assert capability["decides"] is declared.decides
    assert capability["spends"] is declared.spends
    assert capability["why"] == declared.why
    assert [n["key"] for n in capability["needs"]] == [n.key for n in declared.needs]


def test_a_pairing_can_be_judged_on_read(served):
    """`act` is `refuses`' second argument. A deciding shape is refused an
    attested act and allowed an ordinary one, and only the pairing knows."""
    saved(served, name="a-debate", kind="debate")
    plain = served.get("/v1/topologies/a-debate").json()
    assert plain["capability"]["refusal"] is None
    judged = served.get("/v1/topologies/a-debate",
                        params={"act": plane.ATTESTED[0]}).json()
    assert judged["capability"]["refusal"] is not None
    assert "person" in judged["capability"]["refusal"]


def test_an_undeclared_kind_says_so_through_the_refusal_not_empty_fields(served):
    """`mesh` has a config class and no capability. The block must not read
    as a shape that needs nothing.

    Mutation: default `needs` to `[]` for an undeclared kind and this fails.
    """
    body = saved(served, name="a-mesh", kind="mesh")
    capability = body["capability"]
    assert capability["declared"] is False
    assert capability["status"] is None
    assert capability["needs"] is None
    assert "no declared capability" in capability["refusal"]


# --- reading -------------------------------------------------------------------


def test_the_listing_carries_every_row_addressed_and_judged(served):
    saved(served, num_checkers=2)
    saved(served, name="the-council", kind="council")
    body = served.get("/v1/topologies").json()
    assert body["schema"] == 1
    assert body["count"] == 2
    assert [row["name"] for row in body["topologies"]] == ["two-checkers", "the-council"]
    for row in body["topologies"]:
        assert row["address"].startswith(f"{PROJECT}/topology/")
        assert "capability" in row


def test_an_empty_table_is_a_count_of_zero_not_an_error(served):
    """The table was read and holds nothing, which is an answer. An absent
    table would raise, and that is the different case."""
    body = served.get("/v1/topologies").json()
    assert body["count"] == 0 and body["topologies"] == []


def test_one_design_by_id_and_by_name_is_the_same_row(served):
    saved(served, num_checkers=2)
    by_id = served.get("/v1/topologies/1").json()
    by_name = served.get("/v1/topologies/two-checkers").json()
    by_mixed_case = served.get("/v1/topologies/Two-Checkers").json()
    assert by_id["id"] == by_name["id"] == by_mixed_case["id"] == 1
    assert by_id["schema"] == 1


def test_an_unknown_design_is_a_404_naming_where_to_look(served):
    for ref in ("99", "nothing-here"):
        answer = served.get(f"/v1/topologies/{ref}")
        assert answer.status_code == 404
        assert "GET /v1/topologies" in answer.json()["detail"]


def test_a_digit_name_is_reachable_and_the_id_wins_a_collision(served):
    """A name may be all digits. The rule is id first, then name, and this
    pins the rule rather than leaving it to the query order."""
    saved(served, name="7", num_checkers=2)
    assert served.get("/v1/topologies/7").json()["name"] == "7"
    saved(served, name="1", kind="pipeline")
    assert served.get("/v1/topologies/1").json()["name"] == "7"


# --- changing ------------------------------------------------------------------


def test_put_changes_the_mutable_fields_and_moves_updated_at(served):
    before = saved(served, num_checkers=2)
    answer = served.put("/v1/topologies/two-checkers", json={
        "description": "a trio", "config": {"num_checkers": 3}, "version": "1.1.0"})
    assert answer.status_code == 200, answer.json()
    body = answer.json()
    assert body["changed"] == ["config", "description", "version"]
    assert body["description"] == "a trio"
    assert body["config"]["num_checkers"] == 3
    assert body["version"] == "1.1.0"
    assert body["updated_at"] > before["updated_at"]
    assert body["created_at"] == before["created_at"]
    assert served.get("/v1/topologies/two-checkers").json()["description"] == "a trio"


def test_put_validates_the_new_config_through_the_same_class(served):
    """Mutation: skip `validated_config` on change and this fails."""
    saved(served, num_checkers=2)
    answer = served.put("/v1/topologies/two-checkers",
                        json={"config": {"num_checkers": 0}})
    assert answer.status_code == 422
    assert answer.json()["detail"]["where"] == "config"
    assert served.get("/v1/topologies/two-checkers").json()["config"]["num_checkers"] == 2


def test_put_with_nothing_mutable_is_a_400_not_a_silent_200(served):
    saved(served, num_checkers=2)
    for body in ({}, {"name": "other"}, {"topology_type": "debate"}):
        answer = served.put("/v1/topologies/two-checkers", json=body)
        assert answer.status_code == 400, body
    assert served.get("/v1/topologies/two-checkers").json()["name"] == "two-checkers"


def test_put_on_an_unknown_design_is_a_404(served):
    assert served.put("/v1/topologies/99",
                      json={"description": "x"}).status_code == 404


def test_there_is_no_delete(served):
    """A saved design is a record. Mutation: add a DELETE route and this
    fails."""
    methods = {
        method
        for route in served.app.routes
        if getattr(route, "path", "").startswith("/v1/topologies")
        for method in getattr(route, "methods", ())
    }
    assert "DELETE" not in methods
    assert served.delete("/v1/topologies/1").status_code == 405


# --- the address ---------------------------------------------------------------


def test_an_unknown_identity_gives_no_address_and_says_why(tmp_path):
    """Never guessed. `qmcp.identity.UNKNOWN` is a value a caller can check,
    and this route checks it rather than minting `unknown/unknown/...`.

    Mutation: mint the address unconditionally and this fails.
    """
    app = FastAPI()
    designs.register(app, sessions=designs.sessions_at(tmp_path / "d.db"),
                     project=identity.UNKNOWN)
    client = TestClient(app)
    body = saved(client, num_checkers=2)
    assert body["address"] is None
    assert "QMCP_PROJECT" in body["address_unknown"]


def test_a_saved_row_loads_through_get_typed_config(tmp_path):
    """The store and the model agree about what a config is."""
    import asyncio

    from sqlmodel import select

    sessions = designs.sessions_at(tmp_path / "d.db")
    app = FastAPI()
    designs.register(app, sessions=sessions, project=PROJECT)
    saved(TestClient(app), name="a-star", kind="star", hub_agent_name="hub")

    async def read():
        async with sessions() as session:
            return (await session.execute(select(Topology))).scalars().one()

    row = asyncio.run(read())
    assert type(row.get_typed_config()).__name__ == "StarConfig"
    assert row.get_typed_config().hub_agent_name == "hub"


# --- the wiring ----------------------------------------------------------------


def test_the_server_serves_the_designs_and_the_plane_anywhere(client):
    """Through `create_app` and its lifespan, so the table `init_db` creates
    is the one the route writes. The `client` fixture binds loopback; the
    structural check below is what says these are not behind that guard.
    """
    import ast
    import pathlib

    from qmcp import server

    body = client.post("/v1/topologies", json={
        "name": "wired", "description": "through create_app",
        "topology_type": "delegation"})
    assert body.status_code == 201, body.json()
    assert client.get("/v1/topologies/wired").status_code == 200
    assert client.get("/v1/orchestration/plane").status_code == 200
    assert client.get("/v1/topology/schema/delegation").status_code == 200

    tree = ast.parse(pathlib.Path(server.__file__).read_text(encoding="utf-8"))
    guarded_calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Call) \
                and getattr(node.test.func, "id", "") == "is_loopback":
            for inner in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                if isinstance(inner, ast.Call):
                    guarded_calls.add(getattr(inner.func, "id", ""))
    assert "register_topology_designs" not in guarded_calls
    assert "register_orchestration" not in guarded_calls
