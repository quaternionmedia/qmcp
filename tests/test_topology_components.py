"""Reusable topology components and reference integrity."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from qmcp import topology_designs


@pytest.fixture()
def served(tmp_path) -> TestClient:
    app = FastAPI()
    topology_designs.register(
        app,
        sessions=topology_designs.sessions_at(tmp_path / "components.db"),
        project="quaternionmedia/qmcp",
    )
    return TestClient(app)


def component(client: TestClient, name: str) -> dict:
    response = client.post("/v1/topology-components", json={
        "name": name,
        "description": f"{name} role",
        "instruction": f"Review the prompt as {name}.",
    })
    assert response.status_code == 201, response.text
    return response.json()


def design(client: TestClient, name: str, config: dict | None = None) -> dict:
    response = client.post("/v1/topologies", json={
        "name": name,
        "description": f"{name} design",
        "topology_type": "pipeline",
        "config": config or {},
    })
    assert response.status_code == 201, response.text
    return response.json()


def test_components_are_named_reusable_and_editable(served):
    saved = component(served, "reviewer")
    assert saved["name"] == "reviewer"
    assert served.get("/v1/topology-components").json()["count"] == 1

    updated = served.put(
        "/v1/topology-components/reviewer",
        json={"instruction": "Check the claim's implementation and tests."},
    )
    assert updated.status_code == 200
    assert updated.json()["instruction"] == "Check the claim's implementation and tests."


def test_names_are_unique_per_project_and_composition_crosses_projects(served):
    def save(path, project, name, **extra):
        body = {"project": project, "name": name, "description": f"{name} d", **extra}
        return served.post(path, json=body)

    for project in ("alpha", "beta"):
        assert save("/v1/topology-components", project, "judge",
                    instruction="Judge it.").status_code == 201
        assert save("/v1/topologies", project, "flow", topology_type="pipeline",
                    config={"components": [{"name": "judge"}]}).status_code == 201
    assert save("/v1/topologies", "alpha", "flow",
                topology_type="pipeline").status_code == 409

    assert served.get("/v1/topologies", params={"project": "alpha"}).json()["count"] == 1
    assert served.get("/v1/topologies/flow", params={"project": "beta"}
                      ).json()["project"] == "beta"
    assert served.get("/v1/topology-components/judge", params={"project": "gamma"}
                      ).status_code == 404

    # a component must exist in the design's own project
    assert save("/v1/topologies", "gamma", "flow", topology_type="pipeline",
                config={"components": [{"name": "judge"}]}).status_code == 422

    assert save("/v1/topologies", "alpha", "wrap", topology_type="pipeline",
                config={"compose": ["beta/flow"]}).status_code == 201
    assert save("/v1/topologies", "alpha", "bad", topology_type="pipeline",
                config={"compose": ["beta/nope"]}).status_code == 422

    # a cycle is found across the project boundary
    cycle = served.put(
        "/v1/topologies/flow", params={"project": "beta"},
        json={"config": {"compose": ["alpha/wrap"]}},
    )
    assert cycle.status_code == 422
    assert "cycle" in cycle.json()["detail"]


def test_design_references_require_existing_reusable_components(served):
    response = served.post("/v1/topologies", json={
        "name": "review-flow",
        "description": "A review flow",
        "topology_type": "pipeline",
        "config": {"components": [{"name": "missing"}]},
    })
    assert response.status_code == 422
    assert "missing" in response.json()["detail"]


def test_multiple_designs_can_reference_the_same_component(served):
    component(served, "reviewer")
    first = design(served, "first", {"components": [{"name": "reviewer"}]})
    second = design(served, "second", {"components": [{"name": "reviewer"}]})
    assert first["config"]["components"] == second["config"]["components"]


def test_composition_requires_saved_designs_and_rejects_cycles(served):
    design(served, "base")
    outer = served.post("/v1/topologies", json={
        "name": "outer",
        "description": "Nested flow",
        "topology_type": "compound",
        "config": {"compose": ["base"]},
    })
    assert outer.status_code == 201, outer.text

    cycle = served.put("/v1/topologies/base", json={"config": {"compose": ["outer"]}})
    assert cycle.status_code == 422
    assert "cycle" in cycle.json()["detail"]

    missing = served.post("/v1/topologies", json={
        "name": "missing-child",
        "description": "Missing child",
        "topology_type": "compound",
        "config": {"compose": ["absent"]},
    })
    assert missing.status_code == 422
    assert "absent" in missing.json()["detail"]


def test_a_design_cannot_compose_itself(served):
    """Named as itself rather than reported as a cycle, so the message says
    what was done.

    Mutation: drop the `home in keys.values()` check and the cycle walk
    answers instead, with the wrong sentence.
    """
    design(served, "loop")
    answer = served.put("/v1/topologies/loop",
                        json={"config": {"compose": ["loop"]}})
    assert answer.status_code == 422
    assert answer.json()["detail"] == "a topology cannot compose itself"


def test_composition_deeper_than_eight_levels_is_refused(served):
    """Eight nested levels save; a ninth is a 422.

    Mutation: raise the depth limit and this fails.
    """
    design(served, "level-0")
    for level in range(1, 9):
        design(served, f"level-{level}", {"compose": [f"level-{level - 1}"]})
    answer = served.post("/v1/topologies", json={
        "name": "too-deep", "description": "a ninth level",
        "topology_type": "pipeline", "config": {"compose": ["level-8"]}})
    assert answer.status_code == 422
    assert "eight levels" in answer.json()["detail"]


def test_a_component_named_twice_in_one_design_is_refused(served):
    """Case-insensitively, through the config class, before any lookup.

    Mutation: drop the uniqueness check in `ComposableTopologyConfig` and this
    fails.
    """
    component(served, "reviewer")
    answer = served.post("/v1/topologies", json={
        "name": "twice", "description": "names one component twice",
        "topology_type": "pipeline",
        "config": {"components": [{"name": "reviewer"}, {"name": "Reviewer"}]}})
    assert answer.status_code == 422
    assert answer.json()["detail"]["where"] == "config"


def test_a_component_name_collision_is_a_409(served):
    """Mutation: drop the `existing` check and the unique constraint answers
    with a 500, the wrong shape for a caller's mistake."""
    component(served, "reviewer")
    again = served.post("/v1/topology-components", json={
        "name": "reviewer", "description": "again", "instruction": "Again."})
    assert again.status_code == 409
    assert served.get("/v1/topology-components").json()["count"] == 1


def test_the_default_project_is_the_registered_identitys_short_name(tmp_path):
    """A row's project is the short name a spoken command would carry, taken
    from the identity the routes were registered with -- not from whichever
    checkout the process happens to sit in.

    Mutation: call `scope_of` without the registered `project` and this fails
    in any checkout whose remote is not `someone/elsewhere`.
    """
    app = FastAPI()
    topology_designs.register(
        app, sessions=topology_designs.sessions_at(tmp_path / "elsewhere.db"),
        project="someone/elsewhere")
    client = TestClient(app)
    assert component(client, "reviewer")["project"] == "elsewhere"
    assert design(client, "flow")["project"] == "elsewhere"
    assert client.get("/v1/topologies").json()["project"] == "elsewhere"
