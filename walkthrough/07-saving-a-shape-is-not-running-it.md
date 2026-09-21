# 07 — Saving a shape is not running it

Everything on this page runs. It is executed by the ordinary test command, so
an example that stops being true fails the build rather than sitting here
misleading somebody.

**The problem.** A front end growing a topology designer had the shapes to
draw and nothing else. No schema to build a form from, so a form would offer
whatever fields its author remembered. No way to ask the plane what a shape
would do, so the window would have to carry its own copy of which shapes spend,
which decide and which are refused. Nowhere to keep a design the harness could
see, so nothing could tell the designer what the harness thought of it. Each of
those is a route now, and **the window adds no governance of its own**: every
verdict on this page is the harness's, asked for over HTTP.

## One process, the routes registered beside the shapes

The modules take the app rather than creating one, as `qmcp.topology_service`
does and for the same reason. The designs are kept in a database this page
owns, because the configured one holds somebody's queue:

    >>> import tempfile
    >>> from pathlib import Path
    >>> from fastapi import FastAPI
    >>> from fastapi.testclient import TestClient
    >>> from qmcp import orchestration_service, topology_designs, topology_service

    >>> root = Path(tempfile.mkdtemp())
    >>> app = FastAPI()
    >>> topology_service.register(app)
    >>> orchestration_service.register(app)
    >>> topology_designs.register(
    ...     app, sessions=topology_designs.sessions_at(root / "designs.db"),
    ...     project="quaternionmedia/qmcp")
    >>> client = TestClient(app)

## The plane, asked rather than assumed

`uv run qmcp orchestration plane` prints what every shape would do. The same
declaration is a document now, and a window reads it instead of remembering
it:

    >>> plane = client.get("/v1/orchestration/plane").json()
    >>> by_name = {c["topology"]: c for c in plane["capabilities"]}
    >>> by_name["council"]["status"], by_name["council"]["decides"]
    ('refused', True)
    >>> [need["key"] for need in by_name["council"]["needs"]]
    ['person']

Every need names what supplies it, and the vocabulary a need is drawn from
rides along so a legend is built from the words rather than from a guess:

    >>> plane["needs"]
    ['build', 'budget', 'workers', 'model', 'person']

So do the drift reports the module already computes. A window showing
the plane shows where the plane and the registry disagree, because a
declaration for a shape nothing registers is a picture of something that is
not there:

    >>> sorted(plane["drift"])
    ['stubs', 'undeclared', 'unregistered_types']

## What a hand could run now

The journey `06` walked in Python, over HTTP. Two workers is what a router and
a consensus each need:

    >>> client.get("/v1/orchestration/runnable?workers=2").json()["runnable"]
    ['delegation', 'crosscheck']

And per shape, what it is still short of:

    >>> hand = client.get("/v1/orchestration/runnable?workers=2").json()
    >>> [(s["topology"], [n["key"] for n in s["unmet"]])
    ...  for s in hand["shapes"] if s["topology"] in ("delegation", "ensemble")]
    [('delegation', []), ('ensemble', ['build', 'budget'])]

**There is no `person` parameter.** Pass one anyway, with everything else a
caller can have, and the council is still short of a person. A shape whose
need is somebody's judgement is not made runnable by a query string:

    >>> everything = client.get(
    ...     "/v1/orchestration/runnable"
    ...     "?workers=9&budget=99&model=true&built=true&person=true").json()
    >>> [n["key"] for s in everything["shapes"] if s["topology"] == "council"
    ...  for n in s["unmet"]]
    ['person']
    >>> everything["never_supplied"]["key"]
    'person'

## The form is built from the class

A configuration class is what a saved design is validated through, and its
JSON schema is what a form is built from, so the form cannot offer a field the
class does not hold:

    >>> schema = client.get("/v1/topology/schema/crosscheck").json()
    >>> schema["config_class"]
    'CrossCheckConfig'
    >>> schema["json_schema"]["properties"]["num_checkers"]["minimum"]
    2

`governed` has no schema and the answer says why, rather than serving an empty
one that would read as a shape taking no configuration:

    >>> answer = client.get("/v1/topology/schema/governed")
    >>> answer.status_code, "seam" in answer.json()["detail"]
    (404, True)

## Saving a design

The config goes through the kind's class, and what is stored is what the class
accepted with its defaults filled -- so what the window reads back is the
shape as the harness reads it:

    >>> saved = client.post("/v1/topologies", json={
    ...     "name": "two-checkers", "description": "a pair of independent checkers",
    ...     "topology_type": "crosscheck", "config": {"num_checkers": 2}}).json()
    >>> saved["address"]
    'quaternionmedia/qmcp/topology/two-checkers'
    >>> saved["config"]["consensus_method"]
    'majority'

The plane's verdict comes back beside it. This one runs, and wants more than
one checker:

    >>> saved["capability"]["status"], [n["key"] for n in saved["capability"]["needs"]]
    ('runs', ['workers'])

A config the class refuses is refused here, with the field named so a window
can put the message beside the input:

    >>> refused = client.post("/v1/topologies", json={
    ...     "name": "one-checker", "description": "not a consensus",
    ...     "topology_type": "crosscheck", "config": {"num_checkers": 1}})
    >>> refused.status_code, refused.json()["detail"]["errors"][0]["loc"]
    (422, ['num_checkers'])

And the name is the address, so a second design at the same name is a
collision rather than a second row:

    >>> client.post("/v1/topologies", json={
    ...     "name": "two-checkers", "description": "again",
    ...     "topology_type": "crosscheck"}).status_code
    409

## The one the plane refuses to run saves anyway

`council` is refused by the plane because its arbiter decides. The refusal is
of the *run*. Drawing the shape, describing it and keeping it are none of them
the act the constitution reserves, so the design saves and the response says
what is and is not refused:

    >>> council = client.post("/v1/topologies", json={
    ...     "name": "the-council", "description": "nine perspectives and an arbiter",
    ...     "topology_type": "council"}).json()
    >>> council["capability"]["status"]
    'refused'
    >>> council["capability"]["saved_anyway"]
    'designing a shape is not an act; running it is. The design is kept, and the run is what this harness refuses.'

It is in the listing beside the shape that runs. A store that dropped it would
hide the rule at the moment somebody was choosing a shape:

    >>> [row["name"] for row in client.get("/v1/topologies").json()["topologies"]]
    ['two-checkers', 'the-council']

## A refusal is a property of the pairing

`debate` decides and is not refused as a shape. Point it at an attested act and
it is -- and only the pairing knows which, so a window asks with `act`:

    >>> client.post("/v1/topologies", json={
    ...     "name": "a-debate", "description": "positions argued to a conclusion",
    ...     "topology_type": "debate"}).status_code
    201
    >>> client.get("/v1/topologies/a-debate").json()["capability"]["refusal"] is None
    True
    >>> judged = client.get("/v1/topologies/a-debate",
    ...                     params={"act": "ratify a record"}).json()
    >>> "person" in judged["capability"]["refusal"]
    True

## Changing a design, and what cannot change

Description, config and version. The new config goes through the same class,
and `updated_at` moves:

    >>> changed = client.put("/v1/topologies/two-checkers",
    ...                      json={"config": {"num_checkers": 3}}).json()
    >>> changed["changed"], changed["config"]["num_checkers"]
    (['config'], 3)
    >>> changed["updated_at"] > changed["created_at"]
    True

The name and the kind are fixed -- one is the address and the other is what the
config was validated against -- and a body that changes neither of the mutable
fields is a 400 rather than a 200 that looks like a change:

    >>> client.put("/v1/topologies/two-checkers", json={"name": "other"}).status_code
    400

There is no delete. A saved design is a record:

    >>> client.delete("/v1/topologies/1").status_code
    405

## What this page does not claim

That a saved design runs. Nothing here executes a topology: a design with every
need met is a drawing until a command runs it, and the run is what the plane
judges. Nor that the window honours any of this -- the harness answers what it
is asked, and a window that did not ask would be drawing from memory.
