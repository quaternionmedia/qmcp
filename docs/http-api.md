# HTTP routes

Every route the server registers, grouped by the role it serves
([`roles.md`](roles.md)). **Any** routes are served wherever the server is
bound; **loopback** routes are registered only when it binds a loopback
address, because they write, speak, or hold a person's own words, and the
server asks nobody who they are.

## Tool server

| Route | Method | Bind | What it does |
|---|---|---|---|
| `/health` | GET | any | Which server this is: port, start time, pid, session, a fingerprint of its database |
| `/v1/tools` | GET | any | Every registered tool |
| `/v1/tools/{name}` | POST | any | Invoke a tool, recording the invocation |
| `/v1/invocations` | GET | any | Invocation history |
| `/v1/invocations/{id}` | GET | any | One invocation |
| `/metrics` | GET | any | Prometheus metrics |
| `/metrics/json` | GET | any | The same metrics as JSON |

[`tools.md`](tools.md) is the reference.

## Human gate

| Route | Method | Bind | What it does |
|---|---|---|---|
| `/v1/human/requests` | POST | any | Put a question to a person |
| `/v1/human/requests` | GET | any | List questions, without expiring any |
| `/v1/human/requests/{id}` | GET | any | One question with its answer. A pending question past its expiry is marked expired, and the transition is stored |
| `/v1/human/responses` | POST | any | Answer a question |
| `/v1/human/requests/{id}/voice` | POST | loopback | Ask a question aloud and take the spoken answer |
| `/v1/human/voice` | GET | loopback | Whether a spoken run is going, and its last lines |

[`human_in_loop.md`](human_in_loop.md) and
[`integrations/voice.md`](integrations/voice.md) are the references.

## Instruction loop

| Route | Method | Bind | What it does |
|---|---|---|---|
| `/v1/instructions` | POST | loopback | Record an instruction |
| `/v1/instructions` | GET | loopback | The inbox |
| `/v1/instructions/{id}` | GET | loopback | One instruction and what came of it |
| `/v1/instructions/{id}/act` | POST | loopback | Ask consent, then act |
| `/v1/instructions/voice` | POST | loopback | Take an instruction by speech |
| `/v1/instructions/voice` | GET | loopback | The spoken instruction's progress |
| `/v1/voice/vocabulary` | GET | loopback | What can be said, and what each phrase does |

## Harness and thread archive

| Route | Method | Bind | What it does |
|---|---|---|---|
| `/v1/threads` | GET | loopback | Archived conversations |
| `/v1/threads/diverged` | GET | loopback | Conversations whose sources disagree |
| `/v1/threads/reindex` | POST | loopback | Rebuild the archive's index |
| `/v1/threads/import` | POST | loopback | Import a conversation export |
| `/v1/threads/{source}/{identifier}` | GET | loopback | One conversation |
| `/v1/threads/{source}/{identifier}/deltas` | GET | loopback | One conversation as units of work |

[`threads.md`](threads.md) is the reference. The gates themselves have no
route: `qmcp selfcheck` and `qmcp dashboard` read and write the database.

## Topology designer

| Route | Method | Bind | What it does |
|---|---|---|---|
| `/v1/topology` | GET | any | Every topology shape, and the encoding a window must honour |
| `/v1/topology/encoding` | GET | any | Which visual channel carries which data axis |
| `/v1/topology/shape/{kind}` | GET | any | One shape as boxes and arrows, at a level |
| `/v1/topology/schema/{kind}` | GET | any | The JSON schema of one kind's configuration class |
| `/v1/topology/relations/{subject}` | GET | loopback | Relations read from the thread archive |
| `/v1/orchestration/plane` | GET | any | What every shape would do, declared: status, spends, writes, decides, needs, refusals, drift |
| `/v1/orchestration/runnable` | GET | any | What a hand (`workers`, `budget`, `model`, `built`) could run, and what each shape is short of |
| `/v1/topologies` | POST | loopback | Save a design, validated through its kind's configuration class |
| `/v1/topologies` | GET | loopback | Every saved design, with the plane's verdict |
| `/v1/topologies/{ref}` | GET | loopback | One design by id or name; `?act=` judges a pairing |
| `/v1/topologies/{ref}` | PUT | loopback | Change a design's description, config or version |
| `/v1/topology-components` | POST | loopback | Save a reusable, named instruction component |
| `/v1/topology-components` | GET | loopback | Every saved component of one project |
| `/v1/topology-components/{name}` | GET | loopback | One component |
| `/v1/topology-components/{name}` | PUT | loopback | Change a component |

`walkthrough/07-saving-a-shape-is-not-running-it.md` exercises the design
routes, and [`agentframework/topologies.md`](agentframework/topologies.md) is
the reference.
