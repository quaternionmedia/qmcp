# The roles qmcp plays

qmcp is where the stack writes things down. A tool call, a question put to a
person, a person's answer, an instruction somebody gave and what came of it,
a gate that ran: each is a row in this server's database, so the programs
around it can act and show without each holding that state for itself.

That one record serves several jobs. This page names them, says what each
owns and what it leaves to another program, and says what they share.
[`setup/README.md`](setup/README.md) sets qmcp up for one of them.

## The stack around it

```
            dossier: the panel ── shows the record, answers the queue
               │ HTTP, and JSON a person exports and dossier ingests
joe: speech ── qmcp: the record ── codecartographer: maps of a topology's shape
microphone,     tools, the human queue,      │
transcription,  instructions, gate runs      ├── the local model (Ollama): reads a clone
its own page    │                            ├── Metaflow, in Docker: runs the cookbook's flows
                vox (vendor/vox): the        └── alfred: renders and captions, behind approval
                speech contract and a
                local synthesizer            governance/qm: the roster of projects
```

qmcp imports none of the programs on the left or above it, and they import
nothing from qmcp. What crosses is HTTP, a schema, or a file; vox is the one
package qmcp imports, as a path dependency on its submodule.

## The roles

| Role | What it owns | Its way in | Set up with |
|---|---|---|---|
| **Tool server** | publishing tools, running them when asked, a row per invocation | `GET /v1/tools`, `POST /v1/tools/{name}`, `/v1/invocations`; `qmcp.client`; `qmcp_mcp.py` over the Model Context Protocol | [`setup/tool-server.md`](setup/tool-server.md) |
| **Human gate** | the questions waiting on a person, their options and expiry, and who answered | `/v1/human/requests`, `/v1/human/responses`; `qmcp human`; speech through vox and joe | [`setup/tool-server.md`](setup/tool-server.md) |
| **Harness** | running a repository's gates as recorded invocations, turning failures into units of work, saying which server it is; reading assistant conversations as units of work | `qmcp selfcheck`, `dashboard`, `deltas`, `preflight`, `threads`; `/health`; `/v1/threads` | [`setup/harness.md`](setup/harness.md) |
| **Instruction loop** | an instruction recorded against a rostered project, consent asked through the human gate, the act run in the project's clone, and the project's earlier instructions carried into the next | `qmcp instruct`, `instructions`, `converse`, `serve --converse`; `/v1/instructions`; `qmcp localmodel` | [`voice-loop-demo.md`](voice-loop-demo.md) |
| **Topology designer** | the shapes a set of agents can take, what each would do declared before anything runs, and saved designs and components | `qmcp topology`, `orchestration`; `/v1/topology`, `/v1/orchestration/*`, `/v1/topologies`, `/v1/topology-components` | [`agentframework/topologies.md`](agentframework/topologies.md) |
| **Flow host** | the cookbook's Metaflow flows, run in Docker against this server, alfred's renders and captions among them | `qmcp cookbook`; `examples/flows/`; the `alfred_*` tools | [`setup/flows.md`](setup/flows.md) |
| **Agent library** | SQLModel schemas and mixins for agents, and PydanticAI agents whose tool calls are recorded here | `qmcp.agentframework`, `qmcp.integrations.pydantic_ai` | [`integrations/pydantic-ai.md`](integrations/pydantic-ai.md) |

## What it leaves to others

| Not qmcp's | Whose |
|---|---|
| showing the record, and a window to answer the queue in | dossier |
| the microphone, transcription, and the page a speaker watches | joe |
| what a speech engine must answer, and the voice qmcp speaks with | vox |
| generating text from a clone | the local model Ollama serves, or another runtime |
| running a flow's steps | Metaflow, in Docker |
| rendering and captioning media | alfred |
| drawing a topology's shape as a map | codecartographer |
| which projects exist, and the rules every project is held to | `governance/qm` |
| answering a human request | a person; nothing in qmcp answers one on a person's behalf |

## What the roles share

**One database.** Every role writes to the database `QMCP_DATABASE_URL`
names. A consent the instruction loop asks for is a human request; a gate
`selfcheck` ran is a tool invocation. `--database` on the harness commands
keeps a run apart from the clone's own queue.

**One bind.** The routes for saved designs, the thread archive, the voice
routes and the instruction inbox are registered only when the server binds a
loopback address. A server bound to `0.0.0.0`, which the cookbook's flows need
on a Linux Docker engine, serves the tool server, the human gate and the
topology readings, and none of those.

**One install.** `vox` and `pyttsx3` are default dependencies, so every role
needs the `vendor/vox` submodule checked out before any `uv sync`.

**One port.** 3141 is where joe, dossier and qmcp's own client look for the
server. Running a role on another port means pointing each of them at it;
[`setup/README.md`](setup/README.md) says which setting moves which.

## Where the line is not clean

These are true of the code as it stands, and each is a place a reader setting
qmcp up can be surprised:

- The HTTP API is REST in the shape of MCP's tool calls, not the Model Context
  Protocol on the wire. `qmcp_mcp.py`, in the `mcp` extra, is the only part
  that speaks the protocol.
- The server starts processes of its own: `serve --converse` runs the standing
  conversation beside it, and acting on an instruction runs `qmcp instructions
  act`. Orchestration is not only in the client.
- The cookbook command group holds both the Docker flow recipes and the
  offline checks of the voice and instruction paths.
- `qmcp tools list` imports only the built-in tools; the server also registers
  the `alfred_*` tools, which `GET /v1/tools` lists.
