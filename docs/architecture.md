# Architecture

qmcp is the record the stack writes to. This page says what is qmcp's own,
what reaches it from outside, and the rules every part of it holds. Where the
code and a rule below disagree, the code is the defect, and the last section
names each place it does.

The decision behind this page is qmcp's control-plane record,
`governance/qm/adr/DRAFT-control-plane-and-engine-boundary.md`. [`roles.md`](roles.md)
names the jobs the record serves and how each is set up.

---

## Seam, clients and engines

```
   clients                       the seam: qmcp/                 engines
 ┌─────────────────────┐       ┌───────────────────────────┐   ┌─────────────────────┐
 │ cookbook flows      │ HTTP  │ tool registry, invocations│   │ joe: speech in      │
 │ agent library       │──────▶│ human queue               │──▶│ pyttsx3: speech out │
 │ qmcp_mcp.py (stdio) │       │ instructions and acts     │   │ the local model     │
 │ dossier, joe's page │       │ gate runs, threads        │   │ Metaflow in Docker  │
 └─────────────────────┘       │ topologies, designs       │   │ alfred              │
                               │ consent, spend, loopback  │   └─────────────────────┘
                               │ one database              │
                               └───────────────────────────┘
```

**The seam** is `qmcp/` less the clients below. It owns the tool registry and
a row per invocation; the human queue, with each question's options, expiry
and who answered; the instruction record, the consent an act asks for, and the
history of a project's earlier instructions that each act is handed; the
record of a gate run and the units of work drawn from it; the thread archive;
the topology vocabulary, the orchestration plane and saved designs; which
server this is, by its run-file and the identity on `/health`; and the policy
that decides whether an act may run.

**Clients** reach the seam through its HTTP API or its public Python surface
and nothing else: the cookbook's recipes and flows (`qmcp/cookbook/` less
`delta.py`, `examples/flows/`, `docker/`), the agent library
(`qmcp/agentframework/`, `qmcp/integrations/pydantic_ai/`), the test harness
(`qmcp/testing/`) and the stdio MCP server (`qmcp_mcp.py`). The programs
around qmcp — dossier, joe's page, codecartographer — are clients the same way.

**Engines** do the work the seam decides on, and the seam imports none of
them: joe captures and transcribes speech behind vox's engine contract;
`pyttsx3`, reached through vox, speaks; the local model Ollama serves, or
another runtime, reads a clone; Metaflow runs a flow's steps in Docker; alfred
renders and captions. `vox` is the one package the seam imports, as a path
dependency on `vendor/vox`, and the record counts it as seam code until it is
released.

---

## What the server serves

Every bind serves tool discovery, invocation and history, the human queue, the
topology vocabulary and the orchestration plane. A loopback bind also serves
saved designs, the thread archive, the voice routes and the instruction inbox;
off loopback those routes are not registered at all. [`http-api.md`](http-api.md)
lists every route by role.

The server starts processes of its own, and only on a loopback bind: the
standing conversation `qmcp serve --converse` asks for, a spoken answer to one
human request, a spoken instruction, and `qmcp instructions act`, which asks a
person's consent before it acts. Each is started for a person at this machine.

---

## The flows through it

**A tool call.** A client calls `POST /v1/tools/{name}`; the server runs the
tool and records the invocation, its input, its result or its error. A tool's
failure is returned to the client, which decides whether to retry.

**A human request.** A client asks a question; the server records it with its
options and expiry. A person answers through `POST /v1/human/responses`, by
`qmcp human`, or by speaking through joe; the answer is recorded with who gave
it. A request past its expiry is not listed as pending, and nothing in qmcp
answers one for a person.

**An instruction.** A person speaks or types an instruction; the server records
it against the project it names and runs nothing. Acting on it declares what
it would spend, asks consent on the human queue, and hands a runtime a brief:
the instruction, the clone, and the project's earlier instructions and what
they found, read from this record. The outcome is recorded against the
instruction and spoken as a summary. [`voice-loop-demo.md`](voice-loop-demo.md)
walks it.

**A gate run.** `qmcp selfcheck` runs a repository's gates as recorded
invocations, and `qmcp deltas` turns their failures into units of work.
[`self-checks.md`](self-checks.md) is the reference.

**A restart.** Every row survives it: requests, answers, instructions and
invocations are in the database, so a client resumes by reading them back.

---

## Rules that hold

- **The record is the server's.** Every tool call, question, answer,
  instruction and gate run is a row in the one database `QMCP_DATABASE_URL`
  names.
- **Only a person answers a human request.** No tool, route or command
  answers one on a person's behalf.
- **The server starts a process only for a person at this machine**, from a
  route registered only on a loopback bind, and acts only after that person's
  consent.
- **Nothing spends unattended.** A paid call passes through `qmcp.spend`, and
  a model's output comes through `qmcp.governed`; the rule is the corpus's
  `records/DRAFT-no-unattended-spending.md`.
- **Engines are reached, not imported.** Importing a speech, model, media or
  flow engine into the seam breaks the boundary.
- **Tools keep no state but a client's credentials.** A tool does not call
  another tool, wait for a person, or keep results between calls; it may keep
  the login a service it reaches gave it. The `alfred_*` tools keep alfred's
  token and renew it as it nears expiry ([`integrations/alfred.md`](integrations/alfred.md)).
- **Clients reach the seam through its API.** A client imports no server
  internals, and the seam imports no client.
- **Everything is testable**, offline, with every engine standing in.

---

## Where the code stands against these

- **Only a person answers a human request.** `POST /v1/human/responses`
  authenticates nobody, so any process on the machine that can reach the
  server can answer a human gate through it. No tool exposes that route; the
  HTTP app does not yet tell a person's answer from a program's. qmcp's
  adoption record, row C7, holds the open part.
- **The record is the server's** holds, and one read writes:
  `GET /v1/human/requests/{id}` stores `EXPIRED` on a pending request past
  its expiry, where the list leaves the stored status alone. The adoption
  record's row C4 holds it.
