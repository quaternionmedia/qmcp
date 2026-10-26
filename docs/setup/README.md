# Setting qmcp up

qmcp does several jobs, and each needs a different amount of setup. Do the
shared part below once, pick the job at hand, and follow that job's page; none
of them needs another's setup unless it says so. [`../roles.md`](../roles.md)
says what each job owns and where it stops.

| To | Role | Page | Needs beyond the shared part |
|---|---|---|---|
| change qmcp and prove the change | contributor | [`../contributing.md`](../contributing.md) | nothing |
| give an assistant or a flow tools, and stop for a person | tool server and human gate | [`tool-server.md`](tool-server.md) | an MCP client, or Python |
| run a repository's gates and hand the results to dossier | harness | [`harness.md`](harness.md) | a dossier checkout, for the panel |
| speak instructions at the machine and hear what the model found | instruction loop | [`../voice-loop-demo.md`](../voice-loop-demo.md) | Ollama, joe, the projects' clones |
| declare how agents are arranged, and save a design | topology designer | [`../agentframework/topologies.md`](../agentframework/topologies.md) | nothing, on a loopback bind |
| run the cookbook's flows, alfred renders among them | flow host | [`flows.md`](flows.md) | Docker Desktop |
| build agents whose tool calls are recorded here | agent library | [`../integrations/pydantic-ai.md`](../integrations/pydantic-ai.md) | a provider extra and its API key |

## Once, for every role

Needs git and [uv](https://docs.astral.sh/uv/), which installs the Python
3.13 the project pins.

```bash
git clone --recurse-submodules https://github.com/quaternionmedia/qmcp.git
cd qmcp
uv sync --all-extras
uv run pytest tests walkthrough -q
```

In a clone made without `--recurse-submodules`, run `git submodule update
--init` before the sync: `vendor/vox` is a path dependency, so any sync fails
while it is empty, and `governance/qm` holds the roster of projects and what
the suite's address vectors read. Run it again after every branch switch,
because branches pin different `vox` commits.

Install through the lock: `uv sync`, never `uv pip install <package>`, which
resolves a different stack. Every command is `uv run qmcp <command>`, the
server included; `uv run qmcp --help` lists them.

## Which server, and which queue

A qmcp server is one port and one database, and most setup mistakes are a
client talking to a different server, or a server writing to a different
database, than the one the person meant.

**The port.** `uv run qmcp serve` binds `127.0.0.1:3141`, the address the
rest of the stack is configured for: joe's page reaches qmcp there (`QMCP_URL`
moves it), dossier reads its harness there (`DOSSIER_HARNESS_PORT`), and
qmcp's own client and CLI default to it. A second `serve` on the same port
finds the first one healthy, says so, and exits. `--port 0` binds a port the
system picks instead.

**Which server answered.** Every `serve` leaves a run-file in `~/.qmcp/run/`
(`QMCP_RUN_DIR` moves it) naming its port, its resolved database and its start
time, and `GET /health` states the same identity with the session it was
started for (`QMCP_SESSION`) and a fingerprint of its database.

```bash
curl -s http://127.0.0.1:3141/health
```

**The database.** `QMCP_DATABASE_URL` defaults to
`sqlite+aiosqlite:///./qmcp.db`, a file relative to the directory the process
started in, so two clones hold two queues. That file is the human queue:
its pending requests are questions somebody has not answered yet.
`uv run qmcp test` leaves it alone; `--clean` deletes it, before and after the
run, with no prompt.

**One read is a write.** `GET /v1/human/requests/{id}` marks a pending request
past its expiry as expired, and the transition is stored. Do not poll it;
`GET /v1/human/requests` lists without expiring anything.

| Role | Port | Database |
|---|---|---|
| contributor, or any session measuring a server | `--port 0`, with `QMCP_PORT` set to the port its run-file names for the commands that call it | a scratch file of its own |
| tool server, instruction loop, harness beside dossier | 3141, one per workstation | the clone's `qmcp.db`, which is the queue |
| flow host | 3141 through `uv run qmcp cookbook serve` | the clone's `qmcp.db` |

Commands that call the server read `QMCP_PORT` (or take `--base-url`);
commands that read the queue directly, such as `qmcp human list`, `qmcp
dashboard` and `qmcp selfcheck`, read `QMCP_DATABASE_URL` or take
`--database`. Both settings, and every other one in `qmcp/config.py`, are
environment variables with the `QMCP_` prefix, and a `.env` file in the
directory qmcp starts in is read too.
