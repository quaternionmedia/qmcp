# qmcp

An AI assistant can only use a tool somebody has exposed to it, and some of
what it does should wait for a person. qmcp is the server that keeps the
record for both: it runs the tools assistants and flows call, holds the
questions that wait on a person, and remembers each instruction it was given
and what came of it, so work carries from one session to the next.

It is a FastAPI server with its record in SQLite, listening on
`127.0.0.1:3141` (`QMCP_PORT` moves it), with a command line for every part:
`uv run qmcp --help`.

## What it is for

qmcp plays several roles in the Quaternion Media stack, and each is set up on
its own. [`docs/roles.md`](docs/roles.md) says what each owns and what it
leaves to another program.

| Set it up to | Page |
|---|---|
| give an assistant or a flow tools, and stop for a person | [`docs/setup/tool-server.md`](docs/setup/tool-server.md) |
| run a repository's gates and hand the results to dossier | [`docs/setup/harness.md`](docs/setup/harness.md) |
| speak instructions at the machine and hear what the model found | [`docs/voice-loop-demo.md`](docs/voice-loop-demo.md) |
| declare how agents are arranged, and save a design | [`docs/agentframework/topologies.md`](docs/agentframework/topologies.md) |
| run the cookbook's flows, alfred renders among them | [`docs/setup/flows.md`](docs/setup/flows.md) |
| build agents whose tool calls are recorded here | [`docs/integrations/pydantic-ai.md`](docs/integrations/pydantic-ai.md) |
| change qmcp itself | [`docs/contributing.md`](docs/contributing.md) |

## Start

Needs git and [uv](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules https://github.com/quaternionmedia/qmcp.git
cd qmcp
uv sync --all-extras
uv run pytest tests walkthrough -q
uv run qmcp serve
```

Before starting a server on a machine that may already run one, read "Which
server, and which queue" in [`docs/setup/README.md`](docs/setup/README.md):
the clone's database holds questions somebody has not answered yet.

## Documentation

[`docs/README.md`](docs/README.md) lists every page by the role it serves.

## License

MIT, as `REUSE.toml` declares for this project's own files; vendored and
third-party files carry their own.
