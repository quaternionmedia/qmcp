# Flow host

A flow is a sequence of steps that calls tools, waits for a person where it
should, and records what it did. In this role qmcp is the server the
cookbook's Metaflow flows call: each flow runs in a Linux container, reaches
qmcp over HTTP for its tools and its approvals, and keeps its own record in a
flow database. alfred's renders and captions are flows of this kind, put to a
person before anything is delivered.

It needs the shared setup in [`README.md`](README.md), Docker Desktop with the
Linux engine, and, for the local-model recipes, a model served the way
`uv run qmcp localmodel plan` prints.

## What there is to run

```bash
uv run qmcp cookbook list
```

Every recipe runs in Docker. Metaflow does not import on Windows, so a
container is the only way to run one there, and the flow tests skip on
Windows and run on the hosted runner.

## Run one

```bash
uv run qmcp cookbook dev simple-plan --goal "Deploy a web service"
```

`cookbook dev` starts a server when none answers on its port, uses one that
answers over this checkout's database, refuses one that answers over another
database (`--server-port` gives it one of its own), and runs the recipe in
Docker against it. To keep them apart, start the server in
one terminal and run the recipe in another:

```bash
uv run qmcp cookbook serve
uv run qmcp cookbook run approved-deploy --service "api-gateway" --environment "staging"
```

`cookbook serve` binds `127.0.0.1`, which a container on Docker Desktop
reaches. On a Linux engine a container reaches it only with `--host 0.0.0.0`
(`--server-host 0.0.0.0` for `cookbook dev`), and then so does anyone who can
reach this machine; a server bound that way also stops serving the
loopback-only routes ([`../roles.md`](../roles.md), "One bind"). Add
`--no-sync` to skip syncing the image's dependencies once it is built.

The compose file runs a flow directly:

```bash
docker compose -f docker-compose.flows.yml build
docker compose -f docker-compose.flows.yml run --rm flow-runner \
  examples/flows/local_agent_chain.py run --use-mcp True --goal "..."
```

Inside a container, `MCP_URL` and `LLM_BASE_URL` (or `--mcp-url` and
`--llm-base-url`) name the host, as in `http://host.docker.internal:3141`.

## alfred

The `alfred_*` tools drive a running alfred: `QMCP_ALFRED_URL` says where it
answers, and an account or a token says who qmcp is to it. Unset, every alfred
tool says alfred is not configured.

```bash
uv run qmcp cookbook run alfred-render --project-id <id>
```

[`../integrations/alfred.md`](../integrations/alfred.md) is the reference: the
settings, the tools, and the two recipes.

## Reference

- [`../../examples/flows/`](../../examples/flows/): the flows, one file each
- [`../agentframework/runners.md`](../agentframework/runners.md): how a flow's steps are built
- [`../client.md`](../client.md): the client a flow calls qmcp with
