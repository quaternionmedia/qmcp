# Agent Framework: Runners

A runner takes a topology and its agents and runs them somewhere. One runner
runs today, `MetaflowRunner`; the other two registered runners are declared and
not built. This page is what `qmcp/agentframework/runners.py` holds.

## Base classes

```python
class RunConfig(SQLModel):
    timeout_seconds: int = 300
    max_retries: int = 3
    parallelism: int = 1
    save_intermediate: bool = True
    metadata_: dict[str, Any] = {}
    # Metaflow
    docker: bool = False          # run the flow in the flow image, not here
    mcp_url: str | None = None    # passed to the flow as --mcp-url
    metaflow_user: str | None = None
    build_image: bool = True      # with docker: build the image first
    sync_deps: bool = True        # with docker: sync inside the container

class RunResult(SQLModel):
    execution_id: str | None
    status: str                   # "completed", "failed" or "unknown"
    output: dict | None           # stdout, returncode, and stderr when it failed
    error: str | None
    duration_ms: int | None
```

`BaseRunner(config)` declares `async run(topology, agents) -> RunResult`, which
raises `NotImplementedError` until a subclass provides it.

## MetaflowRunner

Runs a topology as a Metaflow flow, in a subprocess.

```python
from qmcp.agentframework.runners import MetaflowRunner, RunConfig

runner = MetaflowRunner(RunConfig(timeout_seconds=600, mcp_url="http://127.0.0.1:3141"))
result = await runner.run(topology, agents, flow_args=["--goal", "Ship a health check"])
```

The flow script is the `flow_path` given to the constructor, or the one the
topology's type maps to in `_TOPOLOGY_FLOW_MAP`:

| Topology type | Flow |
|---|---|
| `pipeline` | `examples/flows/local_agent_chain.py` |
| `council` | `examples/flows/council_deliberation.py` |

It runs `python <flow> run` with `flow_args`, `--mcp-url` and `METAFLOW_USER`,
or, with `RunConfig.docker`, the same inside the flow image as
`qmcp cookbook run` does. A run past `timeout_seconds` ends `failed`. Metaflow
imports only on Linux and macOS, so on Windows run it in the image or under
WSL.

## LocalRunner and AsyncRunner

Both are registered, as `local` and `async`, and neither is built: `run`
raises `NotImplementedError`. An in-process runner waits on topology
execution, and `BaseTopology.run` is still a stub (`docs/ROADMAP.md`, *Next
Steps*).

## Runner registry

```python
from qmcp.agentframework.runners import BaseRunner, RunnerRegistry, runner

metaflow = RunnerRegistry.create("metaflow", RunConfig())

@runner
class MyRunner(BaseRunner):
    runner_type = "my_runner"

    async def run(self, topology, agents):
        ...
```

## Not built

Generating a flow from a topology, a `metaflow_helpers` module for flow steps,
and an `agents` command group for qmcp are planned and do not exist. The flows in
`examples/flows/` are written by hand. The commands that exist for agents and
topologies are `qmcp council` and `qmcp topology`; `uv run qmcp --help` lists
them.
