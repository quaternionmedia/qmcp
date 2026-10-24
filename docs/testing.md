# Testing

The suite runs offline: no network, no paid call, no microphone, no speakers,
no model. This page is what it isolates, and the harness a check uses to stand
up a qmcp of its own.

```
uv sync --all-extras
uv run pytest tests walkthrough -q
```

`.github/workflows/tests.yml` runs the same command on Linux. On Windows the
Metaflow flow tests skip, because `import metaflow` needs `fcntl`.

## What every test is isolated from

| The machine's | What the suite does instead |
|---|---|
| `QMCP_*` variables in the shell that ran pytest | removed before every test and walkthrough page (`qmcp.testing.plugin`, loaded from `addopts`) |
| a `.env` in the working directory | not read |
| `./qmcp.db`, the configured queue | never opened; each test builds its own database, in memory or under `tmp_path` |
| port 3141, the configured port | never bound; a server a test starts listens on port 0 |
| `~/.qmcp/voice-vocabulary.json`, a person's added phrases | an empty file of the test's own |
| the network | refused past loopback, in the test's process and in every Python child it starts; an attempt fails the test even when the code caught the refusal (`qmcp.testing.netguard`) |

The guard is installed before collection, so a module that reaches out when
imported stops the run. A child inherits it through `PYTHONPATH`, whose first
entry is a directory holding a `sitecustomize.py`, and records its attempts in
a file the session reads after each test; `UV_OFFLINE=1` keeps a `uv` the
child runs from fetching. A test marked `live_anthropic` is about a real
service and is let through.

Loopback is let through except on two ports: qmcp's default, where somebody's
queue answers, and the speech engine's, where a microphone opens. A test that
needs either starts its own on port 0. The local model service's port is let
through: a model on this machine is offline, and a test that runs one is an
offline test.

The settings isolation is `qmcp.testing.isolate_settings(monkeypatch)`.
`qmcp.config.get_settings` caches its first answer for the process, and
`qmcp.server` and `qmcp.db.engine` bind it at import, so the `client` fixture
replaces all three bindings with one `Settings` built for the test.

## A qmcp of the check's own

`qmcp.testing.throwaway_server(database)` is a real uvicorn server on
127.0.0.1, port 0, over the database file it is given. It yields the server's
URL and puts the settings cache and database engine back afterwards.
`walkthrough/09-nothing-runs-before-consent.md` runs one, with a client over
it.

Its callers:

- the product's offline checks: `uv run qmcp cookbook voice` and
  `uv run qmcp cookbook instruct`
- walkthrough pages 09, 10 and 11
- the tests that drive a flow over HTTP

The server runs in a thread of the caller's process. It sees the caller's
monkeypatches and shares its settings cache, so it cannot stand for a separate
server process. A path that crosses a process boundary needs a server started
as one.

## Stand-ins

| For | Stand-in | Held to |
|---|---|---|
| a speech engine | vox's deterministic engine (`vendor/vox/vox/engine.py`), on port 0 | vox's engine contract; it proves the wire, not transcription |
| an agent runtime | `qmcp.integrations.agents.scripted` | the runtime contract; spends nothing |
| alfred | `tests/alfred_stub.py`, behind `httpx.MockTransport`, or on a socket | `tests/alfred_health.json`, `alfred_captions.json` and `alfred_refusals.json`, copied from alfred |
| the local model service | `httpx.MockTransport` answering `/api/chat` (`tests/test_agents_local.py`), or the model itself with `-m local_model` | the shape the Ollama adapter reads |

A stand-in written for `httpx.MockTransport` reaches only the client it is
handed. `qmcp.testing.loopback.serve(handler)` puts the same handler on
127.0.0.1, port 0, so code that builds its own client from a configured URL
reaches it as it would the service. The tests marked `integration` cross a
real socket this way: the client library against a server, the alfred tools
logging in from `QMCP_ALFRED_*`, and a server over a database the migration
chain built. They run in the default suite.

## On the local model

`tests/test_local_model.py` runs on the model `uv run qmcp localmodel check`
reports on this machine: `qmcp.localmodel.MODEL` served at
`qmcp.localmodel.ENDPOINT`. A model on this machine is offline. The tests are
marked `local_model` and deselected by default, because CI has no model:

```
uv run pytest -m local_model
```

On a machine without the model they fail and say what is missing; they do not
skip. The runtime asks at temperature 0, so on one model the answers repeat
from run to run.

## What a green run does not show

- That anything outside the stand-ins behaves as they do. A stand-in held to a
  copied file drifts when the original changes, and nothing here compares them.
- A server in its own process. Every server the suite starts shares the test's
  process.
- An attempt the guard cannot see: from a child that is not Python, a child
  given an environment of its own making, or an extension with its own
  resolver.
- The flows on Windows, where they skip. A skip is not a pass.
- `-m live_anthropic` and the `--live` forms of the cookbook checks, which a
  person runs against real services.
