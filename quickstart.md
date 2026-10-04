# QMCP Quickstart

This is the shortest path to validate a fresh install and run a full
end-to-end workflow.

## 1) Install and run the end-to-end HITL test

```bash
git submodule update --init vendor/vox
uv sync --all-extras
uv run pytest tests/test_hitl.py::TestHITLWorkflow::test_complete_approval_workflow -v
```

The submodule line comes first because `vox` is a path dependency in the
default set: on a fresh clone with the submodule empty, any `uv sync` exits
2 before installing anything.

## 2) Start the server

```bash
uv run qmcp serve
```

Every command here is `uv run qmcp <command>`; `uv run qmcp --help` lists
them. Starting `serve` while a healthy server already holds the port says so
and exits, rather than failing at the bind.

For Docker-based flows, use the cookbook wrapper (binds to all interfaces by default):
```bash
uv run qmcp cookbook serve
```

On Windows, a running server holds `Scripts/qmcp.exe`. A sync that has to
reinstall qmcp — after pulling a change to its own dependencies — then fails
with *os error 32*. A sync with nothing to reinstall does not touch the file.
The running server is on the code from before that change anyway, so the
remedy is the restart it needed: stop it, and run the command again.

## 3) Call the server (curl)

```bash
curl -s http://localhost:3141/health
curl -s http://localhost:3141/v1/tools
curl -s -X POST http://localhost:3141/v1/tools/echo \
  -H "Content-Type: application/json" \
  -d '{"input":{"message":"hello"}}'
```

## 4) Call the server (PowerShell)

```powershell
Invoke-RestMethod -Method Get -Uri http://localhost:3141/health
Invoke-RestMethod -Method Get -Uri http://localhost:3141/v1/tools
$payload = @{ input = @{ message = "hello" } } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:3141/v1/tools/echo -ContentType "application/json" -Body $payload
```

## 5) The voice loop

The queue you just exercised can be answered by speaking. Prove the whole
voice path first, with no hardware, no engine and no model. A request is
queued on a throwaway server, answered through vox's deterministic engine and
read back, for four scripted answers: a yes, an option by name, a mismatch and
silence. It takes a few seconds and leaves this clone's queue alone:

```bash
uv run qmcp cookbook voice
```

Against a real engine — one that answers `vox.adapters.joe`'s contract, such
as joe — a pending request is answered by voice. The engine side, once, in
the engine's own checkout:

```bash
uv run joe voice setup   # find, save and prove your microphone
uv run joe backend       # the speech engine, on port 8000
```

Then here, with the server from step 2 running:

```bash
uv run vox doctor                   # engine reachable, a mic on its machine, synthesis
uv run qmcp cookbook voice --live   # one question aloud; say approve or hold
uv run qmcp human list              # what is waiting on a person
uv run qmcp human voice             # hear the question, say one of its options
```

`cookbook voice --live` queues one request of its own, `voice-check-<time>`,
which expires in five minutes, and reports what was recorded.

The question is spoken aloud with its options ("Say approve or hold."), and
recording stops when you do. If the server or the engine is not running,
`human voice` says which and names the command that starts it, before
anything is spoken. `human list` reads this clone's own database, whose
tables are created the first time the server starts — run step 2 before it,
or it fails with `no such table: human_requests`.

`docs/integrations/voice.md` carries the setup that makes this reliable:
which microphone the engine records from, which synthesizer speaks, and how
a spoken answer maps onto a request's own options.

## 6) The spoken instruction, on the local model

The loop also runs the other way: an instruction spoken at the machine,
consented to aloud, carried out by the model `qmcp localmodel` stands up, and
said back -- with each instruction told what the project's earlier ones found.
`docs/voice-loop-demo.md` runs it in three tiers; the first two need no
microphone:

```bash
uv run qmcp cookbook converse                     # one spoken session, every take scripted
uv run qmcp cookbook instruct --runtime local     # continuity, on the local model
```

`--runtime local` needs the model served: `uv run qmcp localmodel check` says
whether it is, and `uv run qmcp localmodel plan` gives the commands that
install it. With the speech engine running (step 5), `uv run qmcp serve
--converse --runtime local` in place of step 2's command makes everything
after it spoken: qmcp asks what should be done, and a person answers.

## Next Steps

- Read `docs/overview.md` for architecture boundaries.
- Read `docs/agentframework/overview.md` for agent schema and mixin status.
- Run example flows in `examples/flows/`.
- Run `qmcp cookbook dev simple-plan` to start the server and flow together.
- Or run `qmcp cookbook run simple-plan` (requires Docker Desktop).
- Other recipes: `qmcp cookbook run approved-deploy --service "api-gateway"`.
