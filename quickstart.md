# QMCP Quickstart

This is the shortest path to validate a fresh install and run a full
end-to-end workflow.

## 1) Install and run the end-to-end HITL test

```bash
git submodule update --init vox
uv sync --all-extras
uv run pytest tests/test_hitl.py::TestHITLWorkflow::test_complete_approval_workflow -v
```

The submodule line comes first because `vox` is a path dependency of the
`voice` extra: on a fresh clone with the submodule empty, `uv sync
--all-extras` exits 2 before installing anything.

## 2) Start the server

```bash
uv run qmcp serve
```

For Docker-based flows, use the cookbook wrapper (binds to all interfaces by default):
```bash
uv run qmcp cookbook serve
```

If the qmcp shim cannot be installed (Windows), use:
```bash
uv run --no-sync python -m qmcp serve
```

While the server runs from this clone, give every further `uv run` here the
`--no-sync` flag: a bare `uv run` re-syncs the environment first, and on
Windows that aborts partway when it cannot replace the running `qmcp.exe`.

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

The queue you just exercised can be answered by speaking. Prove the speech
seam first with no hardware, no engine and no model — the loop closes in
under a second and writes its audio under `Data/`, which is ignored:

```bash
uv run --no-sync vox loop --offline
```

Against a real engine (one that answers `vox.adapters.joe`'s contract, such
as `joe backend` on port 8000), the same loop runs through real
transcription, and a pending request is answered by voice:

```bash
uv run --no-sync vox doctor           # engine reachable, a mic on its machine, synthesis
uv run --no-sync qmcp human list      # what is waiting on a person
uv run --no-sync qmcp human voice     # speak the answer to the oldest of them
```

`human list` reads this clone's own database, whose tables are created the
first time the server starts — run step 2 before it, or it fails with
`no such table: human_requests`.

`docs/integrations/voice.md` carries the setup that makes this reliable:
which microphone the engine records from, which synthesizer speaks, and how
a spoken answer maps onto a request's own options.

## Next Steps

- Read `docs/overview.md` for architecture boundaries.
- Read `docs/agentframework/overview.md` for agent schema and mixin status.
- Run example flows in `examples/flows/`.
- Run `qmcp cookbook dev simple-plan` to start the server and flow together.
- Or run `qmcp cookbook run simple-plan` (requires Docker Desktop).
- Windows fallback: `uv run --no-sync python -m qmcp cookbook run simple-plan`.
- Other recipes: `qmcp cookbook run approved-deploy --service "api-gateway"`.
