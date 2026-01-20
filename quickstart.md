# QMCP Quickstart

This is the shortest path to validate a fresh install and run a full
end-to-end workflow.

## 1) Install and run the end-to-end HITL test

```bash
uv sync --all-extras
uv run pytest tests/test_hitl.py::TestHITLWorkflow::test_complete_approval_workflow -v
```

## 2) Start the server

```bash
uv run qmcp serve
```

## 3) Call the server (curl)

```bash
curl -s http://localhost:3333/health
curl -s http://localhost:3333/v1/tools
curl -s -X POST http://localhost:3333/v1/tools/echo \
  -H "Content-Type: application/json" \
  -d '{"input":{"message":"hello"}}'
```

## 4) Call the server (PowerShell)

```powershell
Invoke-RestMethod -Method Get -Uri http://localhost:3333/health
Invoke-RestMethod -Method Get -Uri http://localhost:3333/v1/tools
$payload = @{ input = @{ message = "hello" } } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:3333/v1/tools/echo -ContentType "application/json" -Body $payload
```

## Next Steps

- Read `docs/overview.md` for architecture boundaries.
- Read `docs/agentframework.md` for agent schema and mixin status.
- Run example flows in `examples/flows/`.
