# Tool server and human gate

An assistant or a flow can only use a tool somebody has exposed to it, and
some of its steps should wait for a person. In this role qmcp publishes tools,
runs them when asked, records every invocation, and holds the questions a
person has to answer. It needs only the shared setup in
[`README.md`](README.md).

## Prove the install

The end-to-end approval test creates a request, answers it, and reads the
answer back through the real HTTP path:

```bash
uv run pytest tests/test_hitl.py::TestHITLWorkflow::test_complete_approval_workflow -v
```

## Serve

```bash
uv run qmcp serve
```

`serve` binds `127.0.0.1:3141` and creates its tables the first time it
starts. On Windows a running server holds `Scripts/qmcp.exe`, so a sync that
reinstalls qmcp fails with os error 32 until the server is stopped; a change to
qmcp's own dependencies needs that restart anyway.

Call it:

```bash
curl -s http://127.0.0.1:3141/health
curl -s http://127.0.0.1:3141/v1/tools
curl -s -X POST http://127.0.0.1:3141/v1/tools/echo \
  -H "Content-Type: application/json" \
  -d '{"input":{"message":"hello"}}'
```

From PowerShell:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:3141/health
$payload = @{ input = @{ message = "hello" } } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:3141/v1/tools/echo -ContentType "application/json" -Body $payload
```

## Connect a caller

There are two ways in, and they reach the same server.

- **Over HTTP**, from code: `qmcp.client.MCPClient` lists and invokes tools,
  creates human requests and waits for their answers. It defaults to the port
  this machine's settings name, so `QMCP_PORT` moves both ends together.
  [`../client.md`](../client.md) is its reference.
- **Over the Model Context Protocol**, from an assistant: `qmcp_mcp.py` is an
  MCP server on stdio, in the `mcp` extra, that lists and invokes the running
  server's tools, reads its invocations, lists its human queue, and runs this
  repository's recipes and tests. It cannot answer a human request. Its module
  docstring carries the client configuration; `QMCP_SERVER_URL` points it at a
  server other than the default.

  ```bash
  uv run --extra mcp python qmcp_mcp.py
  ```

## Answer what is waiting

Answering a request is a person's act, and nothing in qmcp answers one on a
person's behalf.

```bash
uv run qmcp human list
uv run qmcp human respond <request-id> approve --by "<your name>"
```

`qmcp human list` reads the queue from the database directly. The same
questions can be answered in dossier, the control panel, which reads and
answers the queue over HTTP; by speaking (`uv run qmcp human voice`, set up in
[`../integrations/voice.md`](../integrations/voice.md)); or from joe's page.

## Bound off loopback

The server asks nobody who they are: whoever reaches the port can invoke its
tools and answer its human requests. Keep it on `127.0.0.1` unless the
network around it is trusted. The routes that write designs, the thread
archive, the voice routes and the instruction inbox are registered only on a
loopback bind. [`../deployment.md`](../deployment.md) covers running it
elsewhere.

## Reference

- [`../tools.md`](../tools.md): tool discovery, invocation and history
- [`../human_in_loop.md`](../human_in_loop.md): the human queue's API and lifecycle
- [`../client.md`](../client.md): the Python client
