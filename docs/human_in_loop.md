# Human-in-the-Loop (HITL) Guide

This document describes the Human-in-the-Loop capabilities of the QMCP server.

## Overview

HITL enables AI-driven workflows to pause and request human input before proceeding with critical operations. This is essential for:

- **Approval workflows**: Deploy to production, delete resources, send emails
- **Input collection**: Gather additional context, clarify ambiguous requests
- **Review checkpoints**: Validate AI-generated content before publication

## Architecture

```
┌─────────────┐     1. Create Request     ┌─────────────┐
│   Client    │ ──────────────────────────▶│    QMCP     │
│  (Metaflow) │                            │   Server    │
└─────────────┘                            └─────────────┘
      │                                           │
      │         2. Poll for Response              │
      │ ◀────────────────────────────────────────┤
      │                                           │
      │                                     ┌─────────────┐
      │                                     │   Human     │
      │                                     │  Operator   │
      │                                     └─────────────┘
      │                                           │
      │                                    3. Submit Response
      │                                           │
      │         4. Receive Response               ▼
      │ ◀────────────────────────────────────────┤
      ▼                                           │
┌─────────────┐                            ┌─────────────┐
│  Continue   │                            │  Response   │
│  Workflow   │                            │   Stored    │
└─────────────┘                            └─────────────┘
```

## API Reference

### Create a Human Request

```http
POST /v1/human/requests
Content-Type: application/json

{
  "id": "approve-deploy-001",
  "request_type": "approval",
  "prompt": "Approve deployment to production?",
  "options": ["approve", "reject"],
  "timeout_seconds": 3600,
  "context": {
    "service": "api-gateway",
    "environment": "production",
    "commit": "abc123"
  },
  "correlation_id": "flow-123"
}
```

**Response (201 Created):**
```json
{
  "id": "approve-deploy-001",
  "request_type": "approval",
  "prompt": "Approve deployment to production?",
  "status": "pending",
  "created_at": "2026-01-18T22:00:00Z",
  "expires_at": "2026-01-18T23:00:00Z"
}
```

### List Human Requests

```http
GET /v1/human/requests?status=pending&request_type=approval&limit=50
```

**Response:**
```json
{
  "requests": [
    {
      "id": "approve-deploy-001",
      "request_type": "approval",
      "prompt": "Approve deployment to production?",
      "status": "pending",
      "created_at": "2026-01-18T22:00:00Z",
      "expires_at": "2026-01-18T23:00:00Z"
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

### Get a Single Request

```http
GET /v1/human/requests/approve-deploy-001
```

**Response (with response):**
```json
{
  "request": {
    "id": "approve-deploy-001",
    "request_type": "approval",
    "prompt": "Approve deployment to production?",
    "options": ["approve", "reject"],
    "context": {
      "service": "api-gateway",
      "environment": "production"
    },
    "status": "responded",
    "created_at": "2026-01-18T22:00:00Z",
    "expires_at": "2026-01-18T23:00:00Z"
  },
  "response": {
    "id": "resp-001",
    "request_id": "approve-deploy-001",
    "response": "approve",
    "responded_by": "alice@example.com",
    "created_at": "2026-01-18T22:05:00Z"
  }
}
```

### Submit a Human Response

```http
POST /v1/human/responses
Content-Type: application/json

{
  "request_id": "approve-deploy-001",
  "response": "approve",
  "responded_by": "alice@example.com",
  "response_metadata": {
    "notes": "Reviewed and approved"
  }
}
```

**Response (201 Created):**
```json
{
  "id": "resp-001",
  "request_id": "approve-deploy-001",
  "response": "approve",
  "responded_by": "alice@example.com",
  "created_at": "2026-01-18T22:05:00Z"
}
```

## Request Types

| Type | Description | Typical Use |
|------|-------------|-------------|
| `approval` | Yes/no decision | Deploy, delete, send |
| `input` | Free-form text | Clarification, context |
| `review` | Content validation | Document review, code review |

## Status Lifecycle

```
pending ─────┬────▶ responded (human submitted response)
             │
             └────▶ expired (timeout elapsed, no response)
```

## Error Responses

| Status | Condition |
|--------|-----------|
| 404 | Request ID not found |
| 409 | Request ID already exists |
| 409 | Already responded (cannot respond twice) |
| 410 | Request has expired |
| 400 | Response not in allowed options |

## Client Integration Example

```python
import httpx
import time

# Create request
response = httpx.post("http://localhost:3333/v1/human/requests", json={
    "id": f"approve-{run_id}",
    "request_type": "approval",
    "prompt": "Deploy to production?",
    "options": ["approve", "reject"],
    "timeout_seconds": 3600,
})
request_id = response.json()["id"]

# Poll for response
while True:
    result = httpx.get(f"http://localhost:3333/v1/human/requests/{request_id}")
    data = result.json()
    
    if data["request"]["status"] == "responded":
        decision = data["response"]["response"]
        if decision == "approve":
            # Continue with deployment
            pass
        else:
            # Abort deployment
            pass
        break
    elif data["request"]["status"] == "expired":
        raise TimeoutError("Human approval timed out")
    
    time.sleep(30)  # Poll every 30 seconds
```

## End-to-End Walkthrough (Live Server)

This mirrors `tests/test_hitl.py::TestHITLWorkflow::test_complete_approval_workflow`.

Start the server in one terminal:

```bash
uv run qmcp serve
```

Then run the workflow in another terminal.

### Bash (curl)

```bash
curl -s -X POST http://localhost:3333/v1/human/requests \
  -H "Content-Type: application/json" \
  -d '{"id":"workflow-001","request_type":"approval","prompt":"Approve deployment to production?","options":["approve","reject"],"context":{"service":"api-gateway","environment":"prod"}}'

curl -s http://localhost:3333/v1/human/requests/workflow-001

curl -s -X POST http://localhost:3333/v1/human/responses \
  -H "Content-Type: application/json" \
  -d '{"request_id":"workflow-001","response":"approve","responded_by":"ops@example.com","response_metadata":{"reason":"Looks good"}}'

curl -s http://localhost:3333/v1/human/requests/workflow-001
```

### PowerShell

```powershell
$create = @{
  id = "workflow-001"
  request_type = "approval"
  prompt = "Approve deployment to production?"
  options = @("approve","reject")
  context = @{ service = "api-gateway"; environment = "prod" }
} | ConvertTo-Json -Depth 4
Invoke-RestMethod -Method Post -Uri http://localhost:3333/v1/human/requests -ContentType "application/json" -Body $create
Invoke-RestMethod -Method Get -Uri http://localhost:3333/v1/human/requests/workflow-001

$respond = @{
  request_id = "workflow-001"
  response = "approve"
  responded_by = "ops@example.com"
  response_metadata = @{ reason = "Looks good" }
} | ConvertTo-Json -Depth 4
Invoke-RestMethod -Method Post -Uri http://localhost:3333/v1/human/responses -ContentType "application/json" -Body $respond
Invoke-RestMethod -Method Get -Uri http://localhost:3333/v1/human/requests/workflow-001
```

## Where a request comes from, when a model produced it

The API above takes a request from anything that can post one. **When the thing
being reviewed is a model's output, it arrives through one seam and not through
that endpoint directly** -- `qmcp/governed.py`.

    in -> budget -> model -> bound -> draft -> a person
            |                  |
            v                  v
         refused            stopped

A fixed sequence of total stages around exactly one call to something with no
halting guarantee: budgeted before it through `qmcp.spend`, bounded after it,
and ending here. `governed.queued(outcome)` builds the `POST
/v1/human/requests` payload above and does nothing with it. **Building it is not
posting it, and posting it is not answering it.**

Three things follow that a caller of this API should know:

- **A refused run is queued too.** A run that never called anything because the
  budget was zero produces a request with `context.state == "refused"` and an
  empty draft. Somebody deciding whether to authorise a paid retry needs that in
  front of them more than they need the runs that succeeded.
- **`context.this_is_a_draft` is always `true`.** Said outright rather than left
  for a reader to infer from a text field.
- **Answering is a person's.** `governance/qm/ci/attested-registry.yaml` names
  answering a question in this queue as one of the acts reserved for a person,
  and `qmcp.governed` offers no function that does it -- a test fails the moment
  somebody adds one.

And one thing the seam **does not enforce**, said here because a reader who
assumes otherwise is the reader it costs: **a slow call is not stopped.**
`Bound.seconds` is measured, and `context.over_bound` reports whether it was
exceeded, but nothing interrupts the call — a run can exceed its expectation and
still arrive as a draft, and that is the honest report rather than a refusal
nothing carried out. Interrupting a callable that module does not own is not
something Python offers for free, and a check claiming it would be a green one
standing exactly where you believe something is enforced.

The shape is servable and drawable:

```bash
uv run qmcp topology show governed --level 2   # as text
curl localhost:8000/v1/topology/shape/governed # as a payload, for a window
```

Why the seam exists at all is `governance/qm/records/DRAFT-shrink-the-black-box.md`.

## Best Practices

1. **Use meaningful IDs**: Include flow/run context in request IDs for traceability
2. **Set appropriate timeouts**: Match timeout to workflow urgency
3. **Provide rich context**: Include all information needed for decision
4. **Use options for constraints**: Prevent invalid responses with predefined choices
5. **Handle expiration gracefully**: Design workflows to handle timeout scenarios
