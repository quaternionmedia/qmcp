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

## Best Practices

1. **Use meaningful IDs**: Include flow/run context in request IDs for traceability
2. **Set appropriate timeouts**: Match timeout to workflow urgency
3. **Provide rich context**: Include all information needed for decision
4. **Use options for constraints**: Prevent invalid responses with predefined choices
5. **Handle expiration gracefully**: Design workflows to handle timeout scenarios
