# Tools

Tools are the **primary capability exposed by the MCP server**.

In this system:
- Tools are **stateless**
- Tools are **pure functions where possible**
- Tools do not orchestrate or call other tools

There are no persistent "agents".
Planner, executor, and reviewer are simply tools with different roles.

---

## Tool Discovery

### Endpoint

```
GET /v1/tools
```

### Example Response

```json
{
  "tools": [
    {
      "name": "echo",
      "description": "Echo the input message back. Useful for testing.",
      "input_schema": {
        "type": "object",
        "properties": {
          "message": {"type": "string", "description": "The message to echo back"}
        },
        "required": ["message"]
      }
    },
    {
      "name": "planner",
      "description": "Create a step-by-step execution plan from a goal.",
      "input_schema": {
        "type": "object",
        "properties": {
          "goal": {"type": "string", "description": "The goal to create a plan for"},
          "context": {"type": "string", "description": "Optional context to inform planning"}
        },
        "required": ["goal"]
      }
    }
  ]
}
```

---

## Tool Invocation

### Endpoint

```
POST /v1/tools/{tool_name}
```

### Request Example

```json
{
  "input": {
    "goal": "Deploy a new service"
  },
  "correlation_id": "req-123"
}
```

### Response Example

```json
{
  "result": {
    "goal": "Deploy a new service",
    "steps": [
      {"step": 1, "action": "Analyze requirements"},
      {"step": 2, "action": "Identify dependencies"},
      {"step": 3, "action": "Create execution order"},
      {"step": 4, "action": "Validate plan"}
    ],
    "estimated_steps": 4
  },
  "error": null,
  "invocation_id": "08aa2d53-50b9-49d0-afad-61b95f9dd39a"
}
```

Every invocation returns an `invocation_id` for audit and tracing.

---

## Invocation History

All tool invocations are logged to the database for audit.

### List Invocations

```
GET /v1/invocations?tool_name=echo&status=success&limit=50
```

### Get Single Invocation

```
GET /v1/invocations/{invocation_id}
```

### Response Example

```json
{
  "id": "08aa2d53-50b9-49d0-afad-61b95f9dd39a",
  "tool_name": "echo",
  "input_params": {"message": "hello"},
  "result": "hello",
  "status": "success",
  "duration_ms": 1,
  "created_at": "2026-01-18T12:00:00Z",
  "completed_at": "2026-01-18T12:00:00Z",
  "correlation_id": null
}
```

---

## Tool Design Rules

All tools must:

1. Accept a single JSON payload (`input` field)
2. Return a JSON-serializable result
3. Be deterministic when possible
4. Avoid side effects unless explicitly required
5. Be stateless

---

## Built-in Tools

### echo

**Purpose:** Echo the input message back. Useful for testing connectivity.

**Input:**
```json
{"message": "Hello, MCP!"}
```

**Output:**
```json
"Hello, MCP!"
```

### planner

**Purpose:** Create a high-level execution plan from a goal.

**Input:**
```json
{"goal": "Deploy a new service", "context": "Production environment"}
```

**Output:**
```json
{
  "goal": "Deploy a new service",
  "steps": [
    {"step": 1, "action": "Analyze requirements"},
    {"step": 2, "action": "Identify dependencies"},
    {"step": 3, "action": "Create execution order"},
    {"step": 4, "action": "Validate plan"}
  ],
  "estimated_steps": 4
}
```

### executor

**Purpose:** Execute an approved plan. Returns execution summary.

**Input:**
```json
{"plan": {"steps": [...]}, "dry_run": true}
```

**Output:**
```json
{
  "mode": "dry_run",
  "steps_executed": 4,
  "results": [...],
  "success": true
}
```

### reviewer

**Purpose:** Review results and provide a summary assessment.

**Input:**
```json
{"result": {...}, "criteria": ["completeness", "correctness"]}
```

**Output:**
```json
{
  "reviewed_result": {...},
  "checks": [...],
  "overall_status": "approved",
  "recommendation": "Proceed with next steps"
}
```

---

## What Tools Must NOT Do

- ❌ Call other tools
- ❌ Store global state
- ❌ Block on human input
- ❌ Perform orchestration logic
- ❌ Make autonomous decisions

---

## Adding Custom Tools

See [contributing.md](contributing.md) for how to add new tools.

```python
from qmcp.tools import tool_registry

@tool_registry.register(
    name="my_tool",
    description="What this tool does",
    input_schema={
        "type": "object",
        "properties": {
            "param": {"type": "string"}
        },
        "required": ["param"],
    },
)
def my_tool(params: dict) -> Any:
    return params.get("param")
```
