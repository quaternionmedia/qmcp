# Integrations

QMCP integrates with external libraries to provide a complete agent development stack while avoiding duplication of maintained functionality.

## Philosophy

QMCP follows a "best of both worlds" approach:

1. **Use PydanticAI** for agent runtime, typed tools, and output validation
2. **Keep QMCP** for audit trails, HITL, model metadata, and multi-agent topologies

This hybrid architecture leverages actively-maintained libraries while preserving QMCP's unique production features.

## Available Integrations

### [PydanticAI](pydantic-ai.md)

LLM agent framework for Python. Provides:
- Agent execution with dependency injection
- Typed tool definitions with auto-generated schemas
- Output validation against Pydantic models
- Streaming and retry support
- Message history management

**Status**: Production-ready

```python
from qmcp.integrations.pydantic_ai import create_agent, QMCPToolset

agent = create_agent(
    Models.CLAUDE_SONNET_4,
    toolsets=[QMCPToolset()],
)
```

## Decision Matrix

When to use what:

| Need | Use |
|------|-----|
| Run an LLM agent | PydanticAI via `create_agent()` |
| Define typed tools for agents | PydanticAI `@agent.tool` |
| Expose tools via HTTP | QMCP `tool_registry` |
| Audit all tool calls | QMCP MCP Server |
| Human approval workflow | QMCP HITL API |
| Model pricing/limits | QMCP `Models` registry |
| Multi-agent patterns | QMCP Topologies |
| Structured output | PydanticAI `output_type` |
| Token limits | PydanticAI `UsageLimits` |
| Persistent memory | QMCP `MemoryMixin` |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Application                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐     ┌─────────────────────┐       │
│  │   PydanticAI Agent  │────▶│    QMCP Topologies  │       │
│  │   (Runtime)         │     │    (Orchestration)  │       │
│  └─────────────────────┘     └─────────────────────┘       │
│           │                           │                     │
│           ▼                           ▼                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              QMCP Model Registry                     │   │
│  │   (Pricing, Limits, Capabilities, Fallbacks)        │   │
│  └─────────────────────────────────────────────────────┘   │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           QMCP MCP Server (FastAPI)                  │   │
│  │   (Audit Trail, HITL, Metrics)                       │   │
│  └─────────────────────────────────────────────────────┘   │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         QMCPToolset (PydanticAI Adapter)             │   │
│  │   (Connect agents to QMCP server)                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Future Integrations

Planned integrations:

| Integration | Purpose | Status |
|-------------|---------|--------|
| LangChain | Alternative agent framework | Planned |
| Instructor | Structured extraction | Considered |
| Marvin | AI functions | Considered |
