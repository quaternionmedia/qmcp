# PydanticAI vs QMCP: Feature Comparison & Migration Plan

> **Status: IMPLEMENTED**
>
> This analysis has been implemented. See [PydanticAI Integration](integrations/pydantic-ai.md)
> for usage documentation.
>
> Key implementations:
> - `qmcp.integrations.pydantic_ai.create_agent()` - Create agents from QMCP models
> - `qmcp.integrations.pydantic_ai.QMCPToolset` - Connect agents to QMCP server
> - `qmcp.integrations.pydantic_ai.model_to_pydantic_ai()` - Convert model configs

This document analyzes the overlap between PydanticAI and QMCP to identify functionality
that can be replaced by PydanticAI, reducing maintenance burden and leveraging their
actively-maintained library.

## Executive Summary

| Area | PydanticAI | QMCP | Recommendation |
|------|------------|------|----------------|
| **MCP Server** | FastMCP (from MCP SDK) | Custom FastAPI server | **Keep QMCP** - more features |
| **MCP Client** | MCPServerStdio/HTTP/SSE | httpx-based client | **Replace with PydanticAI** |
| **Agent Runtime** | Full agent with tools/deps | Skeleton classes only | **Replace with PydanticAI** |
| **Tool System** | Decorator + type inference | Manual JSON schema | **Replace with PydanticAI** |
| **Model Registry** | Dynamic string-based | Static ModelConfig constants | **Keep QMCP** - richer metadata |
| **Topologies** | Not supported | Debate, Pipeline, Ensemble, etc. | **Keep QMCP** - unique |
| **HITL** | Elicitation callbacks | Full REST API + persistence | **Keep QMCP** - more complete |
| **Mixins** | Not supported | Capability extension system | **Keep QMCP** - unique |

---

## 1. MCP Server Comparison

### PydanticAI Approach
```python
from mcp.server.fastmcp import FastMCP

server = FastMCP('My Server')

@server.tool()
async def my_tool(param: str) -> str:
    return f"Result: {param}"
```

- Uses FastMCP from MCP SDK directly
- Stdio transport primarily
- Agent integration via `MCPSamplingModel`
- No persistence, no audit trail

### QMCP Approach
```python
from qmcp.tools import tool_registry

@tool_registry.register("my_tool", "Description", input_schema={...})
def my_tool(params: dict) -> Any:
    return result
```

- Custom FastAPI server with HTTP transport
- **Unique features**:
  - Audit trail (all invocations persisted to DB)
  - Human-in-the-loop REST API with polling
  - Prometheus metrics (`/metrics`)
  - Request tracing with correlation IDs
  - Invocation history querying

### Recommendation: **Keep QMCP Server**

QMCP's MCP server provides production features (audit, HITL, metrics) that PydanticAI
doesn't offer. The HTTP transport is also more suitable for distributed systems.

---

## 2. MCP Client Comparison

### PydanticAI Approach
```python
from pydantic_ai.mcp import MCPServerStdio, MCPServerStreamableHTTP

# Stdio transport
server = MCPServerStdio('uvx', 'some-mcp-server')

# HTTP transport
server = MCPServerStreamableHTTP('https://api.example.com/mcp')

# Use with agent
agent = Agent('model', toolsets=[server])
```

- Three transport options: Stdio, HTTP (Streamable), SSE
- Tool prefixing to avoid conflicts
- Connection lifecycle management
- Config loading from JSON files

### QMCP Approach
```python
from qmcp.client import MCPClient

with MCPClient("http://localhost:3333") as client:
    tools = client.list_tools()
    result = client.invoke_tool("echo", {"message": "hello"})

    # HITL support
    response = client.request_human_approval("req-1", "Deploy?")
```

- HTTP-only, synchronous
- HITL polling built-in
- No toolset abstraction

### Recommendation: **Replace with PydanticAI**

PydanticAI's client is more flexible (multiple transports) and integrates directly
with agents as toolsets. QMCP's HITL client functionality can be retained as a
standalone utility or wrapped in a custom toolset.

**Migration Path:**
1. Create `QMCPToolset` wrapper that uses PydanticAI's `AbstractToolset`
2. Add QMCP-specific HITL methods as additional toolset functionality
3. Deprecate `qmcp.client.MCPClient`

---

## 3. Agent Runtime Comparison

### PydanticAI Approach
```python
from pydantic_ai import Agent, RunContext

agent = Agent(
    'openai:gpt-4o',
    deps_type=MyDeps,
    output_type=MyOutput,
    system_prompt="You are helpful...",
)

@agent.tool
async def my_tool(ctx: RunContext[MyDeps], query: str) -> str:
    db = ctx.deps.database
    return db.search(query)

result = await agent.run("User prompt", deps=MyDeps(...))
```

- Full agent implementation with:
  - Dependency injection via `RunContext`
  - Typed output validation
  - Dynamic system prompts
  - Tool execution with retries
  - Streaming support
  - Usage limits
  - Message history management

### QMCP Approach
```python
# qmcp/agentframework/runners.py - SKELETON ONLY
class BaseRunner(ABC):
    async def run(self, topology: Topology, agents: dict[str, AgentType]) -> RunResult:
        raise NotImplementedError("Runner execution is not implemented yet.")
```

- **Not implemented** - just type skeletons
- AgentConfig defines configuration structure
- No actual LLM invocation logic
- Designed for future implementation

### Recommendation: **Replace with PydanticAI**

QMCP's agent runtime is unimplemented. PydanticAI provides a complete, production-ready
agent framework that should be adopted entirely.

**Migration Path:**
1. Keep QMCP's `AgentConfig` as configuration layer
2. Create adapter: `AgentConfig` -> PydanticAI `Agent`
3. Use PydanticAI's agent for all LLM interactions
4. Remove skeleton runner classes

---

## 4. Tool System Comparison

### PydanticAI Approach
```python
@agent.tool
async def get_weather(ctx: RunContext[Deps], city: str, units: str = "celsius") -> str:
    """Get weather for a city.

    Args:
        city: The city name
        units: Temperature units (celsius/fahrenheit)
    """
    return await ctx.deps.weather_api.get(city, units)
```

- Schema auto-generated from type hints
- Docstring parsing for descriptions (griffe)
- Context injection via `RunContext`
- Retry support via `ModelRetry` exception
- Prepare functions for dynamic tool modification

### QMCP Approach
```python
@tool_registry.register(
    name="get_weather",
    description="Get weather for a city",
    input_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "units": {"type": "string", "default": "celsius"}
        },
        "required": ["city"]
    }
)
def get_weather(params: dict) -> str:
    return weather_api.get(params["city"], params.get("units", "celsius"))
```

- Manual JSON schema definition
- No type inference
- No context injection
- Simpler but more verbose

### Recommendation: **Replace with PydanticAI**

PydanticAI's tool system is more ergonomic and type-safe. The automatic schema
generation from type hints eliminates boilerplate and reduces errors.

**Migration Path:**
1. Keep `tool_registry` for MCP server tools (HTTP endpoint exposure)
2. Use PydanticAI tools for agent-side tool definitions
3. Tools can be registered in both systems when needed

---

## 5. Model Configuration Comparison

### PydanticAI Approach
```python
from pydantic_ai import Agent
from pydantic_ai.models import FallbackModel

# Simple string
agent = Agent('openai:gpt-4o')

# With fallback
agent = Agent(FallbackModel('openai:gpt-4o', 'anthropic:claude-3-haiku'))

# Dynamic at runtime
result = await agent.run("prompt", model='anthropic:claude-3-opus')
```

- String-based model specification
- Provider auto-detection from prefix
- Fallback chains
- Runtime model override
- Per-model settings

### QMCP Approach
```python
from qmcp.agentframework.models import Models, ModelConfig

# Pre-defined constants with full metadata
agent_config = AgentConfig(model_config_obj=Models.CLAUDE_SONNET_4)

# Access metadata
Models.CLAUDE_SONNET_4.pricing.input_cost  # 3.0 per 1M tokens
Models.CLAUDE_SONNET_4.limits.context_window  # 200000
Models.CLAUDE_SONNET_4.capabilities.supports_tool_choice  # True

# Query registry
anthropic_models = Models.by_provider(ModelProvider.ANTHROPIC)
fast_models = Models.by_tier(ModelTier.FAST)
```

- Rich metadata per model (pricing, limits, capabilities)
- Type-safe constants
- Registry queries
- Fallback configuration per-model

### Recommendation: **Keep QMCP Model Registry**

QMCP's model registry provides valuable metadata (pricing, limits, capabilities) that
PydanticAI doesn't track. This is useful for:
- Cost estimation
- Capability checking before agent creation
- Model selection based on requirements

**Integration Path:**
1. Keep `Models` registry for metadata
2. Create helper: `Models.CLAUDE_SONNET_4.to_pydantic_ai()` -> `"anthropic:claude-sonnet-4"`
3. Use model metadata for agent configuration decisions

---

## 6. Unique QMCP Features (No PydanticAI Equivalent)

### Topologies
Multi-agent orchestration patterns that PydanticAI doesn't support:

| Topology | Description | Use Case |
|----------|-------------|----------|
| Debate | Agents argue, mediator decides | Complex decisions |
| Pipeline | Sequential processing | Workflows |
| Ensemble | Parallel + aggregation | Diverse perspectives |
| CrossCheck | Independent verification | Quality assurance |
| ChainOfCommand | Hierarchical delegation | Large tasks |

**Recommendation**: Keep and implement. These are unique value-add.

### Mixins
Composable agent capabilities:

| Mixin | Description |
|-------|-------------|
| ToolUse | Tool call filtering/limits |
| Memory | Persistent memory with decay |
| Reasoning | CoT prompt injection |
| HumanInLoop | Approval workflow |
| StructuredOutput | JSON schema enforcement |

**Recommendation**: Some overlap with PydanticAI features:
- `StructuredOutput` -> Use PydanticAI's `output_type`
- `ToolUse` limits -> Use PydanticAI's `UsageLimits`
- `Memory`, `Reasoning` -> Keep as QMCP extensions

### Human-in-the-Loop API
Full REST API for human approvals with:
- Persistent requests/responses
- Timeout handling
- Option validation
- Polling support

PydanticAI has "elicitation" but it's callback-based, not persistent.

**Recommendation**: Keep QMCP HITL system.

---

## 7. Migration Priority

### Phase 1: Adopt PydanticAI Agent Runtime
1. Add `pydantic-ai` dependency
2. Create `AgentConfig` -> PydanticAI `Agent` adapter
3. Implement agent execution using PydanticAI
4. Remove skeleton runner classes

### Phase 2: Integrate MCP Client
1. Create `QMCPToolset(AbstractToolset)` wrapper
2. Migrate example flows to use PydanticAI agents
3. Deprecate `qmcp.client.MCPClient`

### Phase 3: Tool System Alignment
1. Use PydanticAI tools for agent definitions
2. Keep `tool_registry` for server-side exposure
3. Create bridge for dual-registration when needed

### Phase 4: Documentation & Examples
1. Update cookbook recipes to use PydanticAI
2. Document hybrid architecture
3. Create migration guide for users

---

## 8. Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Application Layer                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐     ┌─────────────────────┐           │
│  │   PydanticAI Agent  │     │    QMCP Topologies  │           │
│  │   - Tools           │────▶│    - Debate         │           │
│  │   - Dependencies    │     │    - Pipeline       │           │
│  │   - Output Types    │     │    - Ensemble       │           │
│  └─────────────────────┘     └─────────────────────┘           │
│           │                           │                         │
│           ▼                           ▼                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   QMCP Model Registry                    │   │
│  │   Models.CLAUDE_SONNET_4, Models.GPT_4O, etc.           │   │
│  │   - Pricing, Limits, Capabilities                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              QMCP MCP Server (FastAPI)                   │   │
│  │   /v1/tools, /v1/invocations, /v1/human/*               │   │
│  │   - Audit Trail                                          │   │
│  │   - HITL Persistence                                     │   │
│  │   - Prometheus Metrics                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           PydanticAI MCP Clients (Toolsets)              │   │
│  │   MCPServerHTTP -> QMCP Server                           │   │
│  │   MCPServerStdio -> Other MCP Servers                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Dependencies to Add

```toml
# pyproject.toml
[project]
dependencies = [
    "pydantic-ai>=0.1.0",  # Core agent framework
    # Existing QMCP deps...
]

[project.optional-dependencies]
openai = ["pydantic-ai[openai]"]
anthropic = ["pydantic-ai[anthropic]"]
google = ["pydantic-ai[google]"]
all-models = ["pydantic-ai[openai,anthropic,google,groq]"]
```

---

## 10. Code Examples After Migration

### Before (Current QMCP)
```python
# No working agent - just config
from qmcp.agentframework.models import AgentConfig, Models

config = AgentConfig(
    model_config_obj=Models.CLAUDE_SONNET_4,
    system_prompt="You are helpful.",
)
# Cannot actually run - no implementation
```

### After (With PydanticAI)
```python
from pydantic_ai import Agent
from qmcp.agentframework.models import Models
from qmcp.integrations.pydantic_ai import model_to_pydantic_ai

# Use QMCP's rich model metadata
model_config = Models.CLAUDE_SONNET_4
print(f"Cost: ${model_config.pricing.input_cost}/1M input tokens")
print(f"Context: {model_config.limits.context_window} tokens")

# Create working agent with PydanticAI
agent = Agent(
    model_to_pydantic_ai(model_config),  # "anthropic:claude-sonnet-4-20250514"
    system_prompt="You are helpful.",
    output_type=MyResponse,
)

# Actually run the agent
result = await agent.run("Hello!")
print(result.output)
```

---

## Summary

| Keep from QMCP | Replace with PydanticAI |
|----------------|-------------------------|
| MCP Server (FastAPI) | Agent runtime |
| Model Registry (metadata) | MCP client |
| Topologies | Tool definitions |
| Mixins (Memory, Reasoning) | Structured output |
| HITL REST API | Usage limits |
| Audit/Metrics | Message history |

The hybrid approach leverages PydanticAI's mature agent framework while preserving
QMCP's unique production features and multi-agent orchestration capabilities.
