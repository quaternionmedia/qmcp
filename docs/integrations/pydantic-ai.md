# PydanticAI Integration

QMCP integrates with [PydanticAI](https://ai.pydantic.dev/) for LLM agent execution while preserving QMCP's unique capabilities: audit trails, human-in-the-loop, and multi-agent topologies.

## Overview

| QMCP Provides | PydanticAI Provides |
|---------------|---------------------|
| Model registry with pricing/limits | Agent runtime with dependency injection |
| MCP server with audit trail | Typed tool definitions |
| HITL REST API with persistence | Streaming and retries |
| Multi-agent topologies | Output validation |
| Prometheus metrics | Message history management |

## Installation

```bash
# Basic PydanticAI
pip install qmcp[pydantic-ai]

# With Anthropic models (Claude)
pip install qmcp[anthropic]

# With OpenAI models (GPT)
pip install qmcp[openai]

# All major providers
pip install qmcp[all-providers]
```

## Quick Start

### Create an Agent

```python
from qmcp.agentframework.models import Models
from qmcp.integrations.pydantic_ai import create_agent

# Create agent with QMCP's pre-configured model
agent = create_agent(
    Models.CLAUDE_SONNET_4,
    system_prompt="You are a helpful assistant.",
)

# Run the agent
result = await agent.run("Hello!")
print(result.output)
```

### Use QMCP Server as a Toolset

```python
from qmcp.integrations.pydantic_ai import create_agent, QMCPToolset

async with QMCPToolset("http://localhost:3333") as toolset:
    agent = create_agent(
        Models.CLAUDE_SONNET_4,
        system_prompt="You can use tools to help users.",
        toolsets=[toolset],
    )

    result = await agent.run("Use the echo tool to say hello")
```

### Fluent Builder API

```python
from qmcp.integrations.pydantic_ai import AgentBuilder, QMCPToolset
from qmcp.agentframework.models import Models
from pydantic import BaseModel

class TaskResult(BaseModel):
    summary: str
    success: bool

agent = (
    AgentBuilder(Models.CLAUDE_SONNET_4)
    .with_system_prompt("You are a task executor.")
    .with_output_type(TaskResult)
    .with_toolset(QMCPToolset("http://localhost:3333"))
    .with_retries(3)
    .build()
)
```

## Model Conversion

QMCP's `Models` registry provides rich metadata that PydanticAI doesn't track:

```python
from qmcp.agentframework.models import Models
from qmcp.integrations.pydantic_ai import (
    model_to_pydantic_ai,
    get_model_settings,
    estimate_cost,
)

# Access model metadata
model = Models.CLAUDE_SONNET_4
print(f"Context window: {model.limits.context_window}")  # 200000
print(f"Input cost: ${model.pricing.input_cost}/1M tokens")  # $3.00
print(f"Supports tools: {model.capabilities.supports_tool_choice}")  # True

# Convert to PydanticAI string
pydantic_ai_model = model_to_pydantic_ai(model)
# "anthropic:claude-sonnet-4-20250514"

# Get model settings for PydanticAI
settings = get_model_settings(model)
# {"temperature": 0.7, "max_tokens": 4096}

# Estimate cost before running
estimated_cost = estimate_cost(model, input_tokens=1000, output_tokens=500)
print(f"Estimated cost: ${estimated_cost:.4f}")
```

## QMCPToolset

The `QMCPToolset` connects PydanticAI agents to your QMCP server, providing:

- **Audit trail**: All tool calls are logged with correlation IDs
- **HITL support**: Request human approval within tool execution
- **Metrics**: Prometheus metrics for all invocations

### Basic Usage

```python
from qmcp.integrations.pydantic_ai import QMCPToolset

async with QMCPToolset("http://localhost:3333") as toolset:
    # Get available tools
    tools = await toolset.get_tools()
    for tool in tools:
        print(f"{tool.name}: {tool.description}")

    # Call a tool directly
    result = await toolset.call_tool("echo", {"message": "hello"})
```

### With Tool Prefix

Avoid naming conflicts when using multiple toolsets:

```python
async with QMCPToolset(
    "http://localhost:3333",
    tool_prefix="qmcp_",
) as toolset:
    # Tools are now named "qmcp_echo", "qmcp_planner", etc.
    agent = create_agent(
        Models.CLAUDE_SONNET_4,
        toolsets=[toolset, other_toolset],
    )
```

### Human-in-the-Loop

Request human approval from within a tool:

```python
from pydantic_ai import Agent, RunContext

agent = create_agent(Models.CLAUDE_SONNET_4)

@agent.tool
async def deploy_to_production(ctx: RunContext, env: str) -> str:
    """Deploy to a production environment."""
    # Access the toolset from dependencies
    toolset: QMCPToolset = ctx.deps.toolset

    # Request human approval
    approval = await toolset.request_human_approval(
        request_id=f"deploy-{env}-{ctx.run_id}",
        prompt=f"Approve deployment to {env}?",
        options=["approve", "reject"],
        context={"environment": env, "requested_by": "agent"},
    )

    if approval["response"] == "approve":
        # Proceed with deployment
        return f"Deployed to {env}"
    return "Deployment rejected by human"
```

## Creating Agents from AgentConfig

Use QMCP's full `AgentConfig` for deeper integration:

```python
from qmcp.agentframework.models import (
    AgentConfig,
    AgentCapability,
    SecurityConfig,
    AuthScope,
    Models,
)
from qmcp.integrations.pydantic_ai import create_agent_from_config

config = AgentConfig(
    model_config_obj=Models.CLAUDE_SONNET_4,
    system_prompt="You are a secure assistant.",
    max_retries=3,
    capabilities=[
        AgentCapability(name="tool_use"),
        AgentCapability(name="reasoning"),
    ],
    security=SecurityConfig(
        scopes=[AuthScope.READ],
        sandbox_enabled=True,
    ),
)

agent = create_agent_from_config(config)
```

## Combining with QMCP Topologies

Use PydanticAI for individual agent execution within QMCP topologies:

```python
from qmcp.agentframework.models import Models
from qmcp.agentframework.topologies import DebateTopology
from qmcp.integrations.pydantic_ai import create_agent

# Create specialized agents
researcher = create_agent(
    Models.CLAUDE_SONNET_4,
    system_prompt="You research topics thoroughly.",
)

critic = create_agent(
    Models.CLAUDE_SONNET_4,
    system_prompt="You find flaws in arguments.",
)

synthesizer = create_agent(
    Models.CLAUDE_OPUS_4,
    system_prompt="You synthesize debates into conclusions.",
)

# Use in a debate topology (when implemented)
# topology.execute(agents=[researcher, critic, synthesizer])
```

## What QMCP Keeps vs Uses PydanticAI For

### QMCP Provides (Unique Value)

| Feature | Description |
|---------|-------------|
| **MCP Server** | FastAPI server with audit trail, metrics, HITL persistence |
| **Model Registry** | Rich metadata: pricing, limits, capabilities, fallbacks |
| **Topologies** | Debate, Pipeline, Ensemble, CrossCheck, ChainOfCommand |
| **Mixins** | Memory with decay, CoT reasoning, HITL integration |
| **HITL API** | Full REST API with polling, timeouts, DB persistence |

### PydanticAI Provides (Agent Runtime)

| Feature | Description |
|---------|-------------|
| **Agent Execution** | Full agent with deps injection, streaming, retries |
| **Typed Tools** | Auto-generated schemas from type hints |
| **Output Validation** | Pydantic model validation with auto-retry |
| **Message History** | Conversation management across runs |
| **Usage Limits** | Token/request limits to prevent runaway costs |

## API Reference

### `model_to_pydantic_ai(model)`

Convert a QMCP `ModelConfig` or model ID to a PydanticAI model string.

```python
model_to_pydantic_ai(Models.CLAUDE_SONNET_4)
# "anthropic:claude-sonnet-4-20250514"

model_to_pydantic_ai("gpt-4o")
# "openai:gpt-4o"
```

### `create_agent(model, **kwargs)`

Create a PydanticAI `Agent` from a QMCP model.

**Parameters:**
- `model`: `ModelConfig` or model ID string
- `system_prompt`: Optional system prompt
- `deps_type`: Optional dependency type for injection
- `output_type`: Optional Pydantic model for output validation
- `tools`: Optional list of tool functions
- `toolsets`: Optional list of toolsets (including `QMCPToolset`)
- `retries`: Number of retries on validation failure (default: 1)

### `QMCPToolset(base_url, **kwargs)`

PydanticAI toolset connecting to a QMCP server.

**Parameters:**
- `base_url`: QMCP server URL (default: "http://localhost:3333")
- `tool_prefix`: Optional prefix for tool names
- `timeout`: HTTP timeout in seconds (default: 30)
- `correlation_id`: Optional correlation ID for tracing

**Methods:**
- `get_tools()`: Return available tool definitions
- `call_tool(name, args)`: Call a tool on the server
- `request_human_approval(...)`: Create HITL request and wait for response
- `health()`: Check server health

### `estimate_cost(model, input_tokens, output_tokens)`

Estimate cost using QMCP's pricing metadata.

```python
cost = estimate_cost(Models.CLAUDE_SONNET_4, 10000, 2000)
print(f"Estimated: ${cost:.4f}")  # ~$0.06
```

## Migration from Standalone PydanticAI

If you're already using PydanticAI directly:

```python
# Before: PydanticAI only
from pydantic_ai import Agent

agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    system_prompt="You are helpful.",
)

# After: With QMCP integration
from qmcp.agentframework.models import Models
from qmcp.integrations.pydantic_ai import create_agent, QMCPToolset

# Get model metadata (pricing, limits)
print(f"Cost: ${Models.CLAUDE_SONNET_4.pricing.input_cost}/1M tokens")

# Create agent with QMCP model
agent = create_agent(
    Models.CLAUDE_SONNET_4,
    system_prompt="You are helpful.",
)

# Add QMCP tools with audit trail
async with QMCPToolset() as toolset:
    agent = create_agent(
        Models.CLAUDE_SONNET_4,
        toolsets=[toolset],
    )
```

## Next Steps

- [Architecture](../architecture.md) - Understand the system boundaries
- [Agent Framework](../agentframework/overview.md) - Define agent types and topologies
- [Human-in-the-Loop](../human_in_loop.md) - HITL patterns and best practices
- [Tools](../tools.md) - Register custom tools on the MCP server
