# Agent Framework

The Agent Framework provides a hierarchical, mixin-capable system for defining LLM agent types with support for multi-agent collaboration topologies.

## Overview

The framework operates in the client plane, storing data in the QMCP database without adding orchestration logic to the server. It provides:

- **Agent Types** - Reusable agent templates with roles, capabilities, and configurations
- **Capability Mixins** - Composable behaviors (tool use, memory, reasoning)
- **Collaboration Topologies** - Patterns for multi-agent coordination
- **Runners** - Execute topologies locally or via Metaflow workflows

## Quick Reference

| Component | Location | Purpose |
|-----------|----------|---------|
| Models | `qmcp/agentframework/models/` | Data models, enums, schemas |
| Mixins | `qmcp/agentframework/mixins.py` | Capability mixins |
| Topologies | `qmcp/agentframework/topologies.py` | Collaboration patterns |
| Runners | `qmcp/agentframework/runners.py` | Execution environments |

## Example: Define and Persist an Agent Role Config

Here's a complete example showing how to define an agent with a specific role, configure its capabilities, and persist it to the database.

### Using Models Registry (Recommended)

```python
from qmcp.agentframework.models import (
    AgentType, AgentRole, AgentConfig, AgentCapability, Models,
)

# Use a pre-defined model from the registry - no string literals!
agent = AgentType(
    name="research_analyst",
    description="Researches and analyzes topics",
    role=AgentRole.SPECIALIST,
    config=AgentConfig(
        model_config_obj=Models.CLAUDE_SONNET_4,  # Pre-configured model
        temperature=0.7,
        system_prompt="You are a research analyst...",
        capabilities=[
            AgentCapability(name="tool_use"),
            AgentCapability(name="memory"),
        ],
    ).model_dump(),
)
```

### Available Models

```python
from qmcp.agentframework.models import Models, ModelProvider, ModelTier

# Anthropic Claude
Models.CLAUDE_OPUS_4      # Flagship - complex reasoning
Models.CLAUDE_SONNET_4    # Standard - balanced (default)
Models.CLAUDE_HAIKU_35    # Fast - high volume

# OpenAI GPT
Models.GPT_4O             # Flagship
Models.GPT_4O_MINI        # Fast

# Google Gemini
Models.GEMINI_PRO         # Flagship
Models.GEMINI_FLASH       # Fast

# Local (Ollama)
Models.LLAMA_3_70B
Models.LLAMA_3_8B
Models.MISTRAL_7B

# Query by provider or tier
anthropic_models = Models.by_provider(ModelProvider.ANTHROPIC)
fast_models = Models.by_tier(ModelTier.FAST)
```

### Full Configuration with Persistence

```python
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from qmcp.agentframework.models import (
    AgentType,
    AgentRole,
    AgentConfig,
    AgentCapability,
    SkillConfig,
    SkillCategory,
    SecurityConfig,
    AuthScope,
    Models,
)


async def create_and_persist_agent():
    # Setup database
    engine = create_async_engine("sqlite+aiosqlite:///./agents.db")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Use pre-configured model from registry
    config = AgentConfig(
        model_config_obj=Models.CLAUDE_SONNET_4,  # Pre-configured with limits, pricing, fallback
        system_prompt="""You are a senior code reviewer with expertise in Python and security.
Your role is to:
1. Identify potential bugs and security vulnerabilities
2. Suggest improvements for code quality and performance
3. Ensure adherence to best practices and style guidelines
4. Provide constructive, actionable feedback""",
        capabilities=[
            AgentCapability(
                name="tool_use",
                config={"allowed_tools": ["read_file", "search_code", "run_tests"]},
            ),
            AgentCapability(
                name="reasoning",
                config={"reasoning_style": "step_by_step"},
            ),
            AgentCapability(
                name="structured_output",
                config={
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "issues": {"type": "array"},
                            "suggestions": {"type": "array"},
                            "approved": {"type": "boolean"},
                        },
                        "required": ["issues", "approved"],
                    }
                },
            ),
        ],
        skills=[
            SkillConfig(name="python", category=SkillCategory.CODING, proficiency=0.95),
            SkillConfig(name="security", category=SkillCategory.DOMAIN_EXPERT, proficiency=0.9),
            SkillConfig(name="code_review", category=SkillCategory.ANALYSIS, proficiency=0.92),
        ],
        security=SecurityConfig(
            scopes=[AuthScope.READ, AuthScope.EXECUTE],
            sandbox_enabled=True,
            allowed_tools=["read_file", "search_code", "run_tests"],
        ),
    )

    # Create agent type
    agent = AgentType(
        name="senior_code_reviewer",
        description="Expert code reviewer specializing in Python security and best practices",
        role=AgentRole.REVIEWER,
        config=config.model_dump(),
    )

    # Persist to database
    async with session_factory() as session:
        session.add(agent)
        await session.commit()
        await session.refresh(agent)

        print(f"Created agent: {agent.name}")
        print(f"  ID: {agent.id}")
        print(f"  Role: {agent.role.value}")
        print(f"  Version: {agent.version}")
        print(f"  Capabilities: {[c.name for c in config.capabilities]}")

        # Retrieve and verify
        from sqlmodel import select
        stmt = select(AgentType).where(AgentType.name == "senior_code_reviewer")
        result = await session.execute(stmt)
        retrieved = result.scalar_one()

        print(f"\nRetrieved from DB:")
        print(f"  Name: {retrieved.name}")
        print(f"  Has tool_use: {retrieved.has_capability('tool_use')}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(create_and_persist_agent())
```

## Quickstart

### Install Dependencies

```bash
uv add sqlmodel aiosqlite
```

### Run Tests

```bash
uv run pytest tests/test_agentframework_models.py -v
uv run pytest tests/test_agentframework_mixins.py -v
```

### Create a Simple Agent

```python
from qmcp.agentframework import AgentType, AgentRole, AgentConfig

# Option 1: Simple model ID string
agent = AgentType(
    name="my_agent",
    description="A helpful assistant",
    role=AgentRole.EXECUTOR,
    config=AgentConfig(model="claude-sonnet-4-20250514").model_dump(),
)

# Option 2: Full model configuration (for production)
from qmcp.agentframework import ModelConfig, ModelProvider, ModelFamily, ModelTier

model = ModelConfig(
    model_id="claude-sonnet-4-20250514",
    provider=ModelProvider.ANTHROPIC,
    family=ModelFamily.CLAUDE,
    tier=ModelTier.STANDARD,
)

agent = AgentType(
    name="my_agent",
    description="A helpful assistant",
    role=AgentRole.EXECUTOR,
    config=AgentConfig(model_config_obj=model).model_dump(),
)
```

### Create a Topology

```python
from qmcp.agentframework import Topology, TopologyType, DebateConfig

topology = Topology(
    name="tech_debate",
    description="Debate on technology topics",
    topology_type=TopologyType.DEBATE,
    config=DebateConfig(max_rounds=3).model_dump(),
)
```

## What's Implemented

| Component | Status |
|-----------|--------|
| SQLModel tables and schemas | Complete |
| Capability mixins | Complete |
| Topology registry (skeleton) | Complete |
| Runner registry (skeleton) | Complete |
| PydanticAI integration | Complete |
| Unit tests | Complete |

## What's Planned

| Feature | Status |
|---------|--------|
| Full topology execution engine | Design only |
| Metaflow DAG generation | Design only |
| FastAPI router integration | Planned |
| CLI commands | Design only |

## PydanticAI Integration

The agent framework integrates with [PydanticAI](https://ai.pydantic.dev/) for agent execution. QMCP provides:
- Model registry with pricing, limits, and capabilities
- Audit trails for all tool calls
- Human-in-the-loop REST API
- Multi-agent topologies

PydanticAI provides:
- Agent runtime with dependency injection
- Typed tools with auto-generated schemas
- Output validation
- Streaming and retries

### Quick Example

```python
from qmcp.agentframework.models import Models
from qmcp.integrations.pydantic_ai import create_agent, QMCPToolset

# Create agent with QMCP model (includes pricing metadata)
agent = create_agent(
    Models.CLAUDE_SONNET_4,
    system_prompt="You are a helpful assistant.",
)

# Add QMCP server tools with full audit trail
async with QMCPToolset("http://localhost:3333") as toolset:
    agent = create_agent(
        Models.CLAUDE_SONNET_4,
        toolsets=[toolset],
    )
    result = await agent.run("Hello!")
```

See [PydanticAI Integration](../integrations/pydantic-ai.md) for full documentation.

## Documentation

### Design Documents

| Document | Description |
|----------|-------------|
| [Models](models.md) | Data model specification |
| [Mixins](mixins.md) | Mixin system design |
| [Topologies](topologies.md) | Topology patterns |
| [Runners](runners.md) | Runner implementations |

### Related Documentation

| Document | Description |
|----------|-------------|
| [Architecture](../architecture.md) | System boundaries and integration |
| [Deployment](../deployment.md) | Database and production configuration |
| [Module README](../../qmcp/agentframework/README.md) | In-module overview |

## Agent Roles

| Role | Use Case |
|------|----------|
| `PLANNER` | Task decomposition, strategy |
| `EXECUTOR` | Action taking, implementation |
| `REVIEWER` | Quality assurance, validation |
| `CRITIC` | Adversarial analysis, finding flaws |
| `SYNTHESIZER` | Aggregation, summarization |
| `SPECIALIST` | Domain expertise |
| `COORDINATOR` | Orchestration, delegation |
| `OBSERVER` | Monitoring, logging |
| `VALIDATOR` | Input/output validation |
| `TRANSFORMER` | Data transformation |
| `AGGREGATOR` | Result combination |
| `ROUTER` | Task routing |

## Topology Types

| Type | Pattern | Use Case |
|------|---------|----------|
| `DEBATE` | Argumentation | Reaching consensus |
| `ENSEMBLE` | Parallel | Robust answers |
| `PIPELINE` | Sequential | Multi-stage processing |
| `CHAIN_OF_COMMAND` | Hierarchical | Complex decomposition |
| `CROSS_CHECK` | Validation | Quality verification |
| `DELEGATION` | Routing | Dynamic assignment |
| `COMPOUND` | Nested | Complex workflows |
| `MESH` | Interconnected | Peer collaboration |
| `STAR` | Hub-spoke | Centralized coordination |
| `RING` | Circular | Iterative refinement |

## Next Steps

1. Explore the [Models documentation](models.md) for all available types
2. Review [Mixins](mixins.md) for adding capabilities
3. See [Topologies](topologies.md) for multi-agent patterns
4. Check [Runners](runners.md) for execution options
