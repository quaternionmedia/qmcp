# QMCP Agent Framework

A hierarchical, mixin-capable framework for organizing LLM agent types with support for various collaboration topologies.

## Overview

The agent framework provides:

- **Agent Types**: Define reusable agent templates with roles, capabilities, and configurations
- **Capability Mixins**: Composable behaviors (tool use, memory, reasoning, structured output)
- **Collaboration Topologies**: Patterns for multi-agent coordination (debate, ensemble, pipeline, etc.)
- **Runners**: Execute topologies locally or via Metaflow workflows
- **Persistence**: SQLModel-based storage for agents, topologies, and execution history

## Quickstart

### Define an Agent Type

```python
from qmcp.agentframework import (
    AgentType,
    AgentRole,
    AgentConfig,
    AgentCapability,
    Models,  # Pre-defined model configurations
)

# Use a pre-defined model from the registry - no string literals!
agent = AgentType(
    name="research_analyst",
    description="Researches topics and provides analysis",
    role=AgentRole.SPECIALIST,
    config=AgentConfig(
        model_config_obj=Models.CLAUDE_SONNET_4,
        temperature=0.7,
        system_prompt="You are a research analyst...",
        capabilities=[
            AgentCapability(name="tool_use", config={"allowed_tools": ["web_search"]}),
            AgentCapability(name="memory", config={"max_memories": 100}),
        ],
    ).model_dump(),
)
```

### Available Models

```python
from qmcp.agentframework import Models

# Anthropic Claude models
Models.CLAUDE_OPUS_4      # Flagship model for complex tasks
Models.CLAUDE_SONNET_4    # Balanced performance (default)
Models.CLAUDE_HAIKU_35    # Fast and efficient

# OpenAI GPT models
Models.GPT_4O             # Most capable GPT
Models.GPT_4O_MINI        # Fast and affordable

# Google Gemini models
Models.GEMINI_PRO         # Most capable Gemini
Models.GEMINI_FLASH       # Fast Gemini

# Local models (Ollama)
Models.LLAMA_3_70B        # Llama 3 70B
Models.LLAMA_3_8B         # Llama 3 8B
Models.MISTRAL_7B         # Mistral 7B

# Query models by provider or tier
Models.by_provider(ModelProvider.ANTHROPIC)
Models.by_tier(ModelTier.FLAGSHIP)
Models.get("claude-sonnet-4-20250514")  # Lookup by ID
```

### Persist to Database

```python
from sqlmodel.ext.asyncio.session import AsyncSession

async def save_agent(session: AsyncSession, agent: AgentType):
    session.add(agent)
    await session.commit()
    await session.refresh(agent)
    return agent
```

### Create a Topology

```python
from qmcp.agentframework import Topology, TopologyType, DebateConfig

topology = Topology(
    name="tech_debate",
    description="Structured debate on technology topics",
    topology_type=TopologyType.DEBATE,
    config=DebateConfig(
        max_rounds=3,
        consensus_method="mediator",
    ).model_dump(),
)
```

### Run Tests

```bash
uv run pytest tests/test_agentframework_models.py -v
uv run pytest tests/test_agentframework_mixins.py -v
```

## Module Structure

```
qmcp/agentframework/
├── __init__.py          # Public API exports
├── README.md            # This file
├── models/              # Data models (enums, configs, entities, schemas)
├── mixins.py            # Capability mixins (tool_use, memory, reasoning, etc.)
├── topologies.py        # Collaboration topologies (debate, ensemble, pipeline, etc.)
└── runners.py           # Execution runners (local, async, metaflow)
```

## Documentation

| Document | Description |
|----------|-------------|
| [Agent Framework Guide](../../docs/agentframework/overview.md) | User guide and API reference |
| [Architecture](../../docs/architecture.md) | System boundaries and integration |
| [Design: Models](../../docs/agentframework/models.md) | Data model specification |
| [Design: Mixins](../../docs/agentframework/mixins.md) | Mixin system design |
| [Design: Topologies](../../docs/agentframework/topologies.md) | Topology patterns |
| [Design: Runners](../../docs/agentframework/runners.md) | Runner implementations |

## Agent Roles

| Role | Purpose |
|------|---------|
| `PLANNER` | Strategic planning and task decomposition |
| `EXECUTOR` | Task execution and action taking |
| `REVIEWER` | Quality assurance and validation |
| `CRITIC` | Adversarial analysis and finding flaws |
| `SYNTHESIZER` | Information aggregation and summarization |
| `SPECIALIST` | Domain-specific expertise |
| `COORDINATOR` | Orchestration and delegation |
| `OBSERVER` | Monitoring and logging |

## Topology Types

| Type | Pattern | Use Case |
|------|---------|----------|
| `DEBATE` | Structured argumentation | Reaching consensus through opposing views |
| `ENSEMBLE` | Parallel aggregation | Robust answers from multiple perspectives |
| `PIPELINE` | Sequential processing | Multi-stage transformations |
| `CHAIN_OF_COMMAND` | Hierarchical delegation | Complex task breakdown |
| `CROSS_CHECK` | Independent validation | Quality verification |
| `DELEGATION` | Capability-based routing | Dynamic task assignment |
| `COMPOUND` | Nested topologies | Complex workflows |

## What's Implemented

- SQLModel tables and Pydantic schemas for all models
- Mixin system with tool_use, memory, reasoning, human_in_loop, structured_output
- Topology and runner registries with skeleton implementations
- Comprehensive test suite

## What's Planned

- Full topology execution engine
- Metaflow DAG generation
- FastAPI router integration
- CLI commands for agent management

## Links

- [Main Documentation](../../docs/)
- [Contributing Guide](../../docs/contributing.md)
- [Tests](../../tests/test_agentframework_models.py)
