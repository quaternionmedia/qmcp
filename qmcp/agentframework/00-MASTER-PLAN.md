# QMCP Agent Framework: Master Plan

Note: Design reference. Implementation status is documented in docs/agentframework.md.


## Executive Summary

This document outlines a comprehensive framework for organizing LLM agent types within the QMCP (Quaternion MCP) ecosystem. The framework provides hierarchical, mixin-capable agent definitions that integrate seamlessly with SQLModel persistence, Pydantic validation, FastAPI endpoints, CLI tooling, and Metaflow workflow orchestration.

## Design Philosophy

### Core Principles

1. **Composition over Inheritance**: Agent capabilities are composed through mixins rather than deep inheritance hierarchies
2. **Topology-Driven Orchestration**: Agent collaboration patterns (debate, chain-of-command, delegation) are first-class citizens
3. **Schema-First Design**: All agent types and topologies are defined as SQLModel/Pydantic schemas for persistence and validation
4. **Metaflow Integration**: Agent orchestrations map naturally to Metaflow DAGs
5. **Extensibility**: New agent types, capabilities, and topologies can be added without modifying core framework

### Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                     CLI / API Layer                             │
│  (qmcp agents list | qmcp agents create | /v1/agents/*)        │
├─────────────────────────────────────────────────────────────────┤
│                    Metaflow Runner Layer                        │
│  (AgentFlow, TopologyFlow, orchestration DAGs)                 │
├─────────────────────────────────────────────────────────────────┤
│                    Topology Layer                               │
│  (Debate, ChainOfCommand, Delegation, CrossCheck, Ensemble)    │
├─────────────────────────────────────────────────────────────────┤
│                    Agent Layer                                  │
│  (BaseAgent + Capability Mixins + Role Definitions)            │
├─────────────────────────────────────────────────────────────────┤
│                    Model Layer (SQLModel/Pydantic)             │
│  (AgentType, AgentInstance, Topology, Execution, Result)       │
├─────────────────────────────────────────────────────────────────┤
│                    Persistence Layer                            │
│  (SQLite via aiosqlite, existing QMCP database)                │
└─────────────────────────────────────────────────────────────────┘
```

## Agent Type Hierarchy

### Base Agent Model

```python
class AgentType(SQLModel, table=True):
    """Core agent type definition"""
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    description: str
    capabilities: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    role: AgentRole
    config: dict = Field(default_factory=dict, sa_column=Column(JSON))
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

### Agent Roles (Enumeration)

| Role | Description | Primary Function |
|------|-------------|------------------|
| `PLANNER` | Strategic planning and task decomposition | Break complex tasks into subtasks |
| `EXECUTOR` | Task execution and action taking | Perform concrete actions |
| `REVIEWER` | Quality assurance and validation | Verify outputs meet criteria |
| `CRITIC` | Adversarial analysis and devil's advocate | Find flaws and edge cases |
| `SYNTHESIZER` | Information aggregation and summarization | Combine multiple outputs |
| `SPECIALIST` | Domain-specific expertise | Provide specialized knowledge |
| `COORDINATOR` | Orchestration and delegation | Manage multi-agent workflows |
| `OBSERVER` | Monitoring and logging | Track execution and metrics |

### Capability Mixins

Mixins provide composable capabilities that can be added to any agent:

| Mixin | Provides | Use Case |
|-------|----------|----------|
| `ToolUseMixin` | Tool invocation capabilities | Agents that use external tools |
| `MemoryMixin` | Persistent memory across invocations | Long-running or stateful agents |
| `ReasoningMixin` | Chain-of-thought reasoning | Complex analytical tasks |
| `CodeExecutionMixin` | Code generation and execution | Programming tasks |
| `WebSearchMixin` | Web search and retrieval | Research and fact-checking |
| `HumanInLoopMixin` | Human approval/input requests | Critical decision points |
| `StreamingMixin` | Streaming response generation | Real-time output |
| `StructuredOutputMixin` | JSON/schema-constrained output | API responses |

## Collaboration Topologies

### 1. Debate Topology

**Purpose**: Reach consensus through structured argumentation

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Agent A │────▶│ Mediator│◀────│ Agent B │
│(Pro)    │     │         │     │(Con)    │
└─────────┘     └────┬────┘     └─────────┘
                     │
                     ▼
               ┌───────────┐
               │ Synthesis │
               └───────────┘
```

**Configuration**:
- `rounds`: Number of debate rounds
- `mediator_policy`: How mediator resolves conflicts
- `convergence_threshold`: When to stop early

### 2. Chain of Command Topology

**Purpose**: Hierarchical task delegation with authority levels

```
               ┌────────────┐
               │ Commander  │
               └─────┬──────┘
          ┌──────────┼──────────┐
          ▼          ▼          ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │Lieutenant│ │Lieutenant│ │Lieutenant│
    └────┬─────┘ └────┬─────┘ └────┬─────┘
    ┌────┴────┐  ┌────┴────┐  ┌────┴────┐
    ▼         ▼  ▼         ▼  ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│Worker │ │Worker │ │Worker │ │Worker │ │Worker │
└───────┘ └───────┘ └───────┘ └───────┘ └───────┘
```

**Configuration**:
- `authority_levels`: Define command hierarchy
- `escalation_policy`: When to escalate decisions
- `delegation_strategy`: How tasks are assigned

### 3. Delegation Topology

**Purpose**: Dynamic task assignment based on capabilities

```
┌────────────────────────────────────────────┐
│              Coordinator                    │
│  ┌──────────────────────────────────────┐  │
│  │        Capability Registry           │  │
│  └──────────────────────────────────────┘  │
└─────────────────────┬──────────────────────┘
                      │ Dispatch
         ┌────────────┼────────────┐
         ▼            ▼            ▼
   ┌───────────┐ ┌───────────┐ ┌───────────┐
   │Specialist │ │Specialist │ │Specialist │
   │  (Code)   │ │ (Research)│ │ (Writing) │
   └───────────┘ └───────────┘ └───────────┘
```

**Configuration**:
- `routing_strategy`: How to match tasks to specialists
- `load_balancing`: Distribute work across specialists
- `fallback_policy`: Handle unmatched tasks

### 4. Cross-Check Topology

**Purpose**: Verify outputs through independent validation

```
         ┌─────────┐
         │ Primary │
         │ Agent   │
         └────┬────┘
              │ Output
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐
│Checker │ │Checker │ │Checker │
│   1    │ │   2    │ │   3    │
└────┬───┘ └────┬───┘ └────┬───┘
     │          │          │
     └──────────┼──────────┘
                ▼
         ┌───────────┐
         │ Consensus │
         │  Engine   │
         └───────────┘
```

**Configuration**:
- `num_checkers`: Number of independent validators
- `consensus_threshold`: Agreement required
- `conflict_resolution`: Handle disagreements

### 5. Ensemble Topology

**Purpose**: Aggregate multiple agent outputs for robustness

```
                    ┌─────────┐
        ┌──────────▶│ Agent 1 │──────────┐
        │           └─────────┘          │
        │           ┌─────────┐          │
┌───────┴───┐ ────▶ │ Agent 2 │ ────▶ ┌──┴───────┐
│   Input   │       └─────────┘       │Aggregator│
└───────┬───┘ ────▶ ┌─────────┐ ────▶ └──┬───────┘
        │           │ Agent 3 │          │
        └──────────▶└─────────┘──────────┘
                          ▼
                    ┌───────────┐
                    │  Output   │
                    └───────────┘
```

**Configuration**:
- `aggregation_method`: voting, averaging, weighted
- `diversity_enforcement`: Ensure varied approaches
- `failure_handling`: Handle partial failures

### 6. Pipeline Topology

**Purpose**: Sequential processing through specialized stages

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Stage 1 │───▶│ Stage 2 │───▶│ Stage 3 │───▶│ Stage 4 │
│(Parse)  │    │(Analyze)│    │(Generate)│   │(Review) │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
```

**Configuration**:
- `stages`: Ordered list of processing stages
- `checkpoints`: Where to save intermediate state
- `retry_policy`: Handle stage failures

## Compound Topologies

Topologies can be composed to create complex orchestrations:

### Reviewed Debate
```python
topology = Pipeline([
    Debate(agents=[agent_a, agent_b], mediator=mediator),
    CrossCheck(checkers=[reviewer_1, reviewer_2]),
    Synthesizer(output_agent)
])
```

### Delegated Ensemble
```python
topology = Delegation(
    coordinator=coordinator,
    specialists={
        "research": Ensemble([research_1, research_2, research_3]),
        "coding": Pipeline([coder, reviewer, tester]),
        "writing": ChainOfCommand([editor, writer])
    }
)
```

## Metaflow Integration

### Agent Flow Template

```python
from metaflow import FlowSpec, step
from qmcp.agentframework import AgentType, Topology

class AgentOrchestratorFlow(FlowSpec):
    
    @step
    def start(self):
        """Initialize agents and topology"""
        self.topology = Topology.load("debate_with_review")
        self.agents = self.topology.instantiate_agents()
        self.next(self.execute)
    
    @step
    def execute(self):
        """Run the agent topology"""
        self.result = self.topology.run(
            input=self.input_data,
            agents=self.agents
        )
        self.next(self.end)
    
    @step
    def end(self):
        """Finalize and persist results"""
        self.topology.persist_result(self.result)
```

### Topology-to-DAG Mapping

Each topology maps to Metaflow constructs:

| Topology | Metaflow Pattern |
|----------|------------------|
| Pipeline | Linear `@step` chain |
| Ensemble | `foreach` parallel fanout |
| Debate | `foreach` + `join` with rounds |
| ChainOfCommand | Nested `foreach` hierarchy |
| CrossCheck | `foreach` validators + `join` |
| Delegation | Dynamic `foreach` based on task type |

## Database Schema

### Tables

1. **agent_types** - Agent type definitions
2. **agent_instances** - Running agent instances
3. **topologies** - Topology configurations
4. **topology_memberships** - Agent-to-topology mappings
5. **executions** - Execution records
6. **messages** - Inter-agent communications
7. **results** - Final outputs

### Relationships

```
agent_types 1──────< agent_instances
topologies 1──────< topology_memberships >──────1 agent_types
executions 1──────< messages
executions 1──────< results
topologies 1──────< executions
```

## CLI Commands

```bash
# Agent Type Management
qmcp agents list                      # List all agent types
qmcp agents create --name planner     # Create new agent type
qmcp agents show <name>               # Show agent details
qmcp agents delete <name>             # Delete agent type

# Capability Management  
qmcp capabilities list                # List available mixins
qmcp capabilities add <agent> <cap>   # Add capability to agent

# Topology Management
qmcp topologies list                  # List topologies
qmcp topologies create --type debate  # Create topology
qmcp topologies show <name>           # Show topology config
qmcp topologies validate <name>       # Validate topology

# Execution
qmcp run <topology> --input "..."     # Run topology
qmcp run <topology> --flow            # Generate Metaflow DAG
```

## API Endpoints

```
GET    /v1/agents                     # List agent types
POST   /v1/agents                     # Create agent type
GET    /v1/agents/{name}              # Get agent type
PUT    /v1/agents/{name}              # Update agent type
DELETE /v1/agents/{name}              # Delete agent type

GET    /v1/topologies                 # List topologies
POST   /v1/topologies                 # Create topology
GET    /v1/topologies/{name}          # Get topology
PUT    /v1/topologies/{name}          # Update topology
DELETE /v1/topologies/{name}          # Delete topology

POST   /v1/topologies/{name}/run      # Execute topology
GET    /v1/topologies/{name}/dag      # Get Metaflow DAG

GET    /v1/executions                 # List executions
GET    /v1/executions/{id}            # Get execution details
GET    /v1/executions/{id}/messages   # Get execution messages
```

## Implementation Phases

### Phase 1: Core Models (This Document)
- [ ] SQLModel definitions for agents and topologies
- [ ] Pydantic validation schemas
- [ ] Base mixin infrastructure

### Phase 2: Topology Engine
- [ ] Topology execution runtime
- [ ] Inter-agent messaging
- [ ] Result aggregation

### Phase 3: Metaflow Integration  
- [ ] DAG generation from topologies
- [ ] Step decorators for agent invocation
- [ ] State management across steps

### Phase 4: CLI & API
- [ ] CLI commands for management
- [ ] FastAPI router integration
- [ ] OpenAPI documentation

### Phase 5: Advanced Features
- [ ] Dynamic topology adaptation
- [ ] Learning from execution history
- [ ] Cost optimization

## File Structure

```
qmcp/
├── agents/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── agent.py           # AgentType, AgentInstance models
│   │   ├── topology.py        # Topology, TopologyMembership models
│   │   ├── execution.py       # Execution, Message, Result models
│   │   └── enums.py           # AgentRole, TopologyType enums
│   ├── mixins/
│   │   ├── __init__.py
│   │   ├── base.py            # BaseMixin, MixinRegistry
│   │   ├── tool_use.py        # ToolUseMixin
│   │   ├── memory.py          # MemoryMixin
│   │   ├── reasoning.py       # ReasoningMixin
│   │   └── ...
│   ├── topologies/
│   │   ├── __init__.py
│   │   ├── base.py            # BaseTopology
│   │   ├── debate.py          # DebateTopology
│   │   ├── chain.py           # ChainOfCommandTopology
│   │   ├── delegation.py      # DelegationTopology
│   │   ├── crosscheck.py      # CrossCheckTopology
│   │   ├── ensemble.py        # EnsembleTopology
│   │   ├── pipeline.py        # PipelineTopology
│   │   └── compound.py        # CompoundTopology builder
│   ├── runners/
│   │   ├── __init__.py
│   │   ├── base.py            # BaseRunner
│   │   ├── local.py           # LocalRunner (in-process)
│   │   └── metaflow.py        # MetaflowRunner (DAG generation)
│   ├── router.py              # FastAPI router
│   └── cli.py                 # CLI commands
├── tests/
│   └── agents/
│       ├── test_models.py
│       ├── test_mixins.py
│       ├── test_topologies.py
│       └── test_runners.py
└── examples/
    └── flows/
        ├── debate_flow.py
        ├── delegation_flow.py
        └── ensemble_flow.py
```

## Success Criteria

1. **Type Safety**: All agent configurations validated at creation time
2. **Composability**: Any topology can be nested within another
3. **Testability**: Each component independently testable
4. **Observability**: Full execution trace available for debugging
5. **Performance**: Minimal overhead for agent orchestration
6. **Extensibility**: New agent types added without framework changes

## Next Steps

1. Review and approve this plan
2. Implement core models (see `01-MODELS.md`)
3. Implement mixins (see `02-MIXINS.md`)
4. Implement topologies (see `03-TOPOLOGIES.md`)
5. Implement runners (see `04-RUNNERS.md`)
6. Write comprehensive tests (see `05-TESTS.md`)
7. Integrate with QMCP main codebase
