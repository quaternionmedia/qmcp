# Agent Framework: Topologies

Topologies define collaboration patterns for multi-agent systems. Each topology specifies how agents communicate, share information, and reach conclusions.

## Architecture

### Base Classes

```python
class ExecutionContext:
    """Context passed through topology execution."""
    execution_id: UUID
    topology_id: int
    input_data: dict[str, Any]
    shared_state: dict[str, Any]
    round_number: int = 0

class BaseTopology(ABC):
    """Abstract base for all topologies."""
    topology_type: ClassVar[TopologyType]

    async def run(self, context: ExecutionContext) -> dict[str, Any]:
        """Execute the topology."""
```

### Lifecycle

```
1. setup()      - Initialize agents and state
2. _run()       - Execute topology logic
3. teardown()   - Cleanup and finalize
```

## Topology Types

### Debate

Structured argumentation between opposing viewpoints.

**Required Slots:** `proponent`, `opponent`, `mediator`

**Configuration:**
```python
class DebateConfig(SQLModel):
    max_rounds: int = 3
    consensus_method: ConsensusMethod = "mediator"
    convergence_threshold: float = 0.8
    allow_early_termination: bool = True
    require_justification: bool = True
```

**Flow:**
```
┌─────────┐     ┌─────────┐     ┌─────────┐
│Proponent│────▶│Opponent │────▶│Proponent│  (rounds)
└─────────┘     └─────────┘     └─────────┘
                                     │
                                     ▼
                              ┌──────────┐
                              │ Mediator │
                              └──────────┘
                                     │
                                     ▼
                              ┌──────────┐
                              │ Synthesis │
                              └──────────┘
```

**Use Cases:**
- Exploring opposing viewpoints
- Risk analysis (optimist vs pessimist)
- Decision validation

### Ensemble

Parallel execution with aggregated results.

**Required Slots:** `ensemble_0..N`, `aggregator`

**Configuration:**
```python
class EnsembleConfig(SQLModel):
    aggregation_method: AggregationMethod = "synthesis"
    diversity_weight: float = 0.3
    failure_threshold: float = 0.5
    min_responses: int = 2
    weight_by_confidence: bool = True
```

**Aggregation Methods:**
- `vote` - Majority answer
- `average` - Weighted average (numeric)
- `concat` - Concatenate all responses
- `best_of` - Highest confidence
- `synthesis` - LLM synthesis

**Flow:**
```
              ┌──────────┐
         ┌───▶│Ensemble_0│───┐
         │    └──────────┘   │
┌─────┐  │    ┌──────────┐   │    ┌──────────┐
│Input│──┼───▶│Ensemble_1│───┼───▶│Aggregator│
└─────┘  │    └──────────┘   │    └──────────┘
         │    ┌──────────┐   │
         └───▶│Ensemble_2│───┘
              └──────────┘
```

**Use Cases:**
- Robust answers from multiple perspectives
- Error reduction through redundancy
- Covering different expertise areas

### Pipeline

Sequential multi-stage processing.

**Required Slots:** One per stage in `stages` config

**Configuration:**
```python
class PipelineConfig(SQLModel):
    stages: list[str] = []
    checkpoint_after: list[str] = []
    retry_failed_stages: bool = True
    max_stage_retries: int = 2
    error_strategy: ErrorStrategy = "retry"
    parallel_stages: list[list[str]] = []
```

**Flow:**
```
┌───────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐
│ Input │───▶│ Parse   │───▶│ Analyze  │───▶│Generate │
└───────┘    └─────────┘    └──────────┘    └─────────┘
                  │              │
                  ▼              ▼
             checkpoint     checkpoint
```

**Use Cases:**
- Document processing (parse → analyze → summarize)
- Code generation (plan → implement → review)
- Data transformation chains

### Chain of Command

Hierarchical delegation with authority levels.

**Required Slots:** Based on `authority_levels` config

**Configuration:**
```python
class ChainOfCommandConfig(SQLModel):
    authority_levels: list[str] = ["commander", "lieutenant", "worker"]
    escalation_threshold: float = 0.5
    max_delegation_depth: int = 3
    require_acknowledgment: bool = True
```

**Flow:**
```
              ┌───────────┐
              │ Commander │
              └─────┬─────┘
                    │ delegates
         ┌──────────┴──────────┐
         ▼                     ▼
   ┌────────────┐        ┌────────────┐
   │Lieutenant_A│        │Lieutenant_B│
   └──────┬─────┘        └──────┬─────┘
          │                     │
    ┌─────┴─────┐         ┌─────┴─────┐
    ▼           ▼         ▼           ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Worker_1│ │Worker_2│ │Worker_3│ │Worker_4│
└────────┘ └────────┘ └────────┘ └────────┘
```

**Use Cases:**
- Complex task decomposition
- Large-scale project coordination
- Hierarchical decision making

### Cross-Check

Independent validation from multiple reviewers.

**Required Slots:** `primary`, `checker_0..N`

**Configuration:**
```python
class CrossCheckConfig(SQLModel):
    num_checkers: int = 3
    consensus_method: ConsensusMethod = "majority"
    require_unanimous_for_approval: bool = False
    checker_specializations: list[str] = []
    independent_execution: bool = True
```

**Flow:**
```
┌─────────┐
│ Primary │
└────┬────┘
     │
     ▼
┌─────────────────────────────┐
│      Cross-Check Phase      │
│  ┌─────────┐ ┌─────────┐   │
│  │Checker_0│ │Checker_1│   │
│  └─────────┘ └─────────┘   │
└─────────────────────────────┘
     │
     ▼
┌──────────┐
│Consensus │
└──────────┘
```

**Use Cases:**
- Code review automation
- Content moderation
- Fact verification

### Delegation

Capability-based dynamic routing.

**Configuration:**
```python
class DelegationConfig(SQLModel):
    routing_strategy: str = "capability_match"
    load_balance: bool = True
    fallback_agent_name: str | None = None
    max_queue_size: int = 100
    priority_queue: bool = False
```

**Use Cases:**
- Multi-skill task routing
- Load distribution
- Expertise matching

## Topology Registry

```python
from qmcp.agentframework.topologies import TopologyRegistry, TopologyType

# Get topology class
topo_class = TopologyRegistry.get(TopologyType.DEBATE)

# Create topology instance
topology = TopologyRegistry.create(
    topology_model,
    agents_dict,
    db_session,
)

# Execute
result = await topology.run(context)
```

## Usage Example

### Creating and Running a Debate

```python
from qmcp.agentframework import (
    AgentType, AgentRole, AgentConfig, Topology, TopologyType, DebateConfig,
    Models,  # Pre-configured model registry
)
from qmcp.agentframework.topologies import TopologyRegistry, ExecutionContext
from uuid import uuid4

# Create agents using pre-configured model from registry
proponent = AgentType(
    name="optimist",
    description="Argues the benefits",
    role=AgentRole.CRITIC,
    config=AgentConfig(
        model_config_obj=Models.CLAUDE_SONNET_4,
        system_prompt="You argue in favor of the topic.",
    ).model_dump(),
)
opponent = AgentType(
    name="skeptic",
    description="Argues the risks",
    role=AgentRole.CRITIC,
    config=AgentConfig(
        model_config_obj=Models.CLAUDE_SONNET_4,
        system_prompt="You argue against the topic.",
    ).model_dump(),
)
mediator = AgentType(
    name="judge",
    description="Synthesizes the debate",
    role=AgentRole.SYNTHESIZER,
    config=AgentConfig(
        model_config_obj=Models.CLAUDE_SONNET_4,
        system_prompt="You synthesize both perspectives.",
    ).model_dump(),
)

# Create topology
topology = Topology(
    name="ai_debate",
    description="Debate about AI safety",
    topology_type=TopologyType.DEBATE,
    config=DebateConfig(max_rounds=3).model_dump(),
)

# Persist and run
async with session:
    for agent in [proponent, opponent, mediator]:
        session.add(agent)
    session.add(topology)
    await session.commit()

    agents = {
        "proponent": proponent,
        "opponent": opponent,
        "mediator": mediator,
    }

    topo = TopologyRegistry.create(topology, agents, session)
    context = ExecutionContext(
        execution_id=uuid4(),
        topology_id=topology.id,
        input_data={"topic": "Should AI development be regulated?"},
    )

    result = await topo.run(context)
```
