# Agent Framework: Topologies

Topologies define collaboration patterns for multi-agent systems. Each topology specifies how agents communicate, share information, and reach conclusions.

This page describes two separate surfaces. The framework classes and registry
below define topology shapes, and several framework runners are still stubs.
The saved-design voice runner is a separate, consent-gated path: it runs the
reusable components and composed designs a saved design names, with its own
behaviour and limits documented in the [voice loop demo](../voice-loop-demo.md).
A framework class's status does not by itself say whether the voice runner can
run a configuration of that kind; the `voice_runnable` declaration does.

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

The role slots and flows in this section are illustrative framework shapes.
The saved-design voice runner instead runs the components a design selects in
`config.components` and composes the saved designs in `config.compose`. The
schemas and capability responses below are the contract, not the sketches.

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

For the saved-design voice runner, selected component instructions become the
checker perspectives, and their count sets the number of checkers. With no
component references, `num_checkers` chooses the generated perspectives. Other
non-default framework options are refused rather than silently ignored.

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

### Council

Deliberative plurality-seeking topology where an arbiter presides over a council of diverse agent perspectives. Each council member brings a unique viewpoint, and decisions are reached through structured deliberation until consensus or majority is achieved.

**Required Slots:**
- `arbiter` - Council Manager: Facilitates discussion, synthesizes positions, makes final decisions unless the council is advisory
- `storyteller` - Relatable Storyteller: Frames issues in narrative form, makes abstract concepts tangible
- `dreamer` - Infinite Dreamer: Explores possibilities without constraint, generates creative alternatives
- `strategist` - Pragmatic Strategist: Focuses on practical implementation, resource constraints
- `sanity_check` - Sanity Check: Validates feasibility, catches edge cases and potential issues
- `archivist` - Tidy Archivist: Maintains context, references history, ensures consistency
- `efficist` - Brutal Efficist: Cuts through complexity, demands efficiency, eliminates waste
- `accomplisher` - Eager Accomplisher: Drives toward completion, breaks down blockers
- `reflector` - Technical Reflector: Provides deep technical analysis, considers implications

**Configuration:**
```python
class CouncilConfig(SQLModel):
    max_rounds: int = 5
    consensus_threshold: float = 0.67  # 0.5=majority, 0.67=supermajority, 1.0=unanimous
    consensus_method: ConsensusMethod = "quorum"
    allow_early_consensus: bool = True
    require_all_voices: bool = True
    arbiter_can_override: bool = True
    deliberation_style: str = "round_robin"  # round_robin, open_floor, structured
    speaking_order: list[str] = [
        "storyteller", "dreamer", "strategist", "sanity_check",
        "archivist", "efficist", "accomplisher", "reflector"
    ]
    min_contribution_length: int = 100
    track_position_changes: bool = True
    synthesis_after_each_round: bool = True
```

**Two councils, one setting.** With `arbiter_can_override` true, the default,
the arbiter takes the final decision when consensus fails, and
`qmcp.orchestration` refuses that council whatever it is pointed at. With it
false the council is advisory: the arbiter synthesizes, a split is reported as
a split, and the plane declares it separately in `OPTIONS` -- a proposal for
an ordinary question, still refused an attested act, because a consensus is a
conclusion a machine reached. `uv run qmcp council create --no-arbiter-override`
creates one, and a saved design's `capability` block says which it is read as.

The saved-design voice runner runs only the advisory council, in the order its
`speaking_order` gives the selected components, and reports a synthesis without
choosing. A deciding council is refused before the approval is asked.

**Flow:**
```
                              ┌─────────────────────────────────────────────────┐
                              │              COUNCIL CHAMBER                    │
                              │                                                 │
┌───────┐                     │  ┌────────────┐     ┌────────────┐             │
│ Input │────────────────────▶│  │ Storyteller│────▶│  Dreamer   │             │
└───────┘                     │  └────────────┘     └─────┬──────┘             │
                              │                          │                     │
                              │  ┌────────────┐     ┌────▼──────┐             │
                              │  │ Reflector  │◀────│ Strategist│             │
                              │  └─────┬──────┘     └────────────┘             │
                              │        │                                       │
                              │  ┌─────▼──────┐     ┌────────────┐             │
                              │  │Accomplisher│────▶│Sanity Check│             │
                              │  └────────────┘     └─────┬──────┘             │
                              │                          │                     │
                              │  ┌────────────┐     ┌────▼──────┐             │
                              │  │  Efficist  │◀────│  Archivist│             │
                              │  └─────┬──────┘     └────────────┘             │
                              │        │                                       │
                              └────────┼───────────────────────────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │     ARBITER     │
                              │ (Council Manager)│
                              │                 │
                              │  • Synthesizes  │
                              │  • Checks vote  │
                              │  • Decides if   │
                              │    no consensus │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │    DECISION     │
                              │  (or next round)│
                              └─────────────────┘
```

**Agent Role Descriptions:**

| Role | Persona | Contribution Style |
|------|---------|-------------------|
| **Council Manager** (Arbiter) | Impartial facilitator | Synthesizes, mediates, decides |
| **Relatable Storyteller** | Narrative translator | Frames in human terms, uses analogies |
| **Infinite Dreamer** | Unconstrained idealist | "What if we could...", blue-sky thinking |
| **Pragmatic Strategist** | Implementation realist | "Here's how we actually do it..." |
| **Sanity Check** | Devil's advocate | "But have we considered...", edge cases |
| **Tidy Archivist** | Institutional memory | "Previously we decided...", consistency |
| **Brutal Efficist** | Efficiency enforcer | "Cut the fluff", minimal viable path |
| **Eager Accomplisher** | Action driver | "Let's ship it", unblocks progress |
| **Technical Reflector** | Deep analyzer | Technical implications, architecture |

**Use Cases:**
- Complex architectural decisions requiring multiple perspectives
- Product planning where creativity and pragmatism must balance
- Risk assessment needing diverse viewpoints
- Strategic planning with competing priorities
- Code review requiring both technical and practical considerations
- Documentation decisions balancing completeness vs. usability

**Quick Example:**
```python
from qmcp.agentframework import (
    Topology, TopologyType, CouncilConfig,
)
from qmcp.agentframework.topologies import TopologyRegistry, ExecutionContext

# Create council topology
council = Topology(
    name="architecture_council",
    description="Council for architectural decisions",
    topology_type=TopologyType.COUNCIL,
    config=CouncilConfig(
        max_rounds=5,
        consensus_threshold=0.67,  # Supermajority
        arbiter_can_override=True,
    ).model_dump(),
)

# The framework runner is still a stub; `topo.run` raises NotImplementedError.
# A saved advisory council runs through the voice runner instead.
topo = TopologyRegistry.create(council, agents, session)
```

**Full Implementation:**

See [`examples/flows/council_deliberation.py`](../../examples/flows/council_deliberation.py) for a complete working example with:
- All 9 council member personas with detailed system prompts
- Multi-round deliberation with consensus tracking
- Position tracking and synthesis after each round
- Final decision rendering with rationale and recommendations

```bash
# Run with a local LLM
uv run python examples/flows/council_deliberation.py run \
    --question "Should we migrate from REST to GraphQL?" \
    --context "E-commerce platform with 50+ microservices" \
    --llm-base-url "http://localhost:11434/v1" \
    --llm-model "llama3.1"
```

## The capability plane

`qmcp.orchestration.PLANE` declares, for every registered topology, what
running it would do: its status, whether it spends money, writes to a
repository or decides, why, and what a caller must supply first. `uv run qmcp
orchestration plane` prints it, and `GET /v1/orchestration/plane` serves it.
Every declaration is written by hand and read, never discovered by running the
shape, since running it would already be the act being judged.

**Statuses.** `runs` is implemented and safe for ordinary work. `brainstorm` is
a designed shape whose `run` still raises: a proposal, not a runtime.
`refused` is a shape whose design performs an act this organisation reserves
for a person.

**Refusals belong to a pairing.** `governance/qm/ci/attested-registry.yaml`
lists the acts that are a person's by constitution -- ratifying a record,
cutting a tag, authorising a paid call, among others. A topology that decides
is refused when pointed at one of them, and allowed an ordinary question. A
topology that only reports is allowed either, because reporting on an act does
not perform it. `refuses(kind, act, config)` answers for one pairing, and the
design routes ask it with `?act=`.

**Options.** `OPTIONS` declares a shape that behaves differently under one
setting, read from the design's own configuration. A council with
`arbiter_can_override` false is the advisory council: no arbiter decides, so it
is a proposal for an ordinary question and is still refused an attested act.

**Needs and their remedies.** Every need names what supplies it: a build, a
budget, workers, a model, or a person. `unmet` reports what a caller is still
short of, and `GET /v1/orchestration/runnable` answers for a given hand. A need
for a person is never supplied by a parameter, so a refused shape cannot be
made runnable by a request.

**Drift.** `undeclared()` lists registered topologies with no declaration and
declarations for nothing; `stubs()` lists shapes still inheriting the base
`run`, so a `runs` declaration the registry contradicts is visible; and
`unregistered_types()` lists names in the vocabulary that nothing implements.

A window reads all of this from the routes rather than keeping its own copy of
which shapes spend, decide or are refused.

## Views of a topology

`qmcp.topology_view` describes a topology as boxes and arrows, for any window
to draw. A `View` holds kinds and notes and no coordinates, glyphs, colours or
widths, so a terminal panel and a browser graph draw from one description.

**Levels.** Level 0 is the black box: what goes in and what comes out. Level 1
is the parts. Level 2 is the flow between them. Levels 0 and 1 are derived from
level 2, so a summary cannot disagree with the flow it summarises.

**Kinds.** A box is an input, a worker, a gate, a store or an output; an arrow
is a flow, feedback or a refusal. A window may draw kinds alike and may not
merge them: a refusal drawn as a flow would read as a path that is taken. The
plane's marks travel with the view, so a shape that spends, writes or decides
shows it, and a refused shape is drawn refused rather than left out of the
gallery.

**Channels.** `/v1/topology/encoding` declares which visual channel carries
which data axis, so every window maps a line's thickness to the same thing.
`measured` has a channel of its own, separate from `strength`: an unmeasured
edge is an absence of evidence, not a weak edge, and a window that runs out of
channels drops one and says so rather than folding the two together.

**Readings.** `from_relations` draws one subject and what the thread archive
relates it to. Relations reaching the same address share one box, because the
address says they are about one thing; their arrows are not merged, so each
observation keeps its own weight and basis. A relation nobody measured has a
null weight, and stays null.

**What is served where.** The shapes, the encoding and the configuration
schemas name nobody and are served wherever the server is bound. The readings
are derived from a person's conversations and exist only on a loopback bind;
off loopback the route is absent rather than refused, so a response reveals
nothing about whether an archive is there. Saved designs and components write,
and are likewise loopback only.

## Over HTTP, for a designer

The server serves the vocabulary above to a window that draws it, so the window
carries no copy of what a shape is or what running one would do. Every answer
carries `"schema": 1`; a refusal (a 400, 404, 409 or 422) carries FastAPI's
`detail` instead, with the sentence that says what to look at, so a thing that
cannot be answered arrives as a reason rather than an empty list.

| Route | What it answers |
|---|---|
| `GET /v1/topology/shape/{kind}` | the shape as boxes and arrows, at a level; `governed` is served here too |
| `GET /v1/topology/schema/{kind}` | the JSON schema of the kind's configuration class -- what a form is built from; `governed` is a 404 saying it is a seam, not a configurable shape |
| `GET /v1/orchestration/plane` | `qmcp.orchestration.PLANE` as a document: per shape its status, whether it spends, writes or decides, why, and its needs with what supplies each; the `options` one setting selects instead; the needs and attested-act vocabularies; the drift reports |
| `GET /v1/orchestration/runnable?workers=&budget=&model=&built=` | what that hand could run now, and per shape what it is still short of. There is no `person` parameter and there will not be one |
| `POST /v1/topology-components` | save a reusable component by name, with its description and instruction |
| `GET /v1/topology-components` | every saved component of one project |
| `GET /v1/topology-components/{name}` | one component |
| `PUT /v1/topology-components/{name}` | change a component's description, instruction or version; its name is fixed |
| `POST /v1/topologies` | save a design. `config` is validated through the kind's configuration class and stored with its defaults filled; `config.components` names reusable components, each with optional `route_terms`, and `config.compose` names saved designs; the response carries an `address` and a `capability` block |
| `GET /v1/topologies` | every saved design of one project |
| `GET /v1/topologies/{ref}?act=` | one design, by id or by name; `act` asks the plane about a pairing |
| `PUT /v1/topologies/{ref}` | change `description`, `config` or `version`; name and kind are fixed; the references in a new `config` are checked as on save |

**A refused shape can be saved.** `council` is refused by the plane because its
arbiter decides, and the refusal is of the run. The saved row comes back with
the plane's `refusal` and a `saved_anyway` sentence, so a designer sees the
rule at the moment of choosing rather than after. Neither designs nor
components have a `DELETE`.

**References are checked, and components are shared.** An unknown component
or design, a design composing itself, a cycle, and composition deeper than
eight levels are each a 422. A component is shared by reference, so changing
one changes what every design that names it reads.

**Scope and exposure.** These routes write and carry no authorization of their
own, so the server registers them only on a loopback bind. Every design and
component belongs to a `project` (the body's `project` or `?project=`,
defaulting to the repository's short name); a name is unique within its
project and references resolve within it. `config.compose` may name another
project's design as `project/name`, and the cycle and depth checks follow
references across projects. A listing returns one project at a time.

**The block reads the design's own config.** A council saved with
`arbiter_can_override` false comes back with `option` naming that setting and
the advisory declaration in place of the refused one; changing the setting
with `PUT` changes the block.

`walkthrough/07-saving-a-shape-is-not-running-it.md` exercises every route
above through the test client, and `qmcp/topology_designs.py` and
`qmcp/orchestration_service.py` hold the reasoning. These routes save and
judge designs and invoke no model; the voice runner runs a supported saved
design only after a fresh approval, and each `capability` block carries the
runner's `voice_runnable`, `voice_spends`, `voice_writes` and `voice_decides`
beside the framework's declaration.

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

# Persist and build. `topo.run` is still a stub that raises
# `NotImplementedError`; a saved design runs through the voice runner.
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

    # await topo.run(context)  # NotImplementedError until the framework runner exists
```
