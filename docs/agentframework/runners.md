# Agent Framework: Runners

Runners execute topologies in different environments. They handle agent invocation, execution tracking, and result persistence.

## Architecture

### Design Goals

1. **Abstraction** - Topologies don't know about execution environment
2. **Scalability** - Support local development to distributed production
3. **Observability** - Consistent logging and metrics
4. **Checkpointing** - Resume failed executions
5. **Integration** - Seamless QMCP server and Metaflow integration

### Base Classes

```python
class RunConfig(SQLModel):
    """Configuration for a topology run."""
    correlation_id: str | None = None
    timeout_seconds: int = 3600
    checkpoint_enabled: bool = True
    retry_on_failure: bool = True
    max_retries: int = 3
    log_level: str = "INFO"
    trace_enabled: bool = True
    metrics_enabled: bool = True

class RunResult(SQLModel):
    """Result from a topology run."""
    execution_id: UUID
    status: ExecutionStatus
    output: dict[str, Any] | None
    error: str | None
    started_at: datetime
    completed_at: datetime | None
    duration_ms: int | None
    runner_type: str
    topology_name: str
    correlation_id: str | None
```

## Runner Types

### LocalRunner

Executes topologies in-process. Best for development and testing.

```python
from qmcp.agentframework.runners import LocalRunner, RunConfig

runner = LocalRunner(
    config=RunConfig(timeout_seconds=300),
    db_url="sqlite+aiosqlite:///./agent.db",
)

result = await runner.run(
    topology=topology,
    agents=agents,
    input_data={"prompt": "Analyze this data"},
)
```

**Features:**
- Single-process execution
- Direct database access
- Easy debugging
- Fast iteration

### AsyncRunner

Concurrent execution with semaphore-based rate limiting.

```python
from qmcp.agentframework.runners import AsyncRunner

runner = AsyncRunner(
    db_url="sqlite+aiosqlite:///./agent.db",
    max_concurrent=5,
)
```

**Features:**
- Parallel agent execution where topology allows
- Configurable concurrency limits
- Async/await throughout

### MetaflowRunner

Generates Metaflow DAGs for distributed execution.

```python
from qmcp.agentframework.runners import MetaflowRunner

runner = MetaflowRunner(
    output_dir="./generated_flows",
    environment="local",  # or "kubernetes", "batch"
)

# Generate flow file
flow_code = runner._topology_to_flow(topology, agents)

# Or run directly
result = await runner.run(topology, agents, input_data)
```

**Features:**
- Automatic DAG generation from topologies
- Supports Kubernetes, AWS Batch, local execution
- Built-in checkpointing and resume
- Parameter passing between steps

## Runner Registry

```python
from qmcp.agentframework.runners import RunnerRegistry

# List registered runners
runners = ["local", "async", "metaflow"]

# Create runner by type
runner = RunnerRegistry.create("local", config)

# Custom runner registration
@runner
class MyRunner(BaseRunner):
    runner_type = "my_runner"
```

## Generated Metaflow Flows

### Pipeline Topology Flow

```python
from metaflow import FlowSpec, step

class PipelineFlow(FlowSpec):
    @step
    def start(self):
        self.input_data = self.input_json
        self.next(self.parse)

    @step
    def parse(self):
        from qmcp.agentframework.runners.metaflow_helpers import invoke_agent
        self.parse_output = invoke_agent(PARSE_CONFIG, self.input_data)
        self.next(self.analyze)

    @step
    def analyze(self):
        self.analyze_output = invoke_agent(ANALYZE_CONFIG, self.parse_output)
        self.next(self.generate)

    @step
    def generate(self):
        self.output = invoke_agent(GENERATE_CONFIG, self.analyze_output)
        self.next(self.end)

    @step
    def end(self):
        save_execution(self.topology_id, self.output)
```

### Ensemble Topology Flow

```python
from metaflow import FlowSpec, step

class EnsembleFlow(FlowSpec):
    @step
    def start(self):
        self.ensemble_agents = ["ensemble_0", "ensemble_1", "ensemble_2"]
        self.next(self.invoke_ensemble, foreach="ensemble_agents")

    @step
    def invoke_ensemble(self):
        agent_name = self.input
        self.result = invoke_agent(ENSEMBLE_CONFIGS[agent_name], self.input_data)
        self.next(self.aggregate)

    @step
    def aggregate(self, inputs):
        results = [(inp.agent_name, inp.result) for inp in inputs]
        self.output = aggregate_results(results, method="synthesis")
        self.next(self.end)

    @step
    def end(self):
        save_execution(self.topology_id, self.output)
```

## Helper Functions

### metaflow_helpers.py

```python
def invoke_agent(agent_config: dict, input_data: dict) -> dict:
    """Invoke agent synchronously for Metaflow steps."""
    # Builds messages, calls LLM, returns result

def aggregate_results(results: list, method: str) -> dict:
    """Aggregate ensemble outputs."""
    # vote, average, concat, best_of, synthesis

def determine_consensus(primary: dict, validations: list, method: str) -> dict:
    """Determine consensus from cross-check validations."""
    # unanimous, majority, weighted

def save_execution(topology_id: int, output_data: dict) -> None:
    """Save execution result to database."""
```

## CLI Commands

```bash
# List agents
qmcp agents list
qmcp agents list --role planner --format json

# Create agent
qmcp agents create my_planner planner --description "Planning agent"

# Show agent details
qmcp agents show my_planner

# List topologies
qmcp agents topologies list

# Run topology
qmcp agents topologies run my_debate --input-json '{"topic": "AI"}'

# List available mixins
qmcp agents mixins
```

## Usage Examples

### Complete Local Execution

```python
import asyncio
from qmcp.agentframework.models import (
    AgentType, AgentRole, AgentConfig, Topology, TopologyType, EnsembleConfig,
    Models,  # Pre-configured model registry
)
from qmcp.agentframework.runners import LocalRunner, RunConfig

async def main():
    runner = LocalRunner(
        config=RunConfig(timeout_seconds=300),
        db_url="sqlite+aiosqlite:///./demo.db",
    )

    # Use pre-configured model from registry
    agents = {
        "ensemble_0": AgentType(
            name="researcher_1",
            description="Research agent 1",
            role=AgentRole.SPECIALIST,
            config=AgentConfig(model_config_obj=Models.CLAUDE_SONNET_4).model_dump(),
        ),
        "ensemble_1": AgentType(
            name="researcher_2",
            description="Research agent 2",
            role=AgentRole.SPECIALIST,
            config=AgentConfig(model_config_obj=Models.CLAUDE_SONNET_4).model_dump(),
        ),
        "aggregator": AgentType(
            name="synthesizer",
            description="Synthesizes results",
            role=AgentRole.SYNTHESIZER,
            config=AgentConfig(model_config_obj=Models.CLAUDE_SONNET_4).model_dump(),
        ),
    }

    topology = Topology(
        name="research_ensemble",
        description="Parallel research with synthesis",
        topology_type=TopologyType.ENSEMBLE,
        config=EnsembleConfig(aggregation_method="synthesis").model_dump(),
    )

    result = await runner.run(
        topology=topology,
        agents=agents,
        input_data={"prompt": "What are recent quantum computing developments?"},
    )

    print(f"Status: {result.status}")
    print(f"Duration: {result.duration_ms}ms")
    print(f"Output: {result.output}")

asyncio.run(main())
```

### Running Generated Metaflow

```bash
# Generate flow
python -c "from qmcp.agentframework.runners import MetaflowRunner; ..."

# Run locally
python generated_flows/my_debate_flow.py run --input-json '{"topic": "AI"}'

# Run on Kubernetes
python generated_flows/my_debate_flow.py run --with kubernetes --input-json '...'
```
