# QMCP Agent Framework: Runners Specification

Note: Design reference. Implementation status is documented in docs/agentframework.md.


## Overview

Runners are responsible for executing topologies in different environments. The framework provides multiple runner implementations for different use cases, from simple local execution to distributed Metaflow workflows.

## Runner Architecture

### Design Goals

1. **Abstraction**: Topologies don't need to know about execution environment
2. **Scalability**: Support local development to distributed production
3. **Observability**: Consistent logging and metrics across runners
4. **Checkpointing**: Resume failed executions where possible
5. **Integration**: Seamless integration with QMCP server and Metaflow

## Package Structure

```
qmcp/agentframework/runners/
├── __init__.py
├── base.py              # BaseRunner, RunConfig, RunResult
├── local.py             # LocalRunner (in-process)
├── async_runner.py      # AsyncRunner (concurrent)
├── metaflow_runner.py   # MetaflowRunner (DAG generation)
└── metaflow_helpers.py  # Utilities for generated flows
```

## Base Runner

### File: `qmcp/agentframework/runners/base.py`

```python
"""Base runner class and utilities."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional, Type, TypeVar
from uuid import UUID, uuid4

import structlog
from sqlmodel import SQLModel

from qmcp.agentframework.models import (
    AgentType,
    Execution,
    ExecutionStatus,
    Topology,
)
from qmcp.agentframework.topologies import TopologyRegistry


logger = structlog.get_logger(__name__)


class RunConfig(SQLModel):
    """Configuration for a topology run."""
    correlation_id: Optional[str] = None
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
    output: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    runner_type: str
    topology_name: str
    correlation_id: Optional[str] = None


T = TypeVar("T", bound="BaseRunner")


class BaseRunner(ABC):
    """Abstract base class for topology runners."""
    
    runner_type: str = "base"
    
    def __init__(self, config: Optional[RunConfig] = None):
        self.config = config or RunConfig()
        self._logger = logger.bind(runner_type=self.runner_type)
    
    async def run(
        self,
        topology: Topology,
        agents: dict[str, AgentType],
        input_data: dict[str, Any],
        **kwargs,
    ) -> RunResult:
        """Execute a topology with given agents and input."""
        start_time = datetime.utcnow()
        correlation_id = self.config.correlation_id or str(uuid4())
        
        self._logger.info(
            "Starting topology execution",
            topology=topology.name,
            correlation_id=correlation_id,
        )
        
        try:
            self._validate_topology(topology, agents)
            execution = await self._execute_with_retries(
                topology, agents, input_data, correlation_id
            )
            
            end_time = datetime.utcnow()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return RunResult(
                execution_id=execution.id,
                status=execution.status,
                output=execution.output_data,
                error=execution.error,
                started_at=start_time,
                completed_at=end_time,
                duration_ms=duration_ms,
                runner_type=self.runner_type,
                topology_name=topology.name,
                correlation_id=correlation_id,
            )
            
        except Exception as e:
            end_time = datetime.utcnow()
            return RunResult(
                execution_id=uuid4(),
                status=ExecutionStatus.FAILED,
                error=str(e),
                started_at=start_time,
                completed_at=end_time,
                duration_ms=int((end_time - start_time).total_seconds() * 1000),
                runner_type=self.runner_type,
                topology_name=topology.name,
                correlation_id=correlation_id,
            )
    
    @abstractmethod
    async def _execute(
        self,
        topology: Topology,
        agents: dict[str, AgentType],
        input_data: dict[str, Any],
        correlation_id: str,
    ) -> Execution:
        pass
    
    @abstractmethod
    async def get_db_session(self):
        pass
    
    def _validate_topology(self, topology: Topology, agents: dict[str, AgentType]) -> None:
        if TopologyRegistry.get(topology.topology_type) is None:
            raise ValueError(f"Unknown topology type: {topology.topology_type}")
        if not agents:
            raise ValueError("At least one agent required")
    
    async def _execute_with_retries(
        self,
        topology: Topology,
        agents: dict[str, AgentType],
        input_data: dict[str, Any],
        correlation_id: str,
    ) -> Execution:
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return await self._execute(topology, agents, input_data, correlation_id)
            except Exception as e:
                last_error = e
                if not self.config.retry_on_failure or attempt >= self.config.max_retries:
                    raise
                self._logger.warning(f"Retry {attempt + 1}/{self.config.max_retries}")
        raise last_error


class RunnerRegistry:
    """Registry for runner implementations."""
    _runners: dict[str, Type[BaseRunner]] = {}
    
    @classmethod
    def register(cls, runner_class: Type[BaseRunner]) -> Type[BaseRunner]:
        cls._runners[runner_class.runner_type] = runner_class
        return runner_class
    
    @classmethod
    def get(cls, runner_type: str) -> Optional[Type[BaseRunner]]:
        return cls._runners.get(runner_type)
    
    @classmethod
    def create(cls, runner_type: str, config: Optional[RunConfig] = None) -> BaseRunner:
        runner_class = cls.get(runner_type)
        if runner_class is None:
            raise ValueError(f"Unknown runner type: {runner_type}")
        return runner_class(config=config)


def runner(cls: Type[T]) -> Type[T]:
    """Decorator to register a runner class."""
    return RunnerRegistry.register(cls)
```

## Metaflow Helpers

### File: `qmcp/agentframework/runners/metaflow_helpers.py`

```python
"""Helper functions for Metaflow-generated flows."""

from typing import Any, Optional
import json


async def get_db_session():
    """Get database session for Metaflow step."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlmodel import SQLModel
    from sqlmodel.ext.asyncio.session import AsyncSession
    import os
    
    db_url = os.environ.get("QMCP_DATABASE_URL", "sqlite+aiosqlite:///./agent.db")
    engine = create_async_engine(db_url, echo=False)
    
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return session_factory()


def invoke_agent(agent_config: dict[str, Any], input_data: dict[str, Any]) -> dict[str, Any]:
    """
    Invoke an agent synchronously (for Metaflow steps).
    
    In production, integrates with PydanticAI or Anthropic SDK.
    """
    import os
    
    # Build messages
    messages = []
    system_prompt = agent_config.get("system_prompt")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Format input
    if isinstance(input_data, dict):
        content = json.dumps(input_data, indent=2)
    else:
        content = str(input_data)
    messages.append({"role": "user", "content": content})
    
    # Call LLM (placeholder - would use actual SDK)
    model = agent_config.get("model", "claude-sonnet-4-20250514")
    max_tokens = agent_config.get("max_tokens", 4096)
    temperature = agent_config.get("temperature", 0.7)
    
    # In production:
    # from anthropic import Anthropic
    # client = Anthropic()
    # response = client.messages.create(
    #     model=model,
    #     max_tokens=max_tokens,
    #     temperature=temperature,
    #     messages=messages,
    # )
    # return {"content": response.content[0].text}
    
    return {"content": f"[Mock response for: {content[:100]}...]", "model": model}


def aggregate_results(
    results: list[tuple[str, dict[str, Any]]],
    method: str = "synthesis",
) -> dict[str, Any]:
    """
    Aggregate results from multiple agents.
    
    Methods:
    - vote: Most common answer
    - concat: Concatenate all
    - best_of: Highest confidence
    - synthesis: LLM synthesis
    """
    if not results:
        return {"error": "No results to aggregate"}
    
    if method == "vote":
        from collections import Counter
        outputs = [r[1].get("content", "") for r in results]
        counter = Counter(outputs)
        return {"content": counter.most_common(1)[0][0], "method": "vote"}
    
    elif method == "concat":
        outputs = [f"[{r[0]}]: {r[1].get('content', '')}" for r in results]
        return {"content": "\n\n".join(outputs), "method": "concat"}
    
    elif method == "best_of":
        best = max(results, key=lambda r: r[1].get("confidence", 0.5))
        return {"content": best[1].get("content", ""), "method": "best_of", "source": best[0]}
    
    elif method == "synthesis":
        # Use LLM to synthesize
        combined = "\n\n".join(
            f"Response from {name}:\n{data.get('content', '')}"
            for name, data in results
        )
        synthesis_result = invoke_agent(
            {"system_prompt": "Synthesize these responses into a single coherent answer."},
            {"responses": combined}
        )
        return {"content": synthesis_result.get("content", ""), "method": "synthesis"}
    
    return {"error": f"Unknown aggregation method: {method}"}


def determine_consensus(
    primary_output: dict[str, Any],
    validations: list[tuple[str, dict[str, Any]]],
    method: str = "majority",
) -> dict[str, Any]:
    """
    Determine consensus from validation results.
    
    Methods:
    - unanimous: All must approve
    - majority: More than half approve
    - weighted: Weight by confidence
    """
    approvals = []
    rejections = []
    
    for checker_name, validation in validations:
        # Parse validation result
        content = validation.get("content", "{}")
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = {"approved": "approve" in content.lower()}
        else:
            parsed = content
        
        if parsed.get("approved", False):
            approvals.append((checker_name, parsed))
        else:
            rejections.append((checker_name, parsed))
    
    total = len(validations)
    
    if method == "unanimous":
        approved = len(rejections) == 0
    elif method == "majority":
        approved = len(approvals) > total / 2
    elif method == "weighted":
        approval_weight = sum(v[1].get("confidence", 0.5) for v in approvals)
        rejection_weight = sum(v[1].get("confidence", 0.5) for v in rejections)
        approved = approval_weight > rejection_weight
    else:
        approved = len(approvals) >= len(rejections)
    
    return {
        "approved": approved,
        "primary_output": primary_output,
        "approval_count": len(approvals),
        "rejection_count": len(rejections),
        "method": method,
        "issues": [v[1].get("issues", []) for v in rejections],
    }


def save_execution(topology_id: int, output_data: dict[str, Any]) -> None:
    """Save execution result to database."""
    import asyncio
    
    async def _save():
        from qmcp.agentframework.models import Execution, ExecutionStatus
        
        async with await get_db_session() as session:
            # Find or create execution record
            execution = Execution(
                topology_id=topology_id,
                input_data={},  # Would be passed through
                output_data=output_data,
                status=ExecutionStatus.COMPLETED,
            )
            session.add(execution)
            await session.commit()
    
    asyncio.run(_save())
```

## Package Initialization

### File: `qmcp/agentframework/runners/__init__.py`

```python
"""Agent topology runners."""

from .base import (
    BaseRunner,
    RunConfig,
    RunResult,
    RunnerRegistry,
    runner,
)

# Import runners to register them
from .local import LocalRunner
from .async_runner import AsyncRunner
from .metaflow_runner import MetaflowRunner

__all__ = [
    # Base
    "BaseRunner",
    "RunConfig", 
    "RunResult",
    "RunnerRegistry",
    "runner",
    # Implementations
    "LocalRunner",
    "AsyncRunner",
    "MetaflowRunner",
]
```

## CLI Integration

### File: `qmcp/agentframework/cli.py`

```python
"""CLI commands for agent management."""

import json
from typing import Optional

import typer

app = typer.Typer(name="agents", help="Agent management commands")


@app.command("list")
def list_agents(
    role: Optional[str] = typer.Option(None, help="Filter by role"),
    format: str = typer.Option("table", help="Output format: table, json"),
):
    """List all registered agent types."""
    import asyncio
    from rich.console import Console
    from rich.table import Table
    
    async def _list():
        from qmcp.agentframework.runners import LocalRunner
        from sqlmodel import select
        from qmcp.agentframework.models import AgentType
        
        runner = LocalRunner()
        async with await runner.get_db_session() as session:
            stmt = select(AgentType)
            if role:
                stmt = stmt.where(AgentType.role == role)
            result = await session.execute(stmt)
            return result.scalars().all()
    
    agents = asyncio.run(_list())
    console = Console()
    
    if format == "json":
        console.print(json.dumps([a.model_dump() for a in agents], indent=2))
    else:
        table = Table(title="Agent Types")
        table.add_column("ID")
        table.add_column("Name")
        table.add_column("Role")
        table.add_column("Description")
        
        for agent in agents:
            table.add_row(str(agent.id), agent.name, agent.role.value, agent.description[:50])
        
        console.print(table)


@app.command("create")
def create_agent(
    name: str = typer.Argument(..., help="Agent name"),
    role: str = typer.Argument(..., help="Agent role"),
    description: str = typer.Option("", help="Agent description"),
    model: str = typer.Option("claude-sonnet-4-20250514", help="LLM model"),
    system_prompt: Optional[str] = typer.Option(None, help="System prompt"),
):
    """Create a new agent type."""
    import asyncio
    from rich.console import Console
    
    async def _create():
        from qmcp.agentframework.runners import LocalRunner
        from qmcp.agentframework.models import AgentType, AgentRole, AgentConfig
        
        runner = LocalRunner()
        async with await runner.get_db_session() as session:
            agent = AgentType(
                name=name,
                description=description or f"{name} agent",
                role=AgentRole(role),
                config=AgentConfig(
                    model=model,
                    system_prompt=system_prompt,
                ).model_dump(),
            )
            session.add(agent)
            await session.commit()
            await session.refresh(agent)
            return agent
    
    agent = asyncio.run(_create())
    console = Console()
    console.print(f"[green]Created agent:[/green] {agent.name} (ID: {agent.id})")


@app.command("show")
def show_agent(name: str = typer.Argument(..., help="Agent name")):
    """Show agent details."""
    import asyncio
    from rich.console import Console
    from rich.panel import Panel
    
    async def _get():
        from qmcp.agentframework.runners import LocalRunner
        from sqlmodel import select
        from qmcp.agentframework.models import AgentType
        
        runner = LocalRunner()
        async with await runner.get_db_session() as session:
            stmt = select(AgentType).where(AgentType.name == name)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
    
    agent = asyncio.run(_get())
    console = Console()
    
    if agent:
        console.print(Panel(
            f"Name: {agent.name}\n"
            f"Role: {agent.role.value}\n"
            f"Description: {agent.description}\n"
            f"Config: {json.dumps(agent.config, indent=2)}",
            title=f"Agent: {agent.name}"
        ))
    else:
        console.print(f"[red]Agent not found:[/red] {name}")


# Topology commands
topology_app = typer.Typer(name="topologies", help="Topology management")


@topology_app.command("list")
def list_topologies(format: str = typer.Option("table", help="Output format")):
    """List all topologies."""
    import asyncio
    from rich.console import Console
    from rich.table import Table
    
    async def _list():
        from qmcp.agentframework.runners import LocalRunner
        from sqlmodel import select
        from qmcp.agentframework.models import Topology
        
        runner = LocalRunner()
        async with await runner.get_db_session() as session:
            result = await session.execute(select(Topology))
            return result.scalars().all()
    
    topologies = asyncio.run(_list())
    console = Console()
    
    if format == "json":
        console.print(json.dumps([t.model_dump() for t in topologies], indent=2))
    else:
        table = Table(title="Topologies")
        table.add_column("ID")
        table.add_column("Name")
        table.add_column("Type")
        table.add_column("Description")
        
        for topo in topologies:
            table.add_row(str(topo.id), topo.name, topo.topology_type.value, topo.description[:40])
        
        console.print(table)


@topology_app.command("run")
def run_topology(
    name: str = typer.Argument(..., help="Topology name"),
    input_json: str = typer.Option("{}", help="Input data as JSON"),
    runner_type: str = typer.Option("local", help="Runner type: local, async, metaflow"),
):
    """Execute a topology."""
    import asyncio
    from rich.console import Console
    
    async def _run():
        from qmcp.agentframework.runners import RunnerRegistry, RunConfig
        from sqlmodel import select
        from qmcp.agentframework.models import Topology, TopologyMembership, AgentType
        
        runner = RunnerRegistry.create(runner_type, RunConfig())
        
        async with await runner.get_db_session() as session:
            # Get topology
            stmt = select(Topology).where(Topology.name == name)
            result = await session.execute(stmt)
            topology = result.scalar_one_or_none()
            
            if not topology:
                return None, f"Topology not found: {name}"
            
            # Get agents
            stmt = (
                select(TopologyMembership, AgentType)
                .join(AgentType)
                .where(TopologyMembership.topology_id == topology.id)
            )
            result = await session.execute(stmt)
            
            agents = {}
            for membership, agent_type in result:
                agents[membership.slot_name] = agent_type
            
            # Run
            input_data = json.loads(input_json)
            return await runner.run(topology, agents, input_data), None
    
    result, error = asyncio.run(_run())
    console = Console()
    
    if error:
        console.print(f"[red]Error:[/red] {error}")
    else:
        console.print(f"[green]Execution completed:[/green]")
        console.print(f"  Status: {result.status.value}")
        console.print(f"  Duration: {result.duration_ms}ms")
        if result.output:
            console.print(f"  Output: {json.dumps(result.output, indent=2)[:500]}")


app.add_typer(topology_app)


# Mixin commands
@app.command("mixins")
def list_mixins():
    """List available capability mixins."""
    from rich.console import Console
    from rich.table import Table
    from qmcp.agentframework.mixins import MixinRegistry
    
    console = Console()
    table = Table(title="Available Mixins")
    table.add_column("Name")
    table.add_column("Version")
    table.add_column("Description")
    table.add_column("Dependencies")
    
    for mixin_info in MixinRegistry.list_all():
        table.add_row(
            mixin_info["name"],
            mixin_info["version"],
            mixin_info["description"][:40],
            ", ".join(mixin_info["dependencies"]) or "-"
        )
    
    console.print(table)


if __name__ == "__main__":
    app()
```

## Usage Examples

### Local Execution

```python
import asyncio
from qmcp.agentframework.models import AgentType, AgentRole, Topology, TopologyType, EnsembleConfig
from qmcp.agentframework.runners import LocalRunner, RunConfig

async def main():
    # Create runner
    runner = LocalRunner(
        config=RunConfig(timeout_seconds=300),
        db_url="sqlite+aiosqlite:///./demo.db"
    )
    
    # Define agents
    agents = {
        "ensemble_0": AgentType(
            name="researcher_1",
            description="Research agent 1",
            role=AgentRole.SPECIALIST,
            config={"model": "claude-sonnet-4-20250514"}
        ),
        "ensemble_1": AgentType(
            name="researcher_2", 
            description="Research agent 2",
            role=AgentRole.SPECIALIST,
            config={"model": "claude-sonnet-4-20250514"}
        ),
        "aggregator": AgentType(
            name="synthesizer",
            description="Synthesizes results",
            role=AgentRole.SYNTHESIZER,
            config={"model": "claude-sonnet-4-20250514"}
        ),
    }
    
    # Define topology
    topology = Topology(
        name="research_ensemble",
        description="Parallel research with synthesis",
        topology_type=TopologyType.ENSEMBLE,
        config=EnsembleConfig(aggregation_method="synthesis").model_dump()
    )
    
    # Execute
    result = await runner.run(
        topology=topology,
        agents=agents,
        input_data={"prompt": "What are the latest developments in quantum computing?"}
    )
    
    print(f"Status: {result.status}")
    print(f"Output: {result.output}")

asyncio.run(main())
```

### Metaflow DAG Generation

```python
from qmcp.agentframework.models import AgentType, AgentRole, Topology, TopologyType, DebateConfig
from qmcp.agentframework.runners import MetaflowRunner

# Create runner
runner = MetaflowRunner(output_dir="./generated_flows", environment="local")

# Define topology
topology = Topology(
    name="ai_debate",
    description="Debate about AI safety",
    topology_type=TopologyType.DEBATE,
    config=DebateConfig(max_rounds=3).model_dump()
)

# Define agents
agents = {
    "proponent": AgentType(name="optimist", description="...", role=AgentRole.CRITIC),
    "opponent": AgentType(name="skeptic", description="...", role=AgentRole.CRITIC),
    "mediator": AgentType(name="judge", description="...", role=AgentRole.SYNTHESIZER),
}

# Generate flow file
flow_code = runner._topology_to_flow(topology, agents)
print(flow_code)

# Execute (would run the generated Metaflow)
# result = await runner.run(topology, agents, {"topic": "AI safety"})
```

### CLI Usage

```bash
# List agents
qmcp agents list

# Create agent
qmcp agents create planner planner --description "Strategic planning agent"

# List topologies
qmcp agents topologies list

# Run topology
qmcp agents topologies run my_debate --input-json '{"topic": "AI ethics"}'

# List available mixins
qmcp agents mixins
```

## Next Steps

1. Implement comprehensive tests (see `05-TESTS.md`)
2. Add FastAPI router integration
3. Integrate with PydanticAI for LLM calls
4. Add Kubernetes/AWS Batch Metaflow backends
