# QMCP Agent Framework: Topologies Specification

Note: Design reference. Implementation status is documented in docs/agentframework.md.


## Overview

Topologies define how multiple agents collaborate to accomplish tasks. Each topology encapsulates a specific collaboration pattern with well-defined execution semantics, message routing, and result aggregation.

## Topology Architecture

### Design Principles

1. **Declarative Configuration**: Topologies are defined by configuration, not code
2. **Composability**: Topologies can be nested within other topologies
3. **Observability**: Full execution trace available for debugging
4. **Fault Tolerance**: Graceful handling of agent failures
5. **Determinism**: Same inputs produce reproducible behavior

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Topology Execution Flow                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐     ┌──────────────┐     ┌────────────────────┐   │
│  │  Input  │────▶│   Topology   │────▶│ Result Aggregation │   │
│  │  Data   │     │   Executor   │     │                    │   │
│  └─────────┘     └──────┬───────┘     └─────────┬──────────┘   │
│                         │                       │               │
│              ┌──────────┼──────────┐            │               │
│              ▼          ▼          ▼            ▼               │
│         ┌────────┐ ┌────────┐ ┌────────┐  ┌─────────┐          │
│         │Agent 1 │ │Agent 2 │ │Agent N │  │ Output  │          │
│         └────────┘ └────────┘ └────────┘  └─────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Base Topology Implementation

### File: `qmcp/agentframework/topologies/base.py`

```python
"""Base topology class and utilities."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, ClassVar, Optional, Type, TypeVar
from uuid import UUID

from pydantic import BaseModel
from sqlmodel import SQLModel

from qmcp.agentframework.models import (
    AgentInstance,
    AgentType,
    Execution,
    ExecutionStatus,
    Message,
    MessageType,
    Result,
    Topology,
    TopologyMembership,
    TopologyType,
)
from qmcp.agentframework.mixins import BaseMixin, MixinRegistry


class ExecutionContext(SQLModel):
    """Context passed through topology execution."""
    execution_id: UUID
    topology_id: int
    input_data: dict[str, Any]
    metadata: dict[str, Any] = {}
    round_number: int = 0
    parent_context: Optional["ExecutionContext"] = None


class AgentInvocationResult(SQLModel):
    """Result from a single agent invocation."""
    agent_instance_id: UUID
    output: dict[str, Any]
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    token_usage: Optional[dict[str, int]] = None
    duration_ms: int
    error: Optional[str] = None


T = TypeVar("T", bound="BaseTopology")


class BaseTopology(ABC):
    """
    Abstract base class for all topology implementations.
    
    Topologies define collaboration patterns between agents.
    Each topology type has specific execution semantics.
    """
    
    # Class attributes set by subclasses
    topology_type: ClassVar[TopologyType]
    config_class: ClassVar[Type[SQLModel]]
    
    def __init__(
        self,
        topology: Topology,
        agents: dict[str, AgentType],
        db_session: Any,  # AsyncSession
    ):
        """
        Initialize topology with database models.
        
        Args:
            topology: Topology model from database
            agents: Dict mapping slot names to agent types
            db_session: Database session for persistence
        """
        self.topology = topology
        self.agents = agents
        self.db_session = db_session
        self._instances: dict[str, AgentInstance] = {}
        self._mixins: dict[str, list[BaseMixin]] = {}
        self._config = self._parse_config()
    
    def _parse_config(self) -> SQLModel:
        """Parse topology config into typed model."""
        return self.config_class(**self.topology.config)
    
    # ========================================================================
    # Lifecycle Methods
    # ========================================================================
    
    async def setup(self) -> None:
        """
        Initialize topology for execution.
        
        Creates agent instances and loads mixins.
        """
        for slot_name, agent_type in self.agents.items():
            # Create agent instance
            instance = AgentInstance(
                agent_type_id=agent_type.id,
                state={},
                metadata_={"slot": slot_name},
            )
            self.db_session.add(instance)
            self._instances[slot_name] = instance
            
            # Load mixins for agent
            self._mixins[slot_name] = await self._load_mixins(agent_type, instance)
        
        await self.db_session.flush()
    
    async def _load_mixins(
        self, agent_type: AgentType, instance: AgentInstance
    ) -> list[BaseMixin]:
        """Load and bind mixins for an agent."""
        mixins = []
        capabilities = agent_type.get_capabilities()
        
        mixin_names = [c.name for c in capabilities if c.enabled]
        ordered = MixinRegistry.resolve_dependencies(mixin_names)
        
        for name in ordered:
            cap = next((c for c in capabilities if c.name == name), None)
            config = cap.config if cap else {}
            mixin = MixinRegistry.create(name, config)
            mixin.bind(instance)
            await mixin.on_create()
            mixins.append(mixin)
        
        return mixins
    
    async def teardown(self) -> None:
        """Clean up after execution."""
        for slot_name, mixins in self._mixins.items():
            for mixin in mixins:
                await mixin.on_destroy()
    
    # ========================================================================
    # Execution Methods
    # ========================================================================
    
    async def execute(
        self,
        input_data: dict[str, Any],
        correlation_id: Optional[str] = None,
    ) -> Execution:
        """
        Execute the topology with given input.
        
        Args:
            input_data: Input data for the topology
            correlation_id: Optional external correlation ID
            
        Returns:
            Execution record with results
        """
        # Create execution record
        execution = Execution(
            topology_id=self.topology.id,
            input_data=input_data,
            correlation_id=correlation_id,
            status=ExecutionStatus.RUNNING,
        )
        self.db_session.add(execution)
        await self.db_session.flush()
        
        # Update instances with execution ID
        for instance in self._instances.values():
            instance.execution_id = execution.id
        
        # Create execution context
        context = ExecutionContext(
            execution_id=execution.id,
            topology_id=self.topology.id,
            input_data=input_data,
        )
        
        try:
            # Setup
            await self.setup()
            
            # Call mixin on_start hooks
            for slot_name, mixins in self._mixins.items():
                for mixin in mixins:
                    await mixin.on_start(context.model_dump())
            
            # Run topology-specific logic
            output = await self._run(context)
            
            # Call mixin on_complete hooks
            for slot_name, mixins in self._mixins.items():
                for mixin in mixins:
                    output = await mixin.on_complete(output)
            
            # Mark complete
            execution.mark_complete(output)
            
        except Exception as e:
            # Call mixin on_error hooks
            for slot_name, mixins in self._mixins.items():
                for mixin in mixins:
                    await mixin.on_error(e)
            
            execution.mark_failed(str(e), {"type": type(e).__name__})
            raise
        
        finally:
            await self.teardown()
            await self.db_session.commit()
        
        return execution
    
    @abstractmethod
    async def _run(self, context: ExecutionContext) -> dict[str, Any]:
        """
        Execute topology-specific logic.
        
        Subclasses implement this to define collaboration pattern.
        
        Args:
            context: Execution context
            
        Returns:
            Final output data
        """
        pass
    
    # ========================================================================
    # Agent Invocation
    # ========================================================================
    
    async def invoke_agent(
        self,
        slot_name: str,
        messages: list[dict[str, Any]],
        context: ExecutionContext,
    ) -> AgentInvocationResult:
        """
        Invoke an agent with given messages.
        
        Applies all mixin hooks around the invocation.
        """
        instance = self._instances[slot_name]
        agent_type = self.agents[slot_name]
        mixins = self._mixins[slot_name]
        
        # Build request
        config = agent_type.config
        request = {
            "model": config.get("model", "claude-sonnet-4-20250514"),
            "messages": messages,
            "max_tokens": config.get("max_tokens", 4096),
            "temperature": config.get("temperature", 0.7),
        }
        
        # Add system prompt if present
        system_prompt = config.get("system_prompt")
        if system_prompt:
            request["messages"] = [
                {"role": "system", "content": system_prompt}
            ] + request["messages"]
        
        # Apply mixin on_invoke hooks
        for mixin in mixins:
            request = await mixin.on_invoke(request)
        
        # Make LLM call
        start_time = datetime.utcnow()
        try:
            response = await self._call_llm(request)
        except Exception as e:
            return AgentInvocationResult(
                agent_instance_id=instance.id,
                output={},
                duration_ms=0,
                error=str(e),
            )
        end_time = datetime.utcnow()
        
        # Apply mixin on_response hooks
        for mixin in mixins:
            response = await mixin.on_response(response)
        
        # Update instance state
        instance.last_active = end_time
        
        # Build result
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        return AgentInvocationResult(
            agent_instance_id=instance.id,
            output=response,
            confidence=response.get("_confidence"),
            reasoning=response.get("_reasoning", {}).get("steps"),
            token_usage=response.get("usage"),
            duration_ms=duration_ms,
        )
    
    async def _call_llm(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Make actual LLM API call.
        
        This would integrate with PydanticAI or direct API calls.
        """
        # Placeholder - would use actual LLM client
        # In production, integrate with pydantic-ai or anthropic SDK
        raise NotImplementedError("LLM integration required")
    
    # ========================================================================
    # Messaging
    # ========================================================================
    
    async def send_message(
        self,
        sender_slot: str,
        recipient_slot: Optional[str],
        message_type: MessageType,
        content: dict[str, Any],
        context: ExecutionContext,
    ) -> Message:
        """Send a message between agents."""
        sender = self._instances[sender_slot]
        recipient = self._instances.get(recipient_slot) if recipient_slot else None
        
        message = Message(
            execution_id=context.execution_id,
            sender_id=sender.id,
            recipient_id=recipient.id if recipient else None,
            message_type=message_type,
            content=content,
            round_number=context.round_number,
        )
        
        # Apply mixin on_message hooks for recipient
        if recipient_slot:
            for mixin in self._mixins[recipient_slot]:
                modified = await mixin.on_message(message)
                if modified:
                    message = modified
        
        self.db_session.add(message)
        return message
    
    async def broadcast_message(
        self,
        sender_slot: str,
        message_type: MessageType,
        content: dict[str, Any],
        context: ExecutionContext,
        exclude: Optional[list[str]] = None,
    ) -> list[Message]:
        """Broadcast message to all agents except sender and excluded."""
        messages = []
        exclude = exclude or []
        
        for slot_name in self._instances:
            if slot_name != sender_slot and slot_name not in exclude:
                msg = await self.send_message(
                    sender_slot, slot_name, message_type, content, context
                )
                messages.append(msg)
        
        return messages
    
    # ========================================================================
    # Result Recording
    # ========================================================================
    
    async def record_result(
        self,
        invocation: AgentInvocationResult,
        context: ExecutionContext,
    ) -> Result:
        """Record an agent's result."""
        result = Result(
            execution_id=context.execution_id,
            agent_instance_id=invocation.agent_instance_id,
            output=invocation.output,
            confidence=invocation.confidence,
            reasoning=str(invocation.reasoning) if invocation.reasoning else None,
            token_usage=invocation.token_usage,
            completed_at=datetime.utcnow(),
        )
        self.db_session.add(result)
        return result


class TopologyRegistry:
    """Registry for topology implementations."""
    
    _topologies: dict[TopologyType, Type[BaseTopology]] = {}
    
    @classmethod
    def register(cls, topology_class: Type[BaseTopology]) -> Type[BaseTopology]:
        """Register a topology implementation."""
        cls._topologies[topology_class.topology_type] = topology_class
        return topology_class
    
    @classmethod
    def get(cls, topology_type: TopologyType) -> Optional[Type[BaseTopology]]:
        """Get topology class by type."""
        return cls._topologies.get(topology_type)
    
    @classmethod
    def create(
        cls,
        topology: Topology,
        agents: dict[str, AgentType],
        db_session: Any,
    ) -> BaseTopology:
        """Create topology instance from model."""
        topology_class = cls.get(topology.topology_type)
        if topology_class is None:
            raise ValueError(f"Unknown topology type: {topology.topology_type}")
        return topology_class(topology, agents, db_session)


def topology(cls: Type[T]) -> Type[T]:
    """Decorator to register a topology class."""
    return TopologyRegistry.register(cls)
```

## Debate Topology

### File: `qmcp/agentframework/topologies/debate.py`

```python
"""Debate topology implementation."""

from typing import Any, ClassVar, Type

from sqlmodel import SQLModel

from qmcp.agentframework.models import (
    DebateConfig,
    MessageType,
    TopologyType,
)

from .base import (
    AgentInvocationResult,
    BaseTopology,
    ExecutionContext,
    topology,
)


@topology
class DebateTopology(BaseTopology):
    """
    Debate topology for structured argumentation.
    
    Agents take opposing positions and a mediator synthesizes conclusions.
    
    Slots:
    - proponent: Argues in favor
    - opponent: Argues against  
    - mediator: Synthesizes and decides
    
    Flow:
    1. Proponent makes initial argument
    2. Opponent responds with counter-argument
    3. Repeat for configured rounds
    4. Mediator synthesizes final conclusion
    """
    
    topology_type: ClassVar[TopologyType] = TopologyType.DEBATE
    config_class: ClassVar[Type[SQLModel]] = DebateConfig
    
    REQUIRED_SLOTS = ["proponent", "opponent", "mediator"]
    
    @property
    def debate_config(self) -> DebateConfig:
        return self._config  # type: ignore
    
    async def _run(self, context: ExecutionContext) -> dict[str, Any]:
        """Execute debate rounds."""
        # Validate slots
        for slot in self.REQUIRED_SLOTS:
            if slot not in self.agents:
                raise ValueError(f"Debate requires '{slot}' slot")
        
        topic = context.input_data.get("topic", "")
        question = context.input_data.get("question", topic)
        
        debate_history = []
        
        # Initial proponent argument
        proponent_result = await self._get_argument(
            "proponent",
            f"Argue IN FAVOR of: {question}",
            debate_history,
            context,
        )
        debate_history.append({
            "role": "proponent",
            "argument": proponent_result.output.get("content", ""),
        })
        
        # Debate rounds
        for round_num in range(self.debate_config.max_rounds):
            context.round_number = round_num + 1
            
            # Opponent responds
            opponent_result = await self._get_argument(
                "opponent",
                f"Argue AGAINST: {question}\n\nRespond to proponent's argument.",
                debate_history,
                context,
            )
            debate_history.append({
                "role": "opponent",
                "argument": opponent_result.output.get("content", ""),
            })
            
            # Check for early convergence
            if self.debate_config.allow_early_termination:
                if await self._check_convergence(debate_history):
                    break
            
            # Proponent responds (if not last round)
            if round_num < self.debate_config.max_rounds - 1:
                proponent_result = await self._get_argument(
                    "proponent",
                    f"Continue arguing IN FAVOR of: {question}\n\nRespond to opponent.",
                    debate_history,
                    context,
                )
                debate_history.append({
                    "role": "proponent",
                    "argument": proponent_result.output.get("content", ""),
                })
        
        # Mediator synthesizes
        synthesis = await self._synthesize(debate_history, question, context)
        
        return {
            "question": question,
            "rounds": len(debate_history) // 2,
            "debate_history": debate_history,
            "synthesis": synthesis.output.get("content", ""),
            "conclusion": synthesis.output.get("conclusion", ""),
        }
    
    async def _get_argument(
        self,
        slot: str,
        instruction: str,
        history: list[dict],
        context: ExecutionContext,
    ) -> AgentInvocationResult:
        """Get argument from an agent."""
        # Build messages with debate history
        messages = []
        
        for entry in history:
            role_label = "Proponent" if entry["role"] == "proponent" else "Opponent"
            messages.append({
                "role": "assistant" if entry["role"] == slot else "user",
                "content": f"[{role_label}]: {entry['argument']}",
            })
        
        messages.append({
            "role": "user",
            "content": instruction,
        })
        
        result = await self.invoke_agent(slot, messages, context)
        await self.record_result(result, context)
        
        # Send message to other participants
        await self.broadcast_message(
            slot,
            MessageType.RESPONSE,
            {"argument": result.output.get("content", "")},
            context,
        )
        
        return result
    
    async def _check_convergence(self, history: list[dict]) -> bool:
        """Check if debate has converged (agents agreeing)."""
        if len(history) < 4:
            return False
        
        # Simple heuristic: check for agreement language
        # In production, would use LLM to evaluate
        last_arguments = [h["argument"].lower() for h in history[-2:]]
        agreement_phrases = ["i agree", "valid point", "concede", "you're right"]
        
        return any(
            phrase in arg
            for arg in last_arguments
            for phrase in agreement_phrases
        )
    
    async def _synthesize(
        self,
        history: list[dict],
        question: str,
        context: ExecutionContext,
    ) -> AgentInvocationResult:
        """Mediator synthesizes the debate."""
        # Format debate for mediator
        debate_text = "\n\n".join(
            f"[{h['role'].title()}]: {h['argument']}"
            for h in history
        )
        
        messages = [{
            "role": "user",
            "content": f"""You are mediating a debate on: {question}

Here is the debate transcript:

{debate_text}

Please provide:
1. A summary of the key arguments on each side
2. An evaluation of the strongest points
3. Your conclusion and reasoning

Format your response as:
SUMMARY: ...
EVALUATION: ...
CONCLUSION: ...
""",
        }]
        
        result = await self.invoke_agent("mediator", messages, context)
        await self.record_result(result, context)
        
        return result
```

## Chain of Command Topology

### File: `qmcp/agentframework/topologies/chain.py`

```python
"""Chain of Command topology implementation."""

from typing import Any, ClassVar, Optional, Type

from sqlmodel import SQLModel

from qmcp.agentframework.models import (
    ChainOfCommandConfig,
    MessageType,
    TopologyType,
)

from .base import (
    AgentInvocationResult,
    BaseTopology,
    ExecutionContext,
    topology,
)


@topology
class ChainOfCommandTopology(BaseTopology):
    """
    Hierarchical chain of command topology.
    
    Tasks flow down through authority levels with optional escalation.
    
    Default levels: commander -> lieutenant -> worker
    
    Flow:
    1. Commander receives task and creates strategy
    2. Strategy decomposed into sub-tasks for lieutenants
    3. Lieutenants delegate to workers
    4. Results flow back up with aggregation
    """
    
    topology_type: ClassVar[TopologyType] = TopologyType.CHAIN_OF_COMMAND
    config_class: ClassVar[Type[SQLModel]] = ChainOfCommandConfig
    
    @property
    def chain_config(self) -> ChainOfCommandConfig:
        return self._config  # type: ignore
    
    async def _run(self, context: ExecutionContext) -> dict[str, Any]:
        """Execute chain of command."""
        task = context.input_data.get("task", "")
        levels = self.chain_config.authority_levels
        
        # Organize agents by level
        agents_by_level = self._organize_by_level(levels)
        
        # Start at top level (commander)
        top_level = levels[0]
        if top_level not in agents_by_level or not agents_by_level[top_level]:
            raise ValueError(f"No agent assigned to top level: {top_level}")
        
        # Commander creates strategy
        commander_slot = agents_by_level[top_level][0]
        strategy = await self._get_strategy(commander_slot, task, context)
        
        # Execute through hierarchy
        results = await self._execute_level(
            level_idx=0,
            levels=levels,
            agents_by_level=agents_by_level,
            task=strategy,
            context=context,
        )
        
        # Commander synthesizes final result
        final_result = await self._synthesize_results(
            commander_slot, task, results, context
        )
        
        return {
            "task": task,
            "strategy": strategy,
            "level_results": results,
            "final_result": final_result.output.get("content", ""),
        }
    
    def _organize_by_level(self, levels: list[str]) -> dict[str, list[str]]:
        """Group agent slots by authority level."""
        result = {level: [] for level in levels}
        
        for slot_name in self.agents:
            # Match slot to level (e.g., "lieutenant_1" -> "lieutenant")
            for level in levels:
                if slot_name.startswith(level):
                    result[level].append(slot_name)
                    break
        
        return result
    
    async def _get_strategy(
        self,
        slot: str,
        task: str,
        context: ExecutionContext,
    ) -> dict[str, Any]:
        """Commander creates execution strategy."""
        messages = [{
            "role": "user",
            "content": f"""You are the commander. Create a strategy for this task:

{task}

Provide:
1. Overall approach
2. Subtasks to delegate (as a list)
3. Success criteria

Format as JSON:
{{
    "approach": "...",
    "subtasks": ["task1", "task2", ...],
    "success_criteria": ["criterion1", ...]
}}
""",
        }]
        
        result = await self.invoke_agent(slot, messages, context)
        await self.record_result(result, context)
        
        # Extract structured output
        output = result.output.get("content", "{}")
        if isinstance(output, str):
            import json
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                output = {"approach": output, "subtasks": [task], "success_criteria": []}
        
        return output
    
    async def _execute_level(
        self,
        level_idx: int,
        levels: list[str],
        agents_by_level: dict[str, list[str]],
        task: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        """Execute tasks at a hierarchy level."""
        if level_idx >= len(levels):
            return {}
        
        level = levels[level_idx]
        agents = agents_by_level.get(level, [])
        
        if not agents:
            return {}
        
        subtasks = task.get("subtasks", [task.get("approach", "")])
        results = {}
        
        # Distribute subtasks among agents at this level
        for i, subtask in enumerate(subtasks):
            agent_slot = agents[i % len(agents)]
            
            if level_idx < len(levels) - 1:
                # Not at bottom level - delegate further
                sub_strategy = await self._delegate_task(
                    agent_slot, subtask, context
                )
                
                sub_results = await self._execute_level(
                    level_idx + 1,
                    levels,
                    agents_by_level,
                    sub_strategy,
                    context,
                )
                results[f"{agent_slot}:{subtask[:50]}"] = sub_results
            else:
                # Bottom level - execute directly
                execution_result = await self._execute_task(
                    agent_slot, subtask, context
                )
                results[f"{agent_slot}:{subtask[:50]}"] = execution_result
        
        return results
    
    async def _delegate_task(
        self,
        slot: str,
        subtask: str,
        context: ExecutionContext,
    ) -> dict[str, Any]:
        """Mid-level agent delegates task."""
        messages = [{
            "role": "user",
            "content": f"""You have been assigned this task to coordinate:

{subtask}

Break this down into specific work items for your team.
Format as JSON: {{"subtasks": ["item1", "item2", ...]}}
""",
        }]
        
        result = await self.invoke_agent(slot, messages, context)
        await self.record_result(result, context)
        
        # Send delegation message
        await self.send_message(
            slot, None, MessageType.REQUEST,
            {"task": subtask, "type": "delegation"},
            context,
        )
        
        output = result.output.get("content", "{}")
        if isinstance(output, str):
            import json
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                output = {"subtasks": [subtask]}
        
        return output
    
    async def _execute_task(
        self,
        slot: str,
        task: str,
        context: ExecutionContext,
    ) -> dict[str, Any]:
        """Worker executes task."""
        messages = [{
            "role": "user",
            "content": f"""Execute this task:

{task}

Provide your result and any relevant details.
""",
        }]
        
        result = await self.invoke_agent(slot, messages, context)
        await self.record_result(result, context)
        
        return {
            "task": task,
            "result": result.output.get("content", ""),
            "confidence": result.confidence,
        }
    
    async def _synthesize_results(
        self,
        commander_slot: str,
        original_task: str,
        results: dict[str, Any],
        context: ExecutionContext,
    ) -> AgentInvocationResult:
        """Commander synthesizes all results."""
        import json
        
        messages = [{
            "role": "user",
            "content": f"""Original task: {original_task}

Results from your team:
{json.dumps(results, indent=2)}

Synthesize these results into a final deliverable.
""",
        }]
        
        result = await self.invoke_agent(commander_slot, messages, context)
        await self.record_result(result, context)
        
        return result
```

## Ensemble Topology

### File: `qmcp/agentframework/topologies/ensemble.py`

```python
"""Ensemble topology implementation."""

from typing import Any, ClassVar, Type
import asyncio

from sqlmodel import SQLModel

from qmcp.agentframework.models import (
    AggregationMethod,
    EnsembleConfig,
    MessageType,
    TopologyType,
)

from .base import (
    AgentInvocationResult,
    BaseTopology,
    ExecutionContext,
    topology,
)


@topology
class EnsembleTopology(BaseTopology):
    """
    Ensemble topology for parallel aggregation.
    
    Multiple agents process the same input independently,
    then results are aggregated.
    
    Slots: Any number of agents (ensemble_0, ensemble_1, etc.)
    Optional: aggregator (for synthesis aggregation)
    
    Flow:
    1. All ensemble agents receive same input in parallel
    2. Results collected
    3. Aggregation method applied
    """
    
    topology_type: ClassVar[TopologyType] = TopologyType.ENSEMBLE
    config_class: ClassVar[Type[SQLModel]] = EnsembleConfig
    
    @property
    def ensemble_config(self) -> EnsembleConfig:
        return self._config  # type: ignore
    
    async def _run(self, context: ExecutionContext) -> dict[str, Any]:
        """Execute ensemble in parallel."""
        input_prompt = context.input_data.get("prompt", "")
        
        # Get ensemble agents (exclude aggregator)
        ensemble_slots = [
            slot for slot in self.agents
            if slot.startswith("ensemble") or slot not in ["aggregator"]
        ]
        
        if not ensemble_slots:
            raise ValueError("Ensemble requires at least one ensemble agent")
        
        # Invoke all agents in parallel
        tasks = [
            self._invoke_ensemble_member(slot, input_prompt, context)
            for slot in ensemble_slots
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failures
        successful_results = []
        failed_count = 0
        
        for slot, result in zip(ensemble_slots, results):
            if isinstance(result, Exception):
                failed_count += 1
            elif result.error:
                failed_count += 1
            else:
                successful_results.append({
                    "slot": slot,
                    "result": result,
                })
        
        # Check failure threshold
        failure_rate = failed_count / len(ensemble_slots)
        if failure_rate > self.ensemble_config.failure_threshold:
            raise RuntimeError(
                f"Too many ensemble failures: {failed_count}/{len(ensemble_slots)}"
            )
        
        # Aggregate results
        aggregated = await self._aggregate(successful_results, input_prompt, context)
        
        return {
            "input": input_prompt,
            "ensemble_size": len(ensemble_slots),
            "successful": len(successful_results),
            "failed": failed_count,
            "individual_results": [
                {
                    "slot": r["slot"],
                    "output": r["result"].output.get("content", ""),
                    "confidence": r["result"].confidence,
                }
                for r in successful_results
            ],
            "aggregated_result": aggregated,
        }
    
    async def _invoke_ensemble_member(
        self,
        slot: str,
        prompt: str,
        context: ExecutionContext,
    ) -> AgentInvocationResult:
        """Invoke a single ensemble member."""
        messages = [{
            "role": "user",
            "content": prompt,
        }]
        
        result = await self.invoke_agent(slot, messages, context)
        await self.record_result(result, context)
        
        return result
    
    async def _aggregate(
        self,
        results: list[dict[str, Any]],
        original_prompt: str,
        context: ExecutionContext,
    ) -> str:
        """Aggregate ensemble results."""
        method = self.ensemble_config.aggregation_method
        
        if method == AggregationMethod.VOTE:
            return self._vote_aggregate(results)
        elif method == AggregationMethod.CONCAT:
            return self._concat_aggregate(results)
        elif method == AggregationMethod.BEST_OF:
            return self._best_of_aggregate(results)
        elif method == AggregationMethod.SYNTHESIS:
            return await self._synthesis_aggregate(results, original_prompt, context)
        else:
            return self._concat_aggregate(results)
    
    def _vote_aggregate(self, results: list[dict[str, Any]]) -> str:
        """Majority voting aggregation."""
        from collections import Counter
        
        outputs = [r["result"].output.get("content", "") for r in results]
        # Simple voting - in production would use semantic similarity
        counter = Counter(outputs)
        return counter.most_common(1)[0][0] if counter else ""
    
    def _concat_aggregate(self, results: list[dict[str, Any]]) -> str:
        """Concatenate all results."""
        outputs = []
        for r in results:
            outputs.append(f"[{r['slot']}]: {r['result'].output.get('content', '')}")
        return "\n\n".join(outputs)
    
    def _best_of_aggregate(self, results: list[dict[str, Any]]) -> str:
        """Select highest confidence result."""
        best = max(
            results,
            key=lambda r: r["result"].confidence or 0.5
        )
        return best["result"].output.get("content", "")
    
    async def _synthesis_aggregate(
        self,
        results: list[dict[str, Any]],
        original_prompt: str,
        context: ExecutionContext,
    ) -> str:
        """Use LLM to synthesize results."""
        if "aggregator" not in self.agents:
            # Fall back to concat if no aggregator
            return self._concat_aggregate(results)
        
        # Format results for aggregator
        results_text = "\n\n".join(
            f"Response {i+1}:\n{r['result'].output.get('content', '')}"
            for i, r in enumerate(results)
        )
        
        messages = [{
            "role": "user",
            "content": f"""Original question: {original_prompt}

Multiple responses were generated:

{results_text}

Please synthesize these into a single, comprehensive response that:
1. Captures the best insights from each
2. Resolves any contradictions
3. Provides a coherent final answer
""",
        }]
        
        result = await self.invoke_agent("aggregator", messages, context)
        await self.record_result(result, context)
        
        return result.output.get("content", "")
```

## Pipeline Topology

### File: `qmcp/agentframework/topologies/pipeline.py`

```python
"""Pipeline topology implementation."""

from typing import Any, ClassVar, Type

from sqlmodel import SQLModel

from qmcp.agentframework.models import (
    MessageType,
    PipelineConfig,
    TopologyType,
)

from .base import (
    AgentInvocationResult,
    BaseTopology,
    ExecutionContext,
    topology,
)


@topology
class PipelineTopology(BaseTopology):
    """
    Sequential pipeline topology.
    
    Input flows through ordered stages, each transforming the output.
    
    Slots: Named by stage (e.g., parse, analyze, generate, review)
    
    Flow:
    1. Input enters first stage
    2. Each stage processes and passes to next
    3. Optional checkpointing between stages
    4. Final stage output is result
    """
    
    topology_type: ClassVar[TopologyType] = TopologyType.PIPELINE
    config_class: ClassVar[Type[SQLModel]] = PipelineConfig
    
    @property
    def pipeline_config(self) -> PipelineConfig:
        return self._config  # type: ignore
    
    async def _run(self, context: ExecutionContext) -> dict[str, Any]:
        """Execute pipeline stages sequentially."""
        stages = self.pipeline_config.stages
        
        if not stages:
            # Infer stages from agent slots
            stages = sorted(self.agents.keys())
        
        # Validate all stages have agents
        for stage in stages:
            if stage not in self.agents:
                raise ValueError(f"No agent for stage: {stage}")
        
        # Execute pipeline
        current_input = context.input_data
        stage_results = []
        checkpoints = {}
        
        for i, stage in enumerate(stages):
            context.round_number = i
            
            try:
                result = await self._execute_stage(
                    stage, current_input, context
                )
                
                stage_results.append({
                    "stage": stage,
                    "input": current_input,
                    "output": result.output,
                    "duration_ms": result.duration_ms,
                })
                
                # Checkpoint if configured
                if stage in self.pipeline_config.checkpoint_after:
                    checkpoints[stage] = {
                        "input": current_input,
                        "output": result.output,
                    }
                
                # Prepare input for next stage
                current_input = self._prepare_next_input(
                    current_input, result.output, stage
                )
                
            except Exception as e:
                if self.pipeline_config.retry_failed_stages:
                    result = await self._retry_stage(
                        stage, current_input, context
                    )
                    if result:
                        stage_results.append({
                            "stage": stage,
                            "input": current_input,
                            "output": result.output,
                            "retried": True,
                        })
                        current_input = self._prepare_next_input(
                            current_input, result.output, stage
                        )
                        continue
                
                raise RuntimeError(f"Pipeline failed at stage '{stage}': {e}")
        
        return {
            "stages": stages,
            "stage_results": stage_results,
            "checkpoints": checkpoints,
            "final_output": current_input,
        }
    
    async def _execute_stage(
        self,
        stage: str,
        input_data: dict[str, Any],
        context: ExecutionContext,
    ) -> AgentInvocationResult:
        """Execute a single pipeline stage."""
        # Build prompt with input data
        import json
        
        input_str = json.dumps(input_data, indent=2) if isinstance(input_data, dict) else str(input_data)
        
        messages = [{
            "role": "user",
            "content": f"""You are the '{stage}' stage of a processing pipeline.

Input:
{input_str}

Process this input according to your role and provide your output.
""",
        }]
        
        result = await self.invoke_agent(stage, messages, context)
        await self.record_result(result, context)
        
        # Send completion message
        await self.send_message(
            stage, None, MessageType.RESPONSE,
            {"stage": stage, "status": "complete"},
            context,
        )
        
        return result
    
    def _prepare_next_input(
        self,
        previous_input: dict[str, Any],
        stage_output: dict[str, Any],
        stage_name: str,
    ) -> dict[str, Any]:
        """Prepare input for next stage."""
        # Merge stage output into input for next stage
        content = stage_output.get("content", stage_output)
        
        return {
            "previous_stages": previous_input.get("previous_stages", []) + [stage_name],
            "original_input": previous_input.get("original_input", previous_input),
            "current": content,
        }
    
    async def _retry_stage(
        self,
        stage: str,
        input_data: dict[str, Any],
        context: ExecutionContext,
    ) -> AgentInvocationResult | None:
        """Retry a failed stage."""
        max_retries = self.pipeline_config.max_stage_retries
        
        for attempt in range(max_retries):
            try:
                return await self._execute_stage(stage, input_data, context)
            except Exception:
                if attempt == max_retries - 1:
                    return None
                continue
        
        return None
```

## Cross-Check Topology

### File: `qmcp/agentframework/topologies/crosscheck.py`

```python
"""Cross-Check topology implementation."""

from typing import Any, ClassVar, Type
import asyncio

from sqlmodel import SQLModel

from qmcp.agentframework.models import (
    ConsensusMethod,
    CrossCheckConfig,
    MessageType,
    TopologyType,
)

from .base import (
    AgentInvocationResult,
    BaseTopology,
    ExecutionContext,
    topology,
)


@topology
class CrossCheckTopology(BaseTopology):
    """
    Cross-check topology for independent validation.
    
    A primary agent produces output, then multiple checkers
    independently validate it.
    
    Slots:
    - primary: Produces initial output
    - checker_N: Independent validators (checker_0, checker_1, etc.)
    
    Flow:
    1. Primary agent generates output
    2. Checkers independently validate
    3. Consensus determines final result
    """
    
    topology_type: ClassVar[TopologyType] = TopologyType.CROSS_CHECK
    config_class: ClassVar[Type[SQLModel]] = CrossCheckConfig
    
    @property
    def crosscheck_config(self) -> CrossCheckConfig:
        return self._config  # type: ignore
    
    async def _run(self, context: ExecutionContext) -> dict[str, Any]:
        """Execute cross-check validation."""
        if "primary" not in self.agents:
            raise ValueError("Cross-check requires 'primary' agent")
        
        # Get checker agents
        checker_slots = [
            slot for slot in self.agents
            if slot.startswith("checker")
        ]
        
        if len(checker_slots) < 2:
            raise ValueError("Cross-check requires at least 2 checkers")
        
        prompt = context.input_data.get("prompt", "")
        
        # Primary generates output
        primary_result = await self._generate_primary(prompt, context)
        primary_output = primary_result.output.get("content", "")
        
        # Checkers validate in parallel
        check_tasks = [
            self._validate_output(slot, prompt, primary_output, context)
            for slot in checker_slots
        ]
        
        check_results = await asyncio.gather(*check_tasks)
        
        # Determine consensus
        approvals = []
        rejections = []
        
        for slot, result in zip(checker_slots, check_results):
            validation = result.output.get("validation", {})
            if validation.get("approved", False):
                approvals.append({
                    "checker": slot,
                    "confidence": result.confidence,
                    "feedback": validation.get("feedback", ""),
                })
            else:
                rejections.append({
                    "checker": slot,
                    "issues": validation.get("issues", []),
                    "feedback": validation.get("feedback", ""),
                })
        
        # Apply consensus method
        consensus = self._determine_consensus(approvals, rejections)
        
        return {
            "prompt": prompt,
            "primary_output": primary_output,
            "checkers": len(checker_slots),
            "approvals": len(approvals),
            "rejections": len(rejections),
            "approval_details": approvals,
            "rejection_details": rejections,
            "consensus": consensus,
            "final_approved": consensus["approved"],
        }
    
    async def _generate_primary(
        self,
        prompt: str,
        context: ExecutionContext,
    ) -> AgentInvocationResult:
        """Primary agent generates output."""
        messages = [{
            "role": "user",
            "content": prompt,
        }]
        
        result = await self.invoke_agent("primary", messages, context)
        await self.record_result(result, context)
        
        return result
    
    async def _validate_output(
        self,
        slot: str,
        original_prompt: str,
        output: str,
        context: ExecutionContext,
    ) -> AgentInvocationResult:
        """Checker validates primary output."""
        messages = [{
            "role": "user",
            "content": f"""You are an independent validator. Review this output:

ORIGINAL REQUEST:
{original_prompt}

OUTPUT TO VALIDATE:
{output}

Provide your validation as JSON:
{{
    "approved": true/false,
    "confidence": 0.0-1.0,
    "issues": ["issue1", "issue2", ...],
    "feedback": "detailed feedback"
}}
""",
        }]
        
        result = await self.invoke_agent(slot, messages, context)
        await self.record_result(result, context)
        
        # Parse validation result
        content = result.output.get("content", "{}")
        if isinstance(content, str):
            import json
            try:
                validation = json.loads(content)
            except json.JSONDecodeError:
                validation = {"approved": False, "issues": ["Could not parse validation"]}
        else:
            validation = content
        
        result.output["validation"] = validation
        
        return result
    
    def _determine_consensus(
        self,
        approvals: list[dict],
        rejections: list[dict],
    ) -> dict[str, Any]:
        """Determine consensus from checker results."""
        method = self.crosscheck_config.consensus_method
        total = len(approvals) + len(rejections)
        
        if method == ConsensusMethod.UNANIMOUS:
            approved = len(rejections) == 0
        elif method == ConsensusMethod.MAJORITY_VOTE:
            approved = len(approvals) > len(rejections)
        elif method == ConsensusMethod.WEIGHTED_VOTE:
            approval_weight = sum(a.get("confidence", 0.5) for a in approvals)
            rejection_weight = sum(0.5 for _ in rejections)  # Default weight
            approved = approval_weight > rejection_weight
        else:
            approved = len(approvals) >= len(rejections)
        
        # Check unanimous requirement
        if self.crosscheck_config.require_unanimous_for_approval:
            approved = approved and len(rejections) == 0
        
        return {
            "approved": approved,
            "method": method.value,
            "approval_rate": len(approvals) / total if total > 0 else 0,
            "issues": [
                issue
                for r in rejections
                for issue in r.get("issues", [])
            ],
        }
```

## Compound Topology

### File: `qmcp/agentframework/topologies/compound.py`

```python
"""Compound topology for nesting topologies."""

from typing import Any, ClassVar, Type

from sqlmodel import SQLModel

from qmcp.agentframework.models import (
    CompoundConfig,
    TopologyType,
)

from .base import (
    BaseTopology,
    ExecutionContext,
    TopologyRegistry,
    topology,
)


@topology
class CompoundTopology(BaseTopology):
    """
    Compound topology for composing other topologies.
    
    Allows nesting topologies to create complex workflows.
    
    Composition types:
    - sequential: Run sub-topologies in order
    - parallel: Run sub-topologies simultaneously
    - conditional: Route based on input/results
    """
    
    topology_type: ClassVar[TopologyType] = TopologyType.COMPOUND
    config_class: ClassVar[Type[SQLModel]] = CompoundConfig
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sub_topologies: dict[str, BaseTopology] = {}
    
    @property
    def compound_config(self) -> CompoundConfig:
        return self._config  # type: ignore
    
    async def setup(self) -> None:
        """Setup including sub-topologies."""
        await super().setup()
        
        # Load sub-topologies
        for name in self.compound_config.sub_topologies:
            sub_topo = await self._load_sub_topology(name)
            if sub_topo:
                self._sub_topologies[name] = sub_topo
    
    async def _load_sub_topology(self, name: str) -> BaseTopology | None:
        """Load a sub-topology by name."""
        from sqlmodel import select
        from qmcp.agentframework.models import Topology, TopologyMembership, AgentType
        
        # Query topology
        stmt = select(Topology).where(Topology.name == name)
        result = await self.db_session.execute(stmt)
        topology = result.scalar_one_or_none()
        
        if not topology:
            return None
        
        # Query memberships and agents
        stmt = (
            select(TopologyMembership, AgentType)
            .join(AgentType)
            .where(TopologyMembership.topology_id == topology.id)
        )
        result = await self.db_session.execute(stmt)
        
        agents = {}
        for membership, agent_type in result:
            agents[membership.slot_name] = agent_type
        
        # Create topology instance
        return TopologyRegistry.create(topology, agents, self.db_session)
    
    async def _run(self, context: ExecutionContext) -> dict[str, Any]:
        """Execute compound topology."""
        composition = self.compound_config.composition_type
        
        if composition == "sequential":
            return await self._run_sequential(context)
        elif composition == "parallel":
            return await self._run_parallel(context)
        elif composition == "conditional":
            return await self._run_conditional(context)
        else:
            raise ValueError(f"Unknown composition type: {composition}")
    
    async def _run_sequential(self, context: ExecutionContext) -> dict[str, Any]:
        """Run sub-topologies sequentially."""
        results = []
        current_input = context.input_data
        
        for name in self.compound_config.sub_topologies:
            sub_topo = self._sub_topologies.get(name)
            if not sub_topo:
                raise ValueError(f"Sub-topology not found: {name}")
            
            # Create sub-context
            sub_context = ExecutionContext(
                execution_id=context.execution_id,
                topology_id=sub_topo.topology.id,
                input_data=current_input,
                parent_context=context,
            )
            
            # Execute
            execution = await sub_topo.execute(current_input)
            
            results.append({
                "name": name,
                "output": execution.output_data,
                "status": execution.status.value,
            })
            
            # Use output as next input
            if execution.output_data:
                current_input = execution.output_data
        
        return {
            "composition": "sequential",
            "sub_results": results,
            "final_output": current_input,
        }
    
    async def _run_parallel(self, context: ExecutionContext) -> dict[str, Any]:
        """Run sub-topologies in parallel."""
        import asyncio
        
        async def run_sub(name: str):
            sub_topo = self._sub_topologies.get(name)
            if not sub_topo:
                return {"name": name, "error": "Not found"}
            
            execution = await sub_topo.execute(context.input_data)
            return {
                "name": name,
                "output": execution.output_data,
                "status": execution.status.value,
            }
        
        tasks = [run_sub(name) for name in self.compound_config.sub_topologies]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed = []
        for r in results:
            if isinstance(r, Exception):
                processed.append({"error": str(r)})
            else:
                processed.append(r)
        
        return {
            "composition": "parallel",
            "sub_results": processed,
        }
    
    async def _run_conditional(self, context: ExecutionContext) -> dict[str, Any]:
        """Run sub-topology based on condition."""
        # Get routing condition from input
        condition = context.input_data.get("route_to")
        
        if not condition:
            # Default to first
            condition = self.compound_config.sub_topologies[0]
        
        if condition not in self._sub_topologies:
            raise ValueError(f"Invalid route: {condition}")
        
        sub_topo = self._sub_topologies[condition]
        execution = await sub_topo.execute(context.input_data)
        
        return {
            "composition": "conditional",
            "selected_route": condition,
            "output": execution.output_data,
            "status": execution.status.value,
        }
```

## Package Initialization

### File: `qmcp/agentframework/topologies/__init__.py`

```python
"""Agent collaboration topologies."""

from .base import (
    AgentInvocationResult,
    BaseTopology,
    ExecutionContext,
    TopologyRegistry,
    topology,
)

from .debate import DebateTopology
from .chain import ChainOfCommandTopology
from .ensemble import EnsembleTopology
from .pipeline import PipelineTopology
from .crosscheck import CrossCheckTopology
from .compound import CompoundTopology

__all__ = [
    # Base
    "AgentInvocationResult",
    "BaseTopology",
    "ExecutionContext",
    "TopologyRegistry",
    "topology",
    # Implementations
    "DebateTopology",
    "ChainOfCommandTopology",
    "EnsembleTopology",
    "PipelineTopology",
    "CrossCheckTopology",
    "CompoundTopology",
]
```

## Usage Examples

### Creating and Executing a Debate

```python
from qmcp.agentframework.models import (
    AgentType, AgentRole, Topology, TopologyType, DebateConfig
)
from qmcp.agentframework.topologies import TopologyRegistry

# Create agents
proponent = AgentType(
    name="optimist",
    description="Argues positive perspective",
    role=AgentRole.CRITIC,
    config={"system_prompt": "You argue optimistically..."}
)

opponent = AgentType(
    name="skeptic", 
    description="Argues skeptical perspective",
    role=AgentRole.CRITIC,
    config={"system_prompt": "You are skeptical..."}
)

mediator = AgentType(
    name="judge",
    description="Synthesizes debate",
    role=AgentRole.SYNTHESIZER,
    config={"system_prompt": "You are a fair judge..."}
)

# Create topology
topology = Topology(
    name="tech_debate",
    description="Debate on technology topics",
    topology_type=TopologyType.DEBATE,
    config=DebateConfig(max_rounds=3).model_dump()
)

# Execute
agents = {
    "proponent": proponent,
    "opponent": opponent,
    "mediator": mediator,
}

debate = TopologyRegistry.create(topology, agents, session)
result = await debate.execute({
    "topic": "AI will benefit humanity",
    "question": "Will AI ultimately benefit or harm humanity?",
})
```

### Creating a Compound Topology

```python
# Create reviewed ensemble: ensemble -> cross-check
compound = Topology(
    name="reviewed_ensemble",
    description="Ensemble with review stage",
    topology_type=TopologyType.COMPOUND,
    config=CompoundConfig(
        sub_topologies=["research_ensemble", "fact_check"],
        composition_type="sequential",
    ).model_dump()
)
```

## Next Steps

1. Implement Metaflow runner (see `04-RUNNERS.md`)
2. Add comprehensive tests (see `05-TESTS.md`)
3. Integrate with QMCP API router
