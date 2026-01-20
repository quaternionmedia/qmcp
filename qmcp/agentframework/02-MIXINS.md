# QMCP Agent Framework: Mixins Specification

Note: Design reference. Implementation status is documented in docs/agentframework.md.


## Overview

Mixins provide composable capabilities that can be added to any agent type. This design follows the composition-over-inheritance principle, allowing flexible agent configuration without complex class hierarchies.

## Mixin Architecture

### Design Goals

1. **Composability**: Multiple mixins can be combined freely
2. **Independence**: Mixins don't depend on each other (unless explicitly declared)
3. **Configurability**: Each mixin has its own configuration schema
4. **Discoverability**: Mixins are registered and can be queried
5. **Lifecycle Hooks**: Mixins can hook into agent lifecycle events

### Mixin Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Lifecycle                          │
├─────────────────────────────────────────────────────────────────┤
│  1. on_create()     - Agent instance created                    │
│  2. on_start()      - Execution begins                          │
│  3. on_message()    - Message received                          │
│  4. on_invoke()     - Before LLM invocation                     │
│  5. on_response()   - After LLM response                        │
│  6. on_tool_call()  - Tool invocation                           │
│  7. on_complete()   - Execution completes                       │
│  8. on_error()      - Error occurred                            │
│  9. on_destroy()    - Agent instance destroyed                  │
└─────────────────────────────────────────────────────────────────┘
```

## Base Mixin Infrastructure

### File: `qmcp/agentframework/mixins/base.py`

```python
"""Base mixin infrastructure and registry."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional, Type, TypeVar

from pydantic import BaseModel
from sqlmodel import SQLModel


class MixinConfig(SQLModel):
    """Base configuration for all mixins."""
    enabled: bool = True


T = TypeVar("T", bound="BaseMixin")


class BaseMixin(ABC):
    """
    Abstract base class for all agent capability mixins.
    
    Mixins provide composable functionality that can be added to agents.
    Each mixin:
    - Has a unique name for registration
    - Has a configuration schema
    - Can hook into agent lifecycle events
    - Can modify agent behavior
    """
    
    # Class attributes set by subclasses
    name: ClassVar[str]
    description: ClassVar[str]
    version: ClassVar[str] = "1.0.0"
    config_class: ClassVar[Type[MixinConfig]] = MixinConfig
    dependencies: ClassVar[list[str]] = []
    
    def __init__(self, config: Optional[MixinConfig] = None):
        """Initialize mixin with optional configuration."""
        self.config = config or self.config_class()
        self._agent = None
    
    def bind(self, agent: "AgentInstance") -> None:
        """Bind mixin to an agent instance."""
        self._agent = agent
    
    @property
    def agent(self) -> "AgentInstance":
        """Get bound agent instance."""
        if self._agent is None:
            raise RuntimeError(f"Mixin {self.name} not bound to agent")
        return self._agent
    
    # ========================================================================
    # Lifecycle Hooks (override as needed)
    # ========================================================================
    
    async def on_create(self) -> None:
        """Called when agent instance is created."""
        pass
    
    async def on_start(self, execution_context: dict[str, Any]) -> None:
        """Called when execution begins."""
        pass
    
    async def on_message(self, message: "Message") -> Optional["Message"]:
        """
        Called when message received.
        
        Return modified message or None to pass through unchanged.
        """
        return None
    
    async def on_invoke(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Called before LLM invocation.
        
        Can modify the request (messages, tools, etc.).
        """
        return request
    
    async def on_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """
        Called after LLM response.
        
        Can modify the response before processing.
        """
        return response
    
    async def on_tool_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """
        Called when tool is invoked.
        
        Return modified arguments or None for pass-through.
        Can raise to prevent tool call.
        """
        return None
    
    async def on_complete(self, result: dict[str, Any]) -> dict[str, Any]:
        """
        Called when execution completes.
        
        Can modify final result.
        """
        return result
    
    async def on_error(self, error: Exception) -> Optional[Exception]:
        """
        Called when error occurs.
        
        Return modified exception or None for pass-through.
        Return a different exception to change error handling.
        """
        return None
    
    async def on_destroy(self) -> None:
        """Called when agent instance is destroyed."""
        pass


class MixinRegistry:
    """
    Global registry for agent mixins.
    
    Provides mixin discovery and instantiation.
    """
    
    _mixins: dict[str, Type[BaseMixin]] = {}
    
    @classmethod
    def register(cls, mixin_class: Type[BaseMixin]) -> Type[BaseMixin]:
        """Register a mixin class."""
        if not hasattr(mixin_class, "name"):
            raise ValueError(f"Mixin {mixin_class} must have 'name' attribute")
        cls._mixins[mixin_class.name] = mixin_class
        return mixin_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseMixin]]:
        """Get mixin class by name."""
        return cls._mixins.get(name)
    
    @classmethod
    def create(
        cls, name: str, config: Optional[dict[str, Any]] = None
    ) -> BaseMixin:
        """Create mixin instance from name and config."""
        mixin_class = cls.get(name)
        if mixin_class is None:
            raise ValueError(f"Unknown mixin: {name}")
        
        config_obj = None
        if config:
            config_obj = mixin_class.config_class(**config)
        
        return mixin_class(config=config_obj)
    
    @classmethod
    def list_all(cls) -> list[dict[str, Any]]:
        """List all registered mixins with metadata."""
        return [
            {
                "name": m.name,
                "description": m.description,
                "version": m.version,
                "config_schema": m.config_class.model_json_schema(),
                "dependencies": m.dependencies,
            }
            for m in cls._mixins.values()
        ]
    
    @classmethod
    def resolve_dependencies(cls, mixin_names: list[str]) -> list[str]:
        """
        Resolve mixin dependencies and return ordered list.
        
        Raises ValueError if circular dependency detected.
        """
        resolved = []
        seen = set()
        
        def visit(name: str, path: list[str]):
            if name in path:
                cycle = " -> ".join(path + [name])
                raise ValueError(f"Circular mixin dependency: {cycle}")
            
            if name in seen:
                return
            
            mixin_class = cls.get(name)
            if mixin_class is None:
                raise ValueError(f"Unknown mixin: {name}")
            
            for dep in mixin_class.dependencies:
                visit(dep, path + [name])
            
            seen.add(name)
            resolved.append(name)
        
        for name in mixin_names:
            visit(name, [])
        
        return resolved


def mixin(cls: Type[T]) -> Type[T]:
    """Decorator to register a mixin class."""
    return MixinRegistry.register(cls)
```

## Standard Mixins

### Tool Use Mixin

### File: `qmcp/agentframework/mixins/tool_use.py`

```python
"""Tool use capability mixin."""

from typing import Any, ClassVar, Optional, Type

from sqlmodel import SQLModel

from .base import BaseMixin, MixinConfig, mixin


class ToolUseConfig(MixinConfig):
    """Configuration for tool use mixin."""
    allowed_tools: list[str] = []  # Empty = all allowed
    denied_tools: list[str] = []
    require_confirmation: bool = False
    max_tool_calls_per_turn: int = 5
    tool_call_timeout_seconds: int = 60


@mixin
class ToolUseMixin(BaseMixin):
    """
    Provides tool invocation capabilities.
    
    This mixin allows agents to call tools registered with the QMCP server.
    It integrates with the existing /v1/tools/* endpoints.
    """
    
    name: ClassVar[str] = "tool_use"
    description: ClassVar[str] = "Enables tool invocation through QMCP server"
    version: ClassVar[str] = "1.0.0"
    config_class: ClassVar[Type[MixinConfig]] = ToolUseConfig
    
    def __init__(self, config: Optional[ToolUseConfig] = None):
        super().__init__(config)
        self._tool_call_count = 0
        self._pending_confirmations = []
    
    @property
    def tool_config(self) -> ToolUseConfig:
        return self.config  # type: ignore
    
    async def on_start(self, execution_context: dict[str, Any]) -> None:
        """Reset tool call counter at start of execution."""
        self._tool_call_count = 0
        self._pending_confirmations = []
    
    async def on_invoke(self, request: dict[str, Any]) -> dict[str, Any]:
        """Add tool definitions to LLM request."""
        if "tools" not in request:
            request["tools"] = await self._get_available_tools()
        return request
    
    async def on_tool_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Validate and potentially modify tool calls."""
        # Check if tool is allowed
        if self.tool_config.allowed_tools:
            if tool_name not in self.tool_config.allowed_tools:
                raise PermissionError(f"Tool '{tool_name}' not in allowed list")
        
        if tool_name in self.tool_config.denied_tools:
            raise PermissionError(f"Tool '{tool_name}' is denied")
        
        # Check rate limit
        self._tool_call_count += 1
        if self._tool_call_count > self.tool_config.max_tool_calls_per_turn:
            raise RuntimeError(
                f"Exceeded max tool calls ({self.tool_config.max_tool_calls_per_turn})"
            )
        
        # Handle confirmation requirement
        if self.tool_config.require_confirmation:
            self._pending_confirmations.append({
                "tool": tool_name,
                "arguments": arguments
            })
            # This would integrate with QMCP's human-in-the-loop
            # For now, pass through
        
        return None
    
    async def _get_available_tools(self) -> list[dict[str, Any]]:
        """Fetch available tools from QMCP server."""
        # This would use MCPClient to fetch tools
        # Placeholder for integration
        return []
```

### Memory Mixin

### File: `qmcp/agentframework/mixins/memory.py`

```python
"""Persistent memory capability mixin."""

from datetime import datetime
from typing import Any, ClassVar, Optional, Type

from sqlmodel import SQLModel

from .base import BaseMixin, MixinConfig, mixin


class MemoryConfig(MixinConfig):
    """Configuration for memory mixin."""
    max_memories: int = 100
    memory_decay: bool = True
    decay_half_life_hours: float = 24.0
    auto_summarize: bool = True
    summary_threshold: int = 50


class MemoryEntry(SQLModel):
    """A single memory entry."""
    content: str
    importance: float = 0.5
    created_at: datetime = datetime.utcnow()
    last_accessed: datetime = datetime.utcnow()
    access_count: int = 0
    tags: list[str] = []
    metadata: dict[str, Any] = {}


@mixin
class MemoryMixin(BaseMixin):
    """
    Provides persistent memory across invocations.
    
    Memory is stored in agent instance state and persisted to database.
    Supports importance weighting, decay, and automatic summarization.
    """
    
    name: ClassVar[str] = "memory"
    description: ClassVar[str] = "Persistent memory across agent invocations"
    version: ClassVar[str] = "1.0.0"
    config_class: ClassVar[Type[MixinConfig]] = MemoryConfig
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        super().__init__(config)
        self._memories: list[MemoryEntry] = []
    
    @property
    def memory_config(self) -> MemoryConfig:
        return self.config  # type: ignore
    
    async def on_create(self) -> None:
        """Load memories from agent state."""
        state = self.agent.state.get("memories", [])
        self._memories = [MemoryEntry(**m) for m in state]
    
    async def on_invoke(self, request: dict[str, Any]) -> dict[str, Any]:
        """Inject relevant memories into context."""
        if not self._memories:
            return request
        
        # Get relevant memories for current context
        relevant = await self._get_relevant_memories(request)
        
        if relevant:
            memory_context = self._format_memories(relevant)
            # Inject into system message or first user message
            messages = request.get("messages", [])
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] += f"\n\n{memory_context}"
            else:
                messages.insert(0, {
                    "role": "system",
                    "content": memory_context
                })
            request["messages"] = messages
        
        return request
    
    async def on_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Extract and store new memories from response."""
        # Extract important information from response
        content = response.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content if c.get("type") == "text"
            )
        
        # Auto-extract memories (simplified - would use LLM in production)
        await self._maybe_create_memory(content)
        
        return response
    
    async def on_complete(self, result: dict[str, Any]) -> dict[str, Any]:
        """Persist memories to agent state."""
        # Apply decay if enabled
        if self.memory_config.memory_decay:
            self._apply_decay()
        
        # Enforce max memories
        self._prune_memories()
        
        # Summarize if needed
        if (self.memory_config.auto_summarize and 
            len(self._memories) > self.memory_config.summary_threshold):
            await self._summarize_memories()
        
        # Save to state
        self.agent.update_state({
            "memories": [m.model_dump() for m in self._memories]
        })
        
        return result
    
    # ========================================================================
    # Memory Operations
    # ========================================================================
    
    async def remember(
        self,
        content: str,
        importance: float = 0.5,
        tags: Optional[list[str]] = None
    ) -> MemoryEntry:
        """Explicitly store a memory."""
        memory = MemoryEntry(
            content=content,
            importance=importance,
            tags=tags or []
        )
        self._memories.append(memory)
        return memory
    
    async def recall(
        self,
        query: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = 10
    ) -> list[MemoryEntry]:
        """Retrieve memories matching criteria."""
        candidates = self._memories
        
        if tags:
            candidates = [
                m for m in candidates
                if any(t in m.tags for t in tags)
            ]
        
        # Sort by importance and recency
        candidates.sort(
            key=lambda m: (m.importance, m.last_accessed),
            reverse=True
        )
        
        # Update access tracking
        for m in candidates[:limit]:
            m.last_accessed = datetime.utcnow()
            m.access_count += 1
        
        return candidates[:limit]
    
    async def forget(self, content: str) -> bool:
        """Remove a specific memory."""
        for i, m in enumerate(self._memories):
            if m.content == content:
                del self._memories[i]
                return True
        return False
    
    # ========================================================================
    # Internal Methods
    # ========================================================================
    
    async def _get_relevant_memories(
        self, request: dict[str, Any]
    ) -> list[MemoryEntry]:
        """Get memories relevant to current request."""
        # Simple implementation - get most important/recent
        # Production would use semantic search
        return await self.recall(limit=5)
    
    def _format_memories(self, memories: list[MemoryEntry]) -> str:
        """Format memories for context injection."""
        lines = ["## Relevant Memories", ""]
        for m in memories:
            lines.append(f"- {m.content}")
        return "\n".join(lines)
    
    async def _maybe_create_memory(self, content: str) -> None:
        """Potentially create memory from content."""
        # Simplified - would use LLM to extract important info
        if len(content) > 100:
            await self.remember(
                content[:200] + "...",
                importance=0.3
            )
    
    def _apply_decay(self) -> None:
        """Apply importance decay based on time."""
        import math
        now = datetime.utcnow()
        half_life = self.memory_config.decay_half_life_hours * 3600
        
        for m in self._memories:
            age_seconds = (now - m.last_accessed).total_seconds()
            decay_factor = math.exp(-0.693 * age_seconds / half_life)
            m.importance *= decay_factor
    
    def _prune_memories(self) -> None:
        """Remove lowest importance memories if over limit."""
        if len(self._memories) > self.memory_config.max_memories:
            self._memories.sort(key=lambda m: m.importance)
            excess = len(self._memories) - self.memory_config.max_memories
            self._memories = self._memories[excess:]
    
    async def _summarize_memories(self) -> None:
        """Summarize and consolidate memories."""
        # Would use LLM to summarize groups of related memories
        # Simplified: just prune more aggressively
        self._memories = self._memories[-self.memory_config.summary_threshold:]
```

### Reasoning Mixin

### File: `qmcp/agentframework/mixins/reasoning.py`

```python
"""Chain-of-thought reasoning capability mixin."""

from typing import Any, ClassVar, Optional, Type

from sqlmodel import SQLModel

from .base import BaseMixin, MixinConfig, mixin


class ReasoningConfig(MixinConfig):
    """Configuration for reasoning mixin."""
    reasoning_style: str = "step_by_step"  # step_by_step, tree_of_thought, etc.
    show_reasoning: bool = True
    max_reasoning_steps: int = 10
    require_conclusion: bool = True
    

@mixin
class ReasoningMixin(BaseMixin):
    """
    Provides structured chain-of-thought reasoning.
    
    Modifies prompts to encourage explicit reasoning and
    extracts reasoning traces from responses.
    """
    
    name: ClassVar[str] = "reasoning"
    description: ClassVar[str] = "Chain-of-thought reasoning capabilities"
    version: ClassVar[str] = "1.0.0"
    config_class: ClassVar[Type[MixinConfig]] = ReasoningConfig
    
    REASONING_PROMPTS = {
        "step_by_step": (
            "Think through this step by step:\n"
            "1. First, identify the key aspects of the problem\n"
            "2. Consider relevant information and constraints\n"
            "3. Work through the logic systematically\n"
            "4. Verify your reasoning\n"
            "5. State your conclusion clearly\n"
        ),
        "tree_of_thought": (
            "Consider multiple approaches to this problem:\n"
            "- Generate 3 distinct solution paths\n"
            "- Evaluate the pros and cons of each\n"
            "- Select the most promising approach\n"
            "- Develop it fully\n"
            "- Explain your choice\n"
        ),
        "socratic": (
            "Approach this through questioning:\n"
            "- What do we know for certain?\n"
            "- What assumptions are we making?\n"
            "- What are the implications?\n"
            "- Are there counterarguments?\n"
            "- What is the strongest conclusion?\n"
        ),
    }
    
    @property
    def reasoning_config(self) -> ReasoningConfig:
        return self.config  # type: ignore
    
    async def on_invoke(self, request: dict[str, Any]) -> dict[str, Any]:
        """Inject reasoning prompt into request."""
        style = self.reasoning_config.reasoning_style
        prompt = self.REASONING_PROMPTS.get(style, self.REASONING_PROMPTS["step_by_step"])
        
        messages = request.get("messages", [])
        
        # Add reasoning instruction to system message
        system_added = False
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] = f"{msg['content']}\n\n{prompt}"
                system_added = True
                break
        
        if not system_added:
            messages.insert(0, {"role": "system", "content": prompt})
        
        request["messages"] = messages
        return request
    
    async def on_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Extract and annotate reasoning trace."""
        content = response.get("content", "")
        
        # Extract reasoning steps (simplified)
        reasoning_trace = self._extract_reasoning(content)
        
        # Add reasoning metadata
        response["_reasoning"] = {
            "style": self.reasoning_config.reasoning_style,
            "steps": reasoning_trace,
            "step_count": len(reasoning_trace),
        }
        
        return response
    
    def _extract_reasoning(self, content: str) -> list[dict[str, str]]:
        """Extract reasoning steps from response."""
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content if c.get("type") == "text"
            )
        
        steps = []
        lines = content.split("\n")
        current_step = None
        
        for line in lines:
            line = line.strip()
            # Look for numbered steps or bullet points
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
                if current_step:
                    steps.append(current_step)
                current_step = {"type": "step", "content": line}
            elif current_step:
                current_step["content"] += " " + line
        
        if current_step:
            steps.append(current_step)
        
        return steps
```

### Human-in-the-Loop Mixin

### File: `qmcp/agentframework/mixins/human_in_loop.py`

```python
"""Human-in-the-loop capability mixin."""

from typing import Any, ClassVar, Optional, Type

from sqlmodel import SQLModel

from .base import BaseMixin, MixinConfig, mixin


class HumanInLoopConfig(MixinConfig):
    """Configuration for human-in-the-loop mixin."""
    require_approval_for: list[str] = []  # Actions requiring approval
    approval_timeout_seconds: int = 3600
    allow_modification: bool = True
    escalation_on_timeout: str = "abort"  # abort, continue, default


@mixin
class HumanInLoopMixin(BaseMixin):
    """
    Provides human approval/input integration.
    
    Integrates with QMCP's /v1/human/* endpoints for approval workflows.
    """
    
    name: ClassVar[str] = "human_in_loop"
    description: ClassVar[str] = "Human approval and input integration"
    version: ClassVar[str] = "1.0.0"
    config_class: ClassVar[Type[MixinConfig]] = HumanInLoopConfig
    dependencies: ClassVar[list[str]] = ["tool_use"]
    
    def __init__(self, config: Optional[HumanInLoopConfig] = None):
        super().__init__(config)
        self._pending_approvals: dict[str, Any] = {}
    
    @property
    def hitl_config(self) -> HumanInLoopConfig:
        return self.config  # type: ignore
    
    async def on_tool_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Intercept tool calls that require approval."""
        if tool_name not in self.hitl_config.require_approval_for:
            return None
        
        # Request human approval
        approval = await self._request_approval(
            action_type="tool_call",
            details={
                "tool": tool_name,
                "arguments": arguments,
            }
        )
        
        if not approval.get("approved"):
            raise PermissionError(
                f"Human rejected tool call: {tool_name}"
            )
        
        # Return modified arguments if human provided them
        if self.hitl_config.allow_modification:
            return approval.get("modified_arguments", arguments)
        
        return None
    
    async def request_input(
        self,
        prompt: str,
        input_type: str = "text",
        options: Optional[list[str]] = None
    ) -> str:
        """Request input from human."""
        # This would use QMCP's human request API
        request_id = f"input-{self.agent.id}-{datetime.utcnow().timestamp()}"
        
        # Create human request via QMCP client
        # await self._mcp_client.create_human_request(...)
        
        # Wait for response
        # response = await self._mcp_client.wait_for_response(...)
        
        # Placeholder
        return ""
    
    async def _request_approval(
        self, action_type: str, details: dict[str, Any]
    ) -> dict[str, Any]:
        """Request approval for an action."""
        from datetime import datetime
        
        request_id = f"approval-{self.agent.id}-{datetime.utcnow().timestamp()}"
        
        # This would integrate with QMCP human-in-the-loop API
        # For now, return auto-approval for testing
        return {"approved": True}
```

### Structured Output Mixin

### File: `qmcp/agentframework/mixins/structured_output.py`

```python
"""Structured output capability mixin."""

import json
from typing import Any, ClassVar, Optional, Type

from pydantic import BaseModel
from sqlmodel import SQLModel

from .base import BaseMixin, MixinConfig, mixin


class StructuredOutputConfig(MixinConfig):
    """Configuration for structured output mixin."""
    output_schema: Optional[dict[str, Any]] = None
    strict_validation: bool = True
    auto_repair: bool = True
    max_repair_attempts: int = 3


@mixin  
class StructuredOutputMixin(BaseMixin):
    """
    Enforces structured (JSON/schema) output from agents.
    
    Validates responses against provided schema and optionally
    repairs malformed output.
    """
    
    name: ClassVar[str] = "structured_output"
    description: ClassVar[str] = "JSON schema-constrained output generation"
    version: ClassVar[str] = "1.0.0"
    config_class: ClassVar[Type[MixinConfig]] = StructuredOutputConfig
    
    def __init__(self, config: Optional[StructuredOutputConfig] = None):
        super().__init__(config)
        self._repair_attempts = 0
    
    @property
    def output_config(self) -> StructuredOutputConfig:
        return self.config  # type: ignore
    
    async def on_invoke(self, request: dict[str, Any]) -> dict[str, Any]:
        """Add schema instructions to request."""
        if not self.output_config.output_schema:
            return request
        
        schema_instruction = self._build_schema_instruction()
        
        messages = request.get("messages", [])
        
        # Add to system message
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] += f"\n\n{schema_instruction}"
                break
        else:
            messages.insert(0, {"role": "system", "content": schema_instruction})
        
        request["messages"] = messages
        return request
    
    async def on_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Validate and potentially repair response."""
        if not self.output_config.output_schema:
            return response
        
        content = response.get("content", "")
        if isinstance(content, list):
            content = next(
                (c.get("text", "") for c in content if c.get("type") == "text"),
                ""
            )
        
        # Try to extract JSON
        json_data = self._extract_json(content)
        
        if json_data is None:
            if self.output_config.auto_repair:
                json_data = await self._repair_output(content)
            else:
                raise ValueError("Response is not valid JSON")
        
        # Validate against schema
        if self.output_config.strict_validation:
            self._validate_schema(json_data)
        
        response["_structured_output"] = json_data
        return response
    
    def _build_schema_instruction(self) -> str:
        """Build instruction for schema-conformant output."""
        schema = self.output_config.output_schema
        return (
            "You must respond with valid JSON matching this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```\n"
            "Respond ONLY with the JSON, no additional text."
        )
    
    def _extract_json(self, content: str) -> Optional[dict[str, Any]]:
        """Extract JSON from response content."""
        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in markdown blocks
        import re
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None
    
    async def _repair_output(self, content: str) -> dict[str, Any]:
        """Attempt to repair malformed output."""
        self._repair_attempts += 1
        if self._repair_attempts > self.output_config.max_repair_attempts:
            raise ValueError(
                f"Failed to repair output after {self._repair_attempts} attempts"
            )
        
        # This would re-invoke LLM with repair instructions
        # Placeholder
        raise ValueError("Could not repair output")
    
    def _validate_schema(self, data: dict[str, Any]) -> None:
        """Validate data against schema."""
        # This would use jsonschema validation
        # Placeholder - basic structure check
        schema = self.output_config.output_schema
        if not schema:
            return
        
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
```

## Mixin Package Initialization

### File: `qmcp/agentframework/mixins/__init__.py`

```python
"""Agent capability mixins."""

from .base import (
    BaseMixin,
    MixinConfig,
    MixinRegistry,
    mixin,
)

from .tool_use import ToolUseMixin, ToolUseConfig
from .memory import MemoryMixin, MemoryConfig, MemoryEntry
from .reasoning import ReasoningMixin, ReasoningConfig
from .human_in_loop import HumanInLoopMixin, HumanInLoopConfig
from .structured_output import StructuredOutputMixin, StructuredOutputConfig

# Register all standard mixins on import
# (done via @mixin decorator)

__all__ = [
    # Base
    "BaseMixin",
    "MixinConfig",
    "MixinRegistry",
    "mixin",
    # Tool Use
    "ToolUseMixin",
    "ToolUseConfig",
    # Memory
    "MemoryMixin",
    "MemoryConfig",
    "MemoryEntry",
    # Reasoning
    "ReasoningMixin",
    "ReasoningConfig",
    # Human in Loop
    "HumanInLoopMixin",
    "HumanInLoopConfig",
    # Structured Output
    "StructuredOutputMixin",
    "StructuredOutputConfig",
]
```

## Usage Examples

### Creating an Agent with Mixins

```python
from qmcp.agentframework.models import AgentType, AgentConfig, AgentCapability, AgentRole
from qmcp.agentframework.mixins import MixinRegistry

# Define capabilities
capabilities = [
    AgentCapability(
        name="tool_use",
        config={
            "allowed_tools": ["web_search", "calculator"],
            "max_tool_calls_per_turn": 3,
        }
    ),
    AgentCapability(
        name="memory",
        config={
            "max_memories": 50,
            "auto_summarize": True,
        }
    ),
    AgentCapability(
        name="reasoning",
        config={
            "reasoning_style": "step_by_step",
        }
    ),
]

# Create agent type
agent = AgentType(
    name="research_analyst",
    description="Researches topics using web search with persistent memory",
    role=AgentRole.SPECIALIST,
    config=AgentConfig(
        model="claude-sonnet-4-20250514",
        capabilities=capabilities,
        system_prompt="You are a research analyst...",
    ).model_dump()
)
```

### Instantiating Mixins for Runtime

```python
from qmcp.agentframework.mixins import MixinRegistry

# Get capabilities from agent
caps = agent.get_capabilities()

# Resolve dependencies and create instances
mixin_names = [c.name for c in caps if c.enabled]
ordered = MixinRegistry.resolve_dependencies(mixin_names)

mixins = []
for name in ordered:
    cap = next(c for c in caps if c.name == name)
    mixin = MixinRegistry.create(name, cap.config)
    mixins.append(mixin)

# Bind to agent instance
for mixin in mixins:
    mixin.bind(agent_instance)
```

## Next Steps

1. Implement additional mixins (code execution, web search, streaming)
2. Add mixin composition validation
3. Implement topology engine (see `03-TOPOLOGIES.md`)
4. Write mixin tests (see `05-TESTS.md`)
