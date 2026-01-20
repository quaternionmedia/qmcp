"""Capability mixins for the QMCP agent framework."""

from __future__ import annotations

from abc import ABC
from datetime import UTC, datetime
from typing import Any, ClassVar, TypeVar

from sqlmodel import Field, SQLModel


class MixinConfig(SQLModel):
    """Base configuration for all mixins."""

    enabled: bool = True


def utc_now() -> datetime:
    return datetime.now(UTC)


T = TypeVar("T", bound="BaseMixin")


class BaseMixin(ABC):
    """Abstract base class for all agent capability mixins."""

    name: ClassVar[str]
    description: ClassVar[str]
    version: ClassVar[str] = "1.0.0"
    config_class: ClassVar[type[MixinConfig]] = MixinConfig
    dependencies: ClassVar[list[str]] = []

    def __init__(self, config: MixinConfig | None = None):
        self.config = config or self.config_class()
        self._agent = None

    def bind(self, agent: Any) -> None:
        """Bind mixin to an agent instance."""
        self._agent = agent

    @property
    def agent(self) -> Any:
        if self._agent is None:
            raise RuntimeError(f"Mixin {self.name} not bound to agent")
        return self._agent

    async def on_create(self) -> None:
        """Called when agent instance is created."""

    async def on_start(self, execution_context: dict[str, Any]) -> None:
        """Called when execution begins."""

    async def on_message(self, message: Any) -> Any | None:
        """Called when a message is received."""
        return None

    async def on_invoke(self, request: dict[str, Any]) -> dict[str, Any]:
        """Called before LLM invocation."""
        return request

    async def on_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Called after LLM response."""
        return response

    async def on_tool_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Called when a tool is invoked."""
        return None

    async def on_complete(self, result: dict[str, Any]) -> dict[str, Any]:
        """Called when execution completes."""
        return result

    async def on_error(self, error: Exception) -> Exception | None:
        """Called when an error occurs."""
        return None

    async def on_destroy(self) -> None:
        """Called when agent instance is destroyed."""


class MixinRegistry:
    """Global registry for agent mixins."""

    _mixins: dict[str, type[BaseMixin]] = {}

    @classmethod
    def register(cls, mixin_class: type[BaseMixin]) -> type[BaseMixin]:
        if not hasattr(mixin_class, "name"):
            raise ValueError(f"Mixin {mixin_class} must have 'name' attribute")
        cls._mixins[mixin_class.name] = mixin_class
        return mixin_class

    @classmethod
    def get(cls, name: str) -> type[BaseMixin] | None:
        return cls._mixins.get(name)

    @classmethod
    def create(cls, name: str, config: dict[str, Any] | None = None) -> BaseMixin:
        mixin_class = cls.get(name)
        if mixin_class is None:
            raise ValueError(f"Unknown mixin: {name}")

        config_obj = None
        if config:
            config_obj = mixin_class.config_class(**config)

        return mixin_class(config=config_obj)

    @classmethod
    def list_all(cls) -> list[dict[str, Any]]:
        return [
            {
                "name": mixin.name,
                "description": mixin.description,
                "version": mixin.version,
                "config_schema": mixin.config_class.model_json_schema(),
                "dependencies": mixin.dependencies,
            }
            for mixin in cls._mixins.values()
        ]

    @classmethod
    def resolve_dependencies(cls, mixin_names: list[str]) -> list[str]:
        resolved: list[str] = []
        seen: set[str] = set()

        def visit(name: str, path: list[str]) -> None:
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


def mixin(cls: type[T]) -> type[T]:
    """Decorator to register a mixin class."""
    return MixinRegistry.register(cls)


class ToolUseConfig(MixinConfig):
    """Configuration for tool use mixin."""

    allowed_tools: list[str] = Field(default_factory=list)
    denied_tools: list[str] = Field(default_factory=list)
    require_confirmation: bool = False
    max_tool_calls_per_turn: int = 5
    tool_call_timeout_seconds: int = 60


@mixin
class ToolUseMixin(BaseMixin):
    """Enables tool invocation through QMCP server."""

    name: ClassVar[str] = "tool_use"
    description: ClassVar[str] = "Enables tool invocation through QMCP server"
    config_class: ClassVar[type[MixinConfig]] = ToolUseConfig

    def __init__(self, config: ToolUseConfig | None = None):
        super().__init__(config)
        self._tool_call_count = 0

    @property
    def tool_config(self) -> ToolUseConfig:
        return self.config  # type: ignore[return-value]

    async def on_start(self, execution_context: dict[str, Any]) -> None:
        self._tool_call_count = 0

    async def on_tool_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any] | None:
        if self.tool_config.allowed_tools:
            if tool_name not in self.tool_config.allowed_tools:
                raise PermissionError(f"Tool '{tool_name}' not in allowed list")

        if tool_name in self.tool_config.denied_tools:
            raise PermissionError(f"Tool '{tool_name}' is denied")

        self._tool_call_count += 1
        if self._tool_call_count > self.tool_config.max_tool_calls_per_turn:
            raise RuntimeError(
                f"Exceeded max tool calls ({self.tool_config.max_tool_calls_per_turn})"
            )

        return None


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
    created_at: datetime = Field(default_factory=utc_now)
    last_accessed: datetime = Field(default_factory=utc_now)
    access_count: int = 0
    tags: list[str] = Field(default_factory=list)
    metadata_: dict[str, Any] = Field(default_factory=dict)


@mixin
class MemoryMixin(BaseMixin):
    """Persistent memory across agent invocations."""

    name: ClassVar[str] = "memory"
    description: ClassVar[str] = "Persistent memory across agent invocations"
    config_class: ClassVar[type[MixinConfig]] = MemoryConfig

    def __init__(self, config: MemoryConfig | None = None):
        super().__init__(config)
        self._memories: list[MemoryEntry] = []

    @property
    def memory_config(self) -> MemoryConfig:
        return self.config  # type: ignore[return-value]

    async def remember(
        self,
        content: str,
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        memory = MemoryEntry(
            content=content,
            importance=importance,
            tags=tags or [],
        )
        self._memories.append(memory)
        return memory

    async def recall(
        self,
        query: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        candidates = self._memories

        if tags:
            candidates = [mem for mem in candidates if any(tag in mem.tags for tag in tags)]

        candidates.sort(key=lambda mem: (mem.importance, mem.last_accessed), reverse=True)

        for mem in candidates[:limit]:
            mem.last_accessed = utc_now()
            mem.access_count += 1

        return candidates[:limit]

    async def forget(self, content: str) -> bool:
        for index, mem in enumerate(self._memories):
            if mem.content == content:
                del self._memories[index]
                return True
        return False

    def _apply_decay(self) -> None:
        import math

        now = utc_now()
        half_life = self.memory_config.decay_half_life_hours * 3600

        for mem in self._memories:
            age_seconds = (now - mem.last_accessed).total_seconds()
            decay_factor = math.exp(-0.693 * age_seconds / half_life)
            mem.importance *= decay_factor

    def _prune_memories(self) -> None:
        if len(self._memories) > self.memory_config.max_memories:
            self._memories.sort(key=lambda mem: mem.importance)
            excess = len(self._memories) - self.memory_config.max_memories
            self._memories = self._memories[excess:]


class ReasoningConfig(MixinConfig):
    """Configuration for reasoning mixin."""

    reasoning_style: str = "step_by_step"
    show_reasoning: bool = True
    max_reasoning_steps: int = 10
    require_conclusion: bool = True


@mixin
class ReasoningMixin(BaseMixin):
    """Chain-of-thought reasoning capabilities."""

    name: ClassVar[str] = "reasoning"
    description: ClassVar[str] = "Chain-of-thought reasoning capabilities"
    config_class: ClassVar[type[MixinConfig]] = ReasoningConfig

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
        return self.config  # type: ignore[return-value]

    async def on_invoke(self, request: dict[str, Any]) -> dict[str, Any]:
        style = self.reasoning_config.reasoning_style
        prompt = self.REASONING_PROMPTS.get(style, self.REASONING_PROMPTS["step_by_step"])

        messages = request.get("messages", [])

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
        content = response.get("content", "")
        reasoning_trace = self._extract_reasoning(content)

        response["_reasoning"] = {
            "style": self.reasoning_config.reasoning_style,
            "steps": reasoning_trace,
            "step_count": len(reasoning_trace),
        }

        return response

    def _extract_reasoning(self, content: str) -> list[dict[str, str]]:
        if isinstance(content, list):
            content = " ".join(
                chunk.get("text", "") for chunk in content if chunk.get("type") == "text"
            )

        steps: list[dict[str, str]] = []
        lines = content.split("\n")
        current_step: dict[str, str] | None = None

        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith(("-", "*"))):
                if current_step:
                    steps.append(current_step)
                current_step = {"type": "step", "content": line}
            elif current_step:
                current_step["content"] += " " + line

        if current_step:
            steps.append(current_step)

        return steps


class HumanInLoopConfig(MixinConfig):
    """Configuration for human-in-the-loop mixin."""

    require_approval_for: list[str] = Field(default_factory=list)
    approval_timeout_seconds: int = 3600
    allow_modification: bool = True
    escalation_on_timeout: str = "abort"


@mixin
class HumanInLoopMixin(BaseMixin):
    """Human approval and input integration."""

    name: ClassVar[str] = "human_in_loop"
    description: ClassVar[str] = "Human approval and input integration"
    config_class: ClassVar[type[MixinConfig]] = HumanInLoopConfig
    dependencies: ClassVar[list[str]] = ["tool_use"]

    @property
    def hitl_config(self) -> HumanInLoopConfig:
        return self.config  # type: ignore[return-value]

    async def on_tool_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any] | None:
        if tool_name not in self.hitl_config.require_approval_for:
            return None

        # TODO: integrate with QMCP human-in-the-loop API.
        return None


class StructuredOutputConfig(MixinConfig):
    """Configuration for structured output mixin."""

    output_schema: dict[str, Any] | None = None
    strict_validation: bool = True
    auto_repair: bool = True
    max_repair_attempts: int = 3


@mixin
class StructuredOutputMixin(BaseMixin):
    """JSON schema-constrained output generation."""

    name: ClassVar[str] = "structured_output"
    description: ClassVar[str] = "JSON schema-constrained output generation"
    config_class: ClassVar[type[MixinConfig]] = StructuredOutputConfig

    def __init__(self, config: StructuredOutputConfig | None = None):
        super().__init__(config)
        self._repair_attempts = 0

    @property
    def output_config(self) -> StructuredOutputConfig:
        return self.config  # type: ignore[return-value]

    async def on_invoke(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self.output_config.output_schema:
            return request

        schema_instruction = self._build_schema_instruction()
        messages = request.get("messages", [])

        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] += f"\n\n{schema_instruction}"
                break
        else:
            messages.insert(0, {"role": "system", "content": schema_instruction})

        request["messages"] = messages
        return request

    async def on_response(self, response: dict[str, Any]) -> dict[str, Any]:
        if not self.output_config.output_schema:
            return response

        content = response.get("content", "")
        if isinstance(content, list):
            content = next(
                (chunk.get("text", "") for chunk in content if chunk.get("type") == "text"),
                "",
            )

        json_data = self._extract_json(content)

        if json_data is None:
            if self.output_config.auto_repair:
                raise ValueError("Could not extract JSON and repair not implemented")
            raise ValueError("Response is not valid JSON")

        if self.output_config.strict_validation:
            self._validate_schema(json_data)

        response["_structured_output"] = json_data
        return response

    def _build_schema_instruction(self) -> str:
        import json

        schema = self.output_config.output_schema
        return (
            "You must respond with valid JSON matching this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```\n"
            "Respond ONLY with the JSON, no additional text."
        )

    def _extract_json(self, content: str) -> dict[str, Any] | None:
        import json
        import re

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _validate_schema(self, data: dict[str, Any]) -> None:
        schema = self.output_config.output_schema
        if not schema:
            return

        required = schema.get("required", [])
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")


__all__ = [
    "BaseMixin",
    "MixinConfig",
    "MixinRegistry",
    "mixin",
    "ToolUseMixin",
    "ToolUseConfig",
    "MemoryMixin",
    "MemoryConfig",
    "MemoryEntry",
    "ReasoningMixin",
    "ReasoningConfig",
    "HumanInLoopMixin",
    "HumanInLoopConfig",
    "StructuredOutputMixin",
    "StructuredOutputConfig",
]
