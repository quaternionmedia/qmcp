# Agent Framework: Mixins

Mixins provide composable capabilities that can be added to any agent type. This follows composition-over-inheritance, allowing flexible agent configuration.

## Architecture

### Mixin Lifecycle

```
Agent Lifecycle Hooks:
1. on_create()     - Agent instance created
2. on_start()      - Execution begins
3. on_message()    - Message received
4. on_invoke()     - Before LLM invocation
5. on_response()   - After LLM response
6. on_tool_call()  - Tool invocation
7. on_complete()   - Execution completes
8. on_error()      - Error occurred
9. on_destroy()    - Agent instance destroyed
```

### Base Classes

```python
class MixinConfig(SQLModel):
    """Base configuration for all mixins."""
    enabled: bool = True

class BaseMixin(ABC):
    """Abstract base for all capability mixins."""
    name: ClassVar[str]
    description: ClassVar[str]
    version: ClassVar[str] = "1.0.0"
    config_class: ClassVar[Type[MixinConfig]]
    dependencies: ClassVar[list[str]] = []
```

## Available Mixins

### ToolUseMixin

Enables tool invocation through the QMCP server.

```python
class ToolUseConfig(MixinConfig):
    allowed_tools: list[str] = []     # Empty = all allowed
    denied_tools: list[str] = []
    require_confirmation: bool = False
    max_tool_calls_per_turn: int = 5
    tool_call_timeout_seconds: int = 60
```

**Features:**
- Tool allowlist/denylist filtering
- Rate limiting per turn
- Human confirmation integration
- Automatic tool definition injection

### MemoryMixin

Provides persistent memory across invocations.

```python
class MemoryConfig(MixinConfig):
    max_memories: int = 100
    memory_decay: bool = True
    decay_half_life_hours: float = 24.0
    auto_summarize: bool = True
    summary_threshold: int = 50
```

**Features:**
- Importance-weighted storage
- Time-based decay
- Tag-based retrieval
- Automatic summarization
- Context injection

**Methods:**
```python
await mixin.remember(content, importance=0.5, tags=[])
await mixin.recall(query=None, tags=None, limit=10)
await mixin.forget(content)
```

### ReasoningMixin

Provides structured chain-of-thought reasoning.

```python
class ReasoningConfig(MixinConfig):
    reasoning_style: str = "step_by_step"  # step_by_step, tree_of_thought, socratic
    show_reasoning: bool = True
    max_reasoning_steps: int = 10
    require_conclusion: bool = True
```

**Reasoning Styles:**
- `step_by_step` - Linear step-by-step analysis
- `tree_of_thought` - Multiple solution paths
- `socratic` - Question-based exploration

### HumanInLoopMixin

Integrates with QMCP's human approval system.

```python
class HumanInLoopConfig(MixinConfig):
    require_approval_for: list[str] = []  # Actions requiring approval
    approval_timeout_seconds: int = 3600
    allow_modification: bool = True
    escalation_on_timeout: str = "abort"  # abort, continue, default
```

**Dependencies:** `tool_use`

**Methods:**
```python
await mixin.request_input(prompt, input_type="text", options=None)
```

### StructuredOutputMixin

Enforces JSON schema-constrained output.

```python
class StructuredOutputConfig(MixinConfig):
    output_schema: dict[str, Any] | None = None
    strict_validation: bool = True
    auto_repair: bool = True
    max_repair_attempts: int = 3
```

**Features:**
- Schema instruction injection
- JSON extraction from responses
- Automatic repair attempts
- Validation against schema

## Mixin Registry

```python
from qmcp.agentframework.mixins import MixinRegistry

# List all registered mixins
mixins = MixinRegistry.list_all()

# Create mixin instance
mixin = MixinRegistry.create("tool_use", {"max_tool_calls_per_turn": 3})

# Resolve dependencies
ordered = MixinRegistry.resolve_dependencies(["human_in_loop", "memory"])
# Returns: ["tool_use", "human_in_loop", "memory"]
```

## Usage Example

### Configuring Agent Capabilities

```python
from qmcp.agentframework import (
    AgentType, AgentRole, AgentConfig, AgentCapability,
    Models,  # Pre-configured model registry
)

# Use pre-configured model with tool use capability
agent = AgentType(
    name="research_assistant",
    description="Researches topics with memory",
    role=AgentRole.SPECIALIST,
    config=AgentConfig(
        model_config_obj=Models.CLAUDE_SONNET_4,  # Already configured for tool use
        capabilities=[
            AgentCapability(
                name="tool_use",
                config={
                    "allowed_tools": ["web_search", "calculator"],
                    "max_tool_calls_per_turn": 5,
                },
            ),
            AgentCapability(
                name="memory",
                config={
                    "max_memories": 50,
                    "auto_summarize": True,
                },
            ),
            AgentCapability(
                name="reasoning",
                config={
                    "reasoning_style": "step_by_step",
                },
            ),
        ],
    ).model_dump(),
)
```

### Loading Mixins at Runtime

```python
from qmcp.agentframework.mixins import MixinRegistry

# Get capabilities from agent
caps = agent.get_capabilities()
mixin_names = [c.name for c in caps if c.enabled]

# Resolve dependencies
ordered = MixinRegistry.resolve_dependencies(mixin_names)

# Create and bind mixins
mixins = []
for name in ordered:
    cap = next(c for c in caps if c.name == name)
    mixin = MixinRegistry.create(name, cap.config)
    mixin.bind(agent_instance)
    await mixin.on_create()
    mixins.append(mixin)
```

## Creating Custom Mixins

```python
from qmcp.agentframework.mixins import BaseMixin, MixinConfig, mixin

class MyMixinConfig(MixinConfig):
    custom_setting: str = "default"

@mixin
class MyMixin(BaseMixin):
    name = "my_mixin"
    description = "Custom capability mixin"
    version = "1.0.0"
    config_class = MyMixinConfig
    dependencies = []  # Other mixins this depends on

    async def on_invoke(self, request: dict) -> dict:
        # Modify request before LLM call
        return request

    async def on_response(self, response: dict) -> dict:
        # Process response after LLM call
        return response
```
