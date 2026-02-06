# Agent Framework: Models

This document specifies the data models that form the foundation of the agent framework.

## Module Structure

```
qmcp/agentframework/models/
├── __init__.py      # Re-exports all models
├── base.py          # Utilities (utc_now, generate_uuid, validate_identifier)
├── enums.py         # All enumerations
├── configs.py       # Configuration models
├── entities.py      # Database table models
└── schemas.py       # API request/response schemas
```

## Enumerations

### AgentRole

Defines the primary function of an agent within a topology.

| Role | Value | Description |
|------|-------|-------------|
| PLANNER | `planner` | Strategic planning, task decomposition |
| EXECUTOR | `executor` | Task execution, action taking |
| REVIEWER | `reviewer` | Quality assurance, validation |
| CRITIC | `critic` | Adversarial analysis, finding flaws |
| SYNTHESIZER | `synthesizer` | Information aggregation, summarization |
| SPECIALIST | `specialist` | Domain-specific expertise |
| COORDINATOR | `coordinator` | Orchestration, delegation |
| OBSERVER | `observer` | Monitoring, logging, metrics |
| VALIDATOR | `validator` | Input/output validation |
| TRANSFORMER | `transformer` | Data transformation |
| AGGREGATOR | `aggregator` | Result aggregation |
| ROUTER | `router` | Task routing |

### TopologyType

Defines collaboration patterns for agent groups.

| Type | Value | Description |
|------|-------|-------------|
| DEBATE | `debate` | Structured argumentation |
| CHAIN_OF_COMMAND | `chain` | Hierarchical delegation |
| DELEGATION | `delegation` | Capability-based routing |
| CROSS_CHECK | `crosscheck` | Independent validation |
| ENSEMBLE | `ensemble` | Parallel aggregation |
| PIPELINE | `pipeline` | Sequential processing |
| COMPOUND | `compound` | Nested topologies |
| MESH | `mesh` | Interconnected agents |
| STAR | `star` | Hub-and-spoke pattern |
| RING | `ring` | Circular communication |

### ExecutionStatus

| Status | Value | Description |
|--------|-------|-------------|
| PENDING | `pending` | Not yet started |
| QUEUED | `queued` | Waiting in queue |
| RUNNING | `running` | Currently executing |
| PAUSED | `paused` | Awaiting human input |
| COMPLETED | `completed` | Finished successfully |
| FAILED | `failed` | Terminated with error |
| CANCELLED | `cancelled` | Manually stopped |
| TIMEOUT | `timeout` | Exceeded time limit |
| RETRYING | `retrying` | Automatic retry in progress |

### ModelProvider

LLM model providers.

| Provider | Value | Description |
|----------|-------|-------------|
| ANTHROPIC | `anthropic` | Anthropic (Claude models) |
| OPENAI | `openai` | OpenAI (GPT models) |
| GOOGLE | `google` | Google (Gemini models) |
| AZURE | `azure` | Azure OpenAI |
| AWS_BEDROCK | `aws_bedrock` | AWS Bedrock |
| COHERE | `cohere` | Cohere |
| MISTRAL | `mistral` | Mistral AI |
| LOCAL | `local` | Local/self-hosted |
| OLLAMA | `ollama` | Ollama |
| VLLM | `vllm` | vLLM |
| CUSTOM | `custom` | Custom provider |

### ModelFamily

| Family | Value | Description |
|--------|-------|-------------|
| CLAUDE | `claude` | Anthropic Claude models |
| GPT | `gpt` | OpenAI GPT models |
| GEMINI | `gemini` | Google Gemini models |
| LLAMA | `llama` | Meta Llama models |
| MISTRAL | `mistral` | Mistral models |
| COMMAND | `command` | Cohere Command models |
| QWEN | `qwen` | Alibaba Qwen models |
| DEEPSEEK | `deepseek` | DeepSeek models |
| PHI | `phi` | Microsoft Phi models |
| CUSTOM | `custom` | Custom model family |

### ModelTier

| Tier | Value | Description |
|------|-------|-------------|
| FLAGSHIP | `flagship` | Top-tier, most capable |
| STANDARD | `standard` | Balanced performance |
| FAST | `fast` | Optimized for speed |
| MINI | `mini` | Compact, efficient |
| EMBEDDING | `embedding` | Embedding models |
| SPECIALIZED | `specialized` | Task-specific models |

### ModelCapabilityType

| Capability | Value | Description |
|------------|-------|-------------|
| TEXT_GENERATION | `text_generation` | Text generation |
| VISION | `vision` | Image understanding |
| TOOL_USE | `tool_use` | Tool/function calling |
| FUNCTION_CALLING | `function_calling` | Function calling (alias) |
| CODE_EXECUTION | `code_execution` | Code execution |
| STRUCTURED_OUTPUT | `structured_output` | Structured output |
| JSON_MODE | `json_mode` | JSON output mode |
| STREAMING | `streaming` | Streaming responses |
| SYSTEM_PROMPT | `system_prompt` | System prompts |
| MULTI_TURN | `multi_turn` | Multi-turn conversations |
| EMBEDDING | `embedding` | Embedding generation |
| IMAGE_GENERATION | `image_generation` | Image generation |
| AUDIO | `audio` | Audio processing |
| VIDEO | `video` | Video processing |
| THINKING | `thinking` | Extended thinking |
| CITATIONS | `citations` | Citation support |

## Configuration Models

### LLM Model Configuration

#### ModelConfig

Complete configuration for an LLM model.

```python
class ModelConfig(SQLModel):
    model_id: str                    # e.g., "claude-sonnet-4-20250514"
    provider: ModelProvider          # anthropic, openai, google, etc.
    family: ModelFamily              # claude, gpt, gemini, etc.
    tier: ModelTier                  # flagship, standard, fast, mini
    display_name: str | None         # Human-readable name
    description: str | None
    version: str | None
    availability: ModelAvailability  # available, beta, deprecated

    # Nested configurations
    sampling: ModelSamplingParams
    limits: ModelLimits
    capabilities: ModelCapabilities
    pricing: ModelPricing
    endpoint: ModelEndpoint
    fallback: ModelFallbackConfig

    # Inference settings
    max_tokens: int = 4096
    timeout_seconds: int = 300
    stream: bool = False

    # Custom data
    tags: list[str] = []
    extra: dict[str, Any] = {}  # Custom metadata
```

#### ModelSamplingParams

```python
class ModelSamplingParams(SQLModel):
    temperature: float = 0.7         # 0.0-2.0
    top_p: float | None = None       # Nucleus sampling
    top_k: int | None = None         # Top-k sampling
    frequency_penalty: float = 0.0   # -2.0 to 2.0
    presence_penalty: float = 0.0    # -2.0 to 2.0
    stop_sequences: list[str] = []
    seed: int | None = None          # For reproducibility
```

#### ModelLimits

```python
class ModelLimits(SQLModel):
    context_window: int = 200000     # Max context length
    max_output_tokens: int = 4096    # Max output per request
    requests_per_minute: int | None  # Rate limit
    tokens_per_minute: int | None    # Token rate limit
    tokens_per_day: int | None       # Daily token limit
    concurrent_requests: int | None  # Max concurrency
    max_images_per_request: int | None
```

#### ModelCapabilities

```python
class ModelCapabilities(SQLModel):
    supported: list[ModelCapabilityType] = [
        ModelCapabilityType.TEXT_GENERATION,
        ModelCapabilityType.MULTI_TURN,
        ModelCapabilityType.SYSTEM_PROMPT,
    ]
    vision_formats: list[str] = []   # png, jpg, gif, webp
    audio_formats: list[str] = []
    max_tool_definitions: int | None
    supports_parallel_tool_calls: bool = False
    supports_tool_choice: bool = False
    structured_output_formats: list[str] = []  # json_schema, regex
```

#### ModelPricing

```python
class ModelPricing(SQLModel):
    input_cost: float = 0.0          # Cost per unit
    output_cost: float = 0.0
    unit: PricingUnit = PricingUnit.PER_1M_TOKENS
    cache_read_cost: float | None    # Prompt caching
    cache_write_cost: float | None
    image_cost: float | None
    currency: str = "USD"
```

#### ModelEndpoint

```python
class ModelEndpoint(SQLModel):
    base_url: str | None             # Custom API URL
    api_version: str | None
    deployment_name: str | None      # Azure deployment
    region: str | None               # Cloud region
    headers: dict[str, str] = {}     # Additional headers
```

#### ModelFallbackConfig

```python
class ModelFallbackConfig(SQLModel):
    enabled: bool = True
    fallback_models: list[str] = []  # Ordered fallback list
    trigger_on_rate_limit: bool = True
    trigger_on_timeout: bool = True
    trigger_on_error: bool = False
    max_fallback_attempts: int = 2
```

#### ModelRegistryEntry

Compact entry for the model registry.

```python
class ModelRegistryEntry(SQLModel):
    model_id: str
    provider: ModelProvider
    family: ModelFamily
    tier: ModelTier
    display_name: str
    availability: ModelAvailability
    context_window: int
    max_output_tokens: int
    supports_vision: bool = False
    supports_tools: bool = False
    supports_streaming: bool = True
    input_cost_per_1m: float = 0.0
    output_cost_per_1m: float = 0.0
```

### Models Registry

Pre-configured model instances for common LLM models. Use these instead of string literals or manual configuration.

```python
from qmcp.agentframework.models import Models, ModelProvider, ModelTier

# Anthropic Claude models
Models.CLAUDE_OPUS_4      # Flagship - complex reasoning, extended thinking
Models.CLAUDE_SONNET_4    # Standard - balanced performance (recommended)
Models.CLAUDE_HAIKU_35    # Fast - high volume tasks

# OpenAI GPT models
Models.GPT_4O             # Flagship
Models.GPT_4O_MINI        # Fast and affordable

# Google Gemini models
Models.GEMINI_PRO         # Flagship with 2M context
Models.GEMINI_FLASH       # Fast with 1M context

# Local models (Ollama)
Models.LLAMA_3_70B        # Llama 3 70B
Models.LLAMA_3_8B         # Llama 3 8B
Models.MISTRAL_7B         # Mistral 7B

# Query methods
Models.get("claude-sonnet-4-20250514")      # Lookup by ID
Models.by_provider(ModelProvider.ANTHROPIC)  # All Anthropic models
Models.by_tier(ModelTier.FLAGSHIP)           # All flagship models
Models.list_all()                            # All registered models
```

### AgentConfig

Primary configuration for an agent type. Supports both simple model ID strings and full ModelConfig objects.

```python
class AgentConfig(SQLModel):
    # Model configuration - simple or full
    model: str = "claude-sonnet-4-20250514"  # Simple model ID
    model_config_obj: ModelConfig | None     # Full config (overrides model)

    # Sampling (when using simple model string)
    temperature: float = 0.7  # 0.0-2.0
    top_p: float | None = None
    max_tokens: int = 4096

    # Agent settings
    system_prompt: str | None = None
    output_format: str | None = None
    max_retries: int = 3
    timeout_seconds: int = 300
    max_tool_calls: int = 10

    # Nested configurations
    capabilities: list[AgentCapability] = []
    skills: list[SkillConfig] = []
    retry: RetryConfig = RetryConfig()
    timeouts: TimeoutConfig = TimeoutConfig()
    logging: LoggingConfig = LoggingConfig()
    security: SecurityConfig = SecurityConfig()
    resource_limits: ResourceLimits = ResourceLimits()
    communication: CommunicationConfig = CommunicationConfig()
```

### Topology Configurations

Each topology type has a corresponding configuration model:

- `DebateConfig` - max_rounds, consensus_method, convergence_threshold
- `ChainOfCommandConfig` - authority_levels, escalation_threshold
- `DelegationConfig` - routing_strategy, load_balance
- `CrossCheckConfig` - num_checkers, consensus_method
- `EnsembleConfig` - aggregation_method, diversity_weight
- `PipelineConfig` - stages, checkpoint_after
- `CompoundConfig` - sub_topologies, composition_type
- `MeshConfig` - connection_density, bidirectional
- `StarConfig` - hub_agent_name, parallel_spokes
- `RingConfig` - direction, max_iterations

## Database Entities

### AgentType

Persistent agent type definition.

```python
class AgentType(SQLModel, table=True):
    __tablename__ = "agent_types"

    id: int | None = Field(primary_key=True)
    name: str = Field(index=True, unique=True)
    description: str
    role: AgentRole
    version: str = "1.0.0"
    config: dict[str, Any]
    created_at: datetime
    updated_at: datetime
```

### Other Entities

| Entity | Table | Purpose |
|--------|-------|---------|
| AgentInstance | `agent_instances` | Running instance of agent type |
| Topology | `topologies` | Topology definition |
| TopologyMembership | `topology_memberships` | Agent-topology links |
| Execution | `executions` | Execution records |
| Message | `messages` | Inter-agent messages |
| Result | `results` | Agent results |
| AgentSkill | `agent_skills` | Skill mappings |
| ResourceAllocation | `resource_allocations` | Resource tracking |
| Checkpoint | `checkpoints` | Recovery checkpoints |
| AuditLog | `audit_logs` | Action audit trail |
| WorkflowTemplate | `workflow_templates` | Reusable templates |
| MetricRecord | `metric_records` | Execution metrics |
| ToolInvocation | `tool_invocations` | Tool call records |

## API Schemas

### CRUD Schemas

Each entity has corresponding Create, Read, Update, and Summary schemas:

```python
# Agent Type
AgentTypeCreate   # name, description, role, config
AgentTypeUpdate   # description?, role?, version?, config?
AgentTypeRead     # Full agent type data
AgentTypeSummary  # Compact listing view

# Similar patterns for:
# - Topology
# - Execution
# - WorkflowTemplate
# - etc.
```

### Query Schemas

```python
PaginationParams  # page, page_size
FilterParams      # created_after, created_before, status, priority
SortParams        # sort_by, sort_order
PaginatedResponse # items, total, page, page_size, total_pages
```

### Health Schemas

```python
HealthCheck   # status, version, uptime_seconds, active counts
SystemStats   # execution counts, averages, totals
```

## Usage Examples

### Using the Models Registry (Recommended)

```python
from qmcp.agentframework.models import (
    AgentType, AgentRole, AgentConfig, AgentCapability, Models,
)

# Use pre-configured model from registry - no string literals!
agent = AgentType(
    name="research_analyst",
    description="Researches and analyzes topics",
    role=AgentRole.SPECIALIST,
    config=AgentConfig(
        model_config_obj=Models.CLAUDE_SONNET_4,  # Pre-configured
        temperature=0.7,
        capabilities=[
            AgentCapability(name="tool_use"),
            AgentCapability(name="memory"),
        ],
    ).model_dump(),
)
```

### Creating a Custom Model Configuration

For models not in the registry, or when you need custom configuration:

```python
from qmcp.agentframework.models import (
    AgentType, AgentRole, AgentConfig, AgentCapability,
    ModelConfig, ModelProvider, ModelFamily, ModelTier,
    ModelCapabilityType, ModelCapabilities,
)

# Custom model configuration
model = ModelConfig(
    model_id="my-fine-tuned-model",
    provider=ModelProvider.CUSTOM,
    family=ModelFamily.CLAUDE,
    tier=ModelTier.FLAGSHIP,
    capabilities=ModelCapabilities(
        supported=[
            ModelCapabilityType.TEXT_GENERATION,
            ModelCapabilityType.VISION,
            ModelCapabilityType.TOOL_USE,
            ModelCapabilityType.THINKING,
        ],
    ),
)

# Agent with full model config
agent = AgentType(
    name="code_reviewer",
    description="Expert code reviewer",
    role=AgentRole.REVIEWER,
    config=AgentConfig(
        model_config_obj=model,
        system_prompt="You are an expert code reviewer...",
        capabilities=[
            AgentCapability(name="tool_use", config={"allowed_tools": ["read_file"]}),
            AgentCapability(name="reasoning", config={"reasoning_style": "step_by_step"}),
        ],
    ).model_dump(),
)
```

### Persisting to Database

```python
async with session:
    session.add(agent)
    await session.commit()
    await session.refresh(agent)
    print(f"Created agent ID: {agent.id}")
```

### Querying Agents

```python
from sqlmodel import select

stmt = select(AgentType).where(AgentType.role == AgentRole.SPECIALIST)
result = await session.execute(stmt)
agents = result.scalars().all()
```
