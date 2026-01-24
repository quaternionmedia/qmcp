"""Enumerations for the QMCP agent framework."""

from __future__ import annotations

from enum import Enum


class AgentRole(str, Enum):
    """Primary function of an agent within a topology."""

    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    SPECIALIST = "specialist"
    COORDINATOR = "coordinator"
    OBSERVER = "observer"
    VALIDATOR = "validator"
    TRANSFORMER = "transformer"
    AGGREGATOR = "aggregator"
    ROUTER = "router"


class TopologyType(str, Enum):
    """Collaboration pattern for a group of agents."""

    DEBATE = "debate"
    CHAIN_OF_COMMAND = "chain"
    DELEGATION = "delegation"
    CROSS_CHECK = "crosscheck"
    ENSEMBLE = "ensemble"
    PIPELINE = "pipeline"
    COMPOUND = "compound"
    MESH = "mesh"
    STAR = "star"
    RING = "ring"


class ExecutionStatus(str, Enum):
    """Status of a topology execution."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRYING = "retrying"


class MessageType(str, Enum):
    """Type of inter-agent message."""

    REQUEST = "request"
    RESPONSE = "response"
    FEEDBACK = "feedback"
    ESCALATION = "escalation"
    BROADCAST = "broadcast"
    SYSTEM = "system"
    HEARTBEAT = "heartbeat"
    ACK = "ack"
    ERROR = "error"


class ConsensusMethod(str, Enum):
    """Methods for reaching consensus."""

    MAJORITY_VOTE = "majority"
    UNANIMOUS = "unanimous"
    WEIGHTED_VOTE = "weighted"
    MEDIATOR_DECISION = "mediator"
    FIRST_AGREEMENT = "first_agree"
    QUORUM = "quorum"
    RANKED_CHOICE = "ranked_choice"


class AggregationMethod(str, Enum):
    """Methods for aggregating ensemble outputs."""

    VOTE = "vote"
    AVERAGE = "average"
    WEIGHTED = "weighted"
    CONCAT = "concat"
    BEST_OF = "best_of"
    SYNTHESIS = "synthesis"
    MERGE = "merge"
    REDUCE = "reduce"


class Priority(str, Enum):
    """Priority levels for tasks and messages."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class SkillCategory(str, Enum):
    """Categories of agent skills."""

    REASONING = "reasoning"
    CODING = "coding"
    ANALYSIS = "analysis"
    COMMUNICATION = "communication"
    RESEARCH = "research"
    CREATIVITY = "creativity"
    MATH = "math"
    LANGUAGE = "language"
    DOMAIN_EXPERT = "domain_expert"


class CommunicationProtocol(str, Enum):
    """Protocols for agent communication."""

    SYNC = "sync"
    ASYNC = "async"
    STREAMING = "streaming"
    BATCH = "batch"
    PUBSUB = "pubsub"


class ErrorStrategy(str, Enum):
    """Strategies for handling errors."""

    FAIL_FAST = "fail_fast"
    RETRY = "retry"
    FALLBACK = "fallback"
    IGNORE = "ignore"
    ESCALATE = "escalate"
    CIRCUIT_BREAK = "circuit_break"


class ResourceType(str, Enum):
    """Types of resources agents can access."""

    MEMORY = "memory"
    TOOL = "tool"
    DATABASE = "database"
    API = "api"
    FILE = "file"
    MODEL = "model"


class EventType(str, Enum):
    """Types of system events."""

    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_FAILED = "execution_failed"
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    CHECKPOINT_CREATED = "checkpoint_created"
    RESOURCE_ACQUIRED = "resource_acquired"
    RESOURCE_RELEASED = "resource_released"


class HealthStatus(str, Enum):
    """Health status of system components."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class LogLevel(str, Enum):
    """Logging levels for agent activities."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuthScope(str, Enum):
    """Authorization scopes for agent actions."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    DELEGATE = "delegate"


class ModelProvider(str, Enum):
    """LLM model providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    AZURE = "azure"
    AWS_BEDROCK = "aws_bedrock"
    COHERE = "cohere"
    MISTRAL = "mistral"
    LOCAL = "local"
    OLLAMA = "ollama"
    VLLM = "vllm"
    CUSTOM = "custom"


class ModelFamily(str, Enum):
    """LLM model families."""

    CLAUDE = "claude"
    GPT = "gpt"
    GEMINI = "gemini"
    LLAMA = "llama"
    MISTRAL = "mistral"
    COMMAND = "command"
    PALM = "palm"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    PHI = "phi"
    CUSTOM = "custom"


class ModelTier(str, Enum):
    """Performance/capability tier of a model."""

    FLAGSHIP = "flagship"
    STANDARD = "standard"
    FAST = "fast"
    MINI = "mini"
    EMBEDDING = "embedding"
    SPECIALIZED = "specialized"


class ModelCapabilityType(str, Enum):
    """Capabilities that a model may support."""

    TEXT_GENERATION = "text_generation"
    VISION = "vision"
    TOOL_USE = "tool_use"
    FUNCTION_CALLING = "function_calling"
    CODE_EXECUTION = "code_execution"
    STRUCTURED_OUTPUT = "structured_output"
    JSON_MODE = "json_mode"
    STREAMING = "streaming"
    SYSTEM_PROMPT = "system_prompt"
    MULTI_TURN = "multi_turn"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image_generation"
    AUDIO = "audio"
    VIDEO = "video"
    THINKING = "thinking"
    CITATIONS = "citations"


class ModelAvailability(str, Enum):
    """Availability status of a model."""

    AVAILABLE = "available"
    BETA = "beta"
    PREVIEW = "preview"
    DEPRECATED = "deprecated"
    DISCONTINUED = "discontinued"
    RATE_LIMITED = "rate_limited"


class PricingUnit(str, Enum):
    """Units for model pricing."""

    PER_1K_TOKENS = "per_1k_tokens"
    PER_1M_TOKENS = "per_1m_tokens"
    PER_REQUEST = "per_request"
    PER_SECOND = "per_second"
    PER_IMAGE = "per_image"


__all__ = [
    "AgentRole",
    "TopologyType",
    "ExecutionStatus",
    "MessageType",
    "ConsensusMethod",
    "AggregationMethod",
    "Priority",
    "SkillCategory",
    "CommunicationProtocol",
    "ErrorStrategy",
    "ResourceType",
    "EventType",
    "HealthStatus",
    "LogLevel",
    "AuthScope",
    "ModelProvider",
    "ModelFamily",
    "ModelTier",
    "ModelCapabilityType",
    "ModelAvailability",
    "PricingUnit",
]
