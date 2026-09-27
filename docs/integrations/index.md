# Integrations

QMCP integrates with external libraries to provide a complete agent development stack while avoiding duplication of maintained functionality.

## Philosophy

QMCP follows a "best of both worlds" approach:

1. **Use PydanticAI** for agent runtime, typed tools, and output validation
2. **Keep QMCP** for audit trails, HITL, model metadata, and multi-agent topologies

This hybrid architecture leverages actively-maintained libraries while preserving QMCP's unique production features.

## Available Integrations

### [PydanticAI](pydantic-ai.md)

LLM agent framework for Python. Provides:
- Agent execution with dependency injection
- Typed tool definitions with auto-generated schemas
- Output validation against Pydantic models
- Streaming and retry support
- Message history management

**Status**: Production-ready

```python
from qmcp.integrations.pydantic_ai import create_agent, QMCPToolset

agent = create_agent(
    Models.CLAUDE_SONNET_4,
    toolsets=[QMCPToolset()],
)
```

### [Voice](voice.md)

Answers a pending human-in-the-loop request by speech instead of by typing.
Provides:
- `VoiceApprovalLoop`, speaking a prompt and submitting the transcribed reply
- A yes/no parse tolerant of transcript punctuation and casing
- Option selection that reads a request's `options` rather than their order
- A retry budget, so an ambiguous answer is re-asked rather than guessed

Speech-to-text is delegated over HTTP to a speech engine. The
[vox](https://github.com/quaternionmedia/vox) seam states the contract and
names no engine; `vox.adapters.joe` names the one this uses by default. vox is
vendored as a submodule at `./vox`.

**Status**: Works end to end; `run_forever` is not bounded on an idle queue.

```python
from qmcp.integrations.voice import VoiceApprovalLoop
from vox import HttpSTT
from vox.adapters import JOE
from vox.adapters.pyttsx3 import Pyttsx3TTS

stt = HttpSTT("http://127.0.0.1:8000", contract=JOE)
VoiceApprovalLoop(stt=stt, tts=Pyttsx3TTS()).run_once("deploy-001")
```

## Decision Matrix

When to use what:

| Need | Use |
|------|-----|
| Run an LLM agent | PydanticAI via `create_agent()` |
| Define typed tools for agents | PydanticAI `@agent.tool` |
| Expose tools via HTTP | QMCP `tool_registry` |
| Audit all tool calls | QMCP MCP Server |
| Human approval workflow | QMCP HITL API |
| Answer an approval by speaking | QMCP `integrations.voice` over vox |
| Model pricing/limits | QMCP `Models` registry |
| Multi-agent patterns | QMCP Topologies |
| Structured output | PydanticAI `output_type` |
| Token limits | PydanticAI `UsageLimits` |
| Persistent memory | QMCP `MemoryMixin` |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Application                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐     ┌─────────────────────┐       │
│  │   PydanticAI Agent  │────▶│    QMCP Topologies  │       │
│  │   (Runtime)         │     │    (Orchestration)  │       │
│  └─────────────────────┘     └─────────────────────┘       │
│           │                           │                     │
│           ▼                           ▼                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              QMCP Model Registry                     │   │
│  │   (Pricing, Limits, Capabilities, Fallbacks)        │   │
│  └─────────────────────────────────────────────────────┘   │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           QMCP MCP Server (FastAPI)                  │   │
│  │   (Audit Trail, HITL, Metrics)                       │   │
│  └─────────────────────────────────────────────────────┘   │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         QMCPToolset (PydanticAI Adapter)             │   │
│  │   (Connect agents to QMCP server)                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Future Integrations

Planned integrations:

| Integration | Purpose | Status |
|-------------|---------|--------|
| LangChain | Alternative agent framework | Planned |
| Instructor | Structured extraction | Considered |
| Marvin | AI functions | Considered |
