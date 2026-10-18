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
vendored as a submodule at `vendor/vox`.

**Status**: Works end to end; `run_forever` is not bounded on an idle queue.

```python
from qmcp.integrations.voice import VoiceApprovalLoop
from vox import HttpSTT
from vox.adapters import JOE
from vox.adapters.pyttsx3 import Pyttsx3TTS

stt = HttpSTT("http://127.0.0.1:8000", contract=JOE)
VoiceApprovalLoop(stt=stt, tts=Pyttsx3TTS()).run_once("deploy-001")
```

### [alfred](alfred.md)

Drives alfred, a self-hosted video editor, over its HTTP API. Provides:
- `alfred_projects` and `alfred_templates`, what alfred holds
- `alfred_render_enqueue`, a project's saved timeline or an explicit EDL queued
  to render
- `alfred_render_status`, a render followed to its state and a download link
- `alfred_render_deliver`, a finished render sent to a destination alfred offers
- `alfred_captions`, `alfred_captions_transcribe` and `alfred_captions_set`, a
  project's captions read, transcribed by alfred's speech engine, and corrected
- `alfred_health`, whether alfred has a renderer running and a speech engine
- `examples/flows/alfred_render.py`, a render put to a person for approval
- `examples/flows/alfred_caption_render.py`, a transcript put to a person before
  it is burned into a render, the render put to a person, and delivery on
  approval when asked for

Configured by the `QMCP_ALFRED_*` settings; nothing from alfred is imported.
`walkthrough/10-a-render-waits-for-a-person.md` and
`walkthrough/11-captions-are-read-before-they-are-burned-in.md` run it against
an alfred stood in for.

**Status**: Tested against a stand-in alfred; a running alfred is checked by
hand.

## Decision Matrix

When to use what:

| Need | Use |
|------|-----|
| Run an LLM agent | PydanticAI via `create_agent()` |
| Define typed tools for agents | PydanticAI `@agent.tool` |
| Expose tools via HTTP | QMCP `tool_registry` |
| Audit all tool calls | QMCP MCP Server |
| Human approval workflow | QMCP human request API |
| Answer an approval by speaking | QMCP `integrations.voice` over vox |
| Render a video project, behind approval | QMCP `alfred_*` tools and the `alfred-render` flow |
| Caption a video project, the transcript read before it is burned in | QMCP `alfred_captions*` tools and the `alfred-caption-render` flow |
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
