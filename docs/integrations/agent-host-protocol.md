# Agent Host Protocol

qmcp speaks the
[Agent Host Protocol](https://microsoft.github.io/agent-host-protocol/) on
both sides. As a client, a host that speaks it is one of qmcp's runtimes,
`--runtime ahp`; [voice.md](voice.md) describes that runtime beside the
others. As a host, qmcp serves the protocol's clients, and this page is its
reference.

The protocol is a working draft whose wire types may change. Both roles are
written from its type definitions and speak the versions in
`PROTOCOL_VERSIONS` (`qmcp.integrations.agents.adapters.ahp`), the one list
for both.

## The host

A client attaches to a WebSocket at `/v1/ahp`, one JSON-RPC message per text
frame. The route exists only when the server is bound to loopback, as every
route that starts a process does.

| The client sends | qmcp does |
|---|---|
| `initialize` | Chooses the highest version offered that it speaks, or refuses with `UnsupportedProtocolVersion` and closes the socket |
| `subscribe` to `ahp-root://` | Answers with the agents: qmcp's runtime names, as `qmcp instructions act --runtime` takes them |
| `createSession` | Holds a session on the runtime `provider` names, in at most one working directory. No provider is defaulted |
| `createChat` | Adds a chat to the session and tells every client subscribed to the session |
| `dispatchAction` with `chat/turnStarted` | Starts a turn, below |
| `subscribe`, `unsubscribe`, `disposeSession` | As named. A session with a turn running is not disposed |

### A turn is an instruction acted on

A turn's message is recorded as an instruction, with `protocol` as its source
and the session, the chat, the turn and the client in its `detail`. An act on
it is started in the session's directory, as `POST /v1/instructions/{id}/act`
starts one. The chat is told the instruction's id, then each stage the act
reaches, then what the act said, and the turn completes when the act ends. A
runtime that fails ends the turn with the protocol's error.

The instruction row is the record of the work. Sessions and chats are held in
memory and a restart forgets them.

### Consent and budget

The act asks its consent on qmcp's human queue, as every act does, and runs
nothing until a person approves it there. Its budget is the act's default,
which is zero: it declares what it would need, asks nothing and runs nothing.
A client cannot state a budget, so a turn over the protocol ends with the
declaration and no runtime runs. A person who wants the runtime to run issues
the act on that instruction with a budget, from the command line or a page.

A client cannot answer a consent request through this host. The request is
named in the chat, and it is answered where every request is: the queue.

### One act at a time

A turn started while a conversation or an act runs is echoed to the client
that sent it with a `rejectionReason`. Nothing is recorded and nothing is
queued.

### What is not served

| A client asks for | qmcp answers |
|---|---|
| Terminals, resources, change sets, search, automations, authentication, a list of sessions, earlier turns | `MethodNotFound`, naming the method |
| Cancelling a turn | The action echoed with a `rejectionReason`: an act cannot be stopped once it has started |
| Any other action, a tool-call confirmation among them | The action echoed with a `rejectionReason` |
| A turn carried in `createChat` | `InvalidParams`; a turn starts with `chat/turnStarted` |

The handshake's optional capabilities are left out of its result. Nothing is
present and empty.

A client that reconnects is not replayed the actions it missed, and the
transport is not authenticated: the loopback bind is the boundary.

## Trying it

With the server running on its default port, a client connects to
`ws://127.0.0.1:3141/v1/ahp`. qmcp's own client role is such a client:

```bash
QMCP_AHP_URL=ws://127.0.0.1:3141/v1/ahp uv run python -c "
from pathlib import Path
from qmcp.integrations.agents import Brief
from qmcp.integrations.agents.adapters.ahp import Runtime
print(Runtime(provider='scripted').run(Brief(instruction='Say what qmcp is.', cwd=Path.cwd(), project='qmcp')).text)"
```

It prints the declaration, because the turn's budget is zero.

## Testing

`tests/test_agents_host.py` drives the host with a client in the same
process, over the host's own message stream, and runs the WebSocket route on
the test server. One test attaches qmcp's client role to qmcp's host role.
That proves the two agree with each other. Both were read from the same type
definitions, so it does not prove either against another program's client or
host.
