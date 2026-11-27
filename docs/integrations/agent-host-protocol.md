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
| `dispatchAction` with `chat/inputCompleted` or `chat/inputAnswerChanged` | Takes a person's answer to the turn's consent, below |
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

A turn whose `startedAt` is not an ISO 8601 timestamp is echoed with a
`rejectionReason`: the chat's `modifiedAt` is reckoned from it.

### A chat's state

A chat's snapshot is what the protocol's chat reducer makes of the actions
sent on its channel: every turn that ended since the server started, the turn
that runs with the parts said so far, and an input request with its answers.
A client that subscribes during a turn is given the open consent, and one
that subscribes after it is given the turn.

A change to a chat's `status` or `modifiedAt` is told to its session as
`session/chatUpdated`, so the session's catalog and the chat agree. An action
the host starts carries no `origin`; one a client dispatched carries that
client's.

### Consent and budget

A turn's act is issued against one run. It asks consent for that run on
qmcp's human queue, as every act with a budget does, and runs nothing until a
person approves. A turn cannot state another budget.

The chat is shown the request as the queue holds it: `chat/inputRequested`
with one single-select question, `answer`, whose options are the queue's.
No option is marked recommended.

A person answers in either place.

- **On the queue**, as every request is answered. The host then tells the
  chat with `chat/inputCompleted`, carrying the answer, or `cancel` when the
  request expired with none.
- **In a client attached from the machine qmcp runs on.** The client
  dispatches `chat/inputCompleted` with `response: accept` and the selected
  option, in `answers` or drafted before with `chat/inputAnswerChanged`. The
  host writes that answer through the queue's own route, with
  `agent-host-protocol` as `responded_by`, and echoes the action to the chat.

| A client sends | qmcp answers |
|---|---|
| An answer from another address | A `rejectionReason`. The request is shown there and is not answerable there |
| `decline` or `cancel` | A `rejectionReason`: the queue takes one of its options, and neither is one |
| A value that is not one of the options, free text, or a skipped answer | A `rejectionReason` |
| An answer the queue refuses: answered already, or expired | A `rejectionReason` carrying the queue's reason |

`responded_by` names the surface the answer came through. qmcp does not know
who the person at the client is.

### One act at a time

A turn started while a conversation or an act runs is echoed to the client
that sent it with a `rejectionReason`. Nothing is recorded and nothing is
queued.

### What is not served

| A client asks for | qmcp answers |
|---|---|
| Terminals, resources, change sets, search, automations, authentication, a list of sessions, a page of a chat's turns | `MethodNotFound`, naming the method |
| Cancelling a turn | The action echoed with a `rejectionReason`: an act cannot be stopped once it has started |
| Any other action, a tool-call confirmation among them | The action echoed with a `rejectionReason` |
| A budget other than one run | Nothing carries one; the act is issued from the command line or a page instead |
| A session's `inputNeeded` list, and the root's notifications of sessions | Nothing is sent. A pending consent is read from its chat, whose status says input is needed |
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

It prints that the host asked for input and the turn was walked away from:
qmcp's client role answers no question on a person's behalf. The consent
stays on the queue, where `uv run qmcp human list` shows it, and the act runs
if a person approves it there before it expires.

## Testing

`tests/test_agents_host.py` drives the host with a client in the same
process, over the host's own message stream, and runs the WebSocket route on
the test server. One test attaches qmcp's client role to qmcp's host role.
That proves the two agree with each other. Both were read from the same type
definitions, so it does not prove either against another program's client or
host.

The host role holds against the protocol's published TypeScript client,
`@microsoft/agent-host-protocol` 1.1.0, in a run made by hand and repeated by
no test here. In that run the client negotiates its version, creates a
session and a chat, starts a turn, and answers the consent `approve` in one
turn and `hold` in another. Its reducers, given every action the host sent,
hold the same chat and session state as the host's own snapshots, during the
turn and after it, and every envelope and state the host sent is valid under
the protocol's JSON schemas. The run needs Node and the published package,
which this repository does not install.

The client role has run against no host another program provides.
