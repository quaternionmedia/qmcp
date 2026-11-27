"""qmcp as an Agent Host Protocol host: a WebSocket at `/v1/ahp`, on a loopback bind.

A client of the protocol attaches, opens a session on one of qmcp's runtimes,
opens a chat in it, and starts a turn with a message. The turn is an
instruction, recorded as every instruction is, and an act on it, started as
`POST /v1/instructions/{id}/act` starts one. The host streams what the act
reaches as markdown and ends the turn when the act ends. The instruction row
is the record of the work; sessions and chats are kept in memory, and a
restart forgets them.

    initialize        a version from `PROTOCOL_VERSIONS`, or refused
    subscribe         the root (the agents), a session, or a chat
    unsubscribe
    createSession     `provider` is a runtime's name; at most one working directory
    createChat
    disposeSession    refused while one of its turns runs
    dispatchAction    `chat/turnStarted` starts a turn; `chat/inputAnswerChanged`
                      and `chat/inputCompleted` answer its consent; any other
                      action is rejected

The agents the root advertises are `runtime_names()`. A turn's act is issued
against one run, so it asks consent for that run on the human queue and runs
nothing until a person approves. The host shows the request in the chat as
the protocol's input request, one question whose options are the queue's.
An answer a person gives in a client attached from this machine is written
through the queue's own route, with `RELAYED_BY` as `responded_by`, and
meets the refusals the queue makes for any answer. A client attached from
another address is shown the request and cannot answer it. An answer given
on the queue completes the input request in the chat.
`docs/integrations/agent-host-protocol.md` is the reference.

One act runs at a time. A turn started while a conversation or an act runs is
echoed to its client with a `rejectionReason`, and nothing is recorded or
queued.

What a client asks for and qmcp does not serve is refused in the protocol's
terms: a request by `METHOD_NOT_FOUND`, an action by a `rejectionReason`. The
handshake's optional capabilities are left out. Cancelling a turn is among the
refusals: an act cannot be stopped once it has started.

A chat's state is what the protocol's chat reducer makes of the actions sent
on its channel, so a client that subscribes during a turn or after one is
given what a client that followed every action holds. `reduce` is that
reducer for the actions this host sends. A change to a chat's status or
`modifiedAt` is told to its session as `session/chatUpdated`.

`Hub` holds the sessions and is written against two seams, so it runs with no
socket and no database: `Acts` records an instruction, starts an act and
reads how far it got, and each `Connection` has an `outbox` of whole JSON
messages. `ServerActs` is `Acts` over the server's database and its run
tracker; `register` puts a hub behind the WebSocket route.

Read from the protocol's type definitions. What this cannot do: say who the
person at the client is, take a budget other than one run, stop a turn,
replay actions to a client that reconnects, authenticate a client, or page a
chat's earlier turns.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname

# Module level, not inside `register`: the route's annotation is a string
# under `from __future__ import annotations`, and FastAPI resolves it here.
from fastapi import WebSocket, WebSocketDisconnect

from qmcp.integrations.agents import runtime_names
from qmcp.integrations.agents.adapters.ahp import PROTOCOL_VERSIONS, ROOT

# The protocol's error codes this host answers with (its `errors` types).
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
SESSION_NOT_FOUND = -32001
PROVIDER_NOT_FOUND = -32002
SESSION_ALREADY_EXISTS = -32003
TURN_IN_PROGRESS = -32004
UNSUPPORTED_PROTOCOL_VERSION = -32005
NOT_FOUND = -32008
ALREADY_EXISTS = -32010

# The protocol's session and chat status bits. `ACTIVITY` covers the bits of
# which one holds at a time; `IS_READ` is a flag beside them.
IDLE, ERROR, IN_PROGRESS, INPUT_NEEDED = 1, 2, 8, 24
ACTIVITY, IS_READ = 31, 32

# Seconds between two looks at a running act.
POLL_SECONDS = 1.0

# How an act that is not running any more ended, by the row's status, as a
# client is told it. A status missing here is told by its own name.
ENDED = {
    "recorded": "Nothing was asked and nothing ran.",
    "unresolved": "Nothing was asked and nothing ran.",
    "refused": "The consent was not approved; nothing ran.",
    "unanswered": "Nobody answered the consent before it expired; nothing ran.",
    "done": "The runtime finished.",
    "failed": "The runtime failed.",
}
ASKING = ("Consent is asked on qmcp's human queue, as request {request}. Nothing runs"
          " until a person approves it, there or here.")
ACTING = "A person approved it; the runtime is running."

CANNOT_STOP = "qmcp cannot stop an act once it has started"

RUNS = 1
"""The runs a turn's act is issued against."""

QUESTION = "answer"
"""The id of the one question in a consent's input request."""

RELAYED_BY = "agent-host-protocol"
"""`responded_by` on an answer a client relayed."""


def loopback(host: Any) -> bool:
    """Whether `host` is an address on this machine."""
    try:
        return ipaddress.ip_address(str(host)).is_loopback
    except ValueError:
        return str(host) == "localhost"


def negotiate(offered: Iterable[Any]) -> str | None:
    """The highest version offered that this host speaks, or None.

    A version is spoken when it is at or above one of `PROTOCOL_VERSIONS` in
    that version's own compatible range: the same major from 1 up, the same
    minor below it. An offer that is not three plain numbers is not spoken.
    """
    def parse(version: Any) -> tuple[int, int, int] | None:
        parts = str(version).split(".")
        if len(parts) != 3 or not all(p.isdigit() and str(int(p)) == p for p in parts):
            return None
        return int(parts[0]), int(parts[1]), int(parts[2])

    bases = [parse(version) for version in PROTOCOL_VERSIONS]
    chosen: tuple[tuple[int, int, int], str] | None = None
    for version in offered:
        parts = parse(version)
        if parts is None:
            continue
        spoken = any(
            base is not None and parts[0] == base[0]
            and (parts[0] > 0 or parts[1] == base[1]) and parts >= base
            for base in bases)
        if spoken and (chosen is None or parts > chosen[0]):
            chosen = (parts, str(version))
    return chosen[1] if chosen else None


def directory_of(uri: Any) -> Path | None:
    """The directory a `file:` URI names, or None for any other address."""
    split = urlsplit(str(uri))
    if split.scheme != "file":
        return None
    return Path(url2pathname(unquote(split.path)))


# ---------------------------------------------------------------------------
# The seam to the instruction record and the act
# ---------------------------------------------------------------------------


class Busy(Exception):
    """A conversation or an act already runs; the message names it."""


class Refused(Exception):
    """The queue refused an answer; the message is its reason."""


@dataclass(frozen=True)
class Asked:
    """A consent request as the queue holds it."""

    id: str
    prompt: str
    options: tuple[str, ...]
    answer: str | None = None
    """What was answered, by anyone, or None while it is pending or expired."""

    def request(self) -> dict[str, Any]:
        """The protocol's input request: one question, nothing selected."""
        return {"id": self.id, "message": self.prompt, "questions": [{
            "id": QUESTION, "kind": "single-select", "message": self.prompt,
            "required": True,
            "options": [{"id": option, "label": option} for option in self.options]}]}


@dataclass(frozen=True)
class Look:
    """How far an act on one instruction has got."""

    status: str
    running: bool
    outcome_text: str | None = None
    exit_code: int | None = None
    consent_request_id: str | None = None
    output: tuple[str, ...] = ()
    """The last lines the act's command printed."""


class Acts(Protocol):
    """Recording an instruction, starting an act on it, and reading the act."""

    def busy(self) -> str | None:
        """What runs now, by name, or None."""
        ...

    def record(self, text: str, detail: dict[str, Any]) -> str:
        """Record `text` as an instruction; returns its id. Nothing runs."""
        ...

    def start(self, instruction_id: str, runtime: str, cwd: Path | None) -> None:
        """Start an act issued against `RUNS`. Raises `Busy`."""
        ...

    def look(self, instruction_id: str) -> Look:
        ...

    def asked(self, request_id: str) -> Asked | None:
        """The consent request the queue holds under that id, or None."""
        ...

    def answer(self, request_id: str, response: str) -> None:
        """Write one answer through the queue's own route. Raises `Refused`."""
        ...


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    id: str
    chat: str
    instruction_id: str
    started: float
    part_id: str
    told: list[str] = field(default_factory=list)
    asked: Asked | None = None
    """The consent this turn showed its chat."""
    settled: bool = False
    """Whether the chat was told the consent's input request is complete."""
    draft: dict[str, Any] | None = None
    """The answer a client has drafted and not submitted."""


def _instant(stamp: Any) -> datetime | None:
    """The moment an ISO 8601 timestamp names, or None when it names none."""
    try:
        moment = datetime.fromisoformat(str(stamp))
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _stamp(moment: datetime) -> str:
    """A moment as the protocol's reducer writes one: UTC, to the millisecond."""
    moment = moment.astimezone(UTC)
    return f"{moment:%Y-%m-%dT%H:%M:%S}.{moment.microsecond // 1000:03d}Z"


def _status(state: dict[str, Any], ended: int | None = None) -> int:
    """A chat's status from its turn: the activity bits set, every flag kept."""
    active = state.get("activeTurn")
    if ended is not None:
        activity = ended
    elif active is None:
        activity = IDLE
    elif any(part.get("kind") == "inputRequest" and "response" not in part
             for part in active["responseParts"]):
        activity = INPUT_NEEDED
    else:
        activity = IN_PROGRESS
    return state["status"] & ~ACTIVITY | activity


def reduce(state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    """A chat's state after one action, as the protocol's chat reducer leaves it.

    The actions this host sends are covered: a turn started, a response part,
    a delta, an input request asked, drafted and completed, a turn complete,
    and an error. Any other action, and one that names another turn than the
    active one, leaves the state as it was.
    """
    kind = action.get("type")
    if kind == "chat/turnStarted":
        started = {**state, "modifiedAt": action["startedAt"], "activeTurn": {
            "id": action["turnId"], "startedAt": action["startedAt"],
            "message": action["message"], "responseParts": []}}
        return {**started, "status": _status(started) & ~IS_READ}
    active = state.get("activeTurn")
    if active is None:
        return state
    parts = list(active["responseParts"])

    def held(parts: list[dict[str, Any]]) -> dict[str, Any]:
        return {**state, "activeTurn": {**active, "responseParts": parts}}

    if kind in ("chat/inputRequested", "chat/inputAnswerChanged", "chat/inputCompleted"):
        asked = kind == "chat/inputRequested"
        request_id = action["request"]["id"] if asked else action.get("requestId")
        at = next((index for index, part in enumerate(parts)
                   if part.get("kind") == "inputRequest" and "response" not in part
                   and part["request"]["id"] == request_id), None)
        if asked:
            request = dict(action["request"])
            if at is None:
                parts.append({"kind": "inputRequest", "request": request})
            else:
                earlier = parts[at]["request"].get("answers")
                if "answers" not in request and earlier:
                    request["answers"] = earlier
                parts[at] = {"kind": "inputRequest", "request": request}
            shown = held(parts)
            return {**shown, "status": _status(shown) & ~IS_READ}
        if at is None:
            return state
        request = dict(parts[at]["request"])
        answers = dict(request.pop("answers", None) or {})
        if kind == "chat/inputCompleted":
            answers.update(action.get("answers") or {})
        elif action.get("answer") is None:
            answers.pop(action.get("questionId"), None)
        else:
            answers[action["questionId"]] = action["answer"]
        if answers:
            request["answers"] = answers
        parts[at] = {**parts[at], "request": request}
        if kind == "chat/inputAnswerChanged":
            return held(parts)
        parts[at]["response"] = action["response"]
        answered = held(parts)
        return {**answered, "status": _status(answered)}

    if action.get("turnId") != active["id"]:
        return state
    if kind == "chat/responsePart":
        if action["part"].get("kind") == "error":
            return state
        return held([*parts, action["part"]])
    if kind == "chat/delta":
        for index, part in enumerate(parts):
            if part.get("id") == action["partId"]:
                if part.get("kind") != "markdown":
                    return state
                parts[index] = {**part, "content": part["content"] + action["content"]}
                return held(parts)
        return state
    if kind in ("chat/turnComplete", "chat/error"):
        failed = kind == "chat/error"
        duration = max(0, action["duration"])
        if failed:
            parts.append(action["part"])
        began = _instant(active["startedAt"]) or datetime.now(UTC)
        ended = {key: value for key, value in state.items() if key != "activeTurn"}
        ended["turns"] = [*state["turns"], {
            "id": active["id"], "startedAt": active["startedAt"], "duration": duration,
            "message": active["message"], "responseParts": parts,
            "state": "error" if failed else "complete"}]
        ended["modifiedAt"] = _stamp(began + timedelta(milliseconds=duration))
        return {**ended, "status": _status(ended, ERROR if failed else None)}
    return state


@dataclass
class Chat:
    resource: str
    session: str
    held: dict[str, Any]
    """The chat's state, as `reduce` leaves it after each action sent."""
    turn: Turn | None = None

    @classmethod
    def new(cls, resource: str, session: str) -> Chat:
        return cls(resource=resource, session=session, held={
            "resource": resource, "title": "", "status": IDLE,
            "modifiedAt": _stamp(datetime.now(UTC)), "turns": []})

    def summary(self) -> dict[str, Any]:
        return {key: self.held[key] for key in ("resource", "title", "status", "modifiedAt")}

    def state(self) -> dict[str, Any]:
        return dict(self.held)


@dataclass
class Session:
    resource: str
    provider: str
    cwd: Path | None
    chats: dict[str, Chat] = field(default_factory=dict)

    def state(self) -> dict[str, Any]:
        running = any(chat.turn for chat in self.chats.values())
        state: dict[str, Any] = {
            "provider": self.provider, "title": "",
            "status": IN_PROGRESS if running else IDLE,
            "lifecycle": "ready", "activeClients": [],
            "chats": [chat.summary() for chat in self.chats.values()],
        }
        if self.cwd is not None:
            state["workingDirectories"] = [self.cwd.as_uri()]
        return state


class Connection:
    """One attached client: what it subscribed to and what is waiting to be sent."""

    def __init__(self, hub: Hub, loopback: bool = False) -> None:
        self.hub = hub
        self.loopback = loopback
        """Whether the client is attached from this machine."""
        self.outbox: deque[dict[str, Any]] = deque()
        self.subscribed: set[str] = set()
        self.client_id: str | None = None
        self.version: str | None = None
        self.closing = False
        """Set when the handshake is refused: the socket is closed once the
        refusal has been sent."""

    def handle(self, message: dict[str, Any]) -> None:
        """Take one message from the client; answers land in `outbox`."""
        self.hub.handle(self, message)

    def _result(self, message: dict[str, Any], result: Any) -> None:
        self.outbox.append({"jsonrpc": "2.0", "id": message["id"], "result": result})

    def _error(self, message: dict[str, Any], code: int, text: str,
               data: Any = None) -> None:
        error: dict[str, Any] = {"code": code, "message": text}
        if data is not None:
            error["data"] = data
        self.outbox.append({"jsonrpc": "2.0", "id": message["id"], "error": error})


class Hub:
    """Every session this host holds, and the clients attached to them.

    `acts` is the seam to the instruction record. `runtimes` returns the
    names a session may run, read at each use. One lock orders every message
    and every tick, so two clients see one sequence of actions.
    """

    def __init__(self, acts: Acts, *, runtimes: Callable[[], Iterable[str]] = runtime_names,
                 clock: Callable[[], float] = time.monotonic, version: str = "") -> None:
        self.acts = acts
        self.runtimes = runtimes
        self.clock = clock
        self.version = version
        self.sessions: dict[str, Session] = {}
        self.connections: list[Connection] = []
        self.seq = 0
        self.turn: Turn | None = None
        self._lock = threading.RLock()

    # --- connections -------------------------------------------------------------

    def connect(self, loopback: bool = False) -> Connection:
        with self._lock:
            connection = Connection(self, loopback)
            self.connections.append(connection)
            return connection

    def disconnect(self, connection: Connection) -> None:
        with self._lock:
            if connection in self.connections:
                self.connections.remove(connection)

    # --- what is sent ------------------------------------------------------------

    def _envelope(self, channel: str, action: dict[str, Any],
                  origin: dict[str, Any] | None, rejection: str | None) -> dict[str, Any]:
        self.seq += 1
        params: dict[str, Any] = {"channel": channel, "action": action,
                                  "serverSeq": self.seq}
        if origin is not None:
            params["origin"] = origin
        if rejection:
            params["rejectionReason"] = rejection
        return {"jsonrpc": "2.0", "method": "action", "params": params}

    def _emit(self, channel: str, action: dict[str, Any],
              origin: dict[str, Any] | None = None) -> None:
        """One action, to every client subscribed to its channel.

        An action on a chat is reduced into the chat's state, and what that
        changed in the chat's summary is told to its session.
        """
        envelope = self._envelope(channel, action, origin, None)
        for connection in self.connections:
            if channel in connection.subscribed:
                connection.outbox.append(envelope)
        chat = self._chat(channel)
        if chat is None:
            return
        before = chat.summary()
        chat.held = reduce(chat.held, action)
        changes = {key: value for key, value in chat.summary().items()
                   if before[key] != value}
        if changes:
            self._emit(chat.session, {"type": "session/chatUpdated",
                                      "chat": chat.resource, "changes": changes})

    def _reject(self, connection: Connection, channel: str, action: dict[str, Any],
                origin: dict[str, Any], why: str) -> None:
        """An action refused, echoed to the client that dispatched it and no other."""
        connection.outbox.append(self._envelope(channel, action, origin, why))

    # --- snapshots ---------------------------------------------------------------

    def _agents(self) -> list[dict[str, Any]]:
        return [{"provider": name, "displayName": name,
                 "description": f"qmcp's {name} runtime, behind consent on its human queue",
                 "models": []} for name in self.runtimes()]

    def _chat(self, channel: str) -> Chat | None:
        for session in self.sessions.values():
            if channel in session.chats:
                return session.chats[channel]
        return None

    def _snapshot(self, channel: str) -> dict[str, Any] | None:
        if channel == ROOT:
            state: dict[str, Any] = {"agents": self._agents(),
                                     "activeSessions": len(self.sessions)}
        elif channel in self.sessions:
            state = self.sessions[channel].state()
        elif (chat := self._chat(channel)) is not None:
            state = chat.state()
        else:
            return None
        return {"resource": channel, "state": state, "fromSeq": self.seq}

    # --- messages ----------------------------------------------------------------

    def handle(self, connection: Connection, message: dict[str, Any]) -> None:
        with self._lock:
            method = message.get("method")
            if not isinstance(method, str):
                return  # an answer to a request; this host makes none
            params = message.get("params")
            params = params if isinstance(params, dict) else {}
            asked = "id" in message
            if method == "initialize":
                if asked:
                    self._initialize(connection, message, params)
                return
            if connection.version is None:
                if asked:
                    connection._error(message, INVALID_REQUEST,
                                      f"initialize comes before {method}")
                return
            if method == "dispatchAction":
                self._dispatched(connection, params)
                return
            if not asked:
                return
            handler = {"subscribe": self._subscribe, "unsubscribe": self._unsubscribe,
                       "createSession": self._create_session,
                       "createChat": self._create_chat,
                       "disposeSession": self._dispose_session}.get(method)
            if handler is None:
                connection._error(message, METHOD_NOT_FOUND,
                                  f"qmcp's host does not serve {method}")
                return
            handler(connection, message, params)

    def _initialize(self, connection: Connection, message: dict[str, Any],
                    params: dict[str, Any]) -> None:
        offered = params.get("protocolVersions")
        chosen = negotiate(offered if isinstance(offered, list) else [])
        if chosen is None:
            connection._error(
                message, UNSUPPORTED_PROTOCOL_VERSION,
                "qmcp's host speaks none of the versions offered",
                {"supportedVersions": list(PROTOCOL_VERSIONS)})
            connection.closing = True
            return
        connection.version = chosen
        connection.client_id = str(params.get("clientId") or f"client-{uuid.uuid4()}")
        snapshots = []
        for channel in params.get("initialSubscriptions") or []:
            snapshot = self._snapshot(str(channel))
            if snapshot is not None:
                connection.subscribed.add(str(channel))
                snapshots.append(snapshot)
        info: dict[str, Any] = {"name": "qmcp"}
        if self.version:
            info["version"] = self.version
        connection._result(message, {"protocolVersion": chosen, "serverSeq": self.seq,
                                     "serverInfo": info, "snapshots": snapshots})

    def _subscribe(self, connection: Connection, message: dict[str, Any],
                   params: dict[str, Any]) -> None:
        channel = str(params.get("channel") or "")
        snapshot = self._snapshot(channel)
        if snapshot is None:
            connection._error(message, NOT_FOUND, f"qmcp's host holds no {channel or 'channel'}")
            return
        connection.subscribed.add(channel)
        connection._result(message, {"snapshot": snapshot})

    def _unsubscribe(self, connection: Connection, message: dict[str, Any],
                     params: dict[str, Any]) -> None:
        connection.subscribed.discard(str(params.get("channel") or ""))
        connection._result(message, None)

    def _create_session(self, connection: Connection, message: dict[str, Any],
                        params: dict[str, Any]) -> None:
        channel = str(params.get("channel") or "")
        names = tuple(self.runtimes())
        provider = params.get("provider")
        directories = params.get("workingDirectories") or []
        if not channel:
            connection._error(message, INVALID_PARAMS, "createSession names no channel")
        elif channel in self.sessions:
            connection._error(message, SESSION_ALREADY_EXISTS,
                              f"qmcp's host already holds {channel}")
        elif not provider:
            connection._error(message, INVALID_PARAMS,
                              "createSession names no provider, and qmcp defaults none."
                              f" The agents are: {', '.join(names)}")
        elif provider not in names:
            connection._error(message, PROVIDER_NOT_FOUND,
                              f"{provider!r} is not one of qmcp's runtimes."
                              f" The agents are: {', '.join(names)}")
        elif len(directories) > 1:
            connection._error(message, INVALID_PARAMS,
                              "an act runs in one clone; createSession names"
                              f" {len(directories)} working directories")
        else:
            cwd = directory_of(directories[0]) if directories else None
            if directories and (cwd is None or not cwd.is_dir()):
                connection._error(message, INVALID_PARAMS,
                                  f"{directories[0]} is not a directory on this machine")
                return
            self.sessions[channel] = Session(resource=channel, provider=str(provider), cwd=cwd)
            connection._result(message, None)

    def _create_chat(self, connection: Connection, message: dict[str, Any],
                     params: dict[str, Any]) -> None:
        session = self.sessions.get(str(params.get("channel") or ""))
        chat = str(params.get("chat") or "")
        if session is None:
            connection._error(message, SESSION_NOT_FOUND,
                              f"qmcp's host holds no session {params.get('channel')}")
        elif not chat:
            connection._error(message, INVALID_PARAMS, "createChat names no chat")
        elif self._snapshot(chat) is not None:
            connection._error(message, ALREADY_EXISTS, f"qmcp's host already holds {chat}")
        elif params.get("initialMessage") is not None:
            connection._error(message, INVALID_PARAMS,
                              "qmcp's host starts no turn from createChat; dispatch"
                              " chat/turnStarted on the chat")
        else:
            made = Chat.new(chat, session.resource)
            session.chats[chat] = made
            connection._result(message, None)
            self._emit(session.resource, {"type": "session/chatAdded",
                                          "summary": made.summary()})

    def _dispose_session(self, connection: Connection, message: dict[str, Any],
                         params: dict[str, Any]) -> None:
        channel = str(params.get("channel") or "")
        session = self.sessions.get(channel)
        if session is None:
            connection._error(message, SESSION_NOT_FOUND,
                              f"qmcp's host holds no session {channel}")
        elif any(chat.turn for chat in session.chats.values()):
            connection._error(message, TURN_IN_PROGRESS,
                              f"a turn runs in {channel}; {CANNOT_STOP}")
        else:
            del self.sessions[channel]
            for attached in self.connections:
                attached.subscribed -= {channel, *session.chats}
            connection._result(message, None)

    # --- a turn ------------------------------------------------------------------

    def _dispatched(self, connection: Connection, params: dict[str, Any]) -> None:
        channel = str(params.get("channel") or "")
        action = params.get("action")
        action = action if isinstance(action, dict) else {}
        origin = {"clientId": connection.client_id, "clientSeq": params.get("clientSeq")}
        kind = action.get("type")
        chat = self._chat(channel)
        if chat is None:
            self._reject(connection, channel, action, origin,
                         f"qmcp's host holds no chat {channel or 'channel'}")
        elif kind == "chat/turnCancelled":
            self._reject(connection, channel, action, origin, CANNOT_STOP)
        elif kind in ("chat/inputAnswerChanged", "chat/inputCompleted"):
            self._answered(connection, chat, action, origin)
        elif kind != "chat/turnStarted":
            self._reject(connection, channel, action, origin,
                         f"qmcp's host accepts no {kind} from a client")
        else:
            self._start_turn(connection, chat, action, origin)

    def _start_turn(self, connection: Connection, chat: Chat, action: dict[str, Any],
                    origin: dict[str, Any]) -> None:
        message = action.get("message")
        text = str((message or {}).get("text") or "").strip() if isinstance(
            message, dict) else ""
        turn_id = action.get("turnId")

        def reject(why: str) -> None:
            self._reject(connection, chat.resource, action, origin, why)

        if not turn_id or not text:
            reject("a turn carries a turnId and a message with text")
            return
        if _instant(action.get("startedAt")) is None:
            reject("a turn carries startedAt, an ISO 8601 timestamp")
            return
        running = self.acts.busy()
        if self.turn is not None or running is not None:
            reject("a conversation or an act is already running"
                   + (f", for '{running}'" if running else "")
                   + ". One at a time; qmcp queues nothing")
            return
        session = self.sessions[chat.session]
        instruction_id = self.acts.record(text, {
            "surface": "agent-host-protocol", "session": session.resource,
            "chat": chat.resource, "turn": str(turn_id), "client": connection.client_id})
        try:
            self.acts.start(instruction_id, session.provider, session.cwd)
        except Busy as exc:
            reject(f"a conversation or an act started first, for '{exc}'. Instruction"
                   f" {instruction_id} is recorded and nothing ran on it")
            return
        turn = Turn(id=str(turn_id), chat=chat.resource, instruction_id=instruction_id,
                    started=self.clock(), part_id=f"part-{uuid.uuid4()}")
        self.turn = chat.turn = turn
        self._emit(chat.resource, action, origin)
        self._emit(chat.resource, {
            "type": "chat/responsePart", "turnId": turn.id,
            "part": {"kind": "markdown", "id": turn.part_id,
                     "content": f"Recorded as instruction {instruction_id}; an act on it"
                                f" is started with qmcp's {session.provider} runtime."}})

    # --- a turn's consent --------------------------------------------------------

    def _answered(self, connection: Connection, chat: Chat, action: dict[str, Any],
                  origin: dict[str, Any]) -> None:
        """A client drafts or submits an answer to the consent its chat shows."""
        def reject(why: str) -> None:
            self._reject(connection, chat.resource, action, origin, why)

        turn = chat.turn
        asked = turn.asked if turn is not None else None
        if turn is None or asked is None or action.get("requestId") != asked.id:
            reject(f"no consent is asked in this chat as {action.get('requestId')}")
            return
        if turn.settled:
            reject(f"{asked.id} is answered already")
            return
        if not connection.loopback:
            reject("a consent is answered from the machine qmcp runs on, and this"
                   " client is attached from another address")
            return
        if action["type"] == "chat/inputAnswerChanged":
            if action.get("questionId") != QUESTION:
                reject(f"{asked.id} has one question, {QUESTION}")
                return
            turn.draft = action.get("answer") if isinstance(action.get("answer"), dict) else None
            self._emit(chat.resource, action, origin)
            return
        choose = f"choose one of {', '.join(asked.options)} and accept"
        if action.get("response") != "accept":
            reject(f"qmcp's queue takes an answer and no {action.get('response')}; {choose}")
            return
        answers = action.get("answers")
        given = answers.get(QUESTION) if isinstance(answers, dict) else None
        given = given if isinstance(given, dict) else turn.draft or {}
        value = given.get("value") if isinstance(given.get("value"), dict) else {}
        chosen = value.get("value")
        if (given.get("state") == "skipped" or value.get("kind") != "selected"
                or chosen not in asked.options or value.get("freeformValues")):
            reject(f"no option of {asked.id} is selected; {choose}")
            return
        try:
            self.acts.answer(asked.id, str(chosen))
        except Refused as exc:
            reject(f"qmcp's queue refused the answer: {exc}")
            return
        turn.settled = True
        self._emit(chat.resource, action, origin)

    def _consent(self, turn: Turn, look: Look) -> None:
        """Show the consent when it is asked, and close it when it is answered elsewhere."""
        if look.status == "asking" and turn.asked is None and look.consent_request_id:
            turn.asked = self.acts.asked(look.consent_request_id)
            if turn.asked is not None:
                self._emit(turn.chat, {"type": "chat/inputRequested",
                                       "request": turn.asked.request()})
        elif look.status != "asking" and turn.asked is not None and not turn.settled:
            turn.settled = True
            now = self.acts.asked(turn.asked.id)
            answer = now.answer if now is not None else None
            action: dict[str, Any] = {"type": "chat/inputCompleted",
                                      "requestId": turn.asked.id,
                                      "response": "accept" if answer else "cancel"}
            if answer:
                action["answers"] = {QUESTION: {"state": "submitted", "value": {
                    "kind": "selected", "value": answer}}}
            self._emit(turn.chat, action)

    def _say(self, turn: Turn, text: str) -> None:
        self._emit(turn.chat, {"type": "chat/delta", "turnId": turn.id,
                               "partId": turn.part_id, "content": f"\n\n{text}"})

    def tick(self) -> None:
        """Read the running act once, and tell the chat what changed."""
        with self._lock:
            turn = self.turn
            if turn is None:
                return
            look = self.acts.look(turn.instruction_id)
            if look.status not in turn.told:
                turn.told.append(look.status)
                if look.status == "asking":
                    self._say(turn, ASKING.format(request=look.consent_request_id))
                elif look.status == "acting":
                    self._say(turn, ACTING)
            self._consent(turn, look)
            if look.running:
                return
            said = look.outcome_text or "\n".join(look.output)
            ended = ENDED.get(look.status, f"The act ended {look.status}.")
            duration = int((self.clock() - turn.started) * 1000)
            chat = self._chat(turn.chat)
            failed = look.status == "failed"
            if failed:
                self._emit(turn.chat, {
                    "type": "chat/error", "turnId": turn.id, "duration": duration,
                    "part": {"kind": "error", "error": {
                        "errorType": "runtime", "message": said or ended}}})
            else:
                self._say(turn, f"{ended}\n\n{said}" if said else ended)
                self._emit(turn.chat, {"type": "chat/turnComplete", "turnId": turn.id,
                                       "duration": duration})
            self.turn = None
            if chat is not None:
                chat.turn = None


# ---------------------------------------------------------------------------
# The server's own record and tracker
# ---------------------------------------------------------------------------


class ServerActs:
    """`Acts` over the server's database and the machine's one run tracker.

    `rows` is what `qmcp.instructions.act.configured_rows` returns; `runs` is
    the tracker the voice and act routes share, so a turn and a conversation
    exclude each other. `base_url` is where the act's command reaches this
    server's human queue, and where this reads a consent and writes a relayed
    answer: `queue` is the client of those routes, made once on first use.
    """

    def __init__(self, runs: Any, base_url: str, rows: Callable[[], Any] | None = None,
                 names: Callable[[], Iterable[str]] | None = None,
                 queue: Any = None) -> None:
        self.runs = runs
        self.base_url = base_url
        self._rows = rows
        self._names = names
        self._queue = queue

    def queue(self) -> Any:
        if self._queue is None:
            from qmcp.client import MCPClient

            self._queue = MCPClient(self.base_url)
        return self._queue

    def rows(self) -> Any:
        if self._rows is None:
            from qmcp.instructions.act import configured_rows

            self._rows = configured_rows()
        return self._rows()

    def busy(self) -> str | None:
        return self.runs.running()

    def record(self, text: str, detail: dict[str, Any]) -> str:
        from qmcp.db.models import Instruction, InstructionSource
        from qmcp.instructions import resolve, roster_names

        found = resolve(text, self._names() if self._names else roster_names())
        now = datetime.now(UTC)
        row = Instruction(text=text, project=found.project, source=InstructionSource.PROTOCOL,
                          status=found.status, created_at=now, updated_at=now,
                          detail={**found.detail(), **detail})
        with self.rows() as session:
            session.add(row)
            session.commit()
            return row.id

    def start(self, instruction_id: str, runtime: str, cwd: Path | None) -> None:
        argv = [sys.executable, "-m", "qmcp", "instructions", "act", instruction_id,
                "--runtime", runtime, "--budget", str(RUNS), "--base-url", self.base_url]
        if cwd is not None:
            argv += ["--cwd", str(cwd)]
        try:
            self.runs.start(argv, kind="act", request_id=instruction_id)
        except RuntimeError as exc:
            raise Busy(str(exc)) from exc

    def look(self, instruction_id: str) -> Look:
        from qmcp.db.models import Instruction

        with self.rows() as session:
            row = session.get(Instruction, instruction_id)
            status = row.status.value if row is not None else "unresolved"
            text = row.outcome_text if row is not None else None
            code = row.exit_code if row is not None else None
            request = row.consent_request_id if row is not None else None
        tracked = self.runs.status()
        mine = tracked.get("request_id") == instruction_id
        return Look(status=status, running=bool(mine and tracked.get("running")),
                    outcome_text=text, exit_code=code, consent_request_id=request,
                    output=tuple(tracked.get("output") or ()) if mine else ())

    def asked(self, request_id: str) -> Asked | None:
        from qmcp.client import MCPClientError

        try:
            request, response = self.queue().get_human_request(request_id)
        except MCPClientError:
            return None
        return Asked(id=request.id, prompt=request.prompt,
                     options=tuple(request.options or ()),
                     answer=response.response if response is not None else None)

    def answer(self, request_id: str, response: str) -> None:
        from qmcp.client import MCPClientError

        try:
            self.queue().submit_human_response(request_id, response,
                                               responded_by=RELAYED_BY)
        except MCPClientError as exc:
            raise Refused(str(exc)) from exc
        except Exception as exc:
            # A status the queue's route refused with carries its answer;
            # anything else is not a refusal and is raised as it came.
            refusal = getattr(exc, "response", None)
            if refusal is None:
                raise
            try:
                why = refusal.json().get("detail")
            except ValueError:
                why = None
            raise Refused(str(why or exc)) from exc


def register(app: Any, runs: Any, base_url: str, *, acts: Acts | None = None,
             poll_seconds: float = POLL_SECONDS) -> Hub:
    """Attach the host's WebSocket route to a FastAPI app. Returns the hub.

    Called only on a loopback bind, as every route that starts a process is.
    """
    from starlette.concurrency import run_in_threadpool

    from qmcp import __version__

    hub = Hub(acts or ServerActs(runs, base_url), version=__version__)
    app.state.ahp_host = hub

    @app.websocket("/v1/ahp")
    async def agent_host(socket: WebSocket) -> None:
        await socket.accept()
        connection = hub.connect(loopback(socket.client.host) if socket.client else False)
        try:
            while True:
                try:
                    text = await asyncio.wait_for(socket.receive_text(), poll_seconds)
                except TimeoutError:
                    text = None
                if text is not None:
                    try:
                        message = json.loads(text)
                    except ValueError:
                        message = None
                    if isinstance(message, dict):
                        await run_in_threadpool(connection.handle, message)
                await run_in_threadpool(hub.tick)
                while connection.outbox:
                    await socket.send_text(json.dumps(connection.outbox.popleft()))
                if connection.closing:
                    await socket.close()
                    return
        except WebSocketDisconnect:
            pass
        finally:
            hub.disconnect(connection)

    return hub


__all__ = [
    "reduce",
    "Acts",
    "Asked",
    "Busy",
    "Connection",
    "Hub",
    "Look",
    "Refused",
    "ServerActs",
    "negotiate",
    "register",
]
