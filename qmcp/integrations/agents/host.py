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
    dispatchAction    `chat/turnStarted` starts a turn; any other action is rejected

The agents the root advertises are `runtime_names()`. The act runs under its
own default budget, which is zero: it declares what it would need, asks
nothing and runs nothing. A client cannot state a budget, so over the
protocol no runtime runs. `docs/integrations/agent-host-protocol.md` says what
that leaves a client able to do.

One act runs at a time. A turn started while a conversation or an act runs is
echoed to its client with a `rejectionReason`, and nothing is recorded or
queued.

What a client asks for and qmcp does not serve is refused in the protocol's
terms: a request by `METHOD_NOT_FOUND`, an action by a `rejectionReason`. The
handshake's optional capabilities are left out. Cancelling a turn is among the
refusals: an act cannot be stopped once it has started.

`Hub` holds the sessions and is written against two seams, so it runs with no
socket and no database: `Acts` records an instruction, starts an act and
reads how far it got, and each `Connection` has an `outbox` of whole JSON
messages. `ServerActs` is `Acts` over the server's database and its run
tracker; `register` puts a hub behind the WebSocket route.

Read from the protocol's type definitions. What this cannot do: carry a
consent answer from a client to the human queue, state a budget, stop a turn,
replay actions to a client that reconnects, authenticate a client, or page a
chat's earlier turns.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
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

# The protocol's session and chat status bits.
IDLE, ERROR, IN_PROGRESS = 1, 2, 8

# Seconds between two looks at a running act.
POLL_SECONDS = 1.0

# How an act that is not running any more ended, by the row's status, as a
# client is told it. A status missing here is told by its own name.
ENDED = {
    "recorded": "Nothing was asked and nothing ran.",
    "unresolved": "Nothing was asked and nothing ran.",
    "refused": "The consent was refused on the human queue; nothing ran.",
    "unanswered": "Nobody answered the consent before it expired; nothing ran.",
    "done": "The runtime finished.",
    "failed": "The runtime failed.",
}
ASKING = ("Consent is asked on qmcp's human queue, as request {request}. Nothing runs"
          " until a person approves it there.")
ACTING = "A person approved it; the runtime is running."

CANNOT_STOP = "qmcp cannot stop an act once it has started"


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
        """Start an act under its default budget. Raises `Busy`."""
        ...

    def look(self, instruction_id: str) -> Look:
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


@dataclass
class Chat:
    resource: str
    session: str
    modified_at: str
    status: int = IDLE
    turn: Turn | None = None

    def summary(self) -> dict[str, Any]:
        return {"resource": self.resource, "title": "", "status": self.status,
                "modifiedAt": self.modified_at}

    def state(self) -> dict[str, Any]:
        # Completed turns are not kept: the instruction rows hold them.
        return {**self.summary(), "turns": []}


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


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class Connection:
    """One attached client: what it subscribed to and what is waiting to be sent."""

    def __init__(self, hub: Hub) -> None:
        self.hub = hub
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

    def connect(self) -> Connection:
        with self._lock:
            connection = Connection(self)
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
                                  "serverSeq": self.seq, "origin": origin}
        if rejection:
            params["rejectionReason"] = rejection
        return {"jsonrpc": "2.0", "method": "action", "params": params}

    def _emit(self, channel: str, action: dict[str, Any],
              origin: dict[str, Any] | None = None) -> None:
        """One action, to every client subscribed to its channel."""
        envelope = self._envelope(channel, action, origin, None)
        for connection in self.connections:
            if channel in connection.subscribed:
                connection.outbox.append(envelope)

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
            made = Chat(resource=chat, session=session.resource, modified_at=_now())
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
        chat.status, chat.modified_at = IN_PROGRESS, _now()
        self._emit(chat.resource, action, origin)
        self._emit(chat.resource, {
            "type": "chat/responsePart", "turnId": turn.id,
            "part": {"kind": "markdown", "id": turn.part_id,
                     "content": f"Recorded as instruction {instruction_id}; an act on it"
                                f" is started with qmcp's {session.provider} runtime."}})

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
                chat.turn, chat.modified_at = None, _now()
                chat.status = ERROR if failed else IDLE


# ---------------------------------------------------------------------------
# The server's own record and tracker
# ---------------------------------------------------------------------------


class ServerActs:
    """`Acts` over the server's database and the machine's one run tracker.

    `rows` is what `qmcp.instructions.act.configured_rows` returns; `runs` is
    the tracker the voice and act routes share, so a turn and a conversation
    exclude each other. `base_url` is where the act's command reaches this
    server's human queue.
    """

    def __init__(self, runs: Any, base_url: str, rows: Callable[[], Any] | None = None,
                 names: Callable[[], Iterable[str]] | None = None) -> None:
        self.runs = runs
        self.base_url = base_url
        self._rows = rows
        self._names = names

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
                "--runtime", runtime, "--base-url", self.base_url]
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
        connection = hub.connect()
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
    "Acts",
    "Busy",
    "Connection",
    "Hub",
    "Look",
    "ServerActs",
    "negotiate",
    "register",
]
