"""A host that speaks the Agent Host Protocol, as an agent runtime: `--runtime ahp`.

    QMCP_AHP_URL=ws://127.0.0.1:8123 uv run qmcp instructions act <id> --runtime ahp --budget 1

qmcp is the protocol's client. One run opens one connection to the host named
by `QMCP_AHP_URL` (`qmcp.config.Settings.ahp_url`), and sends, in order:
`initialize`; `createSession` on an `ahp-session:/<uuid>` it chooses, with the
brief's directory as the one working directory; `subscribe` to that session,
waiting for `session/ready`; `createChat` on an `ahp-chat:/<uuid>` it chooses;
`subscribe` to that chat; and one `chat/turnStarted` carrying the brief's
prompt. It collects the markdown the host streams (`chat/responsePart`,
`chat/delta`) until `chat/turnComplete` or `chat/error`.

It approves nothing. A tool call the host holds for confirmation is answered
`chat/toolCallConfirmed` with `approved: false`, a tool result held for
confirmation is answered `chat/toolCallResultConfirmed` with `approved: false`,
and the outcome's `detail` names each one. A request for input
(`chat/inputRequested`) has nobody to answer it, so the turn is cancelled.

The exchange is written against `MessageStream` -- send one JSON message,
receive the next -- so it runs with no network; `WebSocketStream` carries it
over a WebSocket, one message per text frame, and needs the `ahp` extra.

Bounds: `MESSAGE_TIMEOUT` seconds for any one message, `TURN_TIMEOUT` seconds
for the whole run, `TEXT_CHARS` characters of collected text. A run that stops
on a bound dispatches `chat/turnCancelled` before it closes the connection.

Read from the protocol's type definitions (`types/` in its repository) and its
specification pages. Where the two differ, the types are followed: the chat
page says the host allocates the chat's address, and `CreateChatParams.chat`
is a required address the client chooses.

What this cannot do: confine the host's agent to the brief's directory, make
it read-only, or know what it spent; reconnect a dropped connection; reassemble
a message a host splits into `channel/frame` notifications; authenticate; or
dispose of the session it created, which stays on the host.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol

from qmcp.config import get_settings
from qmcp.integrations.agents import AgentOutcome, Brief, OnEvent
from qmcp.spend import unknown

NAME = "ahp"
PRODUCT = "AHP"

ROOT = "ahp-root://"
PROTOCOL_VERSIONS = ("1.0.0", "0.9.0")

MESSAGE_TIMEOUT = 90.0
TURN_TIMEOUT = 600.0
TEXT_CHARS = 20_000
CONNECT_TIMEOUT = 10.0

# JSON-RPC's own code, which the protocol's error table lists as MethodNotFound:
# the answer to every request a host makes of this client.
METHOD_NOT_FOUND = -32601


class Unreachable(Exception):
    """No connection to the host; the message says why, in words."""


class Closed(Exception):
    """The host closed the connection."""


class MessageStream(Protocol):
    """An ordered, reliable stream of whole JSON messages, in both directions."""

    def send(self, message: dict[str, Any]) -> None:
        ...

    def receive(self, timeout: float) -> dict[str, Any]:
        """The next message. `TimeoutError` when none arrives within `timeout`
        seconds, `Closed` when the other end is gone."""
        ...

    def close(self) -> None:
        ...


class WebSocketStream:
    """`MessageStream` over a WebSocket: one JSON message per text frame."""

    def __init__(self, url: str, timeout: float = CONNECT_TIMEOUT) -> None:
        try:
            from websockets.exceptions import WebSocketException
            from websockets.sync.client import connect
        except ImportError as exc:
            raise Unreachable("the websockets package is not installed; `uv sync --extra ahp`"
                              " installs it") from exc
        try:
            self.socket = connect(url, open_timeout=timeout)
        except (OSError, WebSocketException, ValueError) as exc:
            raise Unreachable(type(exc).__name__) from exc

    def send(self, message: dict[str, Any]) -> None:
        from websockets.exceptions import ConnectionClosed

        try:
            self.socket.send(json.dumps(message))
        except ConnectionClosed as exc:
            raise Closed(str(exc)) from exc

    def receive(self, timeout: float) -> dict[str, Any]:
        from websockets.exceptions import ConnectionClosed

        try:
            frame = self.socket.recv(timeout=timeout)
        except ConnectionClosed as exc:
            raise Closed(str(exc)) from exc
        message = json.loads(frame)
        if not isinstance(message, dict):
            raise ValueError("a frame that is not a JSON object")
        return message

    def close(self) -> None:
        self.socket.close()


class Failed(Exception):
    """The run ended without an answer; the message is what the person is told."""


class TurnTimeout(TimeoutError):
    """The whole run outlasted its bound."""


def _words(value: Any) -> str:
    """A `StringOrMarkdown` as plain text."""
    if isinstance(value, dict):
        return str(value.get("markdown", ""))
    return str(value or "")


def _error(info: Any) -> str:
    """The message of an `ErrorInfo`, or of a JSON-RPC error object."""
    if isinstance(info, dict) and info.get("message"):
        return str(info["message"])
    return "no reason given"


class _Exchange:
    """One connection's requests, notifications and deadlines."""

    def __init__(self, stream: MessageStream, clock: Callable[[], float],
                 message_timeout: float, turn_timeout: float) -> None:
        self.stream = stream
        self.clock = clock
        self.message_timeout = message_timeout
        self.started = clock()
        self.deadline = self.started + turn_timeout
        self.request_id = 0
        self.client_seq = 0
        self.waiting: deque[dict[str, Any]] = deque()

    def _receive(self) -> dict[str, Any]:
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise TurnTimeout()
        if remaining >= self.message_timeout:
            return self.stream.receive(self.message_timeout)
        try:
            return self.stream.receive(remaining)
        except TimeoutError as exc:
            raise TurnTimeout() from exc

    def _refuse(self, message: dict[str, Any]) -> None:
        self.stream.send({"jsonrpc": "2.0", "id": message["id"], "error": {
            "code": METHOD_NOT_FOUND, "message": f"qmcp answers no {message['method']} request"}})

    def request(self, method: str, params: dict[str, Any]) -> Any:
        """Send a request and return its result; notifications that arrive
        first are kept for `notification`. A JSON-RPC error is `Failed`."""
        self.request_id += 1
        mine = self.request_id
        self.stream.send({"jsonrpc": "2.0", "id": mine, "method": method, "params": params})
        while True:
            message = self._receive()
            if "method" in message:
                if "id" in message:
                    self._refuse(message)
                else:
                    self.waiting.append(message)
                continue
            if message.get("id") != mine:
                continue
            if "error" in message:
                raise Failed(f"The host refused {method}: {_error(message['error'])}.")
            return message.get("result")

    def notification(self) -> dict[str, Any]:
        """The next notification from the host, oldest first."""
        while True:
            if self.waiting:
                return self.waiting.popleft()
            message = self._receive()
            if "method" not in message:
                continue
            if "id" in message:
                self._refuse(message)
                continue
            return message

    def dispatch(self, channel: str, action: dict[str, Any]) -> int:
        """Send one `dispatchAction`; returns the `clientSeq` it carried."""
        self.client_seq += 1
        self.stream.send({"jsonrpc": "2.0", "method": "dispatchAction", "params": {
            "channel": channel, "clientSeq": self.client_seq, "action": action}})
        return self.client_seq


class Runtime:
    """One instruction, as one turn of one new chat on an AHP host.

    `url` defaults to `QMCP_AHP_URL`. `connect` builds the `MessageStream` for
    an address and raises `Unreachable` when there is no host there. `provider`
    is the host's agent provider id, left to the host when None.
    """

    name = NAME

    def __init__(self, url: str | None = None,
                 connect: Callable[[str], MessageStream] = WebSocketStream,
                 clock: Callable[[], float] = time.monotonic,
                 provider: str | None = None,
                 message_timeout: float = MESSAGE_TIMEOUT,
                 turn_timeout: float = TURN_TIMEOUT,
                 text_chars: int = TEXT_CHARS) -> None:
        self.url = url if url is not None else get_settings().ahp_url
        self.connect = connect
        self.clock = clock
        self.provider = provider
        self.message_timeout = message_timeout
        self.turn_timeout = turn_timeout
        self.text_chars = text_chars

    def ready(self) -> str | None:
        """None when a host address is set; otherwise what to set."""
        if not self.url:
            return ("no AHP host is named; set QMCP_AHP_URL to its address, for example"
                    " ws://127.0.0.1:8123")
        return None

    def run(self, brief: Brief, on_event: OnEvent | None = None) -> AgentOutcome:
        started = self.clock()
        detail: dict[str, Any] = {"url": self.url, "denied": []}

        def outcome(text: str, code: int) -> AgentOutcome:
            if on_event:
                on_event("finished", str(code))
            denied = detail["denied"]
            if denied:
                names = ", ".join(call["tool"] for call in denied)
                text = (f"{text}\n\n{len(denied)} tool call(s) the host asked to confirm"
                        f" were denied, because this runtime approves nothing: {names}.")
            detail["denied_count"] = len(denied)
            return AgentOutcome(
                text=text, exit_code=code, elapsed_seconds=self.clock() - started,
                spent=unknown("the host does not report what a turn spent to this runtime"),
                detail=detail)

        missing = self.ready()
        if missing:
            return outcome(f"Nothing ran: {missing}.", 1)
        if on_event:
            on_event("started", f"an AHP host at {self.url}, with {len(brief.history)} earlier"
                                " instruction(s) from qmcp's record")
        try:
            stream = self.connect(self.url)
        except Unreachable as exc:
            return outcome(f"No AHP host answered at {self.url} ({exc}), so nothing ran."
                           " Start the host, or set QMCP_AHP_URL to where it listens.", 1)
        exchange = _Exchange(stream, self.clock, self.message_timeout, self.turn_timeout)
        turn: dict[str, Any] = {}
        try:
            return outcome(self._turn(exchange, brief, detail, turn, on_event), 0)
        except Failed as exc:
            return outcome(str(exc), 1)
        except TurnTimeout:
            self._cancel(exchange, turn)
            return outcome(f"The AHP host at {self.url} did not finish within"
                           f" {self.turn_timeout:g} seconds; the run was given up.", 1)
        except TimeoutError:
            self._cancel(exchange, turn)
            return outcome(f"The AHP host at {self.url} sent nothing for"
                           f" {self.message_timeout:g} seconds; the run was given up.", 1)
        except Closed:
            return outcome(f"The AHP host at {self.url} closed the connection before the"
                           " turn finished.", 1)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            self._cancel(exchange, turn)
            return outcome(f"The AHP host at {self.url} sent a message this runtime could"
                           f" not read ({type(exc).__name__}).", 1)
        finally:
            stream.close()

    def _cancel(self, exchange: _Exchange, turn: dict[str, Any]) -> None:
        """Tell the host to stop a turn this run is walking away from."""
        if not turn:
            return
        try:
            exchange.dispatch(turn["chat"], {
                "type": "chat/turnCancelled", "turnId": turn["id"],
                "duration": int((self.clock() - exchange.started) * 1000)})
        except Closed:
            pass

    def _turn(self, exchange: _Exchange, brief: Brief, detail: dict[str, Any],
              turn: dict[str, Any], on_event: OnEvent | None) -> str:
        """The whole exchange; returns the text, or raises `Failed`."""
        client_id = f"qmcp-{uuid.uuid4()}"
        session = f"ahp-session:/{uuid.uuid4()}"
        chat = f"ahp-chat:/{uuid.uuid4()}"
        turn_id = str(uuid.uuid4())
        detail.update(session=session, chat=chat)

        initialized = exchange.request("initialize", {
            "channel": ROOT, "protocolVersions": list(PROTOCOL_VERSIONS),
            "clientId": client_id, "clientInfo": {"name": "qmcp"}})
        detail["protocol_version"] = (initialized or {}).get("protocolVersion")

        create: dict[str, Any] = {"channel": session,
                                  "workingDirectories": [brief.cwd.resolve().as_uri()]}
        if self.provider:
            create["provider"] = self.provider
        exchange.request("createSession", create)
        subscribed = exchange.request("subscribe", {"channel": session}) or {}
        state = (subscribed.get("snapshot") or {}).get("state") or {}
        lifecycle = state.get("lifecycle", "creating")
        failure = state.get("creationError")
        while lifecycle == "creating":
            message = exchange.notification()
            params = message.get("params") or {}
            if message["method"] != "action" or params.get("channel") != session:
                continue
            action = params.get("action") or {}
            if action.get("type") == "session/ready":
                lifecycle = "ready"
            elif action.get("type") == "session/creationFailed":
                lifecycle, failure = "failed", action.get("error")
        if lifecycle != "ready":
            raise Failed(f"The AHP host at {self.url} could not create a session:"
                         f" {_error(failure)}.")

        exchange.request("createChat", {"channel": session, "chat": chat})
        exchange.request("subscribe", {"channel": chat})
        started_seq = exchange.dispatch(chat, {
            "type": "chat/turnStarted", "turnId": turn_id,
            "startedAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "message": {"text": brief.prompt(), "origin": {"kind": "user"}}})
        turn.update(chat=chat, id=turn_id)

        parts: dict[str, str] = {}
        tools: dict[str, str] = {}

        def deny(call_id: str, action: dict[str, Any]) -> None:
            tool = tools.get(call_id, call_id)
            exchange.dispatch(chat, action)
            detail["denied"].append({"tool": tool, "toolCallId": call_id,
                                     "what": action["type"]})
            if on_event:
                on_event("output", f"denied {tool}")

        def deny_call(call_id: str) -> None:
            deny(call_id, {"type": "chat/toolCallConfirmed", "turnId": turn_id,
                           "toolCallId": call_id, "approved": False, "reason": "denied",
                           "reasonMessage": "qmcp approves no tool call on a person's behalf"})

        while True:
            message = exchange.notification()
            params = message.get("params") or {}
            if message["method"] == "auth/required":
                raise Failed(f"The AHP host at {self.url} asks for authentication, which"
                             " this runtime does not do.")
            if message["method"] != "action" or params.get("channel") != chat:
                continue
            action = params.get("action") or {}
            kind = action.get("type")
            origin = params.get("origin") or {}
            if (params.get("rejectionReason") and origin.get("clientId") == client_id
                    and origin.get("clientSeq") == started_seq):
                turn.clear()
                raise Failed(f"The AHP host at {self.url} rejected the turn:"
                             f" {params['rejectionReason']}.")
            if kind == "chat/inputRequested":
                # Carries no `turnId`; this chat has the one turn.
                self._cancel(exchange, turn)
                raise Failed(f"The AHP host at {self.url} asked for input mid-turn, and this"
                             " runtime has nobody to answer it; the turn was cancelled.")
            if action.get("turnId") != turn_id:
                continue
            if kind == "chat/responsePart":
                part = action.get("part") or {}
                if part.get("kind") == "markdown":
                    parts[part["id"]] = str(part.get("content") or "")
                elif part.get("kind") == "toolCall":
                    call = part.get("toolCall") or {}
                    tools[call["toolCallId"]] = str(call.get("toolName") or call["toolCallId"])
                    if call.get("status") == "pending-confirmation":
                        deny_call(call["toolCallId"])
            elif kind == "chat/delta":
                parts[action["partId"]] = parts.get(action["partId"], "") + str(action["content"])
            elif kind == "chat/toolCallStart":
                tools[action["toolCallId"]] = str(action.get("toolName") or action["toolCallId"])
            elif kind == "chat/toolCallReady":
                # Without `confirmed` the call is held in `pending-confirmation`.
                if not action.get("confirmed"):
                    deny_call(action["toolCallId"])
            elif kind == "chat/toolCallComplete":
                if action.get("requiresResultConfirmation"):
                    deny(action["toolCallId"], {
                        "type": "chat/toolCallResultConfirmed", "turnId": turn_id,
                        "toolCallId": action["toolCallId"], "approved": False})
            elif kind == "chat/turnComplete":
                text = "\n\n".join(text for text in parts.values() if text.strip()).strip()
                if not text:
                    raise Failed(f"The AHP host at {self.url} completed the turn without"
                                 " any text.")
                return text
            elif kind == "chat/error":
                raise Failed(f"The AHP host at {self.url} ended the turn with an error:"
                             f" {_error((action.get('part') or {}).get('error'))}.")
            elif kind == "chat/turnCancelled":
                raise Failed(f"The turn was cancelled on the AHP host at {self.url}.")
            if sum(len(text) for text in parts.values()) > self.text_chars:
                detail["collected"] = "\n\n".join(parts.values())[:self.text_chars]
                self._cancel(exchange, turn)
                raise Failed(f"The AHP host at {self.url} streamed more than"
                             f" {self.text_chars} characters; the turn was cancelled and"
                             " what arrived is kept in the record's detail.")
