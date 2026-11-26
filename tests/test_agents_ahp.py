"""An Agent Host Protocol host as a runtime: the exchange, the denials and the bounds.

The host is `_Host`, in this process, behind the adapter's `MessageStream`: it
answers each request as the protocol's types say a host does, keeps every
message it was sent, and plays the turn a test scripts. Its clock is the
runtime's clock, and a wait with nothing to deliver moves it by the timeout it
was given, so no test waits in real time. One test opens a real socket, to a
loopback port nothing listens on.
"""

from __future__ import annotations

import socket
import subprocess
import sys
from collections import deque

from qmcp.integrations.agents import Brief, runtime_class, runtime_named
from qmcp.integrations.agents.adapters import ahp

URL = "ws://ahp.test:8123"


def _text(part_id, content=""):
    return {"type": "chat/responsePart", "part": {"kind": "markdown", "id": part_id,
                                                  "content": content}}


def _delta(part_id, content):
    return {"type": "chat/delta", "partId": part_id, "content": content}


COMPLETE = {"type": "chat/turnComplete", "duration": 12}
START_TOOL = {"type": "chat/toolCallStart", "toolCallId": "c-1", "toolName": "run_in_terminal",
              "displayName": "Run in terminal"}
READY_TOOL = {"type": "chat/toolCallReady", "toolCallId": "c-1",
              "invocationMessage": "rm -rf build"}


class _Host:
    """An AHP host over the injected stream.

    `turn` is what it streams once a turn starts; `after_denial` is what it
    streams once a tool call or a tool result is answered denied, and an
    approval is answered with "The tool ran.". `creation` is how the
    session's creation ends: `ready`, `failed`, or `snapshot-failed` for a
    session already failed when it is subscribed to.
    """

    def __init__(self, turn=(), after_denial=(), creation="ready", refuse=None, reject=None):
        self.turn = list(turn)
        self.after_denial = list(after_denial)
        self.creation = creation
        self.refuse = refuse or {}
        self.reject = reject
        self.sent: list[dict] = []
        self.queue: deque[dict] = deque()
        self.now = 0.0
        self.seq = 0
        self.closed = False
        self.session = self.chat = self.turn_id = None
        self.connected_to: list[str] = []

    # --- the stream ------------------------------------------------------------

    def connect(self, url):
        self.connected_to.append(url)
        return self

    def clock(self):
        return self.now

    def receive(self, timeout):
        if not self.queue:
            self.now += timeout
            raise TimeoutError
        return self.queue.popleft()

    def close(self):
        self.closed = True

    def send(self, message):
        assert message["jsonrpc"] == "2.0"
        self.sent.append(message)
        method, params = message.get("method"), message.get("params") or {}
        if method is None:
            return
        if method in self.refuse:
            self.queue.append({"jsonrpc": "2.0", "id": message["id"], "error": self.refuse[method]})
        elif method == "initialize":
            self._result(message, {"protocolVersion": params["protocolVersions"][0],
                                   "serverSeq": self.seq, "snapshots": []})
        elif method == "createSession":
            self.session = params["channel"]
            self._result(message, None)
        elif method == "createChat":
            self.chat = params["chat"]
            self._result(message, None)
        elif method == "subscribe" and params["channel"] == self.session:
            failed = {"errorType": "backend", "message": "the agent backend did not start"}
            state = {"lifecycle": "creating"}
            if self.creation == "snapshot-failed":
                state = {"lifecycle": "failed", "creationError": failed}
            self._result(message, {"snapshot": {"resource": self.session, "state": state,
                                                "fromSeq": self.seq}})
            if self.creation == "ready":
                self._action(self.session, {"type": "session/ready"})
            elif self.creation == "failed":
                self._action(self.session, {"type": "session/creationFailed", "error": failed})
        elif method == "subscribe":
            self._result(message, {"snapshot": {"resource": params["channel"],
                                                "state": {"turns": []}, "fromSeq": self.seq}})
        elif method == "dispatchAction":
            self._dispatched(params)

    def _result(self, message, result):
        self.queue.append({"jsonrpc": "2.0", "id": message["id"], "result": result})

    def _action(self, channel, action, origin=None, rejection=None):
        self.seq += 1
        params = {"channel": channel, "action": action, "serverSeq": self.seq, "origin": origin}
        if rejection:
            params["rejectionReason"] = rejection
        self.queue.append({"jsonrpc": "2.0", "method": "action", "params": params})

    def _play(self, actions):
        for action in actions:
            if "method" in action:  # a request or a notification of the host's own
                self.queue.append({"jsonrpc": "2.0", **action})
            elif action["type"] == "chat/inputRequested":
                self._action(self.chat, action)
            else:
                self._action(self.chat, {**action, "turnId": self.turn_id})

    def _dispatched(self, params):
        action = params["action"]
        origin = {"clientId": self.client_id, "clientSeq": params["clientSeq"]}
        if action["type"] == "chat/turnStarted":
            self.turn_id = action["turnId"]
            self._action(params["channel"], action, origin, rejection=self.reject)
            if not self.reject:
                self._play(self.turn)
        elif action["type"] in ("chat/toolCallConfirmed", "chat/toolCallResultConfirmed"):
            self._action(params["channel"], action, origin)
            self._play([_text("p-ran", "The tool ran."), COMPLETE] if action["approved"]
                       else self.after_denial)

    # --- what a test reads ---------------------------------------------------------

    @property
    def client_id(self):
        return self.requests("initialize")[0]["clientId"]

    def methods(self):
        return [m["method"] for m in self.sent if "method" in m]

    def requests(self, method):
        return [m["params"] for m in self.sent if m.get("method") == method]

    def dispatched(self, kind=None):
        return [p["action"] for p in self.requests("dispatchAction")
                if kind is None or p["action"]["type"] == kind]


def _brief(tmp_path):
    return Brief(instruction="What is qmcp?", cwd=tmp_path, project="qmcp")


def _run(host, tmp_path, events=None, **options):
    runtime = ahp.Runtime(url=URL, connect=host.connect, clock=host.clock, **options)
    on_event = (lambda state, text: events.append((state, text))) if events is not None else None
    return runtime.run(_brief(tmp_path), on_event)


# --- a turn that completes --------------------------------------------------------------


def test_a_completed_turn_returns_the_text_the_host_streamed(tmp_path):
    """Mutation: ignore `chat/delta` -- red, the text is the parts' first
    content only."""
    host = _Host(turn=[_text("p-1", "qmcp is "), _delta("p-1", "the local"),
                       _delta("p-1", " backend."), _text("p-2"), _delta("p-2", "It serves MCP."),
                       COMPLETE])
    events: list = []

    outcome = _run(host, tmp_path, events)

    assert outcome.text == "qmcp is the local backend.\n\nIt serves MCP."
    assert outcome.exit_code == 0 and outcome.succeeded
    assert isinstance(outcome.spent, dict)  # unknown, with a reason; never zero
    assert outcome.detail["denied"] == [] and outcome.detail["denied_count"] == 0
    assert outcome.detail["protocol_version"] == ahp.PROTOCOL_VERSIONS[0]
    assert [state for state, _ in events] == ["started", "finished"] and events[-1][1] == "0"
    assert host.connected_to == [URL] and host.closed


def test_the_exchange_is_the_protocols_in_order_and_carries_the_brief(tmp_path):
    """Mutation: send the brief's instruction in place of its prompt -- red;
    every runtime is handed the same words."""
    host = _Host(turn=[_text("p-1", "Yes."), COMPLETE])

    _run(host, tmp_path)

    assert host.methods() == ["initialize", "createSession", "subscribe", "createChat",
                              "subscribe", "dispatchAction"]
    (initialize,) = host.requests("initialize")
    assert initialize["channel"] == "ahp-root://"
    assert initialize["protocolVersions"] == list(ahp.PROTOCOL_VERSIONS)
    (session,) = host.requests("createSession")
    assert session["channel"].startswith("ahp-session:/")
    assert session["workingDirectories"] == [tmp_path.resolve().as_uri()]
    assert "provider" not in session
    (chat,) = host.requests("createChat")
    assert chat["channel"] == session["channel"] and chat["chat"].startswith("ahp-chat:/")
    assert [p["channel"] for p in host.requests("subscribe")] == [session["channel"], chat["chat"]]
    (dispatch,) = host.requests("dispatchAction")
    assert dispatch["channel"] == chat["chat"] and dispatch["clientSeq"] == 1
    started = dispatch["action"]
    assert started["type"] == "chat/turnStarted" and started["turnId"]
    assert started["startedAt"].endswith("Z")
    assert started["message"] == {"text": _brief(tmp_path).prompt(), "origin": {"kind": "user"}}


def test_text_of_another_turn_or_another_channel_is_not_collected(tmp_path):
    """Mutation: drop the turn filter -- red, another turn's text is said back."""
    stray = {"jsonrpc": "2.0", "method": "action", "params": {
        "channel": "ahp-chat:/somebody-elses", "serverSeq": 99, "origin": None,
        "action": {"type": "chat/delta", "turnId": "other", "partId": "p-1", "content": "NO"}}}
    host = _Host(turn=[_text("p-1", "Yes."), stray, COMPLETE])
    original = host._play

    def play(actions):
        host._action(host.chat, {"type": "chat/delta", "turnId": "an-earlier-turn",
                                 "partId": "p-0", "content": "NO"})
        original(actions)

    host._play = play

    assert _run(host, tmp_path).text == "Yes."


# --- a turn that does not ---------------------------------------------------------------


def test_a_turn_that_errors_is_a_failure_carrying_the_hosts_message(tmp_path):
    """Mutation: treat `chat/error` as `chat/turnComplete` -- red, a failed turn
    reads as done."""
    host = _Host(turn=[_text("p-1", "Starting"), {
        "type": "chat/error", "duration": 5,
        "part": {"kind": "error", "error": {"errorType": "model",
                                            "message": "the model is overloaded"}}}])
    events: list = []

    outcome = _run(host, tmp_path, events)

    assert outcome.exit_code == 1 and not outcome.succeeded
    assert "ended the turn with an error: the model is overloaded" in outcome.text
    assert URL in outcome.text and events[-1] == ("finished", "1")
    assert host.closed


def test_a_session_whose_creation_fails_starts_no_turn(tmp_path):
    """Mutation: read `session/creationFailed` as ready -- red, a turn is
    dispatched into a session that does not exist."""
    host = _Host(creation="failed")

    outcome = _run(host, tmp_path)

    assert outcome.exit_code == 1
    assert "could not create a session: the agent backend did not start" in outcome.text
    assert "createChat" not in host.methods() and host.dispatched() == []


def test_a_session_already_failed_in_its_snapshot_starts_no_turn(tmp_path):
    """Mutation: take the lifecycle as `creating` whatever the snapshot says --
    red, the run waits out the message timeout for an action already past."""
    host = _Host(creation="snapshot-failed")

    outcome = _run(host, tmp_path)

    assert "could not create a session: the agent backend did not start" in outcome.text
    assert host.now == 0.0 and host.dispatched() == []


def test_a_refused_request_is_a_sentence_naming_the_method(tmp_path):
    """Mutation: ignore a response's `error` -- red, the run goes on to
    subscribe to a session the host refused."""
    host = _Host(refuse={"createSession": {"code": -32002, "message": "No agent for provider"}})

    outcome = _run(host, tmp_path, provider="nobody")

    assert outcome.exit_code == 1
    assert "refused createSession: No agent for provider" in outcome.text
    assert host.requests("createSession")[0]["provider"] == "nobody"
    assert host.methods() == ["initialize", "createSession"]


def test_a_rejected_turn_is_a_failure_and_is_not_cancelled(tmp_path):
    """Mutation: ignore `rejectionReason` -- red, the run waits for a turn the
    host never started."""
    host = _Host(reject="a turn is already active")

    outcome = _run(host, tmp_path)

    assert outcome.exit_code == 1 and "rejected the turn: a turn is already active" in outcome.text
    assert host.now == 0.0 and host.dispatched("chat/turnCancelled") == []


# --- it approves nothing ---------------------------------------------------------------


def test_a_tool_call_held_for_confirmation_is_denied_and_named(tmp_path):
    """Mutation: answer `approved: True` -- red. This is the consent rule: a
    person approves, and the adapter never stands in for one."""
    host = _Host(turn=[START_TOOL, READY_TOOL],
                 after_denial=[_text("p-1", "I could not run it."), COMPLETE])
    events: list = []

    outcome = _run(host, tmp_path, events)

    (answer,) = host.dispatched("chat/toolCallConfirmed")
    assert answer["approved"] is False and answer["reason"] == "denied"
    assert (answer["turnId"], answer["toolCallId"]) == (host.turn_id, "c-1")
    assert "confirmed" not in answer
    assert outcome.exit_code == 0
    assert outcome.text.startswith("I could not run it.")
    assert "1 tool call(s) the host asked to confirm were denied" in outcome.text
    assert "run_in_terminal" in outcome.text
    assert outcome.detail["denied_count"] == 1
    assert outcome.detail["denied"] == [{"tool": "run_in_terminal", "toolCallId": "c-1",
                                         "what": "chat/toolCallConfirmed"}]
    assert ("output", "denied run_in_terminal") in events


def test_a_tool_call_the_host_confirmed_itself_is_not_answered(tmp_path):
    """Mutation: deny every `chat/toolCallReady` -- red; the host rejects an
    answer to a call that is not held, and the count of denials is wrong."""
    host = _Host(turn=[START_TOOL, {**READY_TOOL, "confirmed": "setting"},
                       _text("p-1", "Done."), COMPLETE])

    outcome = _run(host, tmp_path)

    assert host.dispatched("chat/toolCallConfirmed") == []
    assert outcome.text == "Done." and outcome.detail["denied"] == []


def test_a_tool_call_that_arrives_already_held_is_denied(tmp_path):
    """Mutation: read only `chat/toolCallReady` -- red, a call that arrives as
    a response part in `pending-confirmation` is left held until the timeout."""
    held = {"type": "chat/responsePart", "part": {"kind": "toolCall", "toolCall": {
        "toolCallId": "c-7", "toolName": "write_file", "displayName": "Write file",
        "status": "pending-confirmation", "invocationMessage": "write README.md"}}}
    host = _Host(turn=[held], after_denial=[_text("p-1", "Left as it was."), COMPLETE])

    outcome = _run(host, tmp_path)

    (answer,) = host.dispatched("chat/toolCallConfirmed")
    assert answer["approved"] is False and answer["toolCallId"] == "c-7"
    assert outcome.detail["denied"][0]["tool"] == "write_file"


def test_a_tool_result_held_for_confirmation_is_denied(tmp_path):
    """Mutation: answer the result `approved: True` -- red."""
    host = _Host(turn=[START_TOOL, {**READY_TOOL, "confirmed": "not-needed"},
                       {"type": "chat/toolCallComplete", "toolCallId": "c-1",
                        "result": {"success": True}, "requiresResultConfirmation": True}],
                 after_denial=[_text("p-1", "The result was not used."), COMPLETE])

    outcome = _run(host, tmp_path)

    (answer,) = host.dispatched("chat/toolCallResultConfirmed")
    assert answer == {"type": "chat/toolCallResultConfirmed", "turnId": host.turn_id,
                      "toolCallId": "c-1", "approved": False}
    assert outcome.detail["denied"] == [{"tool": "run_in_terminal", "toolCallId": "c-1",
                                         "what": "chat/toolCallResultConfirmed"}]


def test_a_request_for_input_cancels_the_turn(tmp_path):
    """Mutation: skip `chat/inputRequested` -- red, the run waits out the
    message timeout on a question nobody can answer."""
    host = _Host(turn=[{"type": "chat/inputRequested", "request": {"id": "q-1"}}])

    outcome = _run(host, tmp_path)

    assert outcome.exit_code == 1 and "asked for input" in outcome.text
    (cancel,) = host.dispatched("chat/turnCancelled")
    assert cancel["turnId"] == host.turn_id and host.now == 0.0


def test_a_request_the_host_makes_of_the_client_is_refused(tmp_path):
    """Mutation: drop a host's request unanswered -- red; a host waiting on a
    read of the person's files would wait for ever."""
    asks = {"method": "resourceRead", "id": 70, "params": {"channel": "ahp-root://",
                                                          "uri": "file:///etc/passwd"}}
    host = _Host(turn=[asks, _text("p-1", "Done."), COMPLETE])

    outcome = _run(host, tmp_path)

    (reply,) = [m for m in host.sent if "method" not in m]
    assert reply["id"] == 70 and reply["error"]["code"] == -32601 and "result" not in reply
    assert outcome.text == "Done."


# --- the bounds ---------------------------------------------------------------------------


def test_a_host_that_never_answers_is_given_up_and_the_turn_cancelled(tmp_path):
    """Mutation: do not dispatch `chat/turnCancelled` on a timeout -- red; the
    host's agent would keep working in the clone with nobody reading."""
    host = _Host(turn=[_text("p-1", "Thinking")])

    outcome = _run(host, tmp_path, message_timeout=30.0)

    assert outcome.exit_code == 1
    assert f"The AHP host at {URL} sent nothing for 30 seconds" in outcome.text
    assert host.now == 30.0 and outcome.elapsed_seconds == 30.0
    (cancel,) = host.dispatched("chat/turnCancelled")
    assert cancel == {"type": "chat/turnCancelled", "turnId": host.turn_id, "duration": 30_000}
    assert host.closed


def test_a_host_that_does_not_answer_the_handshake_is_given_up(tmp_path):
    """Mutation: wait without a timeout -- red (the stand-in host raises on a
    wait that is not bounded by the message timeout)."""
    host = _Host()
    host.send = lambda message: host.sent.append(message)  # hears, never replies

    outcome = _run(host, tmp_path, message_timeout=7.0)

    assert outcome.exit_code == 1 and "sent nothing for 7 seconds" in outcome.text
    assert host.now == 7.0 and host.dispatched() == []


def test_a_turn_that_outlasts_its_cap_is_given_up(tmp_path):
    """Mutation: drop the deadline from the wait -- red, a host that streams
    slowly for ever is read for ever."""
    host = _Host(turn=[_text("p-1", "")])
    receive = host.receive

    def drip(timeout):  # one delta every 20 seconds, without end
        if not host.queue:
            host.now += min(timeout, 20.0)
            if timeout < 20.0:
                raise TimeoutError
            return {"jsonrpc": "2.0", "method": "action", "params": {
                "channel": host.chat, "serverSeq": 0, "origin": None,
                "action": _delta("p-1", ".") | {"turnId": host.turn_id}}}
        return receive(timeout)

    host.receive = drip

    outcome = _run(host, tmp_path, message_timeout=30.0, turn_timeout=90.0)

    assert outcome.exit_code == 1 and "did not finish within 90 seconds" in outcome.text
    assert host.now == 90.0
    assert len(host.dispatched("chat/turnCancelled")) == 1


def test_more_text_than_the_cap_cancels_the_turn_and_keeps_what_arrived(tmp_path):
    """Mutation: drop the cap -- red, the whole stream is said back."""
    host = _Host(turn=[_text("p-1", "a" * 40), _delta("p-1", "b" * 40), _delta("p-1", "c" * 40),
                       COMPLETE])

    outcome = _run(host, tmp_path, text_chars=50)

    assert outcome.exit_code == 1 and "streamed more than 50 characters" in outcome.text
    assert outcome.detail["collected"] == "a" * 40 + "b" * 10
    assert len(host.dispatched("chat/turnCancelled")) == 1


def test_a_message_that_cannot_be_read_is_a_sentence(tmp_path):
    """Mutation: let the error out of `run` -- red, a traceback where a
    sentence belongs."""
    host = _Host(turn=[{"type": "chat/delta", "content": "no part named"}])

    outcome = _run(host, tmp_path)

    assert outcome.exit_code == 1 and "could not read (KeyError)" in outcome.text


def test_a_connection_the_host_closes_is_a_sentence(tmp_path):
    """Mutation: let `Closed` out of `run` -- red."""
    host = _Host(turn=[_text("p-1", "Half an ans")])
    receive = host.receive

    def drop(timeout):
        if not host.queue:
            raise ahp.Closed("going away")
        return receive(timeout)

    host.receive = drop

    outcome = _run(host, tmp_path)

    assert outcome.exit_code == 1
    assert "closed the connection before the turn finished" in outcome.text


# --- a host that is not there -----------------------------------------------------------


def test_an_unreachable_host_is_a_sentence_with_the_address_tried(tmp_path):
    """Mutation: let `Unreachable` out of `run` -- red, a traceback where a
    sentence belongs."""
    def connect(url):
        raise ahp.Unreachable("ConnectionRefusedError")

    events: list = []
    outcome = ahp.Runtime(url=URL, connect=connect).run(
        _brief(tmp_path), lambda state, text: events.append((state, text)))

    assert outcome.exit_code == 1
    assert outcome.text.startswith(f"No AHP host answered at {URL} (ConnectionRefusedError)")
    assert "Traceback" not in outcome.text and events[-1] == ("finished", "1")


def test_the_websocket_transport_reports_a_port_nobody_listens_on(tmp_path):
    """The real transport, to a loopback port bound and closed again, so nothing
    answers there. Mutation: let the transport's `OSError` through unwrapped --
    red, it reaches `act` as a raised exception."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"ws://127.0.0.1:{port}"

    outcome = ahp.Runtime(url=url).run(_brief(tmp_path))

    assert outcome.exit_code == 1 and outcome.text.startswith(f"No AHP host answered at {url} (")


def test_no_address_runs_nothing_and_says_what_to_set(tmp_path, monkeypatch):
    """Mutation: fall back to a default address -- red; a run nobody pointed
    at a host must not find one."""
    from qmcp.config import get_settings

    monkeypatch.delenv("QMCP_AHP_URL", raising=False)
    get_settings.cache_clear()
    connected: list = []
    runtime = ahp.Runtime(connect=connected.append)

    outcome = runtime.run(_brief(tmp_path))

    assert runtime.url is None and "QMCP_AHP_URL" in runtime.ready()
    assert outcome.exit_code == 1 and "QMCP_AHP_URL" in outcome.text and connected == []
    get_settings.cache_clear()


def test_the_address_is_read_from_the_environment(monkeypatch):
    """Mutation: rename the setting -- red."""
    from qmcp.config import get_settings

    monkeypatch.setenv("QMCP_AHP_URL", "ws://127.0.0.1:9")
    get_settings.cache_clear()
    try:
        runtime = runtime_named(ahp.NAME)
        assert runtime.url == "ws://127.0.0.1:9" and runtime.ready() is None
    finally:
        get_settings.cache_clear()


# --- the registry and the extra ---------------------------------------------------------


def test_the_registry_finds_the_adapter_by_its_name():
    """Mutation: give the module another `NAME` -- red, `--runtime ahp` finds
    no adapter."""
    assert ahp.NAME == "ahp" and runtime_class("ahp") is ahp.Runtime


def test_the_adapter_imports_without_the_websocket_package():
    """In a fresh interpreter where `websockets` cannot be imported, the
    adapter imports, is discovered, and its transport says which extra to
    install. Mutation: import `websockets` at the top of the module -- red."""
    script = (
        "import sys\n"
        "sys.modules['websockets'] = None\n"
        "from qmcp.integrations.agents import runtime_names\n"
        "from qmcp.integrations.agents.adapters import ahp\n"
        "assert 'ahp' in runtime_names()\n"
        "try:\n"
        "    ahp.WebSocketStream('ws://127.0.0.1:9')\n"
        "except ahp.Unreachable as exc:\n"
        "    print(exc)\n"
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          timeout=120)

    assert done.returncode == 0, done.stderr
    assert "uv sync --extra ahp" in done.stdout
