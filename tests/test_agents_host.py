"""qmcp as an Agent Host Protocol host: the handshake, a turn as an instruction
acted on, its consent shown and relayed, the one-act rule, and what is refused.

The client is `_Client`, in this process: it sends whole JSON messages to a
`Connection` and reads its `outbox`. The instruction record is `_Acts`, which
keeps what was recorded, started and answered and plays the looks a test
scripts, so no database and no process is needed. Some tests go further:
`ServerActs` over a database file with the run tracker's launcher stood in
for, `ServerActs` against the test server's own queue routes, the WebSocket
route on the test server, and qmcp's own client adapter attached to the hub.
Those last prove the two halves agree with each other; both were read
from the same type definitions, so it does not prove either against another
program's host or client.

Each test names the mutation it was seen to fail under.
"""

from __future__ import annotations

import pytest

from qmcp.integrations.agents import Brief, runtime_names
from qmcp.integrations.agents import host as ahp_host
from qmcp.integrations.agents.adapters import ahp
from qmcp.integrations.agents.host import (
    Asked,
    Busy,
    Hub,
    Look,
    Refused,
    ServerActs,
    negotiate,
)

SESSION, CHAT = "ahp-session:/s-1", "ahp-chat:/c-1"
RUNTIMES = ("scripted", "local")
CONSENT = Asked(id="instruction-i-1", prompt="Run scripted once in qmcp?",
                options=("approve", "hold"))


class _Acts:
    """The instruction record, stood in for."""

    def __init__(self, looks=(), running=None, start_busy=None, consent=None,
                 refuses=None):
        self.recorded: list[tuple[str, dict]] = []
        self.started: list[tuple[str, str, object]] = []
        self.answered: list[tuple[str, str]] = []
        self.looks = list(looks)
        self.running = running
        self.start_busy = start_busy
        self.consent = consent
        self.refuses = refuses

    def busy(self):
        return self.running

    def record(self, text, detail):
        self.recorded.append((text, detail))
        return f"i-{len(self.recorded)}"

    def start(self, instruction_id, runtime, cwd):
        if self.start_busy:
            raise Busy(self.start_busy)
        self.started.append((instruction_id, runtime, cwd))

    def look(self, instruction_id):
        return self.looks.pop(0) if len(self.looks) > 1 else self.looks[0]

    def asked(self, request_id):
        return self.consent

    def answer(self, request_id, response):
        if self.refuses:
            raise Refused(self.refuses)
        self.answered.append((request_id, response))


class _Client:
    """One attached client: requests by number, and everything it was sent."""

    def __init__(self, hub, client_id="client-a", loopback=True):
        self.connection = hub.connect(loopback)
        self.client_id = client_id
        self.ids = 0
        self.seq = 0

    def request(self, method, **params):
        self.ids += 1
        self.connection.handle({"jsonrpc": "2.0", "id": self.ids, "method": method,
                                "params": params})
        answer = next(m for m in self.connection.outbox if m.get("id") == self.ids)
        self.connection.outbox.remove(answer)
        return answer

    def result(self, method, **params):
        answer = self.request(method, **params)
        assert "error" not in answer, answer
        return answer["result"]

    def error(self, method, **params):
        return self.request(method, **params)["error"]

    def dispatch(self, channel, action):
        self.seq += 1
        self.connection.handle({"jsonrpc": "2.0", "method": "dispatchAction", "params": {
            "channel": channel, "clientSeq": self.seq, "action": action}})
        return self.seq

    def actions(self):
        """Every action waiting, oldest first, taken off the outbox."""
        sent = [m["params"] for m in self.connection.outbox if m.get("method") == "action"]
        self.connection.outbox.clear()
        return sent

    def initialize(self, versions=ahp.PROTOCOL_VERSIONS, **extra):
        return self.result("initialize", channel=ahp.ROOT, protocolVersions=list(versions),
                           clientId=self.client_id, **extra)


def _hub(acts=None, **options):
    now = [0.0]
    hub = Hub(acts or _Acts(looks=[Look(status="recorded", running=False)]),
              runtimes=lambda: RUNTIMES, clock=lambda: now[0], **options)
    hub.now = now
    return hub


def _attached(hub, tmp_path, client_id="client-a", provider="scripted", loopback=True):
    """A client with a session and a chat, subscribed to both."""
    client = _Client(hub, client_id, loopback)
    client.initialize()
    client.result("createSession", channel=SESSION, provider=provider,
                  workingDirectories=[tmp_path.as_uri()])
    client.result("subscribe", channel=SESSION)
    client.result("createChat", channel=SESSION, chat=CHAT)
    client.result("subscribe", channel=CHAT)
    client.actions()
    return client


def _turn(text="What is qmcp?", turn_id="t-1"):
    return {"type": "chat/turnStarted", "turnId": turn_id, "startedAt": "2026-01-01T00:10:00Z",
            "message": {"text": text, "origin": {"kind": "user"}}}


# --- the handshake -----------------------------------------------------------


@pytest.mark.parametrize("offered,chosen", [
    (["1.0.0", "0.9.0"], "1.0.0"),
    (["0.9.0"], "0.9.0"),
    (["1.2.0", "1.0.0"], "1.2.0"),     # the same major, above the base
    (["0.9.3", "0.8.0"], "0.9.3"),     # below 1, the same minor
    (["0.10.0", "2.0.0"], None),       # another minor below 1; another major
    (["1.0", "v1.0.0", "01.0.0"], None),
    ([], None),
])
def test_the_version_is_the_highest_offered_that_is_spoken(offered, chosen):
    """Mutation: return the first offered -- red, the `1.2.0` and refusal cases."""
    assert negotiate(offered) == chosen


def test_the_versions_spoken_are_the_client_roles():
    """One list for both roles. Mutation: a list of the host's own -- red."""
    assert ahp_host.PROTOCOL_VERSIONS is ahp.PROTOCOL_VERSIONS


def test_a_client_that_offers_no_spoken_version_is_refused_and_closed():
    """Mutation: answer with the host's first version anyway -- red."""
    client = _Client(_hub())
    refused = client.error("initialize", channel=ahp.ROOT, protocolVersions=["2.0.0"],
                           clientId="client-a")
    assert refused["code"] == ahp_host.UNSUPPORTED_PROTOCOL_VERSION
    assert refused["data"] == {"supportedVersions": list(ahp.PROTOCOL_VERSIONS)}
    assert client.connection.closing
    assert client.error("subscribe", channel=ahp.ROOT)["code"] == ahp_host.INVALID_REQUEST


def test_nothing_is_served_before_the_handshake():
    """Mutation: drop the `connection.version is None` check -- red."""
    client = _Client(_hub())
    refused = client.error("createSession", channel=SESSION, provider="scripted")
    assert refused["code"] == ahp_host.INVALID_REQUEST and "initialize" in refused["message"]


def test_the_handshake_declares_what_is_served_and_nothing_else():
    """Mutation: add `telemetry: {}` to the result -- red. What qmcp does not
    serve is left out, never present and empty."""
    result = _Client(_hub(version="9.9.9")).initialize()
    assert set(result) == {"protocolVersion", "serverSeq", "serverInfo", "snapshots"}
    assert result["serverInfo"] == {"name": "qmcp", "version": "9.9.9"}
    assert result["snapshots"] == []


def test_the_agents_advertised_are_qmcps_runtime_names():
    """Mutation: advertise a fixed list -- red."""
    client = _Client(_hub())
    client.initialize()
    state = client.result("subscribe", channel=ahp.ROOT)["snapshot"]["state"]
    assert [agent["provider"] for agent in state["agents"]] == list(RUNTIMES)
    assert all(agent["models"] == [] for agent in state["agents"])
    real = Hub(_Acts())
    assert [agent["provider"] for agent in real._agents()] == list(runtime_names())


def test_the_root_can_be_subscribed_to_in_the_handshake():
    client = _Client(_hub())
    result = client.initialize(initialSubscriptions=[ahp.ROOT, "ahp-session:/nowhere"])
    assert [s["resource"] for s in result["snapshots"]] == [ahp.ROOT]
    assert client.connection.subscribed == {ahp.ROOT}


# --- sessions and chats --------------------------------------------------------


def test_a_session_is_ready_on_a_runtime_in_one_directory(tmp_path):
    client = _Client(_hub())
    client.initialize()
    client.result("createSession", channel=SESSION, provider="local",
                  workingDirectories=[tmp_path.as_uri()])
    state = client.result("subscribe", channel=SESSION)["snapshot"]["state"]
    assert (state["provider"], state["lifecycle"], state["status"]) == (
        "local", "ready", ahp_host.IDLE)
    assert state["workingDirectories"] == [tmp_path.as_uri()]
    assert state["chats"] == [] and state["activeClients"] == []


def test_a_session_is_refused_for_each_thing_it_may_not_be(tmp_path):
    """Mutations: default the provider when none is named -- red; accept a
    name `runtimes` lacks -- red; take the first of two directories -- red."""
    client = _Client(_hub())
    client.initialize()
    here = tmp_path.as_uri()

    def refused(**params):
        return client.error("createSession", **{"channel": SESSION, **params})

    none = refused(workingDirectories=[here])
    assert none["code"] == ahp_host.INVALID_PARAMS and "scripted, local" in none["message"]
    assert refused(provider="copilot")["code"] == ahp_host.PROVIDER_NOT_FOUND
    assert refused(provider="scripted",
                   workingDirectories=[here, here])["code"] == ahp_host.INVALID_PARAMS
    missing = refused(provider="scripted",
                      workingDirectories=[(tmp_path / "absent").as_uri()])
    assert missing["code"] == ahp_host.INVALID_PARAMS and "not a directory" in missing["message"]
    assert refused(provider="scripted",
                   workingDirectories=["https://example.test/x"])["code"] == (
        ahp_host.INVALID_PARAMS)
    client.result("createSession", channel=SESSION, provider="scripted")
    assert refused(provider="scripted")["code"] == ahp_host.SESSION_ALREADY_EXISTS


def test_a_chat_is_added_to_its_session_and_every_subscriber_is_told(tmp_path):
    """Mutation: send `session/chatAdded` to the creating client alone -- red."""
    hub = _hub()
    first = _Client(hub)
    first.initialize()
    first.result("createSession", channel=SESSION, provider="scripted")
    first.result("subscribe", channel=SESSION)
    second = _Client(hub, "client-b")
    second.initialize()
    second.result("subscribe", channel=SESSION)

    first.result("createChat", channel=SESSION, chat=CHAT)

    for client in (first, second):
        (added,) = client.actions()
        assert added["channel"] == SESSION
        assert added["action"]["type"] == "session/chatAdded"
        assert added["action"]["summary"]["resource"] == CHAT
    assert first.error("createChat", channel=SESSION, chat=CHAT)["code"] == (
        ahp_host.ALREADY_EXISTS)
    assert first.error("createChat", channel="ahp-session:/nowhere",
                       chat="ahp-chat:/c-2")["code"] == ahp_host.SESSION_NOT_FOUND
    assert first.error("createChat", channel=SESSION, chat="ahp-chat:/c-2",
                       initialMessage={"text": "go"})["code"] == ahp_host.INVALID_PARAMS


# --- a turn --------------------------------------------------------------------


def test_a_turn_records_an_instruction_and_starts_an_act_on_it(tmp_path):
    """Mutation: start the act without the session's directory -- red."""
    acts = _Acts(looks=[Look(status="recorded", running=True)])
    hub = _hub(acts)
    client = _attached(hub, tmp_path, provider="local")

    seq = client.dispatch(CHAT, _turn("Deploy qmcp to the pi."))

    (text, detail), = acts.recorded
    assert text == "Deploy qmcp to the pi."
    assert detail == {"surface": "agent-host-protocol", "session": SESSION, "chat": CHAT,
                      "turn": "t-1", "client": "client-a"}
    assert acts.started == [("i-1", "local", tmp_path)]
    echoed, said = client.actions()
    assert echoed["action"]["type"] == "chat/turnStarted"
    assert echoed["origin"] == {"clientId": "client-a", "clientSeq": seq}
    assert "rejectionReason" not in echoed
    assert said["action"]["type"] == "chat/responsePart"
    assert said["action"]["part"]["kind"] == "markdown"
    assert "instruction i-1" in said["action"]["part"]["content"]
    assert echoed["serverSeq"] < said["serverSeq"]


def test_an_act_that_declares_and_stops_ends_the_turn_with_what_it_said(tmp_path):
    """The default budget's path. Mutation: complete the turn without the
    command's words -- red."""
    declared = ("issued against 0 runs, so nothing was asked and nothing ran.",
                "Re-issue with --budget 1 to ask consent for one run.")
    acts = _Acts(looks=[Look(status="recorded", running=True),
                        Look(status="recorded", running=False, output=declared)])
    hub = _hub(acts)
    client = _attached(hub, tmp_path)
    client.dispatch(CHAT, _turn())
    client.actions()

    hub.tick()
    assert client.actions() == []          # still running: nothing new to say
    hub.now[0] = 2.5
    hub.tick()

    said, complete = client.actions()
    assert said["action"]["type"] == "chat/delta"
    assert "Nothing was asked and nothing ran." in said["action"]["content"]
    assert declared[0] in said["action"]["content"]
    assert complete["action"] == {"type": "chat/turnComplete", "turnId": "t-1",
                                  "duration": 2500}
    assert hub.turn is None
    hub.tick()
    assert client.actions() == []


def test_a_turn_says_when_consent_is_asked_and_when_the_runtime_runs(tmp_path):
    """Mutation: say `asking` on every tick -- red, it is said once."""
    acts = _Acts(looks=[
        Look(status="asking", running=True, consent_request_id="instruction-i-1"),
        Look(status="asking", running=True, consent_request_id="instruction-i-1"),
        Look(status="acting", running=True),
        Look(status="done", running=False, outcome_text="qmcp is the local backend.",
             exit_code=0)])
    hub = _hub(acts)
    client = _attached(hub, tmp_path)
    client.dispatch(CHAT, _turn())
    client.actions()

    for _ in range(4):
        hub.tick()

    said = [a["action"] for a in client.actions()]
    assert [a["type"] for a in said] == ["chat/delta", "chat/delta", "chat/delta",
                                         "chat/turnComplete"]
    assert "instruction-i-1" in said[0]["content"] and "human queue" in said[0]["content"]
    assert "approved" in said[1]["content"]
    assert "qmcp is the local backend." in said[2]["content"]


def test_a_failed_act_ends_the_turn_with_the_protocols_error(tmp_path):
    """Mutation: complete a failed turn -- red."""
    acts = _Acts(looks=[Look(status="failed", running=False, outcome_text="exit 3: no model",
                             exit_code=3)])
    hub = _hub(acts)
    client = _attached(hub, tmp_path)
    client.dispatch(CHAT, _turn())
    client.actions()

    hub.tick()

    (ended,) = client.actions()
    assert ended["action"]["type"] == "chat/error"
    assert ended["action"]["part"] == {"kind": "error", "error": {
        "errorType": "runtime", "message": "exit 3: no model"}}
    state = client.result("subscribe", channel=CHAT)["snapshot"]["state"]
    assert state["status"] == ahp_host.ERROR


def test_what_a_turn_streams_reaches_every_client_subscribed_to_the_chat(tmp_path):
    """Mutation: emit to the dispatching client alone -- red."""
    acts = _Acts(looks=[Look(status="done", running=False, outcome_text="Done.")])
    hub = _hub(acts)
    first = _attached(hub, tmp_path)
    second = _Client(hub, "client-b")
    second.initialize()
    second.result("subscribe", channel=CHAT)

    first.dispatch(CHAT, _turn())
    hub.tick()

    seen = [[a["action"]["type"] for a in client.actions()] for client in (first, second)]
    assert seen[0] == seen[1] == ["chat/turnStarted", "chat/responsePart", "chat/delta",
                                  "chat/turnComplete"]


# --- one act at a time ---------------------------------------------------------


def test_a_turn_while_another_runs_is_rejected_and_records_nothing(tmp_path):
    """Mutation: drop the `self.turn is not None` check -- red, two acts start."""
    acts = _Acts(looks=[Look(status="acting", running=True)])
    hub = _hub(acts)
    client = _attached(hub, tmp_path)
    other = _Client(hub, "client-b")
    other.initialize()
    other.result("subscribe", channel=CHAT)
    client.dispatch(CHAT, _turn())
    client.actions(), other.actions()

    seq = other.dispatch(CHAT, _turn("And another.", "t-2"))

    (rejected,) = other.actions()
    assert "One at a time" in rejected["rejectionReason"]
    assert rejected["origin"] == {"clientId": "client-b", "clientSeq": seq}
    assert client.actions() == []                 # the rejection is the dispatcher's alone
    assert len(acts.recorded) == 1 and len(acts.started) == 1


def test_a_turn_while_a_conversation_runs_is_rejected_by_name(tmp_path):
    """Mutation: do not ask `acts.busy()` -- red, an instruction is recorded."""
    acts = _Acts(running="conversation")
    client = _attached(_hub(acts), tmp_path)

    client.dispatch(CHAT, _turn())

    (rejected,) = client.actions()
    assert "for 'conversation'" in rejected["rejectionReason"]
    assert acts.recorded == [] and acts.started == []


def test_an_act_that_lost_the_race_leaves_its_instruction_recorded_and_says_so(tmp_path):
    acts = _Acts(start_busy="another")
    hub = _hub(acts)
    client = _attached(hub, tmp_path)

    client.dispatch(CHAT, _turn())

    (rejected,) = client.actions()
    assert "Instruction i-1 is recorded and nothing ran" in rejected["rejectionReason"]
    assert hub.turn is None


# --- a turn's consent ------------------------------------------------------------


ASKING = Look(status="asking", running=True, consent_request_id=CONSENT.id)


def _asking(tmp_path, acts=None, **attached):
    """A hub whose one turn has shown its consent, and the client that started it."""
    acts = acts or _Acts(looks=[ASKING], consent=CONSENT)
    hub = _hub(acts)
    client = _attached(hub, tmp_path, **attached)
    client.dispatch(CHAT, _turn())
    hub.tick()
    return hub, acts, client


def _accept(value="approve", **over):
    answer = {"state": "submitted", "value": {"kind": "selected", "value": value}}
    return {"type": "chat/inputCompleted", "requestId": CONSENT.id, "response": "accept",
            "answers": {"answer": answer}, **over}


def test_a_consent_is_shown_as_the_queue_holds_it_with_nothing_selected(tmp_path):
    """Mutations: mark an option `recommended` -- red; show it on every tick
    -- red, it is shown once."""
    hub, _, client = _asking(tmp_path)
    hub.tick()

    shown = [a["action"] for a in client.actions() if a["action"]["type"] == "chat/inputRequested"]
    assert shown == [{"type": "chat/inputRequested", "request": {
        "id": CONSENT.id, "message": CONSENT.prompt, "questions": [{
            "id": "answer", "kind": "single-select", "message": CONSENT.prompt,
            "required": True,
            "options": [{"id": "approve", "label": "approve"},
                        {"id": "hold", "label": "hold"}]}]}}]


@pytest.mark.parametrize("value", ["approve", "hold"])
def test_an_answer_given_in_a_client_on_this_machine_is_relayed_to_the_queue(tmp_path, value):
    """Mutations: echo the answer without writing it to the queue -- red;
    write `approve` whatever was selected -- red."""
    hub, acts, client = _asking(tmp_path)
    other = _Client(hub, "client-b")
    other.initialize()
    other.result("subscribe", channel=CHAT)
    client.actions()

    seq = client.dispatch(CHAT, _accept(value))

    assert acts.answered == [(CONSENT.id, value)]
    for reader in (client, other):
        (told,) = reader.actions()
        assert told["action"] == _accept(value) and "rejectionReason" not in told
        assert told["origin"] == {"clientId": "client-a", "clientSeq": seq}


def test_a_client_attached_from_another_address_is_shown_the_consent_and_cannot_answer(
        tmp_path):
    """Mutation: drop the loopback check in `_answered` -- red."""
    hub, acts, client = _asking(tmp_path, loopback=False)
    assert "chat/inputRequested" in [a["action"]["type"] for a in client.actions()]

    for action in (_accept(), {"type": "chat/inputAnswerChanged", "requestId": CONSENT.id,
                               "questionId": "answer", "answer": _accept()["answers"]["answer"]}):
        client.dispatch(CHAT, action)
        (rejected,) = client.actions()
        assert "another address" in rejected["rejectionReason"]
    assert acts.answered == [] and hub.turn.draft is None


@pytest.mark.parametrize("action,why", [
    (_accept(response="decline"), "no decline"),
    (_accept(response="cancel"), "no cancel"),
    (_accept(answers=None), "no option"),
    (_accept("yes"), "no option"),
    (_accept(answers={"answer": {"state": "skipped", "value": {
        "kind": "selected", "value": "approve"}}}), "no option"),
    (_accept(answers={"answer": {"state": "submitted", "value": {
        "kind": "text", "value": "approve"}}}), "no option"),
    (_accept(answers={"answer": {"state": "submitted", "value": {
        "kind": "selected", "value": "approve", "freeformValues": ["and more"]}}}),
     "no option"),
    (_accept(requestId="instruction-i-9"), "no consent is asked"),
    ({"type": "chat/inputAnswerChanged", "requestId": CONSENT.id, "questionId": "other",
      "answer": {"state": "draft", "value": {"kind": "selected", "value": "approve"}}},
     "one question"),
])
def test_anything_but_one_option_of_the_queues_is_rejected_and_nothing_is_answered(
        tmp_path, action, why):
    """Mutations: take the answer a `decline` carries -- red; take a value
    that is not one of the request's options -- red; take a skipped answer
    that still carries a value -- red."""
    hub, acts, client = _asking(tmp_path)
    client.actions()

    client.dispatch(CHAT, action)

    (rejected,) = client.actions()
    assert why in rejected["rejectionReason"]
    assert acts.answered == [] and not hub.turn.settled


def test_an_answer_before_any_consent_is_asked_is_rejected(tmp_path):
    acts = _Acts(looks=[Look(status="acting", running=True)])
    hub = _hub(acts)
    client = _attached(hub, tmp_path)
    client.dispatch(CHAT, _turn())
    client.actions()

    client.dispatch(CHAT, _accept())

    assert "no consent is asked" in client.actions()[0]["rejectionReason"]
    assert acts.answered == []


def test_a_drafted_answer_is_the_one_an_accept_without_answers_submits(tmp_path):
    """Mutation: ignore the draft -- red, the accept has no option selected.
    Clearing the draft takes it away again."""
    hub, acts, client = _asking(tmp_path)
    client.actions()
    draft = {"type": "chat/inputAnswerChanged", "requestId": CONSENT.id, "questionId": "answer",
             "answer": {"state": "draft", "value": {"kind": "selected", "value": "hold"}}}

    client.dispatch(CHAT, draft)
    assert client.actions()[0]["action"] == draft
    client.dispatch(CHAT, {k: v for k, v in draft.items() if k != "answer"})
    client.actions()
    client.dispatch(CHAT, _accept(answers=None))
    assert "no option" in client.actions()[0]["rejectionReason"]

    client.dispatch(CHAT, draft)
    client.dispatch(CHAT, _accept(answers=None))

    assert acts.answered == [(CONSENT.id, "hold")]


def test_the_queues_refusal_reaches_the_client_and_the_consent_stays_open(tmp_path):
    """Mutation: swallow `Refused` and echo the answer -- red."""
    acts = _Acts(looks=[ASKING], consent=CONSENT, refuses="Request has expired")
    hub, _, client = _asking(tmp_path, acts)
    client.actions()

    client.dispatch(CHAT, _accept())

    (rejected,) = client.actions()
    assert rejected["rejectionReason"] == "qmcp's queue refused the answer: Request has expired"
    assert not hub.turn.settled


def test_a_consent_is_answered_once_through_the_host(tmp_path):
    """Mutation: drop the `settled` check -- red, the queue is written twice."""
    hub, acts, client = _asking(tmp_path)
    client.dispatch(CHAT, _accept("hold"))
    client.actions()

    client.dispatch(CHAT, _accept("approve"))

    assert "answered already" in client.actions()[0]["rejectionReason"]
    assert acts.answered == [(CONSENT.id, "hold")]


@pytest.mark.parametrize("answer,completed", [
    ("approve", {"response": "accept", "answers": {"answer": {"state": "submitted", "value": {
        "kind": "selected", "value": "approve"}}}}),
    (None, {"response": "cancel"}),
])
def test_an_answer_given_on_the_queue_completes_the_request_in_the_chat(
        tmp_path, answer, completed):
    """Answered where every request is, or expired with no answer.
    Mutation: never tell the chat -- red, the question stays open in it."""
    acts = _Acts(looks=[ASKING, ASKING, Look(status="acting", running=True)], consent=CONSENT)
    hub, _, client = _asking(tmp_path, acts)
    client.actions()
    acts.consent = Asked(id=CONSENT.id, prompt=CONSENT.prompt, options=CONSENT.options,
                         answer=answer)

    hub.tick()
    hub.tick()

    told = [a for a in client.actions() if a["action"]["type"] == "chat/inputCompleted"]
    assert [a["action"] for a in told] == [
        {"type": "chat/inputCompleted", "requestId": CONSENT.id, **completed}]
    assert told[0]["origin"] is None


def test_an_answer_relayed_through_the_host_is_not_told_a_second_time(tmp_path):
    """Mutation: complete the request on every tick after `asking` -- red."""
    acts = _Acts(looks=[ASKING, Look(status="acting", running=True)], consent=CONSENT)
    hub, _, client = _asking(tmp_path, acts)
    client.dispatch(CHAT, _accept())
    client.actions()

    hub.tick()
    hub.tick()

    assert "chat/inputCompleted" not in [a["action"]["type"] for a in client.actions()]


@pytest.mark.parametrize("host,on_this_machine", [
    ("127.0.0.1", True), ("::1", True), ("localhost", True), ("127.8.8.8", True),
    ("192.168.1.5", False), ("0.0.0.0", False), ("testclient", False), ("", False),
    (None, False),
])
def test_an_address_on_this_machine_is_a_loopback_address(host, on_this_machine):
    """Mutation: count the unspecified address as this machine -- red."""
    assert ahp_host.loopback(host) is on_this_machine


# --- what is refused -------------------------------------------------------------


def test_a_turn_cannot_be_cancelled_and_the_client_is_told_why(tmp_path):
    """Mutation: echo the cancellation as accepted -- red."""
    acts = _Acts(looks=[Look(status="acting", running=True)])
    hub = _hub(acts)
    client = _attached(hub, tmp_path)
    client.dispatch(CHAT, _turn())
    client.actions()

    client.dispatch(CHAT, {"type": "chat/turnCancelled", "turnId": "t-1", "duration": 5})

    (rejected,) = client.actions()
    assert rejected["rejectionReason"] == ahp_host.CANNOT_STOP
    assert hub.turn is not None
    assert client.error("disposeSession", channel=SESSION)["code"] == ahp_host.TURN_IN_PROGRESS


@pytest.mark.parametrize("action", [
    {"type": "chat/toolCallConfirmed", "turnId": "t-1", "toolCallId": "c-1", "approved": True},
    {"type": "chat/responsePart", "turnId": "t-1", "part": {"kind": "markdown", "id": "p"}},
    {"type": "session/titleChanged", "title": "mine"},
    {},
])
def test_any_other_action_from_a_client_is_rejected(tmp_path, action):
    """Mutation: emit an action the host does not know to the chat -- red.
    A client confirming a tool call is among them: the host holds none."""
    hub = _hub()
    client = _attached(hub, tmp_path)

    client.dispatch(CHAT, action)

    (rejected,) = client.actions()
    assert "accepts no" in rejected["rejectionReason"]
    assert hub.acts.recorded == []


def test_a_turn_without_text_or_on_no_chat_is_rejected(tmp_path):
    hub = _hub()
    client = _attached(hub, tmp_path)
    client.dispatch(CHAT, _turn("   "))
    client.dispatch("ahp-chat:/nowhere", _turn())
    blank, nowhere = client.actions()
    assert "text" in blank["rejectionReason"] and "no chat" in nowhere["rejectionReason"]
    assert hub.acts.recorded == []


@pytest.mark.parametrize("method", [
    "createTerminal", "resourceRead", "createResourceWatch", "invokeChangesetOperation",
    "fetchTurns", "listSessions", "authenticate", "createSearch", "runAutomation",
])
def test_what_qmcp_does_not_serve_is_refused_by_name(method):
    """Mutation: answer an unknown method with an empty result -- red."""
    client = _Client(_hub())
    client.initialize()
    refused = client.error(method, channel=ahp.ROOT)
    assert refused["code"] == ahp_host.METHOD_NOT_FOUND and method in refused["message"]


def test_a_session_with_no_turn_running_is_disposed(tmp_path):
    hub = _hub()
    client = _attached(hub, tmp_path)
    assert client.result("disposeSession", channel=SESSION) is None
    assert hub.sessions == {} and client.connection.subscribed == set()
    assert client.error("subscribe", channel=CHAT)["code"] == ahp_host.NOT_FOUND


# --- the server's own record ---------------------------------------------------------


class _Process:
    def __init__(self, argv, stdout, lines):
        self.argv, self.code = argv, None
        stdout.write("\n".join(lines) + "\n")

    def poll(self):
        return self.code


def test_the_servers_record_is_an_instruction_row_and_the_act_command(tmp_path):
    """Mutations: record with another source -- red; leave `--budget` off
    the command -- red, the act would declare and stop; issue it against two
    runs -- red."""
    from qmcp.db.models import Instruction, InstructionSource
    from qmcp.instructions.act import rows_at
    from qmcp.integrations.voice.service import VoiceRuns

    started: list[_Process] = []

    def popen(argv, stdout, stderr, env):
        started.append(_Process(argv, stdout, ["asking consent"]))
        return started[-1]

    rows = rows_at(tmp_path / "inbox.db")
    runs = VoiceRuns(log_dir=tmp_path, popen=popen)
    acts = ServerActs(runs, "http://127.0.0.1:3141", rows=rows, names=lambda: ("qmcp", "vox"))

    instruction_id = acts.record("Deploy qmcp to the pi.", {"surface": "agent-host-protocol"})
    with rows() as session:
        row = session.get(Instruction, instruction_id)
        assert (row.text, row.project, row.source) == (
            "Deploy qmcp to the pi.", "qmcp", InstructionSource.PROTOCOL)
        assert row.status.value == "recorded"
        assert row.detail["surface"] == "agent-host-protocol"

    assert acts.busy() is None
    acts.start(instruction_id, "scripted", tmp_path)
    argv = started[0].argv
    assert argv[1:6] == ["-m", "qmcp", "instructions", "act", instruction_id]
    assert argv[6:] == ["--runtime", "scripted", "--budget", "1",
                        "--base-url", "http://127.0.0.1:3141", "--cwd", str(tmp_path)]
    assert acts.busy() == instruction_id
    with pytest.raises(Busy, match=instruction_id):
        acts.start("another", "scripted", None)

    running = acts.look(instruction_id)
    assert (running.status, running.running) == ("recorded", True)
    started[0].code = 0
    ended = acts.look(instruction_id)
    assert (ended.running, ended.output) == (False, ("asking consent",))
    assert acts.look("another").output == ()        # another act's words are not this one's


def test_the_servers_relay_reads_and_writes_the_queues_own_routes(client):
    """`ServerActs` against the test server's human queue, so the refusals
    are the queue's. Mutations: write the response row with no
    `responded_by` -- red; report the queue's 409 as answered -- red."""
    from qmcp.client import MCPClient

    queue = MCPClient("http://testserver")
    queue._client.close()
    queue._client = client
    acts = ServerActs(runs=None, base_url="http://testserver", queue=queue)
    queue.create_human_request(request_id="instruction-i-1", request_type="approval",
                               prompt=CONSENT.prompt, options=list(CONSENT.options))

    assert acts.asked("instruction-i-1") == CONSENT
    assert acts.asked("instruction-i-404") is None
    with pytest.raises(Refused, match="must be one of"):
        acts.answer("instruction-i-1", "yes")
    with pytest.raises(Refused, match="not found"):
        acts.answer("instruction-i-404", "approve")

    acts.answer("instruction-i-1", "approve")

    _, response = queue.get_human_request("instruction-i-1")
    assert (response.response, response.responded_by) == ("approve", "agent-host-protocol")
    assert acts.asked("instruction-i-1").answer == "approve"
    with pytest.raises(Refused, match="already been responded"):
        acts.answer("instruction-i-1", "hold")


# --- the route -------------------------------------------------------------------


def test_the_route_carries_a_turn_over_a_websocket(client, tmp_path):
    """The WebSocket on the test server, with the record stood in for.
    Mutation: never call `hub.tick` in the route -- the turn never ends and
    this test hangs, which a run with a time limit reads as red."""
    hub = client.app.state.ahp_host
    hub.acts = _Acts(looks=[Look(status="recorded", running=False,
                                 output=("issued against 0 runs",))])
    hub.runtimes = lambda: RUNTIMES

    with client.websocket_connect("/v1/ahp") as socket:
        def ask(number, method, **params):
            socket.send_json({"jsonrpc": "2.0", "id": number, "method": method,
                              "params": params})
            while True:
                answer = socket.receive_json()
                if answer.get("id") == number:
                    return answer

        assert ask(1, "initialize", channel=ahp.ROOT, protocolVersions=["1.0.0"],
                   clientId="page")["result"]["protocolVersion"] == "1.0.0"
        ask(2, "createSession", channel=SESSION, provider="scripted",
            workingDirectories=[tmp_path.as_uri()])
        ask(3, "createChat", channel=SESSION, chat=CHAT)
        ask(4, "subscribe", channel=CHAT)
        socket.send_json({"jsonrpc": "2.0", "method": "dispatchAction", "params": {
            "channel": CHAT, "clientSeq": 1, "action": _turn()}})
        kinds = []
        while "chat/turnComplete" not in kinds:
            kinds.append(socket.receive_json()["params"]["action"]["type"])

    assert kinds == ["chat/turnStarted", "chat/responsePart", "chat/delta", "chat/turnComplete"]
    assert hub.connections == []


def test_the_route_knows_a_client_by_the_address_it_is_attached_from(client, tmp_path):
    """The test server's client is attached from a name that is no address
    on this machine. Mutation: attach every socket as loopback -- red."""
    hub = client.app.state.ahp_host
    acts = hub.acts = _Acts(looks=[ASKING], consent=CONSENT)
    hub.runtimes = lambda: RUNTIMES

    with client.websocket_connect("/v1/ahp") as socket:
        def send(number, method, **params):
            socket.send_json({"jsonrpc": "2.0", "id": number, "method": method,
                              "params": params})

        send(1, "initialize", channel=ahp.ROOT, protocolVersions=["1.0.0"], clientId="page")
        send(2, "createSession", channel=SESSION, provider="scripted",
             workingDirectories=[tmp_path.as_uri()])
        send(3, "createChat", channel=SESSION, chat=CHAT)
        send(4, "subscribe", channel=CHAT)
        socket.send_json({"jsonrpc": "2.0", "method": "dispatchAction", "params": {
            "channel": CHAT, "clientSeq": 1, "action": _turn()}})
        kinds = []
        while "chat/inputRequested" not in kinds:
            kinds.append((socket.receive_json().get("params") or {}).get(
                "action", {}).get("type"))
        socket.send_json({"jsonrpc": "2.0", "method": "dispatchAction", "params": {
            "channel": CHAT, "clientSeq": 2, "action": _accept()}})
        while True:
            params = socket.receive_json()["params"]
            if params["action"]["type"] == "chat/inputCompleted":
                break

    assert "another address" in params["rejectionReason"]
    assert acts.answered == []


def test_a_refused_handshake_closes_the_socket(client):
    """Mutation: leave the socket open after the refusal -- this test hangs
    on the read that should have been refused, red under a time limit."""
    from starlette.websockets import WebSocketDisconnect

    with client.websocket_connect("/v1/ahp") as socket:
        socket.send_json({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "channel": ahp.ROOT, "protocolVersions": ["7.0.0"], "clientId": "page"}})
        assert socket.receive_json()["error"]["code"] == ahp_host.UNSUPPORTED_PROTOCOL_VERSION
        with pytest.raises(WebSocketDisconnect):
            socket.receive_json()


def test_the_route_is_not_served_off_loopback(monkeypatch):
    """A turn starts a process on this machine. Mutation: register the host
    outside the `is_loopback` branch in `qmcp.server` -- red."""
    import qmcp.server

    class OffLoopback:
        host = "0.0.0.0"
        port = 3141
        debug = False
        database_url = "sqlite+aiosqlite:///:memory:"
        log_level = "WARNING"
        voice_engine = "joe"
        voice_engine_url = None

    monkeypatch.setattr(qmcp.server, "get_settings", lambda: OffLoopback())
    app = qmcp.server.create_app()

    assert not [r for r in app.routes if "ahp" in getattr(r, "path", "")]
    assert getattr(app.state, "ahp_host", None) is None


# --- qmcp's client against qmcp's host -------------------------------------------


class _Loop:
    """The adapter's `MessageStream`, joined to a hub's connection in process."""

    def __init__(self, hub):
        self.hub = hub
        self.connection = hub.connect(True)

    def send(self, message):
        self.connection.handle(message)

    def receive(self, timeout):
        if not self.connection.outbox:
            self.hub.tick()
        if not self.connection.outbox:
            raise TimeoutError
        return self.connection.outbox.popleft()

    def close(self):
        self.hub.disconnect(self.connection)


def test_qmcps_client_role_runs_a_turn_on_qmcps_host_role(tmp_path):
    """Mutation: end the turn without the outcome's delta -- red, the client
    returns without the runtime's words."""
    acts = _Acts(looks=[Look(status="acting", running=True),
                        Look(status="done", running=False, outcome_text="It serves MCP.")])
    hub = _hub(acts)
    runtime = ahp.Runtime(url="ws://qmcp.test/v1/ahp", connect=lambda url: _Loop(hub),
                          provider="scripted")

    outcome = runtime.run(Brief(instruction="What is qmcp?", cwd=tmp_path, project="qmcp"))

    assert outcome.succeeded, outcome.text
    assert "It serves MCP." in outcome.text and "instruction i-1" in outcome.text
    assert acts.started == [("i-1", "scripted", tmp_path.resolve())]
    assert outcome.detail["protocol_version"] == ahp.PROTOCOL_VERSIONS[0]
    assert hub.connections == []


def test_qmcps_client_role_answers_no_consent_qmcps_host_role_asks(tmp_path):
    """The client role confirms nothing, so a host that asks is walked away
    from, and the consent stays on the queue for a person. Mutation: have the
    client accept the input request with its first option -- red."""
    acts = _Acts(looks=[ASKING], consent=CONSENT)
    hub = _hub(acts)
    runtime = ahp.Runtime(url="ws://qmcp.test/v1/ahp", connect=lambda url: _Loop(hub),
                          provider="scripted")

    outcome = runtime.run(Brief(instruction="What is qmcp?", cwd=tmp_path, project="qmcp"))

    assert not outcome.succeeded and "asked for input" in outcome.text
    assert acts.answered == [] and hub.turn is not None


def test_qmcps_client_is_refused_a_session_when_it_names_no_runtime(tmp_path):
    hub = _hub()
    runtime = ahp.Runtime(url="ws://qmcp.test/v1/ahp", connect=lambda url: _Loop(hub))
    outcome = runtime.run(Brief(instruction="What is qmcp?", cwd=tmp_path, project="qmcp"))
    assert not outcome.succeeded and "names no provider" in outcome.text
