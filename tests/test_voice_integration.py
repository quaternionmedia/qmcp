import pytest

from qmcp.client.mcp_client import HumanRequest, HumanRequestExpiredError, HumanResponse
from qmcp.integrations.voice.adapter import (
    UnclearResponse,
    VoiceApprovalLoop,
    choose_option,
    match_option,
    parse_yes_no,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("yes", True),
        ("Yes!", True),
        ("yeah, go ahead", True),
        ("approved", True),
        ("do it.", True),
        ("no", False),
        ("No.", False),
        ("nope, cancel that", False),
        ("don't", False),
        ("maybe", None),
        ("I'm still thinking", None),
        ("", None),
    ],
)
def test_parse_yes_no(text, expected):
    assert parse_yes_no(text) is expected


class ScriptedSTT:
    """Returns each transcript in order, repeating the last one once exhausted."""

    def __init__(self, transcripts: list[str]):
        self._transcripts = list(transcripts)
        self.calls = 0

    def listen(self, duration: float = 5.0) -> tuple[str, str]:
        text = self._transcripts[min(self.calls, len(self._transcripts) - 1)]
        self.calls += 1
        return text, f"capture_{self.calls}.wav"


class RecordingTTS:
    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text: str, out_path: str | None = None) -> str:
        self.spoken.append(text)
        return out_path or "spoken.wav"


class FakeClient:
    """Stands in for MCPClient's HITL surface: no HTTP, no server."""

    def __init__(self):
        self.requests: dict[str, HumanRequest] = {}
        self.responses: dict[str, HumanResponse] = {}
        self.submitted: list[tuple[str, str]] = []

    def add_pending(self, request_id: str, prompt: str, options: list[str] | None = None):
        self.requests[request_id] = HumanRequest(
            id=request_id,
            request_type="approval",
            prompt=prompt,
            status="pending",
            created_at="now",
            options=options,
        )

    def get_human_request(self, request_id: str):
        return self.requests[request_id], self.responses.get(request_id)

    def list_human_requests(
        self, status_filter=None, request_type=None, limit=50, offset=0, oldest_first=False
    ):
        # Insertion order is creation order; the server lists newest first
        # unless asked otherwise, and so does this.
        pending = [r for rid, r in self.requests.items() if rid not in self.responses]
        return (pending if oldest_first else pending[::-1])[:limit]

    def submit_human_response(self, request_id, response, responded_by=None, metadata=None):
        self.submitted.append((request_id, response))
        result = HumanResponse(
            id=f"resp-{request_id}",
            request_id=request_id,
            response=response,
            responded_by=responded_by,
            created_at="now",
        )
        self.responses[request_id] = result
        return result


def test_run_once_answers_clear_approval():
    client = FakeClient()
    client.add_pending("deploy-001", "Deploy to production?", options=["approve", "reject"])
    stt = ScriptedSTT(["yes, go ahead"])
    tts = RecordingTTS()

    loop = VoiceApprovalLoop(stt=stt, tts=tts, client=client)
    result = loop.run_once("deploy-001")

    assert result.response == "approve"
    assert client.submitted == [("deploy-001", "approve")]
    assert tts.spoken[0] == "Deploy to production? Say approve or reject."
    assert "Recorded: approve" in tts.spoken


def test_run_once_answers_clear_rejection():
    client = FakeClient()
    client.add_pending("deploy-002", "Deploy to production?", options=["approve", "reject"])
    stt = ScriptedSTT(["no, cancel it"])
    tts = RecordingTTS()

    loop = VoiceApprovalLoop(stt=stt, tts=tts, client=client)
    result = loop.run_once("deploy-002")

    assert result.response == "reject"
    assert client.submitted == [("deploy-002", "reject")]


def test_run_once_retries_on_unclear_answer_then_succeeds():
    client = FakeClient()
    client.add_pending("deploy-003", "Deploy?", options=["approve", "reject"])
    stt = ScriptedSTT(["uh", "hmm", "yes"])
    tts = RecordingTTS()

    loop = VoiceApprovalLoop(stt=stt, tts=tts, client=client, max_retries=2)
    result = loop.run_once("deploy-003")

    assert result.response == "approve"
    assert stt.calls == 3
    assert tts.spoken[1] == "I heard: uh. Say approve or reject."
    assert tts.spoken[2] == "I heard: hmm. Say approve or reject."


def test_run_once_raises_when_never_clear():
    client = FakeClient()
    client.add_pending("deploy-004", "Deploy?", options=["approve", "reject"])
    stt = ScriptedSTT(["uh", "hmm", "still not sure"])
    tts = RecordingTTS()

    loop = VoiceApprovalLoop(stt=stt, tts=tts, client=client, max_retries=2)
    with pytest.raises(UnclearResponse):
        loop.run_once("deploy-004")

    assert client.submitted == []


def test_run_once_is_idempotent_for_an_already_answered_request():
    client = FakeClient()
    client.add_pending("deploy-005", "Deploy?", options=["approve", "reject"])
    client.responses["deploy-005"] = HumanResponse(
        id="resp-deploy-005", request_id="deploy-005", response="approve", responded_by="alice", created_at="now"
    )
    stt = ScriptedSTT(["should never be heard"])
    tts = RecordingTTS()

    loop = VoiceApprovalLoop(stt=stt, tts=tts, client=client)
    result = loop.run_once("deploy-005")

    assert result.responded_by == "alice"
    assert stt.calls == 0


def test_run_forever_answers_each_pending_request_in_turn():
    """The continuation this loop exists for: one call answers multiple requests in sequence."""
    client = FakeClient()
    client.add_pending("req-a", "Deploy A?", options=["approve", "reject"])
    client.add_pending("req-b", "Deploy B?", options=["approve", "reject"])
    stt = ScriptedSTT(["yes", "no"])
    tts = RecordingTTS()

    loop = VoiceApprovalLoop(stt=stt, tts=tts, client=client)
    answered = loop.run_forever(max_iterations=5)

    assert answered == 2
    assert set(client.submitted) == {("req-a", "approve"), ("req-b", "reject")}


def test_run_forever_asks_the_oldest_request_first():
    """The server lists newest first unless asked; a queue answered by voice
    is answered in the order it was asked."""
    client = FakeClient()
    client.add_pending("older", "Deploy A?", options=["approve", "reject"])
    client.add_pending("newer", "Deploy B?", options=["approve", "reject"])

    loop = VoiceApprovalLoop(stt=ScriptedSTT(["yes", "no"]), tts=RecordingTTS(), client=client)
    loop.run_forever(max_iterations=5)

    assert client.submitted == [("older", "approve"), ("newer", "reject")]


def test_run_forever_asks_an_unanswered_request_once_and_moves_on():
    """Nobody at the speaker: the loop re-asked the same request at once, for
    as long as it ran. Now one request costs one prompt and its re-asks, it
    stays pending with nothing guessed, and the next request is still asked."""
    client = FakeClient()
    client.add_pending("nobody-home", "Launch?", options=["approve", "hold"])
    client.add_pending("later", "Deploy?", options=["approve", "reject"])
    stt = ScriptedSTT(["", "", "", "yes"])
    tts = RecordingTTS()

    loop = VoiceApprovalLoop(stt=stt, tts=tts, client=client, max_retries=2)
    answered = loop.run_forever(max_iterations=6)

    assert answered == 1
    assert client.submitted == [("later", "approve")]
    assert loop.unanswered == ["nobody-home"]
    # One prompt and two re-asks for the first, one prompt for the second,
    # and nothing more across the remaining iterations.
    assert stt.calls == 4
    assert sum(s.startswith("Launch?") for s in tts.spoken) == 1


def test_run_forever_stops_when_nothing_pending():
    client = FakeClient()
    stt = ScriptedSTT(["yes"])
    tts = RecordingTTS()

    loop = VoiceApprovalLoop(stt=stt, tts=tts, client=client)
    answered = loop.run_forever(max_iterations=5)

    assert answered == 0
    assert stt.calls == 0


# ─── choosing which option a yes or a no means ────────────────────────────────
#
# Every test above passes options=["approve", "reject"], so the conventional
# ordering was the only one ever exercised. Nothing in a request states which
# position is the affirmative one, and a request is free to carry them the
# other way round.

@pytest.mark.parametrize(
    "decision,options,expected",
    [
        (True, ["approve", "reject"], "approve"),
        (False, ["approve", "reject"], "reject"),
        # Reversed: position would give exactly the wrong answer.
        (True, ["reject", "approve"], "approve"),
        (False, ["reject", "approve"], "reject"),
        # Other vocabularies the parser already knows.
        (True, ["deny", "confirm"], "confirm"),
        (False, ["deny", "confirm"], "deny"),
        (True, ["yes", "no"], "yes"),
        (False, ["yes", "no"], "no"),
        # Punctuation and casing, as a transcript or a UI label may carry them.
        (True, ["Reject.", "Approve!"], "Approve!"),
        # Nothing recognisable: position is the only information there is,
        # and the affirmative conventionally comes first.
        (True, ["ship it", "wait"], "ship it"),
        (False, ["ship it", "wait"], "wait"),
        # A single option is the answer whichever way the decision went.
        (True, ["acknowledge"], "acknowledge"),
        (False, ["acknowledge"], "acknowledge"),
    ],
)
def test_choose_option_reads_the_options_rather_than_their_order(decision, options, expected):
    assert choose_option(decision, options) == expected


def test_run_once_submits_the_approving_option_however_it_is_ordered():
    """The defect this guards: a spoken "yes" recorded as a rejection."""
    client = FakeClient()
    client.add_pending("deploy-006", "Deploy?", options=["reject", "approve"])
    loop = VoiceApprovalLoop(
        stt=ScriptedSTT(["yes"]), tts=RecordingTTS(), client=client, max_retries=0
    )

    response = loop.run_once("deploy-006")

    assert response.response == "approve"


def test_run_once_submits_the_rejecting_option_however_it_is_ordered():
    client = FakeClient()
    client.add_pending("deploy-007", "Deploy?", options=["reject", "approve"])
    loop = VoiceApprovalLoop(
        stt=ScriptedSTT(["no"]), tts=RecordingTTS(), client=client, max_retries=0
    )

    response = loop.run_once("deploy-007")

    assert response.response == "reject"


# --- a closed-choice dialog, as VoiceXML frames one ---------------------------
#
# The grammar is the request's own options: the prompt says them, an answer
# naming one is taken, and hearing nothing (noinput) is told apart from
# hearing something unusable (nomatch).


def test_an_option_outside_the_yes_no_vocabulary_is_accepted_by_name():
    """The defect: a request carrying ["approve", "hold"] could not be
    answered "hold" -- it parsed as neither yes nor no, was re-asked, and
    ended unanswered."""
    client = FakeClient()
    client.add_pending("dr-1", "Launch the audit?", options=["approve", "hold"])
    loop = VoiceApprovalLoop(stt=ScriptedSTT(["Hold."]), tts=RecordingTTS(), client=client)

    assert loop.run_once("dr-1").response == "hold"


def test_a_multi_word_option_is_accepted_by_name():
    client = FakeClient()
    client.add_pending("ship-1", "Ready?", options=["ship it", "wait"])
    loop = VoiceApprovalLoop(stt=ScriptedSTT(["wait, please"]), tts=RecordingTTS(), client=client)

    assert loop.run_once("ship-1").response == "wait"


def test_a_negated_option_is_not_taken_as_that_option():
    client = FakeClient()
    client.add_pending("dr-2", "Launch the audit?", options=["approve", "hold"])
    loop = VoiceApprovalLoop(stt=ScriptedSTT(["don't approve it"]), tts=RecordingTTS(), client=client)

    assert loop.run_once("dr-2").response == "hold"


def test_noinput_and_nomatch_are_reprompted_differently():
    client = FakeClient()
    client.add_pending("dr-3", "Launch the audit?", options=["approve", "hold"])
    tts = RecordingTTS()
    loop = VoiceApprovalLoop(stt=ScriptedSTT(["", "banana", "approve"]), tts=tts, client=client)

    assert loop.run_once("dr-3").response == "approve"
    assert tts.spoken[:3] == [
        "Launch the audit? Say approve or hold.",
        "I didn't hear anything. Say approve or hold.",
        "I heard: banana. Say approve or hold.",
    ]


@pytest.mark.parametrize(
    "text,options,expected",
    [
        # A transcript carries no hyphen, so the option's is a word break.
        ("cuelist python", ["rad-godot", "Cuelist-python"], "Cuelist-python"),
        ("rad-godot", ["rad-godot", "Cuelist-python"], "rad-godot"),
        # The longer option says every word the shorter does: it is the one said.
        ("rad godot", ["rad", "rad-godot"], "rad-godot"),
        ("rad", ["rad", "rad-godot"], "rad"),
        # Two options neither of which covers the other is still a nomatch.
        ("rad and vox", ["rad", "vox"], None),
        ("approve all", ["approve", "approve all"], "approve all"),
    ],
)
def test_match_option_reads_a_hyphen_as_a_word_break_and_prefers_the_covering_option(
        text, options, expected):
    """Mutation: keep the hyphen out of the word break (`re.sub` on the
    option alone) -- red on the first row; drop the covering rule -- red on
    `rad godot` and `approve all`, both nomatch."""
    assert match_option(text, options) == expected


def test_three_or_more_options_are_spoken_as_a_list():
    client = FakeClient()
    client.add_pending("pick-1", "Which?", options=["red", "green", "blue"])
    tts = RecordingTTS()
    VoiceApprovalLoop(stt=ScriptedSTT(["green"]), tts=tts, client=client).run_once("pick-1")

    assert tts.spoken[0] == "Which? Say red, green, or blue."


# --- an open question: the transcript is the answer ---------------------------
#
# A request carrying no options used to be given approve and reject, so an
# `input` request answered "yes" recorded `approve` and a question with a
# free-text answer could not be answered at all. Now it is asked open: the
# transcript is read back once, and recorded when the speaker says record.
#
# Each test here was seen red against a mutation of the adapter: `run_once`
# given back its `request.options or ["approve", "reject"]` fallback fails
# every one of them except the closed-choice test; `return heard.strip()`
# in place of the read-back fails the record, again, no, nomatch, again-budget,
# announcement and both route-around tests; dropping the
# `decision is False or named == "again"` branch fails the again, no,
# again-budget, announcement and names-record tests; `return answer or
# heard.strip()` in place of the raise after the loop fails the two exhaustion
# tests; `decision is True or named == "agree"` in place of the
# confirmation's test fails the negated-record test on every confirmation;
# and the three reason mutations named in the reasons test's docstring each
# fail that test alone.

OPEN_PROMPT = "What should the branch be called?"
READBACK = "I heard: {}. Say agree or again."


def test_an_open_question_records_the_transcript_after_record():
    client = FakeClient()
    client.add_pending("name-1", OPEN_PROMPT)
    stt = ScriptedSTT(["release candidate", "agree"])
    tts = RecordingTTS()

    result = VoiceApprovalLoop(stt=stt, tts=tts, client=client).run_once("name-1")

    assert result.response == "release candidate"
    assert result.responded_by == "vox"
    assert client.submitted == [("name-1", "release candidate")]
    assert tts.spoken == [
        OPEN_PROMPT,
        READBACK.format("release candidate"),
        "Recorded: release candidate",
    ]
    assert stt.calls == 2


def test_a_transcripts_closing_stop_is_not_doubled_when_read_back():
    """A transcript ends with its own stop, and the read-back and the re-ask
    each wrap it in a sentence. Mutation: return the transcript unstripped
    from `_said` -- red, "I heard: Release candidate.. Say agree or again."."""
    client = FakeClient()
    client.add_pending("name-9", OPEN_PROMPT)
    client.add_pending("pick-9", "Ship it?", options=["approve", "hold"])
    tts = RecordingTTS()

    VoiceApprovalLoop(stt=ScriptedSTT(["Release candidate.", "agree"]), tts=tts,
                      client=client).run_once("name-9")
    VoiceApprovalLoop(stt=ScriptedSTT(["Banana!", "hold"]), tts=tts, client=client).run_once("pick-9")

    assert READBACK.format("Release candidate") in tts.spoken
    assert "I heard: Banana. Say approve or hold." in tts.spoken


def test_an_open_question_is_not_read_as_a_closed_choice():
    """The defect: "yes" to an open question was recorded as `approve`."""
    client = FakeClient()
    client.add_pending("name-2", OPEN_PROMPT)
    loop = VoiceApprovalLoop(stt=ScriptedSTT(["yes", "agree"]), tts=RecordingTTS(), client=client)

    assert loop.run_once("name-2").response == "yes"


def test_again_listens_again_and_the_second_transcript_is_recorded():
    client = FakeClient()
    client.add_pending("name-3", OPEN_PROMPT)
    stt = ScriptedSTT(["release candy date", "again", "release candidate", "agree"])
    tts = RecordingTTS()

    result = VoiceApprovalLoop(stt=stt, tts=tts, client=client, max_retries=1).run_once("name-3")

    assert result.response == "release candidate"
    assert client.submitted == [("name-3", "release candidate")]
    assert tts.spoken == [
        OPEN_PROMPT,
        READBACK.format("release candy date"),
        OPEN_PROMPT,
        READBACK.format("release candidate"),
        "Recorded: release candidate",
    ]


def test_a_yes_on_the_read_back_records():
    client = FakeClient()
    client.add_pending("name-4", OPEN_PROMPT)
    loop = VoiceApprovalLoop(stt=ScriptedSTT(["main", "yes"]), tts=RecordingTTS(), client=client)

    assert loop.run_once("name-4").response == "main"


def test_a_no_on_the_read_back_listens_again():
    client = FakeClient()
    client.add_pending("name-5", OPEN_PROMPT)
    stt = ScriptedSTT(["main", "no", "trunk", "agree"])

    result = VoiceApprovalLoop(stt=stt, tts=RecordingTTS(), client=client).run_once("name-5")

    assert result.response == "trunk"
    assert stt.calls == 4


def test_silence_on_an_open_question_is_reasked_as_noinput():
    client = FakeClient()
    client.add_pending("name-6", OPEN_PROMPT)
    tts = RecordingTTS()
    stt = ScriptedSTT(["", "main", "agree"])

    VoiceApprovalLoop(stt=stt, tts=tts, client=client, max_retries=1).run_once("name-6")

    assert tts.spoken[1] == f"I didn't hear anything. {OPEN_PROMPT}"
    assert client.submitted == [("name-6", "main")]


def test_an_unusable_confirmation_is_reasked_with_the_read_back_grammar():
    client = FakeClient()
    client.add_pending("name-7", OPEN_PROMPT)
    tts = RecordingTTS()
    stt = ScriptedSTT(["main", "banana", "agree"])

    result = VoiceApprovalLoop(stt=stt, tts=tts, client=client, max_retries=1).run_once("name-7")

    assert result.response == "main"
    assert tts.spoken[2] == "I heard: banana. Say agree or again."


def test_exhausting_the_budget_with_again_raises_and_submits_nothing():
    client = FakeClient()
    client.add_pending("name-8", OPEN_PROMPT)
    stt = ScriptedSTT(["main", "again"])
    loop = VoiceApprovalLoop(stt=stt, tts=RecordingTTS(), client=client, max_retries=1)

    with pytest.raises(UnclearResponse):
        loop.run_once("name-8")

    assert client.submitted == []
    # The answer and its read-back, once per attempt: two attempts, four listens.
    assert stt.calls == 4


def test_exhausting_the_budget_with_silence_raises_and_submits_nothing():
    client = FakeClient()
    client.add_pending("name-9", OPEN_PROMPT)
    tts = RecordingTTS()
    loop = VoiceApprovalLoop(stt=ScriptedSTT([""]), tts=tts, client=client, max_retries=1)

    with pytest.raises(UnclearResponse):
        loop.run_once("name-9")

    assert client.submitted == []
    assert tts.spoken == [OPEN_PROMPT, f"I didn't hear anything. {OPEN_PROMPT}"]


@pytest.mark.parametrize(
    "confirmation", ["don't agree", "no, agree", "do not agree to that", "I do not agree"]
)
def test_a_negated_agree_on_the_read_back_listens_again(confirmation):
    """Routing around the read-back: a no that names the option it negates is
    a no. Seen red with the read-back testing `named == "agree"` before the
    yes/no decision, which recorded "main" on every one of these."""
    client = FakeClient()
    client.add_pending("name-12", OPEN_PROMPT)
    stt = ScriptedSTT(["main", confirmation, "trunk", "agree"])

    result = VoiceApprovalLoop(stt=stt, tts=RecordingTTS(), client=client).run_once("name-12")

    assert result.response == "trunk"
    assert client.submitted == [("name-12", "trunk")]
    assert stt.calls == 4


def test_an_answer_that_names_agree_is_still_read_back():
    """Routing around the read-back: the first transcript is always the answer,
    so saying "agree" inside it records nothing until it has been read back."""
    client = FakeClient()
    client.add_pending("name-11", OPEN_PROMPT)
    stt = ScriptedSTT(["agree on main", "again", "main", "yes"])

    result = VoiceApprovalLoop(stt=stt, tts=RecordingTTS(), client=client).run_once("name-11")

    assert result.response == "main"
    assert client.submitted == [("name-11", "main")]


def test_a_request_with_options_is_still_a_closed_choice():
    """The yes/no fast path is untouched: one listen, no read-back."""
    client = FakeClient()
    client.add_pending("deploy-8", "Deploy?", options=["approve", "reject"])
    stt = ScriptedSTT(["yes", "again"])
    tts = RecordingTTS()

    result = VoiceApprovalLoop(stt=stt, tts=tts, client=client).run_once("deploy-8")

    assert result.response == "approve"
    assert stt.calls == 1
    assert tts.spoken == ["Deploy? Say approve or reject.", "Recorded: approve"]


# --- what the loop tells a display ---------------------------------------------


class AnnouncingSTT(ScriptedSTT):
    """A scripted engine that also keeps what the dialog announced to it."""

    def __init__(self, transcripts, fail=False):
        super().__init__(transcripts)
        self.announced: list[tuple[str, str, str | None]] = []
        self.fail = fail

    def announce(self, state, text="", reason=None):
        if self.fail:
            raise RuntimeError("the display is away")
        self.announced.append((state, text, reason))
        return True


def test_a_clear_answer_is_announced_as_asked_then_recorded():
    client = FakeClient()
    client.add_pending("deploy-1", "Deploy?", options=["approve", "hold"])
    stt = AnnouncingSTT(["yes"])

    VoiceApprovalLoop(stt=stt, tts=RecordingTTS(), client=client).run_once("deploy-1")

    assert stt.announced == [
        ("speaking", "Deploy? Say approve or hold.", None),
        ("recorded", "approve", None),
    ]


def test_each_reask_is_announced_with_its_reason():
    client = FakeClient()
    client.add_pending("deploy-2", "Deploy?", options=["approve", "hold"])
    stt = AnnouncingSTT(["", "banana", "hold"])

    VoiceApprovalLoop(stt=stt, tts=RecordingTTS(), client=client, max_retries=2).run_once("deploy-2")

    assert [(s, r) for s, _, r in stt.announced] == [
        ("speaking", None),
        ("speaking", "noinput"),
        ("speaking", "nomatch"),
        ("recorded", None),
    ]
    assert stt.announced[2][1].startswith("I heard: banana.")


def test_giving_up_is_announced_with_what_was_last_heard():
    client = FakeClient()
    client.add_pending("deploy-3", "Deploy?", options=["approve", "hold"])
    stt = AnnouncingSTT(["banana"])

    with pytest.raises(UnclearResponse):
        VoiceApprovalLoop(stt=stt, tts=RecordingTTS(), client=client, max_retries=1).run_once("deploy-3")

    assert stt.announced[-1] == ("gave_up", "banana", None)
    assert client.submitted == []


def test_an_open_question_announces_its_read_back_as_a_confirmation():
    client = FakeClient()
    client.add_pending("name-10", OPEN_PROMPT)
    stt = AnnouncingSTT(["main", "again", "trunk", "agree"])

    VoiceApprovalLoop(stt=stt, tts=RecordingTTS(), client=client).run_once("name-10")

    assert stt.announced == [
        ("speaking", OPEN_PROMPT, None),
        ("speaking", READBACK.format("main"), "confirm"),
        ("speaking", OPEN_PROMPT, "again"),
        ("speaking", READBACK.format("trunk"), "confirm"),
        ("recorded", "trunk", None),
    ]


def test_each_open_question_reask_is_announced_with_its_reason():
    """Silence before the answer and silence or an unusable phrase on the
    read-back each re-ask under their own reason, and the answer survives a
    re-asked read-back. Seen red against each of three mutations: the open
    question's noinput reason changed to nomatch, the read-back's nomatch
    reason changed to noinput, and the read-back's silence branch deleted so
    silence fell to the nomatch text."""
    client = FakeClient()
    client.add_pending("name-13", OPEN_PROMPT)
    stt = AnnouncingSTT(["", "main", "", "banana", "agree"])

    VoiceApprovalLoop(stt=stt, tts=RecordingTTS(), client=client,
                      max_retries=3).run_once("name-13")

    assert stt.announced == [
        ("speaking", OPEN_PROMPT, None),
        ("speaking", f"I didn't hear anything. {OPEN_PROMPT}", "noinput"),
        ("speaking", READBACK.format("main"), "confirm"),
        ("speaking", "I didn't hear anything. Say agree or again.", "noinput"),
        ("speaking", "I heard: banana. Say agree or again.", "nomatch"),
        ("recorded", "main", None),
    ]
    assert client.submitted == [("name-13", "main")]


def test_a_display_that_fails_costs_the_dialog_nothing():
    client = FakeClient()
    client.add_pending("deploy-4", "Deploy?", options=["approve", "hold"])

    VoiceApprovalLoop(stt=AnnouncingSTT(["yes"], fail=True), tts=RecordingTTS(),
                      client=client).run_once("deploy-4")

    assert client.submitted == [("deploy-4", "approve")]


def test_the_read_back_s_earlier_word_is_still_taken_as_agree():
    """Mutation: drop the alias -- red, "record" is a mismatch."""
    client = FakeClient()
    client.add_pending("name-13", OPEN_PROMPT)

    result = VoiceApprovalLoop(stt=ScriptedSTT(["main", "record"]), tts=RecordingTTS(),
                               client=client).run_once("name-13")

    assert result.response == "main"
