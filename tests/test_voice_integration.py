import pytest

from qmcp.client.mcp_client import HumanRequest, HumanRequestExpiredError, HumanResponse
from qmcp.integrations.voice.adapter import UnclearResponse, VoiceApprovalLoop, parse_yes_no


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

    def list_human_requests(self, status_filter=None, request_type=None, limit=50, offset=0):
        pending = [r for rid, r in self.requests.items() if rid not in self.responses]
        return pending[:limit]

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
    assert tts.spoken[0] == "Deploy to production?"
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
    assert tts.spoken.count("Sorry, I didn't catch that. Yes or no?") == 2


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


def test_run_forever_stops_when_nothing_pending():
    client = FakeClient()
    stt = ScriptedSTT(["yes"])
    tts = RecordingTTS()

    loop = VoiceApprovalLoop(stt=stt, tts=tts, client=client)
    answered = loop.run_forever(max_iterations=5)

    assert answered == 0
    assert stt.calls == 0
