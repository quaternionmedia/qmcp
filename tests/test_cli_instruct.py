"""`qmcp instruct`, typed and spoken, and `qmcp instructions list` and `show`.

The spoken tests stand in for vox the way `tests/test_cli_human_voice.py` does
and for the server with a fake client, so what is tested is the dialog: what is
asked, what is listened for and how, and what is recorded. The server's half is
`tests/test_instructions_route.py`; both halves together are
`qmcp cookbook instruct`.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

import qmcp.cli as cli
from qmcp.instructions import dialog as dialog_module

NAMES = ("qmcp", "dossier", "vox")


class _FakeSTT:
    """Returns each transcript in order, repeating the last once exhausted, and
    keeps what each take asked for."""

    def __init__(self, transcripts: list[str]):
        self._transcripts = list(transcripts)
        self.calls = 0
        self.takes: list[tuple[float, int | None]] = []
        self.announced: list[tuple[str, str, str | None]] = []

    def listen(self, duration: float = 5.0, *, pause_ms: int | None = None) -> tuple[str, str]:
        text = self._transcripts[min(self.calls, len(self._transcripts) - 1)]
        self.calls += 1
        self.takes.append((duration, pause_ms))
        return text, f"capture_{self.calls}.wav"

    def announce(self, state: str, text: str = "", reason: str | None = None) -> bool:
        self.announced.append((state, text, reason))
        return True

    # `HttpSTT(...)` is used as a context manager by the command.
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


class _FakeTTS:
    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text: str, out_path: str | None = None) -> str:
        self.spoken.append(text)
        return out_path or "spoken.wav"


class _FakeClient:
    """Stands in for MCPClient's inbox surface: no HTTP, no server."""

    def __init__(self):
        self.base_url = "http://127.0.0.1:3141"
        self.recorded: list[dict] = []
        self.rows: dict[str, dict] = {}

    def create_instruction(self, text, source="typed", project=None, heard=None):
        row = {"id": f"row-{len(self.recorded) + 1}", "text": text, "source": source,
               "project": project, "status": "recorded" if project else "unresolved",
               "created_at": "2026-10-03T12:00:00", "updated_at": "2026-10-03T12:00:00",
               "detail": {"candidates": [], "rule": "stated" if project else "no project named",
                          **({"heard": heard} if heard is not None else {})}}
        self.recorded.append({"text": text, "source": source, "project": project, "heard": heard})
        self.rows[row["id"]] = row
        return row

    def list_instructions(self, status=None, limit=50):
        rows = list(reversed(self.rows.values()))
        return [r for r in rows if status is None or r["status"] == status][:limit]

    def get_instruction(self, instruction_id):
        from qmcp.client import MCPClientError

        if instruction_id not in self.rows:
            raise MCPClientError(f"Instruction '{instruction_id}' not found")
        return self.rows[instruction_id]


@pytest.fixture(autouse=True)
def _servers_up(monkeypatch):
    monkeypatch.setattr(cli, "_voice_preflight", lambda *a, **k: None)
    monkeypatch.setattr("qmcp.instructions.roster_names", lambda corpus=None: NAMES)


@pytest.fixture
def fake_client(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr("qmcp.client.MCPClient", lambda *a, **k: client)
    return client


def _install_fake_vox(monkeypatch, stt, tts) -> None:
    fake_vox = ModuleType("vox")
    fake_vox.HttpSTT = lambda *a, **k: stt
    fake_contract = MagicMock(name="EngineContract")
    fake_adapters = ModuleType("vox.adapters")
    fake_adapters.JOE = fake_contract
    fake_joe = ModuleType("vox.adapters.joe")
    fake_joe.JOE = fake_contract
    fake_joe.DEFAULT_URL = "http://127.0.0.1:8000"
    fake_pyttsx3 = ModuleType("vox.adapters.pyttsx3")
    fake_pyttsx3.Pyttsx3TTS = lambda *a, **k: tts
    for name, module in [("vox", fake_vox), ("vox.adapters", fake_adapters),
                         ("vox.adapters.joe", fake_joe), ("vox.adapters.pyttsx3", fake_pyttsx3)]:
        monkeypatch.setitem(sys.modules, name, module)


def _spoken(monkeypatch, transcripts, *extra_args):
    stt, tts = _FakeSTT(transcripts), _FakeTTS()
    _install_fake_vox(monkeypatch, stt, tts)
    result = CliRunner().invoke(cli.cli, ["instruct", "--voice", *extra_args])
    return result, stt, tts


# --- typed ---------------------------------------------------------------------


def test_a_typed_instruction_is_recorded_over_http(fake_client):
    result = CliRunner().invoke(cli.cli, ["instruct", "Deploy qmcp to the pi."])

    assert result.exit_code == 0, result.output
    assert fake_client.recorded == [{"text": "Deploy qmcp to the pi.", "source": "typed",
                                     "project": None, "heard": None}]
    assert "row-1" in result.output and "Deploy qmcp to the pi." in result.output


def test_a_stated_project_and_source_travel_with_it(fake_client):
    result = CliRunner().invoke(
        cli.cli, ["instruct", "Rotate the logs.", "--project", "dossier", "--source", "page"])

    assert result.exit_code == 0, result.output
    assert fake_client.recorded[0]["project"] == "dossier"
    assert fake_client.recorded[0]["source"] == "page"
    assert "[=] row-1  dossier  page" in result.output


def test_an_unresolved_row_prints_its_candidates(fake_client):
    result = CliRunner().invoke(cli.cli, ["instruct", "Rotate the logs."])

    assert result.exit_code == 0, result.output
    assert "[?] row-1  unresolved" in result.output
    assert "candidates: none  (no project named)" in result.output


def test_text_or_voice_and_not_both(fake_client):
    neither = CliRunner().invoke(cli.cli, ["instruct"])
    both = CliRunner().invoke(cli.cli, ["instruct", "Deploy qmcp.", "--voice"])

    assert neither.exit_code != 0 and "TEXT, or --voice" in neither.output
    assert both.exit_code != 0 and "pass one" in both.output
    assert fake_client.recorded == []


# --- spoken ----------------------------------------------------------------------


def test_a_spoken_instruction_is_read_back_and_recorded_on_record(fake_client, monkeypatch):
    """Mutation: skip the read-back and record the first take -- red on
    `tts.spoken[1]` and on `heard`."""
    result, stt, tts = _spoken(monkeypatch, ["Deploy qmcp to the pi.", "record"])

    assert result.exit_code == 0, result.output
    assert tts.spoken[0] == "What should be done?"
    assert tts.spoken[1] == "I heard: Deploy qmcp to the pi.. Say record or again."
    assert tts.spoken[2] == "Recorded for qmcp."
    assert fake_client.recorded == [{
        "text": "Deploy qmcp to the pi.", "source": "voice", "project": "qmcp",
        "heard": ["Deploy qmcp to the pi.", "record"]}]
    assert "[=] row-1  qmcp  voice" in result.output


def test_the_instruction_take_is_long_and_the_confirmation_is_not(fake_client, monkeypatch):
    """THE REASON THE STACK EXISTS. Mutation: pass `pause_ms` on every listen
    -- red on the second take; drop it from the first -- red on the first."""
    result, stt, tts = _spoken(monkeypatch, ["Deploy qmcp.", "yes"],
                               "--duration", "20", "--pause-ms", "1200")

    assert result.exit_code == 0, result.output
    assert stt.takes == [(20.0, 1200), (5.0, None)]


def test_the_defaults_are_a_long_cap_and_a_long_pause(fake_client, monkeypatch):
    result, stt, tts = _spoken(monkeypatch, ["Deploy qmcp.", "yes"])

    assert result.exit_code == 0, result.output
    assert stt.takes[0] == (dialog_module.LISTEN_DURATION, dialog_module.PAUSE_MS)
    assert stt.takes[0][0] >= 20 and stt.takes[0][1] >= 1000


def test_again_takes_the_instruction_a_second_time(fake_client, monkeypatch):
    """Mutation: treat `again` as a nomatch -- red: the second take is never
    asked for and the first text is recorded."""
    result, stt, tts = _spoken(monkeypatch, [
        "Deploy qmcp.", "again", "Deploy qmcp to the pi.", "record"])

    assert result.exit_code == 0, result.output
    assert tts.spoken == [
        "What should be done?",
        "I heard: Deploy qmcp.. Say record or again.",
        "What should be done?",
        "I heard: Deploy qmcp to the pi.. Say record or again.",
        "Recorded for qmcp.",
    ]
    assert fake_client.recorded[0]["text"] == "Deploy qmcp to the pi."
    assert fake_client.recorded[0]["heard"] == [
        "Deploy qmcp.", "again", "Deploy qmcp to the pi.", "record"]
    # Both long takes asked for the pause; neither confirmation did.
    assert [pause for _, pause in stt.takes] == [1500, None, 1500, None]


def test_no_asks_again_as_a_no_does_for_an_approval(fake_client, monkeypatch):
    result, stt, tts = _spoken(monkeypatch, ["Deploy qmcp.", "no", "Deploy vox.", "yes"])

    assert result.exit_code == 0, result.output
    assert fake_client.recorded[0]["text"] == "Deploy vox."
    assert fake_client.recorded[0]["project"] == "vox"


def test_an_ambiguous_project_is_asked_back_as_a_closed_choice(fake_client, monkeypatch):
    """Mutation: record `candidates[0]` instead of asking -- red on the
    project and on the question."""
    result, stt, tts = _spoken(monkeypatch, [
        "Move the vectors from vox into qmcp.", "record", "vox"])

    assert result.exit_code == 0, result.output
    assert tts.spoken[2] == "Which project? Say qmcp or vox."
    assert fake_client.recorded[0]["project"] == "vox"
    assert fake_client.recorded[0]["heard"][-1] == "vox"


def test_yes_chooses_no_project(fake_client, monkeypatch):
    """A project is never picked by position. "yes" to "Say qmcp or vox." is
    a nomatch, re-asked; the budget spent, the row is recorded unresolved.
    Mutation: fall back to `choose_option` -- red, `qmcp` recorded."""
    result, stt, tts = _spoken(monkeypatch, [
        "Move the vectors from vox into qmcp.", "record", "yes"], "--max-retries", "1")

    assert result.exit_code == 0, result.output
    assert tts.spoken[3] == "I heard: yes. Say qmcp or vox."
    assert fake_client.recorded[0]["project"] is None
    assert tts.spoken[-1] == "Recorded. The project is unresolved."
    assert "[?] row-1  unresolved  voice" in result.output


def test_a_missing_project_is_asked_for_once(fake_client, monkeypatch):
    """Mutation: skip `_ask_project_once` -- red on the project."""
    result, stt, tts = _spoken(monkeypatch, ["Rotate the logs.", "record", "dossier"])

    assert result.exit_code == 0, result.output
    assert tts.spoken[2] == "Which project?"
    assert fake_client.recorded[0]["project"] == "dossier"


def test_a_missing_project_that_still_matches_nothing_is_recorded_unresolved(
        fake_client, monkeypatch):
    """Asked once, not twice, and the instruction is kept."""
    result, stt, tts = _spoken(monkeypatch, ["Rotate the logs.", "record", "the thing"])

    assert result.exit_code == 0, result.output
    assert tts.spoken.count("Which project?") == 1
    assert stt.calls == 3
    assert fake_client.recorded[0]["project"] is None
    assert fake_client.recorded[0]["text"] == "Rotate the logs."


def test_nothing_heard_is_said_and_the_prompt_repeated(fake_client, monkeypatch):
    result, stt, tts = _spoken(monkeypatch, ["", "Deploy qmcp.", "record"])

    assert result.exit_code == 0, result.output
    assert tts.spoken[1] == "I didn't hear anything. What should be done?"
    assert fake_client.recorded[0]["heard"] == ["", "Deploy qmcp.", "record"]


def test_an_instruction_nobody_confirms_records_nothing(fake_client, monkeypatch):
    """Mutation: record `answer` when the budget runs out -- red."""
    result, stt, tts = _spoken(monkeypatch, ["Deploy qmcp.", "banana"], "--max-retries", "1")

    assert result.exit_code != 0
    assert "No usable instruction" in result.output
    assert fake_client.recorded == []
    assert tts.spoken[2] == "I heard: banana. Say record or again."
    assert stt.announced[-1][0] == "gave_up"


def test_the_dialog_announces_its_states_with_the_shared_reasons(fake_client, monkeypatch):
    """`confirm` on the read-back and `again` on a second take, beside the
    `noinput`/`nomatch` the approval dialog uses."""
    result, stt, tts = _spoken(monkeypatch, [
        "Deploy qmcp.", "again", "", "Deploy qmcp to the pi.", "record"])

    assert result.exit_code == 0, result.output
    assert [(state, reason) for state, _, reason in stt.announced] == [
        ("speaking", None),          # What should be done?
        ("speaking", "confirm"),     # I heard: Deploy qmcp.
        ("speaking", "again"),       # What should be done?
        ("speaking", "noinput"),     # I didn't hear anything.
        ("speaking", "confirm"),     # I heard: Deploy qmcp to the pi.
        ("recorded", None),
    ]


def test_instruct_voice_without_vox_fails_clearly(fake_client, monkeypatch):
    monkeypatch.setitem(sys.modules, "vox", None)

    result = CliRunner().invoke(cli.cli, ["instruct", "--voice"])

    assert result.exit_code != 0
    assert "vox is not importable" in result.output
    assert "`uv run qmcp instruct --voice` again" in result.output


# --- the inbox listed and shown -----------------------------------------------------


def test_instructions_list_prints_what_is_recorded(fake_client):
    fake_client.create_instruction("Deploy qmcp.", project="qmcp")
    fake_client.create_instruction("Rotate the logs.")

    result = CliRunner().invoke(cli.cli, ["instructions", "list"])

    assert result.exit_code == 0, result.output
    assert result.output.index("row-2") < result.output.index("row-1")
    assert "2 instruction(s)." in result.output

    only = CliRunner().invoke(cli.cli, ["instructions", "list", "--status", "unresolved"])
    assert "row-2" in only.output and "row-1" not in only.output


def test_instructions_list_with_nothing_recorded(fake_client):
    result = CliRunner().invoke(cli.cli, ["instructions", "list", "--status", "recorded"])

    assert result.exit_code == 0, result.output
    assert "Nothing recorded as recorded." in result.output


def test_instructions_show_prints_the_evidence(fake_client):
    fake_client.create_instruction("Deploy qmcp.", project="qmcp")

    result = CliRunner().invoke(cli.cli, ["instructions", "show", "row-1"])

    assert result.exit_code == 0, result.output
    assert '"rule": "stated"' in result.output

    missing = CliRunner().invoke(cli.cli, ["instructions", "show", "nobody"])
    assert missing.exit_code != 0 and "not found" in missing.output
