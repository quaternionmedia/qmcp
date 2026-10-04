"""`qmcp.instructions.act`: the clone, the continuity, the budget, the consent,
and what runs when.

The human queue is stood in for by an in-memory object with the client's
surface, so a test can decide how a consent ends -- approved, held, expired --
and count what was read while it waited. The inbox is a database made for the
test, and it is also the record the clone and the history are read from. The
runtime is the scripted one, whose `briefs` say what ran and what it was told.
"""

from __future__ import annotations

import pytest

from qmcp.client import HumanRequestConflictError
from qmcp.client.mcp_client import HumanRequest, HumanResponse
from qmcp.db.models import Instruction, InstructionSource, InstructionStatus
from qmcp.instructions import act as act_module
from qmcp.instructions import continuity
from qmcp.instructions.act import (
    RULE_CWD,
    RULE_RECORD,
    NoSuchInstruction,
    act,
    rows_at,
)
from qmcp.integrations.agents.scripted import ScriptedRuntime
from qmcp.spend import Budget

PROJECT = "qmcp"


class _Queue:
    """The human queue, in memory, with the client's surface.

    `reads` counts calls to the single-request route, which is the one that
    expires a request as a side effect; `script` says how each request ends
    once the act is waiting on it. `clock` is the act's clock, and every
    wait moves it a minute, so a consent the script never settles runs out
    the act's deadline in a few listings rather than in real time.
    """

    base_url = "http://127.0.0.1:3141"

    def __init__(self, script: dict[str, str | None] | None = None):
        self.requests: dict[str, dict] = {}
        self.responses: dict[str, str] = {}
        self.expired: set[str] = set()
        self.reads = 0
        self.reads_while_pending = 0
        self.listings = 0
        self.script = script or {}
        self.now = 0.0

    def clock(self) -> float:
        return self.now

    def create_human_request(self, request_id, request_type, prompt, options=None,
                             context=None, timeout_seconds=3600, correlation_id=None):
        if request_id in self.requests:
            raise HumanRequestConflictError(request_id)
        self.requests[request_id] = {"id": request_id, "prompt": prompt, "options": options,
                                     "context": context, "timeout_seconds": timeout_seconds,
                                     "request_type": request_type}
        return HumanRequest(id=request_id, request_type=request_type, prompt=prompt,
                            status="pending", created_at="now")

    def _pending(self, request_id):
        return request_id not in self.responses and request_id not in self.expired

    def list_human_requests(self, status_filter=None, request_type=None, limit=50,
                            offset=0, oldest_first=False):
        self.listings += 1
        rows = [r for r in self.requests.values()
                if status_filter != "pending" or self._pending(r["id"])]
        return [HumanRequest(id=r["id"], request_type=r["request_type"], prompt=r["prompt"],
                             status="pending", created_at="now", options=r["options"],
                             context=r["context"])
                for r in rows[offset:offset + limit]]

    def get_human_request(self, request_id):
        self.reads += 1
        if self._pending(request_id):
            self.reads_while_pending += 1
        r = self.requests[request_id]
        status = ("responded" if request_id in self.responses
                  else "expired" if request_id in self.expired else "pending")
        request = HumanRequest(id=request_id, request_type=r["request_type"], prompt=r["prompt"],
                               status=status, created_at="now", options=r["options"],
                               context=r["context"])
        response = None
        if request_id in self.responses:
            response = HumanResponse(id="r", request_id=request_id,
                                     response=self.responses[request_id],
                                     responded_by="test", created_at="now")
        return request, response

    def submit_human_response(self, request_id, response, responded_by=None, metadata=None):
        self.responses[request_id] = response
        return HumanResponse(id="r", request_id=request_id, response=response,
                             responded_by=responded_by, created_at="now")

    def settle(self, _seconds: float) -> None:
        """Stands in for `sleep`: the person answers while the act waits."""
        self.now += 60.0
        for request_id, answer in self.script.items():
            if request_id in self.requests and self._pending(request_id):
                if answer is None:
                    self.expired.add(request_id)
                else:
                    self.responses[request_id] = answer


@pytest.fixture
def inbox(tmp_path):
    rows = rows_at(tmp_path / "inbox.db")

    def record(text=f"Deploy {PROJECT} to the pi.", project=PROJECT, status=InstructionStatus.RECORDED):
        with rows() as session:
            row = Instruction(text=text, project=project, source=InstructionSource.TYPED,
                              status=status, detail={"candidates": [project], "rule": "x"})
            session.add(row)
            session.commit()
            return row.id

    def read(instruction_id):
        with rows() as session:
            return session.get(Instruction, instruction_id)

    rows.record, rows.read = record, read
    return rows


@pytest.fixture
def clone(tmp_path):
    path = tmp_path / "checkouts" / PROJECT
    path.mkdir(parents=True)
    return path


def _act(inbox, instruction_id, queue, runtime=None, budget=1, **kw):
    runtime = runtime or ScriptedRuntime(text="pinned")
    return act(instruction_id, runtime, Budget(authorised=budget), client=queue, rows=inbox,
               sleep=queue.settle, clock=queue.clock, poll_interval=0.0, **kw), runtime


def _approved(inbox, text, project=PROJECT, outcome="pinned", **kw):
    """Record an instruction and act on it with an approve, as a person would."""
    instruction_id = inbox.record(text, project=project)
    queue = _Queue({f"instruction-{instruction_id}": "approve"})
    done, runtime = _act(inbox, instruction_id, queue, runtime=ScriptedRuntime(text=outcome), **kw)
    return instruction_id, done, runtime


# --- the clone ------------------------------------------------------------------------


def test_cwd_is_taken_as_given(inbox, clone):
    instruction_id, done, runtime = _approved(inbox, f"Deploy {PROJECT}.", cwd=clone)

    assert done.cwd == str(clone) and done.status == "done"
    assert runtime.calls == [{"instruction": f"Deploy {PROJECT}.", "cwd": str(clone)}]
    assert inbox.read(instruction_id).detail["clone"] == {"cwd": str(clone), "rule": RULE_CWD}


def test_the_clone_is_remembered_from_the_projects_last_act(inbox, clone):
    """A path given once serves the project's later instructions. Mutation:
    drop the `last_clone` fallback -- red, the second act refuses."""
    _approved(inbox, f"Deploy {PROJECT}.", cwd=clone)

    second, done, runtime = _approved(inbox, f"Tag {PROJECT}.")

    assert done.status == "done" and done.cwd == str(clone)
    assert runtime.calls[0]["cwd"] == str(clone)
    assert inbox.read(second).detail["clone"] == {"cwd": str(clone), "rule": RULE_RECORD}


def test_a_remembered_clone_belongs_to_its_project(inbox, clone):
    """Mutation: drop the project filter from `last_clone` -- red, another
    project's instruction runs in this project's clone."""
    _approved(inbox, f"Deploy {PROJECT}.", cwd=clone)

    _, done, runtime = _approved(inbox, "Rotate the logs.", project="vox")

    assert done.status == "recorded" and runtime.calls == []
    assert "no clone for 'vox' in qmcp's record yet" in done.why


def test_cwd_wins_over_the_remembered_clone(inbox, clone, tmp_path):
    """Mutation: consult the record first and fall back to `cwd` -- red."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _approved(inbox, f"Deploy {PROJECT}.", cwd=clone)

    second, done, _ = _approved(inbox, f"Tag {PROJECT}.", cwd=elsewhere)

    assert done.cwd == str(elsewhere)
    assert inbox.read(second).detail["clone"]["rule"] == RULE_CWD


def test_a_remembered_clone_that_is_gone_is_a_refusal(inbox, clone):
    _approved(inbox, f"Deploy {PROJECT}.", cwd=clone)
    clone.rmdir()

    _, done, runtime = _approved(inbox, f"Tag {PROJECT}.")

    assert runtime.calls == [] and "is not a directory" in done.why and "--cwd" in done.why


# --- the continuity -----------------------------------------------------------------------


def test_the_brief_carries_the_projects_earlier_outcomes_oldest_first(inbox, clone):
    """Continuity comes from qmcp, not the model. Mutation: drop the project
    filter from `history` -- red, vox's work is carried; carry held
    instructions -- red, `Rotate` appears; order newest first -- red."""
    first, *_ = _approved(inbox, "Find the README.", outcome="README.md, at the root.", cwd=clone)
    second, *_ = _approved(inbox, "Read it.", outcome="It says qmcp is the local backend.")
    _approved(inbox, "Elsewhere.", project="vox", outcome="vox things", cwd=clone)
    held = inbox.record("Rotate the logs.")
    _act(inbox, held, _Queue({f"instruction-{held}": "hold"}))

    third, done, runtime = _approved(inbox, "Summarise what we found.")

    brief = runtime.briefs[0]
    assert [turn.instruction for turn in brief.history] == ["Find the README.", "Read it."]
    assert [turn.outcome for turn in brief.history] == [
        "README.md, at the root.", "It says qmcp is the local backend."]
    assert all(turn.status == "done" and turn.runtime == "scripted" for turn in brief.history)
    prompt = brief.prompt()
    assert prompt.index("README.md, at the root.") < prompt.index("the local backend")
    assert "vox things" not in prompt and "Rotate" not in prompt
    assert done.carried == (first, second)
    assert inbox.read(third).detail["continuity"] == [first, second]


def test_the_consent_says_how_much_history_is_carried(inbox, clone):
    """The person at the gate is told the runtime will be told about earlier
    work. Mutation: drop `carried` from the prompt -- red."""
    _approved(inbox, "Find the README.", cwd=clone)
    instruction_id = inbox.record("Read it.")
    queue = _Queue({f"instruction-{instruction_id}": "hold"})

    _act(inbox, instruction_id, queue)

    request = queue.requests[f"instruction-{instruction_id}"]
    assert "carrying 1 earlier instruction(s) from qmcp's record" in request["prompt"]
    assert len(request["context"]["carried"]) == 1


def test_continuity_survives_a_change_of_runtime(inbox, clone):
    """The history is in the record, so a different runtime carries the next
    instruction knowing what the first one found."""
    class Other(ScriptedRuntime):
        name = "other"

    _approved(inbox, "Find the README.", outcome="README.md, at the root.", cwd=clone)
    instruction_id = inbox.record("Read it.")
    queue = _Queue({f"instruction-{instruction_id}": "approve"})

    done, runtime = _act(inbox, instruction_id, queue, runtime=Other(text="read"))

    (turn,) = runtime.briefs[0].history
    assert (turn.runtime, turn.outcome) == ("scripted", "README.md, at the root.")
    assert "done by scripted: README.md, at the root." in runtime.briefs[0].prompt()
    assert inbox.read(instruction_id).runtime == "other"


def test_the_history_is_capped_at_its_limit_keeping_the_newest(inbox, clone):
    """Mutation: drop the `limit` -- red."""
    for n in range(continuity.LIMIT + 2):
        _approved(inbox, f"Step {n}.", cwd=clone)

    _, _, runtime = _approved(inbox, "Next.")

    names = [turn.instruction for turn in runtime.briefs[0].history]
    assert names == [f"Step {n}." for n in range(2, continuity.LIMIT + 2)]


def test_an_unresolved_instruction_carries_no_history(inbox, clone):
    _approved(inbox, "Find the README.", cwd=clone)
    assert continuity.history(inbox, None, before="x") == ()
    assert continuity.last_clone(inbox, None, before="x") is None


def test_no_clone_is_a_refusal_that_names_what_to_pass(inbox, tmp_path):
    """Status unchanged, nothing asked, nothing run, and the declaration
    present. Mutation: fall back to the working directory -- red."""
    instruction_id = inbox.record()
    unresolved = inbox.record("Rotate the logs.", project=None, status=InstructionStatus.UNRESOLVED)
    queue = _Queue()

    done, runtime = _act(inbox, instruction_id, queue)
    other, _ = _act(inbox, unresolved, queue)
    missing, _ = _act(inbox, instruction_id, queue, cwd=tmp_path / "nowhere")

    for result in (done, other, missing):
        assert "--cwd" in result.why and result.request_id is None
        assert result.stages == ("instruction", "clone")
        assert result.declared["made"] == 0 and "unknown" in result.declared["would_need"]
    assert f"no clone for {PROJECT!r} in qmcp's record yet" in done.why
    assert "unresolved" in other.why
    assert "not a directory" in missing.why
    assert done.status == "recorded" and other.status == "unresolved"
    assert runtime.calls == [] and queue.requests == {}
    assert inbox.read(instruction_id).status == InstructionStatus.RECORDED


def test_an_unknown_instruction_is_refused_outright(inbox):
    with pytest.raises(NoSuchInstruction, match="nobody"):
        act("nobody", ScriptedRuntime(), Budget(authorised=1), client=_Queue(), rows=inbox)


# --- the budget ---------------------------------------------------------------------


def test_a_zero_budget_declares_and_asks_nothing(inbox, clone):
    """Clause 3 of the record: the free pass resolves the clone, says what
    would be asked, and stops. Mutation: ask consent whatever the budget --
    red on `queue.requests`."""
    instruction_id = inbox.record()
    queue = _Queue()

    done, runtime = _act(inbox, instruction_id, queue, budget=0, cwd=clone)

    assert done.stages == ("instruction", "clone", "budget")
    assert done.declared == {"authorised": 0, "made": 0, "service": "scripted", "free_pass": True,
                             "would_need": done.declared["would_need"]}
    assert "unknown" in done.declared["would_need"]
    assert "--budget 1" in done.why
    assert done.cwd == str(clone)
    assert queue.requests == {} and runtime.calls == []
    assert inbox.read(instruction_id).status == InstructionStatus.RECORDED


# --- the consent ------------------------------------------------------------------------


def test_the_consent_says_everything_a_stranger_needs(inbox, clone):
    instruction_id = inbox.record()
    queue = _Queue({f"instruction-{instruction_id}": "hold"})

    done, _ = _act(inbox, instruction_id, queue, cwd=clone, budget=2)

    request = queue.requests[f"instruction-{instruction_id}"]
    assert request["options"] == ["approve", "hold"]
    assert request["timeout_seconds"] == 600
    for piece in (f"Deploy {PROJECT} to the pi.", PROJECT, str(clone), "scripted", "2 run(s)"):
        assert piece in request["prompt"], piece
    assert request["context"]["instruction_id"] == instruction_id
    assert request["context"]["spend"]["authorised"] == 2
    assert request["context"]["cwd"] == str(clone)


def test_the_wait_reads_the_pending_listing_and_never_the_request_while_pending(inbox, clone):
    """`AGENTS.md`: one read in this API is a write. Mutation: wait with
    `client.get_human_request` -- red on `reads_while_pending`."""
    instruction_id = inbox.record()
    queue = _Queue()
    answered = []

    def settle(_seconds):
        # Three listings pass before the person answers.
        if queue.listings >= 3 and not answered:
            queue.responses[f"instruction-{instruction_id}"] = "approve"
            answered.append(True)

    done = act(instruction_id, ScriptedRuntime(), Budget(authorised=1), client=queue,
               rows=inbox, cwd=clone, sleep=settle, poll_interval=0.0)

    assert done.status == "done"
    assert queue.listings >= 3
    assert queue.reads_while_pending == 0
    assert queue.reads == 1


def test_the_listing_is_searched_past_its_first_page(inbox, clone):
    """A queue longer than one page still finds its request. Mutation: read
    the first page only -- red, the act reads the request while pending."""
    instruction_id = inbox.record()
    queue = _Queue({f"instruction-{instruction_id}": "hold"})
    for n in range(act_module.PAGE + 5):
        queue.create_human_request(f"filler-{n}", "approval", "?", ["approve", "hold"])

    done, _ = _act(inbox, instruction_id, queue, cwd=clone)

    assert done.status == "refused"
    assert queue.reads_while_pending == 0


def test_approve_runs_in_the_clone_and_records_the_outcome(inbox, clone):
    """The one path that runs. Mutation: run before reading the answer --
    red on `test_hold_runs_nothing`; skip `consented` -- red on `seen`."""
    instruction_id = inbox.record()
    queue = _Queue({f"instruction-{instruction_id}": "approve"})
    seen: list[str] = []
    original = act_module._update

    def watching(rows, iid, **fields):
        if "status" in fields:
            seen.append(fields["status"].value)
        return original(rows, iid, **fields)

    act_module._update = watching
    try:
        done, runtime = _act(inbox, instruction_id, queue, cwd=clone)
    finally:
        act_module._update = original

    assert done.status == "done" and done.ran
    assert done.stages == ("instruction", "clone", "budget", "ask", "answer", "run", "record")
    assert seen == ["asking", "consented", "acting", "done"]
    assert done.declared["made"] == 1 and done.declared["authorised"] == 1
    assert done.declared["would_need"] == 0  # the scripted runtime's own count
    assert done.answer == "approve" and done.outcome.text == "pinned"
    row = inbox.read(instruction_id)
    assert row.status == InstructionStatus.DONE
    assert (row.runtime, row.cwd, row.exit_code, row.outcome_text) == ("scripted", str(clone), 0, "pinned")
    assert row.acted_at is not None and row.detail["continuity"] == []
    assert row.consent_request_id == f"instruction-{instruction_id}"
    assert row.declared == done.declared
    assert row.detail["outcome"]["spent"] == 0 and row.detail["candidates"] == [PROJECT]


def test_hold_runs_nothing(inbox, clone):
    """Mutation: run on any answer -- red."""
    instruction_id = inbox.record()
    queue = _Queue({f"instruction-{instruction_id}": "hold"})

    done, runtime = _act(inbox, instruction_id, queue, cwd=clone)

    assert done.status == "refused" and not done.ran
    assert done.stages == ("instruction", "clone", "budget", "ask", "answer")
    assert runtime.calls == []
    assert done.declared["made"] == 0 and done.declared["authorised"] == 1
    assert "nothing ran" in done.why
    row = inbox.read(instruction_id)
    assert row.status == InstructionStatus.REFUSED and row.declared["made"] == 0
    assert row.acted_at is not None and row.exit_code is None


def test_an_answer_outside_the_options_runs_nothing(inbox, clone):
    """The server refuses a response outside the options, so only the exact
    word reaches the act; a queue that let one through still runs nothing.
    Mutation: compare `answer.strip().lower()` -- red on `Approve`."""
    ids = [inbox.record() for _ in range(3)]
    queue = _Queue({f"instruction-{ids[0]}": "Approve", f"instruction-{ids[1]}": "approve ",
                    f"instruction-{ids[2]}": "yes"})

    results = [_act(inbox, i, queue, cwd=clone) for i in ids]

    assert [done.status for done, _ in results] == ["refused"] * 3
    assert all(runtime.calls == [] for _, runtime in results)


def test_a_listing_that_never_drops_the_request_ends_at_the_acts_own_deadline(inbox, clone):
    """The fallback behind the listing: the act's clock passes the expiry
    and two polls, the act reads the request once, finds no answer and
    records `unanswered`, while the request still sits pending on the queue
    for anyone to answer. Mutation: remove the deadline `break` -- red, the
    stand-in for `sleep` fails the test at its tenth call rather than
    letting the wait run forever."""
    instruction_id = inbox.record()
    queue = _Queue()  # never settles, never expires
    waits: list[float] = []

    def wait(seconds):
        queue.settle(seconds)
        waits.append(seconds)
        assert len(waits) < 10, "the wait did not end at the deadline"

    done = act(instruction_id, ScriptedRuntime(), Budget(authorised=1), client=queue, rows=inbox,
               cwd=clone, sleep=wait, clock=queue.clock, poll_interval=0.0,
               consent_seconds=120)

    assert done.status == "unanswered" and not done.ran
    assert queue.reads == 1 and queue.reads_while_pending == 1
    assert queue._pending(f"instruction-{instruction_id}")
    assert inbox.read(instruction_id).status == InstructionStatus.UNANSWERED


def test_expiry_runs_nothing(inbox, clone):
    """The request leaves the pending listing with no answer, and the one read
    afterwards says `expired`. Mutation: treat a missing answer as approve
    -- red."""
    instruction_id = inbox.record()
    queue = _Queue({f"instruction-{instruction_id}": None})

    done, runtime = _act(inbox, instruction_id, queue, cwd=clone)

    assert done.status == "unanswered" and runtime.calls == []
    assert done.answer is None and "expired" in done.why
    assert inbox.read(instruction_id).status == InstructionStatus.UNANSWERED
    assert queue.reads == 1


def test_the_declaration_is_on_the_row_on_every_path(inbox, clone, tmp_path):
    """Refused before the ask, held, expired and run: each leaves
    `qmcp.spend.declare` on the row or in the result. Mutation: write
    `declared` only on the run path -- red on `refused`."""
    ids = [inbox.record() for _ in range(3)]
    queue = _Queue({f"instruction-{ids[0]}": "hold", f"instruction-{ids[1]}": None,
                    f"instruction-{ids[2]}": "approve"})

    results = [_act(inbox, i, queue, cwd=clone)[0] for i in ids]
    free, _ = _act(inbox, inbox.record(), queue, budget=0, cwd=clone)
    no_clone, _ = _act(inbox, inbox.record(), queue)

    for result in (*results, free, no_clone):
        assert set(result.declared) == {"authorised", "made", "would_need", "service", "free_pass"}
    for i in ids:
        assert inbox.read(i).declared is not None
    assert [r.status for r in results] == ["refused", "unanswered", "done"]


def test_a_failing_runtime_is_recorded_as_failed(inbox, clone):
    instruction_id = inbox.record()
    queue = _Queue({f"instruction-{instruction_id}": "approve"})

    done, _ = _act(inbox, instruction_id, queue, cwd=clone,
                   runtime=ScriptedRuntime(text="no such branch", exit_code=2))

    assert done.status == "failed" and done.outcome.exit_code == 2
    assert inbox.read(instruction_id).status == InstructionStatus.FAILED
    assert inbox.read(instruction_id).outcome_text == "no such branch"


def test_a_runtime_that_raises_is_a_failed_run_not_a_crash(inbox, clone):
    class Broken:
        name = "broken"

        def run(self, brief, on_event=None):
            raise OSError("no executable")

    instruction_id = inbox.record()
    queue = _Queue({f"instruction-{instruction_id}": "approve"})

    done, _ = _act(inbox, instruction_id, queue, cwd=clone, runtime=Broken())

    assert done.status == "failed"
    assert "OSError: no executable" in done.outcome.text
    assert "unknown" in done.declared["would_need"]


def test_acting_again_asks_under_a_numbered_request(inbox, clone):
    """The first consent is a record and is not overwritten."""
    instruction_id = inbox.record()
    queue = _Queue({f"instruction-{instruction_id}": "hold",
                    f"instruction-{instruction_id}-2": "approve"})

    first, _ = _act(inbox, instruction_id, queue, cwd=clone)
    second, _ = _act(inbox, instruction_id, queue, cwd=clone)

    assert (first.status, second.status) == ("refused", "done")
    assert second.request_id == f"instruction-{instruction_id}-2"
    assert inbox.read(instruction_id).consent_request_id == second.request_id


# --- answered by voice, in the act -----------------------------------------------------------


class _STT:
    def __init__(self, heard):
        self.heard = heard

    def listen(self, duration=5.0, *, pause_ms=None):
        return self.heard, "take.wav"


class _TTS:
    def __init__(self):
        self.spoken = []

    def speak(self, text, out_path=None):
        self.spoken.append(text)
        return "spoken.wav"


def test_with_speech_the_consent_is_asked_aloud_and_the_answer_recorded(inbox, clone):
    """Through `VoiceApprovalLoop.run_once`, which reads the request once --
    the one read a `qmcp human voice <id>` would make -- and submits.
    Mutation: skip the loop -- red, the request stays pending and expires."""
    instruction_id = inbox.record()
    queue = _Queue({f"instruction-{instruction_id}": None})  # nobody else answers
    tts = _TTS()

    done, runtime = _act(inbox, instruction_id, queue, cwd=clone, stt=_STT("yes, go ahead"), tts=tts)

    assert done.status == "done" and done.answer == "approve"
    assert tts.spoken[0].startswith("Act on the instruction: Deploy")
    assert tts.spoken[0].endswith("Say approve or hold.")
    assert tts.spoken[-1] == "Recorded: approve"
    assert queue.reads == 2  # once by the loop before asking, once by the act afterwards


def test_an_unclear_spoken_answer_leaves_the_request_for_anyone(inbox, clone):
    """Nothing is guessed, and the wait goes on; here nobody else answers and
    the consent expires. Mutation: treat `UnclearResponse` as hold -- red."""
    instruction_id = inbox.record()
    queue = _Queue({f"instruction-{instruction_id}": None})

    done, runtime = _act(inbox, instruction_id, queue, cwd=clone, stt=_STT("banana"), tts=_TTS())

    assert done.status == "unanswered" and runtime.calls == []


# --- the configured inbox -----------------------------------------------------------------


def test_the_configured_database_is_the_inbox_the_command_acts_on(tmp_path, monkeypatch):
    """`instructions act` and the server's conversation read the database the
    settings name, which is an async driver's URL. Mutation: build the engine
    from the URL as it is -- red, a synchronous engine does not open it."""
    from types import SimpleNamespace

    database = tmp_path / "inbox.db"
    rows_at(database)
    monkeypatch.setattr("qmcp.config.get_settings", lambda: SimpleNamespace(
        database_url=f"sqlite+aiosqlite:///{database.as_posix()}"))

    rows = act_module.configured_rows()
    with rows() as session:
        session.add(Instruction(id="from-settings", text="x", project=PROJECT,
                                source=InstructionSource.TYPED))
        session.commit()

    with rows_at(database)() as session:
        assert session.get(Instruction, "from-settings") is not None


def test_a_database_url_naming_no_file_is_refused(monkeypatch):
    """Mutation: fall back to an in-memory engine -- red; an act would record
    into a database nobody reads."""
    from types import SimpleNamespace

    monkeypatch.setattr("qmcp.config.get_settings", lambda: SimpleNamespace(
        database_url="sqlite+aiosqlite:///:memory:"))

    with pytest.raises(RuntimeError, match="names no file"):
        act_module.configured_rows()
