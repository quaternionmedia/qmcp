"""`qmcp.instructions.act`: the clone, the budget, the consent, and what runs when.

The human queue is stood in for by an in-memory object with the client's
surface, so a test can decide how a consent ends -- approved, held, expired --
and count what was read while it waited. The inbox is a database made for the
test; the archive is a stand-in source carrying what `qmcp.threads.claudecode`
puts in `context`. The runtime is the scripted one, whose `calls` say what ran.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qmcp.client import HumanRequestConflictError
from qmcp.client.mcp_client import HumanRequest, HumanResponse
from qmcp.db.models import Instruction, InstructionSource, InstructionStatus
from qmcp.instructions import act as act_module
from qmcp.instructions.act import (
    RULE_ARCHIVE,
    RULE_CWD,
    RULE_NAMED_DIR,
    Clone,
    NoSuchInstruction,
    act,
    clone_for,
    rows_at,
)
from qmcp.integrations.agents.scripted import ScriptedRuntime
from qmcp.spend import Budget
from qmcp.threads.base import Thread, Turn

PROJECT = "qmcp"


class _Queue:
    """The human queue, in memory, with the client's surface.

    `reads` counts calls to the single-request route, which is the one that
    expires a request as a side effect; `script` says how each request ends
    once the act is waiting on it.
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
        for request_id, answer in self.script.items():
            if request_id in self.requests and self._pending(request_id):
                if answer is None:
                    self.expired.add(request_id)
                else:
                    self.responses[request_id] = answer


class _Source:
    """An archive source: threads, and what each knew about its checkout."""

    def __init__(self, *entries: tuple[Thread, dict]):
        self.threads = [thread for thread, _ in entries]
        self.context = {thread.id: context for thread, context in entries}

    def fetch(self, ids, budget):
        return list(self.threads)


def _thread(identifier: str, *texts: str, title: str | None = None) -> Thread:
    return Thread(id=identifier, title=title,
                  turns=tuple(Turn(id=f"{identifier}-{n}", role="user", text=t)
                              for n, t in enumerate(texts)))


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


def _archive(clone: Path, session="s-1", last_at="2026-10-03T10:00:00Z") -> _Source:
    return _Source((_thread("t-1", "working here"),
                    {"cwd": str(clone), "session": session, "last_at": last_at}))


def _act(inbox, instruction_id, queue, runtime=None, budget=1, **kw):
    runtime = runtime or ScriptedRuntime(text="pinned", session_ref="s-2")
    kw.setdefault("sources", [])
    return act(instruction_id, runtime, Budget(authorised=budget), client=queue, rows=inbox,
               sleep=queue.settle, poll_interval=0.0, **kw), runtime


# --- the clone ------------------------------------------------------------------------


def test_the_clone_is_the_newest_threads_checkout_that_exists(tmp_path):
    """Mutation: sort `last_at` ascending -- red on `older`; drop the `exists`
    check -- red on `gone`."""
    newer, older = tmp_path / PROJECT, tmp_path / "elsewhere" / PROJECT
    newer.mkdir(), older.mkdir(parents=True)
    gone = tmp_path / "deleted" / PROJECT
    source = _Source(
        (_thread("old", f"about {PROJECT}", f"{PROJECT} again"),
         {"cwd": str(older), "session": "s-old", "last_at": "2026-09-01T00:00:00Z"}),
        (_thread("new", "nothing named"),
         {"cwd": str(newer), "session": "s-new", "last_at": "2026-10-01T00:00:00Z"}),
        (_thread("gone", "nothing named"),
         {"cwd": str(gone), "session": "s-gone", "last_at": "2026-10-02T00:00:00Z"}),
        (_thread("other", f"{PROJECT} {PROJECT}"),
         {"cwd": str(tmp_path / "vox"), "session": "s-x", "last_at": "2026-10-03T00:00:00Z"}),
    )

    found = clone_for(PROJECT, [source])

    assert found == Clone(cwd=newer, session_ref="s-new", rule=RULE_NAMED_DIR,
                          thread_id="new", last_at="2026-10-01T00:00:00Z")


def test_a_thread_about_the_project_names_its_checkout_whatever_it_is_called(tmp_path):
    """`consolidate.about`'s rule, when the directory is not named for the
    project. Mutation: drop the `about` branch -- red."""
    checkout = tmp_path / "work"
    checkout.mkdir()
    source = _Source((_thread("t", f"{PROJECT} first", f"{PROJECT} second"),
                      {"cwd": str(checkout), "session": "s", "last_at": "x"}))

    found = clone_for(PROJECT, [source])

    assert found is not None and found.rule == RULE_ARCHIVE and found.cwd == checkout
    assert clone_for(PROJECT, [_Source((_thread("t", "one passing qmcp"),
                                        {"cwd": str(checkout)}))]) is None
    assert clone_for(None, [source]) is None


def test_an_owner_prefixed_project_matches_its_repository_name(tmp_path):
    checkout = tmp_path / PROJECT
    checkout.mkdir()
    assert clone_for(f"quaternionmedia/{PROJECT}", [_archive(checkout)]).cwd == checkout


def test_the_archives_clone_is_used_and_its_session_resumed(inbox, clone):
    """Mutation: pass `resume=None` to the runtime -- red."""
    queue = _Queue({f"instruction-{'{id}'}": "approve"})
    instruction_id = inbox.record()
    queue.script = {f"instruction-{instruction_id}": "approve"}

    done, runtime = _act(inbox, instruction_id, queue, sources=[_archive(clone, session="s-7")])

    assert done.cwd == str(clone) and done.status == "done"
    assert runtime.calls == [{"instruction": f"Deploy {PROJECT} to the pi.",
                              "cwd": str(clone), "resume": "s-7"}]
    assert inbox.read(instruction_id).detail["clone"]["rule"] == RULE_NAMED_DIR


def test_cwd_serves_when_the_archives_checkout_is_gone(inbox, clone, tmp_path):
    instruction_id = inbox.record()
    queue = _Queue({f"instruction-{instruction_id}": "approve"})
    gone = tmp_path / "gone" / PROJECT

    done, runtime = _act(inbox, instruction_id, queue, cwd=clone, sources=[_archive(gone)])

    assert done.cwd == str(clone)
    assert runtime.calls[0]["resume"] is None
    assert inbox.read(instruction_id).detail["clone"]["rule"] == RULE_CWD


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
    assert f"no checkout for {PROJECT!r}" in done.why
    assert "unresolved" in other.why
    assert "not a directory" in missing.why
    assert done.status == "recorded" and other.status == "unresolved"
    assert runtime.calls == [] and queue.requests == {}
    assert inbox.read(instruction_id).status == InstructionStatus.RECORDED


def test_an_unknown_instruction_is_refused_outright(inbox):
    with pytest.raises(NoSuchInstruction, match="nobody"):
        act("nobody", ScriptedRuntime(), Budget(authorised=1), client=_Queue(), rows=inbox, sources=[])


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
               rows=inbox, cwd=clone, sources=[], sleep=settle, poll_interval=0.0)

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
    assert row.session_ref == "s-2" and row.acted_at is not None
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

        def run(self, instruction, cwd, resume=None, on_event=None):
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
