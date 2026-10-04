"""The spoken-instruction check: asked, recorded, consented, run, and said back.

`qmcp cookbook instruct` runs it, offline and only offline. A qmcp server is
started on an ephemeral port over a database made for the run, vox's
deterministic engine stands in for the speech engine, and each scripted dialog
travels the real path: the prompt synthesized to a file, every take returned
over the engine contract, the row recorded over HTTP and read back.

**TWO PARTS, AND THE SECOND IS THE LOOP.** The inbox cases prove the
recording -- the long take carries the pause the stack exists for, the
confirmation does not, the project is read and asked back, and the row holds
what the script says. The loop cases then go the whole way in one
conversation: the instruction spoken and recorded, consent asked aloud on the
human queue and answered, the scripted runtime run in a directory standing in
for the clone, and the row's summary said back. Each loop case prints as the
conversation it was, so a person can follow what was said and heard. A held
consent runs nothing and says so.

It makes no claim about transcription, which only a person at a microphone
tests, nor about any real agent: the runtime is `scripted` and spends nothing.
Nothing here writes into the configured inbox. The server, the engine and the
database are all made for the run and gone after it.
"""

from __future__ import annotations

import contextlib
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from qmcp.instructions import RULE_ONE, RULE_STATED
from qmcp.instructions.dialog import CONFIRM, PAUSE_MS, PROMPT, WHICH_PROJECT, InstructionDialog
from qmcp.integrations.voice.adapter import UnclearResponse, say_options
from qmcp.integrations.voice.check import _Recorded, throwaway_server

# Two names the organisation's own roster carries. The check reads the real
# roster through the real server, so these are what resolution is tried on.
NAMED = "qmcp"
OTHER = "dossier"


@dataclass(frozen=True)
class Case:
    """One scripted dialog, and what the inbox should hold afterwards."""

    heard: tuple[str, ...]
    """What each take returns, in order: the instruction, the confirmation,
    and the project where one is asked for."""

    text: str
    """The instruction recorded."""

    project: str | None
    """The project recorded, or None for unresolved."""

    rule: str
    """The rule the row must say settled the project: the server's match
    when the text named it, `stated` when the person chose or spoke it."""

    reasked: str | None = None
    """How a turn after the first prompt must begin."""


# One case per way a spoken instruction can end. One project named and
# confirmed; none named, so the project is asked for and the spoken one
# recorded; several named, so a closed choice over them is asked by name; and
# `again`, which takes the instruction a second time before recording.
OFFLINE_CASES = (
    Case(("Deploy qmcp to the pi.", "record"),
         "Deploy qmcp to the pi.", NAMED, RULE_ONE),
    Case(("Rotate the logs.", "yes", OTHER),
         "Rotate the logs.", OTHER, RULE_STATED, reasked=WHICH_PROJECT),
    # The project chosen is the one the roster lists second, so a dialog that
    # took the first candidate for anything it could not match is caught here.
    Case((f"Move the vectors from {OTHER} into {NAMED}.", "record", NAMED),
         f"Move the vectors from {OTHER} into {NAMED}.", NAMED, RULE_STATED,
         reasked=f"{WHICH_PROJECT} Say {OTHER} or {NAMED}."),
    Case(("Deploy qmcp.", "again", "Deploy qmcp to the pi.", "record"),
         "Deploy qmcp to the pi.", NAMED, RULE_ONE, reasked=PROMPT),
)


class _Scripted:
    """vox's client over the real engine, with the engine told what to hear next.

    The deterministic engine answers one transcript; a dialog takes several.
    Setting the engine's script before each take keeps every take on the wire
    -- the request, the pause parameter, the WAV, the JSON -- rather than
    stubbing the client.
    """

    def __init__(self, stt, state, script: tuple[str, ...], log: list | None = None):
        self.stt = stt
        self.state = state
        self.script = list(script)
        self.takes = 0
        self.log = log

    def listen(self, duration: float = 5.0, *, pause_ms: int | None = None):
        self.state.heard = self.script[min(self.takes, len(self.script) - 1)]
        self.takes += 1
        text, path = self.stt.listen(duration=duration, pause_ms=pause_ms)
        if self.log is not None:
            self.log.append(("heard", text))
        return text, path

    def announce(self, state: str, text: str = "", reason: str | None = None):
        return self.stt.announce(state, text, reason=reason)


def run_offline(echo: Callable[[str], None] = print, cases=OFFLINE_CASES) -> bool:
    """Every case through the real path, against vox's deterministic engine.

    Returns True when every case ended as its script says.
    """
    from vox import HttpSTT
    from vox.adapters.joe import JOE
    from vox.engine import EngineState
    from vox.engine import serve as serve_engine
    from vox.tts import RecordingTTS

    from qmcp.client import MCPClient
    from qmcp.instructions import roster_names

    names = roster_names()
    readback = say_options(CONFIRM)
    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        root = Path(tmp)
        qmcp_url = stack.enter_context(throwaway_server(root / "inbox.db"))
        client = stack.enter_context(MCPClient(base_url=qmcp_url))
        echo(f"qmcp:    {qmcp_url} (a database made for this run)")
        echo("engine:  vox's deterministic engine on an ephemeral port, joe contract")
        if NAMED not in names or OTHER not in names:
            echo(f"  [FAIL] the roster names neither {NAMED!r} nor {OTHER!r};"
                 " nothing here can resolve")
            return False

        passed = 0
        for case in cases:
            state = EngineState(audio_dirs=[root], microphone="deterministic engine",
                                contract=JOE)
            tts = _Recorded(RecordingTTS(out_dir=str(root / "spoken")))
            row = None
            with serve_engine(state) as (engine_url, _), HttpSTT(engine_url, contract=JOE) as stt:
                dialog = InstructionDialog(stt=_Scripted(stt, state, case.heard), tts=tts,
                                           client=client, names=names, max_retries=1,
                                           listen_duration=2.0, answer_duration=1.0)
                with contextlib.suppress(UnclearResponse):
                    row = dialog.run_once()

            problems = []
            if not tts.spoken or tts.spoken[0] != PROMPT:
                problems.append(f"prompt was {tts.spoken[:1]!r}")
            if not any(s.startswith("I heard: ") and s.endswith(readback) for s in tts.spoken):
                problems.append("the instruction was never read back")
            if case.reasked and not any(s.startswith(case.reasked) for s in tts.spoken[1:]):
                problems.append(f"no turn beginning {case.reasked!r}")
            # The whole reason the stack exists: the instruction's take asks
            # the engine for the long pause, and the word that confirms it does not.
            if not state.pauses or state.pauses[0] != PAUSE_MS:
                problems.append(f"the instruction take carried pause {state.pauses[:1]!r},"
                                f" expected {PAUSE_MS}")
            if len(state.pauses) > 1 and state.pauses[1] is not None:
                problems.append(f"the confirmation carried pause {state.pauses[1]!r}")
            recorded = client.get_instruction(row["id"]) if row else None
            if recorded is None:
                problems.append("nothing recorded")
            else:
                if recorded["text"] != case.text:
                    problems.append(f"recorded text {recorded['text']!r}, expected {case.text!r}")
                if recorded["project"] != case.project:
                    problems.append(f"recorded project {recorded['project']!r},"
                                    f" expected {case.project!r}")
                # The row says how the project was settled: by the server's
                # match where the text named it, as stated where the person did.
                if recorded["detail"].get("rule") != case.rule:
                    problems.append(f"rule {recorded['detail'].get('rule')!r},"
                                    f" expected {case.rule!r}")
                if recorded["source"] != "voice":
                    problems.append(f"source {recorded['source']!r}, expected 'voice'")
                if recorded["detail"].get("heard") != list(dialog.heard):
                    problems.append("the row does not carry what was heard")

            script = " / ".join(f'"{h}"' for h in case.heard)
            if problems:
                echo(f"  [FAIL] heard {script}: {'; '.join(problems)}")
            else:
                passed += 1
                echo(f"  [ok]   heard {script}: recorded for {recorded['project'] or 'nobody'}")

        echo(f"{passed} of {len(cases)} ended as scripted.")
        return passed == len(cases)


# --- the loop ---------------------------------------------------------------------


@dataclass(frozen=True)
class Loop:
    """One spoken instruction taken the whole way, and how it must end."""

    heard: tuple[str, ...]
    """What each take returns: the instruction, its confirmation, and the
    answer to the consent."""

    status: str
    """The row's status afterwards."""

    ran: bool
    """Whether the runtime was reached."""

    says: str
    """How the summary said back must begin."""


# What the scripted runtime reports. Two sentences, so the summary is seen to
# say the first and point at the rest.
OUTCOME = ("Added a health route that answers with the version; the suite passes. "
           "Two files changed.")

# The two ways a consent ends aloud: approve runs the instruction once and its
# outcome is said back; hold runs nothing and says so.
LOOP_CASES = (
    Loop((f"Add a health check to {NAMED}.", "record", "approve"), "done", True,
         f"Done in {NAMED}. Added a health route"),
    Loop((f"Rotate the {NAMED} logs.", "record", "hold"), "refused", False,
         "Held. Nothing ran"),
)


class _Said:
    """A synthesizer that also writes what it says into the conversation log."""

    def __init__(self, tts, log: list):
        self.tts = tts
        self.log = log
        self.spoken: list[str] = []

    def speak(self, text: str, out_path: str | None = None) -> str:
        self.spoken.append(text)
        self.log.append(("said", text))
        return self.tts.speak(text, out_path)


def run_loop(echo: Callable[[str], None] = print, cases=LOOP_CASES) -> bool:
    """Every loop case through recording, consent, run and summary, in one conversation.

    Returns True when every case ended as its script says.
    """
    from vox import HttpSTT
    from vox.adapters.joe import JOE
    from vox.engine import EngineState
    from vox.engine import serve as serve_engine
    from vox.tts import RecordingTTS

    from qmcp.client import MCPClient
    from qmcp.instructions import roster_names
    from qmcp.instructions.act import act
    from qmcp.instructions.spoken import say, starting, summarise
    from qmcp.integrations.agents.scripted import ScriptedRuntime
    from qmcp.spend import Budget

    names = roster_names()
    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        root = Path(tmp)
        database = root / "inbox.db"
        qmcp_url = stack.enter_context(throwaway_server(database))
        client = stack.enter_context(MCPClient(base_url=qmcp_url))
        # The inbox the act writes is the file the server serves, so the row
        # read back over HTTP is the row the act recorded. Its engine is
        # disposed before the directory goes: on Windows an open SQLite file
        # cannot be deleted.
        from sqlmodel import Session, create_engine

        engine = create_engine(f"sqlite:///{database.as_posix()}")
        stack.callback(engine.dispose)
        rows = lambda: Session(engine)  # noqa: E731 -- the shape `act` takes
        clone = root / NAMED
        clone.mkdir()
        echo(f"qmcp:    {qmcp_url} (a database made for this run)")
        echo(f"clone:   a directory named {NAMED}, made for this run;"
             " runtime: scripted, which spends nothing")
        if NAMED not in names:
            echo(f"  [FAIL] the roster does not name {NAMED!r}; nothing here can resolve")
            return False

        passed = 0
        for case in cases:
            log: list[tuple[str, str]] = []
            state = EngineState(audio_dirs=[root], microphone="deterministic engine",
                                contract=JOE)
            tts = _Said(RecordingTTS(out_dir=str(root / "spoken")), log)
            runtime = ScriptedRuntime(text=OUTCOME)
            row = recorded = summary = None
            with serve_engine(state) as (engine_url, _), HttpSTT(engine_url, contract=JOE) as stt:
                scripted = _Scripted(stt, state, case.heard, log=log)
                dialog = InstructionDialog(stt=scripted, tts=tts, client=client, names=names,
                                           max_retries=1, listen_duration=2.0,
                                           answer_duration=1.0)
                with contextlib.suppress(UnclearResponse):
                    row = dialog.run_once()
                if row is not None:
                    log.append(("recorded", f"for {row['project'] or 'nobody'}"))

                    def event(kind: str, text: str) -> None:
                        if kind == "acting":
                            log.append(("ran", f"{runtime.name}, in the {NAMED} clone"))
                            say(starting(row["project"]), tts, scripted)

                    done = act(row["id"], runtime, Budget(authorised=1), client=client,
                               rows=rows, cwd=clone, stt=scripted, tts=tts,
                               on_event=event, poll_interval=0.05, consent_seconds=60)
                    recorded = client.get_instruction(row["id"])
                    summary = summarise(recorded, why=done.why)
                    say(summary, tts, scripted)

            problems = []
            if recorded is None:
                problems.append("nothing recorded")
            else:
                if recorded["status"] != case.status:
                    problems.append(f"status {recorded['status']!r}, expected {case.status!r}")
                if not any(s.startswith("Act on the instruction: ") and case.heard[0] in s
                           for s in tts.spoken):
                    problems.append("the consent was never asked aloud with the instruction")
                ran = len(runtime.calls)
                if ran != int(case.ran):
                    problems.append(f"the runtime ran {ran} time(s), expected {int(case.ran)}")
                if not summary or not summary.startswith(case.says):
                    problems.append(f"said {summary!r}, expected it to begin {case.says!r}")
                if tts.spoken[-1:] != [summary]:
                    problems.append("the summary was not the last thing said")
                # The panel is told the summary as it is said, then that the
                # turn is over -- never `recorded`, which it shows as an answer.
                tail = [(a.get("state"), a.get("text")) for a in state.announced[-2:]]
                if tail != [("speaking", summary), ("idle", summary)]:
                    problems.append(f"the panel was last told {tail!r}")

            answer = case.heard[-1]
            echo(f"  loop: consent answered {answer!r}")
            for kind, text in log:
                echo(f"    {kind:<9} {text}")
            if problems:
                echo(f"  [FAIL] {answer}: {'; '.join(problems)}")
            else:
                passed += 1
                times = "once" if case.ran else "nothing"
                echo(f"  [ok]   {answer}: {recorded['status']}, ran {times}, and said so")

        echo(f"{passed} of {len(cases)} loops ended as scripted.")
        return passed == len(cases)


# --- the continuity, on a real runtime ------------------------------------------------


FIRST_ASK = "Which file in {project} says what {project} is? Name the file and quote what it says."
SECOND_ASK = "In one sentence, what did that file say {project} is for?"


def run_continuity(echo: Callable[[str], None] = print, runtime=None,
                   clone: Path | None = None, project: str | None = None) -> bool:
    """Two spoken instructions in one project, on a real runtime: the second told what the first found.

    The first names the clone; the second names none and refers back ("that
    file"), so it can be carried out only if the clone is remembered and the
    first one's outcome reaches the runtime -- both read from this server's
    record. Speech goes through vox's deterministic engine, so each consent is
    answered `approve` by a script standing in for the person; the runtime is
    whatever is passed, and with `local` it is the model on this machine
    reading `clone`. Returns True when both instructions ran, the second
    carried the first, and each summary was the last thing said.
    """
    from vox import HttpSTT
    from vox.adapters.joe import JOE
    from vox.engine import EngineState
    from vox.engine import serve as serve_engine
    from vox.tts import RecordingTTS

    from qmcp.client import MCPClient
    from qmcp.instructions import roster_names
    from qmcp.instructions.act import RULE_CWD, RULE_RECORD, act
    from qmcp.instructions.spoken import say, summarise
    from qmcp.spend import Budget

    clone = Path(clone).resolve()
    project = project or clone.name
    names = roster_names()
    if project not in names:
        echo(f"  [FAIL] {project!r} is not on the roster, so no instruction can resolve to it;"
             " pass a clone named for a rostered project")
        return False
    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        root = Path(tmp)
        database = root / "inbox.db"
        qmcp_url = stack.enter_context(throwaway_server(database))
        client = stack.enter_context(MCPClient(base_url=qmcp_url))
        from sqlmodel import Session, create_engine

        engine = create_engine(f"sqlite:///{database.as_posix()}")
        stack.callback(engine.dispose)
        rows = lambda: Session(engine)  # noqa: E731 -- the shape `act` takes
        echo(f"qmcp:    {qmcp_url} (a database made for this run)")
        echo(f"runtime: {runtime.name}, reading {clone}; each consent is answered by a script")

        ok = True
        first_id = None
        for number, ask in enumerate((FIRST_ASK, SECOND_ASK)):
            text = ask.format(project=project)
            log: list[tuple[str, str]] = []
            state = EngineState(audio_dirs=[root], microphone="deterministic engine",
                                contract=JOE)
            tts = _Said(RecordingTTS(out_dir=str(root / "spoken")), log)
            done = recorded = summary = None
            with serve_engine(state) as (engine_url, _), HttpSTT(engine_url, contract=JOE) as stt:
                scripted = _Scripted(stt, state, (text, "record", "approve"), log=log)
                dialog = InstructionDialog(stt=scripted, tts=tts, client=client, names=names,
                                           max_retries=1, listen_duration=2.0,
                                           answer_duration=1.0)
                row = None
                with contextlib.suppress(UnclearResponse):
                    row = dialog.run_once()
                if row is not None:
                    log.append(("recorded", f"for {row['project'] or 'nobody'}"))

                    def event(kind: str, detail: str) -> None:
                        if kind == "output":
                            log.append(("read", detail))

                    done = act(row["id"], runtime, Budget(authorised=1), client=client,
                               rows=rows, cwd=clone if number == 0 else None, stt=scripted,
                               tts=tts, on_event=event, poll_interval=0.05,
                               consent_seconds=120)
                    if done.carried:
                        log.append(("carried", "instruction 1 and what it found, from qmcp's record"
                                    if done.carried == (first_id,) else ", ".join(done.carried)))
                    recorded = client.get_instruction(row["id"])
                    summary = summarise(recorded, why=done.why)
                    say(summary, tts, scripted)

            problems = []
            if done is None:
                problems.append("nothing recorded")
            else:
                if not done.ran:
                    problems.append(f"the runtime never ran: {done.why or done.status}")
                elif recorded["status"] != "done":
                    problems.append(f"the run ended {recorded['status']}: "
                                    f"{(recorded.get('outcome_text') or '').strip()[:200]}")
                rule = (recorded.get("detail") or {}).get("clone", {}).get("rule")
                if number == 0:
                    first_id = done.instruction_id
                    if rule != RULE_CWD:
                        problems.append(f"the clone was chosen by {rule!r}, not the one passed")
                else:
                    if done.carried != (first_id,):
                        problems.append(f"carried {done.carried!r}, expected the first instruction")
                    if rule != RULE_RECORD:
                        problems.append(f"the clone was chosen by {rule!r}, not remembered")
                if tts.spoken[-1:] != [summary]:
                    problems.append("the summary was not the last thing said")

            echo(f"  instruction {number + 1}: {text}")
            for kind, line in log:
                echo(f"    {kind:<9} {line}")
            if problems:
                ok = False
                echo(f"  [FAIL] {'; '.join(problems)}")
            else:
                echo(f"  [ok]   {'ran with the clone passed' if number == 0 else 'ran in the remembered clone, told what the first found'}")
        echo("The second instruction was carried out knowing what the first found."
             if ok else "Continuity was not shown.")
        return ok


# --- the conversation ------------------------------------------------------------------


WAITING_QUESTION = "Ship the build?"


def conversation_script(project: str) -> tuple[str, ...]:
    """Every take of one spoken session, in order: the waiting question answered,
    a silence, two instructions each recorded and approved, a yes and a no to
    "Anything else?", the words that end it, and silence."""
    return ("approve", "",
            FIRST_ASK.format(project=project), "record", "approve",
            "yes",
            SECOND_ASK.format(project=project), "record", "approve",
            "no",
            "stop listening",
            # Silence after the end: a session that missed its stop runs out on
            # its idle limit rather than hearing the last words for ever.
            "")


def run_conversation(echo: Callable[[str], None] = print, runtime=None,
                     clones: Path | None = None, project: str = NAMED) -> bool:
    """One whole spoken session, with nothing typed: what `qmcp serve --converse` holds.

    A throwaway server and vox's deterministic engine; every take is scripted,
    standing in for the person. An agent's question waits on the queue before
    the session starts. The session must ask it and record the answer, wait
    through a silence, take two instructions -- the first in the clone found
    beside the others, the second in the remembered one and told what the first
    found -- go back to waiting on "no", and end on "stop listening". With no
    runtime it is `scripted`, in a directory made for the run; with `local` it
    is the model reading a real clone among `clones`.
    """
    from vox import HttpSTT
    from vox.adapters.joe import JOE
    from vox.engine import EngineState
    from vox.engine import serve as serve_engine
    from vox.tts import RecordingTTS

    from qmcp.client import MCPClient
    from qmcp.instructions import roster_names
    from qmcp.instructions.act import RULE_RECORD, RULE_SIBLING
    from qmcp.instructions.converse import STOPPING, Conversation
    from qmcp.integrations.agents.scripted import ScriptedRuntime

    names = roster_names()
    if project not in names:
        echo(f"  [FAIL] {project!r} is not on the roster")
        return False
    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        root = Path(tmp)
        if runtime is None:
            runtime = ScriptedRuntime(text="Found it in README.md. It is the local backend.")
        if clones is None:
            clones = root / "clones"
            (clones / project).mkdir(parents=True)
        database = root / "inbox.db"
        qmcp_url = stack.enter_context(throwaway_server(database))
        client = stack.enter_context(MCPClient(base_url=qmcp_url))
        from sqlmodel import Session, create_engine

        engine = create_engine(f"sqlite:///{database.as_posix()}")
        stack.callback(engine.dispose)
        rows = lambda: Session(engine)  # noqa: E731 -- the shape `act` takes
        client.create_human_request(request_id="agent-question", request_type="approval",
                                    prompt=WAITING_QUESTION, options=["approve", "hold"],
                                    timeout_seconds=600)
        echo(f"qmcp:    {qmcp_url} (a database made for this run)")
        echo(f"runtime: {runtime.name}; clones looked for in {clones}; every take scripted")

        log: list[tuple[str, str]] = []
        state = EngineState(audio_dirs=[root], microphone="deterministic engine", contract=JOE)
        tts = _Said(RecordingTTS(out_dir=str(root / "spoken")), log)
        with serve_engine(state) as (engine_url, _), HttpSTT(engine_url, contract=JOE) as stt:
            scripted = _Scripted(stt, state, conversation_script(project), log=log)
            conversation = Conversation(scripted, tts, client, runtime, names, rows=rows,
                                        clones=clones, listen_duration=2.0,
                                        answer_duration=1.0, max_retries=1,
                                        poll_interval=0.05, idle_limit=3,
                                        echo=lambda line: log.append(("", line))
                                        if line.startswith(("recorded:", "read:", "why:",
                                                            "answered:", "turn failed"))
                                        else None)
            ended = conversation.run()

        problems = []
        _, response = client.get_human_request("agent-question")
        if ended.answered != ["agent-question"] or response is None \
                or response.response != "approve":
            problems.append("the waiting question was not asked and answered by voice")
        statuses = [turn.status for turn in ended.turns]
        if statuses != ["done", "done"]:
            problems.append(f"the instructions ended {statuses}, expected two done")
        else:
            first, second = (client.get_instruction(turn.instruction_id) for turn in ended.turns)
            if first["detail"]["clone"]["rule"] != RULE_SIBLING \
                    or Path(first["cwd"]).name != project:
                problems.append("the first instruction did not run in the clone found by name")
            if second["detail"]["clone"]["rule"] != RULE_RECORD:
                problems.append("the second instruction did not run in the remembered clone")
            if ended.turns[1].carried != (ended.turns[0].instruction_id,):
                problems.append("the second instruction was not told what the first found")
        if ended.reason != "told to stop" or tts.spoken[-1:] != [STOPPING]:
            problems.append(f"the session ended {ended.reason!r}, not on being told to stop")

        for kind, line in log:
            echo(f"    {kind:<6} {line}" if kind else f"    {'':<6} {line}")
        if problems:
            echo(f"  [FAIL] {'; '.join(problems)}")
            return False
        echo("  [ok]   one spoken session: a question answered, two instructions carried out,"
             " the second told what the first found, and an end on being told to stop")
        return True
