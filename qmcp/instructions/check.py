"""The spoken-instruction check: asked, confirmed, resolved, recorded, read back.

`qmcp cookbook instruct` runs it, offline and only offline. A qmcp server is
started on an ephemeral port over a database made for the run, vox's
deterministic engine stands in for the speech engine, and each scripted dialog
travels the real path: the prompt synthesized to a file, every take returned
over the engine contract, the row recorded over HTTP and read back. It proves
the wiring -- the long take carries the pause the stack exists for, the
confirmation does not, the project is read and asked back, and the row holds
what the script says -- and makes no claim about transcription, which only a
person at a microphone tests.

Nothing here writes into the configured inbox. The server, the engine and the
database are all made for the run and gone after it.
"""

from __future__ import annotations

import contextlib
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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

    reasked: str | None = None
    """How a turn after the first prompt must begin."""


# One case per way a spoken instruction can end. One project named and
# confirmed; none named, so the project is asked for and the spoken one
# recorded; several named, so a closed choice over them is asked by name; and
# `again`, which takes the instruction a second time before recording.
OFFLINE_CASES = (
    Case(("Deploy qmcp to the pi.", "record"),
         "Deploy qmcp to the pi.", NAMED),
    Case(("Rotate the logs.", "yes", OTHER),
         "Rotate the logs.", OTHER, reasked=WHICH_PROJECT),
    # The project chosen is the one the roster lists second, so a dialog that
    # took the first candidate for anything it could not match is caught here.
    Case((f"Move the vectors from {OTHER} into {NAMED}.", "record", NAMED),
         f"Move the vectors from {OTHER} into {NAMED}.", NAMED,
         reasked=f"{WHICH_PROJECT} Say {OTHER} or {NAMED}."),
    Case(("Deploy qmcp.", "again", "Deploy qmcp to the pi.", "record"),
         "Deploy qmcp to the pi.", NAMED, reasked=PROMPT),
)


class _Scripted:
    """vox's client over the real engine, with the engine told what to hear next.

    The deterministic engine answers one transcript; a dialog takes several.
    Setting the engine's script before each take keeps every take on the wire
    -- the request, the pause parameter, the WAV, the JSON -- rather than
    stubbing the client.
    """

    def __init__(self, stt, state, script: tuple[str, ...]):
        self.stt = stt
        self.state = state
        self.script = list(script)
        self.takes = 0

    def listen(self, duration: float = 5.0, *, pause_ms: int | None = None):
        self.state.heard = self.script[min(self.takes, len(self.script) - 1)]
        self.takes += 1
        return self.stt.listen(duration=duration, pause_ms=pause_ms)

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
