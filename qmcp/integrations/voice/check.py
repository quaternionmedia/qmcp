"""The voice HITL check: a request queued, answered by voice, and read back.

`qmcp cookbook voice` runs it, in one of two forms that answer different
questions:

- **Offline** (the default) needs no engine, no model, no microphone and no
  speakers. A qmcp server is started on an ephemeral port over a database made
  for the run, vox's deterministic engine stands in for the speech engine, and
  each scripted answer travels the real path: a request created over HTTP, the
  prompt synthesized to a file, the answer "heard" over the engine contract,
  the response submitted over HTTP and read back. It proves the wiring.
- **Live** runs the same path against the configured qmcp server and a real
  engine, and speaks. It proves that a person at this machine is heard. It
  asks one short-lived question and is only ever run by that person.

Neither form writes into the configured queue except the live form's one
request, which is named `voice-check-<time>` and expires in five minutes.
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from qmcp.integrations.voice.adapter import UnclearResponse, VoiceApprovalLoop

OPTIONS = ("approve", "hold")
PROMPT = "Voice check."
GRAMMAR = "Approve or hold?"
OPEN_PROMPT = "Voice check. What should the branch be called?"


@dataclass(frozen=True)
class Case:
    """One scripted exchange, and what the queue should hold afterwards."""

    heard: str
    recorded: str | None
    """The answer recorded, or None when nothing may be submitted."""
    reasked: str | None = None
    """How a later turn must begin: the re-ask for an answer that is not
    usable, or the read-back of an open question's transcript."""
    then: tuple[str, ...] = ()
    """What is heard on each listen after the first, for a dialog with turns."""
    prompt: str = PROMPT
    options: tuple[str, ...] | None = OPTIONS
    """None asks the request as an open question."""

    @property
    def asked(self) -> str:
        """What the prompt must be spoken as: the options are the grammar, when there are any."""
        return f"{self.prompt} {GRAMMAR}" if self.options else self.prompt


# One case per way a spoken answer can end. A yes read onto the options, an
# option named, something heard that matches nothing, nothing heard, and an
# open question whose transcript is read back and recorded on "agree".
OFFLINE_CASES = (
    Case("Yes, go ahead.", "approve"),
    Case("Hold.", "hold"),
    Case("banana", None, reasked="Heard banana."),
    Case("", None, reasked="Didn't catch that."),
    Case("release candidate", "release candidate", then=("agree",),
         reasked="I heard: release candidate. Agree or again?",
         prompt=OPEN_PROMPT, options=None),
)


class _Recorded:
    """A synthesizer that also keeps what it was asked to say."""

    def __init__(self, tts):
        self.tts = tts
        self.spoken: list[str] = []

    def speak(self, text: str, out_path: str | None = None) -> str:
        self.spoken.append(text)
        return self.tts.speak(text, out_path)


class _Scripted:
    """An STT client whose engine hears the next scripted line on each listen.

    The deterministic engine answers every listen with its state's `heard`,
    and a dialog with turns needs a different transcript per listen. The state
    is this process's object and the engine reads it on each request, so
    advancing it between listens scripts the speaker without touching the wire:
    each transcript still travels the engine contract over HTTP.
    """

    def __init__(self, stt, state, then: tuple[str, ...]):
        self.stt = stt
        self.state = state
        self.then = list(then)

    def listen(self, duration: float = 5.0) -> tuple[str, str]:
        heard = self.stt.listen(duration=duration)
        if self.then:
            self.state.heard = self.then.pop(0)
        return heard

    def announce(self, state: str, text: str = "", reason: str | None = None) -> bool:
        return self.stt.announce(state, text, reason=reason)


@contextlib.contextmanager
def throwaway_server(database: Path) -> Iterator[str]:
    """A qmcp server on an ephemeral port over its own database. Yields its URL.

    The configured database is somebody's queue, and a check must not write
    into it. The settings cache and the database engine are process-wide, so
    both are swapped for the run and put back afterwards.
    """
    import uvicorn

    import qmcp.config
    import qmcp.db.engine as db_engine
    from qmcp.server import create_app

    saved_url = os.environ.get("QMCP_DATABASE_URL")
    saved_engine = db_engine._engine
    os.environ["QMCP_DATABASE_URL"] = f"sqlite+aiosqlite:///{database.as_posix()}"
    qmcp.config.get_settings.cache_clear()
    db_engine._engine = None

    # The server logs every request at INFO through `basicConfig`, which does
    # nothing once the root logger has a handler; a check's output is its
    # verdicts. A failing request still raises in the client.
    root = logging.getLogger()
    saved_level, quiet = root.level, logging.NullHandler()
    root.addHandler(quiet)
    root.setLevel(logging.WARNING)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(), log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 20
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("the throwaway qmcp server did not start")
            time.sleep(0.02)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        if saved_url is None:
            os.environ.pop("QMCP_DATABASE_URL", None)
        else:
            os.environ["QMCP_DATABASE_URL"] = saved_url
        qmcp.config.get_settings.cache_clear()
        db_engine._engine = saved_engine
        root.removeHandler(quiet)
        root.setLevel(saved_level)


def _outcome(client, request_id: str) -> tuple[str | None, str | None]:
    """(the answer recorded, who recorded it), or (None, None)."""
    _, response = client.get_human_request(request_id)
    if response is None:
        return None, None
    return response.response, response.responded_by


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

    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        root = Path(tmp)
        qmcp_url = stack.enter_context(throwaway_server(root / "queue.db"))
        client = stack.enter_context(MCPClient(base_url=qmcp_url))
        echo(f"qmcp:    {qmcp_url} (a database made for this run)")
        echo("engine:  vox's deterministic engine on an ephemeral port, joe contract")

        passed = 0
        for number, case in enumerate(cases, 1):
            request_id = f"voice-check-{number}"
            # The type is the queue's record of what was asked; the loop reads
            # only `options`, so the verdict below does not depend on it.
            client.create_human_request(
                request_id=request_id,
                request_type="approval" if case.options else "input",
                prompt=case.prompt,
                options=list(case.options) if case.options else None,
                timeout_seconds=300,
            )
            state = EngineState(audio_dirs=[root], microphone="deterministic engine",
                                heard=case.heard, contract=JOE)
            tts = _Recorded(RecordingTTS(out_dir=str(root / "spoken")))
            with serve_engine(state) as (engine_url, _), HttpSTT(engine_url, contract=JOE) as stt:
                loop = VoiceApprovalLoop(stt=_Scripted(stt, state, case.then), tts=tts,
                                         client=client, max_retries=1, listen_duration=1.0)
                with contextlib.suppress(UnclearResponse):
                    loop.run_once(request_id)

            recorded, by = _outcome(client, request_id)
            problems = []
            if not tts.spoken or tts.spoken[0] != case.asked:
                problems.append(f"prompt was {tts.spoken[:1]!r}")
            if recorded != case.recorded:
                problems.append(f"recorded {recorded!r}, expected {case.recorded!r}")
            if recorded is not None and by != "vox":
                problems.append(f"recorded by {by!r}, expected 'vox'")
            if case.reasked and not any(s.startswith(case.reasked) for s in tts.spoken[1:]):
                problems.append(f"no turn beginning {case.reasked!r}")

            heard = '"' + '", "'.join((case.heard, *case.then)) + '"'
            if recorded is not None:
                result = f'recorded "{recorded}" by {by}'
            else:
                result = "nothing recorded" + (f'; re-asked "{case.reasked}"' if case.reasked else "")
            if problems:
                echo(f"  [FAIL] heard {heard:<18} {'; '.join(problems)}")
            else:
                passed += 1
                echo(f"  [ok]   heard {heard:<18} {result}")

        echo(f"{passed} of {len(cases)} ended as scripted.")
        return passed == len(cases)


def run_live(client, stt, tts, echo: Callable[[str], None] = print,
             max_retries: int = 2, listen_duration: float = 5.0) -> bool:
    """One short-lived question, spoken, against the configured server and a real engine.

    Returns True when an option was recorded. An answer that never parsed
    records nothing, and the request expires on its own.
    """
    request_id = f"voice-check-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    client.create_human_request(
        request_id=request_id, request_type="approval", prompt=PROMPT,
        options=list(OPTIONS), timeout_seconds=300,
    )
    echo(f"queued:  {request_id} (expires in five minutes)")
    echo(f'asking:  "{PROMPT} {GRAMMAR}"')
    loop = VoiceApprovalLoop(stt=stt, tts=tts, client=client,
                             max_retries=max_retries, listen_duration=listen_duration)
    try:
        response = loop.run_once(request_id)
    except UnclearResponse as exc:
        echo(f"nothing recorded: {exc}")
        return False
    echo(f'recorded "{response.response}" by {response.responded_by or "?"}')
    return True
