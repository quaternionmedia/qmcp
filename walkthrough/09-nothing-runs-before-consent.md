# 09 — Nothing runs before consent

Everything on this page runs. It is executed by the ordinary test command, so
an example that stops being true fails the build rather than sitting here
misleading somebody.

**The problem.** `08` left an instruction in the inbox and showed that
recording it ran nothing. Acting on it is the step after, and it is where the
money is: an agent in a clone makes as many calls as it needs. So the act is
shaped by `qmcp.governed` with the human gate in front of the runtime rather
than behind it -- the command states how many runs it may make, a consent
request goes on the human queue saying everything a stranger would need, and
the runtime is reached on `approve` and on nothing else. This page shows the
three ways an act ends without running, and the one way it runs, through the
`scripted` runtime, which answers as told and spends nothing.

## One process, a real queue and an inbox of its own

The queue is the real one: a qmcp server on an ephemeral port over a database
made for this page, reached through the ordinary client. The inbox is the same
file, so the row the act writes is the row the server serves. The archive is
empty here, so the clone is passed:

    >>> import tempfile
    >>> from pathlib import Path
    >>> from qmcp.client import MCPClient
    >>> from qmcp.db.models import Instruction, InstructionSource
    >>> from qmcp.instructions.act import act, rows_at
    >>> from qmcp.integrations.agents.scripted import ScriptedRuntime
    >>> from qmcp.integrations.voice.check import throwaway_server
    >>> from qmcp.spend import Budget, render

    >>> root = Path(tempfile.mkdtemp())
    >>> clone = root / "qmcp"
    >>> clone.mkdir()
    >>> server = throwaway_server(root / "queue.db")
    >>> client = MCPClient(base_url=server.__enter__())
    >>> rows = rows_at(root / "queue.db")
    >>> with rows() as session:
    ...     row = Instruction(id="pin-the-vectors", text="Pin rad godot to the vectors.",
    ...                       project="rad-godot", source=InstructionSource.VOICE)
    ...     session.add(row)
    ...     session.commit()
    >>> client.get_instruction("pin-the-vectors")["status"]
    'recorded'

The runtime keeps every call it is asked to make. Through this whole page it
is asked once:

    >>> runtime = ScriptedRuntime(text="pinned to 2c10fd1", session_ref="session-after")

## Zero is the default, and it asks nothing

A budget of zero runs is a real count. The act resolves the clone, declares
what would be asked, and stops; no request reaches the queue and the row is as
it was:

    >>> free = act("pin-the-vectors", runtime, Budget(), client=client, rows=rows,
    ...            sources=[], cwd=clone)
    >>> free.stages
    ('instruction', 'clone', 'budget')
    >>> free.status, free.request_id, runtime.calls
    ('recorded', None, [])
    >>> print(render(free.declared))
      Nothing was spent. This pass was issued against 0 calls.
      What the paid work would cost is unknown: an agent run makes as many calls as it needs; the runtime reports what it made afterwards
      A count nobody took is not a count of zero, so this does not
      say the work is free.
    >>> client.list_human_requests(status_filter="pending")
    []

## No clone is a refusal that says what to pass

The archive names no checkout for the project and nothing was passed, so the
act refuses before it asks, with the row unchanged:

    >>> nowhere = act("pin-the-vectors", runtime, Budget(authorised=1), client=client,
    ...               rows=rows, sources=[])
    >>> nowhere.status, nowhere.stages
    ('recorded', ('instruction', 'clone'))
    >>> nowhere.why
    "no checkout for 'rad-godot' in the thread archive; pass --cwd <path to the project's clone>."

## Held: the consent is asked, answered, and nothing runs

With one run authorised the consent goes on the queue. It is answered here by
voice, through the same loop `qmcp human voice` runs, with a synthesizer and a
transcriber stood in for so the page can show what is said. The person says
hold:

    >>> class Hears:
    ...     def __init__(self, text): self.text = text
    ...     def listen(self, duration=5.0, *, pause_ms=None): return self.text, "take.wav"
    >>> class Says:
    ...     spoken = []
    ...     def speak(self, text, out_path=None): self.spoken.append(text); return "out.wav"

    >>> held = act("pin-the-vectors", runtime, Budget(authorised=1), client=client,
    ...            rows=rows, sources=[], cwd=clone, stt=Hears("hold"), tts=Says(),
    ...            poll_interval=0.05)
    >>> Says.spoken[0] == (
    ...     "Act on the instruction: Pin rad godot to the vectors. "
    ...     f"Project rad-godot, clone {clone}, runtime scripted, budget 1 run(s). "
    ...     "Say approve or hold.")
    True
    >>> held.status, held.answer, held.stages
    ('refused', 'hold', ('instruction', 'clone', 'budget', 'ask', 'answer'))
    >>> runtime.calls
    []

The declaration is on the row, as it is on every path, and says one run was
authorised and none made:

    >>> recorded = client.get_instruction("pin-the-vectors")
    >>> recorded["status"], recorded["declared"]["authorised"], recorded["declared"]["made"]
    ('refused', 1, 0)
    >>> recorded["consent_request_id"]
    'instruction-pin-the-vectors'

The answer went through the queue, where it is a person's response like any
other:

    >>> _, response = client.get_human_request("instruction-pin-the-vectors")
    >>> response.response, response.responded_by
    ('hold', 'vox')

## Unanswered: the consent expires, and nothing runs

The wait watches the pending listing, which applies no expiry, until the
request leaves it, and reads the request itself once afterwards. A consent
lives ten minutes, and the server's floor is one, so rather than wait the page
moves the server's clock past the expiry in place of sleeping between
listings: the listing drops the request, the one read afterwards is what
marks it expired, and nothing runs.

    >>> from datetime import datetime, timedelta
    >>> import qmcp.server
    >>> class Later(datetime):
    ...     @classmethod
    ...     def now(cls, tz=None): return datetime.now(tz) + timedelta(minutes=11)
    >>> def nobody_answers(seconds): qmcp.server.datetime = Later
    >>> expired = act("pin-the-vectors", runtime, Budget(authorised=1), client=client,
    ...               rows=rows, sources=[], cwd=clone, sleep=nobody_answers, poll_interval=0.05)
    >>> qmcp.server.datetime = datetime
    >>> expired.status, expired.answer, runtime.calls
    ('unanswered', None, [])
    >>> client.get_human_request(expired.request_id)[0].status
    'expired'

## Approve: consented before it starts, then run in the clone

The fourth act is approved. The runtime is called once, in the clone, and the
outcome is recorded with the session it left for the next instruction:

    >>> Says.spoken.clear()
    >>> done = act("pin-the-vectors", runtime, Budget(authorised=1), client=client,
    ...            rows=rows, sources=[], cwd=clone, stt=Hears("yes, go ahead"), tts=Says(),
    ...            poll_interval=0.05)
    >>> done.status, done.answer
    ('done', 'approve')
    >>> done.stages
    ('instruction', 'clone', 'budget', 'ask', 'answer', 'run', 'record')
    >>> runtime.calls == [{"instruction": "Pin rad godot to the vectors.",
    ...                    "cwd": str(clone), "resume": None}]
    True
    >>> recorded = client.get_instruction("pin-the-vectors")
    >>> recorded["status"], recorded["exit_code"], recorded["outcome_text"], recorded["session_ref"]
    ('done', 0, 'pinned to 2c10fd1', 'session-after')
    >>> print(render(recorded["declared"]))
      1 of 1 authorised call(s) made against scripted.
      This budget was for this command and is not remembered.

Four acts, one run. The three consents are three records on the queue, each
numbered after the first, and the run is the only one with an outcome:

    >>> [r.id for r in client.list_human_requests(request_type="approval", oldest_first=True)]
    ['instruction-pin-the-vectors', 'instruction-pin-the-vectors-2', 'instruction-pin-the-vectors-3']
    >>> len(runtime.calls)
    1

    >>> client.close()
    >>> _ = server.__exit__(None, None, None)

## What this page does not claim

That an agent did anything: `scripted` answers as told, and the one runtime
that runs a tool lives in `qmcp.integrations.agents.adapters`, where
`tests/test_agents_runtime.py` asserts its command line without launching it.
Nor that the clone the archive would have named is the right one; the page
passed it, and the prompt says which so the person at the gate can refuse.
