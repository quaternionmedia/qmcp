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
three ways an act ends without running, the one way it runs, and what the next
instruction is told -- through the `scripted` runtime, which answers as told,
spends nothing, and keeps every brief it is handed.

## One process, a real queue and an inbox of its own

The queue is the real one: a qmcp server on an ephemeral port over a database
made for this page, reached through the ordinary client. The inbox is the same
file, so the row the act writes is the row the server serves, and it is also
the record the clone and the history are read from. Nothing has been acted on
here yet, so the record names no clone and the first acts pass one:

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

    >>> runtime = ScriptedRuntime(text="pinned to 2c10fd1")

## Zero is the default, and it asks nothing

A budget of zero runs is a real count. The act resolves the clone, declares
what would be asked, and stops; no request reaches the queue and the row is as
it was:

    >>> free = act("pin-the-vectors", runtime, Budget(), client=client, rows=rows,
    ...            cwd=clone)
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

qmcp's record names no clone for the project and nothing was passed, so the
act refuses before it asks, with the row unchanged:

    >>> nowhere = act("pin-the-vectors", runtime, Budget(authorised=1), client=client,
    ...               rows=rows)
    >>> nowhere.status, nowhere.stages
    ('recorded', ('instruction', 'clone'))
    >>> nowhere.why
    "no clone for 'rad-godot' in qmcp's record yet; pass --cwd <path to the project's clone>, and it is remembered for the project's later instructions."

## Held: the consent is asked, answered, and nothing runs

With one run authorised the consent goes on the queue. It is answered here by
voice, through the same loop `qmcp human voice` runs, with a synthesizer and a
transcriber stood in for so the page can show what is said. The consent is
said in a few plain words -- where, what, how, and how many runs -- and the
request written on the queue names the clone's whole path. The person says
hold:

    >>> class Hears:
    ...     def __init__(self, text): self.text = text
    ...     def listen(self, duration=5.0, *, pause_ms=None): return self.text, "take.wav"
    >>> class Says:
    ...     spoken = []
    ...     def speak(self, text, out_path=None): self.spoken.append(text); return "out.wav"

    >>> held = act("pin-the-vectors", runtime, Budget(authorised=1), client=client,
    ...            rows=rows, cwd=clone, stt=Hears("hold"), tts=Says(),
    ...            poll_interval=0.05)
    >>> Says.spoken[0]
    'Run in rad-godot: Pin rad godot to the vectors. A script, one run. Approve or hold?'
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
lives `CONSENT_SECONDS`, and the server's floor is a minute, so rather than
wait the page moves the server's clock past the expiry in place of sleeping
between listings: the listing drops the request, the one read afterwards is
what marks it expired, and nothing runs.

    >>> from datetime import datetime, timedelta
    >>> import qmcp.server
    >>> from qmcp.instructions.act import CONSENT_SECONDS
    >>> class Later(datetime):
    ...     @classmethod
    ...     def now(cls, tz=None): return datetime.now(tz) + timedelta(seconds=CONSENT_SECONDS + 60)
    >>> def nobody_answers(seconds): qmcp.server.datetime = Later
    >>> expired = act("pin-the-vectors", runtime, Budget(authorised=1), client=client,
    ...               rows=rows, cwd=clone, sleep=nobody_answers, poll_interval=0.05)
    >>> qmcp.server.datetime = datetime
    >>> expired.status, expired.answer, runtime.calls
    ('unanswered', None, [])
    >>> client.get_human_request(expired.request_id)[0].status
    'expired'

## Approve: consented before it starts, then run in the clone

The fourth act is approved. The runtime is called once, in the clone, and the
outcome is recorded, where the project's next instruction will find it:

    >>> Says.spoken.clear()
    >>> done = act("pin-the-vectors", runtime, Budget(authorised=1), client=client,
    ...            rows=rows, cwd=clone, stt=Hears("yes, go ahead"), tts=Says(),
    ...            poll_interval=0.05)
    >>> done.status, done.answer
    ('done', 'approve')
    >>> done.stages
    ('instruction', 'clone', 'budget', 'ask', 'answer', 'run', 'record')
    >>> runtime.calls == [{"instruction": "Pin rad godot to the vectors.", "cwd": str(clone)}]
    True
    >>> runtime.briefs[0].history
    ()
    >>> recorded = client.get_instruction("pin-the-vectors")
    >>> recorded["status"], recorded["exit_code"], recorded["outcome_text"]
    ('done', 0, 'pinned to 2c10fd1')
    >>> print(render(recorded["declared"]))
      1 of 1 authorised call(s) made against scripted.
      This budget was for this command and is not remembered.

Four acts, one run. The three consents are three records on the queue, each
numbered after the first, and the run is the only one with an outcome:

    >>> [r.id for r in client.list_human_requests(request_type="approval", oldest_first=True)]
    ['instruction-pin-the-vectors', 'instruction-pin-the-vectors-2', 'instruction-pin-the-vectors-3']
    >>> len(runtime.calls)
    1

## The next instruction is told what the last one found

Continuity comes from qmcp, not the model. A second instruction in the same
project is acted on with no `--cwd` and by a different runtime. The clone is
the one the project's last act ran in, and the brief carries what that act
found, read from this server's record -- nothing about it was kept by the
runtime that ran it:

    >>> with rows() as session:
    ...     session.add(Instruction(id="say-the-pin", text="Say which commit the vectors are pinned to.",
    ...                             project="rad-godot", source=InstructionSource.VOICE))
    ...     session.commit()
    >>> class Another(ScriptedRuntime):
    ...     name = "another"
    >>> another = Another(text="2c10fd1.")
    >>> Says.spoken.clear()
    >>> second = act("say-the-pin", another, Budget(authorised=1), client=client,
    ...              rows=rows, stt=Hears("approve"), tts=Says(), poll_interval=0.05)
    >>> second.status, second.cwd == str(clone), second.carried
    ('done', True, ('pin-the-vectors',))
    >>> Says.spoken[0]
    'Run in rad-godot: Say which commit the vectors are pinned to. Runtime another, one run, with what came before. Approve or hold?'
    >>> (turn,) = another.briefs[0].history
    >>> turn.instruction, turn.runtime, turn.outcome
    ('Pin rad godot to the vectors.', 'scripted', 'pinned to 2c10fd1')
    >>> print("\n".join(another.briefs[0].prompt().splitlines()[2:4]))
    What has been asked in this project so far, from qmcp's record, oldest first:
    1. ... asked: Pin rad godot to the vectors.

The record says which turns were carried and what chose the clone, so a reader
of the outcome can see what the runtime was told:

    >>> detail = client.get_instruction("say-the-pin")["detail"]
    >>> detail["continuity"], detail["clone"]["rule"]
    (['pin-the-vectors'], "the clone the project's last act ran in, from qmcp's record")

    >>> client.close()
    >>> _ = server.__exit__(None, None, None)

## What this page does not claim

That an agent did anything: `scripted` answers as told. The runtimes that run
something -- the local model and a coding assistant's command line -- live in
`qmcp.integrations.agents.adapters`, given the same brief, and
`tests/test_agents_local.py` and `tests/test_agents_runtime.py` assert what each
sends without a model or a tool. Nor that the clone is the right one; the
prompt says which so the person at the gate can refuse.
