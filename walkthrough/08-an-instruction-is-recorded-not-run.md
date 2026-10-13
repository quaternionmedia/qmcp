# 08 — An instruction is recorded, not run

Everything on this page runs. It is executed by the ordinary test command, so
an example that stops being true fails the build rather than sitting here
misleading somebody.

**The problem.** The voice loop answers questions an agent asks: a closed
choice, spoken and recorded on the human queue (`06`, `07` and
`docs/integrations/voice.md`). Nothing ran the other direction. A person who
wanted something done in a project said so in whichever conversation happened
to be open, and when that conversation closed the instruction closed with it.
The inbox is where an instruction is kept instead: in the person's words,
against a project, with the evidence for the project beside it. **Recording one
executes nothing**, and this page shows that by looking.

## One process, the inbox over its own database

The module takes the app rather than creating one, as `qmcp.topology_designs`
does. The rows go in a database this page owns, because the configured one is
somebody's inbox, and the roster is handed in so the page does not depend on
what the governance submodule lists this week:

    >>> import tempfile
    >>> from pathlib import Path
    >>> from fastapi import FastAPI
    >>> from fastapi.testclient import TestClient
    >>> from qmcp.instructions import service
    >>> from qmcp.integrations.voice.service import VoiceRuns
    >>> from qmcp.topology_designs import sessions_at

    >>> root = Path(tempfile.mkdtemp())
    >>> started = []
    >>> def popen(argv, stdout, stderr, env):
    ...     started.append(argv)
    ...     class Process:
    ...         def poll(self):
    ...             return None
    ...     return Process()
    >>> runs = VoiceRuns(log_dir=root, popen=popen)
    >>> app = FastAPI()
    >>> service.register(app, runs, names=lambda: ("qmcp", "vox", "dossier", "rad-godot"),
    ...                  sessions=sessions_at(root / "inbox.db"))
    >>> client = TestClient(app)

## One project named

The text names `qmcp` and nothing else on the roster, so the instruction is
`recorded` against it. The row says which names matched and by what rule, so a
reader can disagree with the reading without losing the evidence:

    >>> row = client.post("/v1/instructions",
    ...                   json={"text": "Deploy qmcp to the pi."}).json()
    >>> row["status"], row["project"], row["source"]
    ('recorded', 'qmcp', 'typed')
    >>> row["detail"]
    {'candidates': ['qmcp'], 'rule': 'the one project named, as a whole word'}

A hyphenated name is matched as a transcript says it, since no transcript
writes the hyphen:

    >>> spoken = client.post("/v1/instructions",
    ...                      json={"text": "Pin rad godot to the vectors."}).json()
    >>> spoken["status"], spoken["project"]
    ('recorded', 'rad-godot')

## None named, and several named

Neither is guessed at. An instruction naming no project is `unresolved` with no
candidates; one naming two is `unresolved` with both, in the roster's order:

    >>> none = client.post("/v1/instructions",
    ...                    json={"text": "Rotate the logs."}).json()
    >>> none["status"], none["project"], none["detail"]["candidates"]
    ('unresolved', None, [])
    >>> both = client.post("/v1/instructions",
    ...                    json={"text": "Move the vectors from vox into qmcp."}).json()
    >>> both["status"], both["detail"]["candidates"], both["detail"]["rule"]
    ('unresolved', ['qmcp', 'vox'], 'several projects named')

A name inside another word is not a match. `rad` is on the real roster and
lives inside `gradient`; here `vox` lives inside a made-up word, and the
instruction stays unresolved:

    >>> client.post("/v1/instructions",
    ...             json={"text": "Fix the voxel shader."}).json()["project"] is None
    True

## Stated outright

A caller that knows the project says so, and the matching is skipped. The rule
records that the project was stated rather than read:

    >>> stated = client.post("/v1/instructions", json={
    ...     "text": "Deploy qmcp to the pi.", "project": "dossier",
    ...     "source": "page"}).json()
    >>> stated["project"], stated["detail"]["rule"]
    ('dossier', 'stated')

## The inbox, newest first

    >>> listed = client.get("/v1/instructions").json()
    >>> listed["count"]
    6
    >>> [r["text"] for r in listed["instructions"]][:2]
    ['Deploy qmcp to the pi.', 'Fix the voxel shader.']
    >>> [r["text"] for r in
    ...  client.get("/v1/instructions", params={"status": "unresolved"}).json()["instructions"]]
    ['Fix the voxel shader.', 'Move the vectors from vox into qmcp.', 'Rotate the logs.']
    >>> client.get(f"/v1/instructions/{row['id']}").json() == row
    True

## What recording did not do

Six instructions are in the inbox. No process was started, and the machine's
one conversation is idle:

    >>> started
    []
    >>> client.get("/v1/instructions/voice").json()["running"]
    False

Nor does a recorded row say otherwise. Recording reaches two words of the
status vocabulary, and neither describes a run; the rest are the path an act
walks (`09`), and a row is on it only because a person issued the command:

    >>> from qmcp.db.models import InstructionStatus
    >>> sorted({r["status"] for r in listed["instructions"]})
    ['recorded', 'unresolved']
    >>> [s.value for s in InstructionStatus][:2]
    ['recorded', 'unresolved']

## Speaking one

`POST /v1/instructions/voice` is how a page offers "Speak an instruction". It
starts `qmcp instruct --voice` in a process of its own, through the tracker the
approval route uses, so an approval being asked and an instruction being taken
cannot overlap:

    >>> client.post("/v1/instructions/voice").status_code
    202
    >>> started[0][1:5]
    ['-m', 'qmcp', 'instruct', '--voice']
    >>> status = client.get("/v1/instructions/voice").json()
    >>> status["running"], status["kind"]
    (True, 'instruction')
    >>> again = client.post("/v1/instructions/voice")
    >>> again.status_code, "one microphone" in again.json()["detail"]
    (409, True)

What that command says and hears is `qmcp cookbook instruct`'s to show, against
vox's deterministic engine; the dialog is `qmcp.instructions.dialog`.

## What this page does not claim

That the project a transcript names is the project the person meant: whether a
name transcribes reliably is unmeasured, which is why the spoken dialog reads
the text back and asks an ambiguous name as a closed choice rather than trusting
either. Nor that anything will act on these rows. The inbox holds them; acting
is a command a person issues, behind consent on the human queue, and `09` is
where it runs.
