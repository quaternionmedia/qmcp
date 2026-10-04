# The voice loop, demonstrated

The voice-driven development loop, end to end, in three tiers. Each tier is a
command that runs, and each adds one thing the tier before it stood in for:
first nothing real, then the real local model, then a person at the
microphone. **Continuity comes from qmcp, not the model**: every tier ends with
an instruction carried out because this server remembered the project's
earlier work, not because a runtime did.

```
speak ──> joe: microphone, transcription ──> qmcp: recorded against a project
                                                         │
                                        consent asked aloud: approve or hold
                                                         │ approve
hear  <── vox: synthesis <── qmcp: summary <── local model reads the clone,
                                                  told the project's history by qmcp
```

`docs/integrations/voice.md` is the reference for every step; this page is
the order to run them in and what each one proves.

## Tier 1 — nothing real: the wiring

No microphone, speakers, model or agent, and nothing spent. A qmcp server is
started on an ephemeral port over a database made for the run, and vox's
deterministic engine stands in for the speech engine, so every turn travels the
real HTTP path:

```bash
uv run qmcp cookbook voice                        # questions answered by voice
uv run qmcp cookbook instruct                     # instructions recorded, consented, run, said back
uv run qmcp cookbook instruct --runtime scripted  # continuity, on a runtime that spends nothing
```

- `cookbook voice` answers queued questions by voice: a yes read onto the
  options, an option by name, a mishearing re-asked, silence re-asked, and an
  open question whose transcript is read back and recorded.
- `cookbook instruct` records spoken instructions -- one project named, none
  named and asked for, several named and chosen between, a second take -- and
  then takes two the whole way: consent asked aloud, the `scripted` runtime run
  in a directory standing in for the clone, and the summary said back. The
  held one runs nothing and says so. Each loop prints as the conversation it
  was.
- `--runtime scripted` is the continuity demonstration below with nothing
  real behind it, so it checks what qmcp owns -- the clone remembered, the
  history carried, the summary said -- on any machine.

Each prints `[ok]` per case and exits non-zero on any `[FAIL]`. The suite runs
all three, and makes sure each can fail.

## Tier 2 — the local model: continuity, really

The model `qmcp localmodel` stands up, reading a real clone. Speech is still the
deterministic engine, so each consent is answered `approve` by a script
standing in for the person; nothing is written to the clone and nothing paid
is called.

```bash
uv run qmcp localmodel check                      # is the model installed and served?
uv run qmcp cookbook instruct --runtime local     # reads this repository
uv run qmcp cookbook instruct --runtime local --clone <path to a rostered project>
```

Two instructions in one project. The first asks which file says what the
project is, and passes the clone. The second asks "what did *that file* say it
is for?", passes no clone, and can be answered only if qmcp remembers the clone
and hands the model what the first instruction found. What to look for:

```
  instruction 2: In one sentence, what did that file say qmcp is for?
    said      Act on the instruction: ... carrying 1 earlier instruction(s) from qmcp's record. ...
    heard     approve
    read      read_file(README.md)
    carried   instruction 1 and what it found, from qmcp's record
    said      Done in qmcp. ...
  [ok]   ran in the remembered clone, told what the first found
```

The model's words will differ from run to run; the structure will not. The
command refuses to start, and says what is missing, when the model is not
served. A seven-billion-parameter model is a quick reader, not a reviewer: it
may quote a heading where a sentence was asked for, and the person who said
approve is the one who judges what it found.

## Tier 3 — a person at the microphone

Everything real. Three processes on one machine, each from its own checkout:

```bash
# in joe's checkout -- the speech engine, which owns the microphone
uv run joe voice setup        # once: find, save and prove the microphone
uv run joe dev                # the page at http://localhost:3000/joe, and the engine

# here -- the backend; the commands below run from the same directory,
# because the database path is relative to it
uv run qmcp serve
```

Then:

1. **Speak an instruction.** Press **Instruct by voice** on joe's page, or run
   `uv run qmcp instruct --voice`. qmcp asks "What should be done?", listens
   with the long pause an instruction needs, reads the transcript back, and
   records it on `record`. The page shows the turn live and then the row.
2. **Act on it.**
   `uv run qmcp instructions act <id> --runtime local --budget 1 --cwd <clone> --voice`.
   The consent is asked aloud through joe's microphone; on `approve` the local
   model reads the clone and the answer is said back. Later instructions in the
   same project leave out `--cwd`: the clone is remembered.
3. **Hear it again.** `uv run qmcp instructions say <id> --speak`;
   `uv run qmcp instructions show <id>` for the whole outcome and what was
   carried.

Questions an agent puts on the human queue are answered the same way, from the
page's **Answer by voice** or with `uv run qmcp human voice`.

## What every tier is held to

```bash
python governance/qm/project-seed/ci/run_workflows_locally.py --event pull_request --base-ref main
```

runs every workflow's steps on this machine, so a hosted run is a mirror and
never the only place a gate runs. A pass is evidence, not proof: `uses:` steps
and the runner image are not reproduced.

## When it does not work

- **The local model does not answer, or answers late.** `uv run qmcp localmodel
  check` says whether it is installed and served. The model service on the
  machine this was built on was seen to stall partway through a reply, with
  the GPU busy and later calls queued behind it; every call qmcp makes is
  capped, so a stall is a failed run naming the endpoint, and unloading the
  model with the service's own command frees it.
- **The microphone hears nothing.** `uv run joe voice setup` in joe's checkout,
  and `docs/integrations/voice.md`, "Which microphone".
- **An act refuses for want of a clone.** Pass `--cwd` once; the project
  remembers it.
- **The page says qmcp cannot be reached.** Start `uv run qmcp serve`; joe's
  dev server reaches it at `http://localhost:3141`, and `QMCP_URL` moves it.
