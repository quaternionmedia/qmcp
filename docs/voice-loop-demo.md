# The voice loop, demonstrated

The voice-driven development loop, end to end, in three tiers. Each tier is a
command that runs, and each adds one thing the tier before it stood in for:
first nothing real, then the real local model, then a person at the
microphone -- where, once two servers are started, nothing is typed at all.
**Continuity comes from qmcp, not the model**: every tier ends with a second
instruction handed what this server recorded of the first, and nothing a
runtime kept.

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
uv run qmcp cookbook converse                     # one whole spoken session, nothing typed
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
- `cookbook converse` is tier 3's conversation with every take scripted: it
  says it is ready, asks a question an agent left waiting, waits through a
  silence, takes two instructions the whole way -- the first in the clone found
  by name, the second in the remembered one, told what the first found --
  goes back to waiting on "no", and ends on "stop listening".

Each prints `[ok]` per case and exits non-zero on any `[FAIL]`. The suite runs
all four, and makes sure each can fail.

## Tier 2 — the local model: continuity, really

The model `qmcp localmodel` stands up, reading a real clone. Speech is still the
deterministic engine, so each consent is answered `approve` by a script
standing in for the person; nothing is written to the clone and nothing paid
is called.

```bash
uv run qmcp localmodel check                      # is the model installed and served?
uv run qmcp cookbook instruct --runtime local     # reads this repository
uv run qmcp cookbook instruct --runtime local --clone <path to a rostered project>
uv run qmcp cookbook converse --runtime local     # the whole session, on the model
```

`cookbook converse --runtime local` looks for a clone named `qmcp` beside this
checkout; from a worktree, or a checkout kept elsewhere, pass `--clones` the
directory that holds one.

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

## Tier 3 — a person at the microphone: two commands, then speech

Everything real, and nothing typed after the servers start:

```bash
# in joe's checkout -- the speech engine, which owns the microphone
uv run joe voice setup                          # once: find, save and prove the microphone
uv run joe dev                                  # the page at http://localhost:3000/joe

# here -- the server, and the conversation beside it
uv run qmcp serve --converse --runtime local
```

Either may start first: the conversation waits for the other. Then it talks:

```
qmcp:   Ready. What should be done?
you:    Which file in qmcp says what qmcp is?
qmcp:   I heard: Which file in qmcp says what qmcp is. Say record or again.
you:    record
qmcp:   Recorded for qmcp.
qmcp:   Act on the instruction: ... Project qmcp, clone ..., runtime local, budget 1 run(s). Say approve or hold.
you:    approve
qmcp:   Approved. Running in qmcp.
qmcp:   Done in qmcp. The file README.md ... The rest is on the record.
qmcp:   Anything else?
you:    yes
qmcp:   What should be done?
you:    In one sentence, what did that file say qmcp is for?
        ... record ... approve -- the consent says it carries 1 earlier instruction ...
qmcp:   Done in qmcp. QMCP is the server that ...
qmcp:   Anything else?
you:    stop listening
qmcp:   Stopping. Start the server again to talk.
```

joe's page shows each turn live: the question, the open microphone, the person
speaking, the pause, the reading. A question an agent has put on the human
queue is asked aloud before the next instruction. The first time a project is
acted on, its clone is the one named for it beside this checkout (`--clones`
moves where it is looked for); after that, the one its last act ran in. A room
where people talk can require a word first with `--wake`. While it runs it
listens take after take, and joe keeps every take as a file in its checkout's
`Data/Voice`; nothing deletes them.

Each step is also a command, for checking and debugging rather than for the
loop: `uv run qmcp instruct --voice`, `uv run qmcp instructions act <id>
--runtime local --budget 1 --voice`, `uv run qmcp instructions say <id> --speak`
and `uv run qmcp instructions show <id>`.

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
  the GPU busy and later calls queued behind it. Every call qmcp makes is
  capped, and a call that stalls unloads the model through the service's own
  keep-alive and is made once more against a fresh load; a second stall is a
  failed run naming the endpoint, and the conversation goes on.
- **The microphone hears nothing.** `uv run joe voice setup` in joe's checkout,
  and `docs/integrations/voice.md`, "Which microphone".
- **An act refuses for want of a clone.** The conversation looks for a clone
  named for the project beside this checkout, or in `--clones`; by command,
  pass `--cwd` once. The project remembers it either way.
- **Nothing is said, or the conversation seems to have gone.** `curl
  http://127.0.0.1:3141/v1/human/voice` says whether it is running and the last
  lines it printed: what it is waiting for, or why it ended.
- **The page's voice buttons say a conversation is running.** The standing
  conversation holds the microphone; speak to it instead, or start the server
  without `--converse`.
- **The page says qmcp cannot be reached.** Start `uv run qmcp serve`; joe's
  dev server reaches it at `http://localhost:3141`, and `QMCP_URL` moves it.
