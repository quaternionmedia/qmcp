# The voice dev loop

An instruction spoken at the machine is recorded against a project, consent is
asked aloud, the local model reads the project's clone, and the answer is said
back -- and once two servers are started, nothing is typed at all.
**Continuity comes from qmcp, not the model**: each instruction is handed the
project's earlier instructions and what they found, from this server's record,
so work carries across sessions and runtimes rather than living in whichever
conversation happens to be open.

```
speak ──> joe: microphone, transcription ──> qmcp: recorded against a project
                                                         │
                                        consent asked aloud: approve or hold
                                                         │ approve
hear  <── vox: synthesis <── qmcp: summary <── local model reads the clone,
                                                  told the project's history by qmcp
```

**Onboarding** sets a workstation up and proves it in three tiers, each adding
one real thing the tier before it stood in for. **Cookbook** is what to say,
and what happens. `docs/integrations/voice.md` is the reference for every step.

## Onboarding

### What runs where

| Part | Its job | Started by |
|---|---|---|
| qmcp, this repository | records each instruction, asks consent, acts, remembers | `uv run qmcp serve`, on `http://127.0.0.1:3141` |
| the local model | reads a project's clone and answers; cannot write | the model service `qmcp localmodel` sets up |
| joe, its own checkout | owns the microphone, transcribes with whisper, shows each turn on its page | `uv run joe dev`: the speech engine on port 8000, the page on 3000 |
| vox, `vendor/vox` | the contract a speech engine answers, and the local synthesizer | nothing; qmcp imports it |
| the projects' clones | what the model reads | nothing; directories beside this checkout |

### The workspace

The conversation finds a project's clone by its name, beside this checkout, so
the workspace keeps its repositories as siblings:

```
<workspace>/
  qmcp/            this repository
  <project>/       one clone per project to talk about, named as on the roster
```

joe can live anywhere: it is a separate server. A project is recognised in
speech by its name on the roster, `ci/workspace.yaml` in the `governance/qm`
submodule; a name that is not on it is not recognised, and the instruction is
recorded unresolved. A checkout kept elsewhere -- a worktree, say -- passes
`--clones <workspace>` wherever this page names it.

### Set up, once

Needed first: `git`, [`uv`](https://docs.astral.sh/uv/), which fetches the
Python each repository pins, and Node.js with npm for joe's page.

1. **qmcp, in a checkout that has the loop**, with both submodules -- the
   roster and vox:

   ```bash
   git clone https://github.com/quaternionmedia/qmcp
   cd qmcp
   git submodule update --init       # first: any uv command fails while vendor/vox is empty
   uv run qmcp serve --help          # must list --converse; see below if it does not
   uv sync --all-extras
   uv run qmcp cookbook converse     # done when it prints [ok]: a whole spoken session, offline
   ```

   A checkout whose `serve` does not list `--converse` predates the loop: it is
   on a branch without it, or behind its remote. Switch to a branch that has
   it -- the one this page was read from -- and run `git submodule update
   --init` after every switch: branches pin different `vox` commits, and a
   stale one still passes `cookbook voice` while every instruction fails.

2. **The local model.** `check` reports the machine and whether the model is
   served; `plan` prints the exact commands that install the model service,
   keep the weights on a drive with room, pull the model once and prove it
   answers. It runs nothing itself:

   ```bash
   uv run qmcp localmodel check
   uv run qmcp localmodel plan       # then run what it prints
   uv run qmcp localmodel check      # done when it says `served:` and names the model
   ```

   `served:` means the service answers and lists the model; the plan's last
   command asks the model for one word, and its answer is what proves it
   generates.

3. **joe**, in its own checkout. `voice setup` counts down, tries every input
   while someone keeps talking, and saves the microphone only once it has
   transcribed a sentence from it. The first transcription fetches whisper's
   model, about 145 MB, into the user's cache:

   ```bash
   git clone https://github.com/quaternionmedia/joe
   cd joe
   uv sync
   npm install
   uv run joe voice setup            # done when it saves the microphone
   ```

4. **The clones.** A clone of each project to talk about, beside this checkout
   and named as on the roster. Nothing is written to them: the local runtime's
   tools read and cannot write.

5. **Hear it.** qmcp's voice plays on the system's default output, which is
   not always the one a person hears -- a second jack, a monitor, a capture
   card:

   ```bash
   uv run vox say "Testing qmcp's voice."      # done when it is heard
   uv run vox outputs                          # if not: the outputs, by name
   uv run vox say "Testing." --output "<a fragment of one>"
   ```

   Once one is heard, set `VOX_OUTPUT_DEVICE` to that fragment in the terminal
   that starts qmcp, and `JOE_OUTPUT_DEVICE` to it in joe's, for joe's tones.

### Prove it, tier by tier

#### Tier 1 — nothing real: the wiring

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
- `--runtime scripted` is tier 2's continuity demonstration with nothing real
  behind it, so it checks what qmcp owns -- the clone remembered, the history
  carried, the summary said -- on any machine.
- `cookbook converse` is tier 3's conversation with every take scripted: it
  says it is ready, asks a question an agent left waiting, waits through a
  silence, takes two instructions the whole way -- the first in the clone found
  by name, the second in the remembered one, told what the first found --
  goes back to waiting on "no", and ends on "stop listening".

Each prints `[ok]` per case and exits non-zero on any `[FAIL]`. The suite runs
all four, and makes sure each can fail.

#### Tier 2 — the local model: continuity, really

The model `qmcp localmodel` stands up, reading a real clone. Speech is still the
deterministic engine, so each consent is answered `approve` by a script
standing in for the person; nothing is written to the clone and nothing paid
is called.

```bash
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
    said      Run in qmcp: ... The local model, one run, with what came before. Approve or hold?
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

#### Tier 3 — a person at the microphone: two commands, then speech

Everything real, and nothing typed after the servers start:

```bash
# in joe's checkout: the speech engine, and the page at http://localhost:3000/joe
uv run joe dev

# here: the server, and the conversation beside it
uv run qmcp serve --converse --runtime local
```

Either may start first: the conversation waits for the other. qmcp's terminal
shows the server's request log, not the conversation: what it hears and says
is on joe's page, and `curl http://127.0.0.1:3141/v1/human/voice` gives its
last lines. Then it talks:

```
qmcp:   Ready. What should be done?
you:    Which file in qmcp says what qmcp is?
qmcp:   I heard: Which file in qmcp says what qmcp is. Agree or again?
you:    agree
qmcp:   Recorded for qmcp.
qmcp:   Run in qmcp: Which file in qmcp says what qmcp is? The local model, one run. Approve or hold?
you:    approve
qmcp:   Approved.
qmcp:   Running in qmcp.
qmcp:   Done in qmcp. The file README.md ...
qmcp:   Anything else?
you:    yes
qmcp:   What should be done?
you:    In one sentence, what did that file say qmcp is for?
        ... agree ... approve -- the consent ends "with what came before" ...
qmcp:   Done in qmcp. QMCP is the server that ...
qmcp:   Anything else?
you:    stop listening
qmcp:   Stopping.
```

joe's page shows each turn live: the question, the open microphone, the person
speaking, the pause, the reading. joe plays two rising notes when it is the
person's turn to speak and one lower note when it has heard them, so the turns
can be followed without looking. While the conversation runs it listens take
after take, and joe keeps every take as a file in its checkout's `Data/Voice`;
nothing deletes them. Onboarding is done when this tier has run once.

## Cookbook

Unless a recipe says otherwise, it assumes tier 3 is running: `uv run joe dev`
in joe's checkout and `uv run qmcp serve --converse --runtime local` here. What
is said to it is in quotes; what it says back is in italics.

### Ask about a project

Say the instruction with the project's name in it: "Which file in qmcp says
what qmcp is?" *I heard: ...* -- and when the engine heard it confidently, that
is all: a moment's silence agrees, and it is recorded. Say anything in that
moment, or press a key, and it is not taken for agreed: "again" takes it
again, "agree" records it, and anything else is asked about outright. Heard
less surely, the read-back asks: *Agree or again?* The consent says what will
run in a few words -- the project, the instruction, the runtime, how many runs,
and whether it carries what came before: *Run in qmcp: ... The local model, one
run. Approve or hold?* Say "approve". *Approved. Running in qmcp.* -- and when the model
has read what it needs, the first sentence of what it found, said back. The
whole answer is on the record: `uv run qmcp instructions show <id>`. Missed a
question? Say "repeat", or press `R`.

### Pick up where the last session left off

Nothing to do. Every instruction in a project is handed that project's most
recent instructions that ran, and what each found, from qmcp's record --
whichever runtime ran them, and however long ago. "What did that file say qmcp
is for?" works the next morning as it does the next minute. The consent says
it carries what came before, and the row's `detail.continuity` names it.

### Talk about a project for the first time

Name it; nothing else is needed if its clone sits beside this checkout under
its roster name. The first act runs there and the clone is remembered, so every
later act in the project runs in the same one. A clone kept elsewhere: start
the server with `--clones <the directory holding it>`, or act once by command
with `--cwd <path>`, which is remembered the same way. With no clone found, the
act refuses before it asks and says why.

### Name no project, or several

"Deploy to the pi." names none: *Which project?* is asked once, and the name
said back is recorded as the project when it is one on the roster. "Move the
vectors from vox into qmcp." names two: *Which project? Qmcp or vox?* --
say one. A project is never picked by position, so "yes" chooses nothing.

### Hold instead of approve

"hold" to the consent: nothing runs, and it says so. The instruction stays
recorded, and `uv run qmcp instructions act <id> --runtime local --budget 1
--voice` asks again later, as a new consent beside the first.

### Answer what an agent is waiting on

Nothing to start. Before each instruction, questions agents have put on the
human queue are asked aloud, oldest first -- *A question is waiting.*, then the
question with its options. Say one. A question nobody answers stays pending for
anyone: `uv run qmcp human list`, `uv run qmcp human voice`, or joe's page.

### No more for now, and stopping

"no" or "that's all" to *Anything else?*: *Listening.* The
microphone stays open, and the next instruction can come at any time; "yes"
asks for it now. "stop listening" or "goodbye" ends the conversation, and so
does stopping the server: *Stopping.*

### Answer before the question ends

Knowing the answer, give it: say "approve" over the consent, or press `1`, and
the question stops mid-sentence and the answer is taken. A spoken answer is
kept from its first word. Through speakers, speak up over the voice -- an
answer is taken over the question only when it is clearly louder than the
question's own echo -- or use the keys; with headphones, any speech does it.
Every closed question works this way: the consent, the read-back, a choice of
project, *Anything else?*. `JOE_BARGE_IN=0` on joe leaves only the keys.

### Answer without speaking, and follow by ear

Every question's answers are on joe's page as numbered buttons, and with the
page focused the keys answer the same way, without looking:

| Key | Does |
|---|---|
| `1`–`9` | answers with the question's options in the order it says them: *"Approve or hold?"* makes `1` approve and `2` hold |
| `R` | says the question again, as saying "repeat" or "what?" does; neither spends one of its retries |
| `Shift`+`Esc` | stops listening |
| `~`, held | keeps the turn open through pauses -- an instruction with a long thought in it; releasing it ends the turn |

A key or button counts as having said the word: the take in progress ends at
once, and with `--wake` no wake word is needed. Two tones carry what the page
shows: two rising notes just before the microphone opens for an answer, and
one lower note once the turn has been heard; `JOE_CUES=0` in joe's environment
turns them off. With the keys, the tones and qmcp's own voice, the loop runs
with nobody looking at a screen.

Each closed question also tells the speech engine the words its answer is
expected to be, which joe hands its transcriber as a prompt: a clipped
"approve" is far likelier to come back as that word, and an instruction's
take is told the project names, so "qmcp" is spelled as a project rather than
as whatever it sounded like.

### A room where people talk

```bash
uv run qmcp serve --converse --runtime local --wake qmcp
```

Everything said to it between turns then begins with the word -- "qmcp, which
file says what dossier is?", "qmcp, yes", "qmcp, stop listening" -- and
anything else heard is ignored. Answers to its own questions, "agree" and
"approve" among them, need no word. Without `--wake`, every utterance is read
back before anything is recorded, so talk in the room interrupts but never
records itself: nothing runs without
"approve".

### Hear a result again

Needs only the server.

```bash
uv run qmcp instructions list                    # newest first, with ids
uv run qmcp instructions say <id> --speak        # what it came to, said aloud; printed without --speak
uv run qmcp instructions show <id>               # the whole row: what it found, where, what it carried
```

### Check it on a machine nobody is at

```bash
uv run qmcp cookbook converse                              # a whole session, every take scripted
uv run qmcp converse --runtime local --synth recording     # against running servers, speech written to files
```

The first needs nothing running. The second runs the conversation on its own
against a running `qmcp serve` and speech engine, and writes each sentence to a
file instead of speaking it.

### One step at a time, by command

For checking and debugging rather than for the loop, with the server started
without `--converse`:

```bash
uv run qmcp instruct --voice                                          # one instruction, spoken; prints the row
uv run qmcp instructions act <id> --runtime local --budget 1 --voice  # consent asked aloud, then the run
uv run qmcp instructions show <id>                                    # what it came to
```

joe's page offers the first as **Instruct by voice**, and **Answer by voice**
for a waiting question. `--budget` counts runs and defaults to zero, which
declares what would be asked and stops.

## What every tier is held to

```bash
python governance/qm/project-seed/ci/run_workflows_locally.py --event pull_request --base-ref main
```

runs every workflow's steps on this machine, so a hosted run is a mirror and
never the only place a gate runs. A pass is evidence, not proof: `uses:` steps
and the runner image are not reproduced.

## When it does not work

- **Nothing is heard, though the page shows the questions being asked.**
  qmcp's voice plays on the default output: step 5 of "Set up, once" finds the
  one that is heard and names it with `VOX_OUTPUT_DEVICE`. The conversation's
  log (below) lists each sentence as `said:`, so a sentence that is logged and
  not heard is on another output.
- **`Error: No such option '--converse'`** -- or `instruct`, `instructions` or
  `converse` is not a command. This checkout predates the loop: see step 1 of
  "Set up, once". `uv run qmcp serve --help` lists what this checkout has.
- **The local model does not answer, or answers late.** `uv run qmcp localmodel
  check` says whether it is installed and served. The model service on the
  machine this was built on was seen to stall partway through a reply, with
  the GPU busy and later calls queued behind it. Every call qmcp makes is
  capped, and a call that stalls unloads the model through the service's own
  keep-alive and is made once more against a fresh load; a second stall is a
  failed run naming the endpoint, and the conversation goes on. When every
  call stalls, fresh loads included, the service is answering without
  generating -- `localmodel check` still says served, and the run says *not
  generating*: restart the service with the commands `uv run qmcp localmodel
  plan` prints, and see that nothing else holds the GPU.
- **The microphone hears nothing.** `uv run joe voice setup` in joe's checkout,
  and `docs/integrations/voice.md`, "Which microphone".
- **It went quiet after a consent.** An answer it could not read clearly
  leaves the consent pending -- it never guesses -- and the conversation waits
  for an answer from anywhere until the consent expires (`CONSENT_SECONDS` in
  `qmcp.instructions.act`): joe's page, or `uv run qmcp human respond <request
  id> approve`. `uv run qmcp human list` shows it.
- **An act refuses for want of a clone.** The conversation looks for a clone
  named for the project beside this checkout, or in `--clones`; by command,
  pass `--cwd` once. The project remembers it either way.
- **"Recorded, no project."** The name heard is not on the roster, or the
  `governance/qm` submodule is not checked out: `git submodule update --init`.
- **Nothing is said, or the conversation seems to have gone.** `curl
  http://127.0.0.1:3141/v1/human/voice` says whether it is running and the last
  lines it printed: what it is waiting for, or why it ended. Its whole output
  is in the system's temporary directory, as `qmcp-voice-<server pid>.log`.
- **`TypeError: HttpSTT.listen() got an unexpected keyword argument
  'pause_ms'`**, from `cookbook instruct`, `cookbook converse` or a spoken
  instruction, while `cookbook voice` passes. The `vox` submodule is not at
  this checkout's pin, usually after a branch switch: `git submodule update
  --init`.
- **The page's voice buttons say a conversation is running.** The standing
  conversation holds the microphone; speak to it instead, or start the server
  without `--converse`.
- **The page says qmcp cannot be reached.** Start `uv run qmcp serve`; joe's
  dev server reaches it at `http://localhost:3141`, and `QMCP_URL` moves it.
