# Voice Integration

Speech reaches qmcp three ways, over one seam. A pending human-in-the-loop
request can be answered by speaking, instead of by typing `qmcp human
respond`: the prompt is spoken aloud, the reply is transcribed, and the
transcribed answer is submitted to the same HITL API a typed answer goes
through — so the audit trail does not distinguish the two except by
`responded_by`. An instruction can be spoken, recorded against a project, and
acted on behind consent asked aloud ("An instruction is recorded, not run" and
the sections after it). And one standing conversation does both, so that once
two servers start nothing is typed ("Talking to qmcp").
`docs/voice-loop-demo.md` is the loop's onboarding and cookbook.

## Overview

| QMCP provides | vox provides |
|---|---|
| The HITL queue, its API and its audit trail | The speech seam: an engine contract, a client, and synthesis |
| `VoiceApprovalLoop`, which turns a transcript into a response | `EngineContract`, `HttpSTT`, `VoiceSession` |
| The retry budget and the yes/no parse | The engine contract and a deterministic stand-in for it |
| The instruction inbox, the act behind consent, and the standing conversation | The same client and synthesizer, for every take and every sentence |

Speech-to-text is not performed locally. `vox.HttpSTT` is an HTTP client
against whatever engine an `EngineContract` describes, and it is that engine's
machine whose microphone is used. vox names no engine; `vox.adapters.joe` is
the contract for the one used here. Text-to-speech is local and offline.

```
prompt --> vox TTS --> speaker
                                 microphone --> speech engine --> transcript
                                                                      |
                                                     parse_yes_no  <--+
                                                                      |
                                              qmcp HITL response  <---+
```

## Installation

vox is vendored as a git submodule at `vendor/vox` and pinned deliberately rather
than floated:

```bash
git submodule update --init vendor/vox
uv sync --all-extras
```

vox and `pyttsx3` are **default dependencies**, not an extra: any `uv sync`
installs them, and none removes them. (They were an extra once, and a plain
`uv sync` then stripped them as extraneous mid-session.) `--all-extras`
rather than `--extra dev` for the rest: the latter omits `pydantic-ai` and
other extras, and the resulting ImportError in unrelated tests reads as a
code regression rather than as a missing dependency. The submodule line
comes first either way: `vox` is a path dependency, and a sync on a fresh
clone exits 2 while the directory is empty.

On Windows, a server started with `uv run qmcp serve` holds
`Scripts/qmcp.exe`, so a sync that has to reinstall qmcp fails with *os
error 32* while it runs. That happens after pulling a change to qmcp's own
dependencies, when the server needs restarting anyway: stop it, sync, and
start it again.

`qmcp.integrations.voice` is structurally typed against vox's shape and does
not import it, so qmcp imports even where vox is absent. Only running
`VoiceApprovalLoop` against real backends requires it.

## Usage

```python
from qmcp.integrations.voice import VoiceApprovalLoop
from vox import HttpSTT
from vox.adapters import JOE
from vox.adapters.pyttsx3 import Pyttsx3TTS

stt = HttpSTT("http://127.0.0.1:8000", contract=JOE)
loop = VoiceApprovalLoop(stt=stt, tts=Pyttsx3TTS())
loop.run_once("deploy-001")   # answer one pending request
loop.run_forever()            # keep answering as new ones arrive
```

From the CLI, alongside `qmcp human list` and `qmcp human respond`:

```bash
uv run qmcp human voice deploy-001     # answer that request
uv run qmcp human voice                # answer whatever is oldest and pending
uv run qmcp human voice --forever      # keep answering; Ctrl+C to stop
uv run qmcp human voice --engine joe   # which vox.adapters entry to talk to
```

The engine must be reachable. `vox doctor` reports which of the three
preconditions is missing — engine unreachable, no microphone on the engine's
machine, or synthesis failing — rather than failing partway through a
recording.

A page can start the same thing over HTTP, which is how joe's front end
offers "Answer by voice" for what is waiting:

| Route | What it does |
|---|---|
| `POST /v1/human/requests/{id}/voice` | asks that request aloud on this machine; `202` once started, `404` for no such request, `409` for one not waiting or while another conversation runs |
| `GET /v1/human/voice` | whether a conversation is running, and how the last one ended: its exit code and the last lines it printed |

The conversation runs as `qmcp human voice <id>` in a process of its own, so
it carries the same preflight and messages as the command, and the
synthesizer gets a process's main thread, which it needs on Windows. One runs
at a time. `QMCP_VOICE_ENGINE` and `QMCP_VOICE_ENGINE_URL` choose the engine.
Both routes are served only when the server is bound to loopback: a caller
elsewhere has no business making this machine speak and listen.

**Typed to answer by voice: two commands**, one per server — `uv run joe dev`
in joe's checkout and `uv run qmcp serve` here. Everything after that happens
in joe's page. **Typed to talk to qmcp: the same two**, with `--converse` on the
second; everything after that is spoken ("Talking to qmcp", below). The count is
kept here because a rise without a stated reason is a regression
(`governance/qm/records/DRAFT-clis-are-for-machines-and-debugging.md`).

## Which microphone

Recording happens on the engine's machine, and its default input is often
not a microphone. One workstation here lists twenty inputs across four host
APIs, with the same microphone appearing four times under a byte-identical
name, and the default is a capture card.

One command, in the engine's own checkout — the engine owns its microphone:

```bash
uv run joe voice setup
```

It counts down and tries every input while you keep talking. Loudness picks
the microphone; among that microphone's entries, the host API decides. WDM-KS
bypasses the system mixer and so reads loudest, and it is also where inputs
open and return garbage, so it is tried last. It finishes by transcribing a
sentence you say, and saves only an entry that recorded it, moving on to the
next entry that heard you when one cannot open.

The choice is saved in the engine's checkout and read at record time, so a
running backend uses it on its next recording with no restart and no
environment variable. `uv run joe voice devices` marks it; `JOE_INPUT_DEVICE`,
where set in the backend's environment, still overrides it.

A name fragment matching several devices is refused rather than guessed.
Devices that open and return samples outside `[-1, 1]` are refused too —
some do, and a level meter reads that as the loudest input on the machine.

## Which voice

The prompt is spoken by a `vox` synthesizer, and there are three:

| backend | audible | needs installing | transcribable |
|---|---|---|---|
| `recording` | no | nothing | only by vox's own engine |
| `formant` | yes | nothing | no |
| `pyttsx3` | yes | a system voice | yes |

`pyttsx3` is what this uses and what a spoken prompt should use: `formant`
is audible but whisper cannot read it, which matters if a loop ever
transcribes its own output. `vox.synth.SPEECH_IS_NOT_TRANSCRIBABLE` carries
the measurement.

## How a spoken answer becomes a response

It is a closed-choice dialog in the shape VoiceXML gives one: the request's
own `options` are the grammar, and the prompt says them — a request carrying
`["approve", "hold"]` is spoken as *"Launch the audit? Say approve or
hold."*

`parse_yes_no` normalizes casing and punctuation before matching, because a
transcript carries both ("Yes.", "Yeah, go ahead."). Negatives are
matched before positives, so "no, don't" is not read as a stray positive.
`choose_option` maps a yes or no onto the options **by reading them**, not by
their position: a request carrying `["reject", "approve"]` records a spoken
"yes" as `approve`.

An answer that is neither yes nor no is matched against the options by name
(`match_option`), so "hold" answers `["approve", "hold"]` and "wait" answers
`["ship it", "wait"]`. Naming none, or more than one, is no match.

The re-ask says which of two things went wrong, up to `max_retries` (default
2): *"I didn't hear anything."* when the transcript is empty (VoiceXML's
noinput), or *"I heard: banana."* when something was heard and was unusable
(nomatch) — echoing the mishearing lets the speaker hear it. Each re-ask
repeats the options. Exhausting the budget raises `UnclearResponse` and
submits nothing: an ambiguous answer is never guessed at.

## Watching the exchange

The loop announces its own states to the STT backend as it goes: `speaking`
before each question or re-ask (a re-ask carries `reason`, `noinput` or
`nomatch`), `recorded` with the option once it is submitted, and `gave_up`
with what was last heard. vox's `HttpSTT.announce` posts them to the engine's
`conversation` route when its contract names one. joe's does, and joe's front
end shows the whole turn live: the question, the open microphone, the person
speaking, the pause, the reading, and the answer. The engine reports the
microphone's states itself.

Announcing is optional on both sides. A backend without `announce`, or an
engine without the route, changes nothing, and an announcement that fails is
dropped rather than stopping the question.

## Bounds and caveats

- **`run_forever` has no upper bound on an idle queue.** With
  `max_iterations` unset it sleeps and polls indefinitely. That is correct at
  a terminal and is not suitable for running unattended.
- **Discovery uses `list_human_requests(status_filter="pending",
  oldest_first=True)`**, which has no side effects. Polling
  `get_human_request` in its place would expire requests as a consequence of
  looking at them. A pending listing leaves out requests past their expiry,
  so nobody is asked a question that can no longer be answered; the server
  lists newest first unless `oldest_first` is passed.
- **A request that already carries a response is returned unchanged.**
  `run_once` does not ask twice.
- **`run_forever` asks each request once per run.** One that gets no usable
  answer stays pending, is not asked again by that run, and is named when
  the loop stops (`VoiceApprovalLoop.unanswered`); the next request is still
  asked. Nobody at the speaker costs one prompt and its re-asks, not the
  same question on repeat. `qmcp human voice <id>` asks it again.
- **`responded_by` is recorded as `vox`.** A voice answer is attributable as a
  voice answer, and is otherwise an ordinary human response.

## An instruction is recorded, not run

Everything above answers a question an agent asked. The inbox runs the other
direction: a person speaks or types an instruction, it is recorded against a
project, and **recording executes nothing**. Recording reaches two statuses,
`recorded` and `unresolved`, and neither describes a run; acting on an
instruction is a command a person issues, behind consent on the human queue,
and is the section after this one. `qmcp.instructions` carries the why, and
`walkthrough/08-an-instruction-is-recorded-not-run.md` runs the routes.

The commands:

```bash
uv run qmcp instruct "Deploy qmcp to the pi."     # typed; prints the row
uv run qmcp instruct --voice                       # spoken
uv run qmcp instructions list [--status unresolved]
uv run qmcp instructions show <id>
```

The project is read from the text by the whole-word match `qmcp threads
consolidate` uses, against the roster in `governance/qm`: a substring inside
another word is not a match, casing is ignored, a name followed by punctuation
still matches, and a hyphenated name matches as a transcript says it, so `rad
godot` is `rad-godot` -- and `rad` too, which leaves that text between the two.
Exactly one match resolves. None or several leaves the row `unresolved` with
the candidates and the rule in `detail`, and `--project` states the project
outright, recorded as `stated`; a blank states nothing.

Spoken, the dialog asks *"What should be done?"*, listens with a long cap and a
long pause (`--duration`, `--pause-ms`; an instruction has pauses mid-thought,
which is what `pause_ms` on the engine contract is for), and reads the
transcript back: *"I heard: Deploy qmcp to the pi. Say record or again."* A yes
or `record` records; a no or `again` listens again; the re-asks are the ones
above. An instruction naming several projects is asked back as a closed choice
by name (*"Which project? Say qmcp or vox."*), where a spoken `rad godot`
chooses `rad-godot` over `rad`; one naming none is asked for the project once.
A text that named its project is sent for the server to read, so the row
carries the match; a project the person chose or spoke is `stated`, with the
answer among the transcripts in `detail.heard`. The states reach the engine's
conversation route as the approval dialog's do, with `confirm` on the read-back
and `again` on a second take.

| Route | What it does |
|---|---|
| `POST /v1/instructions` | records `{text, source, project?, heard?}`; `201` with the row, `recorded` or `unresolved` |
| `GET /v1/instructions?status=` | the inbox, newest first |
| `GET /v1/instructions/{id}` | one row, with its evidence |
| `POST /v1/instructions/voice` | takes one by voice on this machine, as `qmcp instruct --voice` in a process of its own; `202` once started, `409` while any conversation runs |
| `GET /v1/instructions/voice` | whether a conversation is running, its kind, and how the last one ended |

The routes are served only on loopback, as the voice routes are. The spoken
route shares the voice route's tracker, so an approval being asked and an
instruction being taken cannot overlap: there is one microphone.

Without `--runtime`, `uv run qmcp cookbook instruct` is the check, offline: a
server on an ephemeral port over its own database, vox's deterministic engine, and one
scripted dialog per way a spoken instruction can end, through the real path --
one project named and recorded; none named, the project asked for and the
spoken one recorded; several named and chosen by name; and `again`, which takes
the instruction a second time. It also checks that the instruction's take
carried the long pause and the confirmation did not, and that each row says how
its project was settled. `tests/test_cookbook_instruct.py` runs it and makes
sure it can fail.

## Acting on an instruction

```bash
uv run qmcp instructions act <id> --runtime NAME [--budget N] [--cwd PATH] [--voice]
```

A worker takes a recorded instruction, declares what it may spend, asks
consent on the human queue, and runs it in the project's clone only on
`approve`. Every runtime is asked, the local model included; growing a habit
of approval into an automatic one is a later phase. The consent is an ordinary
approval, `instruction-<id>` with the options `approve` and `hold`, so it is
answered wherever approvals are: `qmcp human voice`, `qmcp human respond`, a
page, or in the command itself with `--voice`. Its prompt says the
instruction, the project, the clone, the runtime, the budget and how much
history is carried, and it expires after `CONSENT_SECONDS` in
`qmcp.instructions.act`.

**Continuity comes from qmcp, not the model.** The runtime is handed a brief:
the instruction, the project, the clone, and the project's earlier
instructions that ran, with what each found, read from this server's record
(`qmcp.instructions.continuity`). No runtime resumes a conversation of its
own, so the next instruction can go to a different runtime and still know what
the last one found; the row's `detail` names the turns carried. The clone is
`--cwd` when it is given; without it, the clone the project's last act ran in,
so a path given once is remembered. The standing conversation, failing both,
looks for a clone named for the project ("Talking to qmcp").

`--runtime` has no default (`QMCP_AGENT_RUNTIME` stands in for it). `local` is
the model `qmcp localmodel` stands up on this machine, reading the clone with
tools that cannot write and spending nothing; a coding assistant's command line
is another runtime behind the same contract, given the same brief. A product is
named only in its adapter under `qmcp.integrations.agents.adapters`, and
`scripted` runs nothing and is for checks. `--budget` is runs, and zero -- the
default -- declares and stops. The row's status says where the act got to: `asking`,
then `consented`, `refused` or `unanswered`, then `acting` and `done` or
`failed`; `declared` is written on every path. `qmcp.instructions.act` carries
the why, and `walkthrough/09-nothing-runs-before-consent.md` runs it.

| Route | What it does |
|---|---|
| `POST /v1/instructions/{id}/act` | `{runtime, budget?, cwd?, voice?}`; runs the command in a process of its own, as the spoken route does; `202` once started, `404` for no such instruction, `409` while a conversation or an act runs, `422` for a runtime no adapter declares |

`GET /v1/instructions/voice` reports an act as it reports a conversation, with
`kind: act`.

## The result, spoken

With `--voice`, the act says *"Approved. Running in qmcp."* between the
approval and the run, so the wait is not silence, and when the run ends it
says what the row came to: whether it ran, where, and the outcome's first
sentence -- *"Done in qmcp. Added a health route that answers with the
version; the suite passes. The rest is on the record."* The whole outcome
stays on the row. Without `--voice` the same sentence is printed. A held or
unanswered consent says that nothing ran, and a refusal before anything was
asked says so without reading out the flags it names.

```bash
uv run qmcp instructions say <id> [--speak]   # what an instruction came to, again
```

The summary is announced to the engine's conversation route as `speaking` and
the turn then ends `idle` carrying it, so joe's panel shows it as the last
thing said; it is never `recorded`, which the panel shows as an answer
accepted. `qmcp.instructions.spoken` carries the why.

`uv run qmcp cookbook instruct` is the loop, end to end and offline: after the
inbox's cases it takes one instruction the whole way in one conversation --
spoken and recorded, consent asked aloud and answered `approve`, the
`scripted` runtime run in a directory standing in for the clone, and the
summary said back -- and a second whose consent is answered `hold`, which runs
nothing and says so. Each loop prints as the conversation it was, said and
heard in order, and `tests/test_cookbook_instruct.py` makes sure it can fail.
`--runtime scripted` and `--runtime local` instead take two instructions in
one project, the second answerable only from what qmcp recorded of the first;
`docs/voice-loop-demo.md` says what to look for.

## Talking to qmcp

```bash
uv run joe dev                                  # in joe's checkout: the speech engine
uv run qmcp serve --converse --runtime local    # here: the server, and the conversation
```

Nothing after those two is typed. The conversation starts with the server,
waits for the speech engine in either order, says *"Ready. What should be
done?"*, and from then on is spoken:

| Say | And |
|---|---|
| an instruction | it is read back -- *"I heard: ... Say record or again."* |
| `record` / `again` | it is recorded against its project, or taken again |
| `approve` / `hold` | to the consent -- what will run, where, and how much history is carried; only `approve` runs |
| -- | the runtime carries it out, *"Approved. Running in qmcp."*, and the answer is said back |
| `yes` / `no` | to *"Anything else?"*: asks for the next instruction, or goes back to waiting |
| `stop listening` / `goodbye` | ends the conversation |

Silence is waited through: the microphone stays open, take after take, and the
page shows it listening. Before each instruction, questions agents have put on
the human queue are asked aloud, oldest first, and answered by voice; one
nobody answers stays pending for anyone. A project acted on for the first time
runs in a clone named for it beside this checkout, or in `--clones`, and after
that in the clone its last act ran in, from the record. `--wake WORD` makes an
instruction begin with a word, for a room where people talk; without it every
utterance is read back before anything is recorded, and nothing runs without
`approve`. A turn that fails says so and the conversation goes on.

**What it keeps.** While the conversation runs it listens take after take, and
the speech engine keeps what it records: joe writes every take to `Data/Voice`
in its checkout, as `capture_<time>.wav`, and nothing deletes them. Everything
said near the microphone while the conversation runs stays on that disk until
someone deletes it.

The conversation holds the one conversation this machine can hold, so the
page's **Answer by voice** and **Instruct by voice** answer that one is running
rather than opening the microphone a second time; it ends when the server
stops. `uv run qmcp converse --runtime local` runs it on its own against a
running server, and `--synth recording` writes each sentence to a file instead
of speaking it, for a check on a machine nobody is at. `qmcp.instructions.converse`
carries the why, `uv run qmcp cookbook converse` is one whole session
offline, every take scripted, and the cookbook in `docs/voice-loop-demo.md` is
what to say.

## Testing the integration

`docs/voice-loop-demo.md` is the whole loop's onboarding -- set up once, then
the offline checks, continuity on the local model, and a person at the
microphone -- and its cookbook. This section is the voice-answer check on its
own.

`qmcp cookbook voice` is the check, in two forms that answer different
questions:

```bash
uv run qmcp cookbook voice          # the wiring, with no hardware
uv run qmcp cookbook voice --live   # a person at this machine is heard
```

**Offline**, a qmcp server is started on an ephemeral port over a database
made for the run, and vox's deterministic engine stands in for the speech
engine. Four scripted answers go through the real path: the request created
over HTTP, the prompt synthesized to a file, the answer returned over the
engine contract, and the response submitted and read back. Each ending is
checked against its script:

| heard | the queue afterwards |
|---|---|
| "Yes, go ahead." | `approve`, by `vox`: a yes read onto the options |
| "Hold." | `hold`, by `vox`: an option by name |
| "banana" | nothing; re-asked "I heard: banana." |
| (silence) | nothing; re-asked "I didn't hear anything." |

The configured queue is not touched, and no microphone, speaker or model is
needed. `tests/test_cookbook_voice.py` runs it and makes sure it can fail: a
loop that records the first option for an answer it cannot match turns it red.

**`--live`** asks one question aloud, *"Voice check. Say approve or hold."*,
through the configured server and a running engine, after the same
preflight `human voice` runs. It queues one request, `voice-check-<time>`,
which expires in five minutes, and reports what was recorded.

The deterministic engine is a codec wearing an `EngineContract`. It proves the
wiring and makes no claim about transcription accuracy, which only `--live`
tests. vox's own `uv run vox loop --offline` checks the layer below: the audio
→ text → audio → text round trip, with no queue.
