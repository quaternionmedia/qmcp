# Voice Integration

A pending human-in-the-loop request can be answered by speaking, instead of by
typing `qmcp human respond`. The prompt is spoken aloud, the reply is
transcribed, and the transcribed answer is submitted to the same HITL API a
typed answer goes through — so the audit trail does not distinguish the two
except by `responded_by`.

## Overview

| QMCP provides | vox provides |
|---|---|
| The HITL queue, its API and its audit trail | The speech seam: an engine contract, a client, and synthesis |
| `VoiceApprovalLoop`, which turns a transcript into a response | `EngineContract`, `HttpSTT`, `VoiceSession` |
| The retry budget and the yes/no parse | The engine contract and a deterministic stand-in for it |

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
qmcp human voice deploy-001     # answer that request
qmcp human voice                # answer whatever is oldest and pending
qmcp human voice --forever      # keep answering; Ctrl+C to stop
qmcp human voice --engine joe   # which vox.adapters entry to talk to
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
in joe's page. The count is kept here because a rise without a stated reason is
a regression (`governance/qm/records/DRAFT-clis-are-for-machines-and-debugging.md`).

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

A request with no `options` is an open question, and the transcript is the
answer: *"What should the branch be called?"* is spoken as it is, and what is
heard is read back once as a closed choice — *"I heard: release candidate. Say
record or again."* The read-back goes through the same helpers, so a yes
records and a no listens again; `again` re-speaks the question; silence and a
confirmation that matches neither are re-asked as noinput and nomatch. One
budget covers the dialog: each turn the speaker has to be asked a second time,
whether for the answer or for the confirmation, costs a retry, and exhausting
it raises and submits nothing. An `input` request answered "yes" therefore
records `yes`, not `approve`; a request with options is unaffected.

## Watching the exchange

The loop announces its own states to the STT backend as it goes: `speaking`
before each question or re-ask (a re-ask carries `reason`, `noinput` or
`nomatch`; an open question's read-back carries `confirm`, and the question
re-spoken after `again` carries `again`), `recorded` with the answer once it
is submitted, and `gave_up` with what was last heard. vox's `HttpSTT.announce`
posts them to the engine's `conversation` route when its contract names one.
joe's does, and joe's front end shows the whole turn live: the question, the
open microphone, the person speaking, the pause, the reading, and the answer.
The engine reports the microphone's states itself.

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

## Testing the integration

`qmcp cookbook voice` is the check, in two forms that answer different
questions:

```bash
uv run qmcp cookbook voice          # the wiring, with no hardware
uv run qmcp cookbook voice --live   # a person at this machine is heard
```

**Offline**, a qmcp server is started on an ephemeral port over a database
made for the run, and vox's deterministic engine stands in for the speech
engine. Scripted answers go through the real path: the request created over
HTTP, the prompt synthesized to a file, the answer returned over the engine
contract, and the response submitted and read back. Each ending is checked
against its script:

| heard | the queue afterwards |
|---|---|
| "Yes, go ahead." | `approve`, by `vox`: a yes read onto the options |
| "Hold." | `hold`, by `vox`: an option by name |
| "banana" | nothing; re-asked "I heard: banana." |
| (silence) | nothing; re-asked "I didn't hear anything." |
| "release candidate", then "record" | `release candidate`, by `vox`: an open question, read back and confirmed |

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
