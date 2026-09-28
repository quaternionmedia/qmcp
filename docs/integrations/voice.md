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

While a server started from this clone's **console script** is running,
Windows will not let a sync replace `Scripts/qmcp.exe`, and the sync dies
partway. A server left running belongs on the module form, which opens no
exe and syncs fine beside it:

```bash
uv run python -m qmcp serve
```

With a console-script server already up, add missing packages without a
sync (`uv pip install -e ./vendor/vox pyttsx3`), or run the CLI in an environment
of its own: `uvx --from . --with ./vendor/vox --with pyttsx3 qmcp human voice ...`.

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

## Bounds and caveats

- **`run_forever` has no upper bound on an idle queue.** With
  `max_iterations` unset it sleeps and polls indefinitely. That is correct at
  a terminal and is not suitable for running unattended.
- **Discovery uses `list_human_requests(status_filter="pending")`**, which has
  no side effects. Polling `get_human_request` in its place would expire
  requests as a consequence of looking at them.
- **A request that already carries a response is returned unchanged.**
  `run_once` does not ask twice.
- **`responded_by` is recorded as `vox`.** A voice answer is attributable as a
  voice answer, and is otherwise an ordinary human response.

## Testing without hardware

The integration's own tests use scripted stand-ins for STT and TTS and touch
no audio device. vox carries the layer below: `uv run vox loop --offline`
closes the full audio → text → audio → text round trip against a deterministic
engine on an ephemeral port, with no engine, no model download and no
microphone.

That deterministic engine is a codec wearing an `EngineContract`. It proves
the seam and makes no claim about transcription accuracy, which is what a real
engine answers and what `uv run vox loop` (without `--offline`) exercises.
