# Voice Integration

A pending human-in-the-loop request can be answered by speaking, instead of by
typing `qmcp human respond`. The prompt is spoken aloud, the reply is
transcribed, and the transcribed answer is submitted to the same HITL API a
typed answer goes through — so the audit trail does not distinguish the two
except by `responded_by`.

## Overview

| QMCP provides | vox provides |
|---|---|
| The HITL queue, its API and its audit trail | The speech seam: STT and TTS behind one adapter |
| `VoiceApprovalLoop`, which turns a transcript into a response | `VoiceSession`, `JoeSTT`, `Pyttsx3TTS` |
| The retry budget and the yes/no parse | The engine contract and a deterministic stand-in for it |

Speech-to-text is not performed locally. `vox.JoeSTT` is an HTTP client
against a running [joe](https://github.com/quaternionmedia/joe) engine's
`/api/voice/*` endpoints, and it is that engine's machine whose microphone is
used. Text-to-speech is local and offline (`pyttsx3`).

```
prompt --> vox TTS --> speaker
                                 microphone --> joe (whisper) --> transcript
                                                                      |
                                                     parse_yes_no  <--+
                                                                      |
                                              qmcp HITL response  <---+
```

## Installation

vox is vendored as a git submodule at `./vox` and pinned deliberately rather
than floated:

```bash
git submodule update --init vox
uv sync --all-extras
```

`--all-extras` rather than `--extra dev`: the latter omits `pydantic-ai` and
other extras, and the resulting ImportError in unrelated tests reads as a code
regression rather than as a missing dependency.

`qmcp.integrations.voice` is structurally typed against vox's shape and does
not import it, so qmcp imports without the `voice` extra installed. Only
running `VoiceApprovalLoop` against real backends requires it.

## Usage

```python
from qmcp.integrations.voice import VoiceApprovalLoop
from vox import JoeSTT, Pyttsx3TTS

loop = VoiceApprovalLoop(stt=JoeSTT(), tts=Pyttsx3TTS())
loop.run_once("deploy-001")   # answer one pending request
loop.run_forever()            # keep answering as new ones arrive
```

From the CLI, alongside `qmcp human list` and `qmcp human respond`:

```bash
qmcp human voice deploy-001     # answer that request
qmcp human voice                # answer whatever is oldest and pending
qmcp human voice --forever      # keep answering; Ctrl+C to stop
```

A joe engine must be reachable. `vox doctor` reports which of the three
preconditions is missing — engine unreachable, no microphone on the engine's
machine, or local TTS failing to initialize — rather than failing partway
through a recording.

## How a spoken answer becomes a response

`parse_yes_no` normalizes casing and punctuation before matching, because a
whisper transcript carries both ("Yes.", "Yeah, go ahead."). Negatives are
matched before positives, so "no, don't" is not read as a stray positive.

`choose_option` then maps the decision onto the request's own `options` **by
reading them**, not by their position. A request carrying
`["reject", "approve"]` records a spoken "yes" as `approve`. Where no option
is recognisable — `["ship it", "wait"]` — position is the only information
available and the affirmative is taken to be first.

An answer that parses as neither is re-asked, up to `max_retries` (default 2).
Exhausting the budget raises `UnclearResponse` and submits nothing: an
ambiguous answer is never guessed at.

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
engine on an ephemeral port, with no joe, no model download and no microphone.

That deterministic engine is a codec wearing joe's HTTP contract. It proves the
seam and makes no claim about transcription accuracy, which is what a real joe
answers and what `uv run vox loop` (without `--offline`) exercises.
