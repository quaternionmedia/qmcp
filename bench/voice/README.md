# Voice recognition benchmark

How joe's transcription and qmcp's dialogs read synthesized speech: the word
error and whisper's confidence for each take, over short answers,
instruction-length sentences and jargon-heavy instructions, clean, with noise
and clipped.

| Step | Script | Environment | Writes |
|---|---|---|---|
| Synthesize | `synth.py <voice index>`, `synth_instructions.py`, `synth_jargon.py` | qmcp's | WAVs and a manifest in `clips/`, `instructions/`, `jargon/` |
| Transcribe | `transcribe.py [variants...]` | joe's | `transcripts.jsonl` |
| Score | `score.py [transcripts.jsonl]` | qmcp's | standard output |
| Confidence | `confidence.py` | joe's | `confidence.jsonl` |
| Threshold | `threshold.py` | either | standard output |
| Hint | `jargon_hint.py [joe's Data/Voice]` | joe's | standard output |
| Real takes | `real_hint.py <joe's Data/Voice>`, `real_gain.py <joe's Data/Voice>` | joe's | standard output; reads the takes in place |

Each script finds its files beside itself; run it with the environment named,
for example `uv run --project <joe> python transcribe.py`.

The WAV clips are not committed. The synthesis scripts regenerate them with the
voices installed on the machine that runs them, so a run elsewhere measures
other voices. The manifests, `transcripts.jsonl` and `confidence.jsonl` are the
results of the committed run, and the two logs are its output.
