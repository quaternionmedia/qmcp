"""Does the projects' jargon in the hint help an instruction's take? Run in joe's environment.

Each synthesized clip is transcribed by whisper's base model with four prompts, built the way
joe's `hint_prompt` builds one (comma-joined, a stop at the end): none, the project names, the
names and jargon within joe's bound of twelve entries, and the names with all the jargon. A
clip scores the share of its expected terms found in the transcript. The real takes (joe's
Data/Voice, read in place) are transcribed with the same prompts, to see nothing regresses.
"""
import json
import math
import pathlib
import re
import sys

import librosa
import whisper
from scipy.io import wavfile

HERE = pathlib.Path(__file__).parent
DATA = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
NAMES = ["qmcp", "joe", "vox", "qm", "dossier", "rad-godot"]
TERMS = ["preflight", "cookbook", "converse", "datapoints", "transcript", "segments", "endpointer",
         "mutations", "walkthrough", "handoff", "gates", "estate", "records", "roadmap"]
PROMPTS = {
    "none": None,
    "names": NAMES,
    "names+jargon<=12": (NAMES + TERMS)[:12],
    "names+all jargon": NAMES + TERMS,
}


def prompt(entries):
    return ", ".join(entries) + "." if entries else None


def norm(text):
    return re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("-", " "))


model = whisper.load_model("base")
manifest = json.loads((HERE / "jargon" / "manifest.json").read_text())
scores = {label: [] for label in PROMPTS}
misses = {label: {} for label in PROMPTS}
for clip in manifest:
    audio, _ = librosa.load(str(HERE / "jargon" / clip["clip"]), sr=16000, mono=True)
    for label, entries in PROMPTS.items():
        options = dict(language="en", condition_on_previous_text=False, fp16=False)
        if entries:
            options["initial_prompt"] = prompt(entries)
        heard = norm(model.transcribe(audio, **options)["text"])
        words = heard.split()
        found = [t for t in clip["terms"] if norm(t).strip() in heard or norm(t).split()[0] in words]
        scores[label].append(len(found) / len(clip["terms"]))
        for t in clip["terms"]:
            if t not in found:
                misses[label][t] = misses[label].get(t, 0) + 1
for label in PROMPTS:
    s = scores[label]
    print(f"{label:<18} terms found {sum(s) / len(s):.0%} over {len(s)} clips; missed {misses[label]}")

if DATA:
    print("\nreal takes:")
    for name in ["21122c94696f/000.wav", "f014d306e86f/000.wav", "21887e312c92/000.wav"]:
        rate, pcm = wavfile.read(DATA / "segments" / name)
        audio = pcm.astype("float32") / 32768.0
        row = []
        for label, entries in PROMPTS.items():
            options = dict(language="en", condition_on_previous_text=False, fp16=False, beam_size=5,
                           best_of=5)
            if entries:
                options["initial_prompt"] = prompt(entries)
            result = model.transcribe(audio, **options)
            lps = [s["avg_logprob"] for s in result.get("segments") or []]
            conf = round(min(math.exp(x) for x in lps), 2) if lps else None
            row.append(f"{label}: {result['text'].strip()!r} ({conf})")
        print(" ", name, "|", " | ".join(row))
