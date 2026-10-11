"""The real "camcp" segment and the others with text, transcribed with and without the project
names as the prompt. Reads joe's Data/Voice in place; writes nothing there."""
import math
import pathlib
import sys

import whisper
from scipy.io import wavfile

DATA = pathlib.Path(sys.argv[1])
SEGMENTS = ["21122c94696f/000.wav", "f014d306e86f/000.wav", "21887e312c92/000.wav"]
PROMPTS = {"none": None, "names": "qmcp, vox, joe, qm, dossier, rad-godot.",
           "names, sentence": "Instructions about the projects qmcp, vox, joe, qm, dossier and rad-godot."}
model = whisper.load_model("base")
for name in SEGMENTS:
    rate, pcm = wavfile.read(DATA / "segments" / name)
    audio = pcm.astype("float32") / 32768.0
    print(name)
    for label, prompt in PROMPTS.items():
        options = dict(language="en", condition_on_previous_text=False, fp16=False, beam_size=5, best_of=5)
        if prompt:
            options["initial_prompt"] = prompt
        result = model.transcribe(audio, **options)
        lps = [s["avg_logprob"] for s in result.get("segments") or []]
        conf = round(min(math.exp(x) for x in lps), 2) if lps else None
        print(f"   {label:<16}: {result['text'].strip()!r} conf {conf}")
