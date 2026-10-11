"""Short answers from the demo, synthesized to WAV files with every installed voice at two rates.
Written, never played. Run in qmcp's environment, which has pyttsx3."""
import json
import pathlib
import sys

import pyttsx3

OUT = pathlib.Path(__file__).parent / "clips"
OUT.mkdir(exist_ok=True)

# (spoken text, the question it answers, what the dialog should take it as)
PHRASES = [
    ("record", "confirm", "record"),
    ("again", "confirm", "again"),
    ("approve", "consent", "approve"),
    ("hold", "consent", "hold"),
    ("yes", "consent", "approve"),
    ("yes", "more", "yes"),
    ("no", "more", "no"),
    ("stop listening", "stop", "stop"),
    ("Q M C P", "choice", "qmcp"),
    ("vox", "choice", "vox"),
]
RATES = (160, 230)

engine = pyttsx3.init()
manifest = []
for vi, voice in [(int(sys.argv[1]), engine.getProperty("voices")[int(sys.argv[1])])]:
    short = voice.name.split()[1].lower()
    engine.setProperty("voice", voice.id)
    for rate in RATES:
        engine.setProperty("rate", rate)
        for pi, (text, question, expected) in enumerate(PHRASES):
            name = f"{pi:02d}_{question}_{short}_{rate}.wav"
            engine.save_to_file(text, str(OUT / name))
            manifest.append({"clip": name, "text": text, "question": question,
                             "expected": expected, "voice": short, "rate": rate})
        engine.runAndWait()
(OUT / f"manifest-{vi}.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
print(len(manifest), "clips")
