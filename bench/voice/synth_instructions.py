"""Instruction-length sentences, synthesized to WAVs through SAPI directly (never played):
both installed voices at two rates. Run in qmcp's environment."""
import json
import pathlib

import comtypes.client

OUT = pathlib.Path(__file__).parent / "instructions"
OUT.mkdir(exist_ok=True)
SENTENCES = [
    "Which file in qmcp says what qmcp is?",
    "In one sentence, what did that file say qmcp is for?",
    "Deploy qmcp to the pi.",
    "Move the vectors from vox into qmcp.",
    "Run the test suite in joe and report what fails.",
    "Summarise the open pull requests in qm.",
    "Rotate the qmcp logs.",
    "What changed in vox since yesterday?",
]
manifest = []
probe = comtypes.client.CreateObject("SAPI.SpVoice")
voices = list(probe.GetVoices())
for vi, token in enumerate(voices):
    short = token.GetDescription().split()[1].lower()
    for rate in (-2, 3):
        for si, text in enumerate(SENTENCES):
            name = f"{si:02d}_{short}_{rate:+d}.wav"
            voice = comtypes.client.CreateObject("SAPI.SpVoice")
            voice.Voice = token
            voice.Rate = rate
            stream = comtypes.client.CreateObject("SAPI.SpFileStream")
            stream.Open(str(OUT / name), 3)
            voice.AudioOutputStream = stream
            voice.Speak(text, 0)
            stream.Close()
            manifest.append({"clip": name, "text": text, "voice": short, "rate": rate})
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
print(len(manifest), "clips")
