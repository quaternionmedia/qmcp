"""Jargon-heavy instructions, synthesized to WAVs through SAPI directly (never played):
both installed voices at two rates. Run in qmcp's environment."""
import json
import pathlib

import comtypes.client

OUT = pathlib.Path(__file__).parent / "jargon"
OUT.mkdir(exist_ok=True)
SENTENCES = [
    ("Run preflight in qmcp.", ["preflight", "qmcp"]),
    ("Check the datapoints in joe.", ["datapoints", "joe"]),
    ("Run the cookbook converse check in qmcp.", ["cookbook", "converse", "qmcp"]),
    ("Run the mutations in vox.", ["mutations", "vox"]),
    ("What does the handoff say in qm?", ["handoff", "qm"]),
    ("Run the gates in qm.", ["gates", "qm"]),
    ("Show the transcript segments in joe.", ["transcript", "segments", "joe"]),
    ("Check the estate in qm.", ["estate", "qm"]),
    ("Read the walkthrough in vox.", ["walkthrough", "vox"]),
    ("What does the endpointer do in joe?", ["endpointer", "joe"]),
    ("Which records are drafts in qm?", ["records", "drafts", "qm"]),
    ("Summarise the roadmap in qmcp.", ["roadmap", "qmcp"]),
]
manifest = []
probe = comtypes.client.CreateObject("SAPI.SpVoice")
for vi, token in enumerate(probe.GetVoices()):
    short = token.GetDescription().split()[1].lower()
    for rate in (-1, 2):
        for si, (text, terms) in enumerate(SENTENCES):
            name = f"{si:02d}_{short}_{rate:+d}.wav"
            voice = comtypes.client.CreateObject("SAPI.SpVoice")
            voice.Voice = token
            voice.Rate = rate
            stream = comtypes.client.CreateObject("SAPI.SpFileStream")
            stream.Open(str(OUT / name), 3)
            voice.AudioOutputStream = stream
            voice.Speak(text, 0)
            stream.Close()
            manifest.append({"clip": name, "text": text, "terms": terms, "voice": short, "rate": rate})
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
print(len(manifest), "clips")
