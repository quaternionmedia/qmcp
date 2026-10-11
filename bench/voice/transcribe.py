"""Every clip, under three conditions, transcribed by whisper under each candidate setting.
Run in joe's environment: uv run --project <joe> python transcribe.py [variants...]"""
import json
import pathlib
import sys
import time

import librosa
import numpy as np
import whisper

HERE = pathlib.Path(__file__).parent
CLIPS = HERE / "clips"
HINTS = {"confirm": ["record", "again"], "consent": ["approve", "hold"],
         "more": ["yes", "no"], "choice": ["qmcp", "vox"], "stop": []}

BASE = dict(language="en", condition_on_previous_text=False, fp16=False)
VARIANTS = {
    "A_default": lambda hint: {},
    "B_english": lambda hint: dict(BASE),
    "C_hint": lambda hint: dict(BASE, initial_prompt=", ".join(hint) + "." if hint else None),
    "D_hint_beam": lambda hint: dict(BASE, initial_prompt=", ".join(hint) + "." if hint else None,
                                     beam_size=5, best_of=5),
}


def conditions(audio: np.ndarray, sr: int):
    yield "clean", audio
    rng = np.random.default_rng(7)
    power = float(np.mean(audio ** 2)) or 1e-9
    noise = rng.normal(0.0, np.sqrt(power / 10 ** (15 / 10)), audio.shape)  # 15 dB SNR
    yield "noise15", (audio + noise).astype("float32")
    peak = float(np.max(np.abs(audio))) or 1.0
    onset = int(np.argmax(np.abs(audio) > 0.05 * peak))
    yield "clipped", audio[onset + int(0.06 * sr):].astype("float32")


chosen = sys.argv[1:] or list(VARIANTS)
model = whisper.load_model("base")
manifest = [m for f in sorted(CLIPS.glob("manifest-*.json")) for m in json.loads(f.read_text(encoding="utf-8"))]
out = HERE / "transcripts.jsonl"
with out.open("a", encoding="utf-8") as fh:
    for item in manifest:
        audio, sr = librosa.load(CLIPS / item["clip"], sr=16000, mono=True)
        for cond, samples in conditions(audio, sr):
            for variant in chosen:
                kwargs = {k: v for k, v in VARIANTS[variant](HINTS[item["question"]]).items() if v is not None}
                started = time.perf_counter()
                text = model.transcribe(samples, **kwargs).get("text", "").strip()
                fh.write(json.dumps({**item, "condition": cond, "variant": variant, "heard": text,
                                     "seconds": round(time.perf_counter() - started, 3)}) + "\n")
        print(item["clip"], flush=True)
print("written", out)
