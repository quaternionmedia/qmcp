"""The real segments in joe's Data/Voice/segments, transcribed as recorded and peak-normalised.
Reads them in place and writes nothing there. Run in joe's environment."""
import json
import math
import pathlib
import sys

import numpy as np
import whisper
from scipy.io import wavfile

DATA = pathlib.Path(sys.argv[1])
model = whisper.load_model("base")
records = {(r["take"], r["index"]): r for r in map(json.loads, (DATA / "segments.jsonl").read_text(encoding="utf-8").splitlines())
           if r["kind"] == "segment"}


def decode(audio):
    options = dict(language="en", condition_on_previous_text=False, fp16=False)
    if len(audio) / 16000 <= 3.0:
        options.update(beam_size=5, best_of=5)
    result = model.transcribe(audio, **options)
    lps = [s["avg_logprob"] for s in result.get("segments") or []]
    return result["text"].strip(), (round(min(math.exp(x) for x in lps), 2) if lps else None)


for (take, index), r in sorted(records.items(), key=lambda kv: kv[1]["at"]):
    rate, pcm = wavfile.read(DATA / r["audio"])
    audio = pcm.astype("float32") / 32768.0
    peak = float(np.max(np.abs(audio))) or 1.0
    raw = decode(audio)
    normalised = decode((audio * (0.5 / peak)).astype("float32"))
    print(f"{take}/{index} {r['duration_s']}s peak {peak:.4f}")
    print(f"   as recorded : {raw[0]!r} conf {raw[1]}")
    print(f"   normalised  : {normalised[0]!r} conf {normalised[1]}")
