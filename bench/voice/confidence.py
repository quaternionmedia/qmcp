"""Each instruction clip, clean, with noise and clipped, transcribed as joe now transcribes a take;
records whisper's per-segment confidence and the word error against the sentence.
Run in joe's environment."""
import json
import math
import pathlib
import re

import librosa
import numpy as np
import whisper

HERE = pathlib.Path(__file__).parent
CLIPS = HERE / "instructions"


def words(text):
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r)][len(h)] / max(1, len(r))


def conditions(audio, sr):
    yield "clean", audio
    rng = np.random.default_rng(7)
    power = float(np.mean(audio ** 2)) or 1e-9
    yield "noise15", (audio + rng.normal(0.0, np.sqrt(power / 10 ** 1.5), audio.shape)).astype("float32")
    peak = float(np.max(np.abs(audio))) or 1.0
    onset = int(np.argmax(np.abs(audio) > 0.05 * peak))
    yield "clipped", audio[onset + int(0.06 * sr):].astype("float32")


model = whisper.load_model("base")
rows = []
for item in json.loads((CLIPS / "manifest.json").read_text(encoding="utf-8")):
    audio, sr = librosa.load(CLIPS / item["clip"], sr=16000, mono=True)
    for cond, samples in conditions(audio, sr):
        options = dict(language="en", condition_on_previous_text=False, fp16=False)
        if len(samples) / 16000 <= 3.0:
            options.update(beam_size=5, best_of=5)
        result = model.transcribe(samples, **options)
        segs = result.get("segments") or []
        logprobs = [s["avg_logprob"] for s in segs]
        rows.append({**item, "condition": cond, "heard": result["text"].strip(),
                     "wer": round(wer(item["text"], result["text"]), 3),
                     "conf_min": round(min(math.exp(lp) for lp in logprobs), 3) if logprobs else 0.0,
                     "conf_mean": round(math.exp(sum(logprobs) / len(logprobs)), 3) if logprobs else 0.0,
                     "no_speech": round(max((s["no_speech_prob"] for s in segs), default=1.0), 3)})
    print(item["clip"], flush=True)
(HERE / "confidence.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
print("rows:", len(rows))
