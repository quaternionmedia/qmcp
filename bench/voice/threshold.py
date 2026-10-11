"""For each candidate threshold: how many takes would skip confirmation, and how good those are."""
import json
import pathlib

rows = [json.loads(l) for l in (pathlib.Path(__file__).parent / "confidence.jsonl").read_text().splitlines()]
print(f"{len(rows)} takes; exact {sum(r['wer'] == 0 for r in rows)}")
for key in ("conf_min", "conf_mean"):
    print(f"\nby {key}:")
    print(f"{'threshold':>9} {'above':>6} {'exact':>6} {'wrong':>6} {'worst wer':>9} {'below exact':>11}")
    for t in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        above = [r for r in rows if r[key] >= t]
        below = [r for r in rows if r[key] < t]
        wrong = [r for r in above if r["wer"] > 0]
        worst = max((r["wer"] for r in above), default=0)
        print(f"{t:>9} {len(above):>6} {len(above) - len(wrong):>6} {len(wrong):>6} {worst:>9.2f} "
              f"{sum(r['wer'] == 0 for r in below):>11}")
print("\nwrong takes, by conf_min:")
for r in sorted((r for r in rows if r["wer"] > 0), key=lambda r: -r["conf_min"])[:12]:
    print(f"  {r['conf_min']:.3f} wer {r['wer']:.2f} {r['condition']:<8} {r['text']!r} -> {r['heard']!r}")
