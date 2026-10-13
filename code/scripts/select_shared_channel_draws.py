"""Select the draws from channels shared by both age profiles."""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

BANDS = ["0-4", "5-8"]
TARGET, SLACK = 2_000_000, 500
# Shared channels that no original draw used.
UNUSED = {"TraLaLa DSN - Desene Animate ", "Ștrumfii • Română"}

root, out, seeds = Path(sys.argv[1]), Path(sys.argv[2]), [int(s) for s in sys.argv[3:]]
by_channel = {b: defaultdict(list) for b in BANDS}
for b in BANDS:
    for line in (root / b / "transcripts/corpus.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        by_channel[b][r["channel"]].append((r["video_id"], r["clean_words"], line))

# Documents without a recorded channel belong to no channel.
shared = sorted((set(by_channel["0-4"]) & set(by_channel["5-8"])) - {None})
minima = {c: min(sum(w for _, w, _ in by_channel[b][c]) for b in BANDS) for c in shared}
denominator = sum(minima.values())
quota = {c: TARGET * m // denominator for c, m in minima.items()}
print(f"{len(shared)} shared channels, denominator {denominator}", file=sys.stderr)
assert UNUSED <= set(shared), UNUSED - set(shared)

out.mkdir(parents=True, exist_ok=True)
for seed in seeds:
    for b in BANDS:
        rng, picked = random.Random(seed), []
        for c in shared:
            docs = sorted(by_channel[b][c])
            rng.shuffle(docs)
            if c in UNUSED:
                continue

            realized = 0
            for _, words, line in docs:
                if realized >= quota[c]:
                    break
                if realized + words > quota[c] + SLACK:
                    continue
                picked.append(line)
                realized += words

        (out / f"matched-{b}-seed-{seed}.jsonl").write_text(
            "".join(l + "\n" for l in picked), encoding="utf-8"
        )
