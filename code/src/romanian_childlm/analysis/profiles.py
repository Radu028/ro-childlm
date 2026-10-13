"""Print the t-tests comparing discovery profiles, Table 7 and the punctuation pipeline test."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

FULL = {
    "GPT-2 small, 20 ep": ("summary.jsonl", "kids04-5m-nopunct", "kids58-5m-nopunct"),
    "GPT-2 small, 10 ep": ("summary.jsonl", "kids04-5m-ep10", "kids58-5m-ep10"),
    "GPT-2 17M, 10 ep": ("gpt2-17m-summary.jsonl", "kids04-5m17", "kids58-5m17"),
    "GPT-BERT, 20 ep": ("gptbert-summary.jsonl", "kids04-5m-ctok", "kids58-5m-ctok"),
}
SHARED = {
    "GPT-2 small, 20 ep": ("summary.jsonl", "matched04-d{draw}", "matched58-d{draw}"),
    "GPT-2 small, 10 ep": ("summary.jsonl", "matched04-d{draw}-ep10", "matched58-d{draw}-ep10"),
    "GPT-2 17M, 10 ep": ("gpt2-17m-summary.jsonl", "matched04-d{draw}-17", "matched58-d{draw}-17"),
    "GPT-BERT, 20 ep": ("gptbert-summary.jsonl", "gb-matched04-d{draw}", "gb-matched58-d{draw}"),
    "GPT-BERT, 10 ep": (
        "gptbert-summary.jsonl", "ep10-gb-matched04-d{draw}", "ep10-gb-matched58-d{draw}"
    ),
}


def by_seed(path):
    scores = defaultdict(dict)
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                scores[row["condition"]][row["seed"]] = 100 * row["accuracy"]
    return scores


def main():
    from scipy.stats import ttest_1samp, ttest_rel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    cache = {}
    read = lambda name: cache.setdefault(name, by_seed(args.results / name))

    print("Full-size profiles, 0-4 minus 5-8, paired by seed")
    for label, (summary, young, old) in FULL.items():
        a, b = read(summary)[young], read(summary)[old]
        seeds = sorted(a)
        test = ttest_rel([a[s] for s in seeds], [b[s] for s in seeds])
        diff = statistics.mean(a[s] - b[s] for s in seeds)
        print(
            f"  {label:18s} {diff:+.2f}  "
            f"t({len(seeds) - 1}) = {test.statistic:.2f}, p = {test.pvalue:.3f}"
        )

    print("\nShared channels (Table 7)")
    print(f"  {'protocol':18s} {'0-4':>6s} {'5-8':>6s} {'mean d':>7s}  d range          p")
    for label, (summary, young, old) in SHARED.items():
        scores = read(summary)
        means = [
            (
                statistics.mean(scores[young.format(draw=d)].values()),
                statistics.mean(scores[old.format(draw=d)].values()),
            )
            for d in range(5)
        ]
        assert all(
            len(scores[young.format(draw=d)]) == len(scores[old.format(draw=d)]) == 5
            for d in range(5)
        )
        deltas = [a - b for a, b in means]
        p = ttest_1samp(deltas, 0).pvalue
        print(
            f"  {label:18s} {statistics.mean(a for a, _ in means):6.2f} "
            f"{statistics.mean(b for _, b in means):6.2f} "
            f"{statistics.mean(deltas):+7.2f}  [{min(deltas):+.2f}, {max(deltas):+.2f}]  {p:.2f}"
        )

    scores = read("summary.jsonl")
    kept, stripped = scores["combined-10m"], scores["combined-10m-nopunct"]
    seeds = sorted(kept)
    test = ttest_rel([kept[s] for s in seeds], [stripped[s] for s in seeds])
    print(
        f"\nCombined 10M GPT-2 small, punctuation kept minus removed: "
        f"{statistics.mean(kept[s] - stripped[s] for s in seeds):+.2f}, "
        f"t(4) = {test.statistic:.2f}, p = {test.pvalue:.2f}"
    )


if __name__ == "__main__":
    main()
