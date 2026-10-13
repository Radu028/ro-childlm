"""Measure the bigram closeness of the benchmark to the 1M training splits (Appendix D)."""

from __future__ import annotations

import argparse
import random
import statistics
from collections import defaultdict
from pathlib import Path

from .items import (
    BENCHMARK, COVERAGE, ITEM_SCORES, ItemScores, read_benchmark, read_coverage,
    source_sentences, write_json,
)

MODELS = {
    "gpt2_small_20ep": ("ours-1m-30k", "babybabellm-off-1m"),
    "gpt2_small_10ep": ("ours-1m-ep10", "babybabellm-off-1m-ep10"),
    "gptbert_20ep": ("ours1m-own", "bbl-1m"),
}


def fit(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - slope * mx, slope


def bootstrap(indices, gaps, clusters, stat, n=5000, seed=0):
    groups = defaultdict(list)
    for i, gap in zip(indices, gaps):
        groups[clusters[i]].append((i, gap))
    keys = list(groups)

    rng = random.Random(seed)
    values = []
    for _ in range(n):
        sample = [pair for key in rng.choices(keys, k=len(keys)) for pair in groups[key]]
        values.append(stat([i for i, _ in sample], [g for _, g in sample]))
    values.sort()
    return values[int(0.025 * n)], values[int(0.975 * n)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    parser.add_argument("--coverage", type=Path, default=COVERAGE)
    parser.add_argument("--item-scores", type=Path, default=ITEM_SCORES)
    parser.add_argument("--output", type=Path, default=Path("results/appendix/closeness.json"))
    args = parser.parse_args()

    clusters = source_sentences(read_benchmark(args.benchmark))
    coverage = read_coverage(args.coverage)
    scores = ItemScores(args.item_scores)

    close_cap = [row["captions_d0_train_bigrams"] / row["bigrams"] for row in coverage]
    close_bbl = [row["bbl_train_bigrams"] / row["bigrams"] for row in coverage]
    diff = [b - c for b, c in zip(close_bbl, close_cap)]
    # Compare integer counts, so a difference of exactly five points counts as equal.
    counts = [
        (row["bbl_train_bigrams"] - row["captions_d0_train_bigrams"], row["bigrams"])
        for row in coverage
    ]
    equal = [i for i, (d, n) in enumerate(counts) if 20 * abs(d) <= n]
    everything = list(range(len(coverage)))

    def intercept(indices, gaps):
        return fit([diff[i] for i in indices], gaps)[0]

    report = {
        "closeness": dict(
            captions_mean=100 * statistics.mean(close_cap),
            bbl_mean=100 * statistics.mean(close_bbl),
            bbl_closer_pairs=sum(20 * d > n for d, n in counts),
            captions_closer_pairs=sum(-20 * d > n for d, n in counts),
            equal_pairs=len(equal),
            equal_source_sentences=len({clusters[i] for i in equal}),
        )
    }
    for model, (captions, bbl) in MODELS.items():
        pc, pb = scores.mean_correct(model, captions), scores.mean_correct(model, bbl)
        gaps = [100 * (b - c) for b, c in zip(pb, pc)]
        equal_gaps = [gaps[i] for i in equal]
        a, slope = fit(diff, gaps)
        report[model] = dict(
            all=dict(gap=statistics.mean(gaps)),
            equal_closeness=dict(
                n=len(equal),
                captions=100 * statistics.mean(pc[i] for i in equal),
                bbl=100 * statistics.mean(pb[i] for i in equal),
                gap=statistics.mean(equal_gaps),
                ci95=bootstrap(equal, equal_gaps, clusters, lambda i, g: statistics.mean(g)),
            ),
            regression=dict(
                intercept=a,
                slope_per_unit=slope,
                intercept_ci95=bootstrap(everything, gaps, clusters, intercept),
            ),
        )
    write_json(args.output, report)

    c = report["closeness"]
    print(
        f"bigram share: captions {c['captions_mean']:.1f}, BabyBabelLM {c['bbl_mean']:.1f}; "
        f"BabyBabelLM closer by >5 points on {c['bbl_closer_pairs']}, equal on {c['equal_pairs']}"
    )
    for model in MODELS:
        e, r = report[model]["equal_closeness"], report[model]["regression"]
        print(
            f"{model}: equal {e['gap']:+.1f} [{e['ci95'][0]:.1f}, {e['ci95'][1]:.1f}]; "
            f"intercept {r['intercept']:+.1f} "
            f"[{r['intercept_ci95'][0]:.1f}, {r['intercept_ci95'][1]:.1f}]; "
            f"all {report[model]['all']['gap']:+.1f}"
        )


if __name__ == "__main__":
    main()
