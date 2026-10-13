"""Print Table 9 and the other Appendix F numbers from retraining.json."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

SEEDS = (0, 42, 123, 321, 9000)
T_975_DF4 = 2.776445105
TABLE = [
    ("Captions 1M, 20 ep.", "ours-1m-30k"),
    ("BabyBabelLM 1M, 20 ep.", "babybabellm-off-1m"),
    ("Captions 1M, 10 ep.", "ours-1m-ep10"),
    ("BabyBabelLM 1M, 10 ep.", "babybabellm-off-1m-ep10"),
    ("Ages 0-4 (5M), 20 ep.", "kids04-5m-nopunct"),
    ("Ages 5-8 (5M), 20 ep.", "kids58-5m-nopunct"),
]
FIXED_BUDGET = ("scale-1m-fc", "scale-5m-fc")


def mean_sd(values):
    return f"{statistics.mean(values):.1f} +/- {statistics.stdev(values):.1f}"


def paired(values):
    mean = statistics.mean(values)
    half = T_975_DF4 * statistics.stdev(values) / len(values) ** 0.5
    return f"{mean:+.2f} [{mean - half:.2f}, {mean + half:.2f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results", type=Path, nargs="?", default=Path("results/appendix/retraining.json")
    )
    args = parser.parse_args()

    runs = json.loads(args.results.read_text())

    def curves(condition):
        return {
            s: {e: 100 * a for e, _, a, _ in runs[f"{condition}-seed-{s}"]["curve"]}
            for s in SEEDS
        }

    def selected(condition):
        return [100 * runs[f"{condition}-seed-{s}"]["sel"] for s in SEEDS]

    def original(condition):
        return [100 * runs[f"{condition}-seed-{s}"]["ref"] for s in SEEDS]

    # "ref" is the original run's accuracy at its own checkpoint selected by validation loss.
    print("mean selected accuracy over seeds: retrained, original, difference")
    differences = {}
    for condition in [c for _, c in TABLE] + list(FIXED_BUDGET):
        new, old = statistics.mean(selected(condition)), statistics.mean(original(condition))
        differences[condition] = new - old
        print(f"  {condition:24s} {new:5.2f} {old:5.2f} {new - old:+.2f}")
    largest = max(differences, key=lambda c: abs(differences[c]))
    print(f"largest absolute difference {abs(differences[largest]):.2f} ({largest})")

    print(
        "\ncondition                ep1   ep3   ep5  ep10  ep15  ep20  sel.  "
        "peak epoch of the mean"
    )
    for label, condition in TABLE:
        c = curves(condition)
        means = {e: statistics.mean(c[s][e] for s in SEEDS) for e in c[SEEDS[0]]}
        cells = " ".join(
            f"{means[e]:5.1f}" if e in means else "   --" for e in (1, 3, 5, 10, 15, 20)
        )
        print(
            f"{label:24s} {cells} {statistics.mean(selected(condition)):5.1f}  "
            f"{max(means, key=means.get)}"
        )

    for captions, bbl in [
        ("ours-1m-30k", "babybabellm-off-1m"),
        ("ours-1m-ep10", "babybabellm-off-1m-ep10"),
    ]:
        a, b = curves(captions), curves(bbl)
        epochs = sorted(a[SEEDS[0]])
        gaps = {e: statistics.mean(b[s][e] - a[s][e] for s in SEEDS) for e in epochs}
        always = all(b[s][e] > a[s][e] for s in SEEDS for e in epochs)
        sel_gap = [x - y for x, y in zip(selected(bbl), selected(captions))]
        final_gap = [b[s][epochs[-1]] - a[s][epochs[-1]] for s in SEEDS]

        print(
            f"\n{bbl} minus {captions}: mean gap "
            f"{min(gaps.values()):.2f} to {max(gaps.values()):.2f} over epochs; "
            f"ahead at every epoch in every seed: {always}"
        )
        print(
            f"  selected {statistics.mean(sel_gap):.2f}, final {statistics.mean(final_gap):.2f}, "
            f"selected minus final {paired([x - y for x, y in zip(sel_gap, final_gap)])}"
        )

    young, old = curves("kids04-5m-nopunct"), curves("kids58-5m-nopunct")
    epochs = sorted(young[SEEDS[0]])
    gaps = [statistics.mean(young[s][e] - old[s][e] for s in SEEDS) for e in epochs]
    every_seed = all(young[s][e] > old[s][e] for s in SEEDS for e in epochs)
    print(
        f"\n0-4 minus 5-8, mean over seeds: {min(gaps):.2f} to {max(gaps):.2f} over epochs; "
        f"ahead in every seed at every epoch: {every_seed}"
    )

    print("\nfixed update budget, final and validation-selected checkpoints")
    for condition in FIXED_BUDGET:
        last = [runs[f"{condition}-seed-{s}"]["curve"][-1] for s in SEEDS]
        print(
            f"  {condition}: final {mean_sd([100 * a for _, _, a, _ in last])} "
            f"after {sorted({u for _, u, _, _ in last})} updates "
            f"({sorted({e for e, *_ in last})} epochs); selected {mean_sd(selected(condition))}"
        )


if __name__ == "__main__":
    main()
