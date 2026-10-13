"""Print results by condition and by phenomenon from a summary JSONL file."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

DEFAULT_CONDITIONS = (
    "combined-10m-nopunct",
    "kids04-5m-nopunct",
    "kids58-5m-nopunct",
    "babybabellm-off-1m",
)


def mean_and_std(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values) * 100
    std = statistics.stdev(values) * 100 if len(values) > 1 else 0.0
    return mean, std


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("results/summary.jsonl"))
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    args = parser.parse_args()

    overall = defaultdict(list)
    by_phenomenon = defaultdict(
        lambda: defaultdict(list)
    )
    with args.summary.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            condition = row["condition"]
            if condition not in args.conditions:
                continue
            overall[condition].append(row["accuracy"])
            for phenomenon, accuracy in row["by_phenomenon"].items():
                by_phenomenon[phenomenon][condition].append(accuracy)

    missing = [
        condition for condition in args.conditions if condition not in overall
    ]
    if missing:
        raise SystemExit(
            f"Conditions not found in {args.summary}: {', '.join(missing)}"
        )

    header = ["phenomenon", *args.conditions]
    print("\t".join(header))
    for phenomenon in sorted(by_phenomenon):
        cells = [phenomenon]
        for condition in args.conditions:
            mean, std = mean_and_std(by_phenomenon[phenomenon][condition])
            cells.append(f"{mean:.2f} +/- {std:.2f}")
        print("\t".join(cells))

    cells = ["overall"]
    for condition in args.conditions:
        mean, std = mean_and_std(overall[condition])
        cells.append(f"{mean:.2f} +/- {std:.2f}")
    print("\t".join(cells))


if __name__ == "__main__":
    main()
