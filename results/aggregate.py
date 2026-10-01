"""Print mean and sample standard deviation for each condition."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, nargs="?", default=Path("summary.jsonl"))
    args = parser.parse_args()

    accuracies = defaultdict(list)
    with args.summary.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            accuracies[row["condition"]].append(row["accuracy"] * 100)

    for condition, values in sorted(accuracies.items()):
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        print(
            f"{condition}: mean {statistics.mean(values):.2f} "
            f"+/- {std:.2f} (n={len(values)})"
        )


if __name__ == "__main__":
    main()
