"""Print slopes and R^2 of OLS fits of mean accuracy per budget on log10 training words."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def fit(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    intercept = my - slope * mx
    residual = sum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys))
    return slope, 1 - residual / sum((y - my) ** 2 for y in ys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        type=Path,
        nargs="*",
        default=[
            Path("results/figures/scaling_points.csv"),
            Path("results/figures/gptbert_scaling_points.csv"),
        ],
    )
    args = parser.parse_args()

    for path in args.csv:
        runs = defaultdict(lambda: defaultdict(list))
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                runs[row["condition"]][int(row["budget_words"])].append(float(row["accuracy"]))

        for condition, budgets in runs.items():
            xs = [math.log10(b) for b in sorted(budgets)]
            ys = [statistics.mean(budgets[b]) for b in sorted(budgets)]
            slope, r2 = fit(xs, ys)
            print(
                f"{path.name} {condition}: {len(xs)} budgets, "
                f"{slope:.2f} points per tenfold increase, R^2 = {r2:.2f}"
            )


if __name__ == "__main__":
    main()
