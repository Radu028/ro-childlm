"""Compute ranges over seeds of the checkpoints selected by validation loss (Table 8)."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from .items import write_json


def span(values):
    low, high = min(values), max(values)
    return f"{low:g}" if low == high else f"{low:g}-{high:g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table", type=Path, default=Path("results/appendix/selected-checkpoints.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/appendix/checkpoints.json")
    )
    args = parser.parse_args()

    runs = defaultdict(list)
    with args.table.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            runs[row["protocol"], row["condition"]].append(row)

    report = []
    for (protocol, condition), rows in runs.items():
        assert len(rows) == 5, (protocol, condition)
        report.append(dict(
            protocol=protocol,
            condition=condition,
            epochs=span([float(r["epoch"]) for r in rows]),
            steps=span([int(r["step"]) for r in rows]),
            max_steps=span([int(r["max_step"]) for r in rows]),
        ))
    write_json(args.output, report)

    for r in report:
        print(
            f"{r['protocol']:16s} {r['condition']:24s} epoch {r['epochs']:9s} "
            f"updates {r['steps']:12s} max {r['max_steps']}"
        )


if __name__ == "__main__":
    main()
