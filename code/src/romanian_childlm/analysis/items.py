"""Benchmark rows, pair scores and the bootstrap shared by the Appendix B and D analyses."""

from __future__ import annotations

import ast
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

BENCHMARK = Path("data/multiblimp-ron.tsv")
COVERAGE = Path("data/benchmark-coverage.csv")
ITEM_SCORES = Path("results/appendix/item-scores.jsonl")
SEEDS = (0, 42, 123, 321, 9000)


def read_benchmark(path: Path = BENCHMARK) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def source_sentences(rows: list[dict[str, str]]) -> list[tuple[str, str]]:
    metadata = (ast.literal_eval(row["metadata"]) for row in rows)
    return [(meta["treebank"], meta["sent_id"]) for meta in metadata]


def read_coverage(path: Path = COVERAGE) -> list[dict[str, int]]:
    with path.open(encoding="utf-8", newline="") as file:
        return [{key: int(value) for key, value in row.items()} for row in csv.DictReader(file)]


class ItemScores:
    def __init__(self, path: Path = ITEM_SCORES):
        self.runs = {}
        with path.open(encoding="utf-8") as file:
            for line in file:
                run = json.loads(line)
                self.runs[run["protocol"], run["condition"], run["seed"]] = run

    def mean_correct(self, protocol: str, condition: str) -> list[float]:
        runs = [self.runs[protocol, condition, seed]["correct"] for seed in SEEDS]
        return [statistics.mean(values) for values in zip(*runs)]


def bootstrap_ci(indices, values, clusters, n=10_000, seed=0):
    groups = defaultdict(list)
    for i, value in zip(indices, values):
        groups[clusters[i]].append(value)
    keys = list(groups)

    rng = random.Random(seed)
    means = sorted(
        statistics.mean([v for key in rng.choices(keys, k=len(keys)) for v in groups[key]])
        for _ in range(n)
    )
    return means[int(0.025 * n)], means[int(0.975 * n)]


def compare(indices, captions, bbl, clusters):
    indices = list(indices)
    diffs = [bbl[i] - captions[i] for i in indices]
    low, high = bootstrap_ci(indices, diffs, clusters)
    return dict(
        n=len(indices),
        captions=100 * statistics.mean(captions[i] for i in indices),
        bbl=100 * statistics.mean(bbl[i] for i in indices),
        bbl_minus_captions=100 * statistics.mean(diffs),
        ci95_source_sentences=[100 * low, 100 * high],
        source_sentences=len({clusters[i] for i in indices}),
    )


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
