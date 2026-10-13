"""Print the ranges of draw means and the 0-4 word shares of the 1M caption draws."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

DRAWS = {
    "GPT-2 small, 20 ep., combined": (
        "summary.jsonl",
        ["ours-1m-30k"] + [f"ours-1m-s{k}" for k in range(1, 5)],
    ),
    "GPT-2 small, 10 ep., combined": (
        "summary.jsonl",
        ["ours-1m-ep10"] + [f"ours-1m-s{k}-ep10" for k in range(1, 5)],
    ),
    "GPT-2 17M, combined": (
        "gpt2-17m-summary.jsonl",
        [f"ours1m17-d{k}" for k in range(5)],
    ),
    "GPT-BERT, 20 ep., combined": (
        "gptbert-summary.jsonl",
        ["ours1m-own"] + [f"ours1m-own-s{k}" for k in range(1, 5)],
    ),
    "GPT-2 small, 20 ep., ages 0-4": (
        "summary.jsonl",
        ["kids04-1m-30k"] + [f"kids04-1m-s{k}" for k in range(1, 5)],
    ),
    "GPT-2 small, 20 ep., ages 5-8": (
        "summary.jsonl",
        ["kids58-1m-30k"] + [f"kids58-1m-s{k}" for k in range(1, 5)],
    ),
}


def read_summary(path):
    accuracies = defaultdict(list)
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                accuracies[row["condition"]].append(100 * row["accuracy"])
    return accuracies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--draws", type=Path, default=Path("data/caption-draws.json"))
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=None,
        help="default: data/manifest-0-4.jsonl and data/manifest-5-8.jsonl",
    )
    args = parser.parse_args()
    manifests = args.manifest or [Path("data/manifest-0-4.jsonl"), Path("data/manifest-5-8.jsonl")]

    summaries = {}
    for label, (summary, conditions) in DRAWS.items():
        accuracies = summaries.setdefault(summary, read_summary(args.results / summary))
        means = []
        for condition in conditions:
            assert len(accuracies[condition]) == 5, condition
            means.append(statistics.mean(accuracies[condition]))
        cells = ", ".join(f"{m:.2f}" for m in means)
        print(f"{label}: draw means {min(means):.2f} to {max(means):.2f} ({cells})")

    documents = {}
    for path in manifests:
        with path.open(encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                documents[row["video_id"]] = row

    draws = json.loads(args.draws.read_text())
    shares = []
    for name, draw in draws.items():
        ids = draw["train"] + draw["validation"]
        assert sum(documents[v]["clean_words"] for v in draw["train"]) == draw["train_words"], name
        words = sum(documents[v]["clean_words"] for v in ids)
        young = sum(documents[v]["clean_words"] for v in ids if documents[v]["age_band"] == "0-4")
        shares.append(100 * young / words)
        print(
            f"{name} (selection seed {draw['selection_seed']}): {len(ids)} documents, "
            f"{words:,} words, {100 * young / words:.1f}% from 0-4; "
            f"training split {len(draw['train'])} documents, {draw['train_words']:,} words"
        )

    print(f"0-4 share of words across draws: {min(shares):.1f}% to {max(shares):.1f}%")


if __name__ == "__main__":
    main()
