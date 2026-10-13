"""Compare person agreement (SV-P) by the person of the grammatical verb (Appendix B)."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .items import (
    BENCHMARK, ITEM_SCORES, ItemScores, compare, read_benchmark, source_sentences, write_json,
)

MODELS = {
    "gpt2_small_20ep": ("ours-1m-30k", "babybabellm-off-1m"),
    "gpt2_small_10ep": ("ours-1m-ep10", "babybabellm-off-1m-ep10"),
    "gpt2_17m": ("ours1m17-d0", "bbl17"),
    "gptbert_20ep": ("ours1m-own", "bbl-1m"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    parser.add_argument("--item-scores", type=Path, default=ITEM_SCORES)
    parser.add_argument(
        "--output", type=Path, default=Path("results/appendix/person-models.json")
    )
    args = parser.parse_args()

    rows = read_benchmark(args.benchmark)
    clusters = source_sentences(rows)
    scores = ItemScores(args.item_scores)
    person = [i for i, r in enumerate(rows) if r["phenomenon"] == "SV-P"]
    groups = {
        "third": [i for i in person if rows[i]["grammatical_feature"] == "3"],
        "first_second": [i for i in person if rows[i]["grammatical_feature"] in {"1", "2"}],
        "third_vs_first": [
            i for i in person
            if (rows[i]["grammatical_feature"], rows[i]["ungrammatical_feature"]) == ("3", "1")
        ],
        "third_vs_second": [
            i for i in person
            if (rows[i]["grammatical_feature"], rows[i]["ungrammatical_feature"]) == ("3", "2")
        ],
    }
    assert (len(groups["third"]), len(groups["first_second"])) == (800, 214)

    report = {
        "subjects": {
            group: dict(Counter(rows[i]["child_upos"] for i in groups[group]))
            for group in ("third", "first_second")
        }
    }
    for model, (captions, bbl) in MODELS.items():
        pc, pb = scores.mean_correct(model, captions), scores.mean_correct(model, bbl)
        report[model] = {
            group: compare(indices, pc, pb, clusters) for group, indices in groups.items()
        }
    write_json(args.output, report)

    print("subject parts of speech:", report["subjects"])
    for model in MODELS:
        for group, o in report[model].items():
            low, high = o["ci95_source_sentences"]
            print(
                f"{model:16s} {group:15s} n={o['n']:4d} "
                f"cap={o['captions']:5.1f} bbl={o['bbl']:5.1f} "
                f"d={o['bbl_minus_captions']:+5.1f} [{low:.1f}, {high:.1f}]"
            )


if __name__ == "__main__":
    main()
