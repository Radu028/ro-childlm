"""Compare BabyBabelLM and captions by where the target form occurs in training."""

from __future__ import annotations

import argparse
from pathlib import Path

from .items import (
    COVERAGE, ITEM_SCORES, BENCHMARK, ItemScores, compare, read_benchmark,
    read_coverage, source_sentences, write_json,
)

GROUPS = ("both", "bbl_only", "captions_only", "neither")
# Primary caption condition, draw condition, BabyBabelLM condition and inputs of each protocol.
# GPT-2 17M also trains on the validation documents, so its inputs are train plus validation.
PROTOCOLS = {
    "gpt2_small_20ep": ("ours-1m-30k", "ours-1m-s{draw}", "babybabellm-off-1m", "train"),
    "gpt2_small_10ep": ("ours-1m-ep10", "ours-1m-s{draw}-ep10", "babybabellm-off-1m-ep10", "train"),
    "gptbert_20ep": ("ours1m-own", "ours1m-own-s{draw}", "bbl-1m", "train"),
    "gpt2_17m": ("ours1m17-d0", "ours1m17-d{draw}", "bbl17", "trainval"),
}


def groups(coverage, draw, inputs):
    out = {group: [] for group in GROUPS}
    for i, row in enumerate(coverage):
        captions = row[f"captions_d{draw}_{inputs}_target"]
        bbl = row[f"bbl_{inputs}_target"]
        key = (
            "both" if captions and bbl
            else "bbl_only" if bbl
            else "captions_only" if captions
            else "neither"
        )
        out[key].append(i)
    return out


def coverage_share(coverage, split):
    return sum(row[f"{split}_words"] for row in coverage) / sum(row["words"] for row in coverage)


def by_group(scores, protocol, captions, coverage, draw, clusters, keep=GROUPS):
    _, _, bbl, inputs = PROTOCOLS[protocol]
    cap, base = scores.mean_correct(protocol, captions), scores.mean_correct(protocol, bbl)
    out = {"all": compare(range(len(coverage)), cap, base, clusters)}
    for group, indices in groups(coverage, draw, inputs).items():
        if group in keep:
            out[group] = compare(indices, cap, base, clusters)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    parser.add_argument("--coverage", type=Path, default=COVERAGE)
    parser.add_argument("--item-scores", type=Path, default=ITEM_SCORES)
    parser.add_argument("--output", type=Path, default=Path("results/appendix"))
    args = parser.parse_args()

    clusters = source_sentences(read_benchmark(args.benchmark))
    coverage = read_coverage(args.coverage)
    scores = ItemScores(args.item_scores)

    primary = {
        "token_coverage": {
            "captions_1m_train": coverage_share(coverage, "captions_d0_train"),
            "bbl_1m_train": coverage_share(coverage, "bbl_train"),
            "combined_10m_train": coverage_share(coverage, "combined_10m_train"),
        },
        "target_groups": {
            group: len(indices) for group, indices in groups(coverage, 0, "train").items()
        },
        "target_in_combined_10m_train": sum(row["combined_10m_train_target"] for row in coverage),
    }
    for protocol in ("gpt2_small_20ep", "gpt2_small_10ep"):
        primary[protocol] = by_group(
            scores, protocol, PROTOCOLS[protocol][0], coverage, 0, clusters
        )
    write_json(args.output / "lexical.json", primary)

    models = {}
    for protocol in ("gptbert_20ep", "gpt2_17m"):
        models[protocol] = {
            "inputs": PROTOCOLS[protocol][3],
            **by_group(scores, protocol, PROTOCOLS[protocol][0], coverage, 0, clusters),
        }
    write_json(args.output / "lexical-models.json", models)

    draws = {}
    for protocol, (_, pattern, _, inputs) in PROTOCOLS.items():
        draws[protocol] = {"inputs": inputs}
        for draw in (1, 2, 3, 4):
            draws[protocol][f"draw-{draw}"] = by_group(
                scores, protocol, pattern.format(draw=draw), coverage, draw, clusters,
                keep=("both",),
            )
    write_json(args.output / "lexical-draws.json", draws)

    print("token coverage:", {k: f"{100 * v:.1f}" for k, v in primary["token_coverage"].items()})
    for name, report in [("primary", primary), ("models", models)]:
        for protocol, out in report.items():
            if protocol.startswith("gpt"):
                for group in ("all", *GROUPS):
                    o = out[group]
                    low, high = o["ci95_source_sentences"]
                    print(
                        f"{protocol:16s} {group:14s} n={o['n']:5d} "
                        f"cap={o['captions']:.1f} bbl={o['bbl']:.1f} "
                        f"d={o['bbl_minus_captions']:+.1f} [{low:.1f}, {high:.1f}]"
                    )

    for protocol, out in draws.items():
        for draw in (1, 2, 3, 4):
            o = out[f"draw-{draw}"]["both"]
            low, high = o["ci95_source_sentences"]
            print(
                f"{protocol:16s} draw {draw} both n={o['n']} "
                f"d={o['bbl_minus_captions']:+.1f} [{low:.1f}, {high:.1f}]"
            )


if __name__ == "__main__":
    main()
