"""Compute accuracy by phenomenon and the contributions to the matched 1M gap (Appendix B)."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .items import BENCHMARK, read_benchmark, write_json

PHENOMENA = ("SP-#", "SP-G", "SV-#", "SV-G", "SV-P")
LABELS = ("Combined 10M", "Ages 0-4 (5M)", "Ages 5-8 (5M)", "Captions 1M", "BabyBabelLM 1M")
PROTOCOLS = {
    "gpt2_small_20ep": ("summary.jsonl", [
        "combined-10m-nopunct", "kids04-5m-nopunct", "kids58-5m-nopunct",
        "ours-1m-30k", "babybabellm-off-1m",
    ]),
    "gpt2_small_10ep": ("summary.jsonl", [
        "combined-10m-ep10", "kids04-5m-ep10", "kids58-5m-ep10",
        "ours-1m-ep10", "babybabellm-off-1m-ep10",
    ]),
    "gpt2_17m": ("gpt2-17m-summary.jsonl", [
        "ours10m17", "kids04-5m17", "kids58-5m17",
        "ours1m17-d0", "bbl17",
    ]),
    "gptbert_20ep": ("gptbert-summary.jsonl", [
        "combined-10m", "kids04-5m-ctok", "kids58-5m-ctok",
        "ours1m-own", "bbl-1m",
    ]),
    "gptbert_10ep": ("gptbert-summary.jsonl", [
        "ep10-combined-10m", "ep10-kids04-5m-ctok", "ep10-kids58-5m-ctok",
        None, "ep10-bbl-1m",
    ]),
}


def accuracy(value):
    # The 17M summaries nest the accuracy of each phenomenon.
    return value["accuracy"] if isinstance(value, dict) else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    parser.add_argument("--output", type=Path, default=Path("results/appendix/phenomena.json"))
    args = parser.parse_args()

    pairs = {
        p: sum(r["phenomenon"] == p for r in read_benchmark(args.benchmark)) for p in PHENOMENA
    }
    total = sum(pairs.values())
    report = {"pairs": pairs}

    for protocol, (summary, conditions) in PROTOCOLS.items():
        with (args.results / summary).open(encoding="utf-8") as file:
            records = [json.loads(line) for line in file if line.strip()]

        rows = {}
        for label, condition in zip(LABELS, conditions):
            if condition is None:
                continue
            runs = [r for r in records if r["condition"] == condition]
            assert len(runs) == 5, (condition, len(runs))
            values = {p: [100 * accuracy(r["by_phenomenon"][p]) for r in runs] for p in PHENOMENA}
            values["all"] = [100 * r["accuracy"] for r in runs]
            values["macro"] = [statistics.mean(v) for v in zip(*(values[p] for p in PHENOMENA))]
            rows[label] = {
                "condition": condition,
                **{
                    key: dict(mean=statistics.mean(v), sd=statistics.stdev(v))
                    for key, v in values.items()
                },
            }

        report[protocol] = {"conditions": rows}
        if "Captions 1M" in rows:
            cap, bbl = rows["Captions 1M"], rows["BabyBabelLM 1M"]
            report[protocol]["matched_1m_gap"] = dict(
                all=bbl["all"]["mean"] - cap["all"]["mean"],
                macro=bbl["macro"]["mean"] - cap["macro"]["mean"],
                contributions={
                    p: pairs[p] / total * (bbl[p]["mean"] - cap[p]["mean"]) for p in PHENOMENA
                },
            )
    write_json(args.output, report)

    for protocol in PROTOCOLS:
        print(protocol)
        for label, row in report[protocol]["conditions"].items():
            cells = " ".join(f"{p} {row[p]['mean']:.1f}" for p in PHENOMENA)
            print(
                f"  {label:15s} {cells}  "
                f"macro {row['macro']['mean']:.2f} +/- {row['macro']['sd']:.2f}  "
                f"all {row['all']['mean']:.2f}"
            )

        gap = report[protocol].get("matched_1m_gap")
        if gap:
            print(
                f"  BabyBabelLM minus captions: all {gap['all']:.2f}, macro {gap['macro']:.2f}, "
                f"SV-P contributes {gap['contributions']['SV-P']:.2f}"
            )


if __name__ == "__main__":
    main()
