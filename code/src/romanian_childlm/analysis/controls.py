"""Print Table 10 and the comparisons with BabyBabelLM alone."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .items import write_json

SEEDS = (0, 42, 123, 321, 9000)


def crossed(deltas):
    """Grand mean and 95% interval of a complete seed x order grid of deltas."""
    from scipy.stats import t

    n = len(SEEDS)
    grand = statistics.mean(deltas.values())
    rows = {s: statistics.mean(deltas[s, c] for c in SEEDS) for s in SEEDS}
    columns = {c: statistics.mean(deltas[s, c] for s in SEEDS) for c in SEEDS}

    ms_seed = n * sum((v - grand) ** 2 for v in rows.values()) / (n - 1)
    ms_order = n * sum((v - grand) ** 2 for v in columns.values()) / (n - 1)
    ms_rest = sum(
        (deltas[s, c] - rows[s] - columns[c] + grand) ** 2 for s in SEEDS for c in SEEDS
    ) / (n - 1) ** 2

    # n^2 Var(mean) = ms_seed + ms_order - ms_rest when neither component is clipped.
    positive = [ms for ms in (ms_seed, ms_order) if ms > ms_rest]
    terms = [(1, ms, n - 1) for ms in positive] + [(1 - len(positive), ms_rest, (n - 1) ** 2)]
    total = sum(c * ms for c, ms, _ in terms)
    if total > 0:
        df = total ** 2 / sum((c * ms) ** 2 / f for c, ms, f in terms)
        half = t.ppf(0.975, df) * (total / n ** 2) ** 0.5
    else:  # every cell equal
        df, half = None, 0.0

    return dict(
        mean=grand,
        ci95=[grand - half, grand + half],
        se=(total / n ** 2) ** 0.5,
        df=df,
        seed_variance=max(0, (ms_seed - ms_rest) / n),
        order_variance=max(0, (ms_order - ms_rest) / n),
        residual_variance=ms_rest,
        ms_seed=ms_seed,
        ms_order=ms_order,
        seed_means=rows,
        order_means=columns,
    )


def cells(grid, key, view):
    return {(c["seed"], c["order"]): c["scores"][view] for c in grid[key]["grid"]}


def extra_contrasts(grid):
    out = {}
    for model in ("gpt2", "gptbert"):
        for test in ("keep", "strip"):
            bbl = cells(grid, f"{model}/captions-vs-bbl-keep/test-{test}", "babybabellm-keep")
            captions = cells(grid, f"{model}/captions-vs-bbl-keep/test-{test}", "captions-keep")
            plus = cells(grid, f"{model}/additive/test-{test}", "bbl800-cap800")
            twice = cells(grid, f"{model}/additive/test-{test}", "bbl800-repeat800")

            levels = dict(
                babybabellm_keep=bbl,
                captions_keep=captions,
                bbl800_cap800=plus,
                bbl800_repeat800=twice,
            )
            out[f"{model}/test-{test}"] = dict(
                levels={name: statistics.mean(v.values()) for name, v in levels.items()},
                adding_captions_vs_bbl_alone=crossed({k: plus[k] - bbl[k] for k in bbl}),
                repeating_vs_bbl_alone=crossed({k: twice[k] - bbl[k] for k in bbl}),
            )
    return out


def interval(item):
    low, high = item["ci95"]
    return f"{item['mean']:+.2f} [{low:.2f}, {high:.2f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/appendix"))
    parser.add_argument(
        "--test",
        choices=("keep", "strip"),
        default="keep",
        help="MultiBLiMP form; punctuated (keep) is primary",
    )
    args = parser.parse_args()

    read = lambda name: json.loads((args.results / name).read_text())
    grid, orders = read("crossed-grid.json"), read("document-orders.json")
    equal_updates = {
        "gpt2": read("equal-updates-gpt2.json"),
        "gptbert": read("equal-updates-gptbert.json"),
    }

    # The saved crossed summaries must follow from their cells.
    for key, item in grid.items():
        recomputed = crossed({(c["seed"], c["order"]): c["delta"] for c in item["grid"]})
        assert all(abs(recomputed[k] - item["crossed"][k]) < 1e-9 for k in ("mean", "se")), key
    extra = extra_contrasts(grid)
    write_json(args.results / "crossed-extra-contrasts.json", extra)

    test = args.test
    rows = [
        ("Captions, punctuation kept vs. removed", "Equal words", {
            m: interval(grid[f"{m}/punctuation-captions/test-{test}"]["crossed"])
            for m in ("gpt2", "gptbert")
        }),
        ("", "Equal max. steps", {
            "gpt2": "{:+.2f} [{:.2f}, {:.2f}]".format(
                orders[f"gpt2/fixed-steps/test-{test}"]["delta_pp"],
                *orders[f"gpt2/fixed-steps/test-{test}"]["paired_seed_ci95_pp"],
            ),
            "gptbert": "--",
        }),
        ("BBL + captions vs. BBL alone", "1.6M vs. 0.8M words", {
            m: interval(extra[f"{m}/test-{test}"]["adding_captions_vs_bbl_alone"])
            for m in ("gpt2", "gptbert")
        }),
        ("BBL + captions vs. BBL twice", "Equal words", {
            m: interval(grid[f"{m}/additive/test-{test}"]["crossed"])
            for m in ("gpt2", "gptbert")
        }),
        ("", "Equal max. updates", {
            m: interval(equal_updates[m][f"test-{test}"]["captions - repetition/equal_tokens"])
            for m in ("gpt2", "gptbert")
        }),
    ]

    print(f"MultiBLiMP with punctuation {test}; points, 95% intervals")
    print(f"{'comparison':40s} {'budget':20s} {'GPT-2 small':22s} GPT-BERT")
    for comparison, budget, values in rows:
        print(f"{comparison:40s} {budget:20s} {values['gpt2']:22s} {values['gptbert']}")

    for model in ("gpt2", "gptbert"):
        levels = extra[f"{model}/test-{test}"]["levels"]
        words = equal_updates[model][f"test-{test}"]["captions - repetition/equal_words"]["mean"]
        print(
            f"{model}: BabyBabelLM 800k alone {levels['babybabellm_keep']:.2f}; "
            f"same seeds and orders at equal words {words:+.2f}"
        )


if __name__ == "__main__":
    main()
