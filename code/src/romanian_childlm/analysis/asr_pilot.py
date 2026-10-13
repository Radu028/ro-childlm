"""Compute the caption error rates of the pilot transcription audit (Table 12)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

from .items import write_json

ERRORS = ("feature", "deletion", "lexical", "spelling")
GROUPS = {
    "0-4 automatic": ["0-4/automatic"],
    "0-4 creator": ["0-4/manual"],
    "5-8 automatic": ["5-8/automatic"],
    "5-8 creator": ["5-8/manual"],
    "Automatic, weighted": ["0-4/automatic", "5-8/automatic"],
    "All, weighted": ["0-4/automatic", "0-4/manual", "5-8/automatic", "5-8/manual"],
}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def ratio(windows, strata, weights, errors, words):
    num = sum(weights[h] * sum(errors(w) for w in windows if w["stratum"] == h) for h in strata)
    den = sum(weights[h] * sum(words(w) for w in windows if w["stratum"] == h) for h in strata)
    return num / den


def percentile_ci(values):
    values = sorted(values)
    return [values[math.ceil(0.025 * len(values)) - 1], values[math.ceil(0.975 * len(values)) - 1]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=Path("data/asr-pilot-sample.json"))
    parser.add_argument("--results", type=Path, default=Path("results/appendix"))
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("results/appendix/asr-pilot.json"))
    args = parser.parse_args()

    sample = json.loads(args.sample.read_text())
    weights = {h: v["intervals"] / v["sampled"] for h, v in sample["strata"].items()}
    windows = read_csv(args.results / "asr-pilot-windows.csv")
    for w in windows:
        for key in (
            "words",
            "substitutions",
            "deletions",
            "insertions",
            "names",
            "name_substitutions",
            "name_deletions",
            "agreement_words",
        ):
            w[key] = int(w[key])
    windows.sort(key=lambda w: w["window"])

    # Names the caption spells differently are left out; names it misses count as errors.
    words = lambda w: w["words"] - w["name_substitutions"]
    errors = lambda w: (
        w["substitutions"] + w["deletions"] + w["insertions"] - w["name_substitutions"]
    )
    strata = sorted(weights)

    review = read_csv(args.results / "asr-agreement.csv")
    # Agreement differences per window, so the bootstrap below resamples them with their window.
    found = {
        w["window"]: Counter(r["category"] for r in review if r["window"] == w["window"])
        for w in windows
    }
    bearing_words = lambda w: w["agreement_words"] - found[w["window"]]["uncertain"]
    agreement_errors = lambda w: sum(found[w["window"]][c] for c in ERRORS)
    feature_errors = lambda w: found[w["window"]]["feature"]

    rng = random.Random(42)
    by_stratum = {h: [w for w in windows if w["stratum"] == h] for h in strata}
    boot = {group: {"wer": [], "agreement": [], "feature": []} for group in GROUPS}
    for _ in range(args.resamples):
        resample = [rng.choice(by_stratum[h]) for h in strata for _ in by_stratum[h]]
        for group, members in GROUPS.items():
            boot[group]["wer"].append(ratio(resample, members, weights, errors, words))
            boot[group]["agreement"].append(
                ratio(resample, members, weights, agreement_errors, bearing_words)
            )
            boot[group]["feature"].append(
                ratio(resample, members, weights, feature_errors, bearing_words)
            )

    counts = {h: Counter(r["category"] for r in review if r["stratum"] == h) for h in strata}
    bearing = {
        h: sum(w["agreement_words"] for w in by_stratum[h]) - counts[h]["uncertain"]
        for h in strata
    }

    def agreement(members, categories):
        num = sum(weights[h] * sum(counts[h][c] for c in categories) for h in members)
        return num / sum(weights[h] * bearing[h] for h in members)

    report = {"groups": {}}
    print(
        f"{'stratum':20s} windows  words  WER (95% CI)         "
        "agreement errors (CI)  features (CI)  agreement words"
    )
    for group, members in GROUPS.items():
        low, high = percentile_ci(boot[group]["wer"])
        agreement_ci, feature_ci = (
            percentile_ci(boot[group]["agreement"]),
            percentile_ci(boot[group]["feature"]),
        )
        group_windows = [w for w in windows if w["stratum"] in members]
        wer = ratio(windows, members, weights, errors, words)

        print(
            f"{group:20s} {len(group_windows):7d} {sum(map(words, group_windows)):6,d}  "
            f"{100 * wer:.1f} ({100 * low:.1f}-{100 * high:.1f})      "
            f"{100 * agreement(members, ERRORS):5.1f} "
            f"({100 * agreement_ci[0]:.1f}-{100 * agreement_ci[1]:.1f})  "
            f"{100 * agreement(members, ['feature']):4.1f} "
            f"({100 * feature_ci[0]:.1f}-{100 * feature_ci[1]:.1f})  "
            f"{sum(bearing[h] for h in members):15d}"
        )
        report["groups"][group] = dict(
            windows=len(group_windows),
            words=sum(map(words, group_windows)),
            wer=wer,
            wer_ci95=[low, high],
            agreement_words=sum(bearing[h] for h in members),
            agreement_error_rate=agreement(members, ERRORS),
            agreement_error_ci95=agreement_ci,
            feature_rate=agreement(members, ["feature"]),
            feature_ci95=feature_ci,
        )

    minutes = sum(int(w["end_s"]) - int(w["start_s"]) for w in windows) / 60
    channels = {s["channel"] for s in sample["sample"] if not s["reserve"]}
    print(
        f"\n{len(windows)} windows, {minutes:.1f} minutes, "
        f"{len({w['video_id'] for w in windows})} videos, "
        f"{len(channels)} channels; {sum(w['names'] for w in windows)} name tokens, "
        f"{sum(w['name_substitutions'] for w in windows)} spelled differently and left out"
    )

    report["audio"] = {}
    for audio in ("speech", "singing", "mixed"):
        group = [w for w in windows if w["audio"] == audio]
        rate = sum(map(errors, group)) / sum(map(words, group))
        report["audio"][audio] = dict(windows=len(group), unweighted_wer=rate)
        print(f"{audio}: {len(group)} windows, unweighted WER {100 * rate:.1f}")

    categories = Counter(r["category"] for r in review)
    print(
        f"\n{len(review)} differences on agreement-bearing words: "
        + ", ".join(f"{k} {v}" for k, v in categories.items())
    )
    print(
        f"errors {sum(categories[c] for c in ERRORS)} of {sum(bearing.values())} "
        "agreement-bearing words"
    )

    features = [r for r in review if r["category"] == "feature"]
    word_end = sum(r["word_end"] == "1" for r in features)
    second_person_i = sum(r["second_person_i"] == "1" for r in features)
    print(
        f"of {len(features)} feature changes, {word_end} differ only at the end of the word "
        f"and {second_person_i} add or drop the second-person -i"
    )

    checks = read_csv(args.results / "asr-agreement-checks.csv")
    for kind in ("selected", "other"):
        group = [c for c in checks if c["sample"] == kind]
        print(
            f"{kind} words checked by hand: "
            f"{sum(c['agreement_bearing'] == 'yes' for c in group)} of {len(group)} "
            "agreement-bearing"
        )
    others = sum(w["words"] for w in windows) - sum(w["agreement_words"] for w in windows)
    other_checks = [c for c in checks if c["sample"] == "other"]
    missed = sum(c["agreement_bearing"] == "yes" for c in other_checks)
    print(
        f"about {round(others * missed / len(other_checks))} of the {others} "
        "other words bear agreement"
    )

    report.update(
        minutes=minutes,
        videos=len({w["video_id"] for w in windows}),
        channels=len(channels),
        name_tokens=sum(w["names"] for w in windows),
        names_left_out=sum(w["name_substitutions"] for w in windows),
        agreement_categories=dict(categories),
        feature_changes=dict(
            total=len(features), word_end=word_end, second_person_i=second_person_i
        ),
        checks={
            kind: dict(
                checked=sum(c["sample"] == kind for c in checks),
                agreement_bearing=sum(
                    c["sample"] == kind and c["agreement_bearing"] == "yes" for c in checks
                ),
            )
            for kind in ("selected", "other")
        },
    )
    write_json(args.output, report)


if __name__ == "__main__":
    main()
