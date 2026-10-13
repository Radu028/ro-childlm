"""Build data/benchmark-coverage.csv from the training splits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from ..training.babyscale_gpt2 import normalize_derived_text, read_strict_corpus
from .items import BENCHMARK, read_benchmark


def words(text):
    return [w.lower() for w in normalize_derived_text(text, strip_punctuation=True)[0].split()]


def split_texts(path):
    with path.open(encoding="utf-8") as file:
        return [words(json.loads(line)["text"]) for line in file if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-0-4", type=Path, required=True)
    parser.add_argument("--corpus-5-8", type=Path, required=True)
    parser.add_argument("--draws", type=Path, default=Path("data/caption-draws.json"))
    parser.add_argument(
        "--bbl-prepared",
        type=Path,
        required=True,
        help="BabyBabelLM preparation of code/README.md (its splits/ folder is read)",
    )
    parser.add_argument(
        "--combined-prepared",
        type=Path,
        required=True,
        help="Combined 10M preparation with --strip-punctuation",
    )
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    parser.add_argument("--output", type=Path, default=Path("data/benchmark-coverage.csv"))
    args = parser.parse_args()

    documents = {}
    for band, path in [("0-4", args.corpus_0_4), ("5-8", args.corpus_5_8)]:
        for d in read_strict_corpus(path, band, strip_punctuation=True)[0]:
            documents[d.video_id] = [w.lower() for w in d.text.split()]

    splits = {}
    for k, draw in enumerate(json.loads(args.draws.read_text()).values()):
        missing = set(draw["train"] + draw["validation"]) - documents.keys()
        assert not missing, f"{len(missing)} documents of draw {k} are not in the corpus"
        splits[f"captions_d{k}_train"] = [documents[v] for v in draw["train"]]
        splits[f"captions_d{k}_trainval"] = [
            documents[v] for v in draw["train"] + draw["validation"]
        ]
    splits["bbl_train"] = split_texts(args.bbl_prepared / "splits/train.jsonl")
    splits["bbl_trainval"] = splits["bbl_train"] + split_texts(
        args.bbl_prepared / "splits/validation.jsonl"
    )
    splits["combined_10m_train"] = split_texts(args.combined_prepared / "splits/train.jsonl")

    seen = {}
    for name, texts in splits.items():
        vocabulary, bigrams = set(), set()
        for text in texts:
            vocabulary.update(text)
            bigrams.update(zip(text, text[1:]))
        seen[name] = vocabulary, bigrams

    header = ["row", "words", "bigrams"] + [
        f"{name}_{kind}" for name in splits for kind in ("target", "words", "bigrams")
    ]
    rows = []
    for i, pair in enumerate(read_benchmark(args.benchmark), 1):
        good, bad = words(pair["sen"]), words(pair["wrong_sen"])
        target = [a for a, b in zip(good, bad) if a != b]
        assert len(good) == len(bad) and len(target) == 1, i
        pairs = list(zip(good, good[1:]))
        row = [i, len(good), len(pairs)]
        for vocabulary, bigrams in seen.values():
            row += [
                int(target[0] in vocabulary),
                sum(w in vocabulary for w in good),
                sum(p in bigrams for p in pairs),
            ]
        rows.append(row)

    with (sys.stdout if str(args.output) == "-" else args.output.open("w", newline="")) as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
