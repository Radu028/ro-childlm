"""Count finite verbs by person in the two 1M training splits with Stanza (Appendix B)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ..training.babyscale_gpt2 import read_strict_corpus
from .items import write_json


def count(nlp, texts):
    import stanza

    persons, words = Counter(), 0
    for doc in nlp.bulk_process([stanza.Document([], text=t) for t in texts]):
        for sentence in doc.sentences:
            for word in sentence.words:
                words += word.upos != "PUNCT"
                feats = dict(f.split("=", 1) for f in (word.feats or "").split("|") if "=" in f)
                if (
                    word.upos in {"VERB", "AUX"}
                    and feats.get("VerbForm") == "Fin"
                    and "Person" in feats
                ):
                    persons[word.upos, feats["Person"], feats.get("Number", "")] += 1

    return dict(
        words=words,
        finite_by_upos_person_number={"|".join(k): v for k, v in sorted(persons.items())},
    )


def rates(report):
    for split, counts in report.items():
        by_person = Counter()
        for key, n in counts["finite_by_upos_person_number"].items():
            by_person[key.split("|")[1]] += n
        cells = ", ".join(
            f"person {p}: {1000 * by_person[p] / counts['words']:.1f}" for p in sorted(by_person)
        )
        print(f"{split} ({counts['words']:,} words), finite forms per 1,000 words: {cells}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts", type=Path, help="print the rates of a saved person-corpus.json and stop"
    )
    parser.add_argument("--corpus-0-4", type=Path)
    parser.add_argument("--corpus-5-8", type=Path)
    parser.add_argument("--draws", type=Path, default=Path("data/caption-draws.json"))
    parser.add_argument(
        "--bbl-corpus", type=Path, help="corpus.jsonl from build_babybabellm_official_corpus.py"
    )
    parser.add_argument(
        "--bbl-train", type=Path, help="splits/train.jsonl of the prepared BabyBabelLM corpus"
    )
    parser.add_argument("--stanza-dir", type=Path, help="Stanza resources directory")
    parser.add_argument("--output", type=Path, default=Path("person-corpus.json"))
    args = parser.parse_args()

    if args.counts:
        rates(json.loads(args.counts.read_text()))
        return
    for name in ("corpus_0_4", "corpus_5_8", "bbl_corpus", "bbl_train"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")

    import stanza

    train = set(json.loads(args.draws.read_text())["ours-1m-30k"]["train"])
    captions = []
    for band, path in [("0-4", args.corpus_0_4), ("5-8", args.corpus_5_8)]:
        documents, _ = read_strict_corpus(path, band, strip_punctuation=False)
        captions += [d.text for d in documents if d.video_id in train]
    assert len(captions) == len(train), f"{len(train) - len(captions)} training documents missing"

    with args.bbl_train.open(encoding="utf-8") as file:
        ids = {json.loads(line)["video_id"] for line in file if line.strip()}
    with args.bbl_corpus.open(encoding="utf-8") as file:
        bbl = [row["text"] for row in map(json.loads, file) if row["video_id"] in ids]
    assert len(bbl) == len(ids), (len(bbl), len(ids))

    options = dict(dir=str(args.stanza_dir)) if args.stanza_dir else {}
    nlp = stanza.Pipeline(
        "ro", processors="tokenize,pos", use_gpu=False, logging_level="WARN", **options
    )
    report = {"captions_1m_train": count(nlp, captions), "bbl_1m_train": count(nlp, bbl)}
    write_json(args.output, report)
    rates(report)


if __name__ == "__main__":
    main()
