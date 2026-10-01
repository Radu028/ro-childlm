"""Evaluate a GPT-2 run with its training-text normalization.

Use the BabyBabelLM causal scoring protocol. Ties count as errors and are
also reported separately.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .babyscale_gpt2 import (
    canonical_json_sha256,
    normalize_derived_text,
    sha256_file,
    write_json,
)
from .scoring import read_multiblimp


def summarize_pairs(
    rows: list[dict[str, str]], scores: list[float | None]
):
    """Summarize scores ordered as grammatical, ungrammatical, and so on."""
    details = []
    by_phenomenon = defaultdict(
        lambda: {"correct": 0, "ties": 0, "pairs": 0}
    )
    deltas = []
    skipped = 0
    ties = 0
    for index, row in enumerate(rows):
        grammatical = scores[index * 2]
        ungrammatical = scores[index * 2 + 1]
        if grammatical is None or ungrammatical is None:
            skipped += 1
            continue
        delta = grammatical - ungrammatical
        tie = delta == 0.0
        correct = delta > 0.0
        ties += tie
        deltas.append(delta)
        bucket = by_phenomenon[row["phenomenon"]]
        bucket["pairs"] += 1
        bucket["correct"] += correct
        bucket["ties"] += tie
        details.append(
            {
                "row": index + 1,
                "phenomenon": row["phenomenon"],
                "sentence": row["sen"],
                "wrong_sentence": row["wrong_sen"],
                "grammatical_log_probability": grammatical,
                "ungrammatical_log_probability": ungrammatical,
                "delta": delta,
                "tie": tie,
                "correct": correct,
            }
        )
    evaluated = len(details)
    if not evaluated:
        raise ValueError("No MultiBLiMP pair could be evaluated")
    summary = {
        "accuracy": sum(detail["correct"] for detail in details) / evaluated,
        "evaluated_pairs": evaluated,
        "skipped_pairs": skipped,
        "ties": ties,
        "mean_delta": sum(deltas) / evaluated,
        "by_phenomenon": {
            phenomenon: {
                "accuracy": bucket["correct"] / bucket["pairs"],
                "pairs": bucket["pairs"],
                "ties": bucket["ties"],
            }
            for phenomenon, bucket in sorted(by_phenomenon.items())
        },
    }
    return summary, details


def load_run_normalization(run: Path, prepared: Path) -> bool:
    """Check the preparation matches the run and read its punctuation setting."""
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    identity = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "identity_sha256"}
    )
    if identity != manifest.get("identity_sha256"):
        raise SystemExit(f"Preparation manifest identity is invalid: {prepared}")
    run_config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    if run_config["preparation_identity"] != manifest["identity_sha256"]:
        raise SystemExit(f"Run {run} was not trained on preparation {prepared}")
    return manifest["normalization"]["punctuation"].startswith("removed")


def evaluate_run(args) -> int:
    import torch
    from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
    from .scoring import _score_sentences

    run = Path(args.run).resolve()
    prepared = Path(args.prepared).resolve()
    benchmark = Path(args.benchmark).resolve()
    strip_punctuation = load_run_normalization(run, prepared)

    rows = read_multiblimp(benchmark)
    sentences = []
    for row in rows:
        for sentence in (row["sen"], row["wrong_sen"]):
            normalized, _ = normalize_derived_text(
                sentence, strip_punctuation=strip_punctuation
            )
            sentences.append(normalized)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    model_dir = run / "best_model"
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    scores = _score_sentences(
        model,
        tokenizer,
        sentences,
        args.batch_size,
        args.max_length,
        device,
        torch,
    )
    summary, details = summarize_pairs(rows, scores)

    training_complete = json.loads(
        (run / "training_complete.json").read_text(encoding="utf-8")
    )
    report = {
        "run": str(run),
        "prepared": str(prepared),
        "benchmark": str(benchmark),
        "benchmark_sha256": sha256_file(benchmark),
        "normalization_strip_punctuation": strip_punctuation,
        "protocol": (
            "lm-eval 0.4.9 empty-context: leading space, BOS-conditioned first "
            "token, summed token log probabilities, strict > comparison"
        ),
        "best_validation_loss": training_complete["best_validation_loss"],
        "multiblimp": summary,
    }
    write_json(run / "multiblimp_eval.json", report)
    with (run / "multiblimp_details.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    print(
        f"run={run.name} accuracy={summary['accuracy']:.4f} "
        f"pairs={summary['evaluated_pairs']} skipped={summary['skipped_pairs']} "
        f"ties={summary['ties']} mean_delta={summary['mean_delta']:.3f}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Romanian MultiBLiMP evaluation for one Baby Scale run"
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser


def main():
    evaluate_run(build_parser().parse_args())


if __name__ == "__main__":
    main()
