"""Run the GPT-BERT protocol used in the paper."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, choices=(10, 20), default=20)
    parser.add_argument("--steps-per-epoch", type=positive,
                        help="Override the default of training tokens / (256 x 128), rounded up; "
                             "used by the controls with equal maximum steps in code/experiments")
    args = parser.parse_args()

    metadata = json.loads((args.data / "metadata.json").read_text())
    tokens_per_step = 256 * 128
    steps_per_epoch = (int(metadata["train_tokens"]) + tokens_per_step - 1) // tokens_per_step
    if args.steps_per_epoch is not None:
        steps_per_epoch = args.steps_per_epoch
    total_steps = steps_per_epoch * args.epochs
    vendor = Path(__file__).resolve().parents[1] / "vendor" / "gpt_bert"
    command = [
        sys.executable,
        "train.py",
        "--train_path", str((args.data / "train").resolve()),
        "--valid_path", str((args.data / "valid").resolve()),
        "--tokenizer_path", str((args.data / "tokenizer.json").resolve()),
        "--config_file", str((vendor / "base.json").resolve()),
        "--name", "run",
        "--output_dir", str(args.output.resolve()),
        "--hybrid_numerator", "8",
        "--hybrid_denominator", "16",
        "--seq_length", "128",
        "--local_batch_size", "64",
        "--global_batch_size", "256",
        "--batch_reduction", "1",
        "--learning_rate", "5e-3",
        "--max_steps", str(total_steps),
        "--validate_every", str(steps_per_epoch),
        "--save_every", str(total_steps),
        "--mixed_precision",
        "--seed", str(args.seed),
    ]
    environment = os.environ.copy()
    environment.setdefault("WANDB_MODE", "disabled")
    subprocess.run(command, cwd=vendor, env=environment, check=True)


if __name__ == "__main__":
    main()
