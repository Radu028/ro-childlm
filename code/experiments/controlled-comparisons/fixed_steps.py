"""Train GPT-2 small on the same number of full batches in every epoch."""

import argparse
from pathlib import Path

from romanian_childlm.training import babyscale_gpt2 as core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batches", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    original_dataset, original_batches = core.TokenBlockDataset, core.epoch_batches
    real_blocks = None

    def dataset(path, sequence_length, torch):
        nonlocal real_blocks
        result = original_dataset(path, sequence_length, torch)
        if path.resolve() == (args.prepared / "tokens/train.uint16").resolve():
            real_blocks = result.block_count
            assert args.batches * 8 <= real_blocks
            result.block_count = args.batches * 8  # what the trainer counts updates from
        return result

    def batches(block_count, batch_size, seed, epoch, torch):
        assert block_count == args.batches * 8 and batch_size == 8
        full = [
            b for b in original_batches(real_blocks, batch_size, seed, epoch, torch)
            if len(b) == batch_size
        ]
        assert len(full) >= args.batches
        return full[:args.batches]

    core.TokenBlockDataset, core.epoch_batches = dataset, batches
    train_args = core.build_parser().parse_args([
        "train",
        "--prepared", str(args.prepared),
        "--output", str(args.output),
        "--seed", str(args.seed),
        "--epochs", "20",
        "--batch-size", "8",
        "--checkpoint-every", "500",
        "--device", args.device,
    ])
    core.run_training(train_args, benchmark=False)


if __name__ == "__main__":
    main()
