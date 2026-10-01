"""Prepare tokenizer and token shards for GPT-BERT training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"] + [
    f"<special_{index}>" for index in range(11)
]


def read_texts(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line)["text"] for line in file if line.strip()]


def build_tokenizer(texts: list[str], vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        texts,
        BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS),
    )
    return tokenizer


def write_shards(
    texts: list[str], tokenizer: Tokenizer, output: Path, shard_bytes: int
) -> tuple[int, int]:
    output.mkdir(parents=True, exist_ok=True)
    tokens_per_shard = max(
        128, shard_bytes // torch.tensor([], dtype=torch.long).element_size()
    )
    eos_id = tokenizer.token_to_id("</s>")
    buffer = []
    shard_count = 0
    token_count = 0

    def save(tokens: list[int]) -> None:
        nonlocal shard_count, token_count
        torch.save(
            torch.tensor(tokens, dtype=torch.long),
            output / f"shard_{shard_count:03d}.bin",
        )
        shard_count += 1
        token_count += len(tokens)

    for text in texts:
        buffer.extend(tokenizer.encode(text).ids)
        buffer.append(eos_id)
        while len(buffer) >= tokens_per_shard:
            save(buffer[:tokens_per_shard])
            buffer = buffer[tokens_per_shard:]
    if buffer:
        save(buffer)
    return shard_count, token_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-from", type=Path)
    parser.add_argument("--vocab-size", type=int, default=16_384)
    parser.add_argument("--shard-bytes", type=int, default=100_000_000)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output}")

    train_texts = read_texts(args.prepared / "splits" / "train.jsonl")
    validation_texts = read_texts(args.prepared / "splits" / "validation.jsonl")
    args.output.mkdir(parents=True, exist_ok=True)

    tokenizer = (
        Tokenizer.from_file(str(args.tokenizer_from))
        if args.tokenizer_from
        else build_tokenizer(train_texts, args.vocab_size)
    )
    tokenizer_path = args.output / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    train_shards, train_tokens = write_shards(
        train_texts, tokenizer, args.output / "train", args.shard_bytes
    )
    validation_shards, validation_tokens = write_shards(
        validation_texts,
        tokenizer,
        args.output / "valid",
        args.shard_bytes,
    )
    metadata = {
        "vocab_size": args.vocab_size,
        "train_shards": train_shards,
        "train_tokens": train_tokens,
        "validation_shards": validation_shards,
        "validation_tokens": validation_tokens,
        "eos_id": tokenizer.token_to_id("</s>"),
        "sequence_length": 128,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
