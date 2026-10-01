"""Train the 17M-parameter GPT-2 model with the BabyBabelLM recipe."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, items: list[tuple[list[int], list[int]]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        input_ids, attention_mask = self.items[index]
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def load_texts(corpus_paths: str, split_directory: Path | None) -> list[str]:
    rows = []
    for path in corpus_paths.split(","):
        with Path(path).open(encoding="utf-8") as file:
            rows.extend(json.loads(line) for line in file if line.strip())

    if split_directory is not None:
        selected_ids = set()
        for filename in ("train.jsonl", "validation.jsonl"):
            with (split_directory / "splits" / filename).open(encoding="utf-8") as file:
                selected_ids.update(
                    str(json.loads(line)["video_id"]) for line in file if line.strip()
                )
        rows = [row for row in rows if str(row.get("video_id")) in selected_ids]

    return [str(row["text"]) for row in rows if str(row.get("text") or "").strip()]


def train(args) -> None:
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output}")
    texts = load_texts(args.corpus, args.ids_from)
    if not texts:
        raise SystemExit("No documents loaded; check --corpus and --ids-from")
    print(f"documents={len(texts)}")

    args.output.mkdir(parents=True, exist_ok=True)
    if args.tokenizer_dir is not None:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    else:
        byte_tokenizer = ByteLevelBPETokenizer()
        byte_tokenizer.train_from_iterator(
            texts,
            vocab_size=args.vocab_size,
            min_frequency=2,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=byte_tokenizer._tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
        )
    tokenizer.save_pretrained(args.output)

    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
        return_overflowing_tokens=True,
    )
    chunks = list(zip(encoded["input_ids"], encoded["attention_mask"], strict=True))
    random.Random(args.seed).shuffle(chunks)
    validation_size = max(1, int(len(chunks) * 0.05))
    validation_chunks = chunks[:validation_size]
    training_chunks = chunks[validation_size:]
    print(f"chunks={len(chunks)}")

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.max_length,
        n_ctx=args.max_length,
        n_embd=512,
        n_layer=4,
        n_head=8,
        n_inner=2048,
        activation_function="gelu",
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = GPT2LMHeadModel(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters={parameter_count / 1_000_000:.1f}M")

    training_args = TrainingArguments(
        output_dir=str(args.output),
        bf16=True,
        dataloader_num_workers=4,
        learning_rate=1e-4,
        lr_scheduler_type="linear",
        warmup_steps=0,
        weight_decay=0.0,
        max_grad_norm=1.0,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        remove_unused_columns=False,
        save_strategy="no",
        logging_strategy="epoch",
        eval_strategy="epoch",
        seed=args.seed,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TokenizedDataset(training_chunks),
        eval_dataset=TokenizedDataset(validation_chunks),
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(args.output)
    print(f"saved={args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--ids-from", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    return parser


def main():
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
