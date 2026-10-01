from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import signal
import sys
import time
from array import array
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from ..postprocess import remove_paralinguistic_markers
from ..words import count_words

# GPT-2 convention: one special token, used both as document separator and as
# padding at evaluation time. All 256 byte tokens are guaranteed in the
# vocabulary, so no unknown token is needed.
END_OF_TEXT = "<|endoftext|>"
PREPARATION_VERSION = 1
RUN_VERSION = 1

DEFAULT_CORPORA = {
    "0-4": {
        "path": "data/0-4/transcripts/corpus.jsonl",
        "sha256": "5df7fee5bd75afdb1bf07de6c88098725ad9f84abc8843cf77624c6c5809a5e1",
        "documents": 2_533,
        "clean_words": 5_000_127,
    },
    "5-8": {
        "path": "data/5-8/transcripts/corpus.jsonl",
        "sha256": "5cd21e635ec22312915adfcc4e2dfc2b501aa9f1d71db8d60cb5ba9645204a05",
        "documents": 2_823,
        "clean_words": 5_000_881,
    },
}


@dataclass(frozen=True)
class PreparedDocument:
    age_band: str
    video_id: str
    text: str
    word_count: int
    channel: str | None

    @property
    def document_id(self) -> str:
        return f"{self.age_band}:{self.video_id}"


@dataclass
class TrainState:
    epoch: int = 0
    batch_in_epoch: int = 0
    global_step: int = 0
    best_validation_loss: float = float("inf")
    stop_reason: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


# Legacy cedilla forms (U+015F/U+0163) appear in a small number of captions and
# would otherwise become distinct tokens from the standard comma-below forms.
CEDILLA_TO_COMMA = str.maketrans({"ş": "ș", "ţ": "ț", "Ş": "Ș", "Ţ": "Ț"})

# For the no-punctuation corpus variant: drop everything except word characters,
# whitespace, and the joiners Romanian uses inside words (s-a, într-o, d'ale).
# Joiners survive only between word characters, so word counts stay identical.
_STRIPPABLE_PUNCTUATION = re.compile(r"[^\w\s'’-]", re.UNICODE)
_DANGLING_JOINER = re.compile(r"(?<!\w)['’-]+|['’-]+(?!\w)", re.UNICODE)


def normalize_derived_text(
    text: str, *, strip_punctuation: bool = False
) -> tuple[str, int]:
    """Create a marker-free training view of one transcript."""
    removal = remove_paralinguistic_markers(text)
    normalized = removal.text.replace(">>", " ")
    if strip_punctuation:
        normalized = _STRIPPABLE_PUNCTUATION.sub(" ", normalized)
        normalized = _DANGLING_JOINER.sub(" ", normalized)
    normalized = " ".join(normalized.split())
    return normalized.translate(CEDILLA_TO_COMMA), len(removal.removed_markers)


def read_strict_corpus(
    path: Path,
    age_band: str,
    *,
    expected_sha256: str | None = None,
    expected_documents: int | None = None,
    expected_clean_words: int | None = None,
    strip_punctuation: bool = False,
):
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"Unexpected SHA-256 for {path}: {actual_sha256}; expected {expected_sha256}"
        )

    documents = []
    removed_markers = 0
    for line_number, line in enumerate(path.open(encoding="utf-8"), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"Expected a JSON object at {path}:{line_number}")
        video_id = str(row.get("video_id") or "")
        row_age_band = str(row.get("age_band") or age_band)
        if not video_id or row_age_band != age_band:
            raise ValueError(f"Invalid video_id or age_band at {path}:{line_number}")
        text, marker_count = normalize_derived_text(
            str(row.get("text") or ""), strip_punctuation=strip_punctuation
        )
        word_count = count_words(text)
        declared_words = row.get("clean_words")
        if not text or (declared_words is not None and word_count != int(declared_words)):
            raise ValueError(
                f"Derived word count mismatch at {path}:{line_number}: "
                f"computed {word_count}, declared {declared_words}"
            )
        removed_markers += marker_count
        documents.append(
            PreparedDocument(
                age_band=age_band,
                video_id=video_id,
                text=text,
                word_count=word_count,
                channel=str(row["channel"]) if row.get("channel") else None,
            )
        )

    words = sum(document.word_count for document in documents)
    if expected_documents is not None and len(documents) != expected_documents:
        raise ValueError(
            f"Unexpected document count for {path}: {len(documents)}; "
            f"expected {expected_documents}"
        )
    if expected_clean_words is not None and words != expected_clean_words:
        raise ValueError(
            f"Unexpected clean word count for {path}: {words}; "
            f"expected {expected_clean_words}"
        )
    return documents, {
        "path": str(path.resolve()),
        "sha256": actual_sha256,
        "documents": len(documents),
        "clean_words": words,
        "removed_paralinguistic_markers": removed_markers,
    }


def split_documents(
    documents: Sequence[PreparedDocument],
    *,
    validation_fraction: float,
    split_seed: int,
) -> tuple[list[PreparedDocument], list[PreparedDocument]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    # Stratified by age band so validation mirrors the corpus age mixture.
    train = []
    validation = []
    for age_band in sorted({document.age_band for document in documents}):
        ordered = sorted(
            (document for document in documents if document.age_band == age_band),
            key=lambda document: document.document_id,
        )
        random.Random(split_seed).shuffle(ordered)
        validation_size = min(
            len(ordered) - 1,
            max(1, round(len(ordered) * validation_fraction)),
        )
        validation.extend(ordered[:validation_size])
        train.extend(ordered[validation_size:])
    return train, validation


def write_jsonl(path: Path, documents: Iterable[PreparedDocument]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")


def train_tokenizer(corpus_file: Path, output_directory: Path, vocab_size: int):
    from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer

    backend = Tokenizer(BPE())
    backend.normalizer = normalizers.NFC()
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train(
        [str(corpus_file)],
        BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=[END_OF_TEXT],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        ),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    tokenizer_file = output_directory / "tokenizer.json"
    backend.save(str(tokenizer_file))
    return backend


def tokenizer_size(tokenizer) -> int:
    if hasattr(tokenizer, "get_vocab_size"):
        return int(tokenizer.get_vocab_size())
    return len(tokenizer)


def tokenizer_token_id(tokenizer, token: str) -> int:
    if hasattr(tokenizer, "token_to_id"):
        token_id = tokenizer.token_to_id(token)
    else:
        token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        raise ValueError(f"Tokenizer is missing required token: {token}")
    return int(token_id)


def tokenizer_encode(tokenizer, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return list(encoded.ids if hasattr(encoded, "ids") else encoded)


def encode_documents_to_uint16(
    documents: Sequence[PreparedDocument], tokenizer, output_path: Path
) -> int:
    if tokenizer_size(tokenizer) > 65_536:
        raise ValueError("The tokenizer vocabulary does not fit in uint16")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eos_id = tokenizer_token_id(tokenizer, END_OF_TEXT)
    token_count = 1
    with output_path.open("wb") as file:
        initial = array("H", [eos_id])
        if sys.byteorder != "little":
            initial.byteswap()
        initial.tofile(file)
        for document in documents:
            ids = tokenizer_encode(tokenizer, document.text)
            ids.append(eos_id)
            encoded = array("H", ids)
            if sys.byteorder != "little":
                encoded.byteswap()
            encoded.tofile(file)
            token_count += len(ids)
    return token_count


def prepare(args) -> int:
    output = Path(args.output).resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(
            f"already prepared: {output} "
            f"manifest_sha256={canonical_json_sha256(manifest)}"
        )
        return 0
    if output.exists() and any(output.iterdir()):
        raise SystemExit(
            f"Incomplete or foreign preparation directory exists: {output}. "
            "Choose a new --output path."
        )
    output.mkdir(parents=True, exist_ok=True)

    corpus_paths = {
        "0-4": Path(args.corpus_0_4).resolve(),
        "5-8": Path(args.corpus_5_8).resolve(),
    }
    bands = [band.strip() for band in args.bands.split(",") if band.strip()]
    unknown = [band for band in bands if band not in corpus_paths]
    if not bands or unknown:
        raise SystemExit(f"--bands must name age bands from {sorted(corpus_paths)}")
    all_documents = []
    sources = {}
    for age_band in bands:
        path = corpus_paths[age_band]
        expected = DEFAULT_CORPORA[age_band]
        documents, source = read_strict_corpus(
            path,
            age_band,
            expected_sha256=None if args.allow_other_corpora else expected["sha256"],
            expected_documents=None if args.allow_other_corpora else expected["documents"],
            expected_clean_words=None if args.allow_other_corpora else expected["clean_words"],
            strip_punctuation=args.strip_punctuation,
        )
        all_documents.extend(documents)
        sources[age_band] = source

    ids = [document.video_id for document in all_documents]
    if len(ids) != len(set(ids)):
        raise ValueError("A video_id occurs more than once across the two age bands")
    word_budget = None
    if args.max_words:
        # Sample whole documents until the budget is reached. The draw seed
        # can vary independently of the train/validation split seed.
        selection_seed = (
            args.selection_seed
            if args.selection_seed is not None
            else args.split_seed
        )
        ordered = sorted(all_documents, key=lambda document: document.document_id)
        random.Random(selection_seed).shuffle(ordered)
        selected = []
        selected_words = 0
        for document in ordered:
            if selected_words >= args.max_words:
                break
            selected.append(document)
            selected_words += document.word_count
        all_documents = selected
        word_budget = {
            "max_words": args.max_words,
            "selected_documents": len(selected),
            "selected_words": selected_words,
            "selection_seed": selection_seed,
            "selection": "deterministic shuffle, whole documents until budget reached",
        }
    train, validation = split_documents(
        all_documents,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
    )
    write_jsonl(output / "splits" / "train.jsonl", train)
    write_jsonl(output / "splits" / "validation.jsonl", validation)

    if args.tokenizer_from:
        # Reuse the shared tokenizer (e.g. for age-band ablations) so every
        # condition scores under the exact same vocabulary.
        from tokenizers import Tokenizer

        source_file = (
            Path(args.tokenizer_from).resolve() / "tokenizer" / "tokenizer.json"
        )
        (output / "tokenizer").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, output / "tokenizer" / "tokenizer.json")
        tokenizer = Tokenizer.from_file(str(output / "tokenizer" / "tokenizer.json"))
        tokenizer_scope = f"reused from {source_file.parent.parent}"
    else:
        tokenizer_corpus = output / "tokenizer_corpus.txt"
        with tokenizer_corpus.open("w", encoding="utf-8") as file:
            for document in sorted(all_documents, key=lambda item: item.document_id):
                file.write(document.text + "\n")
        tokenizer = train_tokenizer(
            tokenizer_corpus, output / "tokenizer", args.vocab_size
        )
        tokenizer_scope = "trained on this preparation's full corpus"
    train_tokens = encode_documents_to_uint16(
        train, tokenizer, output / "tokens" / "train.uint16"
    )
    validation_tokens = encode_documents_to_uint16(
        validation, tokenizer, output / "tokens" / "validation.uint16"
    )

    split_ids = {
        "train": [document.document_id for document in train],
        "validation": [document.document_id for document in validation],
    }
    manifest = {
        "preparation_version": PREPARATION_VERSION,
        "method": "Baby Scale-style GPT-2 small preparation",
        "sources": sources,
        "normalization": {
            "unicode": "tokenizer NFC",
            "paralinguistic_markers": "known bracketed markers removed",
            "speaker_prefix": ">> removed",
            "punctuation": "removed except intra-word hyphens and apostrophes"
            if args.strip_punctuation
            else "preserved",
            "case": "preserved",
            "diacritics": "cedilla ş/ţ mapped to comma-below ș/ț",
        },
        "age_bands": bands,
        "word_budget": word_budget,
        "split": {
            "unit": "whole video transcript",
            "stratification": "independent 85/15 split within each age band",
            "validation_fraction": args.validation_fraction,
            "seed": args.split_seed,
            "documents": {
                "train": len(train),
                "validation": len(validation),
            },
            "clean_words": {
                "train": sum(document.word_count for document in train),
                "validation": sum(document.word_count for document in validation),
            },
            "document_ids_sha256": canonical_json_sha256(split_ids),
        },
        "tokenizer": {
            "algorithm": "byte-level BPE",
            "training_scope": tokenizer_scope,
            "requested_vocab_size": args.vocab_size,
            "actual_vocab_size": tokenizer_size(tokenizer),
            "special_tokens": [END_OF_TEXT],
            "directory": "tokenizer",
            "tokenizer_json_sha256": sha256_file(
                output / "tokenizer" / "tokenizer.json"
            ),
        },
        "tokens": {
            "dtype": "little-endian uint16",
            "document_separator": END_OF_TEXT,
            "train": {
                "count": train_tokens,
                "path": "tokens/train.uint16",
                "sha256": sha256_file(output / "tokens" / "train.uint16"),
            },
            "validation": {
                "count": validation_tokens,
                "path": "tokens/validation.uint16",
                "sha256": sha256_file(output / "tokens" / "validation.uint16"),
            },
        },
    }
    manifest["identity_sha256"] = canonical_json_sha256(manifest)
    write_json(manifest_path, manifest)
    print(
        f"prepared={output} documents={len(all_documents):,} "
        f"words={sum(document.word_count for document in all_documents):,} "
        f"train_tokens={train_tokens:,} validation_tokens={validation_tokens:,}"
    )
    return 0


class TokenBlockDataset:
    def __init__(self, path: Path, sequence_length: int, torch) -> None:
        import numpy

        self.tokens = numpy.memmap(path, dtype="<u2", mode="r")
        self.sequence_length = sequence_length
        self.torch = torch
        self.block_count = (len(self.tokens) - 1) // sequence_length
        if self.block_count < 1:
            raise ValueError(f"Not enough tokens for one block in {path}")

    def batch(self, indices: Sequence[int], device):
        import numpy

        inputs = []
        labels = []
        for index in indices:
            start = int(index) * self.sequence_length
            inputs.append(
                numpy.array(self.tokens[start : start + self.sequence_length])
            )
            labels.append(
                numpy.array(self.tokens[start + 1 : start + self.sequence_length + 1])
            )
        input_ids = self.torch.from_numpy(numpy.stack(inputs)).long().to(device)
        target_ids = self.torch.from_numpy(numpy.stack(labels)).long().to(device)
        return input_ids, target_ids


def epoch_batches(
    block_count: int, batch_size: int, seed: int, epoch: int, torch
) -> list[list[int]]:
    generator = torch.Generator().manual_seed(seed + epoch)
    permutation = torch.randperm(block_count, generator=generator).tolist()
    return [
        permutation[start : start + batch_size]
        for start in range(0, len(permutation), batch_size)
    ]


def linear_lr_multiplier(step: int, *, total_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    return max(0.0, 1.0 - (step + 1) / total_steps)


def resume_signature(config):
    keys = (
        "mode",
        "preparation_identity",
        "seed",
        "sequence_length",
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "max_steps",
        "benchmark_steps",
    )
    return {key: config[key] for key in keys}


def assert_resume_compatible(current, saved) -> None:
    if resume_signature(current) != resume_signature(saved):
        raise ValueError("Checkpoint configuration does not match this run")


def seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(tokenizer, torch):
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        attn_implementation="sdpa",
    )
    return GPT2LMHeadModel(config)


def checkpoint_payload(model, optimizer, scheduler, state: TrainState, config, torch):
    return {
        "run_version": RUN_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "state": asdict(state),
        "config": config,
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        },
    }


def save_checkpoint(path: Path, payload, torch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, config, torch) -> TrainState:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert_resume_compatible(config, payload["config"])
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    random.setstate(payload["rng"]["python"])
    torch.set_rng_state(payload["rng"]["torch"])
    if torch.cuda.is_available() and payload["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
    return TrainState(**payload["state"])


def validation_loss(
    model, dataset: TokenBlockDataset, batch_size: int, device, torch, *,
    max_batches: int | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, dataset.block_count, batch_size)):
            if max_batches is not None and batch_index >= max_batches:
                break
            indices = list(range(start, min(start + batch_size, dataset.block_count)))
            input_ids, labels = dataset.batch(indices, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(input_ids=input_ids).logits
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    reduction="sum",
                )
            total_loss += float(loss.item())
            total_tokens += labels.numel()
    return total_loss / total_tokens


def append_metric(path: Path, metric) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metric, ensure_ascii=False) + "\n")


def run_training(args, *, benchmark: bool) -> int:
    import tokenizers
    import torch
    import transformers
    from transformers import PreTrainedTokenizerFast

    prepared = Path(args.prepared).resolve()
    output = Path(args.output).resolve()
    benchmark_report_path = output / "benchmark_report.json"
    if benchmark and benchmark_report_path.exists():
        report = json.loads(benchmark_report_path.read_text(encoding="utf-8"))
        print(
            f"benchmark already complete: {benchmark_report_path} "
            f"status={report.get('status', 'unknown')}"
        )
        return 0
    manifest_path = prepared / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing prepared manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("identity_sha256") != canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "identity_sha256"}
    ):
        raise SystemExit("Prepared manifest identity is invalid")
    for split in ("train", "validation"):
        token_info = manifest["tokens"][split]
        token_path = prepared / token_info["path"]
        if sha256_file(token_path) != token_info["sha256"]:
            raise SystemExit(f"Prepared token file changed: {token_path}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise SystemExit("This pipeline requires BF16-capable CUDA hardware")
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(prepared / "tokenizer" / "tokenizer.json"),
        bos_token=END_OF_TEXT,
        eos_token=END_OF_TEXT,
        pad_token=END_OF_TEXT,
        model_max_length=1024,
    )
    train_dataset = TokenBlockDataset(
        prepared / manifest["tokens"]["train"]["path"],
        args.sequence_length,
        torch,
    )
    validation_dataset = TokenBlockDataset(
        prepared / manifest["tokens"]["validation"]["path"],
        args.sequence_length,
        torch,
    )
    batches_per_epoch = math.ceil(train_dataset.block_count / args.batch_size)
    natural_steps = args.epochs * batches_per_epoch
    training_target_steps = args.steps if benchmark else natural_steps
    config = {
        "run_version": RUN_VERSION,
        "mode": "benchmark" if benchmark else "train",
        "preparation_identity": manifest["identity_sha256"],
        "seed": args.seed,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_steps": natural_steps,
        "benchmark_steps": args.steps if benchmark else None,
        "checkpoint_every": args.checkpoint_every,
        "validation_batches": args.validation_batches if benchmark else None,
        "architecture": {
            "name": "GPT-2 small",
            "layers": 12,
            "heads": 12,
            "embedding_size": 768,
            "positions": 1024,
            "attention": "PyTorch SDPA",
        },
        "optimizer": {
            "name": "Adam",
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "gradient_clip_norm": 1.0,
        },
        "scheduler": "linear decay, no warmup",
    }
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed, torch)
    model = build_model(tokenizer, torch).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: linear_lr_multiplier(step, total_steps=natural_steps),
    )
    state = TrainState()
    checkpoint_path = output / "checkpoints" / "last.pt"
    if checkpoint_path.exists():
        state = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, config, torch
        )
        model.to(device)
        print(f"resumed={checkpoint_path} global_step={state.global_step}")
    write_json(output / "run_config.json", config)

    stop_requested = False

    def request_stop(signum: int, _) -> None:
        nonlocal stop_requested
        stop_requested = True
        state.stop_reason = signal.Signals(signum).name
        print("stop requested; saving after the current optimizer step", flush=True)

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    durations = []
    tokens_seen = 0
    started = time.perf_counter()
    try:
        while (
            state.global_step < training_target_steps and state.epoch < args.epochs
        ):
            batches = epoch_batches(
                train_dataset.block_count,
                args.batch_size,
                args.seed,
                state.epoch,
                torch,
            )
            while state.batch_in_epoch < len(batches):
                if state.global_step >= training_target_steps:
                    break
                step_started = time.perf_counter()
                input_ids, labels = train_dataset.batch(
                    batches[state.batch_in_epoch], device
                )
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    logits = model(input_ids=input_ids).logits
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss at step {state.global_step + 1}"
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                duration = time.perf_counter() - step_started
                durations.append(duration)
                tokens = labels.numel()
                tokens_seen += tokens
                state.global_step += 1
                state.batch_in_epoch += 1
                metric = {
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "batch_in_epoch": state.batch_in_epoch,
                    "loss": float(loss.item()),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "tokens": tokens,
                    "seconds": duration,
                }
                append_metric(output / "metrics.jsonl", metric)
                if state.global_step % args.log_every == 0:
                    print(
                        f"step={state.global_step} loss={metric['loss']:.4f} "
                        f"tokens_per_second={tokens / duration:,.0f}",
                        flush=True,
                    )
                if (
                    state.global_step % args.checkpoint_every == 0
                    or stop_requested
                ):
                    save_checkpoint(
                        checkpoint_path,
                        checkpoint_payload(
                            model, optimizer, scheduler, state, config, torch
                        ),
                        torch,
                    )
                if stop_requested:
                    return 130

            if state.batch_in_epoch == len(batches):
                state.epoch += 1
                state.batch_in_epoch = 0
                if not benchmark:
                    value = validation_loss(
                        model,
                        validation_dataset,
                        args.evaluation_batch_size,
                        device,
                        torch,
                    )
                    append_metric(
                        output / "validation.jsonl",
                        {
                            "epoch": state.epoch,
                            "global_step": state.global_step,
                            "loss": value,
                            "perplexity": math.exp(min(value, 20)),
                        },
                    )
                    if value < state.best_validation_loss:
                        state.best_validation_loss = value
                        model.save_pretrained(output / "best_model")
                        tokenizer.save_pretrained(output / "best_model")
                save_checkpoint(
                    checkpoint_path,
                    checkpoint_payload(
                        model, optimizer, scheduler, state, config, torch
                    ),
                    torch,
                )

        save_checkpoint(
            checkpoint_path,
            checkpoint_payload(model, optimizer, scheduler, state, config, torch),
            torch,
        )
        if benchmark:
            checkpoint_step = state.global_step
            del model, optimizer, scheduler
            if device.type == "cuda":
                torch.cuda.empty_cache()
            model = build_model(tokenizer, torch).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: linear_lr_multiplier(
                    step, total_steps=natural_steps
                ),
            )
            state = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, config, torch
            )
            model.to(device)
            if state.batch_in_epoch == batches_per_epoch:
                state.epoch += 1
                state.batch_in_epoch = 0
            batches = epoch_batches(
                train_dataset.block_count,
                args.batch_size,
                args.seed,
                state.epoch,
                torch,
            )
            input_ids, labels = train_dataset.batch(
                batches[state.batch_in_epoch], device
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(input_ids=input_ids).logits
                resume_loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                )
            resume_loss.backward()
            optimizer.step()
            scheduler.step()
            state.global_step += 1
            state.batch_in_epoch += 1
            save_checkpoint(
                checkpoint_path,
                checkpoint_payload(model, optimizer, scheduler, state, config, torch),
                torch,
            )
            value = validation_loss(
                model,
                validation_dataset,
                args.evaluation_batch_size,
                device,
                torch,
                max_batches=args.validation_batches,
            )
            measured = durations[min(10, len(durations)) :] or durations
            seconds_per_step = sum(measured) / len(measured)
            report = {
                "status": "passed",
                "checkpoint_reload_and_next_step": state.global_step
                == checkpoint_step + 1,
                "training_steps_measured": checkpoint_step,
                "resume_step": state.global_step,
                "resume_loss": float(resume_loss.item()),
                "validation_loss": value,
                "validation_perplexity": math.exp(min(value, 20)),
                "seconds": time.perf_counter() - started,
                "seconds_per_step_after_warmup": seconds_per_step,
                "tokens_per_second_after_warmup": (
                    args.batch_size * args.sequence_length / seconds_per_step
                ),
                "estimated_seconds_per_epoch": seconds_per_step * batches_per_epoch,
                "estimated_seconds_for_20_epochs": (
                    seconds_per_step * batches_per_epoch * 20
                ),
                "peak_gpu_memory_bytes": (
                    torch.cuda.max_memory_allocated() if device.type == "cuda" else None
                ),
                "system": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "tokenizers": tokenizers.__version__,
                    "device": str(device),
                    "gpu": torch.cuda.get_device_name(0)
                    if device.type == "cuda"
                    else None,
                },
            }
            write_json(benchmark_report_path, report)
            print(
                f"benchmark passed tokens_per_second={report['tokens_per_second_after_warmup']:,.0f} "
                f"estimated_20_epochs_hours={report['estimated_seconds_for_20_epochs'] / 3600:.2f}"
            )
        else:
            write_json(
                output / "training_complete.json",
                {
                    "status": "complete",
                    "epochs": state.epoch,
                    "global_step": state.global_step,
                    "best_validation_loss": state.best_validation_loss,
                    "seconds": time.perf_counter() - started,
                },
            )
            print(f"training complete epochs={state.epoch} steps={state.global_step}")
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible Baby Scale-style GPT-2 small experiment"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--corpus-0-4", default=DEFAULT_CORPORA["0-4"]["path"])
    prepare_parser.add_argument("--corpus-5-8", default=DEFAULT_CORPORA["5-8"]["path"])
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--validation-fraction", type=float, default=0.15)
    prepare_parser.add_argument("--split-seed", type=int, default=42)
    prepare_parser.add_argument(
        "--selection-seed",
        type=int,
        default=None,
        help="Seed for the word-budget document selection only; defaults to --split-seed.",
    )
    prepare_parser.add_argument("--vocab-size", type=int, default=50_000)
    prepare_parser.add_argument("--allow-other-corpora", action="store_true")
    prepare_parser.add_argument("--bands", default="0-4,5-8")
    prepare_parser.add_argument("--strip-punctuation", action="store_true")
    prepare_parser.add_argument("--tokenizer-from", default=None)
    prepare_parser.add_argument("--max-words", type=int, default=None)
    prepare_parser.set_defaults(handler=prepare)

    def add_run_arguments(run_parser: argparse.ArgumentParser) -> None:
        run_parser.add_argument("--prepared", required=True)
        run_parser.add_argument("--output", required=True)
        run_parser.add_argument("--seed", type=int, default=42)
        run_parser.add_argument("--batch-size", type=int, default=8)
        run_parser.add_argument("--sequence-length", type=int, default=1024)
        run_parser.add_argument("--epochs", type=int, default=20)
        run_parser.add_argument("--learning-rate", type=float, default=1e-4)
        run_parser.add_argument("--weight-decay", type=float, default=0.0)
        run_parser.add_argument("--checkpoint-every", type=int, default=100)
        run_parser.add_argument("--log-every", type=int, default=10)
        run_parser.add_argument("--evaluation-batch-size", type=int, default=8)
        run_parser.add_argument(
            "--device", choices=("auto", "cuda", "cpu"), default="cuda"
        )

    benchmark_parser = subparsers.add_parser("benchmark")
    add_run_arguments(benchmark_parser)
    benchmark_parser.add_argument("--steps", type=int, default=200)
    benchmark_parser.add_argument("--validation-batches", type=int, default=20)
    benchmark_parser.set_defaults(
        handler=lambda args: run_training(args, benchmark=True)
    )

    train_parser = subparsers.add_parser("train")
    add_run_arguments(train_parser)
    train_parser.set_defaults(
        steps=0,
        validation_batches=0,
        handler=lambda args: run_training(args, benchmark=False),
    )
    return parser


def main():
    args = build_parser().parse_args()
    status = args.handler(args)
    if status:
        raise SystemExit(status)


if __name__ == "__main__":
    main()
