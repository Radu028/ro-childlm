"""Convert a GPT-BERT checkpoint to a local Hugging Face model directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import PreTrainedTokenizerFast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.run / "run_8_16_best.bin"
    if not checkpoint.is_file():
        raise SystemExit(f"Missing checkpoint: {checkpoint}")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    vendor = Path(__file__).resolve().parents[1] / "vendor" / "gpt_bert"
    sys.path.insert(0, str(vendor))
    import convert_helpers as conversion

    base_config = json.loads((vendor / "base.json").read_text())
    config = conversion._save_single_variant(
        checkpoint, base_config, args.output
    )
    config["force_causal_mask"] = True
    config["n_positions"] = config.get("max_position_embeddings", 512)
    conversion.write_remote_code_files(
        args.output, True, True, False, False, 0.1, 1e-5, 2
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(args.data / "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    tokenizer.save_pretrained(args.output)
    conversion._write_primary_config(
        args.output,
        config,
        prefer_causal=True,
        force_causal_mask=True,
        include_sequence_classification=False,
    )

    print(f"converted={args.output}")


if __name__ == "__main__":
    main()
