from __future__ import annotations

import argparse
import json
from pathlib import Path

from romanian_childlm.training.scoring import _multiblimp_evaluation, read_multiblimp
from romanian_childlm.training.babyscale_gpt2 import normalize_derived_text


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a Hugging Face causal LM on Romanian MultiBLiMP"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--multiblimp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--strip-punctuation", action="store_true",
                        help="Apply the GPT-BERT training normalization to both sentences")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    rows = read_multiblimp(args.multiblimp)
    if args.strip_punctuation:
        for row in rows:
            for key in ("sen", "wrong_sen"):
                row[key], _ = normalize_derived_text(row[key], strip_punctuation=True)
    summary, details = _multiblimp_evaluation(
        model,
        tokenizer,
        rows,
        args.batch_size,
        int(model.config.n_positions),
        torch.device(args.device),
        torch,
    )
    result = {
        "model": args.model,
        "revision": args.revision,
        "benchmark": str(args.multiblimp),
        "strip_punctuation": args.strip_punctuation,
        "scoring": (
            "lm-eval 0.4.9 empty-context multiple-choice: target delimiter is "
            "one space, BOS (or EOS fallback) is the context token, all "
            "continuation tokens are scored"
        ),
        "summary": summary,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Evaluated {summary['evaluated_pairs']} pairs: "
        f"accuracy={summary['accuracy']:.3%}"
    )


if __name__ == "__main__":
    main()
