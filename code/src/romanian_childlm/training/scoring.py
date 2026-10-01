"""Causal sentence scoring shared by the reported evaluation protocols."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def read_multiblimp(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    required = {"sen", "wrong_sen", "phenomenon"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid Romanian MultiBLiMP file: {path}")
    return rows


def _encode_lm_eval_continuation(
    tokenizer, sentence: str, max_length: int
) -> list[int] | None:
    ids = tokenizer.encode(
        f" {sentence}",
        add_special_tokens=False,
        verbose=False,
    )
    if not 1 <= len(ids) <= max_length:
        return None
    prefix_token_id = (
        tokenizer.bos_token_id
        if tokenizer.bos_token_id is not None
        else tokenizer.eos_token_id
    )
    return [int(prefix_token_id), *ids]


def _score_sentences(
    model, tokenizer, sentences: list[str], batch_size: int, max_length: int, device, torch,
) -> list[float | None]:
    encoded = []
    for sentence in sentences:
        # Match lm-eval 0.4.9's empty-context multiple-choice protocol used by
        # BabyBabelLM: prepend the task delimiter and condition the first
        # continuation token on BOS (falling back to EOS when BOS is absent).
        encoded.append(_encode_lm_eval_continuation(tokenizer, sentence, max_length))

    scores = [None] * len(sentences)
    valid_indices = [index for index, ids in enumerate(encoded) if ids is not None]
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(valid_indices), batch_size):
            indices = valid_indices[start : start + batch_size]
            sequences = [encoded[index] for index in indices]
            width = max(len(sequence) for sequence in sequences)
            input_ids = torch.full(
                (len(sequences), width),
                int(tokenizer.pad_token_id),
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row_index, sequence in enumerate(sequences):
                input_ids[row_index, : len(sequence)] = torch.tensor(
                    sequence, dtype=torch.long, device=device
                )
                attention_mask[row_index, : len(sequence)] = 1
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits[:, :-1, :]
            targets = input_ids[:, 1:]
            target_mask = attention_mask[:, 1:].bool()
            log_probabilities = torch.nn.functional.log_softmax(logits.float(), dim=-1)
            token_scores = log_probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            sentence_scores = (token_scores * target_mask).sum(dim=1).cpu().tolist()
            for original_index, score in zip(indices, sentence_scores, strict=True):
                scores[original_index] = float(score)
    return scores


def _multiblimp_evaluation(
    model, tokenizer, rows: list[dict[str, str]], batch_size: int, max_length: int,
    device, torch,
):
    sentences = []
    for row in rows:
        sentences.extend((row["sen"], row["wrong_sen"]))
    scores = _score_sentences(model, tokenizer, sentences, batch_size, max_length, device, torch)

    details = []
    by_phenomenon = defaultdict(list)
    deltas = []
    skipped = 0
    for index, row in enumerate(rows):
        grammatical_score = scores[index * 2]
        ungrammatical_score = scores[index * 2 + 1]
        if grammatical_score is None or ungrammatical_score is None:
            skipped += 1
            continue
        delta = grammatical_score - ungrammatical_score
        correct = delta > 0
        deltas.append(delta)
        by_phenomenon[row["phenomenon"]].append(correct)
        details.append(
            {
                "row": index + 1,
                "phenomenon": row["phenomenon"],
                "inflect_item": row.get("inflect_item", ""),
                "agreement_certainty": row.get("agreement_certainty", ""),
                "grammatical_feature": row.get("grammatical_feature", ""),
                "ungrammatical_feature": row.get("ungrammatical_feature", ""),
                "sentence": row["sen"],
                "wrong_sentence": row["wrong_sen"],
                "grammatical_log_probability": grammatical_score,
                "ungrammatical_log_probability": ungrammatical_score,
                "delta": delta,
                "correct": correct,
            }
        )
    evaluated = len(details)
    summary = {
        "accuracy": sum(detail["correct"] for detail in details) / evaluated,
        "evaluated_pairs": evaluated,
        "skipped_pairs": skipped,
        "mean_delta": sum(deltas) / evaluated,
        "by_phenomenon": {
            phenomenon: {
                "accuracy": sum(values) / len(values),
                "pairs": len(values),
            }
            for phenomenon, values in sorted(by_phenomenon.items())
        },
    }
    return summary, details
