#!/usr/bin/env python3
"""Convert the official BabyBabelLM Romanian parquet to document-level JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as parquet

from romanian_childlm.training.babyscale_gpt2 import normalize_derived_text
from romanian_childlm.words import count_words


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = parquet.read_table(args.input).to_pylist()
    written = 0
    words = 0
    mismatches = 0
    categories = {}
    with args.output.open("x", encoding="utf-8") as output:
        for index, row in enumerate(rows):
            text = str(row.get("text") or "")
            kept, _ = normalize_derived_text(text)
            stripped, _ = normalize_derived_text(text, strip_punctuation=True)
            count = count_words(kept)
            # Preserve the filtering used for the reported corpus.
            if count != count_words(stripped):
                mismatches += 1
                continue
            if not stripped:
                continue
            doc_id = str(row.get("doc-id") or f"official-{index:06d}")
            category = str(row.get("category") or "unknown")
            categories[category] = categories.get(category, 0) + 1
            output.write(
                json.dumps(
                    {
                        "video_id": doc_id,
                        "text": text,
                        "clean_words": count,
                        "channel": str(row.get("data-source") or category),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            words += count
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(
        f"documents={written} words={words:,} mismatches={mismatches} "
        f"categories={categories} sha256={digest}"
    )


if __name__ == "__main__":
    main()
