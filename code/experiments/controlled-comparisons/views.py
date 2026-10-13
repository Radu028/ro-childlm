"""Views, tokenizers and MultiBLiMP files of the controlled comparisons."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from romanian_childlm.training import babyscale_gpt2 as core
from romanian_childlm.words import _WORD, count_words

SEEDS = [42, 0, 123, 321, 9000]
PUNCT = ["keep", "strip"]
FAMILIES = {
    "punctuation": [
        f"{source}-{punct}" for source in ["captions", "babybabellm", "mix"] for punct in PUNCT
    ],
    "additive": ["bbl800-cap800", "bbl800-repeat800"],
    "fixed-steps": ["fixed-keep", "fixed-strip"],
    "subtitles": ["bbl-full-500k", "bbl-noos-500k", "mix-full-500k", "mix-noos-500k"],
    "scale": [f"captions-{n}m-{punct}" for n in [2, 5] for punct in PUNCT],
}
# Hashes of the original split files; a rehydrated corpus may differ.
ORIGINAL_SPLITS = {
    ("captions", "train"):
        "254c1e6a7be8ca485ea79ccdb788eae8d5788f3ad4d299244ea33d80cf6548e3",
    ("captions", "validation"):
        "debe7646694b2d4b62b78a8547d2bc74198b8b2b6c7a68728ba1f07762ca424f",
    ("babybabellm", "train"):
        "d2668dff12b3e05a5dc33ce7b039c9c349ad47bde769046f6167a599f35dd124",
    ("babybabellm", "validation"):
        "3827758be3f1693df982ba3049d77ba9f1791a4abc5607943f9bffc15737273f",
}
NORMALIZATION = {"keep": "preserved", "strip": "removed except intra-word hyphens and apostrophes"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text, punct):
    text, _ = core.normalize_derived_text(text, strip_punctuation=punct == "strip")
    return unicodedata.normalize("NFC", text)


def component(channel):
    if channel.startswith("CHILDES"):
        return "childes"
    if channel == "OpenSubtitles":
        return "opensubtitles"
    if channel in ["LEMI & Romanian Corpus", "GlotStoryBook - GlotStoryBook"]:
        return "books"
    raise ValueError(f"Unknown BabyBabelLM component: {channel}")


def load_pools(captions_prepared, bbl_prepared, corpus_0_4, corpus_5_8, bbl_corpus):
    originals = {}
    for band, path in [("0-4", corpus_0_4), ("5-8", corpus_5_8)]:
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            originals[f"captions:{band}:{row['video_id']}"] = row["text"]
    for line in open(bbl_corpus, encoding="utf-8"):
        row = json.loads(line)
        originals[f"babybabellm:0-4:{row['video_id']}"] = row["text"]

    pools, hashes = {}, {}
    for source, prepared in [("captions", captions_prepared), ("babybabellm", bbl_prepared)]:
        for split in ["train", "validation"]:
            path = Path(prepared) / "splits" / f"{split}.jsonl"
            hashes[f"{source}/{split}"] = digest = sha(path)
            if digest != ORIGINAL_SPLITS[source, split]:
                print(f"note: {path} differs from the original {source} {split} split")

            rows = []
            for line in open(path, encoding="utf-8"):
                row = json.loads(line)
                row["source"] = source
                row["parent_id"] = row["video_id"] = f"{source}:{row['age_band']}:{row['video_id']}"
                row["age_band"] = "0-4"  # one slot for the trainer, whatever the source
                text = normalize(originals[row["parent_id"]], "keep")
                assert (
                    normalize(text, "strip") == unicodedata.normalize("NFC", row["text"])
                ), row["parent_id"]
                assert count_words(text) == row["word_count"], row["parent_id"]
                row["text"] = text
                row["component"] = source if source == "captions" else component(row["channel"])
                rows.append(row)
            pools[source, split] = rows

        assert not (
            {r["parent_id"] for r in pools[source, "train"]}
            & {r["parent_id"] for r in pools[source, "validation"]}
        )

    return pools, hashes


def proportional_quotas(counts, total):
    denominator = sum(counts.values())
    quotas = {k: total * v // denominator for k, v in counts.items()}
    order = sorted(counts, key=lambda k: (-(total * counts[k] % denominator), k))
    for k in order[:total - sum(quotas.values())]:
        quotas[k] += 1
    return quotas


def take(rows, total, seed=42):
    """Shuffled whole documents up to `total` words; the last one is cut to fit."""
    candidates = sorted(rows, key=lambda r: r["parent_id"])
    random.Random(seed).shuffle(candidates)

    kept, remaining = [], total
    for original in candidates:
        if not remaining:
            break
        row = dict(original)
        if row["word_count"] > remaining:
            words = list(_WORD.finditer(row["text"]))
            # Keep punctuation after the last retained word, not the next word.
            row["text"] = row["text"][:words[remaining].start()].rstrip()
            row["word_count"] = remaining
            row["video_id"] += f":prefix-{remaining}"
        assert count_words(row["text"]) == row["word_count"]
        kept.append(row)
        remaining -= row["word_count"]

    assert remaining == 0, (total, remaining)
    return kept


def stratified(rows, total, reference_counts):
    groups = defaultdict(list)
    for row in rows:
        groups[row["component"]].append(row)
    quotas = proportional_quotas(reference_counts, total)
    return [r for k in sorted(quotas) for r in take(groups[k], quotas[k])]


def components(rows):
    counts = Counter()
    for row in rows:
        counts[row["component"]] += row["word_count"]
    return counts


def documents(pools, base):
    captions, bbl = pools["captions", "train"], pools["babybabellm", "train"]
    counts = components(bbl)
    noos = [r for r in bbl if r["component"] != "opensubtitles"]
    noos_counts = {k: v for k, v in counts.items() if k != "opensubtitles"}

    stem, _, form = base.rpartition("-")
    if stem == "captions":
        return take(captions, 800_000), form, form
    if stem == "babybabellm":
        return stratified(bbl, 800_000, counts), form, form
    if stem == "mix":
        return take(captions, 400_000) + stratified(bbl, 400_000, counts), form, form
    if stem == "fixed":  # both forms validate on punctuated text
        return take(captions, 800_000), form, "keep"
    if stem in ["captions-2m", "captions-5m"]:
        return take(captions, int(stem[-2]) * 1_000_000), form, form

    b800 = stratified(bbl, 800_000, counts)
    rows = {
        "bbl800-cap800": lambda: b800 + take(captions, 800_000),
        "bbl800-repeat800": lambda: b800 + [
            {**r, "video_id": r["video_id"] + ":repeat"} for r in b800
        ],
        "bbl-full-500k": lambda: stratified(bbl, 500_000, counts),
        "bbl-noos-500k": lambda: stratified(noos, 500_000, noos_counts),
        "mix-full-500k": lambda: stratified(bbl, 250_000, counts) + take(captions, 250_000),
        "mix-noos-500k": lambda: stratified(noos, 250_000, noos_counts) + take(captions, 250_000),
    }[base]()
    return rows, "keep", "keep"


def ordered(rows, order, punct):
    rows = list(rows)
    random.Random(order).shuffle(rows)
    return [{**r, "text": normalize(r["text"], punct)} for r in rows]


def view(pools, base, order):
    train, form, valid_form = documents(pools, base)
    # Every view validates on the same 80k caption and 80k BabyBabelLM words.
    valid = take(pools["captions", "validation"], 80_000) + stratified(
        pools["babybabellm", "validation"], 80_000, components(pools["babybabellm", "train"])
    )
    return ordered(train, order, form), ordered(valid, order, valid_form)


def train_tokenizers(pools, folder):
    from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer

    rows, _, _ = documents(pools, "mix-keep")
    texts = [r["text"] for punct in PUNCT for r in ordered(rows, 42, punct)]
    folder.mkdir(parents=True, exist_ok=True)

    for kind, size in [("gpt2", 30000), ("gptbert", 16384)]:
        path = folder / f"{kind}.json"
        if path.exists():
            continue

        special = [core.END_OF_TEXT] if kind == "gpt2" else (
            ["<unk>", "<s>", "</s>", "<pad>", "<mask>"] + [f"<special_{i}>" for i in range(11)]
        )
        tokenizer = Tokenizer(BPE() if kind == "gpt2" else BPE(unk_token="<unk>"))
        tokenizer.normalizer = normalizers.NFC() if kind == "gpt2" else normalizers.NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=kind == "gptbert")
        tokenizer.decoder = decoders.ByteLevel()

        trainer = BpeTrainer(
            vocab_size=size,
            min_frequency=2,
            special_tokens=special,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tokenizer.train_from_iterator(texts, trainer)
        assert tokenizer.get_vocab_size() == size, kind
        tokenizer.save(str(path))


def write_benchmarks(benchmark, tokenizers, folder):
    from tokenizers import Tokenizer

    with open(benchmark, encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        fields, rows = reader.fieldnames, list(reader)
    assert len(rows) == 2056

    encoders = [
        Tokenizer.from_file(str(tokenizers / f"{kind}.json")) for kind in ["gpt2", "gptbert"]
    ]
    for punct in PUNCT:
        with open(folder / f"multiblimp-{punct}.tsv", "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for row in rows:
                row = {**row, **{k: normalize(row[k], punct) for k in ["sen", "wrong_sen"]}}
                assert row["sen"] != row["wrong_sen"]
                assert all(
                    len(t.encode(row[k]).ids) + 2 <= 512
                    for t in encoders
                    for k in ["sen", "wrong_sen"]
                )
                writer.writerow(row)


def write_view(folder, train, valid, tokenizer_file, base, order, input_hashes):
    """A preparation folder in the format of run_babyscale_gpt2_small.py prepare."""
    from tokenizers import Tokenizer

    if (folder / "manifest.json").exists():
        return

    temporary = folder.with_name(folder.name + ".partial")
    shutil.rmtree(temporary, ignore_errors=True)
    (temporary / "tokenizer").mkdir(parents=True)
    assert not {r["parent_id"] for r in train} & {r["parent_id"] for r in valid}
    shutil.copyfile(tokenizer_file, temporary / "tokenizer/tokenizer.json")
    tokenizer = Tokenizer.from_file(str(tokenizer_file))

    tokens, files, words, composition = {}, {}, {}, {}
    for split, rows in [("train", train), ("validation", valid)]:
        path = temporary / "splits" / f"{split}.jsonl"
        path.parent.mkdir(exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                assert count_words(row["text"]) == row["word_count"]
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        files[f"splits/{split}.jsonl"] = sha(path)

        docs = [
            core.PreparedDocument(
                r["age_band"], r["video_id"], r["text"], r["word_count"], r["channel"]
            )
            for r in rows
        ]
        path = temporary / "tokens" / f"{split}.uint16"
        count = core.encode_documents_to_uint16(docs, tokenizer, path)
        tokens[split] = dict(count=count, path=f"tokens/{split}.uint16", sha256=sha(path))
        words[split] = sum(r["word_count"] for r in rows)
        composition[split] = dict(components(rows))

    form = "strip" if base.endswith("-strip") else "keep"
    manifest = dict(
        condition=folder.name,
        base=base,
        order_seed=order,
        normalization=dict(punctuation=NORMALIZATION[form], unicode="NFC"),
        split=dict(clean_words=words, files=files),
        composition=composition,
        tokens=tokens,
        tokenizer=dict(sha256=sha(tokenizer_file)),
        inputs=input_hashes,
    )
    manifest["identity_sha256"] = core.canonical_json_sha256(manifest)
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    temporary.rename(folder)
