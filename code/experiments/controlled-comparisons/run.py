"""Prepare, train and summarize the controlled comparisons of Appendix G."""

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import views as v

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[1] / "scripts"
MODELS = ["gpt2", "gptbert"]
CROSSED = v.FAMILIES["punctuation"][:4] + v.FAMILIES["additive"]
T_975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}  # t quantile of a 95% interval by n (df = n - 1)
STEP_TOKENS = 256 * 128  # GPT-BERT global batch x sequence length


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=1) + "\n")


def run(command, log):
    with open(log, "a") as file:
        subprocess.run(
            [str(c) for c in command],
            check=True,
            stdout=file,
            stderr=subprocess.STDOUT,
        )


def prepare(args):
    root = args.root
    pools, hashes = v.load_pools(
        args.captions_prepared,
        args.bbl_prepared,
        args.corpus_0_4,
        args.corpus_5_8,
        args.bbl_corpus,
    )
    v.train_tokenizers(pools, root / "tokenizers")
    v.write_benchmarks(args.benchmark, root / "tokenizers", root)

    for bases in v.FAMILIES.values():
        for base in bases:
            for order in v.SEEDS:
                train, valid = v.view(pools, base, order)
                v.write_view(
                    root / "views" / f"{base}-o{order}",
                    train,
                    valid,
                    root / "tokenizers/gpt2.json",
                    base,
                    order,
                    hashes,
                )
                print("prepared", base, "order", order, flush=True)


def gptbert_data(root, base, order):
    data = root / "views-gptbert" / f"{base}-o{order}"
    if not (data / "metadata.json").exists():
        shutil.rmtree(data, ignore_errors=True)
        data.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                sys.executable, SCRIPTS / "prepare_gptbert.py",
                "--prepared", root / "views" / f"{base}-o{order}",
                "--output", data,
                "--tokenizer-from", root / "tokenizers/gptbert.json",
            ],
            data.parent / "prepare.log",
        )
    return data


def full_batches(root, base, order):
    """Full GPT-2 batches per epoch; the trainer predicts count - 1 tokens."""
    return (
        read(root / "views" / f"{base}-o{order}" / "manifest.json")["tokens"]["train"]["count"] - 1
    ) // 1024 // 8


def gptbert_tokens(root, base, order):
    return read(gptbert_data(root, base, order) / "metadata.json")["train_tokens"]


def cell(root, model, base, seed, order, name, batches=None, steps_per_epoch=None):
    output = root / "runs" / name
    if (output / "DONE.json").exists():
        return

    # A resumed GPT-2 run can skip its final validation, so unfinished cells start over.
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    view = root / "views" / f"{base}-o{order}"

    if model == "gpt2":
        if batches:
            run(
                [
                    sys.executable, HERE / "fixed_steps.py",
                    "--prepared", view,
                    "--output", output,
                    "--seed", seed,
                    "--batches", batches,
                ],
                output / "train.log",
            )
        else:
            run(
                [
                    sys.executable, SCRIPTS / "run_babyscale_gpt2_small.py", "train",
                    "--prepared", view,
                    "--output", output,
                    "--seed", seed,
                    "--epochs", 20,
                    "--batch-size", 8,
                    "--checkpoint-every", 500,
                    "--device", "cuda",
                ],
                output / "train.log",
            )
        scored = output / "best_model"
        rows = [json.loads(line) for line in open(output / "validation.jsonl") if line.strip()]
        selected = {"best_epoch": min(rows, key=lambda r: r["loss"])["epoch"]}
    else:
        data = gptbert_data(root, base, order)
        run(
            [
                sys.executable, SCRIPTS / "run_gptbert.py",
                "--data", data,
                "--output", output,
                "--seed", seed,
                "--epochs", 20,
                *(["--steps-per-epoch", steps_per_epoch] if steps_per_epoch else []),
            ],
            output / "train.log",
        )
        scored = output / "hf"
        run(
            [
                sys.executable, SCRIPTS / "convert_gptbert_to_hf.py",
                "--run", output,
                "--data", data,
                "--output", scored,
            ],
            output / "eval.log",
        )
        selected = {"best_step": read(next(output.glob("*_best.json")))["global_step"]}

    scores = {}
    for punct in v.PUNCT:
        result = output / f"eval-{punct}.json"
        run(
            [
                sys.executable, SCRIPTS / "evaluate_hf_causal_model.py",
                "--model", scored,
                "--multiblimp", root / f"multiblimp-{punct}.tsv",
                "--output", result,
                "--device", "cuda",
                *(["--trust-remote-code"] if model == "gptbert" else []),
            ],
            output / "eval.log",
        )
        summary = read(result)["summary"]
        assert summary["evaluated_pairs"] == 2056 and summary["skipped_pairs"] == 0
        scores[punct] = summary["accuracy"]

    write(
        output / "DONE.json",
        dict(model=model, base=base, seed=seed, order=order, scores=scores, **selected),
    )
    print("done", name, scores, flush=True)


def name(model, base, seed, order):
    return f"{model}-{base}-seed-{seed}-order-{order}"


def train(args):
    root = args.root
    wanted = lambda model, seed: args.model in (None, model) and args.seed in (None, seed)

    if args.design == "orders":
        for family, bases in v.FAMILIES.items():
            for seed in v.SEEDS:
                for model in MODELS:
                    for base in bases:
                        if not wanted(model, seed):
                            continue
                        budget = {}
                        if family == "fixed-steps" and model == "gpt2":
                            budget["batches"] = min(full_batches(root, b, seed) for b in bases)
                        elif family == "fixed-steps":
                            budget["steps_per_epoch"] = min(
                                gptbert_tokens(root, b, seed) // STEP_TOKENS for b in bases
                            )
                        cell(root, model, base, seed, seed, name(model, base, seed, seed), **budget)

    elif args.design == "crossed":
        for order in v.SEEDS:  # order by order, so each finished order completes a column
            for model in MODELS:
                for seed in v.SEEDS:
                    for base in CROSSED:
                        if wanted(model, seed):
                            cell(root, model, base, seed, order, name(model, base, seed, order))

    else:
        bases = v.FAMILIES["additive"]
        for seed in v.SEEDS:
            for model in MODELS:
                if not wanted(model, seed):
                    continue
                if model == "gpt2":
                    budget = {"batches": min(full_batches(root, b, seed) for b in bases)}
                else:  # rounded up, as in the GPT-BERT runs with equal words
                    budget = {
                        "steps_per_epoch": min(
                            -(-gptbert_tokens(root, b, seed) // STEP_TOKENS) for b in bases
                        )
                    }
                for base in bases:
                    cell(
                        root, model, base, seed, seed,
                        f"equal-updates/{model}-{base}-seed-{seed}",
                        **budget,
                    )


def paired(values):
    values = [x for x in values if x is not None]
    item = {"n": len(values), "values": values}
    if values:
        item["mean"] = statistics.mean(values)
    if len(values) > 1:
        half = T_975[len(values)] * statistics.stdev(values) / len(values) ** 0.5
        item["ci95"] = [item["mean"] - half, item["mean"] + half]
    return item


def score(root, run_name, test):
    path = root / "runs" / run_name / "DONE.json"
    return 100 * read(path)["scores"][test] if path.exists() else None


def summarize_orders(root):
    pairs = [
        (f"punctuation-{s}", f"{s}-keep", f"{s}-strip")
        for s in ["captions", "babybabellm", "mix"]
    ]
    pairs += [(f"captions-vs-bbl-{p}", f"captions-{p}", f"babybabellm-{p}") for p in v.PUNCT]
    pairs += [
        ("additive", "bbl800-cap800", "bbl800-repeat800"),
        ("fixed-steps", "fixed-keep", "fixed-strip"),
        ("scale-2m", "captions-2m-keep", "captions-2m-strip"),
        ("scale-5m", "captions-5m-keep", "captions-5m-strip"),
        ("subtitles-bbl", "bbl-full-500k", "bbl-noos-500k"),
        ("subtitles-mix", "mix-full-500k", "mix-noos-500k"),
        ("captions-with-subtitles", "mix-full-500k", "bbl-full-500k"),
        ("captions-without-subtitles", "mix-noos-500k", "bbl-noos-500k"),
    ]

    results = {}
    for model in MODELS:
        for label, a, b in pairs:
            for test in v.PUNCT:
                rows = []
                for seed in v.SEEDS:
                    first, second = (score(root, name(model, x, seed, seed), test) for x in (a, b))
                    if first is not None and second is not None:
                        rows.append(dict(seed=seed, a=first, b=second, delta=first - second))

                item = dict(a=a, b=b, n=len(rows), seeds=rows)
                if rows:
                    item.update(
                        mean_a=statistics.mean(r["a"] for r in rows),
                        mean_b=statistics.mean(r["b"] for r in rows),
                        delta_pp=statistics.mean(r["delta"] for r in rows),
                    )
                if len(rows) > 1:
                    item["paired_seed_ci95_pp"] = paired([r["delta"] for r in rows])["ci95"]
                results[f"{model}/{label}/test-{test}"] = item

    return results


def summarize_crossed(root):
    from romanian_childlm.analysis.controls import crossed

    contrasts = {
        "punctuation-captions": {"captions-keep": 1, "captions-strip": -1},
        "punctuation-babybabellm": {"babybabellm-keep": 1, "babybabellm-strip": -1},
        # Whether punctuation helps the two sources differently, cell by cell.
        "punctuation-captions-minus-babybabellm": {
            "captions-keep": 1,
            "captions-strip": -1,
            "babybabellm-keep": -1,
            "babybabellm-strip": 1,
        },
        "captions-vs-bbl-keep": {"captions-keep": 1, "babybabellm-keep": -1},
        "captions-vs-bbl-strip": {"captions-strip": 1, "babybabellm-strip": -1},
        "additive": {"bbl800-cap800": 1, "bbl800-repeat800": -1},
    }

    results = {}
    for model in MODELS:
        for label, weights in contrasts.items():
            for test in v.PUNCT:
                grid = {}
                for seed in v.SEEDS:
                    for order in v.SEEDS:
                        scores = {
                            view: score(root, name(model, view, seed, order), test)
                            for view in weights
                        }
                        if None not in scores.values():
                            grid[seed, order] = dict(
                                scores=scores,
                                delta=sum(w * scores[x] for x, w in weights.items()),
                            )

                item = dict(
                    weights=weights,
                    cells=len(grid),
                    grid=[dict(seed=s, order=o, **c) for (s, o), c in grid.items()],
                )
                if len(grid) == len(v.SEEDS) ** 2:
                    deltas = {key: c["delta"] for key, c in grid.items()}
                    item["crossed"] = crossed(deltas)
                    item["order_42_column"] = paired([deltas[s, 42] for s in v.SEEDS])
                    item["diagonal"] = paired([deltas[s, s] for s in v.SEEDS])
                results[f"{model}/{label}/test-{test}"] = item

    return results


def summarize_equal_updates(root, model):
    captions, repetition = v.FAMILIES["additive"]
    key = "best_epoch" if model == "gpt2" else "best_step"
    results = {
        "note": (
            "points; captions = bbl800-cap800, repetition = bbl800-repeat800; "
            "equal tokens = the smaller condition's updates per epoch for both; "
            "equal words = the same seeds with order = seed"
        )
    }

    for test in v.PUNCT:
        rows = []
        for seed in v.SEEDS:
            row = {"seed": seed}
            for base in (captions, repetition):
                equal = root / "runs/equal-updates" / f"{model}-{base}-seed-{seed}" / "DONE.json"
                words = root / "runs" / name(model, base, seed, seed) / "DONE.json"
                row[f"{base}/equal_tokens"] = (
                    100 * read(equal)["scores"][test] if equal.exists() else None
                )
                row[f"{base}/equal_words"] = (
                    100 * read(words)["scores"][test] if words.exists() else None
                )
                row[f"{base}/equal_tokens/{key}"] = read(equal)[key] if equal.exists() else None
                row[f"{base}/equal_words/{key}"] = read(words)[key] if words.exists() else None

            for match in ("equal_tokens", "equal_words"):
                a, b = row[f"{captions}/{match}"], row[f"{repetition}/{match}"]
                row[f"captions - repetition/{match}"] = None if a is None or b is None else a - b

            t, w = (
                row["captions - repetition/equal_tokens"],
                row["captions - repetition/equal_words"],
            )
            row["contrast change (equal_tokens - equal_words)"] = (
                None if t is None or w is None else t - w
            )
            rows.append(row)

        results[f"test-{test}"] = {
            "per_seed": rows,
            **{
                k: paired([r[k] for r in rows])
                for k in (
                    "captions - repetition/equal_tokens",
                    "captions - repetition/equal_words",
                    "contrast change (equal_tokens - equal_words)",
                )
            },
            **{
                f"{base}: equal_tokens - equal_words": paired([
                    None if None in (r[f"{base}/equal_tokens"], r[f"{base}/equal_words"])
                    else r[f"{base}/equal_tokens"] - r[f"{base}/equal_words"]
                    for r in rows
                ])
                for base in (captions, repetition)
            },
        }

    return results


def summarize(args):
    write(args.output / "document-orders.json", summarize_orders(args.root))
    write(args.output / "crossed-grid.json", summarize_crossed(args.root))
    for model in MODELS:
        write(
            args.output / f"equal-updates-{model}.json",
            summarize_equal_updates(args.root, model),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("prepare")
    p.add_argument("--root", type=Path, required=True)
    for option in (
        "--captions-prepared",
        "--bbl-prepared",
        "--corpus-0-4",
        "--corpus-5-8",
        "--bbl-corpus",
    ):
        p.add_argument(option, type=Path, required=True)
    p.add_argument("--benchmark", type=Path, default=Path("data/multiblimp-ron.tsv"))

    t = commands.add_parser("train")
    t.add_argument("--root", type=Path, required=True)
    t.add_argument("--design", choices=["orders", "crossed", "equal-updates"], required=True)
    t.add_argument(
        "--model",
        choices=MODELS,
        help="only this model, for example to use several GPUs",
    )
    t.add_argument("--seed", type=int, choices=v.SEEDS, help="only this training seed")

    s = commands.add_parser("summarize")
    s.add_argument("--root", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    {"prepare": prepare, "train": train, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
