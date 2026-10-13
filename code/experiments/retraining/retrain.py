"""Retrain a GPT-2 small run with every epoch scored, or collect finished runs."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=1) + "\n")


def run(command, log):
    with open(log, "a") as file:
        subprocess.run(
            [str(c) for c in command],
            check=True,
            stdout=file,
            stderr=subprocess.STDOUT,
        )


def validation(run_dir):
    return [json.loads(line) for line in open(run_dir / "validation.jsonl") if line.strip()]


def evaluate(run_dir, args):
    if not (
        (run_dir / "multiblimp_eval.json").exists()
        and (run_dir / "multiblimp_details.csv").exists()
    ):
        run(
            [
                sys.executable, "-m", "romanian_childlm.training.evaluate_babyscale",
                "--run", run_dir,
                "--prepared", args.prepared,
                "--benchmark", args.benchmark,
                "--device", args.device,
            ],
            run_dir / "eval.log",
        )

    result = read(run_dir / "multiblimp_eval.json")["multiblimp"]
    assert result["evaluated_pairs"] == 2056 and result["skipped_pairs"] == 0
    return result


def retrain(args):
    reference, output = args.reference, args.output
    config = read(reference / "run_config.json")
    assert (
        read(args.prepared / "manifest.json")["identity_sha256"] == config["preparation_identity"]
    ), "--prepared is not the preparation of the reference run"
    if (output / "DONE.json").exists():
        return

    epochs = config["epochs"]
    wanted = [
        e for e in range(1, epochs + 1) if (args.every and e % args.every == 0) or e == epochs
    ]
    if not (output / "training_complete.json").exists():
        # A resumed run can skip its final validation, so unfinished runs start over.
        shutil.rmtree(output, ignore_errors=True)
        output.mkdir(parents=True)
        run(
            [
                sys.executable, HERE / "epoch_train.py", args.every, epochs, "--",
                "--prepared", args.prepared,
                "--output", output,
                "--seed", config["seed"],
                "--epochs", epochs,
                "--batch-size", config["batch_size"],
                "--sequence-length", config["sequence_length"],
                "--learning-rate", config["learning_rate"],
                "--weight-decay", config["weight_decay"],
                "--checkpoint-every", config["checkpoint_every"],
                "--device", args.device,
            ],
            output / "train.log",
        )
    assert read(output / "run_config.json") == config, "run_config differs from the reference run"

    step_of = {r["epoch"]: r["global_step"] for r in validation(output)}
    curve = []
    for epoch in wanted:
        scored = output / "epoch_evals" / f"epoch-{epoch:03d}"
        if not (scored / "multiblimp_details.csv").exists():
            scored.mkdir(parents=True, exist_ok=True)
            model = scored / "best_model"
            shutil.rmtree(model, ignore_errors=True)
            shutil.copytree(output / "epoch_models" / f"epoch-{epoch:03d}", model)
            for name in TOKENIZER_FILES:
                if (output / "best_model" / name).exists():
                    shutil.copy2(output / "best_model" / name, model / name)
            for name in ("run_config.json", "training_complete.json"):
                shutil.copy2(output / name, scored / name)
            # best_validation_loss in this folder's multiblimp_eval.json refers to the whole run.
            evaluate(scored, args)
            shutil.rmtree(model)

        result = read(scored / "multiblimp_eval.json")["multiblimp"]
        curve.append({
            "epoch": epoch,
            "global_step": step_of[epoch],
            "accuracy": result["accuracy"],
            "by_phenomenon": {k: v["accuracy"] for k, v in result["by_phenomenon"].items()},
        })

    with open(output / "curve.jsonl", "w") as file:
        file.writelines(json.dumps(row) + "\n" for row in curve)

    selected = evaluate(output, args)
    write(output / "DONE.json", {
        "epochs_evaluated": wanted,
        "selected_accuracy": selected["accuracy"],
        "reference_selected_accuracy": (
            read(reference / "multiblimp_eval.json")["multiblimp"]["accuracy"]
        ),
        "final_accuracy": curve[-1]["accuracy"],
    })
    shutil.rmtree(output / "epoch_models", ignore_errors=True)  # regenerable


def collect(args):
    runs = {}
    for output in sorted(args.runs, key=lambda p: p.name):
        done = read(output / "DONE.json")
        curve = [json.loads(line) for line in open(output / "curve.jsonl")]
        runs[output.name] = {
            "curve": [
                [r["epoch"], r["global_step"], r["accuracy"], r["by_phenomenon"]] for r in curve
            ],
            "ref": done["reference_selected_accuracy"],
            "sel": done["selected_accuracy"],
        }
    write(args.output, runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    one = commands.add_parser("run")
    one.add_argument("--reference", type=Path, required=True, help="the original run's folder")
    one.add_argument("--prepared", type=Path, required=True)
    one.add_argument("--output", type=Path, required=True, help="name it CONDITION-seed-SEED")
    one.add_argument(
        "--every", type=int, default=1, help="save every k epochs; 0 = final epoch only"
    )
    one.add_argument("--benchmark", type=Path, default=Path("data/multiblimp-ron.tsv"))
    one.add_argument("--device", default="cuda")

    many = commands.add_parser("collect")
    many.add_argument("runs", type=Path, nargs="+")
    many.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    (retrain if args.command == "run" else collect)(args)


if __name__ == "__main__":
    main()
