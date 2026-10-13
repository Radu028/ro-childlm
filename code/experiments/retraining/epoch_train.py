"""Run the unchanged GPT-2 small trainer and save the model after chosen epochs."""

import sys
from pathlib import Path

from romanian_childlm.training import babyscale_gpt2 as core

every, final_epoch = int(sys.argv[1]), int(sys.argv[2])
assert sys.argv[3] == "--"
train_arguments = sys.argv[4:]
output = Path(train_arguments[train_arguments.index("--output") + 1])
assert not (output / "validation.jsonl").exists(), "fresh runs only"
original_validation_loss = core.validation_loss


def validation_loss_and_save(model, *args, **kwargs):
    value = original_validation_loss(model, *args, **kwargs)

    # The trainer appends this epoch's line to validation.jsonl after we return; runs are
    # never resumed, so the line count gives the epoch.
    log = output / "validation.jsonl"
    epoch = (sum(1 for line in log.open() if line.strip()) if log.exists() else 0) + 1
    if (every and epoch % every == 0) or epoch == final_epoch:
        model.save_pretrained(output / "epoch_models" / f"epoch-{epoch:03d}")
    return value


core.validation_loss = validation_loss_and_save
sys.argv = ["run_babyscale_gpt2_small.py", "train", *train_arguments]
core.main()
