# Retraining with scores after every epoch

Appendix F. Each job retrains an existing GPT-2 small condition with its
original preparation, seed and `run_config.json`, saves the model after the
chosen epochs and scores every saved model with the unchanged
`evaluate_babyscale`. The trainer is imported, not edited (`epoch_train.py`).
Results: `results/appendix/retraining.json`.

## Runs

Seeds 0, 42, 123, 321 and 9000 for each condition:

- `ours-1m-30k` and `babybabellm-off-1m`, 20 epochs, every epoch.
- `scale-1m-fc` and `scale-5m-fc`, only the final checkpoint and the model selected
  by validation loss. Their maximum budgets are approximately, not exactly, equal
  (20,460 and 20,254 updates).
- `ours-1m-ep10` and `babybabellm-off-1m-ep10`, 10 epochs, every epoch.
- `kids04-5m-nopunct` and `kids58-5m-nopunct`, 20 epochs, every epoch.

```bash
python code/experiments/retraining/retrain.py run \
  --reference runs/ours-1m-30k-seed-42 --prepared prepared-ours-1m-30k \
  --output retraining/ours-1m-30k-seed-42 --every 1
python code/experiments/retraining/retrain.py collect retraining/* \
  --output retraining.json
```

`--reference` is the original run's folder and `--prepared` its preparation;
use `--every 0` for the two conditions with fixed update budgets. `collect` stores
each run's accuracy at the checkpoint selected by validation loss next to the
original run's.

## Analysis rules, fixed before the runs

- Selection by validation loss stays the primary result.
- The final checkpoint is reported for every condition as a sensitivity analysis.
- All curves and all seeds are reported. No epoch is chosen from MultiBLiMP scores.
- Accuracy over all pairs stays primary; the macro average and results by phenomenon are secondary.
- Source differences and the changes from the selected to the final checkpoint are
  reported with uncertainty across seeds, whatever their direction.
- The 1M and 5M runs with fixed update budgets are compared at their final
  checkpoints with the difference in updates reported. This is not a strict control
  for equal updates or equal tokens.

```bash
python -m romanian_childlm.analysis.retraining
```

prints Table 9 and the other numbers of Appendix F from the saved results, including
the mean accuracy of each condition next to that of the original runs.
