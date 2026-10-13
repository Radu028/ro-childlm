# Results

The three JSONL files contain one record per condition and seed.
`summary.jsonl` is GPT-2 small; the other filenames identify their models.
Accuracies in them use a 0-1 scale. `aggregate.py` prints percentages and
sample SDs.

The retraining runs with scores after every epoch, the controlled comparisons,
scores for each pair and the inputs and outputs of the appendix analyses are in
[appendix/](appendix/README.md).

## Main conditions

| Paper condition | GPT-2 small, 20 ep. | GPT-2 17M | GPT-BERT, 20 ep. |
| --- | --- | --- | --- |
| Captions 10M | `combined-10m-nopunct` | `ours10m17` | `combined-10m` |
| Ages 0-4, 5M | `kids04-5m-nopunct` | `kids04-5m17` | `kids04-5m-ctok` |
| Ages 5-8, 5M | `kids58-5m-nopunct` | `kids58-5m17` | `kids58-5m-ctok` |
| Captions 1M | `ours-1m-30k` | `ours1m17-d0` | `ours1m-own` |
| BabyBabelLM 1M | `babybabellm-off-1m` | `bbl17` | `bbl-1m` |

The GPT-2 small ten-epoch conditions are `combined-10m-ep10`, `kids04-5m-ep10`,
`kids58-5m-ep10`, `ours-1m-ep10` and `babybabellm-off-1m-ep10`.
GPT-BERT uses an `ep10-` prefix. These are separately trained schedules.

- Matched caption draws: GPT-2 small uses `ours-1m-s1` through `s4`, also with
  `-ep10`; 17M uses `ours1m17-d0` through `d4`; GPT-BERT uses `ours1m-own`
  plus `-s1` through `-s4`. Unsuffixed GPT-2 small and GPT-BERT runs are the first draw.
- Scaling: `scale`, `scale04`, `scale58`. Targets are nominal word counts.
  GPT-BERT `scale-1m` shares the scaling tokenizer; `ours1m-own` does not.
- Controls on shared channels: `matched04-d0` through `d4` and `matched58-d0`
  through `d4`, with model/schedule prefixes or suffixes. These are separate draws.
- Fixed updates: GPT-2 small `scale-1m-fc`, `scale-5m-fc`; GPT-BERT `fc-1m`, `fc-5m`.
- Shared tokenizer: GPT-2 small `ours-1m-mixtok`, `babybabellm-1m-mixtok`.

The published BabyBabelLM score of 74.1 is external, not one of these runs.
Our evaluator gave the released model 74.12 (`appendix/babybabellm-baseline.json`).

## Scoring

Scores sum continuation log probabilities after a space delimiter and BOS
context token. Ties count as errors. The GPT-2 small and GPT-BERT evaluations use
the training normalization; 17M retains punctuation. All 260 GPT-BERT
records use normalized evaluations of the best checkpoints.
`benchmark_sha256` identifies the original TSV. GPT-BERT also records
hashes of the normalized benchmark, tokenizer and checkpoint.

## Figures and scores by phenomenon

Run from the archive root with the code installed:

```bash
python -m romanian_childlm.analysis.multiblimp_table
pip install matplotlib
python results/figures/plot.py --output figures
```

`multiblimp_table` prints scores by phenomenon for any conditions;
`romanian_childlm.analysis.phenomena` gives Table 3 with macro averages.
`plot.py` redraws both panels of Figure 1 (`scaling-curve.png`,
`scaling-gbert.png`) from the scaling CSVs, which use percentages.
