# Appendix results

Results behind the appendices that the three JSONL summaries do not hold.
Condition names follow [../README.md](../README.md). Protocol keys are
`gpt2_small_20ep`, `gpt2_small_10ep`, `gpt2_17m` and `gptbert_20ep`.

## Files

Results by phenomenon and person agreement (Appendix B):

- `phenomena.json`: accuracy by phenomenon and macro accuracy, and the phenomenon contributions to the matched 1M gap. Percent.
- `person-models.json`: SV-P pairs by the grammatical verb's person, BabyBabelLM minus captions. Points.
- `person-corpus.json`: finite verbs by person and number in the two 1M training splits (Stanza counts).

Lexical coverage (Appendix D):

- `item-scores.jsonl`: one record per run of the 24 matched 1M conditions behind B and D,
  with `correct` (0 or 1) and `margin` (grammatical minus ungrammatical log probability,
  as recorded by the evaluator) for every benchmark row, row 1 first. A pair is correct
  when its margin is positive; ties count as errors. `accuracy` is 0-1 and equals the
  summary files.
- `lexical.json`: coverage on the training splits and the groups by target form for GPT-2 small
  (Table 5).
- `lexical-models.json`: the same groups for GPT-BERT and GPT-2 17M.
- `lexical-draws.json`: pairs whose target form both inputs contain, caption draws 1-4.
- `closeness.json`: bigram shares, the pairs with equal closeness and the regression intercepts.

Checkpoints (Appendix E):

- `selected-checkpoints.csv`: epoch and update of every checkpoint selected by validation loss and
  the schedule's maximum, one row per run, including the runs with fixed update budgets.
- `checkpoints.json`: their ranges over seeds (Table 8).

Retraining (Appendix F):

- `retraining.json`: 40 fresh GPT-2 small runs. Per run, the curve over epochs
  (`[epoch, update, accuracy, accuracy by phenomenon]`), the accuracy of the checkpoint
  selected by validation loss (`sel`) and the original run's (`ref`). 0-1.

Controlled comparisons (Appendix G, Table 10), all in points:

- `document-orders.json`: five seeds with document order equal to the seed, paired t intervals.
- `crossed-grid.json`: the 25 seed x order cells per contrast, with the interval from the crossed model.
- `crossed-extra-contrasts.json`: BabyBabelLM alone, and adding captions or a second copy to it.
- `equal-updates-gpt2.json`, `equal-updates-gptbert.json`: the additive pair at equal updates
  (5,660 for GPT-2 small, 1,480 for GPT-BERT) and at equal words.

Pilot transcription audit (Appendix I):

- `asr-pilot-windows.csv`: alignment counts for each window, agreement-bearing words and
  audio label.
- `asr-agreement.csv`: the 34 classified caption differences on agreement-bearing words, with
  `word_end` and `second_person_i` flags for the 9 feature changes.
- `asr-agreement-checks.csv`: the two 60-word checks of the agreement-bearing selection.
- `asr-pilot.json`: Table 12 and the counts for the sample and agreement, written by `asr_pilot`.

Released baseline (Appendix J):

- `babybabellm-baseline.json`: our score for the released BabyBabelLM Romanian model.

The retraining runs are fresh models, not the checkpoints behind Table 1. The
controlled comparisons use their own views, tokenizer and punctuated MultiBLiMP,
so their scores are not comparable with Table 1.

## Commands

From the archive root. The analyses need only the standard library, except
`controls` and `profiles` (scipy) and the tagging step of `person_corpus`
(Stanza and the rehydrated corpus). Without installing the code, prefix each
command with `PYTHONPATH=code/src`.

```bash
python -m romanian_childlm.analysis.phenomena      # Table 3, gap contributions, macro
python -m romanian_childlm.analysis.person         # third versus first/second person
python -m romanian_childlm.analysis.person_corpus --counts results/appendix/person-corpus.json
python -m romanian_childlm.analysis.draws          # Appendix C draw ranges and 0-4 shares
python -m romanian_childlm.analysis.lexical        # Table 5 and the other architectures and draws
python -m romanian_childlm.analysis.closeness      # bigram closeness
python -m romanian_childlm.analysis.scaling        # Figure 1 slopes and R^2
python -m romanian_childlm.analysis.profiles       # profile t-tests and Table 7
python -m romanian_childlm.analysis.checkpoints    # Table 8
python -m romanian_childlm.analysis.retraining     # Table 9 and Appendix F
python -m romanian_childlm.analysis.controls       # Table 10
python -m romanian_childlm.analysis.asr_pilot      # Table 12, sample and agreement counts
```

`phenomena`, `person`, `lexical`, `closeness`, `checkpoints`, `controls` and
`asr_pilot` rewrite their JSON files here; `lexical` takes about four minutes.
In `person` and `lexical` (and `closeness`), intervals resample UD source
sentences (1,204 for the 2,056 pairs), conditional on the trained models, with
seed 0: 10,000 resamples, 5,000 in `closeness`. The bootstrap of `asr_pilot`
instead resamples windows within strata, 10,000 times with seed 42.

To recount persons in the corpus, pass `--corpus-0-4`, `--corpus-5-8`,
`--bbl-corpus` and `--bbl-train` to `person_corpus` instead of `--counts`.
`romanian_childlm.analysis.coverage` rebuilds `data/benchmark-coverage.csv`;
like the recount, it needs the caption text (data/README.md).
