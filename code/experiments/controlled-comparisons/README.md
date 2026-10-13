# Controlled comparisons

Appendix G: punctuation, BabyBabelLM plus captions against BabyBabelLM twice,
and the controls with equal maximum steps and updates, for GPT-2 small and
GPT-BERT base. Results are in `results/appendix/`:

- `document-orders.json`: five seeds, document order equal to the seed.
- `crossed-grid.json`: every training seed with every document order.
- `crossed-extra-contrasts.json`: comparisons with BabyBabelLM alone, from the same grid.
- `equal-updates-gpt2.json`, `equal-updates-gptbert.json`: the additive pair at equal updates.

`python -m romanian_childlm.analysis.controls` prints Table 10 from these files.

## Files

- `views.py`: document pools, sampling, the views, shared tokenizers and the two MultiBLiMP files.
- `run.py`: `prepare`, `train` (one design at a time) and `summarize`.
- `fixed_steps.py`: GPT-2 with a fixed number of full batches per epoch.

```bash
python code/experiments/controlled-comparisons/run.py prepare --root suite \
  --captions-prepared prepared --bbl-prepared prepared-babybabellm \
  --corpus-0-4 rehydrated/corpus-0-4.jsonl --corpus-5-8 rehydrated/corpus-5-8.jsonl \
  --bbl-corpus babybabellm-source/corpus.jsonl
python code/experiments/controlled-comparisons/run.py train --root suite --design orders
python code/experiments/controlled-comparisons/run.py train --root suite --design crossed
python code/experiments/controlled-comparisons/run.py train --root suite --design equal-updates
python code/experiments/controlled-comparisons/run.py summarize --root suite --output summaries
```

`prepared` is the Combined 10M preparation with `--strip-punctuation` and
`prepared-babybabellm` the BabyBabelLM one (code/README.md). Scoring uses
`evaluate_hf_causal_model.py`; GPT-BERT runs through `run_gptbert.py` after
`setup_gptbert.py`. `prepare` prints a note when an input split differs from
the split used for the reported runs. With the same splits it rebuilds both
MultiBLiMP files byte for byte, and the additive views have the token counts
given under "Equal updates" below. Preparation identities differ from those of
the reported runs, so compare splits and token counts instead.

## Views

- 800k training words of captions, of BabyBabelLM, or 400k of each (`mix`),
  with punctuation kept or stripped. Keep and strip views hold exactly the same
  documents, prefixes, words and order; only punctuation changes.
- Documents come from the train and validation splits of the two preparations;
  punctuation is restored from the corpus files and checked against the stripped
  split. Shuffled whole documents are taken until the budget is reached, and the
  last one is cut to fit.
- BabyBabelLM samples keep the training pool's proportions of books, CHILDES
  and OpenSubtitles to within one word.
- NFC, cedilla to comma below and known nonverbal markers removed in both
  forms; case and word-internal hyphens and apostrophes kept.
- Validation: the same 80k caption and 80k BabyBabelLM words from held-out
  documents in every view, in the view's punctuation form.
- One tokenizer per architecture (30k GPT-2, 16,384 GPT-BERT), trained once on
  both punctuation forms of the 400k + 400k mixture, never on validation.
- Additive: the BabyBabelLM 800k sample plus 800k caption words, against the
  same 800k twice. Both see 1.6M words per epoch with punctuation kept. This
  controls word exposure, not tokens or updates.
- Fixed steps: the 800k caption view with and without punctuation, trained for
  the same number of steps (the smaller budget of the pair), both validated on
  punctuated text. These are 20 virtual epochs, not 20 passes over both views.
- Also prepared and reported in `document-orders.json`: 2M and 5M captions with
  and without punctuation, and 500k BabyBabelLM with and without OpenSubtitles,
  alone and in a 50/50 mixture with captions. Removing subtitles gives their
  budget to books and CHILDES; it does not isolate subtitle quality.

The document order shuffles the training and validation documents of a view.
For GPT-2 it decides which documents share a 1,024-token block and which tail
is dropped; for GPT-BERT, the 128-token segments and which incomplete batches
are dropped. Through the validation documents it also affects checkpoint
selection.

## Training and scoring

The usual recipes, 20 epochs, seeds 42, 0, 123, 321 and 9000, selection by
validation loss. Every model is scored on all 2,056 MultiBLiMP pairs with
punctuation kept and stripped; punctuation kept is the primary test. No
setting, seed or stopping time is chosen from test scores.

## Document order (five seeds)

This design shuffles each run's documents with its own training seed. Paired
differences by seed, mean and 95% t interval, both test forms, are reported
whatever their direction. The intervals describe seed variation, not
uncertainty over linguistic items; secondary contrasts are exploratory.

## Seed by order grid

Training seed and document order crossed, 5 x 5 cells, for captions and
BabyBabelLM with punctuation kept and stripped and the two additive views. Nine
cells per grid, the diagonal (the five-seed runs above) and the column with
order 42, were trained first; the analysis was fixed before the other 16:

- The primary result for each contrast, model and test form is the grand mean
  of the 25 paired differences; test-keep is primary.
- The 95% interval is mean +- t(0.975, df) x se from the two-way crossed model
  with variance components for seed, order and residual (method of moments,
  negative components set to zero, Satterthwaite degrees of freedom). The
  residual includes the seed x order interaction and run noise. Intervals are
  pointwise and do not cover uncertainty over benchmark items.
- Contrasts: punctuation (keep minus strip) for captions and for BabyBabelLM;
  captions minus BabyBabelLM in each form; additive. Whether punctuation helps
  the two sources differently is read from the difference of their
  punctuation effects in each cell, never from one interval excluding zero and the other not.
- Also reported: the full grid, the column with document order 42 and the diagonal as
  five-seed paired t intervals. The 25 cells are not 25 independent seeds and
  are not merged with other designs as more seeds.

The comparisons with BabyBabelLM alone (`crossed-extra-contrasts.json`) were
not specified in advance and are exploratory.

## Equal updates

The additive runs match words, not tokens: captions tokenize shorter, so the
GPT-2 additive view has 2,268 training blocks against 2,659 for repetition
(5,680 against 6,660 updates), and GPT-BERT runs 74 against 87 steps per epoch
(1,480 against 1,740). Here both views train on the smaller count, with order
equal to the seed:

- GPT-2: 283 full batches of 8 x 1,024 tokens per epoch, 5,660 updates, with
  `fixed_steps.py`. Each epoch takes the first 283 full batches of the
  trainer's own shuffled order, so the repetition view sees a different 85% of
  its blocks each epoch.
- GPT-BERT: 74 steps per epoch, 1,480 steps, with only the step budget and
  validation interval changed (`run_gptbert.py --steps-per-epoch`). The captions run repeats its
  recipe at equal words; the repetition run gets 15% fewer tokens, a learning rate
  schedule shortened to 1,480 steps instead of 1,740 and validation every 74
  steps, so this is a control at equal maximum updates, not a pure token change.

Reported per test form: captions minus repetition at equal updates and at
equal words (the diagonal runs), per seed, with mean and 95% t interval; the
paired change between the two; each view's result at equal updates minus its result at equal words;
and the selected epoch (GPT-2) or step (GPT-BERT) of every model, since equal
budgets can still select models after different numbers of updates.
