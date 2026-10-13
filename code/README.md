# Training and evaluation

Run these commands from the archive's root directory after installation.

## GPT-BERT setup

```bash
git clone https://github.com/babylm-org/multilingual-training.git ../multilingual-training
git -C ../multilingual-training checkout e6bb176f4ae5eac3d4b660a480a94182146eac6d
python code/scripts/setup_gptbert.py --upstream ../multilingual-training
```

Setup applies our training patch and builds the local converter in
`code/vendor/gpt_bert/`. Upstream code is not bundled; see `LICENSE`.

## GPT-2 small

Retrieve captions for `data/manifest-*.jsonl` through authorized platform
access. Save one JSONL per band with `video_id` and `text` on each line.
Keep input punctuation; preparation removes it on a copy. Use the manifests
to check coverage. Optional `clean_words` must match the normalized text.

```bash
python code/scripts/run_babyscale_gpt2_small.py prepare \
  --corpus-0-4 rehydrated/corpus-0-4.jsonl \
  --corpus-5-8 rehydrated/corpus-5-8.jsonl \
  --allow-other-corpora \
  --strip-punctuation \
  --output prepared

python code/scripts/run_babyscale_gpt2_small.py train \
  --prepared prepared \
  --output run-42 \
  --seed 42 \
  --batch-size 8

python -m romanian_childlm.training.evaluate_babyscale \
  --run run-42 \
  --prepared prepared \
  --benchmark data/multiblimp-ron.tsv
```

Use seeds `42`, `0`, `123`, `321`, `9000` and separate output directories
for each preparation and run. `--allow-other-corpora` permits re-serialized
input files; the checks for duplicate IDs and supplied word counts still apply.

- Age runs: add `--bands 0-4` or `--bands 5-8`, plus `--tokenizer-from prepared`
  to preparation.
- Matched captions: add `--max-words 956285 --vocab-size 30000` to preparation,
  without `--tokenizer-from`. Whole documents may exceed the target slightly.
- Ten-epoch schedule: add `--epochs 10` to training; the default is 20.

The primary seed for document selection is 42. Condition names are in
[results/README.md](../results/README.md).

## GPT-2 17M

```bash
python code/scripts/replicate_babybabellm_recipe.py \
  --corpus rehydrated/corpus-0-4.jsonl,rehydrated/corpus-5-8.jsonl \
  --ids-from prepared \
  --output run-17m-42 \
  --seed 42

python code/scripts/evaluate_hf_causal_model.py \
  --model run-17m-42 \
  --multiblimp data/multiblimp-ron.tsv \
  --output run-17m-42/evaluation.json
```

This four-layer model uses an 8,192-token vocabulary, a 95/5 fragment split
and the final checkpoint after 10 epochs. Punctuation is retained.
`--ids-from` selects documents only, not their normalization or split.

## GPT-BERT

Use a 16,384-token vocabulary and a 1:1 ratio of masked and causal objectives:

```bash
python code/scripts/prepare_gptbert.py \
  --prepared prepared \
  --output prepared-gptbert

python code/scripts/run_gptbert.py \
  --data prepared-gptbert \
  --output run-gptbert-42 \
  --seed 42 \
  --epochs 20

python code/scripts/convert_gptbert_to_hf.py \
  --run run-gptbert-42 \
  --data prepared-gptbert \
  --output run-gptbert-42-hf

python code/scripts/evaluate_hf_causal_model.py \
  --model run-gptbert-42-hf \
  --multiblimp data/multiblimp-ron.tsv \
  --output run-gptbert-42/evaluation.json \
  --strip-punctuation \
  --trust-remote-code
```

For age and scaling runs, reuse the combined tokenizer with
`prepare_gptbert.py --tokenizer-from prepared-gptbert/tokenizer.json`.
The comparison with matched budgets instead trains a tokenizer on each corpus.

## BabyBabelLM comparison

Request access to the
[Romanian dataset](https://huggingface.co/datasets/BabyLM-community/babylm-ron)
and run `hf auth login`. Use the document-level parquet, not
`training_text.txt`, to preserve document boundaries and punctuation.

```bash
hf download BabyLM-community/babylm-ron \
  --repo-type dataset --revision caf295e824ff0fb6f071ea635fdb7aced75c1b1b \
  --local-dir babybabellm-source data/train-00000-of-00001.parquet

python code/scripts/build_babybabellm_official_corpus.py \
  --input babybabellm-source/data/train-00000-of-00001.parquet \
  --output babybabellm-source/corpus.jsonl

python code/scripts/run_babyscale_gpt2_small.py prepare \
  --corpus-0-4 babybabellm-source/corpus.jsonl --bands 0-4 \
  --allow-other-corpora --strip-punctuation --vocab-size 30000 \
  --output prepared-babybabellm
```

Expected: 18,739 documents, 956,285 normalized words. Here `--bands 0-4`
selects an input slot, not an age label. Use `prepared-babybabellm` in the
GPT-2 small commands above. For GPT-BERT, prepare it with a fresh tokenizer.

For 17M, use the tokenizer from the
[released Romanian baseline](https://huggingface.co/BabyLM-community/ron-baseline-small):

```bash
hf download BabyLM-community/ron-baseline-small \
  --revision 34d70b0fbcb15765a8ed7a43a097651c84d4576b \
  --local-dir babybabellm-tokenizer \
  tokenizer.json tokenizer_config.json special_tokens_map.json

python code/scripts/replicate_babybabellm_recipe.py \
  --corpus babybabellm-source/corpus.jsonl \
  --tokenizer-dir babybabellm-tokenizer \
  --output run-babybabellm-17m-42 --seed 42
```

Evaluate without `--strip-punctuation`. Use the released tokenizer, but train
the model from scratch; no pretrained model weights enter this replication.

## Appendix analyses and experiments

`src/romanian_childlm/analysis/` reproduces the appendix numbers from the
saved results; [results/appendix/README.md](../results/appendix/README.md)
lists the commands. `pip install -e "code[analysis]"` adds scipy for the
t-tests and crossed intervals, and `code[stanza]` the tagger for the corpus
person counts. The code for the appendix experiments is in
[experiments/](experiments/README.md).

## Draws from shared channels

`scripts/select_shared_channel_draws.py` selects the draws for the analysis of sensitivity
to media source from the two full subsets:

```bash
python code/scripts/select_shared_channel_draws.py CORPUS_ROOT OUT_DIR 42 1 2 3 4
```

`CORPUS_ROOT` holds `0-4/transcripts/corpus.jsonl` and `5-8/transcripts/corpus.jsonl`. For
each seed and profile the script writes `matched-<profile>-seed-<seed>.jsonl` to `OUT_DIR`,
with the lines copied unchanged from the corpus.

Both profiles draw only from the 27 channels found under both, and each channel gets the same
word quota in both: 2,000,000 x min(w04, w58) / 2,857,135, rounded down, where w04 and w58 are
the channel's clean words in each full subset and 2,857,135 is the sum of these minima. Each
profile and draw uses one `random.Random(seed)`. The script goes through the shared channels
in name order, sorts each channel's documents by video ID and shuffles them. No draw uses two
of the shared channels, but the script still shuffles them, so they advance the generator.
In the other 25 channels the script takes documents in shuffled order until the quota is
reached, skipping any document that would end more than 500 words above it.

The script reproduces the ten selections behind the reported runs byte for byte.
