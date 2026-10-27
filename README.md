# Romanian ChildLM experiments

Code and results for [Romanian Small Language Models Trained on Native
Child-Targeted Media](https://openreview.net/forum?id=JitnCb895x), by Radu-Stefan Popa,
Adina Camelia Bleotu and Sergiu Nisioi. NeurIPS 2026 Workshop on Linguistic
Principles for Foundation Models (LP4FM), accepted poster, non-archival.

We collect 10M words of Romanian captions from child-targeted videos, starting
from YouTube Kids, train GPT-2 and GPT-BERT on them under four protocols, and
evaluate agreement accuracy on Romanian MultiBLiMP.

- With GPT-BERT, accuracy rises from 68.2% at 1M words to 74.7% at 10M.
- At a matched 1M-word budget, models trained on the BabyBabelLM Romanian corpus
  outperform models trained on caption subsamples. Most of this gap comes from
  subject-verb person agreement.
- Added to BabyBabelLM, the captions help GPT-BERT more than repeating the
  BabyBabelLM sample.

## Contents

- [code/](code/README.md): training and evaluation for all three models.
- [code/experiments/](code/experiments/README.md): the retraining runs, controlled comparisons and
  transcription audit reported in the appendices.
- [data/](data/README.md): corpus identifiers, statistics and Romanian MultiBLiMP.
- [results/](results/README.md): results by seed, aggregation and figure data;
  [results/appendix/](results/appendix/README.md) holds the appendix results.

## Check the results without a GPU

From this directory, with Python 3.10 or newer:

```bash
python results/aggregate.py results/summary.jsonl
python results/aggregate.py results/gpt2-17m-summary.jsonl
python results/aggregate.py results/gptbert-summary.jsonl
```

These commands use only the standard library. They compute means and sample SDs
from saved evaluations. The appendix analyses also run without a GPU, from saved
pair scores and counts (two of them need scipy):

```bash
PYTHONPATH=code/src python -m romanian_childlm.analysis.lexical
PYTHONPATH=code/src python -m romanian_childlm.analysis.asr_pilot
```

[results/appendix/README.md](results/appendix/README.md) lists all of them.

## Train and evaluate

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e "code[babybabellm]"
```

Follow [code/README.md](code/README.md). Training needs a GPU and separately
obtained text; GPT-BERT also needs upstream code. Captions and model weights
are not included. Missing or changed captions may prevent exact reproduction.

The reported GPT-2 small runs used Linux, RTX 5090, Python 3.10.12,
torch 2.13.0+cu130, transformers 5.15.1 and tokenizers 0.22.2.
Transformers and tokenizers are pinned; other dependencies have minimum versions.

`sha256sum -c SHA256SUMS` checks the archive's file contents.
Licenses and third-party notices are in [LICENSE](LICENSE).
