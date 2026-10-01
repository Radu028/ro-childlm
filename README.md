# Romanian ChildLM experiments

Anonymous supplementary code and results for the LP4FM submission.

- [code/](code/README.md): training and evaluation for all three models.
- [data/](data/README.md): corpus identifiers, statistics and Romanian MultiBLiMP.
- [results/](results/README.md): per-seed results, aggregation and figure data.

## Check the results without a GPU

From this directory, with Python 3.10 or newer:

```bash
python results/aggregate.py results/summary.jsonl
python results/aggregate.py results/gpt2-17m-summary.jsonl
python results/aggregate.py results/gptbert-summary.jsonl
```

These standard-library commands compute means and sample SDs from saved
evaluations. They do not rerun the models.

## Train and evaluate

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e "code[babybabellm]"
```

Follow [code/README.md](code/README.md). Training needs a GPU and separately
obtained text; GPT-BERT also needs upstream code. Captions and model weights
are not included. Missing or changed captions may prevent exact reproduction.

The reported 124M runs used Linux, RTX 5090, Python 3.10.12,
torch 2.13.0+cu130, transformers 5.15.1 and tokenizers 0.22.2.
Transformers and tokenizers are pinned; other dependencies have minimum versions.

`sha256sum -c SHA256SUMS` checks the archive's file contents.
Licenses and third-party notices are in [LICENSE](LICENSE).
