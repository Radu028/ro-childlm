# Appendix experiments

Code for the experiments in Appendices F, G and I. Each folder has a README with
the protocol and the analysis rules fixed before the runs.

- [retraining/](retraining/README.md): GPT-2 small retrained with every epoch scored (Appendix F).
- [controlled-comparisons/](controlled-comparisons/README.md): punctuation, additive, document order
  and seed x order runs, plus runs with equal maximum updates (Appendix G).
- [asr-pilot/](asr-pilot/README.md): the sample of the pilot transcription audit (Appendix I).

The first two need the rehydrated captions, the preparations of
[code/README.md](../README.md) and a GPU; GPT-BERT also needs `setup_gptbert.py`.
Run everything from the archive root. The outputs are already in
`results/appendix/`, so you can check the paper's numbers without
retraining.
