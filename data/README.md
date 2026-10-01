# Data

- `manifest-*.jsonl`: one record per caption document, with IDs and provenance.
- `stats-*.json`: corpus statistics from the appendix.
- `benchmark-overlap.json`: word coverage and target-form diagnostics.
- `multiblimp-ron.tsv`: unchanged Romanian MultiBLiMP 1.0 data.

## Reading the manifests

`age_band` is the recorded discovery profile, not a speaker's age. Each video
appears once. Where both profiles yielded an eligible copy, 0-4 took precedence.

`discovery_method`, `discovery_query`, `discovery_rank` and `discovery_date`
record where and when the video was found. Methods beginning with
`kids_fixed_search`, `kids_home`, `kids_watch` or `kids_category` refer to
YouTube Kids. Methods containing `channel` mean enumeration of a channel
first confirmed there. Older records may use equivalent unprefixed labels.

`caption_kind` distinguishes creator captions from automatic captions.
`caption_language` is the provider's label, `ro` or `asr-ro`; six automatic
tracks were labelled `ro`. `detected_language` and `romanian_confidence` come
from lingua; `marker_words` counts Romanian marker words.

`clean_words` and `text_sha256` describe normalized text, which is not included.
All records have `status: accepted`. `historical_duplicate_*` records earlier
copies that were merged or superseded; `historical_corpus_exclusions` records
cross-profile discovery. Final video IDs and text hashes are unique.

Null means unavailable, not zero. A null `license` does not grant permission
to redistribute captions.

## Coverage

Word coverage uses full corpora and counts normalized, lowercased occurrences
in grammatical benchmark sentences. `train_only_target_form` instead excludes
validation data and groups pairs by whether the target verb form was seen.
Hashes are included. The JSON's historical benchmark path refers to the file
now bundled as `data/multiblimp-ron.tsv`.

## Benchmark attribution

Jaap Jumelet, Leonie Weissweiler, Joakim Nivre, and Arianna Bisazza.
[MultiBLiMP 1.0: A Massively Multilingual Benchmark of Linguistic Minimal Pairs](https://arxiv.org/abs/2504.02768).
2025. Romanian split, unchanged, CC-BY-4.0.
