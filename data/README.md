# Data

- `manifest-*.jsonl`: one record per caption document, with IDs and provenance.
- `stats-*.json`: corpus statistics from the appendix.
- `caption-draws.json`: the five matched 1M caption draws, with their training and
  validation video IDs, word counts and preparation identities.
- `benchmark-coverage.csv`: per benchmark row, presence of the target form and covered
  words and bigrams in each training split of Appendix D.
- `caption-hash-verification.tsv`: the verifier's status for each video on the
  caption downloads used to build the corpus.
- `asr-pilot-sample.json`: the sample of the pilot transcription audit (Appendix I).
- `multiblimp-ron.tsv`: unchanged Romanian MultiBLiMP 1.0 data.

## Reading the manifests

`age_band` is the recorded discovery profile, not a speaker's age. Each video
appears once. Where both profiles yielded an eligible copy, 0-4 took precedence.

`discovery_method`, `discovery_query`, `discovery_rank` and `discovery_date`
record where and when the video was found. Methods beginning with
`kids_fixed_search`, `kids_home`, `kids_watch` or `kids_category` refer to
YouTube Kids. Methods containing `channel` mean enumeration of a channel
first confirmed there. A few records use the same labels without the `kids_` prefix.

`caption_kind` distinguishes creator captions from automatic captions.
`caption_language` is the provider's label, `ro` or `asr-ro`; six automatic
tracks were labelled `ro`. `detected_language` and `romanian_confidence` come
from lingua; `marker_words` counts Romanian marker words.

`clean_words` and `text_sha256` describe normalized text, which is not included.
All records have `status: accepted`. `historical_duplicate_*` names another copy
of the same content that deduplication matched to this record, and the kind of
match; `historical_corpus_exclusions` records discovery under both profiles.
Final video IDs and text hashes are unique.

Null means unavailable, not zero. A null `license` does not grant permission
to redistribute captions.

## Retrieving and checking captions

Caption text is not included, so each video's Romanian track (`caption_language`)
has to be retrieved again. Any tool that returns that track works. We first used
yt-dlp, which is free, but YouTube's request limits stopped it before the corpus
was complete, so the corpus captions were retrieved through TranscriptAPI, a paid
caption service. The same route is scripted here:

```bash
TRANSCRIPTAPI_KEY=... python code/scripts/fetch_captions_transcriptapi.py \
  --manifest data/manifest-0-4.jsonl --manifest data/manifest-5-8.jsonl \
  --out rehydrated/captions
```

The script spaces requests to stay below the service's limit of 200 per minute
and takes about half an hour. It saves one `<video_id>.transcriptapi.json` per
video and skips videos that already have one, so you can rerun it after an
interruption. It lists videos without a usable track in `unavailable.tsv` and
stops if the service rejects the key or the credits run out.

At the prices listed on 11 October 2026
(https://transcriptapi.com/pricing), each successful request uses one credit and
failed requests are free; one month of the US$5 plan includes 1,000 credits, and
five top-ups of 1,000 credits at US$2.50 bring the total to about US$17.50.
Prices and caption availability may change. With yt-dlp, save the tracks as
`.json3` or `.vtt` into one folder instead; it is free but may take longer under
YouTube's request limits. These yt-dlp options fetch the Romanian tracks:
`--skip-download --write-subs --write-auto-subs --sub-langs ro --sub-format json3 -o "%(id)s"`,
so the files are named after the video ID.

Then run:

```bash
python code/scripts/verify_caption_hashes.py \
  --manifest data/manifest-0-4.jsonl --manifest data/manifest-5-8.jsonl \
  --captions rehydrated/captions --report verify-report.tsv
```

`--captions` holds one file per video: `.json3`, `.vtt` or TranscriptAPI `.json`,
with a name that starts with the video ID and a dot, such as `<video_id>.ro.json3`.

The script applies the collection's normalization, hashes the result and
reports `match`, `mismatch`, `missing`, `ambiguous` (several files for one
video) or `unreadable` per video. On the caption downloads used to build the
corpus, run once per profile folder, it matches all 5,356 hashes
(`caption-hash-verification.tsv`). Revised automatic captions, a different
track and deleted videos appear as `mismatch` or `missing`.

## Where the content comes from

`channel-origin.csv` labels each of the 64 channels by where its content was
first made: `original` (made in Romanian), `dubbed` (cartoons re-voiced or
songs translated from another language) or `mixed` (a lot of both; a few
adapted songs don't make an original channel mixed). Row 32 groups 20 videos
with no channel name. The labels come
from public sources: the producer, the same shows on channels in other
languages and their upload dates, and dubbing or translation credits. `source`
gives one link per channel. Six channels, and 20 videos with no channel name,
were labeled video by video in `video-origin.csv` (`by video` in the channel
file). Weighted by caption words, dubbed content is 82.3% of the corpus, content of
mixed origin (mixed channels and two compilations) 12.8% and Romanian
productions 4.9% (Appendix H).

## Coverage

`benchmark-coverage.csv` has one row per benchmark row (`row` 1 is the first
pair of `multiblimp-ron.tsv`). `words` and `bigrams` count the grammatical
sentence's tokens and token bigrams after the training normalization and
lowercasing. For each training split, `_target` is 1 if the pair's target form
(the grammatical word that differs between its two sentences) occurs in the
split, `_words` counts the sentence's tokens that occur in it and `_bigrams`
its bigrams that occur in it. Splits: `captions_d0` to `captions_d4` are the
five matched 1M caption draws (`d0` is `ours-1m-30k`, then `ours-1m-s1` to
`s4`), `bbl` the BabyBabelLM corpus, each as its training split (`_train`) or
with its validation documents (`_trainval`, the input of GPT-2 17M), and
`combined_10m_train` the Combined 10M training split. No text is included.
It needs caption text to rebuild:

```bash
python -m romanian_childlm.analysis.coverage \
  --corpus-0-4 rehydrated/corpus-0-4.jsonl --corpus-5-8 rehydrated/corpus-5-8.jsonl \
  --bbl-prepared prepared-babybabellm --combined-prepared prepared
```

`prepared` is the Combined 10M preparation with `--strip-punctuation` and
`prepared-babybabellm` the BabyBabelLM one (code/README.md). With the corpus
and splits used in the paper, the command reproduces this file byte for byte.

## Benchmark attribution

Jaap Jumelet, Leonie Weissweiler, Joakim Nivre, and Arianna Bisazza.
[MultiBLiMP 1.0: A Massively Multilingual Benchmark of Linguistic Minimal Pairs](https://arxiv.org/abs/2504.02768).
2025. Romanian split, unchanged, CC-BY-4.0.
