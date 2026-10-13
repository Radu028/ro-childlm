# Pilot transcription audit

Appendix I. Forty one-minute windows of the final corpus, transcribed from the
audio with captions off and compared with the captions.

## Sample

- Unit: a 60-second interval of a corpus video; the last interval of a video may be shorter.
- Strata: profile (0-4, 5-8) x caption kind (automatic, creator), 16 automatic
  and 4 creator windows per profile. The creator strata are descriptive only.
- Uniform draw without replacement within each stratum, seed 42, plus reserves
  (4 automatic and 2 creator per stratum) for unavailable videos only. Every
  video was available, so no reserve was used.
- No selection on text quality and no channel quotas after the draw: 40 videos
  from 19 channels.

`sample.py` redraws the sample from the manifests;
`--check data/asr-pilot-sample.json` confirms it matches the saved one.

## Transcription and scoring

The reference was transcribed verbatim, with repetitions, hesitations and
speech errors kept. Each window
was labelled speech, singing or both.

Both texts share one normalization: lowercase, NFC, cedilla to comma below,
punctuation and bracketed tags removed, hyphenated forms split, caption digits
written out as words. Sounds (laughter, hesitations, onomatopoeia, emotive
exclamations) are removed from both texts. Words are aligned by Levenshtein
with unit costs; on paths of equal cost the backtrace prefers a match or
substitution, then a deletion, then an insertion. Caption cues that cross a
window edge are trimmed only where trimming lowers the edit distance to the
reference.

WER leaves out the names the caption spells differently: (S on other tokens +
all D + all I) / (reference tokens minus those names). A name the caption
misses counts as a deletion. Rates pool errors over windows; pooled
groups weight each stratum by its intervals in the corpus over its sampled
windows. Intervals come from a stratified bootstrap over windows (10,000
resamples, seed 42, nearest-rank percentiles) and cover sampling error only.

Where the two texts repeat a word or line a different number of times (five
passages), the extra repetitions do not count as errors. Exclamations and
sounds written in only one text are left out, as are caption spellings of a
name that do not line up with it. One word that the caption has just after the
window's end does not count as an error. Short and full forms of the same word (îmi/mi, e/i, este/e) and m for metri count as matches, and a
word the caption splits in two with the same letters (la/l-a) is one error.

Agreement-bearing words are chosen on the reference alone: Stanza's Romanian
tagger marks adjectives, determiners, finite verbs and auxiliaries with person;
names and a fixed list of invariable words never count, and four ambiguous
forms are judged by hand at each occurrence. Every substitution or deletion of
such a word is classified by hand as a feature change, a deletion, a lexical
substitution or a word split in two with the same letters (errors), or as a word
present at an adjacent position, another form with the same features, or
uncertain (left out of the denominator). Two words the tagger gets wrong in context are left out by hand.
Two random samples of 60 words check the selection.

## What is released

- `data/asr-pilot-sample.json`: the sample, strata, weights and reserves.
- `results/appendix/asr-pilot-windows.csv`: counts for each window (reference
  words, substitutions, deletions, insertions, names and errors on names,
  agreement-bearing words) and the audio label.
- `results/appendix/asr-agreement.csv`: the 34 classified caption
  differences, by window and category. IDs are window:token position in the
  normalized reference. For the 9 feature changes, `word_end` marks a change
  only at the end of the word and `second_person_i` one that adds or drops the
  second-person -i.
- `results/appendix/asr-agreement-checks.csv`: the two 60-word checks.
- `results/appendix/asr-pilot.json`: the numbers the command below prints.

```bash
python -m romanian_childlm.analysis.asr_pilot
```

writes `asr-pilot.json` and prints Table 12; the sample's minutes, videos,
channels and name tokens; WER by audio label; the categories of the 34
differences, including how many of the 9 feature changes differ only at the
word end or involve the second-person -i; and the two 60-word checks.
Statements that need the words themselves (the examples, which deletions fall
in repeated song lines, the parts of speech of the missed words) are not
reproducible from these files. The reference transcriptions, the alignments and
the normalization word lists contain caption and speech text, so they are not
released, and the counts cannot be recomputed from the audio without a new
reference.
