"""Draw the pilot transcription sample from the corpus manifests."""

import argparse
import json
import math
import random
from pathlib import Path

SEED = 42
ALLOCATION = {"automatic": 16, "manual": 4}
RESERVES = {"automatic": 4, "manual": 2}


def draw(manifests):
    strata, sample = {}, []
    for band, path in manifests.items():
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        for kind in ["automatic", "manual"]:
            intervals = []
            for r in sorted(rows, key=lambda r: r["video_id"]):
                if r["caption_kind"] == kind:
                    for k in range(math.ceil(r["duration_seconds"] / 60)):
                        intervals.append((r, 60 * k, min(60 * (k + 1), r["duration_seconds"])))

            drawn = random.Random(f"{SEED}:{band}:{kind}").sample(
                intervals, ALLOCATION[kind] + RESERVES[kind]
            )
            strata[f"{band}/{kind}"] = dict(
                intervals=len(intervals),
                sampled=ALLOCATION[kind],
                reserves=RESERVES[kind],
                weight=len(intervals) / ALLOCATION[kind],
            )
            for rank, (r, start, end) in enumerate(drawn):
                sample.append(dict(
                    stratum=f"{band}/{kind}",
                    rank=rank,
                    reserve=rank >= ALLOCATION[kind],
                    video_id=r["video_id"],
                    channel=r["channel"],
                    start_s=start,
                    end_s=end,
                    text_sha256=r["text_sha256"],
                ))

    # Audit IDs come from a separate shuffle, so an ID does not reveal its stratum.
    order = list(range(len(sample)))
    random.Random(f"{SEED}:order").shuffle(order)
    for position, i in enumerate(order, start=1):
        sample[i]["audit_id"] = f"A{position:02d}"
    return strata, sample


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest-0-4", type=Path, default=Path("data/manifest-0-4.jsonl"))
    parser.add_argument("--manifest-5-8", type=Path, default=Path("data/manifest-5-8.jsonl"))
    parser.add_argument(
        "--check", type=Path, help="compare with a saved sample instead of printing it"
    )
    args = parser.parse_args()

    strata, sample = draw({"0-4": args.manifest_0_4, "5-8": args.manifest_5_8})
    if args.check:
        saved = json.loads(args.check.read_text())
        assert saved["strata"] == strata, "strata differ"
        ours = sorted(sample, key=lambda s: s["audit_id"])
        theirs = sorted(saved["sample"], key=lambda s: s["audit_id"])
        assert [{k: s[k] for k in ours[0]} for s in theirs] == ours, "sample differs"
        print(f"{len(sample)} windows and reserves match {args.check}")
    else:
        print(json.dumps(
            dict(seed=SEED, unit="60 s interval", strata=strata, sample=sample),
            indent=1,
            ensure_ascii=False,
        ))


if __name__ == "__main__":
    main()
