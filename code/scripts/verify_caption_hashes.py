"""Check rehydrated caption tracks against the released `text_sha256` values."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import unicodedata
from pathlib import Path

# The normalization that produced `text_sha256` when the corpus was built.
_TIMESTAMP_LINE = re.compile(
    r"^\s*(?:\d+:)?\d{2}:\d{2}[.,]\d{3}\s+-->\s+(?:\d+:)?\d{2}:\d{2}[.,]\d{3}"
)
_HTML_TAG = re.compile(r"<[^>]+>")


def parse_json3(payload: str) -> list[str]:
    cues = []
    for event in json.loads(payload).get("events", []):
        text = "".join(str(s.get("utf8", "")) for s in event.get("segs") or [])
        text = text.replace("\n", " ").strip()
        if text:
            cues.append(text)
    return cues


def parse_vtt(payload: str) -> list[str]:
    cues: list[str] = []
    current: list[str] = []
    for raw_line in payload.splitlines() + [""]:
        line = raw_line.strip("﻿ \t")
        if not line:
            if current and (cue := " ".join(current).strip()):
                cues.append(cue)
            current.clear()
        elif not (
            line == "WEBVTT"
            or line.startswith(("Kind:", "Language:", "NOTE"))
            or _TIMESTAMP_LINE.match(line)
            or line.isdigit()
        ):
            current.append(line)
    return cues


def parse_transcript_json(payload: str) -> list[str]:
    data = json.loads(payload)
    if not isinstance(data, dict) or not isinstance(data.get("transcript"), list):
        raise TypeError("expected {'transcript': [...]}")
    return [
        str(s.get("text", ""))
        for s in data["transcript"]
        if isinstance(s, dict) and s.get("text")
    ]


def plain_text(value: str) -> str:
    value = _HTML_TAG.sub("", html.unescape(value))
    return " ".join(unicodedata.normalize("NFC", value).split())


def _append_with_overlap(accumulated: list[str], incoming: list[str]) -> None:
    for size in range(min(len(accumulated), len(incoming), 30), 2, -1):
        if accumulated[-size:] == incoming[:size]:
            accumulated.extend(incoming[size:])
            return
    accumulated.extend(incoming)


def merge_cues(cues: list[str], automatic: bool) -> str:
    """Join cues, merging the rolling cues that automatic tracks repeat."""
    cleaned = [c for cue in cues if (c := plain_text(cue))]
    if not automatic:
        deduplicated: list[str] = []
        for cue in cleaned:
            if not deduplicated or cue != deduplicated[-1]:
                deduplicated.append(cue)
        return " ".join(deduplicated)

    tokens: list[str] = []
    previous: list[str] | None = None
    for cue in cleaned:
        incoming = cue.split()
        if incoming == previous:
            continue
        if previous and len(previous) >= 3 and incoming[: len(previous)] == previous:
            tokens.extend(incoming[len(previous):])
        elif not (previous and len(incoming) >= 3 and previous[: len(incoming)] == incoming):
            _append_with_overlap(tokens, incoming)
        previous = incoming
    return " ".join(tokens)


PARSERS = {".json3": parse_json3, ".vtt": parse_vtt, ".json": parse_transcript_json}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="TSV with video_id, status and hashes")
    args = parser.parse_args()

    files: dict[str, list[Path]] = {}
    for path in sorted(args.captions.iterdir()):
        if path.suffix in PARSERS:
            files.setdefault(path.name.split(".")[0], []).append(path)

    rows = []
    counts = {"match": 0, "mismatch": 0, "missing": 0, "ambiguous": 0, "unreadable": 0}
    for manifest in args.manifest:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            video_id, expected = record["video_id"], record["text_sha256"]
            candidates, actual = files.get(video_id, []), ""
            if not candidates:
                status = "missing"
            elif len(candidates) > 1:  # keep one file per video, the manifest's track
                status = "ambiguous"
            else:
                path = candidates[0]
                try:
                    cues = PARSERS[path.suffix](path.read_text(encoding="utf-8"))
                    text = merge_cues(cues, record["caption_kind"] == "automatic")
                    actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    status = "match" if actual == expected else "mismatch"
                except (ValueError, TypeError, KeyError, AttributeError):
                    status = "unreadable"
            counts[status] += 1
            rows.append((video_id, status, expected, actual))

    if args.report:
        with args.report.open("w", encoding="utf-8") as out:
            out.write("video_id\tstatus\texpected_sha256\tactual_sha256\n")
            out.writelines("\t".join(row) + "\n" for row in rows)

    total = len(rows)
    print(" ".join(f"{k}={v}" for k, v in counts.items()), f"total={total}")
    return 0 if counts["match"] == total else 1


if __name__ == "__main__":
    sys.exit(main())
