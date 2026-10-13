"""Retrieve the corpus caption tracks through TranscriptAPI for hash checking."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

URL = "https://transcriptapi.com/api/v2/youtube/transcript"
MAX_PER_MINUTE = 180
WORKERS = 6
MAX_ATTEMPTS = 5


class Stop(Exception):
    """HTTP 401 or 402, after which no request can succeed."""


class Limiter:
    def __init__(self, per_minute: float, stop: threading.Event) -> None:
        self.spacing, self.next_start, self.lock = 60 / per_minute, 0.0, threading.Lock()
        self.stop = stop

    def wait(self) -> bool:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_start)
            self.next_start = start + self.spacing
        return not self.stop.wait(start - now)


def retry_wait(header: str | None, attempt: int) -> float:
    wait = 30.0 * attempt
    if header:
        try:
            wait = float(header)
        except ValueError:
            try:
                wait = parsedate_to_datetime(header).timestamp() - time.time()
            except (TypeError, ValueError):
                pass
    return min(max(wait, 1.0), 600.0) if math.isfinite(wait) else 30.0 * attempt


def request(video_id: str, key: str) -> bytes:
    query = urlencode({
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "language": "ro",
        "send_metadata": "true",
        "format": "json",
        "include_timestamp": "true",
    })
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    with urlopen(Request(f"{URL}?{query}", headers=headers), timeout=60) as response:
        return response.read()


def valid(payload: bytes) -> bool:
    try:
        segments = json.loads(payload).get("transcript")
    except (ValueError, AttributeError):
        return False
    return (
        isinstance(segments, list)
        and all(isinstance(s, dict) for s in segments)
        and any(isinstance(s.get("text"), str) and s["text"].strip() for s in segments)
    )


def fetch(video_id: str, key: str, out: Path, limiter: Limiter, stop: threading.Event) -> str:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if not limiter.wait():
            return "stopped"
        try:
            payload = request(video_id, key)
        except HTTPError as error:
            if error.code in (401, 402):
                stop.set()
                raise Stop(f"HTTP {error.code}: check the key or the remaining credits.") from None
            if error.code == 429 or error.code >= 500:
                wait = retry_wait(
                    error.headers.get("Retry-After") if error.headers else None, attempt
                )
                print(f"{video_id}: HTTP {error.code}, waiting {wait:.0f} s", file=sys.stderr)
                stop.wait(wait)
                continue
            return str(error.code)  # deleted video or no Romanian track
        except (URLError, HTTPException, OSError) as error:
            print(f"{video_id}: {type(error).__name__}, retrying", file=sys.stderr)
            stop.wait(10 * attempt)
            continue

        if not valid(payload):
            continue
        path = out / f"{video_id}.transcriptapi.json"
        path.with_suffix(".tmp").write_bytes(payload)
        path.with_suffix(".tmp").replace(path)
        return "saved"
    return "failed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-minute", type=float, default=MAX_PER_MINUTE)
    args = parser.parse_args()

    if not (math.isfinite(args.per_minute) and 0 < args.per_minute <= MAX_PER_MINUTE):
        sys.exit(f"--per-minute must be above 0 and at most {MAX_PER_MINUTE}.")
    key = os.environ.get("TRANSCRIPTAPI_KEY", "").strip()
    if not key or not key.isascii() or not key.isprintable() or " " in key:
        sys.exit("Set TRANSCRIPTAPI_KEY to the key alone, without spaces.")
    args.out.mkdir(parents=True, exist_ok=True)

    video_ids = list(dict.fromkeys(  # unique, in manifest order
        json.loads(line)["video_id"]
        for manifest in args.manifest
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ))
    todo = [v for v in video_ids if not (args.out / f"{v}.transcriptapi.json").exists()]
    print(f"{len(video_ids)} videos, {len(video_ids) - len(todo)} already retrieved")

    stop = threading.Event()
    limiter, counts, message = Limiter(args.per_minute, stop), {}, None
    pool = ThreadPoolExecutor(WORKERS)
    try:
        with (args.out / "unavailable.tsv").open("w", encoding="utf-8") as unavailable:
            futures = {pool.submit(fetch, v, key, args.out, limiter, stop): v for v in todo}
            for done, future in enumerate(as_completed(futures), 1):
                try:
                    status = future.result()
                except Stop as error:
                    message = message or str(error)
                    continue
                counts[status] = counts.get(status, 0) + 1
                if status not in ("saved", "stopped"):
                    unavailable.write(f"{futures[future]}\t{status}\n")
                    unavailable.flush()
                if done % 250 == 0:
                    print(f"{done}/{len(todo)} done")
    except KeyboardInterrupt:
        message = "Interrupted; run again to continue."
    finally:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)

    print(" ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if message:
        sys.exit(message)
    return 1 if counts.get("failed") or counts.get("stopped") else 0


if __name__ == "__main__":
    sys.exit(main())
