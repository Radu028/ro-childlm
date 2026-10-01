"""Count words, keeping apostrophes and hyphens inside word forms."""

from __future__ import annotations

import re

_WORD = re.compile(r"[^\W\d_]+(?:[-’'][^\W\d_]+)*", re.UNICODE)


def count_words(text: str) -> int:
    return len(_WORD.findall(text))
