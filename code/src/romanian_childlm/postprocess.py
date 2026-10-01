from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_BRACKETED_MARKER = re.compile(r"\[([^\[\]\n]{1,60})\]")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?])")

# Keep bracketed text unless it matches a known event label.
_PARALINGUISTIC_LABELS = frozenset({
    "aplauze", "applause", "bell", "cantec", "cantat", "cascat", "cheering",
    "clopot", "cough", "crying", "gafait", "gasp", "gasping", "icnet",
    "inaudible", "instrument", "instrumental", "laughter", "laughs", "music",
    "muzica", "neinteligibil", "ovatii", "panting", "plans", "rade", "rasete",
    "scream", "screaming", "sigh", "sighing", "singing", "snoring", "song",
    "sforait", "throat clearing", "tipat", "tuse", "voce dreasa", "whistling",
    "yawning",
})


@dataclass(frozen=True)
class MarkerRemoval:
    text: str
    removed_markers: tuple[str, ...]


def _normalized_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    without_diacritics = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return " ".join(without_diacritics.split())


def remove_paralinguistic_markers(text: str) -> MarkerRemoval:
    """Remove only known bracketed event labels from a derived training view."""
    removed = []

    def replace(match: re.Match[str]) -> str:
        if _normalized_label(match.group(1)) not in _PARALINGUISTIC_LABELS:
            return match.group(0)
        removed.append(match.group(0))
        return " "

    processed = _BRACKETED_MARKER.sub(replace, text)
    processed = " ".join(processed.split())
    processed = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", processed)
    return MarkerRemoval(text=processed, removed_markers=tuple(removed))


def remove_bracketed_annotations(text: str) -> MarkerRemoval:
    """Remove every bracketed caption annotation from a derived training view."""
    removed = tuple(match.group(0) for match in _BRACKETED_MARKER.finditer(text))
    processed = _BRACKETED_MARKER.sub(" ", text)
    processed = " ".join(processed.split())
    processed = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", processed)
    return MarkerRemoval(text=processed, removed_markers=removed)
