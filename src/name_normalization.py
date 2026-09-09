"""
Name normalization pipeline (Charter Section 2.2 / Module 4 fix).

Pipeline order, matching the charter exactly:
    lowercase -> strip punctuation -> extract legal-suffix tokens into a
    separate flag (not deleted) -> tokenize -> light stemming.

This runs BEFORE rapidfuzz scoring, as a distinct step -- not folded into
the matching call -- so it's demonstrable as its own text-analytics
component rather than a fuzzy-matching black box.
"""

import re
import unicodedata

try:
    from unidecode import unidecode
    _HAS_UNIDECODE = True
except ImportError:
    _HAS_UNIDECODE = False

LEGAL_SUFFIXES = {
    "ltd", "limited", "llp", "llc", "plc", "inc", "trust",
    "holdings", "co", "company", "corp", "corporation",
}

# Very light stemming: strip a small set of common trailing inflections.
# Deliberately conservative -- aggressive stemming on proper names causes
# more false merges than it prevents false misses.
_STEM_SUFFIXES = ("'s",)


def _strip_accents(text: str) -> str:
    """
    Transliterate to ASCII so matching isn't defeated by script differences --
    both accented Latin ('Câmpean' -> 'Campean') and non-Latin scripts
    ('Костенко' -> 'Kostenko'), per the transliteration step in the Final
    Scope Summary. Uses unidecode if installed (handles Cyrillic, Greek,
    etc. properly); falls back to NFKD diacritic-stripping, which only
    handles accented Latin and leaves other scripts unchanged, if not.
    """
    if _HAS_UNIDECODE:
        return unidecode(text)
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def _light_stem(token: str) -> str:
    for suf in _STEM_SUFFIXES:
        if token.endswith(suf):
            return token[: -len(suf)]
    return token


def normalize_name(raw_name: str) -> dict:
    """
    Returns a dict with:
        normalized   -- the cleaned, space-joined token string used for matching
        tokens       -- list of tokens (legal suffixes removed)
        has_legal_suffix -- bool flag, the extracted information (not discarded)
    """
    if not raw_name or not raw_name.strip():
        return {"normalized": "", "tokens": [], "has_legal_suffix": False}

    text = _strip_accents(raw_name)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # strip punctuation
    raw_tokens = text.split()

    has_legal_suffix = any(t in LEGAL_SUFFIXES for t in raw_tokens)
    tokens = [_light_stem(t) for t in raw_tokens if t not in LEGAL_SUFFIXES]

    return {
        "normalized": " ".join(tokens).strip(),
        "tokens": tokens,
        "has_legal_suffix": has_legal_suffix,
    }
