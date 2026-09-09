"""
Encoding utilities.

The PEP export has a specific, known corruption: UTF-8 bytes that were
decoded as Latin-1 at some point in the export pipeline, producing mojibake
like 'CÃ®mpean' instead of 'Câmpean'. fix_mojibake() detects and repairs this.

It is applied defensively across all three sources (PEP, sanctions, PSC)
because there's no guarantee only one file is affected, but it is a no-op on
clean text -- it only rewrites a string if doing so actually produces valid,
different UTF-8, so correctly-encoded Cyrillic/accented names (e.g. in the
sanctions file) pass through unchanged.
"""

import re

# A cheap heuristic pre-check: mojibake of this kind almost always produces
# the byte sequence for 'Ã' (U+00C3) followed by another Latin-1 supplement
# character. If neither telltale character is present, skip the round-trip
# entirely rather than risk corrupting genuinely clean text.
_MOJIBAKE_HINT = re.compile(r"[ÃÂ]")


def fix_mojibake(text: str) -> str:
    """Repair UTF-8-decoded-as-Latin-1 mojibake. Safe no-op on clean text."""
    if not text or not _MOJIBAKE_HINT.search(text):
        return text
    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text  # not this kind of corruption -- leave untouched
    return repaired


def fix_mojibake_row(row: dict, fields: list[str]) -> dict:
    """Apply fix_mojibake to a specific set of fields in a CSV DictReader row."""
    for field in fields:
        if field in row and row[field]:
            row[field] = fix_mojibake(row[field])
    return row


def sniff_delimiter(sample: str) -> str:
    """Distinguish tab-delimited vs comma-delimited exports (the PEP preview
    you pasted was tab-delimited; some OpenSanctions exports are comma-delimited).
    """
    return "\t" if sample.count("\t") > sample.count(",") else ","
