"""
PEP dataset cleaning (Charter Section 1.2 known issues, Part 2 recommendation).

Three fixes applied, in this order:
    1. Encoding repair (mojibake) on name/aliases/addresses fields.
    2. Filter to rows where 'countries' contains 'gb' (semicolon-separated field).
    3. Regex-exclude positional-title rows (Lords Spiritual / bishops etc.)
       that are unmatchable against PSC personal-name fields.

Run directly:
    python src/pep_cleaning.py --input data/raw/pep/pep_export_2026-08-02.csv \
                                --output data/interim/pep_clean.csv
"""

import argparse
import csv
import re
import sys

from text_encoding import fix_mojibake, fix_mojibake_row, sniff_delimiter
from name_normalization import normalize_name

MOJIBAKE_FIELDS = ["name", "aliases", "addresses"]

# Positional titles that appear in the PEP list instead of a personal name --
# these can never match a PSC name field and must be excluded up front rather
# than silently failing every match attempt against them.
# 'bishop of' / 'archbishop' are matched anywhere in the name, since titles
# like "The Rt Rev. the Lord Bishop of St. Albans" put an honorific first.
# 'rt rev' alone is deliberately NOT a pattern: it also prefixes real
# people's names, e.g. "The Rt Rev. the Lord Harries of Pentregarth".
POSITIONAL_TITLE_RE = re.compile(
    r"^(?:(the\s+)?(lord\s+bishop\s+of|bishop\s+of|lord\s+(?!\w+\s+\w+$)|"
    r"baroness\s+of|rt\s+hon\s+the|speaker\s+of\s+the|president\s+of\s+the)\b"
    r"|.*(bishop\s+of\b|archbishop))",
    re.IGNORECASE,
)


def run(input_path: str, output_path: str):
    """
    Reads the raw PEP export, applies encoding repair + gb filter +
    positional-title exclusion + normalization, writes the cleaned CSV,
    and returns a dict of summary counts for the pilot log / report.
    """
    counts = {
        "total_rows": 0,
        "kept_gb": 0,
        "dropped_not_gb": 0,
        "dropped_positional_title": 0,
        "dropped_empty_name": 0,
    }

    with open(input_path, encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = sniff_delimiter(sample)
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = list(reader.fieldnames) + ["normalized_name", "has_legal_suffix"]

        rows_out = []
        for row in reader:
            counts["total_rows"] += 1
            row = fix_mojibake_row(row, MOJIBAKE_FIELDS)

            countries = [c.strip().lower() for c in (row.get("countries") or "").split(";")]
            if "gb" not in countries:
                counts["dropped_not_gb"] += 1
                continue

            name = (row.get("name") or "").strip()
            if not name:
                counts["dropped_empty_name"] += 1
                continue

            if POSITIONAL_TITLE_RE.match(name):
                counts["dropped_positional_title"] += 1
                continue

            norm = normalize_name(name)
            row["normalized_name"] = norm["normalized"]
            row["has_legal_suffix"] = norm["has_legal_suffix"]

            clean_aliases = []
            for alias in (row.get("aliases") or "").split(";"):
                alias = alias.strip()
                if alias and not POSITIONAL_TITLE_RE.match(alias):
                    clean_aliases.append(alias)
            row["aliases"] = ";".join(clean_aliases)

            counts["kept_gb"] += 1
            rows_out.append(row)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    result_counts = run(args.input, args.output)

    print("PEP cleaning summary:", file=sys.stderr)
    for k, v in result_counts.items():
        print(f"  {k}: {v}", file=sys.stderr)
    print(f"Cleaned file written to: {args.output}", file=sys.stderr)
