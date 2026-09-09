"""
UK FCDO Sanctions List loader (targets_simple.csv).

Per Charter Section 1.1 / prior validation: NO country filter is applied --
this dataset is inherently UK-scoped by dataset membership (every row is a
'gb_fcdo_sanctions' entry), not by the 'countries' field, which mostly shows
the *sanctioned entity's* nationality (Russia, Iran, etc.), not the
designating jurisdiction. Filtering on 'countries' here would reproduce the
64-row mistake documented earlier in this project.

Each row's name AND each semicolon-separated alias becomes its own seed row,
tagged by source, so a match against an alias is distinguishable from a
match against the canonical name during the manual spot-check.

Run directly:
    python src/sanctions_loader.py --input data/raw/fcdo_sanctions/gb_fcdo_sanctions_2026-08-02.csv \
                                    --output data/interim/sanctions_clean.csv
"""

import argparse
import csv
import sys

from text_encoding import fix_mojibake
from name_normalization import normalize_name

OUTPUT_FIELDS = [
    "source_id", "schema", "seed_name", "seed_source", "normalized_name",
    "has_legal_suffix", "countries", "dataset",
]


def expand_rows(input_path: str):
    """Yields one output row per (canonical name) and per alias."""
    with open(input_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = fix_mojibake((row.get("name") or "").strip())
            if name:
                norm = normalize_name(name)
                yield {
                    "source_id": row.get("id", ""),
                    "schema": row.get("schema", ""),
                    "seed_name": name,
                    "seed_source": "sanctions_canonical",
                    "normalized_name": norm["normalized"],
                    "has_legal_suffix": norm["has_legal_suffix"],
                    "countries": row.get("countries", ""),
                    "dataset": row.get("dataset", ""),
                }

            for alias in (row.get("aliases") or "").split(";"):
                alias = fix_mojibake(alias.strip())
                if not alias:
                    continue
                norm = normalize_name(alias)
                if not norm["normalized"]:
                    continue
                yield {
                    "source_id": row.get("id", ""),
                    "schema": row.get("schema", ""),
                    "seed_name": alias,
                    "seed_source": "sanctions_alias",
                    "normalized_name": norm["normalized"],
                    "has_legal_suffix": norm["has_legal_suffix"],
                    "countries": row.get("countries", ""),
                    "dataset": row.get("dataset", ""),
                }


def run(input_path: str, output_path: str):
    counts = {"canonical_names": 0, "aliases": 0, "total_seed_rows": 0}
    rows_out = []
    for row in expand_rows(input_path):
        rows_out.append(row)
        counts["total_seed_rows"] += 1
        if row["seed_source"] == "sanctions_canonical":
            counts["canonical_names"] += 1
        else:
            counts["aliases"] += 1

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)

    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    result_counts = run(args.input, args.output)

    print("Sanctions loader summary:", file=sys.stderr)
    for k, v in result_counts.items():
        print(f"  {k}: {v}", file=sys.stderr)
    print(f"Cleaned file written to: {args.output}", file=sys.stderr)
