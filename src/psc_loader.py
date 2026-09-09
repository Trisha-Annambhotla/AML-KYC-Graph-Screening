"""
Companies House PSC snapshot loader.

Reads whatever .txt (JSONL) parts are actually present in the snapshot
directory -- currently just part 1 of 32 -- and logs coverage explicitly so
match-count results are never silently treated as full-register figures.

Handles both PSC record kinds seen in the snapshot:
    - individual-person-with-significant-control: name from name_elements,
      or identity_verification_details.preferred_name if present.
    - corporate-entity-person-with-significant-control: name field directly.

Streams line-by-line (snapshot parts can be large) rather than loading the
whole file into memory, and writes output incrementally.

Run directly:
    python src/psc_loader.py --input-dir data/raw/psc/psc_snapshot_2026-08-02 \
                              --output data/interim/psc_clean.csv \
                              --expected-parts 32
"""

import argparse
import csv
import glob
import json
import os
import sys

from text_encoding import fix_mojibake
from name_normalization import normalize_name

OUTPUT_FIELDS = [
    "company_number", "kind", "name", "normalized_name", "has_legal_suffix",
    "address", "nationality", "country_of_residence", "notified_on",
    "natures_of_control", "source_part",
]


def _extract_name(data: dict) -> str:
    # Individuals: prefer the verified preferred_name if present (it can
    # differ from name_elements, e.g. post-marriage or corrected spelling).
    ivd = data.get("identity_verification_details") or {}
    if ivd.get("preferred_name"):
        return ivd["preferred_name"]

    if "name_elements" in data:
        ne = data["name_elements"]
        parts = [ne.get("forename"), ne.get("middle_name"), ne.get("surname")]
        name = " ".join(p for p in parts if p)
        if name:
            return name

    return data.get("name", "")


def _extract_address(data: dict) -> str:
    addr = data.get("address") or {}
    parts = [
        addr.get("premises"), addr.get("address_line_1"), addr.get("address_line_2"),
        addr.get("locality"), addr.get("region"), addr.get("postal_code"),
        addr.get("country"),
    ]
    return ", ".join(p for p in parts if p)


def iter_psc_records(part_path: str, part_label: str):
    with open(part_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [warn] skipping malformed JSON at {part_label}:{line_num}", file=sys.stderr)
                continue

            data = rec.get("data", {})
            name = fix_mojibake(_extract_name(data))
            if not name:
                continue

            norm = normalize_name(name)
            yield {
                "company_number": rec.get("company_number", ""),
                "kind": data.get("kind", ""),
                "name": name,
                "normalized_name": norm["normalized"],
                "has_legal_suffix": norm["has_legal_suffix"],
                "address": _extract_address(data),
                "nationality": data.get("nationality", ""),
                "country_of_residence": data.get("country_of_residence", ""),
                "notified_on": data.get("notified_on", ""),
                "natures_of_control": ";".join(data.get("natures_of_control", []) or []),
                "source_part": part_label,
            }


def run(input_dir: str, output_path: str, expected_parts: int = 32):
    part_files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))

    if not part_files:
        raise FileNotFoundError(f"No .txt parts found in {input_dir}")

    print(f"Found {len(part_files)} of {expected_parts} expected PSC parts.", file=sys.stderr)
    if len(part_files) < expected_parts:
        print(
            f"  [COVERAGE WARNING] Only {len(part_files)}/{expected_parts} parts present. "
            "Any match counts from this run are a PARTIAL-COVERAGE pilot result, "
            "not a full-register figure. Log this explicitly in docs/pilot_results.md.",
            file=sys.stderr,
        )

    counts = {"parts_processed": 0, "records_written": 0, "records_skipped_no_name": 0}

    with open(output_path, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for part_path in part_files:
            part_label = os.path.basename(part_path)
            print(f"  Processing {part_label} ...", file=sys.stderr)
            part_record_count = 0
            for record in iter_psc_records(part_path, part_label):
                writer.writerow(record)
                part_record_count += 1
            counts["records_written"] += part_record_count
            counts["parts_processed"] += 1
            print(f"    -> {part_record_count} named PSC records", file=sys.stderr)

    counts["parts_found"] = len(part_files)
    counts["parts_expected"] = expected_parts
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Directory containing PSC snapshot .txt parts")
    ap.add_argument("--output", required=True)
    ap.add_argument("--expected-parts", type=int, default=32)
    args = ap.parse_args()

    result_counts = run(args.input_dir, args.output, args.expected_parts)

    print("\nPSC loader summary:", file=sys.stderr)
    for k, v in result_counts.items():
        print(f"  {k}: {v}", file=sys.stderr)
    print(f"Cleaned file written to: {args.output}", file=sys.stderr)
