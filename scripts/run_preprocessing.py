"""
Orchestrates Week 1 preprocessing: sanctions loading, PEP cleaning, PSC
loading. Writes cleaned outputs to data/interim/ and a dated summary to
docs/preprocessing_log.md (Charter Section 1.1 Step 4 / Section 3, Week 1).

Run from the project root:
    python scripts/run_preprocessing.py
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sanctions_loader  # noqa: E402
import pep_cleaning  # noqa: E402
import psc_loader  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw")
INTERIM = os.path.join(ROOT, "data", "interim")
DOCS = os.path.join(ROOT, "docs")

SANCTIONS_INPUT = os.path.join(RAW, "fcdo_sanctions", "fcdo.csv")
PEP_INPUT = os.path.join(RAW, "pep", "pep.csv")
PSC_INPUT_DIR = os.path.join(RAW, "psc")

SANCTIONS_OUTPUT = os.path.join(INTERIM, "sanctions_clean.csv")
PEP_OUTPUT = os.path.join(INTERIM, "pep_clean.csv")
PSC_OUTPUT = os.path.join(INTERIM, "psc_clean.csv")


def main():
    os.makedirs(INTERIM, exist_ok=True)
    os.makedirs(DOCS, exist_ok=True)

    log_lines = [
        f"# Preprocessing Log — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    print("=== 1/3: Sanctions loader ===")
    sanctions_counts = sanctions_loader.run(SANCTIONS_INPUT, SANCTIONS_OUTPUT)
    log_lines += ["## Sanctions (gb_fcdo_sanctions)", ""]
    log_lines += [f"- {k}: {v}" for k, v in sanctions_counts.items()]
    log_lines.append("")

    print("\n=== 2/3: PEP cleaning ===")
    pep_counts = pep_cleaning.run(PEP_INPUT, PEP_OUTPUT)
    log_lines += ["## PEP dataset", ""]
    log_lines += [f"- {k}: {v}" for k, v in pep_counts.items()]
    log_lines.append("")

    print("\n=== 3/3: PSC loader ===")
    psc_counts = psc_loader.run(PSC_INPUT_DIR, PSC_OUTPUT, expected_parts=32)
    log_lines += ["## PSC snapshot", ""]
    log_lines += [f"- {k}: {v}" for k, v in psc_counts.items()]
    if psc_counts["parts_found"] < psc_counts["parts_expected"]:
        log_lines.append(
            f"- **PARTIAL COVERAGE**: only {psc_counts['parts_found']}/"
            f"{psc_counts['parts_expected']} PSC parts processed. "
            "Any downstream match counts are a partial-coverage pilot "
            "result, not a full-register figure (Charter Section 1.1, Step 4)."
        )
    log_lines.append("")

    log_path = os.path.join(DOCS, "preprocessing_log.md")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"\nAll three cleaned files written to {INTERIM}")
    print(f"Summary log written to {log_path}")


if __name__ == "__main__":
    main()
