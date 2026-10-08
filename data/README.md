# Data Provenance

Raw data is not committed to this repository (see .gitignore). This file is
the record of what was used, where it came from, and when — per the
point-in-time snapshot policy (Charter Section 1.4). No re-pulls mid-project.

## Files this project actually used

The analysis (Phases 1-5) ran on the three cleaned files in `data/interim/`,
**received from a teammate (Trisha)**. The raw downloads in `data/raw/` were
never available on the analysis machine, which is why the SHA256 hashes below
are blank: they can only be computed on the machine that holds the raw files.

| File | Rows | Produced | Notes |
|---|---|---|---|
| `psc_clean.csv` | 499,971 | 2026-10-08 (re-run of the loader) | adds `birth_month`, `birth_year` (filled for 461,760 of 461,775 individual PSCs) |
| `sanctions_clean.csv` | 19,610 (6,261 entities + 13,349 aliases) | 2026-10-08 (re-run of the loader) | adds `birth_date` (filled for 3,313 of 3,992 persons), `addresses`, `identifiers` |
| `pep_clean.csv` | 7,990 | 2026-10-08 | 19 bishop/archbishop title rows removed after preprocessing (see `docs/preprocessing_log.md`) |

**Matching used Mode A**: an owner is a strong match when the name scores
>= 90 and the birth month and year agree with the list entry. Mode A turns on
automatically because both the PSC and the sanctions file now have birth
columns (`src/matching.py`). The earlier interim files had no birth columns
and used Mode B (exact name + 3 words + nationality); those results are
superseded.

**Manual spot-check still pending.** `docs/spot_check_sample.csv` (100
matches) has not been hand-labelled yet, so `docs/match_precision.md` has no
precision figures. Until it is done, the share of strong matches that are
really the listed person is unknown.

## 1. UK FCDO Sanctions List

- Source: OpenSanctions, `gb_fcdo_sanctions` dataset
- URL: https://www.opensanctions.org/datasets/gb_fcdo_sanctions/
- Downloaded: 2026-08-02
- Downloaded 2026-08-02 (unchanged); re-processed 2026-10-08 to keep
  birth_month/birth_year (PSC) and birth_date/addresses (sanctions).
- Local file: `data/raw/fcdo_sanctions/fcdo.csv`
- Row count: 6,261
- License: OpenSanctions — free for non-commercial use
- SHA256: <paste output of `sha256sum` / `Get-FileHash` here>

## 2. PEP Dataset

- Source: OpenSanctions, PEP dataset
- URL: https://www.opensanctions.org/datasets/peps/
- Downloaded: 2026-08-02
- Local file: `data/raw/pep/pep.csv`
- Row count (pre-filter): 769,044
- Row count (post `contains gb` filter, applied in code not at source): 8,356;
  8,009 after the original job-title filter removed 347 rows (`docs/preprocessing_log.md`)
- Row count after removing job-title rows ("bishop of" / "archbishop"): **7,990**
  (19 rows removed on 2026-10-08; original kept as `data/interim/pep_clean_backup.csv`)
- Known issue: raw export has UTF-8-as-Latin-1 mojibake in non-ASCII names —
  fixed in `src/pep_cleaning.py`, not at the source file.
- License: OpenSanctions — free for non-commercial use
- SHA256: <paste hash here>

## 3. UK Companies House PSC Snapshot

- Source: Companies House bulk PSC snapshot
- URL: http://download.companieshouse.gov.uk/en_pscdata.html
- Snapshot date: 2026-08-02 (confirmed by the `source_part` column of the
  interim file: `psc-snapshot-2026-08-02_1of32.txt`)
- Downloaded 2026-08-02 (unchanged); re-processed 2026-10-08 to keep
  birth_month/birth_year (PSC) and birth_date/addresses (sanctions).
- Coverage: **1 of 32 parts** used — a deliberate sample.
- Local files: `data/raw/psc/psc-snapshot-2026-08-02_1of32.txt/` (32 parts, JSONL/.txt)
- Format: one JSON object per line
- License: Open Government Licence
- SHA256 (per part): <paste hashes here, or a manifest file if easier>

## Policy

- Single fixed download date per source, logged above. No mid-project re-pulls.
- Any dataset added later must get its own dated entry here before use.
