# Data Provenance

Raw data is not committed to this repository (see .gitignore). This file is
the record of what was used, where it came from, and when — per the
point-in-time snapshot policy (Charter Section 1.4). No re-pulls mid-project.

## 1. UK FCDO Sanctions List

- Source: OpenSanctions, `gb_fcdo_sanctions` dataset
- URL: https://www.opensanctions.org/datasets/gb_fcdo_sanctions/
- Downloaded: 2026-08-02
- Local file: `data/raw/fcdo_sanctions/gb_fcdo_sanctions_2026-08-02.csv`
- Row count: 6,261
- License: OpenSanctions — free for non-commercial use
- SHA256: <paste output of `sha256sum` / `Get-FileHash` here>

## 2. PEP Dataset

- Source: OpenSanctions, PEP dataset
- URL: https://www.opensanctions.org/datasets/peps/
- Downloaded: 2026-08-02
- Local file: `data/raw/pep/pep_export_2026-08-02.csv`
- Row count (pre-filter): <fill in>
- Row count (post `contains gb` filter, applied in code not at source): <fill in>
- Known issue: raw export has UTF-8-as-Latin-1 mojibake in non-ASCII names —
  fixed in `src/pep_cleaning.py`, not at the source file.
- License: OpenSanctions — free for non-commercial use
- SHA256: <paste hash here>

## 3. UK Companies House PSC Snapshot

- Source: Companies House bulk PSC snapshot
- URL: http://download.companieshouse.gov.uk/en_pscdata.html
- Snapshot date: 2026-08-02
- Local files: `data/raw/psc/psc_snapshot_2026-08-02/` (32 parts, JSONL/.txt)
- Format: one JSON object per line
- License: Open Government Licence
- SHA256 (per part): <paste hashes here, or a manifest file if easier>

## Policy

- Single fixed download date per source, logged above. No mid-project re-pulls.
- Any dataset added later must get its own dated entry here before use.